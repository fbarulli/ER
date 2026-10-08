"""scripts/laya_build_dataset.py — build the laya FINE-TUNE corpus (JSONL).

Owner order: "we will finetune laya with the correct dataset". The laya
trainer (`laya-train`, `laya.train.read_data`/`read_jsonl`/`items_from_rows`)
consumes ONE JSON case per line:

    {"state": <str>,
     "questions": {<config/laya.question.json "questions" verbatim>},
     "expected": {<qid>: <label>}}

A `choice` question expects one of its criteria labels; a `noul` (yes/no)
question expects `"true"`/`"false"` (laya.train.target_from_expected accepts
a bool, 0/1, or the strings true/false/1/0/yes/no). Questions with no
`expected` entry are simply not labelled for that row — the laya trainer
skips them (never counts them as errors), so a partial `expected` is safe.

Sources (read, never invented):
  * data/track_setup/eligible_catalog.csv — `attribute` = the standardized
    state string (one STATE case per row);
  * data/track_setup/listing_pairs.csv — 576 labelled pairs (564 same / 12
    different) joined to the catalog and composed side-by-side exactly like
    scripts/laya_metrics_pairs.py (reused, not copied);
  * data/gate_results.csv — the verified DIFFERENT population: the
    `hard_no` rows are identity negatives; the `fallback` rows are UNKNOWN
    and are quarantined to data/laya/unknown_pairs.csv, never in the corpus;
  * data/final_validation.csv — mirrored only through
    scripts/laya_metrics_pairs.py's composed `attribute_pairs` shape.

Growth sources (owner order "laya is overfitting" 2026-10-08 — fold the
masking + augmentation the pipeline already mints ON TOP of the sources
above; both are OPT-IN so the hermetic builder tests are unchanged):
  * data/prepared/full/worker_1_baseline.pkl.gz — the prepared text bundle's
    `mask_audit` and `hard_negative_mask_audit`:
      - every `mask_audit` positive (`target_mode != "swap_values"`) becomes
        a masked-positive STATE variant (`state = masked_text`, the pipeline's
        own normalized input) labelled `package_state` from the ANCHOR row's
        standardized attribute (context masking retains every structured
        token, so the anchor's package evidence is the variant's);
      - every `hard_negative_mask_audit` counterfactual/twin and every
        `swap_counterpart` audit becomes a side-by-side PAIR (`identity_claim`)
        composed through the reused `compose_state`/`compose_side` after
        rendering each side's cleaned token tail back to the six identity
        slice fields (`_side_from_payload`). A counterfactual whose flipped
        field is not one of the six slice fields (package_material, pulp,
        sweetening, ...) collapses to two identical sides and is SKIPPED and
        COUNTED — the six-field state cannot carry it and is never invented.
  * data/labeled_pairs.csv — gate `proceed`/`hard_no` pairs above the
    similarity thresholds, composed side-by-side from the catalog exactly
    like the listing pairs (adds the `proceed` positives the gate hard_no
    sample omits).

Deterministic: seed 1729, no wall-clock, no set-iteration order leaks into
the output; a rerun reproduces every byte.

Outputs (data/laya/): train.jsonl, dev.jsonl, test.jsonl, unknown_pairs.csv,
receipt.json.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import json
import pickle
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = ROOT / "data/track_setup/eligible_catalog.csv"
PAIRS_PATH = ROOT / "data/track_setup/listing_pairs.csv"
GATE_PATH = ROOT / "data/gate_results.csv"
QUESTION_PATH = ROOT / "config/laya.question.json"
OUTPUT_DIR = ROOT / "data/laya"
# The pipeline's minted masking/augmentation live in the prepared text
# bundle. `main()` passes these when present; `build()` defaults to None so
# hermetic callers (the builder tests) keep the pre-growth corpus exactly.
BUNDLE_PATH = ROOT / "data/prepared/full/worker_1_baseline.pkl.gz"
LABELED_PAIRS_PATH = ROOT / "data/labeled_pairs.csv"

SEED = 1729
# Balanced identity negatives: total negatives target the positive count,
# capped here (the "up to ~1000" ceiling), then the ground-truth listing
# negatives are subtracted to get the hard_no sample size.
HARD_NO_CAP = 1000
SPLIT_ORDER = ("train", "dev", "test")

# package_state rule. The laya question asks for "explicit package-quantity
# evidence (a pack count or a unit volume)". A measured unit volume
# (`Volume:` carrying a finite number) or an explicit numeric pack count
# qualifies; a `Pack Type:` alone is a package FORM, not a quantity, and is
# deliberately NOT counted (its count is still reported for the operator).
NUMERIC_VALUE_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)?$")
PACK_COUNT_KEYS = frozenset({
    "pack", "pack size", "pack count", "pack quantity", "packaging",
    "units", "number of items", "item count",
})
MULTIPACK_RE = re.compile(
    r"\b[0-9]+\s*[xX]\s*[0-9]+(?:[.,][0-9]+)?\s*(?:ml|l|cl|dl|g|kg|oz)\b")
PACKAGE_STATE_RULE = (
    "package_state=true iff the standardized state carries a measured unit "
    "volume (a 'Volume:' field with a finite numeric value) OR an explicit "
    "numeric pack count (a numeric value under a pack-quantity key: "
    "Pack/Pack Size/Pack Count/Pack Quantity/Packaging/Units/Number of "
    "Items, or an 'NxM<unit>' multipack token); a 'Pack Type:' value alone "
    "is a package form, not a quantity, and does NOT qualify."
)


def _load_pairs_builder():
    """Reuse scripts/laya_metrics_pairs.py's composer verbatim (DRY: the
    pair state must be byte-identical to the metrics lane's composition)."""
    path = Path(__file__).resolve().parent / "laya_metrics_pairs.py"
    spec = importlib.util.spec_from_file_location("laya_metrics_pairs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PAIRS_BUILDER = _load_pairs_builder()
compose_side = _PAIRS_BUILDER.compose_side
compose_state = _PAIRS_BUILDER.compose_state


def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _field(attribute: str, name: str) -> str | None:
    for part in str(attribute).split(";"):
        if ":" in part:
            key, value = part.split(":", 1)
            if key.strip().lower() == name:
                return value.strip()
    return None


def package_state(attribute: str) -> bool:
    """The package_state rule above, over one standardized attribute string."""
    for part in str(attribute).split(";"):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key == "volume" and NUMERIC_VALUE_RE.match(value):
            return True
        if key in PACK_COUNT_KEYS and NUMERIC_VALUE_RE.match(value):
            return True
    return bool(MULTIPACK_RE.search(str(attribute)))


def _pack_type_only(attribute: str) -> bool:
    """Diagnostic: carries a Pack Type but no qualifying evidence."""
    return (not package_state(attribute)
            and bool(_field(attribute, "pack type")))


def _allocate(total: int, ratios: dict[str, float]) -> dict[str, int]:
    """Largest-remainder allocation of `total` at `ratios` (sums to total)."""
    raw = {key: total * ratios[key] for key in ratios}
    allocation = {key: int(value) for key, value in raw.items()}
    remainder = total - sum(allocation.values())
    for key in sorted(ratios, key=lambda k: (-(raw[k] - allocation[k]), k)):
        if remainder <= 0:
            break
        allocation[key] += 1
        remainder -= 1
    return allocation


def _split_ratios(listing_counts: dict[str, int]) -> dict[str, float]:
    """The documented ratios: the listing_pairs split proportions, so the
    state and gate-negative draws land in the same train/dev/test shape."""
    total = sum(listing_counts.get(key, 0) for key in SPLIT_ORDER)
    return {key: listing_counts.get(key, 0) / total for key in SPLIT_ORDER}


def _assign_splits(items: list, ratios: dict[str, float],
                   seed: int) -> dict[str, list]:
    """Deterministically shuffle (seed) and slice into the three splits."""
    allocation = _allocate(len(items), ratios)
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    out: dict[str, list] = {}
    cursor = 0
    for key in SPLIT_ORDER:
        out[key] = shuffled[cursor:cursor + allocation[key]]
        cursor += allocation[key]
    return out


def _stratified_sample(rows: list[dict], target: int, reason_of,
                       seed: int) -> list[dict]:
    """Proportional-by-stratum sample (largest remainder), deterministic."""
    by_reason: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_reason[reason_of(row)].append(row)
    for members in by_reason.values():
        members.sort(key=lambda r: (r["gtin1"], r["gtin2"]))
    ratios = {reason: len(members) / len(rows)
              for reason, members in by_reason.items()}
    allocation = _allocate(min(target, len(rows)), ratios)
    rng = random.Random(seed)
    sampled: list[dict] = []
    for reason in sorted(by_reason):
        count = min(allocation[reason], len(by_reason[reason]))
        sampled.extend(rng.sample(by_reason[reason], count))
    return sampled


def _record(state: str, questions: dict, expected: dict) -> dict:
    return {"state": state, "questions": questions, "expected": expected}


def _dump_line(record: dict) -> str:
    return json.dumps(record, ensure_ascii=False)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_prepared_bundle(path: Path) -> dict:
    """Read the prepared text bundle (gzip pickle of plain dict/ndarray/
    DataFrame values — no custom classes)."""
    with gzip.open(Path(path), "rb") as handle:
        return pickle.load(handle)


# Cleaned payload token prefix -> standardized attribute key. This is the
# inverse vocabulary mapping the six identity slice fields need: the
# prepared bundle's structured tail spells a field as `volume_ml_500` /
# `flavor_apple`, the reused `compose_side` reads `Volume: 500` /
# `Flavour: apple`. Longest/most-specific prefixes first so
# `sweetener_type_` never falls through to `sweetener_`.
_PAYLOAD_ATTR_PREFIXES: tuple[tuple[str, str], ...] = (
    ("sweetener_diet_", "Sweetener"),
    ("sweetener_type_", "Sweetener"),
    ("sweetening_", "Sweetening"),
    ("volume_ml_", "Volume"),
    ("pack_qty_", "Pack Size"),
    ("package_type_", "Pack Type"),
    ("flavor_", "Flavour"),
    ("carbonation_", "Carbonization"),
    ("sweetener_", "Sweetener"),
)
_PAYLOAD_ATTR_ORDER = (
    "Volume", "Pack Type", "Flavour", "Carbonization", "Sweetener",
)


def _attr_from_payload(text: str) -> str:
    """Cleaned payload text -> a standardized `Key: value` attribute string.

    Only the structured tail is mapped (the vocabulary `compose_side`
    parses); prose is dropped exactly as the existing pair cases drop it.
    Values keep their payload spelling (underscores are token separators
    except volume/pack, where `_` is the decimal point). No token is
    invented: every value is read off the payload.
    """
    values: dict[str, list[str]] = {}
    for token in str(text).split():
        for prefix, key in _PAYLOAD_ATTR_PREFIXES:
            if token.startswith(prefix):
                value = token[len(prefix):]
                if prefix in ("volume_ml_", "pack_qty_"):
                    value = value.replace("_", ".")
                if value:
                    values.setdefault(key, []).append(value)
                break
    return "; ".join(
        f"{key}: {', '.join(values[key])}"
        for key in _PAYLOAD_ATTR_ORDER if key in values)


def _side_from_payload(text: str) -> dict[str, str]:
    """One cleaned payload text -> the six-field side dict `compose_side`
    emits (reused, never re-implemented: the render lives in the metrics
    pairs composer)."""
    return compose_side(_attr_from_payload(text))


def _has_evidence(side: dict[str, str]) -> bool:
    """Any of the six identity slice fields measured on one side."""
    return any(side.get(field) for field in _PAIRS_BUILDER.SLICE_FIELDS)


def _ingest_masking_and_augmentation(
    *, bundle_path, labeled_pairs_path, by_gtin, existing_pair_states,
    questions,
) -> tuple[list[dict], list[dict], dict]:
    """Fold the pipeline's minted masking/augmentation into corpus records.

    Returns `(mask_records, aug_records, census)`. Deterministic: inputs are
    walked in file/pickle order, membership sets are never iterated, and the
    only RNG (split assignment) lives in the caller.
    """
    census: dict = {
        "bundle": str(bundle_path) if bundle_path else None,
        "labeled_pairs": (str(labeled_pairs_path)
                          if labeled_pairs_path else None),
        "bundle_mask_audit_total": 0,
        "bundle_hard_negative_audit_total": 0,
        "mask_positive_audits": 0,
        "mask_cases": 0,
        "mask_package_state_true": 0,
        "mask_package_state_false": 0,
        "mask_skipped_duplicate": 0,
        "aug_pair_audits": 0,
        "aug_pair_cases": 0,
        "aug_pair_positive": 0,
        "aug_pair_negative": 0,
        "aug_skipped_duplicate": 0,
        "aug_skipped_unrepresentable": 0,
        "labeled_pairs_rows": 0,
        "labeled_pairs_added": 0,
        "labeled_pairs_missing_gtin": 0,
        "labeled_pairs_skipped_duplicate": 0,
    }
    mask_records: list[dict] = []
    aug_records: list[dict] = []
    mask_seen: set[str] = set()
    aug_seen: set[str] = set(existing_pair_states)

    if bundle_path is not None and Path(bundle_path).is_file():
        bundle = _load_prepared_bundle(bundle_path)
        frame = bundle["df"]
        payload = bundle["payload"]
        mask_audits = list(bundle.get("mask_audit", []))
        hard_negatives = list(bundle.get("hard_negative_mask_audit", []))
        census["bundle_mask_audit_total"] = len(mask_audits)
        census["bundle_hard_negative_audit_total"] = len(hard_negatives)

        for audit in [*mask_audits, *hard_negatives]:
            population = audit.get("population")
            target_mode = audit.get("target_mode")
            if population == "positive" and target_mode != "swap_values":
                # a masked-positive STATE variant
                census["mask_positive_audits"] += 1
                state = audit.get("masked_text")
                if not state or state in mask_seen:
                    census["mask_skipped_duplicate"] += 1
                    continue
                anchor = int(audit["anchor_payload_idx"])
                attribute = str(frame.iloc[anchor]["attribute"])
                label = "true" if package_state(attribute) else "false"
                mask_records.append(
                    _record(state, questions, {"package_state": label}))
                mask_seen.add(state)
                census[f"mask_package_state_{label}"] += 1
                continue
            # a side-by-side augmentation PAIR (counterfactual/twin/minted)
            census["aug_pair_audits"] += 1
            label = "true" if population == "swap_counterpart" else "false"
            copy_text = (audit.get("masked_text")
                         or payload[int(audit["copy_payload_idx"])])
            pair_text = payload[int(audit["pair_payload_idx"])]
            side_one = _side_from_payload(copy_text)
            side_two = _side_from_payload(pair_text)
            if side_one == side_two or not (
                    _has_evidence(side_one) or _has_evidence(side_two)):
                census["aug_skipped_unrepresentable"] += 1
                continue
            state = compose_state(side_one, side_two)
            if state in aug_seen:
                census["aug_skipped_duplicate"] += 1
                continue
            aug_records.append(
                _record(state, questions, {"identity_claim": label}))
            aug_seen.add(state)
            census[f"aug_pair_{'positive' if label == 'true' else 'negative'}"] += 1

    if labeled_pairs_path is not None and Path(labeled_pairs_path).is_file():
        header, rows = _read_csv(labeled_pairs_path)
        if header != ["gtin1", "gtin2", "true_label"]:
            raise RuntimeError(f"labeled_pairs header drifted: {header}")
        census["labeled_pairs_rows"] = len(rows)
        for row in rows:
            one = by_gtin.get((row["gtin1"] or "").strip())
            two = by_gtin.get((row["gtin2"] or "").strip())
            if one is None or two is None:
                census["labeled_pairs_missing_gtin"] += 1
                continue
            label = "true" if int(row["true_label"]) == 1 else "false"
            state = compose_state(compose_side(one["attribute"]),
                                  compose_side(two["attribute"]))
            if state in aug_seen:
                census["labeled_pairs_skipped_duplicate"] += 1
                continue
            aug_records.append(
                _record(state, questions, {"identity_claim": label}))
            aug_seen.add(state)
            census["labeled_pairs_added"] += 1
            census[f"aug_pair_{'positive' if label == 'true' else 'negative'}"] += 1

    census["mask_cases"] = len(mask_records)
    census["aug_pair_cases"] = len(aug_records)
    return mask_records, aug_records, census


def build(*, catalog_path: Path = CATALOG_PATH, pairs_path: Path = PAIRS_PATH,
          gate_path: Path = GATE_PATH, question_path: Path = QUESTION_PATH,
          output_dir: Path = OUTPUT_DIR, seed: int = SEED,
          hard_no_cap: int = HARD_NO_CAP, bundle_path: Path | None = None,
          labeled_pairs_path: Path | None = None) -> dict:
    catalog_path = Path(catalog_path)
    pairs_path = Path(pairs_path)
    gate_path = Path(gate_path)
    question_path = Path(question_path)
    output_dir = Path(output_dir)

    catalog_header, catalog = _read_csv(catalog_path)
    required = ("sku_id", "gtin", "attribute")
    missing_cols = [key for key in required if key not in catalog_header]
    if missing_cols:
        raise RuntimeError(
            f"eligible_catalog is missing required columns {missing_cols}")
    by_sku = {row["sku_id"]: row for row in catalog}
    # A gtin may repeat across sku_id rows; pick the smallest sku_id so the
    # join is deterministic (documented; the catalog keeps both spellings).
    by_gtin: dict[str, dict] = {}
    for row in sorted(catalog, key=lambda r: r["sku_id"]):
        by_gtin.setdefault((row["gtin"] or "").strip(), row)

    questions = json.loads(question_path.read_text(encoding="utf-8"))["questions"]
    question_sha = _sha256(question_path)

    pairs_header, pairs = _read_csv(pairs_path)
    if pairs_header != ["sku_id1", "sku_id2", "label", "split"]:
        raise RuntimeError(f"listing_pairs header drifted: {pairs_header}")
    missing_skus = sorted(
        {sku for pair in pairs
         for sku in (pair["sku_id1"], pair["sku_id2"]) if sku not in by_sku})
    if missing_skus:
        raise RuntimeError(
            f"{len(missing_skus)} listing_pair sku_id(s) resolve to no "
            f"catalog row: {missing_skus[:10]}")

    listing_counts = Counter(pair["split"] for pair in pairs)
    ratios = _split_ratios(listing_counts)

    # ── STATE cases: one per catalog row ────────────────────────────────
    state_records = []
    state_pkg = Counter()
    for row in catalog:
        attribute = row["attribute"]
        labeled = package_state(attribute)
        state_pkg[str(labeled).lower()] += 1
        state_records.append(_record(
            attribute, questions, {"package_state": "true" if labeled
                                   else "false"}))
    state_splits = _assign_splits(state_records, ratios, seed)

    # ── PAIR cases: the ground-truth listing pairs ──────────────────────
    pair_by_split: dict[str, list[dict]] = {key: [] for key in SPLIT_ORDER}
    identity_labels = Counter()
    for pair in pairs:
        one = by_sku[pair["sku_id1"]]["attribute"]
        two = by_sku[pair["sku_id2"]]["attribute"]
        label = "true" if int(pair["label"]) == 1 else "false"
        identity_labels[label] += 1
        pair_by_split[pair["split"]].append(_record(
            compose_state(compose_side(one), compose_side(two)),
            questions, {"identity_claim": label}))
    positives = identity_labels["true"]
    listing_negatives = identity_labels["false"]

    # ── GATE cases: hard_no negatives + fallback quarantine ─────────────
    gate_header, gate = _read_csv(gate_path)
    hard_no, fallback = [], []
    dropped_missing = Counter()
    for row in gate:
        both = ((row["gtin1"] or "").strip() in by_gtin
                and (row["gtin2"] or "").strip() in by_gtin)
        if not both:
            dropped_missing[row["gate_decision"]] += 1
        if row["gate_decision"] == "hard_no":
            if both:
                hard_no.append(row)
        elif row["gate_decision"] == "fallback":
            fallback.append(row)

    target_total_negatives = min(hard_no_cap, positives)
    hard_no_target = max(0, target_total_negatives - listing_negatives)
    sampled = _stratified_sample(
        hard_no, hard_no_target, lambda row: row["gate_reason"], seed)
    gate_reason_sample = Counter(row["gate_reason"] for row in sampled)
    gate_splits = _assign_splits(sampled, ratios, seed)

    def _gate_state(row: dict) -> str:
        one = by_gtin[(row["gtin1"] or "").strip()]["attribute"]
        two = by_gtin[(row["gtin2"] or "").strip()]["attribute"]
        return compose_state(compose_side(one), compose_side(two))

    gate_records_by_split: dict[str, list[dict]] = {key: [] for key in SPLIT_ORDER}
    for key in SPLIT_ORDER:
        for row in gate_splits[key]:
            gate_records_by_split[key].append(_record(
                _gate_state(row), questions, {"identity_claim": "false"}))

    # ── MASKING + AUGMENTATION: fold the pipeline's minted data in ──────
    # The bundles/labeled pairs are opt-in (None in the hermetic builder
    # tests), so a caller without them reproduces the pre-growth corpus.
    existing_pair_states = {
        record["state"]
        for key in SPLIT_ORDER
        for record in (pair_by_split[key] + gate_records_by_split[key])
    }
    mask_records, aug_records, growth_census = (
        _ingest_masking_and_augmentation(
            bundle_path=bundle_path, labeled_pairs_path=labeled_pairs_path,
            by_gtin=by_gtin, existing_pair_states=existing_pair_states,
            questions=questions))
    mask_splits = _assign_splits(mask_records, ratios, seed)
    aug_splits = _assign_splits(aug_records, ratios, seed)

    # ── emit the splits (deterministic order: pairs, gate, aug, state,
    #    mask) ────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    split_sizes: dict[str, int] = {}
    split_counts: dict[str, dict] = {}
    split_paths: dict[str, Path] = {}
    for key in SPLIT_ORDER:
        path = output_dir / f"{key}.jsonl"
        lines = (pair_by_split[key] + gate_records_by_split[key]
                 + aug_splits[key] + state_splits[key] + mask_splits[key])
        with path.open("w", encoding="utf-8") as handle:
            for record in lines:
                handle.write(_dump_line(record) + "\n")
        split_paths[key] = path
        split_sizes[key] = len(lines)
        split_counts[key] = {
            "listing_positive": sum(
                1 for r in pair_by_split[key]
                if r["expected"]["identity_claim"] == "true"),
            "listing_negative": sum(
                1 for r in pair_by_split[key]
                if r["expected"]["identity_claim"] == "false"),
            "gate_negative": len(gate_records_by_split[key]),
            "aug_pair_positive": sum(
                1 for r in aug_splits[key]
                if r["expected"]["identity_claim"] == "true"),
            "aug_pair_negative": sum(
                1 for r in aug_splits[key]
                if r["expected"]["identity_claim"] == "false"),
            "state": len(state_splits[key]),
            "mask_state": len(mask_splits[key]),
        }

    # ── quarantine the fallback (UNKNOWN) rows ──────────────────────────
    unknown_path = output_dir / "unknown_pairs.csv"
    unknown_columns = list(gate_header) + [
        "gtin1_in_catalog", "gtin2_in_catalog", "attribute_pairs"]
    with unknown_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=unknown_columns,
                                lineterminator="\n")
        writer.writeheader()
        for row in fallback:
            in_one = (row["gtin1"] or "").strip() in by_gtin
            in_two = (row["gtin2"] or "").strip() in by_gtin
            out = dict(row)
            out["gtin1_in_catalog"] = str(in_one)
            out["gtin2_in_catalog"] = str(in_two)
            out["attribute_pairs"] = (
                _gate_state(row) if (in_one and in_two) else "")
            writer.writerow(out)

    # ── receipt ─────────────────────────────────────────────────────────
    receipt = {
        "seed": seed,
        "package_state_rule": PACKAGE_STATE_RULE,
        "identity_label_values": {"true": positives, "false": listing_negatives},
        "counts": {
            "state_cases": len(catalog),
            "state_package_state_true": state_pkg["true"],
            "state_package_state_false": state_pkg["false"],
            "state_pack_type_only_no_quantity": sum(
                1 for row in catalog if _pack_type_only(row["attribute"])),
            "listing_pairs_positive": positives,
            "listing_pairs_negative": listing_negatives,
            "gate_hard_no_available": len(hard_no),
            "gate_hard_no_sampled": len(sampled),
            "gate_fallback_quarantined": len(fallback),
            "gate_hard_no_cap": hard_no_cap,
            "identity_positive_total": positives,
            "identity_negative_total": listing_negatives + len(sampled),
            "dropped_missing": {
                "hard_no": dropped_missing["hard_no"],
                "fallback": dropped_missing["fallback"],
                "total": sum(dropped_missing.values()),
            },
            # ── the growth censuses (masking + augmentation) ────────────
            "mask_cases": growth_census["mask_cases"],
            "mask_package_state_true": growth_census["mask_package_state_true"],
            "mask_package_state_false": growth_census["mask_package_state_false"],
            "aug_pairs": growth_census["aug_pair_cases"],
            "aug_pairs_positive": growth_census["aug_pair_positive"],
            "aug_pairs_negative": growth_census["aug_pair_negative"],
            # corpus-wide identity prior AFTER folding the augmentation in:
            # the counterfactual/twin negatives are hard and numerous, so the
            # operator can see the balance (and tune the source) at a glance.
            "identity_positive_total_with_growth": (
                positives + growth_census["aug_pair_positive"]),
            "identity_negative_total_with_growth": (
                listing_negatives + len(sampled)
                + growth_census["aug_pair_negative"]),
        },
        "growth": growth_census,
        "gate_reason_sample": dict(sorted(gate_reason_sample.items())),
        "split_ratios": {key: ratios[key] for key in SPLIT_ORDER},
        "split_sizes": split_sizes,
        "split_counts": split_counts,
        "question_schema_sha256": question_sha,
        "sha256": {
            **{f"{key}.jsonl": _sha256(split_paths[key])
               for key in SPLIT_ORDER},
            "unknown_pairs.csv": _sha256(unknown_path),
        },
    }
    # One stable digest over the whole corpus body (the split files in
    # frozen SPLIT_ORDER), for the report and the determinism test.
    corpus_digest = hashlib.sha256()
    for key in SPLIT_ORDER:
        corpus_digest.update(split_paths[key].read_bytes())
    receipt["corpus_sha256"] = corpus_digest.hexdigest()
    receipt_path = output_dir / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return receipt


def main() -> None:
    for path in (CATALOG_PATH, PAIRS_PATH, GATE_PATH, QUESTION_PATH):
        if not Path(path).is_file():
            raise FileNotFoundError(f"required source missing: {path}")
    # Fold the pipeline's minted masking/augmentation in when the artifacts
    # are present (read, never invented); absent sources are reported, not
    # fabricated. `build()` still runs the base corpus without them.
    bundle = BUNDLE_PATH if BUNDLE_PATH.is_file() else None
    labeled = LABELED_PAIRS_PATH if LABELED_PAIRS_PATH.is_file() else None
    if bundle is None:
        print(f"[laya-build-dataset] masking/augmentation source absent: "
              f"{BUNDLE_PATH} (base corpus only)")
    if labeled is None:
        print(f"[laya-build-dataset] labeled pairs source absent: "
              f"{LABELED_PAIRS_PATH} (base corpus only)")
    receipt = build(bundle_path=bundle, labeled_pairs_path=labeled)
    counts = receipt["counts"]
    print(
        "[laya-build-dataset] states=%d (pkg true=%d false=%d) "
        "listing_pairs=%d+,%d- gate_hard_no_sampled=%d/%d "
        "fallback_quarantined=%d dropped_missing=%d"
        % (counts["state_cases"], counts["state_package_state_true"],
           counts["state_package_state_false"], counts["listing_pairs_positive"],
           counts["listing_pairs_negative"], counts["gate_hard_no_sampled"],
           counts["gate_hard_no_available"], counts["gate_fallback_quarantined"],
           counts["dropped_missing"]["total"]))
    print(
        "[laya-build-dataset] growth: mask_cases=%d (pkg true=%d false=%d) "
        "aug_pairs=%d (+%d/-%d) labeled_pairs_added=%d"
        % (counts["mask_cases"], counts["mask_package_state_true"],
           counts["mask_package_state_false"], counts["aug_pairs"],
           counts["aug_pairs_positive"], counts["aug_pairs_negative"],
           receipt["growth"]["labeled_pairs_added"]))
    print("[laya-build-dataset] split_sizes="
          + json.dumps(receipt["split_sizes"]))
    print("[laya-build-dataset] split_counts="
          + json.dumps(receipt["split_counts"]))
    print("[laya-build-dataset] growth=" + json.dumps(receipt["growth"]))
    for name, digest in receipt["sha256"].items():
        print(f"[laya-build-dataset] sha256 {name} {digest}")
    print(f"[laya-build-dataset] corpus_sha256={receipt['corpus_sha256']}")
    print("[laya-build-dataset] -> " + str(OUTPUT_DIR / "receipt.json"))


if __name__ == "__main__":
    main()
