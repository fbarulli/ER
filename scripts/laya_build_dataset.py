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

Deterministic: seed 1729, no wall-clock, no set-iteration order leaks into
the output; a rerun reproduces every byte.

Outputs (data/laya/): train.jsonl, dev.jsonl, test.jsonl, unknown_pairs.csv,
receipt.json.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
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


def build(*, catalog_path: Path = CATALOG_PATH, pairs_path: Path = PAIRS_PATH,
          gate_path: Path = GATE_PATH, question_path: Path = QUESTION_PATH,
          output_dir: Path = OUTPUT_DIR, seed: int = SEED,
          hard_no_cap: int = HARD_NO_CAP) -> dict:
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

    # ── emit the splits (deterministic order: pairs, gate, state) ───────
    output_dir.mkdir(parents=True, exist_ok=True)
    split_sizes: dict[str, int] = {}
    split_counts: dict[str, dict] = {}
    split_paths: dict[str, Path] = {}
    for key in SPLIT_ORDER:
        path = output_dir / f"{key}.jsonl"
        lines = (pair_by_split[key] + gate_records_by_split[key]
                 + state_splits[key])
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
            "state": len(state_splits[key]),
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
        },
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
    receipt_path = output_dir / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return receipt


def main() -> None:
    for path in (CATALOG_PATH, PAIRS_PATH, GATE_PATH, QUESTION_PATH):
        if not Path(path).is_file():
            raise FileNotFoundError(f"required source missing: {path}")
    receipt = build()
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
    print("[laya-build-dataset] split_sizes="
          + json.dumps(receipt["split_sizes"]))
    for name, digest in receipt["sha256"].items():
        print(f"[laya-build-dataset] sha256 {name} {digest}")
    print("[laya-build-dataset] -> " + str(OUTPUT_DIR / "receipt.json"))


if __name__ == "__main__":
    main()
