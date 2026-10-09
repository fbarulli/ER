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

Config (config/laya.question.json, optional top-level "corpus" block — the
config SSOT for this builder; every key defaults to the historical behaviour,
so a schema without the block reproduces the landed corpus byte-for-byte):
  * seed                            -> the deterministic shuffle seed;
  * hard_no_cap                     -> the gate hard_no negative ceiling;
  * identity_negative_target_ratio  -> negatives-per-positive target applied
    AFTER the minted growth (the corpus ran ~1:14); only pipeline-MINTED
    negatives are thinned, ground-truth negatives are never dropped;
  * sources                         -> optional path overrides (catalog,
    pairs, gate, bundle, labeled_pairs, output_dir) resolved against ROOT.

Outputs (data/laya/): train.jsonl, dev.jsonl, test.jsonl, unknown_pairs.csv,
receipt.json.
"""
from __future__ import annotations

import csv
import gzip
from core.portable_archive import ByteCount
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

# ── the corpus-composition config (config-owned; every knob defaults to the
#    historical behaviour) ───────────────────────────────────────────────────
# `config/laya.question.json` is ALREADY this builder's schema source, so its
# optional top-level `corpus` block owns the composition knobs rather than
# code literals (config SSOT). A schema without the block (or with every key
# null) resolves to the defaults below and reproduces the landed corpus
# byte-for-byte. Keys:
#   * seed                          — the deterministic shuffle seed;
#   * hard_no_cap                   — gate hard_no negative ceiling;
#   * identity_negative_target_ratio— negatives-per-positive TARGET applied
#     AFTER the minted growth is folded in (item: the corpus ran ~1:14). Only
#     pipeline-MINTED negatives are thinned to reach it; ground-truth
#     negatives (listing pairs, gate hard_no, labeled pairs) are never
#     dropped. null = no cap (the historical behaviour);
#   * sources                       — optional path overrides for the
#     hardcoded defaults (portable names relative to the repo root).
CORPUS_CONFIG_DEFAULTS: dict = {
    "seed": None,
    "hard_no_cap": None,
    "identity_negative_target_ratio": None,
    "sources": None,
}
CORPUS_SOURCE_KEYS = ("catalog", "pairs", "gate", "bundle", "labeled_pairs",
                      "output_dir")
CORPUS_SOURCE_DEFAULTS = {
    "catalog": CATALOG_PATH,
    "pairs": PAIRS_PATH,
    "gate": GATE_PATH,
    "bundle": BUNDLE_PATH,
    "labeled_pairs": LABELED_PAIRS_PATH,
    "output_dir": OUTPUT_DIR,
}

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


# ── per-field / per-attribute pair labels (reuse the composed sides) ────────
# Every pair label below is read off ONE of two things the builder already
# has: the six-field side literals `compose_side` emits (the reused metrics
# composer, never re-implemented) or the standardized attribute string. No
# label is invented, and a question the caller's schema does not declare is
# simply not emitted (the membership gate in `_pair_expected`).
PAIR_FIELDS = _PAIRS_BUILDER.SLICE_FIELDS
FIELD_SAME_QIDS = tuple(f"field_same:{field}" for field in PAIR_FIELDS)

# The nine bounded gate-reason families (data/gate_results.csv carries a
# free-form `gate_reason`; the family is its text before the first ':'
# -- read off the source, never a new taxonomy). An unrecognized reason
# falls back to `unclassified` and is counted, never silently dropped.
GATE_REASON_FAMILIES: dict[str, str] = {
    "Pack blocker": "pack_blocker",
    "Critical attribute mismatch": "critical_attribute",
    "Package material mismatch": "package_material",
    "Contradictory source attribute evidence": "contradictory_evidence",
    "Declared product identity differs or is incomplete":
        "declared_identity_incomplete",
    "Missing flavor evidence with differing supporting attributes":
        "missing_flavor_evidence",
    "Low raw volume confidence": "low_volume_confidence",
    "Low raw pack confidence": "low_pack_confidence",
    "Known critical attributes compatible": "compatible",
}


def _field_same_label(side_one: dict[str, str], side_two: dict[str, str],
                      field: str) -> str:
    """same / different / unknown for one slice field, from the side literals.

    Both sides measured and equal -> same; both measured and unequal ->
    different; either side unmeasured ('') -> unknown (never guessed).
    """
    one, two = side_one.get(field, ""), side_two.get(field, "")
    if one and two:
        return "same" if one == two else "different"
    return "unknown"


def _has_evidence(side: dict[str, str]) -> bool:
    """Any of the six identity slice fields measured on one side."""
    return any(side.get(field) for field in PAIR_FIELDS)


def _package_signature(attribute: str) -> tuple:
    """(volume, pack-count, multipack) evidence read off the attribute.

    Reuses the same keys the `package_state` rule reads (Volume plus the
    PACK_COUNT_KEYS / MULTIPACK_RE vocabulary); an unmeasured field is
    simply absent, never zero.
    """
    parts: list[tuple[str, str]] = []
    for part in str(attribute).split(";"):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key, value = key.strip().lower(), value.strip()
        if key == "volume" and NUMERIC_VALUE_RE.match(value):
            parts.append(("volume", value))
        elif key in PACK_COUNT_KEYS and NUMERIC_VALUE_RE.match(value):
            parts.append(("pack", value))
    match = MULTIPACK_RE.search(str(attribute))
    if match:
        parts.append(("multipack", match.group(0).lower().replace(" ", "")))
    return tuple(sorted(parts))


def pack_volume_equal(attr_one: str, attr_two: str) -> str:
    """true iff both sides carry package-quantity evidence and it agrees."""
    signature_one = _package_signature(attr_one)
    signature_two = _package_signature(attr_two)
    return "true" if (signature_one and signature_two
                      and signature_one == signature_two) else "false"


def pack_format_equivalent(attr_one: str, attr_two: str) -> str:
    """true iff both sides measure a Pack Type and the forms agree."""
    one = (_field(attr_one, "pack type") or "").strip().lower()
    two = (_field(attr_two, "pack type") or "").strip().lower()
    return "true" if (one and two and one == two) else "false"


def evidence_sufficient(side_one: dict[str, str],
                        side_two: dict[str, str]) -> str:
    """true iff both sides carry at least one measured slice field."""
    return "true" if (_has_evidence(side_one) and _has_evidence(side_two)) \
        else "false"


def same_brand_only(brand_one: str | None, brand_two: str | None,
                    identity_label: str | None) -> str | None:
    """true iff the brands agree and the pair is NOT the same item.

    `None` (omit the label) when either brand is unknown: the state does not
    carry brand evidence, so the label is only emitted where the catalog
    source supplies both brands AND the GTIN truth supplies the identity
    label.
    """
    if not brand_one or not brand_two or identity_label is None:
        return None
    same = brand_one.strip().lower() == brand_two.strip().lower()
    return "true" if (same and identity_label == "false") else "false"


def _difficulty_slice(side_one: dict[str, str],
                      side_two: dict[str, str]) -> str:
    """A pair's difficulty from the same field-level agreement the questions
    use: how many measured fields disagree (never invented from a model)."""
    if not (_has_evidence(side_one) and _has_evidence(side_two)):
        return "insufficient"
    differing = sum(1 for field in PAIR_FIELDS
                    if _field_same_label(side_one, side_two, field) == "different")
    if differing == 0:
        return "all_same"
    if differing == 1:
        return "one_diff"
    return "multi_diff"


def _primary_attribute(side_one: dict[str, str],
                       side_two: dict[str, str]) -> str:
    """The row's attribute tag: the first differing measured field, else the
    first measured field in the frozen slice order, else 'none'."""
    for field in PAIR_FIELDS:
        if _field_same_label(side_one, side_two, field) == "different":
            return field
    for field in PAIR_FIELDS:
        if side_one.get(field) or side_two.get(field):
            return field
    return "none"


def gate_reason_family(reason: str) -> str:
    """Free-form gate_reason -> its bounded family (text before ':')."""
    prefix = str(reason).split(":", 1)[0].strip()
    return GATE_REASON_FAMILIES.get(prefix, "unclassified")


def _corpus_config(document: dict, override: dict | None = None) -> dict:
    """Resolve the `corpus` composition block from the question-schema file.

    `override` (when given) replaces the file's block wholesale — the explicit
    caller wins over config SSOT. Unknown keys fail loud (a typo must never
    silently fall back to a default), a null value means "keep the default",
    and a config with no block resolves to `CORPUS_CONFIG_DEFAULTS` exactly.
    """
    config = dict(CORPUS_CONFIG_DEFAULTS)
    block = override if override is not None else (document.get("corpus") or {})
    if not isinstance(block, dict):
        raise ValueError(
            f"laya corpus config must be a mapping, got {type(block).__name__}")
    unknown = sorted(set(block) - set(config))
    if unknown:
        raise ValueError(f"unknown laya corpus knob(s): {unknown}")
    for key, value in block.items():
        if value is not None:
            config[key] = value
    if isinstance(config["sources"], dict):
        unknown = sorted(set(config["sources"]) - set(CORPUS_SOURCE_KEYS))
        if unknown:
            raise ValueError(f"unknown laya corpus source(s): {unknown}")
    return config


def resolve_corpus_sources(config: dict) -> dict:
    """The builder's input paths: the hardcoded defaults + config overrides.

    A config path is resolved against the repo root (`ROOT`), never the cwd,
    so the same config resolves identically from any working directory.
    """
    sources = dict(CORPUS_SOURCE_DEFAULTS)
    for key, value in (config.get("sources") or {}).items():
        if value is None:
            continue
        candidate = Path(value)
        sources[key] = candidate if candidate.is_absolute() else ROOT / candidate
    return sources


def corpus_config_is_default(config: dict) -> bool:
    """True when NO knob (nor any source path) is set — the null block case."""
    if any(config.get(key) is not None for key in
           ("seed", "hard_no_cap", "identity_negative_target_ratio")):
        return False
    return not any(value is not None
                   for value in (config.get("sources") or {}).values())


def _rebalance_identity_negatives(
    aug_records: list[dict], origins: dict[str, str], *,
    positives_total: int, ground_negative_total: int, ratio: float | None,
    seed: int,
) -> tuple[list[dict], dict]:
    """Thin the pipeline-MINTED identity negatives toward `ratio` x positives.

    `ratio is None` (the default) is a no-op, so the landed corpus is
    reproduced byte-for-byte. When set, only records the growth fold MINTED
    (origin `bundle`: counterfactual/twin/swap augmentation) may be dropped;
    ground-truth negatives (listing pairs, gate `hard_no`, labeled pairs) are
    never touched. The drop is deterministic (`seed` over a sorted state
    list), so a rerun reproduces every byte, and a target already satisfied
    drops nothing.
    """
    if ratio is None:
        return aug_records, {"enabled": False}
    if ratio < 0:
        raise ValueError(
            f"identity_negative_target_ratio must be >= 0, got {ratio!r}")
    minted = sorted(
        record["state"] for record in aug_records
        if origins.get(record["state"]) == "bundle"
        and record["expected"].get("identity_claim") == "false")
    target_negatives = int(round(ratio * positives_total))
    keep = max(0, target_negatives - ground_negative_total)
    census = {
        "enabled": True,
        "target_ratio": ratio,
        "positives_total": positives_total,
        "ground_truth_negatives": ground_negative_total,
        "target_negatives": target_negatives,
        "minted_negatives_available": len(minted),
        "minted_negatives_kept": min(len(minted), keep),
        "minted_negatives_dropped": max(0, len(minted) - keep),
    }
    if len(minted) <= keep:
        return aug_records, census
    dropped = set(random.Random(seed).sample(minted, len(minted) - keep))
    kept = [record for record in aug_records if record["state"] not in dropped]
    return kept, census


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


def _record(state: str, questions: dict, expected: dict, *,
            difficulty_slice: str, gate_reason: str = "",
            attribute: str = "none") -> dict:
    """One corpus case: the laya contract keys PLUS the traceability tags.

    `difficulty_slice`/`gate_reason`/`attribute` ride at the top level: the
    laya trainer (`items_from_rows`) reads only `state`/`questions`/`gold`/
    `expected`, so the extra keys never affect training, and the eval report
    reads them back to slice accuracy/ECE without re-deriving membership.
    """
    return {"state": state, "questions": questions, "expected": expected,
            "difficulty_slice": difficulty_slice,
            "gate_reason": gate_reason, "attribute": attribute}


def _pair_expected(questions: dict, side_one: dict[str, str],
                   side_two: dict[str, str], *, attr_one: str | None = None,
                   attr_two: str | None = None,
                   identity: str | None = None,
                   brand_one: str | None = None,
                   brand_two: str | None = None,
                   counterfactual: str | None = None,
                   gate_verdict: str | None = None,
                   gate_reason: str | None = None) -> dict:
    """Every pair label this row's sources support, gated by the schema.

    A label is emitted only when the caller's `questions` dict declares the
    qid (the hermetic fixtures with a 3-question schema stay byte-identical)
    AND the source supplies the truth. All field labels come off the SAME
    composed sides the state was rendered from.
    """
    expected: dict[str, str] = {}

    def put(qid: str, label: str | None) -> None:
        if label is not None and qid in questions:
            expected[qid] = label

    put("identity_claim", identity)
    put("counterfactual", counterfactual)
    for field in PAIR_FIELDS:
        put(f"field_same:{field}",
            _field_same_label(side_one, side_two, field))
    if attr_one is not None and attr_two is not None:
        put("pack_volume_equal", pack_volume_equal(attr_one, attr_two))
        put("pack_format_equivalent",
            pack_format_equivalent(attr_one, attr_two))
    put("evidence_sufficient", evidence_sufficient(side_one, side_two))
    put("same_brand_only", same_brand_only(brand_one, brand_two, identity))
    put("gate_verdict", gate_verdict)
    put("gate_reason", gate_reason)
    return expected


def _pair_meta(side_one: dict[str, str], side_two: dict[str, str], *,
               gate_reason: str = "") -> dict:
    return {"difficulty_slice": _difficulty_slice(side_one, side_two),
            "gate_reason": gate_reason,
            "attribute": _primary_attribute(side_one, side_two)}


def _single_meta(attribute: str) -> dict:
    """A single-state row's tags: no pair slice, the first measured field."""
    side = compose_side(attribute)
    for field in PAIR_FIELDS:
        if side.get(field):
            return {"difficulty_slice": "single_state", "gate_reason": "",
                    "attribute": field}
    return {"difficulty_slice": "single_state", "gate_reason": "",
            "attribute": "none"}


def _render_triple_side(side: dict[str, str]) -> str:
    """One listing's six-field literals for the better_match state."""
    return "[" + ", ".join(
        f"{field}:{side.get(field) or '-'}" for field in PAIR_FIELDS) + "]"


def better_match_records(pairs: list[dict], by_sku: dict[str, dict],
                         questions: dict) -> list[dict]:
    """Pairwise `better_match` (choice(2)) cases from the GTIN truth.

    For every confirmed-different listing pair (A, B) that also has a
    confirmed-same partner C for A, the state carries the anchor A and two
    candidates (B, C); the answer is the candidate whose GTIN equals A's.
    The two candidates are placed deterministically by GTIN (the smaller
    GTIN is `candidate_1`) so position is not a free signal. Emitted only
    when the schema declares `better_match`.
    """
    if "better_match" not in questions:
        return []
    same_by_sku: dict[str, dict[str, dict]] = {}
    for pair in pairs:
        if int(pair["label"]) != 1:
            continue
        for anchor, partner in ((pair["sku_id1"], pair["sku_id2"]),
                                (pair["sku_id2"], pair["sku_id1"])):
            same_by_sku.setdefault(anchor, {})[partner] = pair
    records: list[dict] = []
    seen: set[str] = set()
    for pair in pairs:
        if int(pair["label"]) != 0:
            continue
        anchor = pair["sku_id1"]
        different = pair["sku_id2"]
        for same in sorted(same_by_sku.get(anchor, {})):
            if same == different:
                continue
            gtin_different = by_sku[different]["gtin"]
            gtin_same = by_sku[same]["gtin"]
            # deterministic candidate order by GTIN; the GTIN truth decides
            first_is_same = gtin_same <= gtin_different
            cand_one = same if first_is_same else different
            cand_two = different if first_is_same else same
            side_anchor = compose_side(by_sku[anchor]["attribute"])
            side_one = compose_side(by_sku[cand_one]["attribute"])
            side_two = compose_side(by_sku[cand_two]["attribute"])
            state = (f"anchor: {_render_triple_side(side_anchor)}; "
                     f"candidate_1: {_render_triple_side(side_one)}; "
                     f"candidate_2: {_render_triple_side(side_two)}")
            if state in seen:
                continue
            seen.add(state)
            records.append(_record(
                state, questions,
                {"better_match": "candidate_1" if first_is_same
                 else "candidate_2"},
                difficulty_slice="pairwise",
                attribute=_primary_attribute(side_one, side_two)))
    return records


def _dump_line(record: dict) -> str:
    return json.dumps(record, ensure_ascii=False)


def _file_bytes(path: Path) -> int:
    return Path(path).stat().st_size


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


def _ingest_masking_and_augmentation(
    *, bundle_path, labeled_pairs_path, by_gtin, existing_pair_states,
    questions,
) -> tuple[list[dict], list[dict], dict, dict]:
    """Fold the pipeline's minted masking/augmentation into corpus records.

    Returns `(mask_records, aug_records, census, origins)`. `origins` maps each
    augmentation PAIR state to its source ("bundle" for the pipeline-minted
    counterfactual/twin/swap negatives, "labeled_pairs" for the ground-truth
    labeled pairs), so the rebalance knob can thin ONLY the minted population.
    Deterministic: inputs are walked in file/pickle order, membership sets are
    never iterated, and the only RNG (split assignment) lives in the caller.
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
        "aug_pair_counterfactual": 0,
        "aug_skipped_duplicate": 0,
        "aug_skipped_unrepresentable": 0,
        "labeled_pairs_rows": 0,
        "labeled_pairs_added": 0,
        "labeled_pairs_missing_gtin": 0,
        "labeled_pairs_skipped_duplicate": 0,
    }
    mask_records: list[dict] = []
    aug_records: list[dict] = []
    origins: dict[str, str] = {}
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
                expected = {"package_state": label}
                if "evidence_sufficient" in questions:
                    side = compose_side(attribute)
                    expected["evidence_sufficient"] = (
                        "true" if _has_evidence(side) else "false")
                mask_records.append(
                    _record(state, questions, expected,
                            **_single_meta(attribute)))
                mask_seen.add(state)
                census[f"mask_package_state_{label}"] += 1
                continue
            # a side-by-side augmentation PAIR (counterfactual/twin/minted)
            census["aug_pair_audits"] += 1
            label = "true" if population == "swap_counterpart" else "false"
            is_counterfactual = "true" if target_mode == "counterfactual" \
                else "false"
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
            expected = _pair_expected(
                questions, side_one, side_two,
                attr_one=_attr_from_payload(copy_text),
                attr_two=_attr_from_payload(pair_text),
                identity=label, counterfactual=is_counterfactual)
            if label == "true" and "identity_claim" not in expected:
                # the fixture schema labelled no identity_claim: keep the
                # historical positive/negative census meaningful anyway.
                pass
            aug_records.append(
                _record(state, questions, expected,
                        **_pair_meta(side_one, side_two)))
            origins[state] = "bundle"
            aug_seen.add(state)
            census[f"aug_pair_{'positive' if label == 'true' else 'negative'}"] += 1
            if is_counterfactual == "true":
                census["aug_pair_counterfactual"] = (
                    census.get("aug_pair_counterfactual", 0) + 1)

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
            side_one = compose_side(one["attribute"])
            side_two = compose_side(two["attribute"])
            state = compose_state(side_one, side_two)
            if state in aug_seen:
                census["labeled_pairs_skipped_duplicate"] += 1
                continue
            expected = _pair_expected(
                questions, side_one, side_two,
                attr_one=one["attribute"], attr_two=two["attribute"],
                identity=label,
                brand_one=one.get("brand"), brand_two=two.get("brand"),
                counterfactual="false")
            aug_records.append(
                _record(state, questions, expected,
                        **_pair_meta(side_one, side_two)))
            origins[state] = "labeled_pairs"
            aug_seen.add(state)
            census["labeled_pairs_added"] += 1
            census[f"aug_pair_{'positive' if label == 'true' else 'negative'}"] += 1

    census["mask_cases"] = len(mask_records)
    census["aug_pair_cases"] = len(aug_records)
    return mask_records, aug_records, census, origins


def build(*, catalog_path: Path = CATALOG_PATH, pairs_path: Path = PAIRS_PATH,
          gate_path: Path = GATE_PATH, question_path: Path = QUESTION_PATH,
          output_dir: Path = OUTPUT_DIR, seed: int | None = None,
          hard_no_cap: int | None = None, bundle_path: Path | None = None,
          labeled_pairs_path: Path | None = None,
          identity_negative_target_ratio: float | None = None,
          corpus_config: dict | None = None) -> dict:
    """Build the corpus.

    Every composition knob resolves `explicit argument > config/laya.question
    .json "corpus" block > historical default`, so an `int`/`float` argument
    wins over config SSOT and a caller that passes NOTHING (or a schema with no
    block) reproduces the landed corpus byte-for-byte.
    """
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

    document = json.loads(question_path.read_text(encoding="utf-8"))
    questions = document["questions"]
    question_size = _file_bytes(question_path)
    # corpus composition: explicit argument > config block > historical default.
    corpus = _corpus_config(document, corpus_config)
    if seed is None:
        seed = corpus["seed"] if corpus["seed"] is not None else SEED
    if hard_no_cap is None:
        hard_no_cap = (corpus["hard_no_cap"]
                       if corpus["hard_no_cap"] is not None else HARD_NO_CAP)
    if identity_negative_target_ratio is None:
        identity_negative_target_ratio = corpus[
            "identity_negative_target_ratio"]

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
        expected = {"package_state": "true" if labeled else "false"}
        if "evidence_sufficient" in questions:
            expected["evidence_sufficient"] = (
                "true" if _has_evidence(compose_side(attribute)) else "false")
        state_records.append(_record(attribute, questions, expected,
                                     **_single_meta(attribute)))
    state_splits = _assign_splits(state_records, ratios, seed)

    # ── PAIR cases: the ground-truth listing pairs ──────────────────────
    pair_by_split: dict[str, list[dict]] = {key: [] for key in SPLIT_ORDER}
    identity_labels = Counter()
    for pair in pairs:
        row_one = by_sku[pair["sku_id1"]]
        row_two = by_sku[pair["sku_id2"]]
        one, two = row_one["attribute"], row_two["attribute"]
        side_one, side_two = compose_side(one), compose_side(two)
        label = "true" if int(pair["label"]) == 1 else "false"
        identity_labels[label] += 1
        expected = _pair_expected(
            questions, side_one, side_two, attr_one=one, attr_two=two,
            identity=label, brand_one=row_one.get("brand"),
            brand_two=row_two.get("brand"), counterfactual="false")
        pair_by_split[pair["split"]].append(_record(
            compose_state(side_one, side_two), questions, expected,
            difficulty_slice=_difficulty_slice(side_one, side_two),
            attribute=_primary_attribute(side_one, side_two)))
    positives = identity_labels["true"]
    listing_negatives = identity_labels["false"]

    # ── GATE cases: hard_no negatives + proceed + fallback quarantine ───
    gate_header, gate = _read_csv(gate_path)
    hard_no, proceed, fallback = [], [], []
    dropped_missing = Counter()
    for row in gate:
        both = ((row["gtin1"] or "").strip() in by_gtin
                and (row["gtin2"] or "").strip() in by_gtin)
        if not both:
            dropped_missing[row["gate_decision"]] += 1
        if row["gate_decision"] == "hard_no":
            if both:
                hard_no.append(row)
        elif row["gate_decision"] == "proceed":
            if both:
                proceed.append(row)
        elif row["gate_decision"] == "fallback":
            fallback.append(row)

    target_total_negatives = min(hard_no_cap, positives)
    hard_no_target = max(0, target_total_negatives - listing_negatives)
    sampled = _stratified_sample(
        hard_no, hard_no_target, lambda row: row["gate_reason"], seed)
    gate_reason_sample = Counter(row["gate_reason"] for row in sampled)
    gate_splits = _assign_splits(sampled, ratios, seed)
    # Every joinable `proceed` gate row rides the corpus for its gate verdict
    # / reason labels (identity_claim stays unlabelled: the gate verdict is
    # not a GTIN truth). Deterministic row order preserved, then split.
    proceed_splits = _assign_splits(list(proceed), ratios, seed)
    gate_reason_families = Counter(
        gate_reason_family(row["gate_reason"]) for row in gate)

    def _gate_sides(row: dict) -> tuple[dict, dict]:
        one = by_gtin[(row["gtin1"] or "").strip()]["attribute"]
        two = by_gtin[(row["gtin2"] or "").strip()]["attribute"]
        return compose_side(one), compose_side(two)

    def _gate_state(row: dict) -> str:
        side_one, side_two = _gate_sides(row)
        return compose_state(side_one, side_two)

    gate_records_by_split: dict[str, list[dict]] = {key: [] for key in SPLIT_ORDER}
    for key in SPLIT_ORDER:
        for row in gate_splits[key]:
            side_one, side_two = _gate_sides(row)
            family = gate_reason_family(row["gate_reason"])
            expected = _pair_expected(
                questions, side_one, side_two, identity="false",
                gate_verdict="hard_no", gate_reason=family,
                attr_one=by_gtin[(row["gtin1"] or "").strip()]["attribute"],
                attr_two=by_gtin[(row["gtin2"] or "").strip()]["attribute"],
                brand_one=by_gtin[(row["gtin1"] or "").strip()].get("brand"),
                brand_two=by_gtin[(row["gtin2"] or "").strip()].get("brand"),
                counterfactual="false")
            gate_records_by_split[key].append(_record(
                compose_state(side_one, side_two), questions, expected,
                difficulty_slice=_difficulty_slice(side_one, side_two),
                gate_reason=family,
                attribute=_primary_attribute(side_one, side_two)))

    proceed_records_by_split: dict[str, list[dict]] = {
        key: [] for key in SPLIT_ORDER}
    for key in SPLIT_ORDER:
        for row in proceed_splits[key]:
            side_one, side_two = _gate_sides(row)
            family = gate_reason_family(row["gate_reason"])
            expected = _pair_expected(
                questions, side_one, side_two,
                gate_verdict="proceed", gate_reason=family,
                attr_one=by_gtin[(row["gtin1"] or "").strip()]["attribute"],
                attr_two=by_gtin[(row["gtin2"] or "").strip()]["attribute"],
                counterfactual="false")
            proceed_records_by_split[key].append(_record(
                compose_state(side_one, side_two), questions, expected,
                difficulty_slice=_difficulty_slice(side_one, side_two),
                gate_reason=family,
                attribute=_primary_attribute(side_one, side_two)))

    # ── MASKING + AUGMENTATION: fold the pipeline's minted data in ──────
    # The bundles/labeled pairs are opt-in (None in the hermetic builder
    # tests), so a caller without them reproduces the pre-growth corpus.
    existing_pair_states = {
        record["state"]
        for key in SPLIT_ORDER
        for record in (pair_by_split[key] + gate_records_by_split[key])
    }
    mask_records, aug_records, growth_census, aug_origins = (
        _ingest_masking_and_augmentation(
            bundle_path=bundle_path, labeled_pairs_path=labeled_pairs_path,
            by_gtin=by_gtin, existing_pair_states=existing_pair_states,
            questions=questions))
    # ── IDENTITY REBALANCE (config-owned; no-op by default) ─────────────
    # The minted counterfactual/twin negatives dominate the identity prior
    # (~1:14). The knob thins ONLY that minted population, deterministically,
    # never the ground-truth negatives.
    identity_negatives_growth = growth_census["aug_pair_negative"]
    positives_growth = growth_census["aug_pair_positive"]
    minted_negatives = sum(
        1 for record in aug_records
        if aug_origins.get(record["state"]) == "bundle"
        and record["expected"].get("identity_claim") == "false")
    # Ground truth = every negative that is NOT pipeline-minted: the listing
    # pairs, the gate hard_no sample and the labeled pairs.
    ground_negative_total = (listing_negatives + len(sampled)
                             + identity_negatives_growth - minted_negatives)
    aug_records, rebalance_census = _rebalance_identity_negatives(
        aug_records, aug_origins,
        positives_total=positives + positives_growth,
        ground_negative_total=ground_negative_total,
        ratio=identity_negative_target_ratio, seed=seed)
    identity_negatives_emitted = sum(
        1 for record in aug_records
        if record["expected"].get("identity_claim") == "false")
    identity_negatives_total = (listing_negatives + len(sampled)
                                + identity_negatives_emitted)
    mask_splits = _assign_splits(mask_records, ratios, seed)
    aug_splits = _assign_splits(aug_records, ratios, seed)

    # ── BETTER_MATCH cases: pairwise choice(2) from the GTIN truth ──────
    better_records = better_match_records(pairs, by_sku, questions)
    better_splits = _assign_splits(better_records, ratios, seed)

    # ── emit the splits (deterministic order: pairs, gate, proceed, aug,
    #    better_match, state, mask) ───────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    split_sizes: dict[str, int] = {}
    split_counts: dict[str, dict] = {}
    split_paths: dict[str, Path] = {}
    for key in SPLIT_ORDER:
        path = output_dir / f"{key}.jsonl"
        lines = (pair_by_split[key] + gate_records_by_split[key]
                 + proceed_records_by_split[key] + aug_splits[key]
                 + better_splits[key] + state_splits[key] + mask_splits[key])
        with path.open("w", encoding="utf-8") as handle:
            for record in lines:
                handle.write(_dump_line(record) + "\n")
        split_paths[key] = path
        split_sizes[key] = len(lines)
        split_counts[key] = {
            "listing_positive": sum(
                1 for r in pair_by_split[key]
                if r["expected"].get("identity_claim") == "true"),
            "listing_negative": sum(
                1 for r in pair_by_split[key]
                if r["expected"].get("identity_claim") == "false"),
            "gate_negative": len(gate_records_by_split[key]),
            "gate_proceed": len(proceed_records_by_split[key]),
            "aug_pair_positive": sum(
                1 for r in aug_splits[key]
                if r["expected"].get("identity_claim") == "true"),
            "aug_pair_negative": sum(
                1 for r in aug_splits[key]
                if r["expected"].get("identity_claim") == "false"),
            "better_match": len(better_splits[key]),
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

    # ── traceability census: per-question label + per-tag populations ────
    all_records = [
        record
        for key in SPLIT_ORDER
        for record in (pair_by_split[key] + gate_records_by_split[key]
                       + proceed_records_by_split[key] + aug_splits[key]
                       + better_splits[key] + state_splits[key]
                       + mask_splits[key])
    ]
    question_label_census: dict[str, Counter] = defaultdict(Counter)
    difficulty_slice_census: Counter = Counter()
    gate_reason_census: Counter = Counter()
    attribute_census: Counter = Counter()
    for record in all_records:
        for qid, label in record["expected"].items():
            question_label_census[qid][label] += 1
        difficulty_slice_census[record["difficulty_slice"]] += 1
        gate_reason_census[record["gate_reason"] or "none"] += 1
        attribute_census[record["attribute"]] += 1

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
            "gate_proceed_available": len(proceed),
            "gate_proceed_in_corpus": sum(
                len(proceed_records_by_split[key]) for key in SPLIT_ORDER),
            "gate_fallback_quarantined": len(fallback),
            "gate_hard_no_cap": hard_no_cap,
            "better_match_cases": sum(
                len(better_splits[key]) for key in SPLIT_ORDER),
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
            "aug_pairs_counterfactual":
                growth_census.get("aug_pair_counterfactual", 0),
            "labeled_pairs_added": growth_census["labeled_pairs_added"],
            # corpus-wide identity prior AFTER folding the augmentation in:
            # the counterfactual/twin negatives are hard and numerous, so the
            # operator can see the balance (and tune the source) at a glance.
            "identity_positive_total_with_growth": (
                positives + positives_growth),
            "identity_negative_total_with_growth": identity_negatives_total,
        },
        "growth": growth_census,
        "gate_reason_sample": dict(sorted(gate_reason_sample.items())),
        "gate_reason_families": dict(sorted(gate_reason_families.items())),
        # traceability: every emitted question's label distribution, plus the
        # per-row tag populations the eval report slices on. Deterministic
        # (counters over the frozen SPLIT_ORDER), so a rerun reproduces it.
        "question_label_census": {
            qid: dict(sorted(labels.items()))
            for qid, labels in sorted(question_label_census.items())
        },
        "difficulty_slice_census": dict(sorted(difficulty_slice_census.items())),
        "attribute_census": dict(sorted(attribute_census.items())),
        "tag_gate_reason_census": dict(sorted(gate_reason_census.items())),
        "split_ratios": {key: ratios[key] for key in SPLIT_ORDER},
        "split_sizes": split_sizes,
        "split_counts": split_counts,
        "question_schema_size": question_size,
        "size": {
            **{f"{key}.jsonl": _file_bytes(split_paths[key])
               for key in SPLIT_ORDER},
            "unknown_pairs.csv": _file_bytes(unknown_path),
        },
    }
    # One stable size over the whole corpus body (the split files in
    # frozen SPLIT_ORDER), for the report and the determinism test.
    corpus_size = ByteCount()
    for key in SPLIT_ORDER:
        corpus_size.update(split_paths[key].read_bytes())
    receipt["corpus_size"] = corpus_size.total
    # Additive: the composition-census blocks land ONLY when a knob is set, so
    # a default config keeps the landed receipt bytes exactly.
    if rebalance_census.get("enabled"):
        receipt["identity_rebalance"] = rebalance_census
    if not corpus_config_is_default(corpus):
        receipt["corpus_config"] = corpus
    receipt_path = output_dir / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return receipt


def main() -> None:
    # The question schema is the config SSOT for this builder: its optional
    # `corpus` block owns the composition knobs AND the input/output paths
    # (hardcoded defaults otherwise), so a run never depends on code literals.
    if not Path(QUESTION_PATH).is_file():
        raise FileNotFoundError(f"required source missing: {QUESTION_PATH}")
    corpus_cfg = _corpus_config(
        json.loads(Path(QUESTION_PATH).read_text(encoding="utf-8")))
    sources = resolve_corpus_sources(corpus_cfg)
    for key in ("catalog", "pairs", "gate"):
        if not Path(sources[key]).is_file():
            raise FileNotFoundError(
                f"required source missing ({key}): {sources[key]}")
    # Fold the pipeline's minted masking/augmentation in when the artifacts
    # are present (read, never invented); absent sources are reported, not
    # fabricated. `build()` still runs the base corpus without them.
    bundle = sources["bundle"] if Path(sources["bundle"]).is_file() else None
    labeled = (sources["labeled_pairs"]
               if Path(sources["labeled_pairs"]).is_file() else None)
    if bundle is None:
        print(f"[laya-build-dataset] masking/augmentation source absent: "
              f"{sources['bundle']} (base corpus only)")
    if labeled is None:
        print(f"[laya-build-dataset] labeled pairs source absent: "
              f"{sources['labeled_pairs']} (base corpus only)")
    if not corpus_config_is_default(corpus_cfg):
        print("[laya-build-dataset] corpus_config=" + json.dumps(corpus_cfg))
    receipt = build(
        catalog_path=sources["catalog"], pairs_path=sources["pairs"],
        gate_path=sources["gate"], output_dir=sources["output_dir"],
        question_path=QUESTION_PATH, corpus_config=corpus_cfg,
        bundle_path=bundle, labeled_pairs_path=labeled)
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
    print("[laya-build-dataset] question_label_census="
          + json.dumps(receipt["question_label_census"]))
    print("[laya-build-dataset] difficulty_slice_census="
          + json.dumps(receipt["difficulty_slice_census"]))
    print("[laya-build-dataset] attribute_census="
          + json.dumps(receipt["attribute_census"]))
    print("[laya-build-dataset] tag_gate_reason_census="
          + json.dumps(receipt["tag_gate_reason_census"]))
    print("[laya-build-dataset] growth=" + json.dumps(receipt["growth"]))
    if "identity_rebalance" in receipt:
        print("[laya-build-dataset] identity_rebalance="
              + json.dumps(receipt["identity_rebalance"]))
    for name, size in receipt["size"].items():
        print(f"[laya-build-dataset] size {name} {size}")
    print(f"[laya-build-dataset] corpus_size={receipt['corpus_size']}")
    print("[laya-build-dataset] -> "
          + str(Path(sources["output_dir"]) / "receipt.json"))


if __name__ == "__main__":
    main()
