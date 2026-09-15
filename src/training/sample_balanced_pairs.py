"""Build a deterministic, type-balanced candidate-pair pool and sample.

The label universe is intentionally identical to ``training/labeled_pairs.py``:

* positive: ``gate_decision == "proceed"`` and similarity is at least
  ``pairs.proceed_sim_threshold``;
* negative: ``gate_decision == "hard_no"`` and similarity is at least
  ``pairs.hardneg_sim_threshold``;
* fallback and below-threshold candidates are excluded and accounted for.

``gate_reason`` is label-specific: positives have one "all compatible" reason,
whereas hard negatives name the incompatible attribute.  A directly shared
gate-reason stratum therefore does not exist.  ``pair_type`` is defined as the
hard-negative contrast family (volume, pack, flavor, package type, or package
material).  Hard negatives inherit that family from ``gate_reason``.  Eligible
compatible positives are deterministically partitioned among the same families
in proportion to the available negatives.  No pair is copied.  The resulting
pool has exactly as many positives as negatives inside every ``pair_type``.

The final sample preserves the balanced pool's joint distribution over pair
type, endpoint SKU-count buckets/retailers/countries, endpoint categories and
attribute-name signatures, and time period using deterministic proportional
allocation.  If the source SKU data has no temporal column, every row receives
the explicit ``snapshot_unknown`` time stratum; dates are never invented.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from core.common import load_config


DEFAULT_GATE_INPUT = Path("results/gate_results.csv")
DEFAULT_SKU_INPUT = Path("artifacts/data/dataset_deduped.csv")
DEFAULT_BALANCED_OUTPUT = Path("results/training/balanced_pairs.csv")
DEFAULT_SAMPLE_OUTPUT = Path("results/training/balanced_pairs_sample_3000.csv")
DEFAULT_MANIFEST_OUTPUT = Path("results/training/balanced_pairs_sample_manifest.json")
DEFAULT_SWEEP_OUTPUT = Path("results/training/balanced_pairs_sample_threshold_sweep.csv")
SWEEP_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)

REASON_PREFIX_TO_TYPE = (
    ("No volume overlap", "volume"),
    ("No pack overlap", "pack"),
    ("Flavor mismatch:", "flavor"),
    ("Critical attribute mismatch:", "critical_attribute"),
    ("Package type mismatch", "package_type"),
    ("Package material mismatch", "package_material"),
)
TIME_COLUMN_CANDIDATES = (
    "timestamp",
    "event_time",
    "observed_at",
    "collected_at",
    "scraped_at",
    "created_at",
    "updated_at",
    "date",
)


def _stable_digest(seed: int, *values: object) -> str:
    payload = "\x1f".join([str(seed), *(str(value) for value in values)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", prefix=f".{path.name}.", dir=path.parent,
        encoding="utf-8", newline="", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(temporary, path)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix=f".{path.name}.", dir=path.parent,
        encoding="utf-8", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _normalise(value: object) -> str:
    text = " ".join(str(value).strip().lower().split())
    return text or "unknown"


def _stable_mode(values: Iterable[object]) -> str:
    cleaned = [_normalise(value) for value in values]
    counts = pd.Series(cleaned, dtype="string").value_counts()
    if counts.empty:
        return "unknown"
    maximum = int(counts.max())
    return sorted(str(index) for index, count in counts.items() if count == maximum)[0]


def _attribute_signature(values: Iterable[object]) -> str:
    names: set[str] = set()
    for value in values:
        for item in str(value).split(";"):
            name = item.split(":", 1)[0]
            normalised = _normalise(name)
            if normalised != "unknown":
                names.add(normalised)
    return "|".join(sorted(names)) if names else "unknown"


def _sku_count_bucket(count: int) -> str:
    if count <= 1:
        return "1"
    lower = 2 ** int(math.floor(math.log2(count)))
    upper = 2 * lower - 1
    return f"{lower}-{upper}"


def _detect_schema(frame: pd.DataFrame) -> dict[str, str | None]:
    choices = {
        "gtin": ("barcode", "gtin"),
        "sku": ("product_id", "sku_id"),
        "attribute": ("attributes", "attribute"),
        "retailer": ("retailer",),
        "country": ("country",),
        "category": ("category",),
    }
    schema: dict[str, str | None] = {}
    for key, candidates in choices.items():
        schema[key] = next((name for name in candidates if name in frame.columns), None)
    missing = [key for key in ("gtin", "sku") if schema[key] is None]
    if missing:
        raise ValueError(f"SKU input is missing required semantic columns: {missing}")
    schema["time"] = next(
        (name for name in TIME_COLUMN_CANDIDATES if name in frame.columns), None
    )
    return schema


def _time_periods(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None:
        return pd.Series("snapshot_unknown", index=frame.index, dtype="string")
    parsed = pd.to_datetime(frame[column], errors="coerce", utc=True)
    periods = parsed.dt.strftime("%Y-%m").fillna("time_unknown")
    if not bool(parsed.notna().any()):
        raise ValueError(f"Temporal column {column!r} exists but contains no parseable dates")
    return periods.astype("string")


def _build_gtin_metadata(
    sku_frame: pd.DataFrame, schema: Mapping[str, str | None]
) -> dict[str, dict[str, str]]:
    gtin_col = str(schema["gtin"])
    sku_col = str(schema["sku"])
    work = sku_frame.copy()
    work[gtin_col] = work[gtin_col].astype(str).str.strip()
    work = work[~work[gtin_col].isin(["", "NA", "N/A", "nan", "None"])]
    work["__time_period"] = _time_periods(work, schema["time"])
    metadata: dict[str, dict[str, str]] = {}
    for gtin, group in work.groupby(gtin_col, sort=True):
        def values(name: str) -> Iterable[object]:
            column = schema[name]
            return group[column] if column is not None else ["unknown"]

        sku_count = int(group[sku_col].astype(str).nunique())
        metadata[str(gtin)] = {
            "sku_count_bucket": _sku_count_bucket(sku_count),
            "retailer": _stable_mode(values("retailer")),
            "country": _stable_mode(values("country")),
            "category": _stable_mode(values("category")),
            "attribute_signature": _attribute_signature(values("attribute")),
            "time_period": _stable_mode(group["__time_period"]),
        }
    return metadata


def _pair_value(left: str, right: str) -> str:
    return " <> ".join(sorted((left, right)))


def _enrich_distribution_strata(
    pairs: pd.DataFrame, metadata: Mapping[str, Mapping[str, str]]
) -> pd.DataFrame:
    result = pairs.copy()
    missing = {
        "sku_count_bucket": "missing_gtin",
        "retailer": "unknown",
        "country": "unknown",
        "category": "unknown",
        "attribute_signature": "unknown",
        "time_period": "snapshot_unknown",
    }
    for field, default in missing.items():
        result[f"{field}_pair"] = [
            _pair_value(
                metadata.get(str(left), {}).get(field, default),
                metadata.get(str(right), {}).get(field, default),
            )
            for left, right in zip(result["gtin1"], result["gtin2"], strict=True)
        ]
    result["sku_distribution_stratum"] = (
        result["sku_count_bucket_pair"] + " || "
        + result["retailer_pair"] + " || " + result["country_pair"]
    )
    result["attribute_distribution_stratum"] = (
        result["category_pair"] + " || " + result["attribute_signature_pair"]
    )
    result["time_period"] = result["time_period_pair"]
    result["distribution_stratum"] = (
        result["sku_distribution_stratum"] + " || "
        + result["attribute_distribution_stratum"] + " || "
        + result["time_period"]
    )
    return result.drop(columns=["time_period_pair"])


def _reason_type(reason: object) -> str | None:
    text = str(reason).strip()
    return next((family for prefix, family in REASON_PREFIX_TO_TYPE if text.startswith(prefix)), None)


def _largest_remainder(
    capacities: Mapping[str, int], target: int, seed: int, namespace: str
) -> dict[str, int]:
    capacities = {str(key): int(value) for key, value in capacities.items() if value > 0}
    total = sum(capacities.values())
    if target < 0 or target > total:
        raise ValueError(f"Cannot allocate target={target} from capacity={total}")
    if target == 0:
        return {key: 0 for key in capacities}
    exact = {key: target * value / total for key, value in capacities.items()}
    allocated = {key: min(capacities[key], int(math.floor(value))) for key, value in exact.items()}
    remaining = target - sum(allocated.values())
    order = sorted(
        capacities,
        key=lambda key: (
            -(exact[key] - math.floor(exact[key])),
            _stable_digest(seed, namespace, key),
            key,
        ),
    )
    while remaining:
        progressed = False
        for key in order:
            if allocated[key] < capacities[key]:
                allocated[key] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise RuntimeError("Proportional allocation exhausted capacity unexpectedly")
    return allocated


def _largest_remainder_with_coverage(
    capacities: Mapping[str, int], target: int, seed: int, namespace: str
) -> dict[str, int]:
    """Allocate proportionally while retaining every represented type."""
    positive = {str(key): int(value) for key, value in capacities.items() if value > 0}
    if target < len(positive):
        raise ValueError(
            f"Target {target} cannot retain all {len(positive)} represented types"
        )
    residual = {key: value - 1 for key, value in positive.items()}
    extra = _largest_remainder(residual, target - len(positive), seed, namespace)
    return {key: 1 + extra.get(key, 0) for key in positive}


def _stable_take(frame: pd.DataFrame, count: int, seed: int, namespace: str) -> pd.DataFrame:
    if count > len(frame):
        raise ValueError(f"Requested {count} rows from a population of {len(frame)}")
    ranked = frame.copy()
    ranked["__rank"] = [
        _stable_digest(seed, namespace, min(a, b), max(a, b))
        for a, b in zip(ranked["gtin1"], ranked["gtin2"], strict=True)
    ]
    return ranked.sort_values(["__rank", "gtin1", "gtin2"], kind="mergesort").head(count).drop(columns="__rank")


def _stratified_take(
    frame: pd.DataFrame, count: int, seed: int, namespace: str
) -> pd.DataFrame:
    capacities = frame.groupby("distribution_stratum", sort=True).size().to_dict()
    allocation = _largest_remainder(capacities, count, seed, namespace)
    pieces = []
    for stratum in sorted(allocation):
        amount = allocation[stratum]
        if amount:
            subset = frame[frame["distribution_stratum"] == stratum]
            pieces.append(_stable_take(subset, amount, seed, f"{namespace}|{stratum}"))
    if not pieces and count:
        raise RuntimeError("Stratified selection produced no rows")
    return pd.concat(pieces, ignore_index=True) if pieces else frame.head(0).copy()


def _assert_pair_contract(frame: pd.DataFrame, expected_rows: int | None = None) -> None:
    if expected_rows is not None and len(frame) != expected_rows:
        raise AssertionError(f"Expected {expected_rows:,} rows, found {len(frame):,}")
    bad_labels = set(frame["true_label"].unique()) - {0, 1}
    if bad_labels:
        raise AssertionError(f"Unexpected labels: {sorted(bad_labels)}")
    canonical = pd.DataFrame({
        "left": frame[["gtin1", "gtin2"]].min(axis=1),
        "right": frame[["gtin1", "gtin2"]].max(axis=1),
    })
    duplicates = int(canonical.duplicated().sum())
    if duplicates:
        raise AssertionError(f"Found {duplicates:,} duplicate unordered GTIN pairs")
    counts = frame.groupby(["pair_type", "true_label"]).size().unstack(fill_value=0)
    if 0 not in counts or 1 not in counts:
        raise AssertionError("Both labels must occur in every balanced output")
    unequal = counts[counts[0] != counts[1]]
    if not unequal.empty:
        raise AssertionError(f"Per-type label imbalance:\n{unequal.to_string()}")


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def threshold_sweep(
    scores: pd.Series,
    labels: pd.Series,
    thresholds: Iterable[float] = SWEEP_THRESHOLDS,
) -> pd.DataFrame:
    """Return deterministic confusion counts and rates for fixed thresholds."""
    numeric_scores = pd.to_numeric(scores, errors="raise").astype(float)
    numeric_labels = pd.to_numeric(labels, errors="raise").astype(int)
    if not set(numeric_labels.unique()).issubset({0, 1}):
        raise ValueError("Threshold sweep labels must be binary {0, 1}")
    rows: list[dict[str, int | float]] = []
    for threshold in thresholds:
        predicted = numeric_scores >= float(threshold)
        positive = numeric_labels == 1
        tp = int((predicted & positive).sum())
        fp = int((predicted & ~positive).sum())
        fn = int((~predicted & positive).sum())
        tn = int((~predicted & ~positive).sum())
        rows.append({
            "threshold": float(threshold),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": _safe_ratio(tp, tp + fp),
            "recall": _safe_ratio(tp, tp + fn),
            "fpr": _safe_ratio(fp, fp + tn),
        })
    return pd.DataFrame(rows)


def build_outputs(
    gate_path: Path,
    sku_path: Path,
    balanced_path: Path,
    sample_path: Path,
    manifest_path: Path,
    sweep_path: Path,
    sample_size: int,
    seed: int,
    allow_unmatched_types: bool,
) -> dict[str, Any]:
    if sample_size <= 0 or sample_size % 2:
        raise ValueError("--size must be a positive even integer for exact class balance")
    config = load_config()
    positive_threshold = float(config["pairs"]["proceed_sim_threshold"])
    negative_threshold = float(config["pairs"]["hardneg_sim_threshold"])

    gate = pd.read_csv(gate_path, dtype={"gtin1": str, "gtin2": str}, keep_default_na=False)
    required = {"gtin1", "gtin2", "gate_decision", "gate_reason", "similarity"}
    missing = sorted(required - set(gate.columns))
    if missing:
        raise ValueError(f"Gate input is missing required columns: {missing}")
    gate["similarity"] = pd.to_numeric(gate["similarity"], errors="raise")
    positive = gate[(gate["gate_decision"] == "proceed") & (gate["similarity"] >= positive_threshold)].copy()
    negative = gate[(gate["gate_decision"] == "hard_no") & (gate["similarity"] >= negative_threshold)].copy()
    positive["true_label"] = 1
    negative["true_label"] = 0
    negative["pair_type"] = negative["gate_reason"].map(_reason_type)
    unmatched_counts = negative.loc[negative["pair_type"].isna(), "gate_reason"].value_counts().sort_index().to_dict()
    if unmatched_counts and not allow_unmatched_types:
        raise ValueError(
            "Unmatched hard-negative gate_reason values; update REASON_PREFIX_TO_TYPE "
            f"or pass --allow-unmatched-types to exclude and record them: {unmatched_counts}"
        )
    negative = negative[negative["pair_type"].notna()].copy()
    if positive.empty or negative.empty:
        raise ValueError("No eligible positive or typed hard-negative candidates")

    negative_capacities = negative["pair_type"].value_counts().sort_index().to_dict()
    per_class = min(len(positive), len(negative))
    type_quota = _largest_remainder(negative_capacities, per_class, seed, "balanced-type-quota")
    positive_selected = _stable_take(positive, per_class, seed, "balanced-positive")
    positive_pieces = []
    cursor = 0
    for pair_type in sorted(type_quota):
        amount = type_quota[pair_type]
        piece = positive_selected.iloc[cursor:cursor + amount].copy()
        piece["pair_type"] = pair_type
        piece["pair_type_source"] = "compatible_positive_partition"
        positive_pieces.append(piece)
        cursor += amount
    if cursor != per_class:
        raise AssertionError("Positive type partition did not close")
    negative_pieces = []
    for pair_type in sorted(type_quota):
        subset = negative[negative["pair_type"] == pair_type]
        piece = _stable_take(subset, type_quota[pair_type], seed, f"balanced-negative|{pair_type}")
        piece["pair_type_source"] = "hard_no_gate_reason_family"
        negative_pieces.append(piece)
    balanced = pd.concat([*positive_pieces, *negative_pieces], ignore_index=True)

    sku = pd.read_csv(sku_path, dtype=str, keep_default_na=False)
    sku_schema = _detect_schema(sku)
    metadata = _build_gtin_metadata(sku, sku_schema)
    balanced = _enrich_distribution_strata(balanced, metadata)
    balanced["candidate_generation_logic"] = (
        f"labeled_pairs:v1;proceed>={positive_threshold:g};hard_no>={negative_threshold:g};fallback=excluded"
    )
    _assert_pair_contract(balanced)
    if sample_size > len(balanced):
        raise ValueError(
            f"Requested {sample_size:,} rows, but balanced pool has only {len(balanced):,}"
        )

    sample_per_class = sample_size // 2
    balanced_type_counts = balanced[balanced["true_label"] == 1]["pair_type"].value_counts().sort_index().to_dict()
    sample_type_quota = _largest_remainder_with_coverage(
        balanced_type_counts, sample_per_class, seed, "sample-type-quota"
    )
    sample_pieces = []
    for pair_type in sorted(sample_type_quota):
        amount = sample_type_quota[pair_type]
        for label in (0, 1):
            cell = balanced[(balanced["pair_type"] == pair_type) & (balanced["true_label"] == label)]
            sample_pieces.append(
                _stratified_take(cell, amount, seed, f"sample|{pair_type}|{label}")
            )
    sample = pd.concat(sample_pieces, ignore_index=True)
    order = ["pair_type", "true_label", "gtin1", "gtin2"]
    balanced = balanced.sort_values(order, kind="mergesort").reset_index(drop=True)
    sample = sample.sort_values(order, kind="mergesort").reset_index(drop=True)
    _assert_pair_contract(balanced)
    _assert_pair_contract(sample, expected_rows=sample_size)

    _atomic_csv(balanced, balanced_path)
    _atomic_csv(sample, sample_path)
    sweep = threshold_sweep(sample["similarity"], sample["true_label"])
    _atomic_csv(sweep, sweep_path)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "seed": seed,
        "requested_sample_size": sample_size,
        "pair_type_definition": {
            "name": "hard_negative_gate_reason_family",
            "families": {prefix: family for prefix, family in REASON_PREFIX_TO_TYPE},
            "positive_assignment": "stable SHA-256 partition proportional to eligible negative families; no replacement",
            "negative_assignment": "gate_reason prefix",
        },
        "candidate_generation": {
            "positive": f"gate_decision=proceed and similarity>={positive_threshold:g}",
            "negative": f"gate_decision=hard_no and similarity>={negative_threshold:g}",
            "excluded": "fallback and below-class-threshold rows",
            "threshold_source": "config/training.yaml pairs.proceed_sim_threshold and pairs.hardneg_sim_threshold",
        },
        "distribution_contract": {
            "sku": "joint endpoint SKU-count bucket, retailer, and country",
            "attribute": "joint endpoint category and parsed attribute-name signature",
            "time": (
                f"calendar month parsed from {sku_schema['time']}"
                if sku_schema["time"] is not None else "snapshot_unknown: source has no temporal column"
            ),
            "selection": "largest-remainder proportional allocation, then seeded SHA-256 rank",
        },
        "inputs": {
            "gate_results": {"path": str(gate_path), "sha256": _file_sha256(gate_path), "rows": len(gate)},
            "sku_data": {"path": str(sku_path), "sha256": _file_sha256(sku_path), "rows": len(sku), "schema": sku_schema},
        },
        "accounting": {
            "eligible_positive": len(positive),
            "eligible_typed_negative": len(negative),
            "excluded_fallback": int((gate["gate_decision"] == "fallback").sum()),
            "excluded_below_positive_threshold": int(
                ((gate["gate_decision"] == "proceed") & (gate["similarity"] < positive_threshold)).sum()
            ),
            "excluded_below_negative_threshold": int(
                ((gate["gate_decision"] == "hard_no") & (gate["similarity"] < negative_threshold)).sum()
            ),
            "excluded_other_gate_decision": int(
                (~gate["gate_decision"].isin(["proceed", "hard_no", "fallback"])).sum()
            ),
            "unmatched_negative_reasons": unmatched_counts,
            "balanced_rows": len(balanced),
            "sample_rows": len(sample),
            "balanced_by_type_and_label": {
                str(pair_type): {str(label): int(count) for label, count in values.items()}
                for pair_type, values in balanced.groupby(["pair_type", "true_label"]).size().unstack(fill_value=0).to_dict("index").items()
            },
            "sample_by_type_and_label": {
                str(pair_type): {str(label): int(count) for label, count in values.items()}
                for pair_type, values in sample.groupby(["pair_type", "true_label"]).size().unstack(fill_value=0).to_dict("index").items()
            },
            "threshold_sweep": sweep.to_dict(orient="records"),
        },
        "outputs": {
            "balanced": {"path": str(balanced_path), "sha256": _file_sha256(balanced_path)},
            "sample": {"path": str(sample_path), "sha256": _file_sha256(sample_path)},
            "threshold_sweep": {"path": str(sweep_path), "sha256": _file_sha256(sweep_path)},
        },
    }
    _atomic_json(manifest, manifest_path)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-input", type=Path, default=DEFAULT_GATE_INPUT)
    parser.add_argument("--sku-input", type=Path, default=DEFAULT_SKU_INPUT)
    parser.add_argument("--balanced-output", type=Path, default=DEFAULT_BALANCED_OUTPUT)
    parser.add_argument("--sample-output", type=Path, default=DEFAULT_SAMPLE_OUTPUT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST_OUTPUT)
    parser.add_argument("--threshold-sweep-output", type=Path, default=DEFAULT_SWEEP_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=3000)
    parser.add_argument(
        "--allow-unmatched-types", action="store_true",
        help="Exclude unknown hard-negative reason families and record them in the manifest.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = build_outputs(
        gate_path=args.gate_input,
        sku_path=args.sku_input,
        balanced_path=args.balanced_output,
        sample_path=args.sample_output,
        manifest_path=args.manifest_output,
        sweep_path=args.threshold_sweep_output,
        sample_size=args.size,
        seed=args.seed,
        allow_unmatched_types=args.allow_unmatched_types,
    )
    accounting = manifest["accounting"]
    print(
        f"balanced pool: {accounting['balanced_rows']:,} rows -> {args.balanced_output}\n"
        f"deterministic sample: {accounting['sample_rows']:,} rows -> {args.sample_output}\n"
        f"manifest: {args.manifest_output}\n"
        f"threshold sweep: {args.threshold_sweep_output}\n"
        f"sample balance: {json.dumps(accounting['sample_by_type_and_label'], sort_keys=True)}"
    )


if __name__ == "__main__":
    main()
