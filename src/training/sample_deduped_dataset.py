"""Create a deterministic stratified sample of the deduplicated SKU dataset.

This is intentionally separate from ``sample_balanced_pairs``: the output is
source-SKU data suitable as a trainer input, not labeled candidate pairs.
The validation population is reduced by exactly one half.  It preserves the
joint retailer/country/brand/category/attribute-signature populations with a
stable SHA-256 rank.  Odd groups receive either floor(n/2) or ceil(n/2)
members through a deterministic marginal-balancing pass, and the manifest
records every before/after count for audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from core.common import SEED, TRAIN_ROOT, F

DEFAULT_INPUT = Path(F["dataset_deduped"])
DEFAULT_OUTPUT = TRAIN_ROOT / "artifacts/data/dataset_deduped_sample_1500.csv"
DEFAULT_REMAINDER_OUTPUT = (
    TRAIN_ROOT / "artifacts/data/dataset_deduped_train_minus_1500.csv"
)
DEFAULT_MANIFEST = TRAIN_ROOT / "results/manifests/dataset_deduped_sample_1500.json"


def _digest(seed: int, *values: object) -> str:
    value = "\x1f".join([str(seed), *(str(item) for item in values)])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv",
        prefix=f".{path.name}.",
        dir=path.parent,
        encoding="utf-8",
        newline="",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(temporary, path)


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix=f".{path.name}.",
        dir=path.parent,
        encoding="utf-8",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _attribute_signature(value: object) -> str:
    names = set()
    for item in str(value).split(";"):
        name = item.split(":", 1)[0].strip().lower()
        if name:
            names.add(" ".join(name.split()))
    return "|".join(sorted(names)) or "unknown"


def _stratified_population(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"product_id", "retailer", "country", "brand", "category", "attributes"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"validation population missing required columns: {missing}")
    work = frame.copy()
    if work["product_id"].duplicated().any():
        raise ValueError("validation population contains duplicate product_id values")
    for column in ("retailer", "country", "brand", "category", "attributes"):
        work[column] = work[column].fillna("").astype(str).str.strip()
    work["__attribute_signature"] = work["attributes"].map(_attribute_signature)
    work["__stratum"] = (
        work["retailer"].replace("", "unknown")
        + " || "
        + work["country"].replace("", "unknown")
        + " || "
        + work["brand"].replace("", "unknown")
        + " || "
        + work["category"].replace("", "unknown")
        + " || "
        + work["__attribute_signature"]
    )
    return work


def _half_allocation(work: pd.DataFrame, size: int, seed: int) -> dict[str, int]:
    if size * 2 != len(work):
        raise ValueError(
            "validation reduction must retain exactly half its population: "
            f"population={len(work)}, requested={size}"
        )
    capacities = work.groupby("__stratum", sort=True).size().astype(int).to_dict()
    allocation = {stratum: count // 2 for stratum, count in capacities.items()}
    # An even total implies an even number of odd groups. Choose exactly half
    # of them for ceil(n/2), balancing brand/category and the other owned
    # sampling dimensions as closely as integer arithmetic permits.
    odd = [stratum for stratum, count in capacities.items() if count % 2]
    extras = size - sum(allocation.values())
    if extras != len(odd) // 2:
        raise AssertionError("half-sample odd-stratum accounting did not close")
    dimensions = ("retailer", "country", "brand", "category", "__attribute_signature")
    marginal_total = {
        dimension: work.groupby(dimension, sort=True).size().astype(int).to_dict()
        for dimension in dimensions
    }
    stratum_values = {
        stratum: {dimension: str(part[dimension].iloc[0]) for dimension in dimensions}
        for stratum, part in work.groupby("__stratum", sort=True)
    }
    marginal_selected = {
        dimension: {
            str(value): int(
                sum(
                    allocation[stratum]
                    for stratum in capacities
                    if stratum_values[stratum][dimension] == str(value)
                )
            )
            for value in totals
        }
        for dimension, totals in marginal_total.items()
    }
    # The expression above is intentionally correct but expensive only for
    # the small validation population.  Cache each odd group's dimension
    # values so tie-breaking does not depend on frame order.
    odd_values = {stratum: stratum_values[stratum] for stratum in odd}
    selected_odd: set[str] = set()
    for _ in range(extras):
        choices = [stratum for stratum in odd if stratum not in selected_odd]

        def cost(stratum: str) -> tuple[float, str]:
            delta = 0.0
            for dimension, value in odd_values[stratum].items():
                target = marginal_total[dimension][value] / 2.0
                current = marginal_selected[dimension][value]
                delta += (current + 1 - target) ** 2 - (current - target) ** 2
            return delta, _digest(seed, "odd-stratum", stratum)

        chosen = min(choices, key=cost)
        selected_odd.add(chosen)
        allocation[chosen] += 1
        for dimension, value in odd_values[chosen].items():
            marginal_selected[dimension][value] += 1
    if sum(allocation.values()) != size:
        raise AssertionError("half-sample allocation did not close")
    return allocation


def _sample_half(
    frame: pd.DataFrame, size: int, seed: int
) -> tuple[pd.DataFrame, dict[str, int], list[dict[str, object]]]:
    work = _stratified_population(frame)
    allocation = _half_allocation(work, size, seed)
    pieces = []
    for stratum in sorted(allocation):
        count = allocation[stratum]
        if not count:
            continue
        subset = work[work["__stratum"] == stratum].copy()
        subset["__rank"] = [
            _digest(
                seed,
                row.product_id,
                row.retailer,
                row.country,
                row.category,
                row.attributes,
            )
            for row in subset.itertuples(index=False)
        ]
        pieces.append(
            subset.sort_values(["__rank", "product_id"], kind="mergesort").head(count)
        )
    result = pd.concat(pieces, ignore_index=True)
    result = (
        result.sort_values("product_id", kind="mergesort")
        .drop(columns=["__attribute_signature", "__stratum", "__rank"])
        .reset_index(drop=True)
    )
    if len(result) != size:
        raise AssertionError(f"sample size did not close: {len(result)} != {size}")
    if result["product_id"].duplicated().any():
        raise AssertionError("sample contains duplicate product_id values")
    audit = [
        {
            "stratum": stratum,
            "population_rows": int(count),
            "retained_rows": int(allocation[stratum]),
            "excluded_rows": int(count - allocation[stratum]),
            "retention_ratio": float(allocation[stratum] / count),
        }
        for stratum, count in sorted(
            work.groupby("__stratum", sort=True).size().astype(int).to_dict().items()
        )
    ]
    return result, {str(key): int(value) for key, value in allocation.items()}, audit


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--population",
        type=Path,
        required=True,
        help="existing validation population to reduce by exactly half",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--remainder-output", type=Path, default=DEFAULT_REMAINDER_OUTPUT
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--size", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    source = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    population = pd.read_csv(args.population, dtype=str, keep_default_na=False)
    source_ids = set(source["product_id"])
    population_ids = set(population["product_id"])
    if len(source_ids) != len(source):
        raise ValueError("deduplicated source contains duplicate product_id values")
    if not population_ids <= source_ids:
        raise ValueError(
            "validation population contains product IDs absent from deduplicated source: "
            f"count={len(population_ids - source_ids)}"
        )
    sampled, allocation, stratum_audit = _sample_half(population, args.size, args.seed)
    remainder = source[~source["product_id"].isin(set(sampled["product_id"]))].copy()
    remainder = remainder.sort_values("product_id", kind="mergesort").reset_index(
        drop=True
    )
    if len(remainder) + len(sampled) != len(source):
        raise AssertionError("sample/remainder row accounting did not close")
    _atomic_csv(sampled, args.output)
    _atomic_csv(remainder, args.remainder_output)
    manifest = {
        "schema_version": 2,
        "seed": int(args.seed),
        "sample_size": int(args.size),
        "time_period": "snapshot_unknown: source has no temporal column",
        "stratification": "retailer + country + brand + category + attribute-name signature",
        "retention_contract": "exactly half of the supplied validation population; odd strata use deterministic floor/ceil allocation",
        "inputs": {
            "path": str(args.input),
            "rows": len(source),
            "sha256": _sha256(args.input),
        },
        "output": {
            "path": str(args.output),
            "rows": len(sampled),
            "sha256": _sha256(args.output),
        },
        "validation_population": {
            "rows": len(population),
            "sha256": _sha256(args.population),
            "definition": "pre-reduction validation population; retained IDs are a deterministic half",
        },
        "training_remainder": {
            "path": str(args.remainder_output),
            "rows": len(remainder),
            "sha256": _sha256(args.remainder_output),
            "definition": "input rows whose product_id is not in the held-out sample",
        },
        "strata_retained_rows": allocation,
        "stratum_retention_audit": stratum_audit,
        "attribute_slice_population_and_retained_rows": {
            column: [
                {
                    "slice": str(value),
                    "population_rows": len(
                        population.loc[
                            population[column]
                            .fillna("")
                            .astype(str)
                            .str.strip()
                            .eq(str(value))
                        ]
                    ),
                    "retained_rows": len(
                        sampled.loc[
                            sampled[column]
                            .fillna("")
                            .astype(str)
                            .str.strip()
                            .eq(str(value))
                        ]
                    ),
                }
                for value in sorted(
                    population[column].fillna("").astype(str).str.strip().unique()
                )
            ]
            for column in ("retailer", "country", "brand", "category")
        },
    }
    _atomic_json(manifest, args.manifest)
    print(
        f"source rows: {len(source):,}\n"
        f"sample rows: {len(sampled):,} -> {args.output}\n"
        f"training rows: {len(remainder):,} -> {args.remainder_output}\n"
        f"manifest: {args.manifest}\n"
        f"seed: {args.seed}",
        flush=True,
    )


if __name__ == "__main__":
    main()
