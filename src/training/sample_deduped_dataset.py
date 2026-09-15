"""Create a deterministic stratified sample of the deduplicated SKU dataset.

This is intentionally separate from ``sample_balanced_pairs``: the output is
source-SKU data suitable as a trainer input, not labeled candidate pairs.
Sampling preserves the joint retailer/country/category/attribute-signature
distribution as closely as possible with largest-remainder allocation and a
stable SHA-256 row rank. The frozen source has no temporal column, so the
manifest records the explicit ``snapshot_unknown`` period.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import pandas as pd

from core.common import F, SEED, TRAIN_ROOT


DEFAULT_INPUT = Path(F["dataset_deduped"])
DEFAULT_OUTPUT = Path(F["dataset_deduped_sample_3000"])
DEFAULT_REMAINDER_OUTPUT = Path(F["dataset_deduped_train_minus_3000"])
DEFAULT_MANIFEST = TRAIN_ROOT / "results/manifests/dataset_deduped_sample_3000.json"


def _digest(seed: int, *values: object) -> str:
    value = "\x1f".join([str(seed), *(str(item) for item in values)])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", prefix=f".{path.name}.", dir=path.parent,
        encoding="utf-8", newline="", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(temporary, path)


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix=f".{path.name}.", dir=path.parent,
        encoding="utf-8", delete=False,
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


def _largest_remainder(capacities: dict[str, int], target: int) -> dict[str, int]:
    total = sum(capacities.values())
    exact = {key: target * value / total for key, value in capacities.items()}
    allocated = {key: int(math.floor(value)) for key, value in exact.items()}
    remaining = target - sum(allocated.values())
    order = sorted(
        capacities,
        key=lambda key: (-(exact[key] - math.floor(exact[key])), key),
    )
    for key in order[:remaining]:
        allocated[key] += 1
    return allocated


def _sample(frame: pd.DataFrame, size: int, seed: int) -> tuple[pd.DataFrame, dict[str, int]]:
    required = {"product_id", "retailer", "country", "category", "attributes"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"deduplicated dataset missing required columns: {missing}")
    if size <= 0 or size > len(frame):
        raise ValueError(f"sample size must be in [1, {len(frame)}], got {size}")

    work = frame.copy()
    for column in ("retailer", "country", "category", "attributes"):
        work[column] = work[column].fillna("").astype(str).str.strip()
    work["__attribute_signature"] = work["attributes"].map(_attribute_signature)
    work["__stratum"] = (
        work["retailer"].replace("", "unknown") + " || "
        + work["country"].replace("", "unknown") + " || "
        + work["category"].replace("", "unknown") + " || "
        + work["__attribute_signature"]
    )
    capacities = work.groupby("__stratum", sort=True).size().astype(int).to_dict()
    allocation = _largest_remainder(capacities, size)
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
    result = result.sort_values("product_id", kind="mergesort").drop(
        columns=["__attribute_signature", "__stratum", "__rank"]
    ).reset_index(drop=True)
    if len(result) != size:
        raise AssertionError(f"sample size did not close: {len(result)} != {size}")
    if result["product_id"].duplicated().any():
        raise AssertionError("sample contains duplicate product_id values")
    return result, {str(key): int(value) for key, value in allocation.items() if value}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--remainder-output", type=Path, default=DEFAULT_REMAINDER_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--size", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    source = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    sampled, allocation = _sample(source, args.size, args.seed)
    remainder = source[~source["product_id"].isin(set(sampled["product_id"]))].copy()
    remainder = remainder.sort_values("product_id", kind="mergesort").reset_index(drop=True)
    if len(remainder) + len(sampled) != len(source):
        raise AssertionError("sample/remainder row accounting did not close")
    _atomic_csv(sampled, args.output)
    _atomic_csv(remainder, args.remainder_output)
    manifest = {
        "schema_version": 1,
        "seed": int(args.seed),
        "sample_size": int(args.size),
        "time_period": "snapshot_unknown: source has no temporal column",
        "stratification": "retailer + country + category + attribute-name signature",
        "inputs": {
            "path": str(args.input),
            "rows": int(len(source)),
            "sha256": _sha256(args.input),
        },
        "output": {
            "path": str(args.output),
            "rows": int(len(sampled)),
            "sha256": _sha256(args.output),
        },
        "training_remainder": {
            "path": str(args.remainder_output),
            "rows": int(len(remainder)),
            "sha256": _sha256(args.remainder_output),
            "definition": "input rows whose product_id is not in the held-out sample",
        },
        "strata": allocation,
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
