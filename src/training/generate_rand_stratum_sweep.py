"""Build a non-degenerate GTIN-status fixture for Rand threshold sweeps.

Each selected canonical identity contributes multiple source SKUs.  The fixture
therefore retains positive and negative pair signal in every GTIN stratum,
unlike a one-SKU-per-identity truth export.  ``source_gtin`` is an evaluation
override: callers must apply it to the source SKU barcode before scoring.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from core.common import TRAIN_ROOT, canonical_records_frame, load_dataset_deduped, rand_matching_cfg
from core.gtin import is_valid_gtin_checksum
from core.manifest import atomic_write_csv


STATUSES = ("both_equal", "different", "one_missing")


def _rank(seed: int, *parts: str) -> str:
    return hashlib.sha256("\x1f".join((str(seed), *parts)).encode()).hexdigest()


def _absent_valid_gtin(item_id: str, canonical_ids: set[str], seed: int) -> str:
    """Make a valid GTIN that cannot activate an exact-canonical lock."""
    if not is_valid_gtin_checksum(item_id):
        raise ValueError(f"cannot derive an override from invalid GTIN {item_id!r}")
    body = item_id[:-1]
    for salt in range(1, 100):
        position = int(_rank(seed, "different", item_id, str(salt))[:8], 16) % len(body)
        step = int(_rank(seed, "step", item_id, str(salt))[:8], 16) % 9 + 1
        digits = list(body)
        digits[position] = str((int(digits[position]) + step) % 10)
        mutated_body = "".join(digits)
        weighted = sum(int(digit) * (3 if index % 2 == 0 else 1) for index, digit in enumerate(mutated_body[::-1]))
        candidate = mutated_body + str((10 - weighted % 10) % 10)
        if candidate not in canonical_ids and is_valid_gtin_checksum(candidate):
            return candidate
    raise RuntimeError(f"could not derive an absent valid GTIN for {item_id!r}")


def generate_stratum_sweep(
    output_dir: Path,
    output_name: str,
    identities_per_status: int,
    skus_per_identity: int,
    calibration_folds: int,
    seed: int,
) -> Path:
    source = load_dataset_deduped().copy()
    required = {"product_id", "barcode"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"deduplicated dataset missing columns: {sorted(missing)}")
    source["SKU_ID"] = source["product_id"].astype(str).str.strip()
    source["true_item_id"] = source["barcode"].astype(str).str.strip()
    if source["SKU_ID"].eq("").any() or source["SKU_ID"].duplicated().any():
        raise ValueError("deduplicated dataset has blank or duplicate product IDs")
    canonical_ids = set(canonical_records_frame()["gtin"].astype(str))
    eligible = source[
        source["true_item_id"].isin(canonical_ids)
        & source["true_item_id"].map(is_valid_gtin_checksum)
    ].copy()
    grouped = eligible.groupby("true_item_id", sort=False)
    groups = [
        (str(item_id), group.sort_values("SKU_ID", kind="stable").head(skus_per_identity))
        for item_id, group in grouped
        if len(group) >= skus_per_identity
    ]
    required_groups = identities_per_status * len(STATUSES)
    if len(groups) < required_groups:
        raise ValueError(
            f"need {required_groups:,} canonical groups with {skus_per_identity} SKUs; "
            f"found {len(groups):,}"
        )
    groups = sorted(groups, key=lambda item: _rank(seed, "identity", item[0]))[
        :required_groups
    ]
    selected = {
        status: groups[index * identities_per_status : (index + 1) * identities_per_status]
        for index, status in enumerate(STATUSES)
    }
    rows: list[dict[str, str]] = []
    for status, status_groups in selected.items():
        for position, (item_id, group) in enumerate(status_groups):
            if status == "both_equal":
                source_gtin = item_id
            elif status == "one_missing":
                source_gtin = ""
            else:
                source_gtin = _absent_valid_gtin(item_id, canonical_ids, seed)
            for sku_id in group["SKU_ID"].astype(str):
                rows.append(
                    {
                        "SKU_ID": sku_id,
                        "true_item_id": item_id,
                        "source_gtin": source_gtin,
                        "gtin_status": status,
                        "calibration_fold": str(position % calibration_folds),
                    }
                )
    fixture = pd.DataFrame(rows).sort_values(["gtin_status", "SKU_ID"], kind="stable")
    if fixture["SKU_ID"].duplicated().any():
        raise RuntimeError("stratum fixture reused a source SKU")
    if (fixture.groupby(["gtin_status", "true_item_id"])["SKU_ID"].nunique() != skus_per_identity).any():
        raise RuntimeError("stratum fixture lost a required positive pair")
    for status in STATUSES:
        actual = fixture.loc[fixture["gtin_status"].eq(status), "calibration_fold"].nunique()
        if actual != calibration_folds:
            raise RuntimeError(f"{status} fixture has {actual} folds, expected {calibration_folds}")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / output_name
    atomic_write_csv(fixture, path, index=False)
    print(
        "[rand-stratum-sweep] "
        f"rows={len(fixture):,} identities={fixture.true_item_id.nunique():,} "
        f"counts={fixture.gtin_status.value_counts().sort_index().to_dict()}",
        flush=True,
    )
    print(f"[rand-stratum-sweep] output={path}", flush=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--identities-per-status", type=int, default=None)
    args = parser.parse_args()
    cfg = rand_matching_cfg()["stratum_sweep"]
    output_dir = Path(args.output_dir or cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = TRAIN_ROOT / output_dir
    generate_stratum_sweep(
        output_dir.resolve(),
        str(cfg["output"]),
        int(args.identities_per_status or cfg["identities_per_status"]),
        int(cfg["skus_per_identity"]),
        int(cfg["calibration_folds"]),
        int(cfg["seed"]),
    )


if __name__ == "__main__":
    main()
