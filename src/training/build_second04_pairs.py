"""Build the valid-GTIN cross-country hard-positive manifest.

The manifest is derived from the frozen deduplicated dataset: rows sharing a
valid GTIN are paired when both countries are present and different.  The
existing ``volume_verified_cross_country`` consumer applies the final volume
agreement gate before returning training pairs.

Run::

    python -m training.build_second04_pairs
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd

from core.common import F, load_dataset_deduped
from core.gtin import is_valid_gtin_checksum
from core.schemas import (
    CROSS_COUNTRY_PAIR_COLUMNS,
    CrossCountryPairRow,
    check_cross_country_pair_frame,
)

MANIFEST_COLUMNS = CROSS_COUNTRY_PAIR_COLUMNS


def _usable_rows(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"product_id", "barcode", "country"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"deduplicated dataset missing columns: {missing}")
    usable = frame.copy()
    usable["product_id"] = usable["product_id"].fillna("").astype(str).str.strip()
    usable["barcode"] = usable["barcode"].fillna("").astype(str).str.strip()
    usable["country"] = usable["country"].fillna("").astype(str).str.strip()
    usable = usable.loc[
        usable["product_id"].ne("")
        & usable["barcode"].ne("")
        & usable["country"].ne("")
        & usable["barcode"].map(is_valid_gtin_checksum)
    ]
    return usable.sort_values(
        ["barcode", "product_id", "country"],
        kind="mergesort",
    )


def build_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one deterministic row per valid-GTIN cross-country pair."""
    rows: list[dict[str, object]] = []
    for gtin, group in _usable_rows(frame).groupby("barcode", sort=True):
        records = group[["product_id", "country"]].to_dict("records")
        for left, right in combinations(records, 2):
            country_a = str(left["country"])
            country_b = str(right["country"])
            if country_a == country_b:
                continue
            rows.append(
                CrossCountryPairRow(
                    sku_id_a=str(left["product_id"]),
                    sku_id_b=str(right["product_id"]),
                    cross_country=True,
                    gtin=str(gtin),
                    country_a=country_a,
                    country_b=country_b,
                ).model_dump()
            )
    manifest = pd.DataFrame(rows, columns=CROSS_COUNTRY_PAIR_COLUMNS)
    return check_cross_country_pair_frame(manifest)


def write_manifest(output_path: Path) -> pd.DataFrame:
    """Build and atomically write the configured manifest."""
    manifest = build_manifest(load_dataset_deduped())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f"{output_path.name}.tmp")
    manifest.to_csv(temporary, index=False)
    temporary.replace(output_path)
    return manifest


def main() -> None:
    output_path = Path(F["second04_pairs_positive"])
    manifest = write_manifest(output_path)
    print(
        f"[second04] wrote {output_path}: {len(manifest):,} cross-country pairs "
        f"across {manifest['gtin'].nunique():,} valid GTINs",
        flush=True,
    )


if __name__ == "__main__":
    main()
