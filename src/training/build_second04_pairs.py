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
from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.common import F, load_dataset_deduped
from core.gtin import is_valid_gtin_checksum
from core.schemas import (
    CROSS_COUNTRY_PAIR_COLUMNS,
    CrossCountryPairRow,
    check_cross_country_pair_frame,
)

MANIFEST_COLUMNS = CROSS_COUNTRY_PAIR_COLUMNS


class ExclusionCensus(BaseModel):
    """Closed accounting for every source row entering the pair builder."""

    model_config = ConfigDict(extra="forbid", strict=True)

    input_rows: int = Field(ge=0)
    accepted_rows: int = Field(ge=0)
    excluded_rows: int = Field(ge=0)
    missing_product_id: int = Field(ge=0)
    missing_barcode: int = Field(ge=0)
    missing_country: int = Field(ge=0)
    invalid_gtin: int = Field(ge=0)

    @model_validator(mode="after")
    def _accounting_closes(self) -> "ExclusionCensus":
        reason_total = (
            self.missing_product_id
            + self.missing_barcode
            + self.missing_country
            + self.invalid_gtin
        )
        if self.excluded_rows != reason_total:
            raise ValueError(
                "second04 exclusion census does not close: "
                f"excluded_rows={self.excluded_rows} != reasons={reason_total}"
            )
        if self.input_rows != self.accepted_rows + self.excluded_rows:
            raise ValueError(
                "second04 source census does not close: "
                f"input_rows={self.input_rows} != accepted+excluded="
                f"{self.accepted_rows + self.excluded_rows}"
            )
        return self


def _usable_rows_with_census(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, ExclusionCensus]:
    required = {"product_id", "barcode", "country"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"deduplicated dataset missing columns: {missing}")
    usable = frame.copy()
    usable["product_id"] = usable["product_id"].fillna("").astype(str).str.strip()
    usable["barcode"] = usable["barcode"].fillna("").astype(str).str.strip()
    usable["country"] = usable["country"].fillna("").astype(str).str.strip()

    # Apply the same filters as the former boolean mask, but retain one
    # deterministic reason per source row.  The order is intentional: a row
    # missing multiple fields is counted once at its first failing gate, so
    # input_rows == accepted_rows + excluded_rows always closes.
    reason = pd.Series("accepted", index=usable.index, dtype="string")
    pending = reason.eq("accepted")
    missing_product_id = usable["product_id"].eq("")
    reason.loc[pending & missing_product_id] = "missing_product_id"
    pending = reason.eq("accepted")
    missing_barcode = usable["barcode"].eq("")
    reason.loc[pending & missing_barcode] = "missing_barcode"
    pending = reason.eq("accepted")
    missing_country = usable["country"].eq("")
    reason.loc[pending & missing_country] = "missing_country"
    pending = reason.eq("accepted")
    invalid_gtin = usable["barcode"].map(is_valid_gtin_checksum).eq(False)
    reason.loc[pending & invalid_gtin] = "invalid_gtin"

    census = ExclusionCensus(
        input_rows=int(len(usable)),
        accepted_rows=int(reason.eq("accepted").sum()),
        excluded_rows=int(reason.ne("accepted").sum()),
        missing_product_id=int(reason.eq("missing_product_id").sum()),
        missing_barcode=int(reason.eq("missing_barcode").sum()),
        missing_country=int(reason.eq("missing_country").sum()),
        invalid_gtin=int(reason.eq("invalid_gtin").sum()),
    )
    return usable.loc[reason.eq("accepted")].sort_values(
        ["barcode", "product_id", "country"],
        kind="mergesort",
    ), census


def _usable_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return valid rows while retaining the census in the build path."""
    usable, _ = _usable_rows_with_census(frame)
    return usable


def _build_manifest_with_census(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, ExclusionCensus]:
    usable, census = _usable_rows_with_census(frame)
    rows: list[dict[str, object]] = []
    for gtin, group in usable.groupby("barcode", sort=True):
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
    return check_cross_country_pair_frame(manifest), census


def build_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one deterministic row per valid-GTIN cross-country pair."""
    manifest, census = _build_manifest_with_census(frame)
    manifest.attrs["exclusion_census"] = census.model_dump()
    print(
        f"[second04] exclusion census: {census.model_dump_json()}",
        flush=True,
    )
    return manifest


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
