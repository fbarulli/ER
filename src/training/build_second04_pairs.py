"""Build the valid-GTIN cross-country hard-positive manifest.

The manifest is derived from the frozen deduplicated dataset: rows sharing a
valid GTIN are paired when both countries are present and different.  The
existing ``volume_verified_cross_country`` consumer applies the final volume
agreement gate before returning training pairs.

Run::

    python -m training.build_second04_pairs
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.common import F, ensure_parent, load_dataset_deduped
from core.gtin import is_valid_gtin_checksum
from core.manifest import atomic_write_csv
from core.run_log import RunLogger
from core.schemas import (
    CROSS_COUNTRY_PAIR_COLUMNS,
    CrossCountryPairRow,
    check_cross_country_pair_frame,
)
from core.step_trace import timed

log = RunLogger(__name__)

MANIFEST_COLUMNS = CROSS_COUNTRY_PAIR_COLUMNS

REQUIRED_COLUMNS = ("sku_id", "gtin", "country")


class ExclusionCensus(BaseModel):
    """Closed accounting for every source row entering the pair builder."""

    model_config = ConfigDict(extra="forbid", strict=True)

    input_rows: int = Field(ge=0)
    accepted_rows: int = Field(ge=0)
    excluded_rows: int = Field(ge=0)
    missing_sku_id: int = Field(ge=0)
    missing_gtin: int = Field(ge=0)
    missing_country: int = Field(ge=0)
    invalid_gtin: int = Field(ge=0)

    @model_validator(mode="after")
    def _accounting_closes(self) -> "ExclusionCensus":
        reason_total = (
            self.missing_sku_id
            + self.missing_gtin
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


def _require_columns(frame: pd.DataFrame) -> None:
    """Fail loud when the deduplicated dataset lacks a required column."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"deduplicated dataset missing columns: {missing}")


def _normalized_copy(frame: pd.DataFrame) -> pd.DataFrame:
    """A copy with the three key columns stripped of NA/whitespace noise."""
    usable = frame.copy()
    for column in REQUIRED_COLUMNS:
        usable[column] = usable[column].fillna("").astype(str).str.strip()
    return usable


def _exclusion_reasons(usable: pd.DataFrame) -> pd.Series:
    """One deterministic exclusion reason per source row ("" = accepted... no).

    Applies the same filters as the former boolean mask but retains one
    reason per row. The order is intentional: a row missing multiple fields
    is counted once at its first failing gate, so
    input_rows == accepted_rows + excluded_rows always closes.
    """
    reason = pd.Series("accepted", index=usable.index, dtype="string")
    gates = (
        ("missing_sku_id", usable["sku_id"].eq("")),
        ("missing_gtin", usable["gtin"].eq("")),
        ("missing_country", usable["country"].eq("")),
        ("invalid_gtin", usable["gtin"].map(is_valid_gtin_checksum).eq(False)),
    )
    for label, mask in gates:
        pending = reason.eq("accepted")
        reason.loc[pending & mask] = label
    return reason


def _census_from_reasons(reason: pd.Series, input_rows: int) -> ExclusionCensus:
    """Fold the per-row reason column into the closed census model."""
    return ExclusionCensus(
        input_rows=input_rows,
        accepted_rows=int(reason.eq("accepted").sum()),
        excluded_rows=int(reason.ne("accepted").sum()),
        missing_sku_id=int(reason.eq("missing_sku_id").sum()),
        missing_gtin=int(reason.eq("missing_gtin").sum()),
        missing_country=int(reason.eq("missing_country").sum()),
        invalid_gtin=int(reason.eq("invalid_gtin").sum()),
    )


def _usable_rows_with_census(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, ExclusionCensus]:
    """The accepted rows in canonical order, plus their closed exclusion census."""
    _require_columns(frame)
    usable = _normalized_copy(frame)
    reason = _exclusion_reasons(usable)
    census = _census_from_reasons(reason, int(len(usable)))
    return usable.loc[reason.eq("accepted")].sort_values(
        ["gtin", "sku_id", "country"],
        kind="mergesort",
    ), census


def _country_positions(records: list[dict]) -> dict[str, list[int]]:
    """Record positions bucketed by country (order-preserving)."""
    positions: dict[str, list[int]] = {}
    for position, record in enumerate(records):
        positions.setdefault(str(record["country"]), []).append(position)
    return positions


def _cross_country_pairs(records: list[dict]) -> list[tuple[dict, dict]]:
    """(left, right) record pairs with different countries.

    The pair sequence is EXACTLY the one combinations() + country-skip
    produces — (i, j) with i < j ascending — so the manifest's row order is
    unchanged; the per-country bucket only spares same-country comparisons.
    """
    positions = _country_positions(records)
    pairs: list[tuple[dict, dict]] = []
    for i, left in enumerate(records):
        differing = sorted(
            j
            for country, js in positions.items()
            if country != str(left["country"])
            for j in js
            if j > i
        )
        pairs.extend((left, records[j]) for j in differing)
    return pairs


def _pair_rows(usable: pd.DataFrame) -> list[dict[str, object]]:
    """One validated row dict per cross-country pair, grouped per gtin."""
    rows: list[dict[str, object]] = []
    groups = list(usable.groupby("gtin", sort=True))
    for gtin, group in log.progress(
        groups, desc="cross_country_pairs", unit="gtin"
    ):
        for left, right in _cross_country_pairs(
            group[["sku_id", "country"]].to_dict("records")
        ):
            rows.append(
                CrossCountryPairRow(
                    sku_id_a=str(left["sku_id"]),
                    sku_id_b=str(right["sku_id"]),
                    cross_country=True,
                    gtin=str(gtin),
                    country_a=str(left["country"]),
                    country_b=str(right["country"]),
                ).model_dump()
            )
    return rows


@timed
def _build_manifest_with_census(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, ExclusionCensus]:
    """The pair manifest plus its closed exclusion census."""
    usable, census = _usable_rows_with_census(frame)
    manifest = pd.DataFrame(_pair_rows(usable), columns=CROSS_COUNTRY_PAIR_COLUMNS)
    return check_cross_country_pair_frame(manifest), census


def build_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one deterministic row per valid-GTIN cross-country pair."""
    manifest, census = _build_manifest_with_census(frame)
    manifest.attrs["exclusion_census"] = census.model_dump()
    log.info(f"[second04] exclusion census: {census.model_dump_json()}")
    return manifest


def write_manifest(output_path: Path) -> pd.DataFrame:
    """Build and atomically write the configured manifest."""
    manifest = build_manifest(load_dataset_deduped())
    atomic_write_csv(manifest, ensure_parent(Path(output_path)), index=False)
    return manifest


@timed
def main() -> None:
    output_path = Path(F["second04_pairs_positive"])
    manifest = write_manifest(output_path)
    log.info(
        f"[second04] wrote {output_path}: {len(manifest):,} cross-country pairs "
        f"across {manifest['gtin'].nunique():,} valid GTINs"
    )


if __name__ == "__main__":
    main()
