"""Build the valid-GTIN cross-country hard-positive manifest.

The manifest is derived from the frozen deduplicated dataset: rows sharing a
valid GTIN are paired when both countries are present and different.  The
existing ``volume_verified_cross_country`` consumer applies the final volume
agreement gate before returning training pairs.

TRACE ROWS (core.tracing, the ONE consolidated trace)
-----------------------------------------------------
Stage ``build_second04_pairs``. Emitted:
  run   source.rows_classified        every source row -> the accepted rows,
                                      with the closed exclusion census in detail
  group source.exclusion.reason_census  the EXACT per-reason census (accepted /
                                      missing_sku_id / missing_gtin /
                                      missing_country / invalid_gtin)
  ent   source.exclusion.*            the named ROWS behind those reasons
                                      (sku_id + that row's exact reason)
  run   pairs.batch_<i>              BATCH grain: one row per traced chunk of
                                      GTIN groups (in = groups, out = pairs)
  run   pairs.batch_census            how many batches there were, how many were
                                      traced, and how many were omitted
  run   pairs.cross_country_built     accepted rows -> cross-country pair rows
  run   pairs.manifest_validated      the frame contract check (in == out)
  run   pairs.manifest_written        the atomic publication
Batch caps: ``_BATCH_GTINS`` GTIN groups per traced batch row and at most
``_MAX_BATCH_ROWS`` batch rows; both are written into the batch rows' detail.
Never unbounded.

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
from core.tracing import (ENTITY_ROW_CAP, ENTITY_SAMPLE_PER_REASON,
                          TRACE_BATCH_ROWS, TRACE_MAX_BATCH_ROWS, TraceRun)

_LOG = RunLogger(__name__)

#: The pipeline stage these rows belong to (core.tracing ``stage`` column).
STAGE = "build_second04_pairs"

# ── batch-grain budget (documented where it is spent) ──────────────────────
# The pairing pass walks one GTIN group at a time; a group is the natural chunk
# (its rows can form pairs only with each other). 4,096 groups per BATCH row
# makes a real cohort a handful of rows while staying small enough to read, and
# 16 traced batches keep the file bounded on a 25k-GTIN corpus; the remainder is
# announced in ``pairs.batch_census`` instead of vanishing.
_BATCH_GTINS = TRACE_BATCH_ROWS
_MAX_BATCH_ROWS = TRACE_MAX_BATCH_ROWS

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


# ── the accepted source population ──────────────────────────────────────────
class SourcePopulation:
    """The deduplicated dataset's buildable rows plus their closed census.

    One deterministic exclusion reason per source row ("" is never a reason —
    rows are accepted or named). The order is intentional: a row missing
    multiple fields is counted once at its first failing gate, so
    input_rows == accepted_rows + excluded_rows always closes.
    """

    @staticmethod
    def require_columns(frame: pd.DataFrame) -> None:
        """Fail loud when the deduplicated dataset lacks a required column."""
        missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"deduplicated dataset missing columns: {missing}")

    @classmethod
    def normalized_copy(cls, frame: pd.DataFrame) -> pd.DataFrame:
        """A copy with the three key columns stripped of NA/whitespace noise."""
        usable = frame.copy()
        for column in REQUIRED_COLUMNS:
            usable[column] = usable[column].fillna("").astype(str).str.strip()
        return usable

    @staticmethod
    def exclusion_reasons(usable: pd.DataFrame) -> pd.Series:
        """One deterministic exclusion reason per source row.

        Applies the same filters as the former boolean mask but retains one
        reason per row, each counted at its first failing gate.
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

    @staticmethod
    def census_from_reasons(reason: pd.Series, input_rows: int) -> ExclusionCensus:
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

    @classmethod
    def usable_rows_with_census_and_reasons(
        cls, frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, ExclusionCensus]:
        """The accepted rows, ALL normalized rows, every row's reason, and the census.

        The normalized frame is returned beside the accepted one because the
        trace's ENTITY rows must name the DROPPED rows too (a dropped row is a
        named ``sku_id`` with the exact gate that removed it, never a count).
        """
        cls.require_columns(frame)
        usable = cls.normalized_copy(frame)
        reason = cls.exclusion_reasons(usable)
        census = cls.census_from_reasons(reason, int(len(usable)))
        accepted = usable.loc[reason.eq("accepted")].sort_values(
            ["gtin", "sku_id", "country"],
            kind="mergesort",
        )
        return accepted, usable, reason, census

    @classmethod
    def usable_rows_with_census(
        cls, frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, ExclusionCensus]:
        """The accepted rows in canonical order, plus the closed census."""
        accepted, _usable, _reason, census = cls.usable_rows_with_census_and_reasons(
            frame
        )
        return accepted, census


def _usable_rows_with_census(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, ExclusionCensus]:
    """The accepted rows, all normalized rows, every row's reason, and the census."""
    return SourcePopulation.usable_rows_with_census_and_reasons(frame)


# ── the source-row trace (stage + entity grain) ─────────────────────────────
class SourcePopulationTrace:
    """Records every source row's fate: the census, and the named rows behind it."""

    @staticmethod
    def records(usable: pd.DataFrame, reason: pd.Series) -> list[dict[str, str]]:
        """One record per SOURCE row: its key, its fields, its exact reason.

        ``usable`` is the normalized copy the classifier ran on (all rows, accepted
        or not), so its index is the frame's own and ``reason`` aligns row for row.
        """
        return [
            {
                "sku_id": str(row["sku_id"]),
                "gtin": str(row["gtin"]),
                "country": str(row["country"]),
                "reason": str(reason.loc[index]),
            }
            for index, row in usable.iterrows()
        ]

    @classmethod
    def record(cls, trace: TraceRun, usable: pd.DataFrame, reason: pd.Series,
               census: ExclusionCensus) -> None:
        """The stage row plus the exact reason census and its entity sample."""
        trace.add(
            "source",
            "rows_classified",
            in_count=census.input_rows,
            out_count=census.accepted_rows,
            reason=(
                "a source row is buildable only when it carries sku_id, gtin AND "
                "country and its gtin passes the GS1 check digit; each rejected "
                "row is charged to its FIRST failing gate, once"
            ),
            detail=census.model_dump(),
            source="dataset_deduped (core.common.load_dataset_deduped)",
        )
        trace.add_entities(
            "source.exclusion",
            cls.records(usable, reason),
            key_of=lambda record: record["sku_id"],
            reason_of=lambda record: record["reason"],
            detail_of=lambda record: {
                "gtin": record["gtin"],
                "country": record["country"],
            },
            source="dataset_deduped (core.common.load_dataset_deduped)",
            per_reason=ENTITY_SAMPLE_PER_REASON,
            total_cap=ENTITY_ROW_CAP,
        )


# ── cross-country pairing ───────────────────────────────────────────────────
class CrossCountryPairs:
    """The (left, right) pairs of one gtin group, countries differing.

    The pair sequence is EXACTLY the one combinations() + country-skip
    produces — (i, j) with i < j ascending — so the manifest's row order is
    unchanged; the per-country bucket only spares same-country comparisons.
    """

    @staticmethod
    def country_positions(records: list[dict]) -> dict[str, list[int]]:
        """Record positions bucketed by country (order-preserving)."""
        positions: dict[str, list[int]] = {}
        for position, record in enumerate(records):
            positions.setdefault(str(record["country"]), []).append(position)
        return positions

    @classmethod
    def differing_country_pairs(cls, records: list[dict]) -> list[tuple[dict, dict]]:
        """(left, right) record pairs with different countries."""
        positions = cls.country_positions(records)
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

    @classmethod
    def pair_rows(
        cls, usable: pd.DataFrame, trace: TraceRun | None = None
    ) -> list[dict[str, object]]:
        """One validated row dict per cross-country pair, grouped per gtin.

        ``trace`` adds BATCH-grain rows: one row per ``_BATCH_GTINS`` GTIN groups
        (in = groups walked, out = pairs they produced), capped at
        ``_MAX_BATCH_ROWS`` rows. The grouping, order and pair sequence are
        untouched — the batch boundary is read off the existing loop, it does not
        re-chunk it.
        """
        rows: list[dict[str, object]] = []
        groups = list(usable.groupby("gtin", sort=True))
        groups_in_batch = 0
        pairs_in_batch = 0
        first_gtin = last_gtin = ""
        batches = traced = 0
        for gtin, group in _LOG.progress(
            groups, desc="cross_country_pairs", unit="gtin"
        ):
            if groups_in_batch == 0:
                first_gtin = str(gtin)
                pairs_in_batch = 0
            last_gtin = str(gtin)
            groups_in_batch += 1
            for left, right in cls.differing_country_pairs(
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
                pairs_in_batch += 1
            if groups_in_batch >= _BATCH_GTINS or groups_in_batch == len(groups):
                batches += 1
                if trace is not None and traced < _MAX_BATCH_ROWS:
                    traced += 1
                    # Deliberately NO in/out pair: a group's rows can form many
                    # pairs (fan-out), so "groups -> pairs" is a census, not a
                    # funnel, and forcing an in/out pair would be a lie.
                    trace.add(
                        "pairs",
                        f"batch_{batches - 1:04d}",
                        reason=(
                            "GTIN groups walked; the pairs they produced are a "
                            "fan-out over their rows (a census, not a funnel), "
                            "so no in/out pair is stated"
                        ),
                        detail={
                            "first_gtin": first_gtin,
                            "last_gtin": last_gtin,
                            "gtin_groups": groups_in_batch,
                            "pairs": pairs_in_batch,
                            "batch_gtins": _BATCH_GTINS,
                            "max_batch_rows": _MAX_BATCH_ROWS,
                        },
                        source="dataset_deduped gtin groups",
                    )
                groups_in_batch = 0
        if trace is not None:
            trace.add(
                "pairs",
                "batch_census",
                in_count=batches,
                out_count=traced,
                reason=(
                    "batches traced individually; the remainder is summed here so "
                    "no chunk is silent"
                ),
                detail={
                    "gtin_groups": len(groups),
                    "batches": batches,
                    "batches_traced": traced,
                    "batches_omitted": batches - traced,
                    "batch_gtins": _BATCH_GTINS,
                    "max_batch_rows": _MAX_BATCH_ROWS,
                    "pairs": len(rows),
                },
                source="dataset_deduped gtin groups",
            )
        return rows


def _pair_rows(
    usable: pd.DataFrame, trace: TraceRun | None = None
) -> list[dict[str, object]]:
    """One validated row dict per cross-country pair (see CrossCountryPairs)."""
    return CrossCountryPairs.pair_rows(usable, trace)


@timed
def _build_manifest_with_census(
    frame: pd.DataFrame, trace: TraceRun | None = None
) -> tuple[pd.DataFrame, ExclusionCensus]:
    """The pair manifest plus its closed exclusion census."""
    usable, all_rows, reason, census = _usable_rows_with_census(frame)
    if trace is not None:
        SourcePopulationTrace.record(trace, all_rows, reason, census)
    manifest = pd.DataFrame(_pair_rows(usable, trace), columns=CROSS_COUNTRY_PAIR_COLUMNS)
    checked = check_cross_country_pair_frame(manifest)
    if trace is not None:
        # Census, not a funnel: a group's rows fan out into pairs, so a
        # pairs-per-row in/out pair would be false for the smallest populations.
        trace.add(
            "pairs",
            "cross_country_built",
            reason=(
                "one row per (left, right) pair of a GTIN group whose countries "
                "differ — a fan-out over the accepted rows, so the population is "
                "recorded as a census"
            ),
            detail={
                "accepted_rows": census.accepted_rows,
                "gtins": int(checked["gtin"].nunique()) if len(checked) else 0,
                "pairs": int(len(checked)),
            },
            source="dataset_deduped gtin groups",
        )
        trace.add(
            "pairs",
            "manifest_validated",
            in_count=int(len(checked)),
            out_count=int(len(checked)),
            reason="the frame contract (endpoints, cross_country, distinct countries) holds",
            detail={"columns": list(CROSS_COUNTRY_PAIR_COLUMNS)},
            source="core.schemas.check_cross_country_pair_frame",
        )
    return checked, census


def build_manifest(
    frame: pd.DataFrame, trace: TraceRun | None = None
) -> pd.DataFrame:
    """Return one deterministic row per valid-GTIN cross-country pair."""
    own = trace is None
    if own:
        trace = TraceRun(STAGE)
    manifest, census = _build_manifest_with_census(frame, trace)
    manifest.attrs["exclusion_census"] = census.model_dump()
    _LOG.info(f"[second04] exclusion census: {census.model_dump_json()}")
    if own:
        trace.write()
    return manifest


def write_manifest(output_path: Path, trace: TraceRun | None = None) -> pd.DataFrame:
    """Build and atomically write the configured manifest."""
    own = trace is None
    if own:
        trace = TraceRun(STAGE)
    manifest = build_manifest(load_dataset_deduped(), trace)
    path = ensure_parent(Path(output_path))
    atomic_write_csv(manifest, path, index=False)
    trace.add(
        "pairs",
        "manifest_written",
        in_count=int(len(manifest)),
        out_count=int(len(manifest)),
        reason="cross-country pair manifest published atomically",
        detail={"path": str(path), "gtins": int(manifest["gtin"].nunique())},
        source=str(path),
    )
    if own:
        trace.write()
    return manifest


@timed
def main() -> None:
    output_path = Path(F["second04_pairs_positive"])
    # ONE writer for the stage: the census, the batches and the publication are
    # one flow in the trace, committed once.
    trace = TraceRun(STAGE)
    manifest = write_manifest(output_path, trace)
    trace.write()
    _LOG.info(
        f"[second04] wrote {output_path}: {len(manifest):,} cross-country pairs "
        f"across {manifest['gtin'].nunique():,} valid GTINs"
    )


if __name__ == "__main__":
    main()
