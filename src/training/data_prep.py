"""data_prep.py — run the OFFICIAL data-prep pipeline (pipeline.run_within_brand_pipeline).

Loads the raw export (raw column names, dtype=str), runs the within-brand
pipeline (extraction → canonical → gating → similarity), writes
canonical_records.csv + gate_results.csv into the config results dir, and
records every step in the ONE consolidated trace (core.tracing).

The two-stage flow this file is half of:
    stage 1  data_prep.main()                 RAW export columns
             (gtin / sku_name_eng / attribute)  -> canonical_records.csv,
                                                   gate_results.csv, trace
    stage 2  training.data_prep.build (train)  CANONICAL columns
             (gtin / title / attributes)     -> training pairs, trace
Stage 2 does NOT consume stage 1's dataframe — it reloads the deduped dataset
(core.common.load_dataset_deduped) and only MEETS stage 1 at the two artifacts
above. That handoff is pinned in the trace by each stage's column-contract row.

TRACE ROWS (core.tracing, the ONE consolidated trace)
-----------------------------------------------------
This module OWNS the ``data_prep`` stage writer and drives stage 1 with it
(``run_within_brand_pipeline(df, trace)``), so the pipeline's step rows and the
manifest's accounting land in ONE commit of ONE run — the consolidated trace's
one-writer-per-stage contract. Rows this module adds after the pipeline returns:
  run   guard.recomputed_census   raw rows -> GS1-valid rows, with the
                                  missing/checksum/review drops named exactly
  run   manifest.row_accounting   the closure input == kept-visible + dropped +
                                  collapsed_same_gtin, with the manifest's own
                                  accounting in detail
  group flags.reason_census       one EXACT census row per attribute-consistency
                                  flag (how many GTINs carry it)
  ent   flags.*                   the sampled GTINs carrying each flag
  run   batch_grain               this stage reads ONE file in one pass, so
                                  there is no chunk fan-out to trace (stated,
                                  not silently omitted)
Sampling caps are core.tracing's (ENTITY_SAMPLE_PER_REASON / ENTITY_ROW_CAP).
"""

from __future__ import annotations

import pandas as pd

from core.common import DATA_PATH, SEED, F, CONFIG_PATH, VOCABULARY_CONFIG_PATH, load_raw_export
from core.manifest import begin_manifest, finish_manifest
from core.run_log import RunLogger
from core.step_trace import timed
from core.tracing import ENTITY_ROW_CAP, ENTITY_SAMPLE_PER_REASON, TraceRun, trace_path
from pipeline import run_within_brand_pipeline

log = RunLogger(__name__)


def _load_raw() -> pd.DataFrame:
    """The raw export exactly as the pipeline expects it (raw column names)."""
    return load_raw_export()


def _regex_fallback_census(pipeline_module) -> int:
    """Digit tokens the numbers lane resolved by regex fallback, not reference."""
    return int(pipeline_module._UNSEEN_TOKEN_TOTAL)


def _report_regex_fallback_census() -> None:
    """AUDIT 2026-09-09: print the one degradation the numbers lane allows."""
    import pipeline as _dp

    log.info(
        f"[numbers] {_regex_fallback_census(_dp):,} digit-token resolutions "
        f"via regex fallback (not in reference CSV)"
    )


def _identifiable_gtin_mask(df: pd.DataFrame) -> pd.Series:
    """Rows whose gtin cell names a barcode at all (vs missing/NaN/'nan')."""
    gtin = df["gtin"].astype(str)
    return (
        df["gtin"].notna()
        & (gtin.str.strip() != "")
        & (gtin.str.lower() != "nan")
    )


def _gtin_guard_census(df: pd.DataFrame) -> dict[str, int]:
    """Recompute the gtin-guard populations exactly as the pipeline drops them.

    The counters below are recomputed from the frame the same way the guard
    computes them, so the manifest can never drift from the code's truth.
    """
    from core.gtin import gtin_validity
    from core.identity_policy import reviewed_row_mask

    reviewed = reviewed_row_mask(df)
    barcode = df["gtin"].fillna("").astype(str).str.strip()
    bc_valid = gtin_validity(barcode)
    bc_valid.index = df.index
    bc_valid &= ~reviewed
    return {
        "n_missing_gtin": int((~_identifiable_gtin_mask(df)).sum()),
        "n_checksum": int((_identifiable_gtin_mask(df) & ~bc_valid & ~reviewed).sum()),
        "n_valid": int((_identifiable_gtin_mask(df) & bc_valid).sum()),
        "n_reviewed": int(reviewed.sum()),
    }


def _row_closures(census: dict[str, int], canon: pd.DataFrame, pairs, df) -> dict:
    """Assemble the manifest's capture-only row accounting.

    input_rows: every raw-export row the pipeline read. dropped: the
    gtin-guard populations (missing/NaN gtin, failed GS1 checksum) —
    run_within_brand_pipeline prints both. Rows with a valid gtin collapse
    into one canonical record per gtin (not a "drop" — they're aggregated);
    that population is recorded under collapsed_same_gtin so the closure
    reads input == output_rows + dropped + collapsed.
    """
    n_canon = len(canon)
    collapsed = census["n_valid"] - n_canon
    return {
        "input_rows": len(df),
        "output_rows": n_canon,
        "dropped": {
            "gtin_missing_or_nan": census["n_missing_gtin"],
            "gtin_checksum_failed": census["n_checksum"],
            "identity_review_quarantined": census["n_reviewed"],
        },
        # kept-and-aggregated, NOT dropped (closure-extension key)
        "collapsed_same_gtin": collapsed,
        # pair-level population (not row accounting, but pinned here so
        # the gate census can't silently thin)
        "gate_pairs": len(pairs),
    }


def _dp_manifest_accounting(df, pairs, canon) -> dict:
    """Capture-only row accounting for the data_prep manifest."""
    from core.identity_policy import apply_identity_links

    linked = apply_identity_links(df)
    census = _gtin_guard_census(linked)
    return _row_closures(census, canon, pairs, linked)


def _flag_census(canon) -> dict[str, int]:
    """Count attribute-consistency flags per canonical record (gtin level).

    ``canon`` is the in-memory canonical-records frame returned by
    ``run_within_brand_pipeline``; its ``attribute_consistency_flags`` column
    holds a sorted list of flags per gtin (aggregated from extract_all in
    generate_canonical). Counting these makes the previously in-memory-only
    flags verifiable from committed data, since the census rides the data_prep
    manifest. Each gtin carrying a flag counts once; a gtin can carry several.
    """
    from collections import Counter

    counts: Counter[str] = Counter()
    if "attribute_consistency_flags" not in canon:
        return {}
    flags_column = canon["attribute_consistency_flags"]
    for flags in log.progress(flags_column, desc="flag_census", unit="gtin"):
        if not isinstance(flags, (list, tuple, set)):
            continue
        for flag in flags:
            counts[str(flag)] += 1
    return dict(sorted(counts.items()))


def _stage_outputs() -> tuple[list, list[str]]:
    """The stage's two frozen artifacts plus the consolidated trace.

    The trace used to be listed here by its OLD per-stage name
    (results/logs/gate_visibility.csv), whose writer this directive deleted —
    finish_manifest hashes every listed output, so a stale name made the
    stage die with FileNotFoundError after all the work was done. The name
    now comes from the layout that owns it (core.tracing → training_trace).
    """
    outputs = [F["canonical_records"], F["gate_results"], trace_path()]
    return outputs, [path.name for path in outputs]


@timed
def main() -> None:
    from core.timing import Timing

    timing = Timing("data_prep")
    # Stage manifest (SILENT_DROPS task 6) — begin BEFORE the work: the
    # raw export is hashed now (53MB, chunked) so the record pins exactly
    # what this stage read. Seed = the SSOT seed; the pipeline is
    # deterministic, no RNG is consumed.
    manifest = begin_manifest("data_prep", inputs=[DATA_PATH, CONFIG_PATH, VOCABULARY_CONFIG_PATH, F["number_reference"]], seed=SEED)
    timing.mark("manifest_begin")
    with log.section("data_prep.pipeline"):
        df = _load_raw()
        timing.mark("load_raw_export")
        log.info(f"[data_prep] loaded {len(df):,} raw rows")
        # ONE writer for the stage: the pipeline adds its step rows to THIS run and
        # leaves the commit to us, so the manifest's accounting joins them.
        trace = TraceRun("data_prep")
        pairs, canon = run_within_brand_pipeline(df, trace)
        timing.mark("run_within_brand_pipeline")
    log.info(f"[data_prep] pairs: {len(pairs):,} | canonical records: {len(canon):,}")
    _report_regex_fallback_census()
    _close_manifest(manifest, timing, df, pairs, canon, trace)


def _close_manifest(manifest, timing, df: pd.DataFrame, pairs, canon, trace) -> None:
    """Assemble the stage's ledger into the manifest and print the closure.

    Row accounting (gtin-guard drops vs kept-visible canonical records), the
    attribute-consistency flag census, the frozen artifact list, the atomic
    manifest publication and the closing closure line — the one
    responsibility the stage owes AFTER the pipeline ran. The consolidated-trace
    rows are added to the stage's single writer and committed BEFORE the manifest
    hashes it, so the shipped manifest pins the trace it describes.
    """
    # ---- row accounting (SILENT_DROPS task 6; capture-only) ────────────────
    # The pipeline's gtin-guard drops rows for two loud reasons (both
    # printed by run_within_brand_pipeline); every other input row either
    # lands in a canonical record or is a same-gtin duplicate collapsed
    # into one record — all three populations sum back to input_rows.
    # The gate is pair-level, not row-level: every candidate pair gets a
    # decision (no pair is dropped), so pairs carry no dropped bucket.
    row_accounting = _dp_manifest_accounting(df, pairs, canon)
    # Attribute-consistency flags (volume_inconsistency, ambiguous_volume,
    # ...) are computed in extract_all and aggregated per gtin into the
    # in-memory canonical frame, but were never persisted — the counts
    # drifted between snapshots (volume_inconsistency 232 -> 197,
    # ambiguous_volume 49 -> 69 on the 2026-09-28 rebuild) with nothing
    # pinning them. Persist the census here so the counts are verifiable
    # from committed data (results/manifests/data_prep.json).
    row_accounting["flags_census"] = _flag_census(canon)
    _record_stage_rows(trace, df, canon, row_accounting)
    trace.write()
    out_paths, expected = _stage_outputs()
    timing.mark("accounting_and_flags")
    mpath = finish_manifest(
        manifest, out_paths, row_accounting, expected_outputs=expected
    )
    timing.mark("finish_manifest")
    n_in = row_accounting["input_rows"]
    n_out = row_accounting["output_rows"]
    n_drop = sum(row_accounting["dropped"].values())
    log.info(
        f"[manifest] data_prep complete -> {mpath} | "
        f"closure {n_in:,} == {n_out:,} kept-visible + {n_drop:,} dropped"
    )


def _record_stage_rows(
    trace: TraceRun, df: pd.DataFrame, canon, row_accounting: dict
) -> None:
    """The manifest's row accounting, the flag census and the guard readback.

    ``guard.recomputed_census`` re-derives the guard's populations from the frame
    the same way the guard does (``_gtin_guard_census``), so the trace carries the
    stage's own closure arithmetic and cannot drift from the manifest.
    """
    from core.identity_policy import apply_identity_links

    linked = apply_identity_links(df)
    census = _gtin_guard_census(linked)
    # The invariant, stated in the trace itself: ONE writer commits stage
    # 'data_prep'. This module drives stage 1 with its own writer
    # (run_within_brand_pipeline(df, trace)), which defers its commit, so a second
    # writer for this stage would silently REPLACE these rows. A reader seeing two
    # of these rows knows the deferral was bypassed.
    trace.add(
        "stage_ownership",
        "single_writer",
        reason=(
            "training.data_prep is the ONE writer of stage 'data_prep': it hands "
            "its writer to run_within_brand_pipeline, which defers its commit, so "
            "the pipeline's step rows and this manifest accounting commit together"
        ),
        detail={
            "writer": "training.data_prep",
            "stage1_handoff": "run_within_brand_pipeline(df, trace)",
        },
        source="src/pipeline.py _PipelineSteering.open_stage",
    )
    trace.add(
        "guard",
        "recomputed_census",
        in_count=int(len(df)),
        out_count=int(census["n_valid"]),
        reason=(
            "rows keep identity only with a present, GS1-valid gtin (and no "
            "identity-review hold), recomputed from the frame the guard used"
        ),
        detail={**census, "input_rows": int(len(df))},
        source="raw export (core.common.load_raw_export)",
    )
    dropped = row_accounting["dropped"]
    trace.add(
        "manifest",
        "row_accounting",
        in_count=int(row_accounting["input_rows"]),
        out_count=int(row_accounting["output_rows"]),
        reason=(
            "closure: input_rows == output_rows + sum(dropped) + "
            "collapsed_same_gtin; the last term is rows KEPT and AGGREGATED into "
            "their gtin's record, not lost"
        ),
        detail={
            "output_rows": int(row_accounting["output_rows"]),
            "dropped": {key: int(value) for key, value in dropped.items()},
            "collapsed_same_gtin": int(row_accounting["collapsed_same_gtin"]),
            "gate_pairs": int(row_accounting["gate_pairs"]),
        },
        source="data_prep stage manifest row accounting",
    )
    trace.add(
        "batch_grain",
        "not_applicable",
        reason=(
            "this stage reads ONE raw-export frame in one pass and writes two "
            "artifacts; there is no chunk or file fan-out to trace at batch grain"
        ),
        detail={
            "rows": int(len(df)),
            "artifacts": [path.name for path in _stage_outputs()[0]],
        },
        source="raw export (core.common.load_raw_export)",
    )
    _record_flag_census(trace, canon)


def _record_flag_census(trace: TraceRun, canon) -> None:
    """One group row per attribute-consistency flag, plus the flagged GTINs.

    A GTIN can carry several flags, so the entity here is the (gtin, flag) pair
    and the census counts GTINs per flag — exactly what ``_flag_census`` counts.
    """
    records: list[dict[str, str]] = []
    if "attribute_consistency_flags" in canon:
        for gtin, flags in zip(canon["gtin"], canon["attribute_consistency_flags"], strict=True):
            if not isinstance(flags, (list, tuple, set)):
                continue
            for flag in flags:
                records.append({"gtin": str(gtin), "flag": str(flag)})
    trace.add_entities(
        "flags",
        records,
        key_of=lambda record: record["gtin"],
        reason_of=lambda record: record["flag"],
        detail_of=lambda record: {"gtin": record["gtin"]},
        source="canonical_records.csv attribute_consistency_flags",
        per_reason=ENTITY_SAMPLE_PER_REASON,
        total_cap=ENTITY_ROW_CAP,
    )


if __name__ == "__main__":
    main()
