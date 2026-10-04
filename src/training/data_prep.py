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
"""

from __future__ import annotations


from core.common import DATA_PATH, SEED, F, CONFIG_PATH, VOCABULARY_CONFIG_PATH, load_raw_export
from core.manifest import begin_manifest, finish_manifest
from core.tracing import trace_path
from pipeline import run_within_brand_pipeline


def main() -> None:
    # Stage manifest (SILENT_DROPS task 6) — begin BEFORE the work: the
    # raw export is hashed now (53MB, chunked) so the record pins exactly
    # what this stage read. Seed = the SSOT seed; the pipeline is
    # deterministic, no RNG is consumed.
    manifest = begin_manifest("data_prep", inputs=[DATA_PATH, CONFIG_PATH, VOCABULARY_CONFIG_PATH, F["number_reference"]], seed=SEED)
    df = load_raw_export()
    pairs, canon = run_within_brand_pipeline(df)
    print(f"pairs: {len(pairs):,} | canonical records: {len(canon):,}")
    # AUDIT 2026-09-09: digit tokens resolved by the regex fallback (not in
    # the reference CSV) — the one degradation the numbers lane allows;
    # printed so it can never be silent.
    import pipeline as _dp

    print(
        f"[numbers] {_dp._UNSEEN_TOKEN_TOTAL:,} digit-token resolutions "
        f"via regex fallback (not in reference CSV)"
    )

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
    # The stage's outputs are its two frozen artifacts plus the consolidated
    # trace. The trace used to be listed here by its OLD per-stage name
    # (results/logs/gate_visibility.csv), whose writer this directive deleted —
    # finish_manifest hashes every listed output, so a stale name made the
    # stage die with FileNotFoundError after all the work was done. The name
    # now comes from the layout that owns it (core.tracing → training_trace).
    out_paths = [
        F["canonical_records"],
        F["gate_results"],
        trace_path(),
    ]
    expected = [
        F["canonical_records"].name,
        F["gate_results"].name,
        trace_path().name,
    ]
    mpath = finish_manifest(
        manifest, out_paths, row_accounting, expected_outputs=expected
    )
    n_in = row_accounting["input_rows"]
    n_out = row_accounting["output_rows"]
    n_drop = sum(row_accounting["dropped"].values())
    print(
        f"[manifest] data_prep complete -> {mpath} | "
        f"closure {n_in:,} == {n_out:,} kept-visible + {n_drop:,} dropped"
    )


def _dp_manifest_accounting(df, pairs, canon) -> dict:
    """Capture-only row accounting for the data_prep manifest.

    input_rows: every raw-export row the pipeline read. dropped: the
    gtin-guard populations (missing/NaN gtin, failed GS1 checksum) —
    run_within_brand_pipeline prints both, and the numbers below are
    recomputed from the frame the same way the guard computes them, so
    the manifest can never drift from the code's truth. Rows with a
    valid gtin collapse into one canonical record per gtin (not a
    "drop" — they're aggregated); that population is recorded under
    collapsed_same_gtin so the closure reads input == output_rows +
    dropped + collapsed.
    """

    from core.gtin import gtin_validity

    from core.identity_policy import apply_identity_links
    df = apply_identity_links(df)
    gtin_valid = (
        df["gtin"].notna()
        & (df["gtin"].astype(str).str.strip() != "")
        & (df["gtin"].astype(str).str.lower() != "nan")
    )
    bc_valid = gtin_validity(df["gtin"].fillna("").astype(str).str.strip())
    bc_valid.index = df.index
    n_missing_gtin = int((~gtin_valid).sum())
    from core.identity_policy import reviewed_row_mask
    reviewed = reviewed_row_mask(df)
    bc_valid &= ~reviewed
    n_checksum = int((gtin_valid & ~bc_valid & ~reviewed).sum())
    n_valid = int((gtin_valid & bc_valid).sum())
    # canonical records = distinct valid gtins; collapsed = valid rows
    # folded into them
    n_canon = len(canon)
    collapsed = n_valid - n_canon
    return {
        "input_rows": len(df),
        "output_rows": n_canon,
        "dropped": {
            "gtin_missing_or_nan": n_missing_gtin,
            "gtin_checksum_failed": n_checksum,
            "identity_review_quarantined": int(reviewed.sum()),
        },
        # kept-and-aggregated, NOT dropped (closure-extension key)
        "collapsed_same_gtin": collapsed,
        # pair-level population (not row accounting, but pinned here so
        # the gate census can't silently thin)
        "gate_pairs": len(pairs),
    }


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
    column = getattr(canon, "get", None)
    if column is None or "attribute_consistency_flags" not in canon:
        return {}
    for flags in canon["attribute_consistency_flags"]:
        if not isinstance(flags, (list, tuple, set)):
            continue
        for flag in flags:
            counts[str(flag)] += 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    main()
