"""Owner's zero-shot embedding-similarity script (bundle-adapted paths).

Encodes canonical strings for every non-hard_no candidate pair with each of
the 3 models and stores per-model cosine similarity in embedding_similarities.csv.

MANIFEST (SILENT_DROPS task 7): the stage snapshots its inputs
(canonical_records.csv + gate_results.csv, plus the existing
embedding_similarities.csv when a resume read is about to happen), writes
the sims CSV atomically (atomic_write_csv — the incremental per-model
write can no longer leave a truncated CSV on the final path, which a
skip-if-exists rerun would then treat as good), and publishes
results/manifests/zero_shot_sims.json LAST. The per-model `.model_fp`
fingerprint stamps become manifest OUTPUTS (sha256-pinned), so a torn or
tampered stamp is detectable without a re-run.

ROW ACCOUNTING (code truth): every gate pair in gate_results.csv is
scored and every scored pair is written — the `[keep]` projection at the
write is a COLUMN selection (identity columns + every sim_* column), not
a row filter, so no row ever vanishes between input and output and
`dropped` is EMPTY (closure input == output). The docstring's historic
"non-hard_no" phrasing is stale: the code scores EVERY pair (hard_no
included — the eval lane draws its negatives from hard_no rows; skipping
them was the silent class drop that rule closed).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model directories resolve through the shared local-only registry. The old
# local _m() copy and Hub fallback were removed; missing DVC bundles fail
# before any encoder is constructed.
from core.common import SEED, F, ensure_parent, load_config, resolve_model
from core.manifest import atomic_write_csv, begin_manifest, finish_manifest

_cfg = load_config()

MODELS = {k: resolve_model(sub) for k, sub in _cfg["models"].items()}
SIM_COLUMNS = {k: v for k, v in _cfg["sim_columns"].items()}

# --models lane selector: score ONLY the named models (comma-separated).
# The deberta lane is GPU-only in practice (14h+ on CPU) and used to hold
# the whole script hostage: completing the two MiniLM columns required
# manual kills that raced the incremental CSV writes. Explicit lanes:
#   python src/training/zero_shot_sims.py --models minilm_l6,multilingual_l12
# --models lane selector moved INTO main() (audit 2026-09-09): was a manual
# sys.argv.index("--models") parse that (a) crashed with IndexError when
# --models was the last token, (b) had no --help, and (c) ran at MODULE
# level — importing this module executed the whole scoring pipeline.
# CPU note: deberta-v3's relative attention is ~2000x slower than MiniLM on
# this torch-CPU build (measured 3.9 s/text vs 2 ms/text) — run deberta on
# the GPU lane; a CPU sweep leaves its column absent (04 warns, skips it).


def main() -> None:
    import argparse


    models = dict(MODELS)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--models",
        type=str,
        default=None,
        help="comma-separated model keys to score (default: all in config)",
    )
    args = ap.parse_args()
    if args.models is not None:
        sel = [m for m in args.models.split(",") if m]
        missing = [m for m in sel if m not in MODELS]
        if missing:
            raise SystemExit(
                f"unknown --models entries: {missing} (have {list(MODELS)})"
            )
        models = {k: v for k, v in MODELS.items() if k in sel}
        print(f"[models] scoring lanes only: {list(models)}", flush=True)

    df_canon_path = F["canonical_records"]
    df_gate_path = F["gate_results"]
    out = F["embedding_similarities"]
    # Stage manifest (SILENT_DROPS task 7) — begin BEFORE the work. The
    # fresh-resume read below consumes the PREVIOUS run's CSV, so when that
    # read is about to happen it is recorded as an input too (hashed
    # before the file is replaced). Seed = the SSOT seed; encode calls are
    # deterministic, the component split downstream (not this stage) is
    # what consumes RNG.
    manifest_inputs = [df_canon_path, df_gate_path]
    if out.exists():
        manifest_inputs.append(out)
    manifest = begin_manifest("zero_shot_sims", inputs=manifest_inputs, seed=SEED)

    df_canon = pd.read_csv(df_canon_path)
    df_gate = pd.read_csv(df_gate_path, dtype={"gtin1": str, "gtin2": str})
    assert "gtin" in df_canon.columns and "canonical" in df_canon.columns
    # MODEL-side canonical (number-free + schema-free) — the SAME text the
    # trainer encodes (consistency: eval measures the payload the model runs
    # on; the gate's raw numeric canonical never reaches an encoder)
    from pipeline import canonical_model_text, strip_schema_words

    gtin_to_canon = {
        g: strip_schema_words(canonical_model_text(c))
        for g, c in zip(df_canon["gtin"].astype(str), df_canon["canonical"].astype(str), strict=True)
    }

    # score EVERY gate pair (candidates AND hard_no): the evaluation set
    # (labeled_pairs) draws its negatives from hard_no rows — skipping them
    # would leave 04 with positives only (silent class drop)
    candidates = df_gate.copy()
    print(f"Total gate pairs to score: {len(candidates)}")

    unique_gtins = sorted(set(candidates["gtin1"]).union(set(candidates["gtin2"])))
    texts = [gtin_to_canon[g] for g in unique_gtins]
    print(f"Unique GTINs to encode: {len(unique_gtins)}")

    results = candidates[["gtin1", "gtin2", "gate_decision", "gate_reason"]].copy()

    # resume: models already scored in a previous (crashed) run are skipped —
    # but ONLY when the stored pair SET matches the current gate rows exactly.
    # The old check (column exists & notna) silently reused stale sims when
    # data_prep regenerated gate_results with different pairs/canonicals —
    # a drift bug: sims from the OLD canonicals attached to NEW gate rows.
    # CANONICAL-TEXT GUARD: the pair sequence alone is NOT sufficient — the
    # phrase-variation fix (2026-09-07) changed 3,205 canonical texts without
    # touching a single gate pair; the encoded texts changed, so every stored
    # sim is stale even though the pairs match. A cheap text fingerprint (the
    # sha256 of the joined canonical texts actually about to be encoded) pins
    # the sims to the exact canonical content they were computed from.
    import hashlib as _hashlib

    _canon_fp = _hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()[:16]
    have: list[str] = []
    # fresh-resumed columns (already on disk with a valid fingerprint stamp) must
    # ride along on EVERY incremental write — the old write rebuilt the CSV from
    # `results` alone, silently DROPPING every resumed column (empirically
    # verified: score minilm after multilingual -> multilingual column vanished
    # while its stamp stayed). Carry them in `done` so the write keeps them.
    resumed: dict[str, pd.Series] | None = None
    if out.exists():
        done = pd.read_csv(out, dtype={"gtin1": str, "gtin2": str})
        key_new = list(zip(results["gtin1"], results["gtin2"], strict=True))
        key_old = list(zip(done["gtin1"], done["gtin2"], strict=True))
        if key_new != key_old:
            print(
                "[resume] gate pair sequence changed since the last scoring — "
                "re-scoring ALL models (stale sims discarded)",
                flush=True,
            )
            done = None
        else:
            # PER-MODEL fingerprint stamps: a column resumes ONLY when its own
            # stamp matches the canonical texts about to be encoded. No stamp
            # (column predates the contract) = provenance unverifiable = the
            # column re-scores; deleting stamps must never upgrade stale to
            # fresh. All-columns-stale still discards the whole CSV for a clean
            # rewrite (no half-CSV mixes old and new canonical scores).
            stale_cols = []
            for c in done.columns:
                if not (c.startswith("sim_") and done[c].notna().all()):
                    continue
                stamp = out.parent / f"{out.name}.model_fp.{c}"
                fp_col = stamp.read_text().strip() if stamp.exists() else None
                if fp_col != _canon_fp:
                    stale_cols.append(c)
            if stale_cols:
                print(
                    f"[resume] canonical texts changed (fp {_canon_fp}); stale "
                    f"columns re-scored: {stale_cols}",
                    flush=True,
                )
                done = None
            else:
                have = [
                    c
                    for c in done.columns
                    if c.startswith("sim_") and done[c].notna().all()
                ]
                if have:
                    # keep the fresh columns' data for the incremental write
                    resumed = {c: done[c].copy() for c in have}
                    print(f"resuming — already scored: {have}", flush=True)

    # fingerprint stamps written by THIS run become manifest outputs (the
    # resumed-fresh ones are re-written only if their lane re-scores; a
    # skipped lane keeps its existing stamp on disk — not an output of
    # this run, so it is not claimed as one)
    manifest_stamps: list[Path] = []

    for model_key, model_path in models.items():
        col = SIM_COLUMNS[model_key]
        if col in have:
            print(f"--- Model: {model_key} already scored, skip ---", flush=True)
            continue
        print(f"\n--- Model: {model_key} ---", flush=True)
        model = SentenceTransformer(model_path, device=DEVICE)
        model.max_seq_length = int(load_config()["training"]["max_seq_length"])  # SSOT
        embeddings = model.encode(
            texts,
            batch_size=int(load_config()["training"]["batch_size_embed"]),  # SSOT
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        # vectorized: row-indexed gather, one einsum (no iterrows)
        gtin_idx = {g: i for i, g in enumerate(unique_gtins)}
        ai = results["gtin1"].map(gtin_idx).to_numpy()
        bi = results["gtin2"].map(gtin_idx).to_numpy()
        results[col] = np.einsum("ij,ij->i", embeddings[ai], embeddings[bi])
        del model, embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # carry resumed-fresh columns through the write (they are still valid:
        # pair sequence AND fingerprint both verified above)
        if resumed:
            for c, vals in resumed.items():
                if c not in results.columns:
                    results[c] = vals
        # incremental write: a crash never loses the completed models —
        # atomic_write_csv (SILENT_DROPS task 7) additionally guarantees the
        # final path never holds a truncated CSV mid-write, so a crashed run
        # leaves the PREVIOUS complete CSV + a re-scored lane, never a
        # half-written file a resume would trust.
        keep = ["gtin1", "gtin2", "gate_decision", "gate_reason"] + [
            c for c in results.columns if c.startswith("sim_")
        ]
        atomic_write_csv(results[keep], ensure_parent(out), index=False)
        print(f"    wrote {col} ({len(results):,} rows)", flush=True)
        # per-model fingerprint stamp: certifies THIS column's scores against
        # the exact canonical texts they were computed from. Written after the
        # column's own successful write — a crash in a LATER model (deberta on
        # CPU) never invalidates the completed ones, and a changed-canonical
        # run leaves every stamp mismatched so only truly-fresh columns resume.
        stamp_path = out.parent / f"{out.name}.model_fp.{col}"
        stamp_path.write_text(_canon_fp)
        manifest_stamps.append(stamp_path)

    # ---- row accounting (SILENT_DROPS task 7; capture-only) ────────────────
    # CODE TRUTH: the `[keep]` projection above is a COLUMN selection
    # (identity columns + every sim_* column), NOT a row filter — every
    # gate pair is scored and every scored row is written, so NO row is
    # ever dropped between input and output and `dropped` is EMPTY: the
    # closure is the identity input_rows == output_rows. (A lane-limited
    # --models run still scores ALL pairs, just with fewer sim columns.)
    # The sim-column population is recorded OUTSIDE `dropped` (it is a
    # coverage census, not a drop) so a silently-missing model column is
    # visible in the manifest without breaking the closure invariant.
    final = pd.read_csv(out, dtype={"gtin1": str, "gtin2": str})
    sim_cols = [c for c in final.columns if c.startswith("sim_")]
    row_accounting = {
        "input_rows": len(candidates),
        "output_rows": len(final),
        "dropped": {},
        # coverage census (not drops): which sim columns are on disk and
        # fully populated after this run
        "sim_columns": sorted(sim_cols),
        "sim_columns_complete": sorted(
            c for c in sim_cols if final[c].notna().all()
        ),
        "unique_gtins_encoded": len(unique_gtins),
        "canonical_text_fp": _canon_fp,
        "lanes_scored_this_run": sorted(models),
    }
    outputs = [out] + manifest_stamps
    # expected_outputs = what THIS run must have produced: the CSV always,
    # a fingerprint stamp only when its lane actually scored (a resumed-
    # skipped lane keeps its OLD stamp on disk — valid, but not written by
    # this run, so it is an output of the run that wrote it, not this one)
    expected = [F["embedding_similarities"]] + [
        f"{out.name}.model_fp.{SIM_COLUMNS[k]}"
        for k in models
        if SIM_COLUMNS[k] not in have  # only lanes that scored this run
    ]
    manifest_path = finish_manifest(
        manifest,
        outputs=outputs,
        row_accounting=row_accounting,
        expected_outputs=expected,
    )
    print(
        f"[manifest] zero_shot_sims complete -> {manifest_path} | "
        f"closure {row_accounting['input_rows']:,} == "
        f"{row_accounting['output_rows']:,} kept + 0 dropped "
        f"(column-only [keep] projection; sim columns: "
        f"{', '.join(sorted(sim_cols))})"
    )

    print(f"\nSaved {out} ({len(results):,} rows)")


if __name__ == "__main__":
    main()
