"""Owner's zero-shot embedding-similarity script (bundle-adapted paths).

Encodes canonical strings for every non-hard_no candidate pair with each of
the 3 models and stores per-model cosine similarity in embedding_similarities.csv.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# model dirs resolve via lib.common.resolve_model (config models_dir /
# models_dir_sibling); hub id is the offline-last-resort fallback. The
# old local _m() copy was removed 2026-09-08 — ONE registry-aware resolver
# for the whole tree (TRAIN + run_all).
from euromonitor.core.common import RESULTS, F, load_config, resolve_model

_cfg = load_config()

MODELS = {k: resolve_model(sub) for k, sub in _cfg["models"].items()}
SIM_COLUMNS = {k: v for k, v in _cfg["sim_columns"].items()}

# --models lane selector: score ONLY the named models (comma-separated).
# The deberta lane is GPU-only in practice (14h+ on CPU) and used to hold
# the whole script hostage: completing the two MiniLM columns required
# manual kills that raced the incremental CSV writes. Explicit lanes:
#   python src/euromonitor/training/zero_shot_sims.py --models minilm_l6,multilingual_l12
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

    df_canon = pd.read_csv(RESULTS / F["canonical_records"])
    df_gate = pd.read_csv(RESULTS / F["gate_results"], dtype={"gtin1": str, "gtin2": str})
    assert "gtin" in df_canon.columns and "canonical" in df_canon.columns
    # MODEL-side canonical (number-free + schema-free) — the SAME text the
    # trainer encodes (consistency: eval measures the payload the model runs
    # on; the gate's raw numeric canonical never reaches an encoder)
    from euromonitor.pipeline import canonical_model_text, strip_schema_words

    gtin_to_canon = {
        g: strip_schema_words(canonical_model_text(c))
        for g, c in zip(df_canon["gtin"].astype(str), df_canon["canonical"].astype(str))
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
    out = RESULTS / F["embedding_similarities"]

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
        key_new = list(zip(results["gtin1"], results["gtin2"]))
        key_old = list(zip(done["gtin1"], done["gtin2"]))
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
        # incremental write: a crash never loses the completed models
        keep = ["gtin1", "gtin2", "gate_decision", "gate_reason"] + [
            c for c in results.columns if c.startswith("sim_")
        ]
        results[keep].to_csv(out, index=False)
        print(f"    wrote {col} ({len(results):,} rows)", flush=True)
        # per-model fingerprint stamp: certifies THIS column's scores against
        # the exact canonical texts they were computed from. Written after the
        # column's own successful write — a crash in a LATER model (deberta on
        # CPU) never invalidates the completed ones, and a changed-canonical
        # run leaves every stamp mismatched so only truly-fresh columns resume.
        (out.parent / f"{out.name}.model_fp.{col}").write_text(_canon_fp)

    print(f"\nSaved {out} ({len(results):,} rows)")


if __name__ == "__main__":
    main()

