"""Owner's zero-shot embedding-similarity script (bundle-adapted paths).

Encodes configured canonical model inputs for every gate pair and stores
per-model cosine similarity in embedding_similarities.csv. The configured
masking policy is applied once per canonical GTIN, and every output row carries
the raw/model input text plus source and canonical metadata needed to audit it.

MANIFEST (SILENT_DROPS task 7): the stage snapshots its inputs
(canonical_records.csv + gate_results.csv, plus the existing
embedding_similarities.csv when a resume read is about to happen), writes
the sims CSV atomically (atomic_write_csv — the incremental per-model
write can no longer leave a truncated CSV on the final path, which a
skip-if-exists rerun would then treat as good), and publishes
results/manifests/zero_shot_sims.json LAST. The per-model `.model_fp`
fingerprint stamps become manifest OUTPUTS (sha256-pinned), so a torn or
tampered stamp is detectable without a re-run.

ROW ACCOUNTING (code truth): every selected gate pair is scored and every
selected pair is written — the `[keep]` projection at the write is a COLUMN
selection (identity columns + every sim_* column), not a row filter. A
`--sample` run records the intentionally excluded tail as `sample_cap` in
the manifest; a full run has no dropped rows. The code scores EVERY selected
pair (hard_no included — the eval lane draws its negatives from hard_no
rows; skipping them was the silent class drop that rule closed).
"""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model directories resolve through the shared local-only registry. The old
# local _m() copy and Hub fallback were removed; missing DVC bundles fail
# before any encoder is constructed.
from core.common import (
    F,
    SEED,
    embedding_model_keys,
    ensure_parent,
    load_dataset,
    load_config,
    load_local_sentence_transformer,
    resolve_model,
    training_cfg,
)
from core.manifest import atomic_write_csv, begin_manifest, finish_manifest
from core.schemas import ZERO_SHOT_TRACE_COLUMNS, check_zero_shot_similarity_frame
from training.masking import mask_text

_cfg = load_config()

MODEL_KEYS = embedding_model_keys()
SIM_COLUMNS = {k: v for k, v in _cfg["sim_columns"].items()}

# --models lane selector: score ONLY the named models (comma-separated).
# The deberta lane is GPU-only in practice (14h+ on CPU) and used to hold
# the whole script hostage: completing the two MiniLM columns required
# manual kills that raced the incremental CSV writes. Explicit lanes:
#   python -m training.zero_shot_sims --models minilm_l6,multilingual_l12
# --models lane selector moved INTO main() (audit 2026-09-09): was a manual
# sys.argv.index("--models") parse that (a) crashed with IndexError when
# --models was the last token, (b) had no --help, and (c) ran at MODULE
# level — importing this module executed the whole scoring pipeline.
# CPU note: deberta-v3's relative attention is ~2000x slower than MiniLM on
# this torch-CPU build (measured 3.9 s/text vs 2 ms/text) — run deberta on
# the GPU lane; a CPU sweep leaves its column absent (04 warns, skips it).


def _json_text(value: object) -> str:
    """Serialize a trace value without turning missing data into ``nan``."""
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _source_trace_map(source: pd.DataFrame) -> dict[str, list[dict[str, str]]]:
    """Index every raw source row by GTIN without collapsing its metadata."""
    required = {"barcode", "product_id"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"source dataset missing trace columns: {missing}")
    indexed: dict[str, list[dict[str, str]]] = {}
    for source_row_id, row in source.reset_index(drop=True).iterrows():
        gtin = _json_text(row["barcode"]).strip()
        if not gtin:
            continue
        indexed.setdefault(gtin, []).append(
            {
                "source_row_id": str(source_row_id),
                **{str(column): _json_text(value) for column, value in row.items()},
            }
        )
    return indexed


def _masked_inputs(
    gtins: list[str],
    texts: dict[str, str],
    *,
    seed: int,
) -> dict[str, dict[str, object]]:
    """Apply configured masking once per canonical input.

    Zero-shot has no labels, so ``masking.frac`` selects canonical inputs,
    rather than pair rows. Every pair then uses the same traceable input for a
    GTIN; this avoids row-order-dependent masking and keeps resume fingerprints
    stable. Both endpoints can be masked when both are selected.
    """
    spec = training_cfg().masking
    rng = random.Random(seed)
    selected_count = int(len(gtins) * spec.frac) if spec.enabled else 0
    selected = set(rng.sample(gtins, selected_count)) if selected_count else set()
    output: dict[str, dict[str, object]] = {}
    for gtin in gtins:
        raw = texts[gtin]
        if not spec.enabled:
            status = "disabled"
            model_input, extent = raw, 0.0
        elif gtin not in selected:
            status = "not_selected"
            model_input, extent = raw, 0.0
        else:
            model_input, extent = mask_text(
                raw,
                mask_prob=spec.mask_prob,
                rng=rng,
                lo=spec.mask_lo,
                hi=spec.mask_hi,
            )
            status = "masked" if extent > 0.0 and model_input != raw else "selected_noop"
        output[gtin] = {
            "raw": raw,
            "model_input": model_input,
            "status": status,
            "applied": status == "masked",
            "extent": float(extent),
        }
    if len(output) != len(gtins):
        raise AssertionError("zero-shot masking lost canonical inputs")
    return output


def _mask_config_fingerprint(seed: int) -> str:
    """Fingerprint the validated masking SSOT and deterministic seed."""
    spec = training_cfg().masking
    payload = {
        "seed": seed,
        "enabled": spec.enabled,
        "frac": spec.frac,
        "mask_prob": spec.mask_prob,
        "mask_lo": spec.mask_lo,
        "mask_hi": spec.mask_hi,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _build_trace_frame(
    candidates: pd.DataFrame,
    canonical_records: pd.DataFrame,
    source_rows: dict[str, list[dict[str, str]]],
    masked: dict[str, dict[str, object]],
    selected_keys: tuple[str, ...],
    mask_fp: str,
) -> pd.DataFrame:
    """Build one fully traceable output row for every gate pair."""
    canonical_by_gtin = canonical_records.set_index("gtin", drop=False)
    endpoint_gtins = set(candidates["gtin1"]) | set(candidates["gtin2"])
    missing = sorted(endpoint_gtins - set(canonical_by_gtin.index))
    if missing:
        raise ValueError(f"canonical records missing gate GTINs: {missing[:5]}")
    missing_source = sorted(endpoint_gtins - set(source_rows))
    if missing_source:
        raise ValueError(f"source rows missing gate GTINs: {missing_source[:5]}")

    model_keys_json = json.dumps(list(selected_keys), separators=(",", ":"))
    trace_rows: list[dict[str, object]] = []
    for row in candidates.itertuples(index=False):
        gtin1, gtin2 = str(row.gtin1), str(row.gtin2)
        c1 = canonical_by_gtin.loc[gtin1].to_dict()
        c2 = canonical_by_gtin.loc[gtin2].to_dict()
        m1, m2 = masked[gtin1], masked[gtin2]
        lineage_payload = {
            "gtin1": gtin1,
            "gtin2": gtin2,
            "model_keys": selected_keys,
            "mask_fp": mask_fp,
            "model_input_text1": m1["model_input"],
            "model_input_text2": m2["model_input"],
        }
        lineage_id = hashlib.sha256(
            json.dumps(lineage_payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
        source1, source2 = source_rows[gtin1], source_rows[gtin2]
        trace_rows.append(
            {
                "gtin1": gtin1,
                "gtin2": gtin2,
                "gate_decision": str(row.gate_decision),
                "gate_reason": str(row.gate_reason),
                "canonical1": str(c1["canonical"]),
                "canonical2": str(c2["canonical"]),
                "canonical_model_text1": str(m1["raw"]),
                "canonical_model_text2": str(m2["raw"]),
                "model_input_text1": str(m1["model_input"]),
                "model_input_text2": str(m2["model_input"]),
                "source_row_ids1": json.dumps([x["source_row_id"] for x in source1], separators=(",", ":")),
                "source_row_ids2": json.dumps([x["source_row_id"] for x in source2], separators=(",", ":")),
                "source_sku_ids1": json.dumps([x["product_id"] for x in source1], separators=(",", ":")),
                "source_sku_ids2": json.dumps([x["product_id"] for x in source2], separators=(",", ":")),
                "source_metadata1": json.dumps(source1, sort_keys=True, separators=(",", ":")),
                "source_metadata2": json.dumps(source2, sort_keys=True, separators=(",", ":")),
                "canonical_metadata1": json.dumps(c1, sort_keys=True, default=str, separators=(",", ":")),
                "canonical_metadata2": json.dumps(c2, sort_keys=True, default=str, separators=(",", ":")),
                "mask_status1": str(m1["status"]),
                "mask_status2": str(m2["status"]),
                "mask_applied1": bool(m1["applied"]),
                "mask_applied2": bool(m2["applied"]),
                "mask_realized_extent1": float(m1["extent"]),
                "mask_realized_extent2": float(m2["extent"]),
                "mask_config_fingerprint": mask_fp,
                "model_keys": model_keys_json,
                "lineage_id": lineage_id,
            }
        )
    result = pd.DataFrame(trace_rows, columns=list(ZERO_SHOT_TRACE_COLUMNS))
    if len(result) != len(candidates):
        raise AssertionError("zero-shot trace rows do not close against gate pairs")
    return result


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--models",
        type=str,
        default=None,
        help="comma-separated model keys to score (default: all in config)",
    )
    ap.add_argument(
        "--sample",
        type=int,
        default=None,
        help="debug: cap gate rows for an explicit, traceable smoke run",
    )
    args = ap.parse_args()
    if args.sample is not None and args.sample < 1:
        raise SystemExit("--sample must be >= 1 when provided")
    selected_keys = MODEL_KEYS
    if args.models is not None:
        selected_keys = tuple(m for m in args.models.split(",") if m)
        missing = [m for m in selected_keys if m not in MODEL_KEYS]
        if missing:
            raise SystemExit(
                f"unknown --models entries: {missing} (have {list(MODEL_KEYS)})"
            )
    models = {key: resolve_model(key) for key in selected_keys}
    model_provenance = {
        key: {
            "model_key": key,
            "bundle_path": str(path),
            "source": "config.paths.models_registry",
        }
        for key, path in models.items()
    }
    if args.models is not None:
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
    manifest_inputs = [df_canon_path, df_gate_path, F["dataset"]]
    if out.exists():
        manifest_inputs.append(out)
    manifest = begin_manifest("zero_shot_sims", inputs=manifest_inputs, seed=SEED)

    df_canon = pd.read_csv(df_canon_path, dtype=str, keep_default_na=False)
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
    full_gate_rows = len(df_gate)
    candidates = df_gate.copy()
    if args.sample is not None:
        candidates = candidates.head(args.sample).reset_index(drop=True)
        print(
            f"SAMPLE MODE: selected first {len(candidates):,} of "
            f"{full_gate_rows:,} gate pairs",
            flush=True,
        )
    print(f"Total gate pairs to score: {len(candidates)}")

    unique_gtins = sorted(set(candidates["gtin1"]).union(set(candidates["gtin2"])))
    raw_texts = {g: gtin_to_canon[g] for g in unique_gtins}
    masked = _masked_inputs(unique_gtins, raw_texts, seed=SEED)
    model_texts = [str(masked[g]["model_input"]) for g in unique_gtins]
    mask_fp = _mask_config_fingerprint(SEED)
    source_rows = _source_trace_map(load_dataset())
    results = _build_trace_frame(
        candidates,
        df_canon,
        source_rows,
        masked,
        tuple(selected_keys),
        mask_fp,
    )
    print(f"Unique GTINs to encode: {len(unique_gtins)}")

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

    _canon_fp = _hashlib.sha256(
        json.dumps(
            {"model_inputs": model_texts, "mask_config": mask_fp},
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
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
        elif any(column not in done.columns for column in ZERO_SHOT_TRACE_COLUMNS):
            print(
                "[resume] existing similarity output lacks the traceability "
                "contract — rebuilding all selected model lanes",
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
        model = load_local_sentence_transformer(model_key, device=DEVICE)
        model.max_seq_length = int(load_config()["training"]["max_seq_length"])  # SSOT
        embeddings = model.encode(
            model_texts,
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
        keep = list(ZERO_SHOT_TRACE_COLUMNS) + [
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
    check_zero_shot_similarity_frame(final)
    sim_cols = [c for c in final.columns if c.startswith("sim_")]
    row_accounting = {
        "input_rows": full_gate_rows,
        "output_rows": len(final),
        "dropped": (
            {"sample_cap": full_gate_rows - len(candidates)}
            if args.sample is not None
            else {}
        ),
        "sample": args.sample or "full",
        "selected_rows": len(candidates),
        # coverage census (not drops): which sim columns are on disk and
        # fully populated after this run
        "sim_columns": sorted(sim_cols),
        "sim_columns_complete": sorted(
            c for c in sim_cols if final[c].notna().all()
        ),
        "unique_gtins_encoded": len(unique_gtins),
        "canonical_text_fp": _canon_fp,
        "mask_config_fingerprint": mask_fp,
        "masking_status_counts": {
            side: final[f"mask_status{side}"].value_counts().to_dict()
            for side in ("1", "2")
        },
        "model_provenance": model_provenance,
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
        f"{row_accounting['output_rows']:,} kept + "
        f"{sum(row_accounting['dropped'].values()):,} intentionally excluded "
        f"(column-only [keep] projection; sim columns: "
        f"{', '.join(sorted(sim_cols))})"
    )

    print(f"\nSaved {out} ({len(results):,} rows)")


if __name__ == "__main__":
    main()
