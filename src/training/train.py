"""train.py — the TRAIN_GPU training entry point (the ONE callable in
TRAIN_GPU root; the training internals live in src/training/).

Trains on the OFFICIAL clean-sku scheme (DATA_PIPE.pairs):
  positive : (clean_sku_text, canonical_gtin)
  negative : (clean_sku_text, canonical_other_gtin) — gate hard-no pairs
Run:  python train.py --model models/<name> [--split holdout|cv ...]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from core.common import (
    RESULTS,
    SEED,
    F,
    artifact,
    ensure_parent,
    load_config,
    load_dataset_deduped,
    plot_dpi,
    runtime,
    set_determinism,
    training_cfg,
)
from core.common import SSOT_LOSS as _SSOT_LOSS
from core.common import SSOT_CONTRASTIVE_MARGIN as _SSOT_CONTRASTIVE_MARGIN
from training.folds import component_folds
from training.training import ES_PATIENCE, ES_THRESHOLD, train_one_config

# Colab workers should spend GPU time only on training and the score exports
# needed by the local post-processing lane.  Reports, plots, artifact bundles,
# and final DVC publishing are CPU/network work and are intentionally deferred
# until the worker outputs have been downloaded locally.
_REMOTE_TRAINING = os.environ.get("EUROMONITOR_REMOTE_TRAINING") == "1"
_UNIFORMITY_CFG = load_config()["training"]["uniformity_regularization"]

# thresholds live in config/paths.yaml (pairs.proceed_sim_threshold /
# pairs.hardneg_sim_threshold) and are read by pipeline.build_training_data
# directly — no duplicate constants here (they were dead globals: set but
# never read anywhere).


def load_training_data(df: pd.DataFrame, payload_variant: str = "full") -> dict:
    """OFFICIAL pair construction (DATA_PIPE.pairs): payload = clean sku
    text per row + canonical per GTIN; pos = (sku, own canonical), neg =
    (sku, other canonical) from gate hard-no pairs."""
    from pipeline import build_training_data

    return build_training_data(df, payload_variant=payload_variant)


def _emit_07_series(ok_rows: list[dict], args) -> None:
    """07b/07c/07d CSV emission — the report_plots inputs (owner ruling).

    07c (field ablation) and 07d (data scaling) are APPEND-MODE: each
    src/training/train invocation adds its variant's row, so run_all's sweep
    accumulates instead of overwriting (the plots read whatever rows exist).
    07b (four-population scores) is written by the rerank lane, which has
    the trained embeddings; a plain run notes its absence.
    """

    from training.hpo_metrics import CALIBRATION_AGGREGATE_FIELDS

    calibration_rows = [
        row for row in ok_rows
        if row.get("calibration_status", "available") == "available"
    ]
    missing_fields = sorted(
        {
            field
            for field in CALIBRATION_AGGREGATE_FIELDS
            if any(field not in row for row in calibration_rows)
        }
    )
    if missing_fields and calibration_rows:
        raise ValueError(
            "calibration metric contract missing from fold rows: "
            f"{missing_fields}"
        )
    if not calibration_rows:
        print(
            "[calibration] unavailable for all completed folds; "
            "07-series calibration aggregates will be NaN",
            flush=True,
        )

    # ---- 07c: one aggregate row per payload variant ----
    aggregate_cache: dict[str, float] = {}

    def agg(field: str) -> float:
        """Compute each fold aggregate once for both 07-series rows."""
        if field not in aggregate_cache:
            vals = [r.get(field) for r in ok_rows if r.get(field) is not None]
            aggregate_cache[field] = float(np.mean(vals)) if vals else float("nan")
        return aggregate_cache[field]

    row_07c = {
        "variant": args.payload,
        "average_precision": round(agg("pr_auc"), 4),
        "precision_at_1": round(agg("precision_at_1"), 4),
        "recall_at_1": round(agg("recall_at_1"), 4),
        "precision_at_5": round(agg("precision_at_5"), 4),
        "recall_at_5": round(agg("recall_at_5"), 4),
        "precision_at_10": round(agg("precision_at_10"), 4),
        "recall_at_10": round(agg("recall_at_10"), 4),
        "hits_at_1": round(agg("hits_at_1"), 4),
        "precision_at_90pct_recall": round(agg("precision_at_90pct_recall"), 4),
        "tp_at_90pct_recall": round(agg("tp_at_90pct_recall"), 1),
        "fp_at_90pct_recall": round(agg("fp_at_90pct_recall"), 1),
        "threshold_at_90pct_recall": round(agg("threshold_at_90pct_recall"), 4),
        "auc": round(agg("auc"), 4),
        "n_folds": len(ok_rows),
        "model": args.model,
    }
    row_07c.update(
        {
            field: round(agg(field), 4)
            for field in CALIBRATION_AGGREGATE_FIELDS
        }
    )
    _append_csv(F["field_ablation"], [row_07c], "variant")

    # ---- 07d: one row per train fraction ----
    row_07d = {
        "fraction": args.train_frac,
        "n_triples": round(agg("n_train")),
        "average_precision": round(agg("pr_auc"), 4),
        "precision_at_1": round(agg("precision_at_1"), 4),
        "recall_at_1": round(agg("recall_at_1"), 4),
        "precision_at_5": round(agg("precision_at_5"), 4),
        "recall_at_5": round(agg("recall_at_5"), 4),
        "precision_at_10": round(agg("precision_at_10"), 4),
        "recall_at_10": round(agg("recall_at_10"), 4),
        "hits_at_1": round(agg("hits_at_1"), 4),
        "precision_at_90pct_recall": round(agg("precision_at_90pct_recall"), 4),
        "tp_at_90pct_recall": round(agg("tp_at_90pct_recall"), 1),
        "fp_at_90pct_recall": round(agg("fp_at_90pct_recall"), 1),
        "threshold_at_90pct_recall": round(agg("threshold_at_90pct_recall"), 4),
        "repeat": "single",
        "model": args.model,
        "payload": args.payload,
    }
    row_07d.update(
        {
            field: round(agg(field), 4)
            for field in CALIBRATION_AGGREGATE_FIELDS
        }
    )
    _append_csv(
        F["data_scaling"], [row_07d], ["fraction", "payload"]
    )


def _append_csv(
    path: Path, new_rows: list[dict], key_fields: str | list[str]
) -> None:
    """Append with replace-by-key semantics: re-running the SAME variant
    updates its row IN PLACE (row order preserved — plots read the sweep
    order from it; the old concat+drop_duplicates moved re-run rows to the
    END, silently corrupting the 07 series order) instead of duplicating
    it. Idempotent regeneration — the reproducibility contract."""
    import pandas as pd

    ensure_parent(path)
    if isinstance(key_fields, str):
        key_fields = [key_fields]
    new_df = pd.DataFrame(new_rows)
    if not path.exists():
        new_df.to_csv(path, index=False)
        return
    old = pd.read_csv(path)
    key = [k for k in key_fields if k in old.columns and k in new_df.columns]
    if not key:
        pd.concat([old, new_df], ignore_index=True).to_csv(path, index=False)
        return
    # index the old rows by key tuple -> row position. Keys compare on
    # STR — pandas reads "0.25" back as float 0.25, so a raw-tuple match
    # would miss on every numeric-looking key (07d's fraction column).
    pos = {
        tuple(str(r[k]) for k in key): i
        for i, r in enumerate(old.to_dict("records"))
    }
    out_rows = old.to_dict("records")
    appended = []
    for r in new_df.to_dict("records"):
        k = tuple(str(r[k_]) for k_ in key)
        if k in pos:
            out_rows[pos[k]] = {**out_rows[pos[k]], **r}  # replace in place
        else:
            appended.append(r)
    pd.DataFrame(out_rows + appended).to_csv(path, index=False)


def _log_run_artifacts_to_wandb(_wandb, *, run_tag: str, model_tag: str, metrics_path: Path, rows: list[dict]) -> None:
    """Publish every run-scoped result and checkpoint as one downloadable artifact."""
    artifact_paths: list[Path] = [metrics_path]
    latest_metrics = F["fold_metrics"]
    if latest_metrics.is_file() and latest_metrics != metrics_path:
        artifact_paths.append(latest_metrics)

    run_logs = RESULTS / "logs" / run_tag
    if run_logs.is_dir():
        artifact_paths.append(run_logs)
    report_dir = RESULTS / f"report_{run_tag}"
    if report_dir.is_dir():
        artifact_paths.append(report_dir)

    for row in rows:
        fold = row.get("fold")
        if not isinstance(fold, (int, np.integer)):
            continue
        artifact_paths.extend(
            RESULTS / f"train_{model_tag}_{run_tag}_fold{int(fold)}_{suffix}"
            for suffix in ("pairs.csv", "train_scores.csv", "random_easy_scores.csv")
        )
        artifact_paths.append(
            RESULTS / f"wandb_loss_by_epoch_{run_tag}_fold{int(fold)}.png"
        )
        artifact_paths.append(
            RESULTS / f"wandb_score_distributions_{run_tag}_fold{int(fold)}.png"
        )
        artifact_paths.append(
            artifact(
                "checkpoint_repo",
                {
                    "model_tag": model_tag,
                    "run_tag": run_tag,
                    "fold": int(fold),
                    "step": 0,
                },
            ).parent
        )
    artifact_paths.append(RESULTS / f"mask_effect_{run_tag}.png")

    pointer = F["results_pointer"]
    if pointer.is_file() and "_sample" not in metrics_path.name:
        artifact_paths.append(pointer)
    dvc_resume = RESULTS / ".resume"
    if dvc_resume.is_dir():
        artifact_paths.append(dvc_resume)

    # Keep the artifact deterministic if a path was discovered by more than
    # one glob, while preserving the first occurrence's useful directory name.
    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in artifact_paths:
        # realpath deduplicates aliases while retaining a stable canonical
        # identity across symlinked result/checkpoint mounts.
        identity = os.path.realpath(os.fspath(path))
        if identity not in seen and path.exists():
            seen.add(identity)
            unique_paths.append(path)
    if unique_paths:
        _wandb.log_artifacts(unique_paths, f"run-{run_tag}-downloadable")


def main() -> None:
    """Entry: everything runs inside one mlflow parent run (local backend
    + artifact store at artifacts/mlruns — browsable with
    `mlflow ui --backend-store-uri file:artifacts/mlruns`; MLFLOW_TRACKING_URI
    overrides, =off disables)."""
    from core.mlflow_ctx import MlflowCtx
    from core.wandb_ctx import WandbCtx

    run_name = "hpo" if "--hpo" in sys.argv else "train_gpu"
    with MlflowCtx(run_name) as _mlf, WandbCtx(run_name) as _wandb:
        _main_inner(_mlf, _wandb)


def _main_inner(_mlf, _wandb) -> None:
    # determinism FIRST (2026-10-06): one call before any model/data
    # randomness — masking augmentation, zero-shot encode, fold carving
    # and the grid/tpe sweep lanes (dispatched below, same entry) all
    # ride this seed. The component split keeps passing SEED explicitly
    # (its own contract); the GLOBAL RNGs (random/numpy/torch/cudnn)
    # are pinned here, once, at entry.
    set_determinism(SEED)
    cfg = load_config()
    # NO FALLBACKS (owner Q27): every section/key below is read with hard
    # indexing — a missing key crashes at startup, never a silent default.
    tr = cfg["training"]
    mining_cfg = cfg["mining"]
    mining_profile = os.environ.get("EUROMONITOR_MINING_PROFILE")
    if mining_profile:
        profile_cfg = cfg["mining_profiles"][mining_profile]
        mining_cfg = {
            **mining_cfg,
            "ann": {
                **mining_cfg["ann"],
                "enabled": bool(profile_cfg["ann_enabled"]),
            },
            "attribute_conflict": {
                **mining_cfg["attribute_conflict"],
                "enabled": bool(profile_cfg["attribute_conflict_enabled"]),
            },
        }
    ann_cfg = mining_cfg["ann"]
    attr_cfg = mining_cfg["attribute_conflict"]
    ann_mining_enabled = bool(ann_cfg["enabled"])
    attribute_conflict_enabled = bool(attr_cfg["enabled"])
    mining_enabled = ann_mining_enabled or attribute_conflict_enabled
    mask_cfg = cfg["masking"]
    split_cfg = cfg["split"]

    # Trainer models resolve through the project-owned registry. Resolution is
    # local-only: a missing DVC bundle fails before data preparation begins.
    from core.common import resolve_model

    default_model = resolve_model(tr["base_model"])
    default_band = str(ann_cfg["band"])

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=int(tr["epochs"]))
    ap.add_argument("--lr", type=float, default=float(tr["lr"]))
    ap.add_argument(
        "--warmup-ratio", type=float, default=None,
        help="final selected-run override; omitted values use training.yaml",
    )
    ap.add_argument(
        "--weight-decay", type=float, default=None,
        help="final selected-run override; omitted values use training.yaml",
    )
    ap.add_argument(
        "--split",
        choices=["holdout", "cv"],
        default=str(split_cfg["mode"]),
        help="holdout: ONE component-aware 50/25/25 "
        "train/dev/test split (the owner's spec) | "
        "cv: k component folds (research mode)",
    )
    ap.add_argument(
        "--folds",
        type=int,
        default=int(split_cfg["cv_folds"]),
        help="cv mode only: number of component folds",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=default_model,
        help="model id OR local path inside the bundle (models/...)",
    )
    ap.add_argument(
        "--band",
        type=str,
        default=default_band,
        help="mining band for in-batch hard negatives",
    )
    ap.add_argument(
        "--n-target-mining",
        type=int,
        default=int(ann_cfg["target"]),
        help="ANN mining target (SSOT: mining.ann.target)",
    )
    ap.add_argument(
        "--dev-fraction",
        type=float,
        default=runtime("dev_fraction"),  # SSOT training.dev_fraction — no inline literal
        help="dev share of the train side (cv mode); holdout uses the component split",
    )
    ap.add_argument(
        "--sample",
        type=int,
        default=None,
        help="debug: cap dataset rows (full chain, tiny data)",
    )
    ap.add_argument(
        "--mask-effect",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run the optional post-training masking robustness audit",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="resume each fold from its newest compatible checkpoint in this run directory",
    )
    # ── 07-series mirrors (all GPU-only experiments) ──────────────────────
    ap.add_argument(
        "--payload",
        choices=["full", "title_only"],
        default="full",
        help="07c field ablation: payload composition variant",
    )
    ap.add_argument(
        "--loss",
        choices=["contrastive", "mnrl", "triplet"],
        default=_SSOT_LOSS,  # training.loss SSOT — no fallback (owner Q27)
        help="training loss (contrastive = OnlineContrastiveLoss over labeled "
        "hard pos/neg pairs — the lane default, owner ruling 2026-09-07; "
        "mnrl = in-batch ranking; triplet = mined triplets). "
        "SSOT: training.loss",
    )
    # NOTE: --no-hard-positives REMOVED — hard positives are wired per the
    # owner ruling (volume-verified lane ON + loud when empty; see the
    # HARD POSITIVES block below). A flag for an always-off lane was a
    # silent no-op; the lane is now always-on by config, not by flag.
    # ── HPO lanes (second07 grid / second08 TPE — both with masking) ─────
    ap.add_argument(
        "--grid",
        action="store_true",
        help="second07-style fixed grid sweep (epochs x lr x warmup, "
        "11 configs, config_mean summary) — masking applied in every config",
    )
    ap.add_argument(
        "--hpo",
        action="store_true",
        help="second08-style optuna TPE sweep over HPO_SPACE — masking "
        "applied in every trial (resume-safe sqlite study)",
    )
    ap.add_argument("--quick", action="store_true", help="grid lane: 3-config smoke")
    ap.add_argument(
        "--n-trials",
        type=int,
        default=int(cfg["hpo"]["n_trials"]),  # SSOT hpo.n_trials
        help="tpe lane: trial budget",
    )
    ap.add_argument(
        "--n-jobs",
        type=int,
        default=int(cfg["hpo"]["n_jobs"]),  # SSOT hpo.n_jobs
        help="tpe lane: parallel optuna workers (1 = sequential; >1 "
        "multiplies GPU/CPU memory — beware)",
    )
    ap.add_argument(
        "--train-frac",
        type=float,
        default=1.0,
        help="07d data scaling: fraction of TRAIN pairs used "
        "(dev/test pools stay full — the curve measures "
        "train-data effect, not eval noise)",
    )
    ap.add_argument(
        "--rerank",
        type=str,
        default=None,
        help="07e two-stage: cross-encoder model id (e.g. "
        "cross-encoder/ms-marco-MiniLM-L-6-v2) re-scoring "
        "the confusion band after bi-encoder training",
    )
    ap.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="07f report + training-loss plots (default on; --no-plot to skip)",
    )
    ap.add_argument(
        "--mask-frac",
        type=float,
        default=None,
        help="masking augmentation (config/paths.yaml masking.*): "
        "fraction of positive anchors to mask. Default "
        "comes from the config — enabled:false -> 0 "
        "(off), enabled:true -> masking.frac. Explicit "
        "CLI value always wins.",
    )
    args = ap.parse_args()

    # Resolve both the bi-encoder and optional cross-encoder through the same
    # local-only registry contract. This prevents SentenceTransformers from
    # interpreting a registry key or Hub-looking string as a download target.
    args.model = resolve_model(args.model)
    if args.rerank:
        args.rerank = resolve_model(args.rerank)

    # An HPO winner must be replayable exactly. These are explicit only for
    # the selected final run; ordinary runs remain config-driven.
    if args.warmup_ratio is None:
        args.warmup_ratio = runtime("warmup_ratio")
    if args.weight_decay is None:
        args.weight_decay = runtime("weight_decay")

    # masking resolution: CLI flag > config; config default OFF.
    # NO FALLBACKS (owner Q27): masking.frac / masking.mask_prob are read
    # with hard indexing — a missing key crashes, a silent 0.15 default
    # that disagrees with the YAML (1.00) can never ship.
    if args.mask_frac is None:
        args.mask_frac = float(mask_cfg["frac"]) if mask_cfg["enabled"] else 0.0
    mask_hard_negatives = bool(mask_cfg["mask_hard_negatives"])
    mask_hard_negative_frac = (
        float(mask_cfg["hard_negative_frac"]) if mask_hard_negatives else 0.0
    )
    mask_prob = mask_cfg["mask_prob"]
    mask_prob = float(mask_prob) if mask_prob is not None else None

    import torch

    on_cuda = torch.cuda.is_available()
    print(f"device: {'cuda' if on_cuda else 'cpu'}", flush=True)

    lo, hi = (float(x) for x in args.band.split("-"))
    band = (lo, hi)

    df = load_dataset_deduped()
    if args.sample:
        df = df.head(args.sample).reset_index(drop=True)
        print(f"SAMPLE MODE: first {args.sample} rows", flush=True)
    # run_tag (owner ruling): every varying axis — model, payload variant,
    # train fraction, split, AND sample mode — is part of every artifact
    # name this run touches (fold metrics, pair dumps, checkpoints,
    # visibility logs). Constructed EARLY: the visibility dumps below write
    # into run-tagged dirs before any training happens.
    model_tag = args.model.rstrip("/").split("/")[-1]
    frac_tag = "full" if args.train_frac >= 1.0 else f"frac{args.train_frac:g}"
    sample_tag = f"sample{args.sample}" if args.sample else ""
    base_run_tag = (
        f"train-{args.split}-{model_tag}-{args.payload}-{frac_tag}-{sample_tag}"
    ).replace("/", "-").rstrip("-")
    # The launcher supplies one immutable ID for the entire remote run. Keep
    # it in every artifact/W&B namespace so two otherwise identical retries
    # cannot overwrite or become indistinguishable from one another.
    global_run_id = os.environ.get("EUROMONITOR_RUN_ID", "").strip()
    run_tag = global_run_id or base_run_tag
    # 07c field ablation — payload variants (only the TEXT the encoder sees
    # changes, so fold metrics are comparable):
    #   full       = clean sku (title + attributes, the owner's cleaning)
    #                — pairs (sku, own canonical)
    #   title_only = clean sku from title alone
    data = load_training_data(df, payload_variant=args.payload)
    payload, structured_features, row_bc, pos, neg = (
        data["payload"],
        np.asarray(data["structured_features"], dtype=np.float32),
        data["row_bc"],
        data["pos"],
        data["neg"],
    )
    s = data["stats"]
    _wandb.log_config(
        {
            "model": args.model,
            "split": args.split,
            "payload": args.payload,
            "loss": args.loss,
            "contrastive_margin": float(_SSOT_CONTRASTIVE_MARGIN),
            "architecture": runtime("architecture"),
            "mask_frac": args.mask_frac,
            "masking_enabled": bool(args.mask_frac > 0),
            "mask_hard_negatives": mask_hard_negatives,
            "mask_hard_negative_frac": mask_hard_negative_frac,
            "mask_prob": mask_prob,
            "mask_lo": float(mask_cfg["mask_lo"]),
            "mask_hi": float(mask_cfg["mask_hi"]),
            "mask_track_visibility": bool(mask_cfg["track_visibility"]),
            "mask_track_per_epoch": bool(mask_cfg["track_per_epoch"]),
            "structured_features_enabled": bool(
                tr["structured_features"]["enabled"]
            ),
            "structured_features_append_to_text": bool(
                tr["structured_features"]["append_to_text"]
            ),
            "structured_features_feed_to_loss": bool(
                tr["structured_features"]["feed_to_loss"]
            ),
            "structured_features_embedding_weight": float(
                tr["structured_features"]["embedding_weight"]
            ),
            "structured_features_volume_scale_ml": float(
                tr["structured_features"]["volume_scale_ml"]
            ),
            "structured_features_pack_scale": float(
                tr["structured_features"]["pack_scale"]
            ),
            "structured_features_max_set_size": int(
                tr["structured_features"]["max_set_size"]
            ),
            "train_frac": args.train_frac,
            "batch_size_cuda": tr["batch_size_cuda"],
            "track_datapoint_usage": bool(tr["track_datapoint_usage"]),
            "sample": args.sample or "full",
            "n_source_rows": s["n_rows"],
            "n_canonicals": s["n_canonicals"],
            "n_gate_hard_negatives": s["n_neg_gate_rows"],
            "ann_mining_enabled": ann_mining_enabled,
            "ann_target": int(ann_cfg["target"]),
            "ann_k": int(ann_cfg["k"]),
            "ann_refresh_enabled": bool(ann_cfg["refresh_enabled"]),
            "ann_refresh_every_epochs": int(ann_cfg["refresh_every_epochs"]),
            "ann_band_mode": str(ann_cfg["band_mode"]),
            "ann_candidate_multiplier": int(ann_cfg["candidate_multiplier"]),
            "ann_score_quantiles": str(ann_cfg["score_quantiles"]),
            "ann_max_per_canonical": int(ann_cfg["max_per_canonical"]),
            "ann_max_per_brand": int(ann_cfg["max_per_brand"]),
            "attribute_conflict_enabled": attribute_conflict_enabled,
            "attribute_conflict_target": int(attr_cfg["target"]),
            "mining_profile": mining_profile or "config_default",
        }
    )
    print(
        f"[pairs] clean-sku: {s['n_sku_with_canonical']:,} rows with canonical "
        f"/ {s['n_rows']:,} rows | canonicals: {s['n_canonicals']:,}",
        flush=True,
    )
    print(
        f"[negatives] {s['n_neg_gate_rows']:,} hard-no gate rows -> "
        f"{s['n_neg_resolved']:,} (sku, other-canonical) pairs "
        f"({s['n_neg_dropped']:,} endpoints unresolved)",
        flush=True,
    )
    print(
        f"[negatives] excluded {s['n_neg_same_canonical_dropped']:,} "
        "hard-no rows whose two GTINs share the same canonical item",
        flush=True,
    )
    if args.payload != "full":
        print(f"[07c] payload variant: {args.payload}", flush=True)
    country = df["country"].fillna("").astype(str).to_numpy()
    if len(pos) == 0:
        raise SystemExit(
            "no positives resolved — check canonical_records.csv GTINs "
            "vs dataset_deduped.csv barcodes"
        )

    # ── masking augmentation (src/training/masking.py, config-driven) ──
    # Owner's masking augmentation, corrected for MNRL semantics: the original
    # the original masking script fed label=0.0 hard-negative pairs to MNRL — MNRL
    # IGNORES labels and would train them as POSITIVES (different products
    # pulled together). Here masking augments POSITIVES only: for a fraction
    # of pairs, the ANCHOR text gets config-band variable token masking
    # (U(mask_lo, mask_hi) per masked copy — config/training.yaml masking band;
    # AUDIT round 2 F08: this comment still said "15% random token
    # masking") and the masked pair is appended as an EXTRA positive (same
    # pair semantics, noised anchor). When configured, hard negatives receive
    # the same label-preserving augmentation: the masked anchor remains paired
    # with its different-product target and stays label 0. Masked texts are
    # NEW payload entries (row_bc = same barcode), so folds/components are
    # unaffected.
    mask_audit: list[dict] = []
    hard_negative_mask_audit: list[dict] = []
    # Keep evaluation negatives immutable. Masked hard-negative copies are
    # training-only rows so dev/holdout metrics cannot include augmentation.
    train_neg = neg
    # Keep provenance aligned with every negative row before any fold split.
    # The baseline resolved gate population is immutable; supplemental
    # miners append their own source label rather than collapsing into a
    # generic ``hard_negative`` bucket.
    neg_sources = np.full(len(neg), "gate", dtype=object)
    train_neg_sources = neg_sources.copy()
    if args.mask_frac > 0:
        from training.masking import augment_positives

        pos, payload, row_bc, n_added, mask_audit = augment_positives(
            pos, payload, row_bc, frac=args.mask_frac, mask_prob=mask_prob, seed=SEED
        )
        if n_added:
            structured_features = np.vstack(
                [
                    structured_features,
                    np.asarray(
                        [structured_features[int(row["anchor_payload_idx"])] for row in mask_audit],
                        dtype=np.float32,
                    ),
                ]
            )
        if len(structured_features) != len(payload):
            raise RuntimeError(
                "structured feature/payload length mismatch after masking: "
                f"{len(structured_features)} != {len(payload)}"
            )
        # MASK VISIBILITY (owner directive 2026-09-07): per-copy realized
        # extents — the high-vs-low-extent effect on overfitting is
        # measurable only when each copy's TRUE masked fraction is logged
        # next to its texts. Rewritten every run — run-tagged dir + latest
        # pointer (sample runs never move the shared pointer).
        import pandas as _pd

        from core.common import write_visibility_log as _wvl

        _ma = _pd.DataFrame(mask_audit)
        if bool(mask_cfg["track_visibility"]):
            _wvl(_ma, "mask_visibility.csv", run_tag, bool(args.sample))
        if hard_negative_mask_audit:
            _wvl(
                _pd.DataFrame(hard_negative_mask_audit),
                "mask_hard_negative_visibility.csv",
                run_tag,
                bool(args.sample),
            )
        # high/low halves of the extent distribution — the split point is
        # the MIDPOINT of the config extent band (masking.mask_lo..
        # mask_hi), derived here so a band change can never leave the
        # buckets misaligned with the distribution (was inline 0.10).
        _mid = (float(mask_cfg["mask_lo"]) + float(mask_cfg["mask_hi"])) / 2.0
        if len(_ma):
            _hi = _ma[_ma.realized_extent >= _mid]
            _lo = _ma[_ma.realized_extent < _mid]
            _extent_desc = (
                f"variable {float(mask_cfg['mask_lo']):.0%}-"
                f"{float(mask_cfg['mask_hi']):.0%}"
                if mask_prob is None
                else mask_prob
            )
            print(
                f"[masking] +{n_added:,} masked-anchor positives "
                f"(frac={args.mask_frac:.0%}, extent={_extent_desc}) | "
                f"high-extent(>={_mid:.2f}): {len(_hi):,} copies, mean {_hi.realized_extent.mean():.3f} | "
                f"low-extent(<{_mid:.2f}): {len(_lo):,} copies, mean {_lo.realized_extent.mean() if len(_lo) else 0:.3f}",
                flush=True,
            )
            if mask_hard_negatives:
                print(
                    f"[masking] hard negatives use dynamic per-presentation "
                    f"masking (label=0, frac={mask_hard_negative_frac:.0%})",
                    flush=True,
                )

    # ── COMPONENT-AWARE SPLITS ──────────────────────────────────────────────
    # UNEXPECTED-BEHAVIOR FIX: pipeline positives connect TWO DIFFERENT
    # barcodes, so barcode-level folds straddle pairs (one endpoint per
    # side) and pairs_in_set silently drops them — measured 7,489
    # positives → ~1,500 straddling per fold boundary, test sets at ~130
    # pairs. The split unit is the CONNECTED COMPONENT of the positive-pair
    # graph; the dev boundary must ALSO be component-aligned (a barcode-level
    # rng carve splits 7,808 of 37,445 train-side pair-uses between train/dev).
    # holdout = the owner's 50/25/25: quarters q0+q1 train, q2 dev, q3 test.
    if args.split == "holdout":
        quarters = component_folds(
            pos,
            row_bc,
            int(cfg["split"]["holdout_component_folds"]),
            SEED,
        )
        test_bc, dev_bc = quarters[3], quarters[2]
        folds_override = test_bc  # single-set: train = all others
        dev_override = dev_bc
        from core.hard_negatives import pairs_in_set

        n_tr = len({b for b in row_bc if b and b not in test_bc and b not in dev_bc})
        tr_pos = len(
            pos[pairs_in_set(pos, row_bc, set(row_bc.tolist()) - test_bc - dev_bc)]
        )
        print(
            f"[holdout 50/25/25] train≈{n_tr:,} / dev {len(dev_bc):,} / "
            f"test {len(test_bc):,} barcodes | train_pos {tr_pos:,} (component-aware)",
            flush=True,
        )
    else:
        quarters = None
        folds_override = component_folds(pos, row_bc, args.folds, SEED)
        dev_override = None
        print(f"[cv] {args.folds} component folds", flush=True)

    # ── HARD POSITIVES  ─────────────────────────
    # The manifest is regenerated from the frozen deduplicated dataset before
    # the existing volume verifier consumes it, so this lane cannot silently
    # train with zero volume-verified cross-country positives.
    from training.build_second04_pairs import write_manifest
    from core.volume_verified import volume_verified_cross_country

    hard_positives_enabled = bool(training_cfg().training.hard_positives)
    if hard_positives_enabled:
        write_manifest(Path(F["second04_pairs_positive"]))
        hp_pairs = volume_verified_cross_country(df)
    else:
        hp_pairs = np.empty((0, 2), dtype=int)
        print("[hard-positives] disabled by training.hard_positives", flush=True)
    if len(hp_pairs):
        print(
            f"[hard-positives] volume-verified cross-country: {len(hp_pairs):,} "
            f"pairs joining the positive pool",
            flush=True,
        )
    # hard negatives = the gate hard-no pairs (neg) — passed as neg_pairs
    # below; the contrastive loss trains label=0 rows on them directly and
    # MNRL uses them as explicit in-batch negatives.

    # ANN is deliberately deferred until a fine-tuned checkpoint exists.
    # There is no zero-shot embedding fallback here: the first training phase
    # uses the frozen gate population and masking, then the refresh callback
    # mines from the live fine-tuned model after checkpoint saves.
    emb0 = np.empty((0, 0), dtype=np.float32)
    if mining_enabled:
        print(
            "[ANN] deferred until a fine-tuned checkpoint; zero-shot embeddings disabled",
            flush=True,
        )
    else:
        print("[mining] disabled by config; skipping ANN and supplemental mining", flush=True)

    # Supplemental mining targets the same-brand/category attribute-conflict
    # population that the gate's 6,051 hard negatives cannot exhaust. The
    # original gate negatives remain intact; these are additional label-0
    # training rows selected from the configured cosine band.
    if attribute_conflict_enabled and emb0.size:
        from core.hard_negatives import mine_attribute_conflict_negatives

        _attr_lo, _attr_hi = (float(x) for x in attr_cfg["band"].split("-"))
        _attr_neg, _attr_scores = mine_attribute_conflict_negatives(
            df,
            payload,
            row_bc,
            emb0,
            existing=neg,
            n_target=int(attr_cfg["target"]),
            cosine_lo=_attr_lo,
            cosine_hi=_attr_hi,
        )
    elif attribute_conflict_enabled:
        print(
            "[attribute-conflicts] deferred until a fine-tuned checkpoint; "
            "no zero-shot mining embeddings",
            flush=True,
        )
        _attr_neg = np.empty((0, 2), dtype=int)
    else:
        _attr_neg = np.empty((0, 2), dtype=int)
    if len(_attr_neg):
        neg = np.vstack([neg, _attr_neg]) if len(neg) else _attr_neg
        # Keep supplemental negatives unexpanded; dynamic masking happens at
        # each training presentation, so every epoch can see a fresh variant.
        train_neg = np.vstack([train_neg, _attr_neg])
        _attr_sources = np.full(len(_attr_neg), "attribute_conflict", dtype=object)
        neg_sources = np.concatenate([neg_sources, _attr_sources])
        train_neg_sources = np.concatenate([train_neg_sources, _attr_sources])
    if len(neg_sources) != len(neg) or len(train_neg_sources) != len(train_neg):
        raise RuntimeError(
            "negative source provenance length mismatch before fold split: "
            f"eval={len(neg_sources)}/{len(neg)} "
            f"train={len(train_neg_sources)}/{len(train_neg)}"
        )
    _wandb.log_config(
        {
            "n_gate_negative_pairs": int(np.sum(neg_sources == "gate")),
            "n_attribute_conflict_negative_pairs": int(
                np.sum(neg_sources == "attribute_conflict")
            ),
        }
    )
    _wandb.log_config(
        {
            "n_attribute_conflict_negatives": int(len(_attr_neg)),
            "n_hard_negative_training_pairs": int(len(train_neg)),
            "n_hard_negative_eval_pairs": int(len(neg)),
        }
    )
    print(
        f"[attribute-conflicts] +{len(_attr_neg):,} supplemental label-0 pairs "
        f"(baseline gate hard-negatives preserved; total {len(neg):,})",
        flush=True,
    )

    # masked anchors extend the row-index space beyond df; every df-indexed
    # side array must cover them (row_bc/payload/emb0 already do; country
    # doesn't — pad with the anchor's own country)
    # AUDIT FIX 2026-09-08 (agent finding 7): padding "" for EVERYTHING past
    # df made cross_mask ≈ 'sku side has a country' — 85% of positives
    # flagged cross-country and auc_cross ≈ auc. The canonical block gets
    # the REAL country of its GTIN's df rows (mode over the rows that map
    # to it); masked copies inherit the ANCHOR's country via their barcode.
    if len(country) < len(payload):
        import collections as _coll

        _n_df = len(country)
        _bc_arr = row_bc if isinstance(row_bc, np.ndarray) else np.array(row_bc)
        _by_bc = _coll.defaultdict(_coll.Counter)
        for _i in range(_n_df):
            _g = str(_bc_arr[_i])
            if _g:
                _by_bc[_g][country[_i]] += 1
        _gtin_country = {
            _g: _cnt.most_common(1)[0][0] for _g, _cnt in _by_bc.items()
        }
        _pad = [_gtin_country.get(str(_g), "") for _g in _bc_arr[_n_df:]]
        _resolved = sum(1 for c in _pad if c)
        country = np.concatenate(
            [country, np.array(_pad, dtype=country.dtype)]
        )
        print(
            f"[country-pad] {_resolved:,}/{len(_pad):,} extended-payload "
            f"entries resolved to a real country "
            f"(canonicals by their GTIN's rows; masked copies by barcode)",
            flush=True,
        )

    data = (df, payload, structured_features, row_bc, country, pos, hp_pairs, emb0)

    # ── HPO lanes: second07 fixed grid / second08 optuna TPE ───────────────
    if args.grid or args.hpo:
        from training.hpo import run_grid, run_tpe

        # TEST-LEAK WIRING (2026-09-12): both sweep lanes ride the SAME
        # component split the main lane just built — holdout mode passes
        # the test quarter (folds_override) + dev quarter (dev_override)
        # so the sweep trains q0+q1 and selects on q2; cv mode passes the
        # fold list. Never let a sweep rebuild its own folds over ALL
        # barcodes: that put the dev/test quarters into the sweep's train
        # side — the leak the previous fix closed (asserts in hpo.py).
        # MASKING: the entry already augmented (pre-encode, emb0-aligned);
        # the sweep lanes inherit it — len(mask_audit) is the applied
        # count they print for provenance (they must NOT re-augment:
        # that extended payload past emb0 and died at DataTuple).
        # NEGATIVES: the gate hard-no pairs ride the same neg channel the
        # main lane uses (below) — the contrastive SSOT loss needs them.
        if args.grid:
            run_grid(
                args, data, mask_cfg, folds_override, dev_override,
                n_masked=len(mask_audit), neg_pairs=neg,
                train_neg_pairs=train_neg,
                neg_pair_sources=neg_sources,
                train_neg_pair_sources=train_neg_sources,
                dynamic_mask_hard_negatives=mask_hard_negatives,
                dynamic_mask_frac=mask_hard_negative_frac,
                dynamic_mask_prob=mask_prob,
                mask_audit=mask_audit,
                hard_negative_mask_audit=hard_negative_mask_audit,
            )
        else:
            run_tpe(
                args, data, mask_cfg, folds_override, dev_override,
                n_masked=len(mask_audit), neg_pairs=neg,
                train_neg_pairs=train_neg,
                neg_pair_sources=neg_sources,
                train_neg_pair_sources=train_neg_sources,
                dynamic_mask_hard_negatives=mask_hard_negatives,
                dynamic_mask_frac=mask_hard_negative_frac,
                dynamic_mask_prob=mask_prob,
                mask_audit=mask_audit,
                hard_negative_mask_audit=hard_negative_mask_audit,
                wandb_ctx=_wandb,
                mlf_ctx=_mlf,
            )
        return

    # src/training/training's DEFAULT_CFG key set (train_one_config reads these)

    # NO FALLBACK (owner Q27): every optimizer knob comes from the SSOT
    # training: block via runtime() — the inline 0.05/0.01/linear/1.0
    # literals duplicated config/training.yaml and could silently diverge.
    cfg = {
        "architecture": runtime("architecture"),
        "epochs": args.epochs,
        "lr": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "lr_scheduler": runtime("lr_scheduler"),
        "max_grad_norm": runtime("max_grad_norm"),
        "patience": ES_PATIENCE,
        "es_threshold": ES_THRESHOLD,
        "uniformity_weight": (
            float(_UNIFORMITY_CFG["weight"])
            if bool(_UNIFORMITY_CFG["enabled"])
            else 0.0
        ),
    }
    t0 = time.perf_counter()
    # run_tag carries EVERY varying axis (owner ruling): the run_all
    # ablation series ran 12 variants into the SAME train_fold_metrics.csv
    # and r{tag}_f{fold} checkpoint dirs — each run silently overwrote the
    # last. Model, payload variant, and train fraction are now part of the
    # tag so artifacts collide-proof across the whole series.
    # SAMPLE COLLISION FIX (owner audit 2026-09-07): --sample runs share
    # every tag axis with the real run (same split/model/payload/frac) and
    # SILENTLY OVERWROTE the full run's fold-metrics + pair-dump CSVs —
    # measured: 3h full-frac-0.25 result destroyed by a 1k chain check.
    # Sample runs write to their own suffixed artifacts; never to the
    # shared names. (run_tag itself is built right after df loads.)
    rows = train_one_config(
        cfg,
        loss=args.loss,
        model_id=args.model,
        # hard positives ride in the data tuple (position 5, set above);
        # hard negatives ride in neg_pairs — ONE channel each, no shadow copies
        use_hp=hard_positives_enabled and len(hp_pairs) > 0,
        band=band,
        data=data,
        seed=SEED,
        on_cuda=on_cuda,
        cv_folds=None,
        folds_override=folds_override,
        dev_fraction=args.dev_fraction,
        dev_override=dev_override,
        neg_pairs=neg,
        train_neg_pairs=train_neg,
        neg_pair_sources=neg_sources,
        train_neg_pair_sources=train_neg_sources,
        dynamic_mask_hard_negatives=mask_hard_negatives,
        dynamic_mask_frac=mask_hard_negative_frac,
        dynamic_mask_prob=mask_prob,
        mask_audit=mask_audit,
        hard_negative_mask_audit=hard_negative_mask_audit,
        ann_refresh_enabled=ann_mining_enabled,
        attribute_conflict_refresh_enabled=attribute_conflict_enabled,
        train_frac=args.train_frac if args.train_frac < 1.0 else None,
        run_tag=run_tag,
        sample=bool(args.sample),
        resume=args.resume,
        wandb_ctx=_wandb,
    )
    elapsed = time.perf_counter() - t0

    # ── MASK-EXTENT EFFECT (owner directive 2026-09-07) ─────────────────────
    # Does masking actually fight overfitting, and does the HIGH-extent
    # half behave differently from the LOW half? Score every masked copy
    # against its positive target under the trained model, bucket by
    # realized extent (>= midpoint high / < midpoint low — midpoint of the
    # config extent band masking.mask_lo..mask_hi), and dump per-copy sims
    # to results/logs/mask_effect.csv.
    # Overfit signature = low-extent copies score HIGH (memorized surface
    # forms) while high-extent copies score much lower (the model leaned
    # on tokens that got masked).
    mask_effect_metrics: dict[str, float] = {}
    try:
        _ok = [r for r in rows if r.get("status") == "ok"]
        if (
            not _REMOTE_TRAINING
            and args.mask_effect
            and getattr(args, "mask_frac", 0) > 0
            and mask_audit
            and _ok
        ):
            from sentence_transformers import SentenceTransformer as _ST

            _best = artifact(
                "checkpoint_repo",
                {
                    "model_tag": model_tag,
                    "run_tag": run_tag,
                    "fold": int(_ok[0]["fold"]),
                    "step": 0,
                },
            ).parent
            # pick the BEST checkpoint (dev-AP), not the highest-numbered
            # one: with save_total_limit=2 the dir holds [best, last] and
            # the last step is NOT the shipped model when early stopping
            # fired (verified on the real run: best=checkpoint-20,
            # highest=checkpoint-21). trainer_state.json records the best
            # checkpoint path; when the recorded best was pruned (limit=2)
            # fall back to the highest surviving checkpoint.
            _ckpts = sorted(
                _best.glob("checkpoint-*"),
                key=lambda p: int(p.name.split("-")[1]),
            )
            _src = _ckpts[-1] if _ckpts else _best
            for _c in _ckpts:
                _ts = _c / "trainer_state.json"
                if not _ts.exists():
                    continue
                try:
                    _bm = json.loads(_ts.read_text()).get("best_model_checkpoint")
                except (ValueError, OSError):
                    continue
                if not _bm:
                    continue
                _cand = _best / Path(_bm).name
                if _cand.exists():
                    _src = _cand
                    break
            print(f"[mask-effect] scoring model from {_src}", flush=True)
            _model = _ST(str(_src), device="cpu")
            _texts = [m["masked_text"] for m in mask_audit]
            _unmasked_texts = [m["anchor_text"] for m in mask_audit]
            _targets = [payload[m["pair_payload_idx"]] for m in mask_audit]
            _em = _model.encode(
                _texts + _unmasked_texts + _targets,
                batch_size=runtime("batch_size_embed"),  # SSOT
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            _n = len(_texts)
            _masked_sims = np.einsum("ij,ij->i", _em[:_n], _em[2 * _n:])
            _unmasked_sims = np.einsum(
                "ij,ij->i", _em[_n : 2 * _n], _em[2 * _n:]
            )
            _eff = pd.DataFrame(
                {
                    "realized_extent": [m["realized_extent"] for m in mask_audit],
                    "sim_to_target": _masked_sims.round(4),
                    "masked_score": _masked_sims.round(4),
                    "unmasked_score": _unmasked_sims.round(4),
                    "masked_minus_unmasked": (
                        _masked_sims - _unmasked_sims
                    ).round(4),
                    "barcode": [m["barcode"] for m in mask_audit],
                    "anchor_text": _unmasked_texts,
                    "masked_text": _texts,
                }
            )
            _mid_eff = (
                float(mask_cfg["mask_lo"]) + float(mask_cfg["mask_hi"])
            ) / 2.0
            _eff["bucket"] = np.where(
                _eff.realized_extent >= _mid_eff,
                f"high(>={_mid_eff:.2f})",
                f"low(<{_mid_eff:.2f})",
            )
            from core.common import write_visibility_log as _wvl

            _wvl(_eff, "mask_effect.csv", run_tag, bool(args.sample))
            _g = _eff.groupby("bucket")[[
                "masked_score", "unmasked_score", "masked_minus_unmasked"
            ]].agg(["count", "mean"])
            mask_effect_metrics = {
                "mask_n": float(len(_eff)),
                "mask_masked_mean_cosine": float(_eff["masked_score"].mean()),
                "mask_masked_median_cosine": float(_eff["masked_score"].median()),
                "mask_unmasked_mean_cosine": float(_eff["unmasked_score"].mean()),
                "mask_unmasked_median_cosine": float(_eff["unmasked_score"].median()),
                "mask_mean_cosine_delta": float(_eff["masked_minus_unmasked"].mean()),
                "mask_median_cosine_delta": float(_eff["masked_minus_unmasked"].median()),
            }
            if _wandb is not None:
                _wandb.log_metrics(
                    {
                        f"masking/{key}": value
                        for key, value in mask_effect_metrics.items()
                    }
                )
                _wandb.set_summary(
                    {
                        f"masking/{key}": value
                        for key, value in mask_effect_metrics.items()
                    }
                )
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            _mask_plot = RESULTS / f"mask_effect_{run_tag}.png"
            _fig, _ax = plt.subplots(figsize=(7, 4.5))
            _ax.hist(
                _eff["unmasked_score"], bins=20, alpha=0.55, label="non-masked"
            )
            _ax.hist(
                _eff["masked_score"], bins=20, alpha=0.55, label="masked"
            )
            _ax.set(
                xlabel="cosine similarity to target",
                ylabel="count",
                title="Masked vs non-masked positive performance",
            )
            _ax.grid(alpha=0.25)
            _ax.legend()
            _fig.tight_layout()
            _fig.savefig(_mask_plot, dpi=plot_dpi())
            plt.close(_fig)
            if _wandb is not None:
                _wandb.log_image(_mask_plot, f"masking/{run_tag}/score_distributions")
            print(
                "[mask-effect] trained-model masked vs non-masked cosine "
                "by extent bucket:",
                flush=True,
            )
            for _b, _r in _g.iterrows():
                print(
                    f"    {_b:12s} n={int(_r[('masked_score', 'count')]):>6,} "
                    f"masked={_r[('masked_score', 'mean')]:.4f} "
                    f"non_masked={_r[('unmasked_score', 'mean')]:.4f} "
                    f"delta={_r[('masked_minus_unmasked', 'mean')]:+.4f}",
                    flush=True,
                )
            print(
                f"    (dumped {len(_eff):,} rows -> results/logs/mask_effect.csv)",
                flush=True,
            )
    except Exception:
        import traceback as _tb

        print(
            "[mask-effect] FAILED (non-fatal — training results stand):\n"
            + _tb.format_exc(),
            flush=True,
        )

    if mask_effect_metrics:
        for _row in rows:
            if _row.get("status") == "ok":
                _row.update(mask_effect_metrics)
    for _row in rows:
        if _row.get("status") == "ok":
            _row["n_masked_pos"] = len(mask_audit)
            _row["n_masked_hard_negatives"] = int(
                _row.get("n_masked_hard_negatives", len(hard_negative_mask_audit))
            )

    # persist metrics — SUFFIXED per run (see run_tag note): the fixed
    # F["fold_metrics"] name meant the run_all step-4 series left only
    # the LAST variant's fold metrics on disk. Per-run copy keeps every
    # variant; the shared name stays for the "latest run" consumer
    # (report_plots greps train_*_fold_metrics.csv).
    # SAMPLE COLLISION FIX (audit round 2): this per-run name lacked the
    # sample axis — a --sample run still overwrote the full run's suffixed
    # metrics CSV (the pointer at F["fold_metrics"] was already protected;
    # this was the last unprotected surface of the collision class).
    RESULTS.mkdir(parents=True, exist_ok=True)
    _metrics_tag = (
        f"train_{model_tag}_{args.split}_{args.payload}_{frac_tag}"
        + (f"_sample{args.sample}" if args.sample else "")
    )
    out = RESULTS / f"{_metrics_tag}_fold_metrics.csv"
    all_rows = []
    for r in rows:
        row = dict(r)
        row["model"] = args.model
        row["payload"] = args.payload
        row["train_frac"] = args.train_frac
        all_rows.append(row)
    pd.DataFrame(all_rows).to_csv(out, index=False)
    # latest-run pointer for consumers that want one canonical name —
    # SAMPLE runs must not move it (same overwrite class as the collision
    # above: a 1k chain check replaced the full run's latest-metrics CSV)
    if not args.sample:
        pd.DataFrame(all_rows).to_csv(F["fold_metrics"], index=False)
        # results pointer (owner Q21): ONE json always naming the most
        # recent full-run artifacts — consumers never glob for "latest"
        _ok_rows = [r for r in all_rows if r.get("status") == "ok"]
        pointer = {
            "run_tag": run_tag,
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": args.model,
            "split": args.split,
            "payload": args.payload,
            "loss": args.loss,
            "train_frac": args.train_frac,
            "mask_frac": args.mask_frac,
            "fold_metrics_csv": out.name,
            "visibility_log_dir": f"logs/{run_tag}",
            "n_folds_ok": len(_ok_rows),
            "mean_auc": (
                float(np.mean([r["auc"] for r in _ok_rows]))
                if _ok_rows
                else None
            ),
            "mean_pr_auc": (
                float(np.mean([r["pr_auc"] for r in _ok_rows]))
                if _ok_rows
                else None
            ),
            "mean_calibration_rand_index": (
                float(np.mean([r["calibration_rand_index"] for r in _ok_rows]))
                if _ok_rows
                and all("calibration_rand_index" in r for r in _ok_rows)
                else None
            ),
            "mean_calibration_adjusted_rand": (
                float(np.mean([r["calibration_adjusted_rand"] for r in _ok_rows]))
                if _ok_rows
                and all("calibration_adjusted_rand" in r for r in _ok_rows)
                else None
            ),
        }
        F["results_pointer"].write_text(
            json.dumps(pointer, indent=2, sort_keys=True)
        )
        print(
            f"[results-pointer] {F['results_pointer']} -> run {run_tag}",
            flush=True,
        )
    else:
        print(
            "[fold-metrics] sample run — latest-run pointer NOT updated",
            flush=True,
        )

    # ── 07-series CSV emission (owner ruling: emit from src/training/train) ────────
    # report_plots.py reads 07b/07c/07d; their monorepo producers were never
    # ported. The 07 lanes ARE this script's flag-mirrors (--payload = 07c,
    # --train-frac = 07d), so the CSVs regenerate from these runs:
    #   07c_field_ablation.csv — one row per payload variant (APPEND mode:
    #     run_all sweeps full + title_only into ONE csv)
    #   07d_data_scaling.csv  — one row per train fraction (append, same)
    #   07b_four_pop_scores.csv — four-population cosines (needs re-scoring;
    #     emitted by the rerank lane, see _emit_four_pop)
    # SMOKE GUARD: --sample runs are chain validation, not results — their
    # rows must not sit in the ablation CSVs until a real run replaces them.
    ok_rows_07 = [r for r in all_rows if r.get("status") == "ok"]
    if ok_rows_07 and not args.sample and not _REMOTE_TRAINING:
        _emit_07_series(ok_rows_07, args)
    elif ok_rows_07 and args.sample:
        print(
            "[07] sample mode — 07c/07d emission SKIPPED (smoke runs don't "
            "count as ablation data)",
            flush=True,
        )

    ok_rows = [r for r in rows if r.get("status") != "skipped"]
    aucs = [r.get("auc") for r in ok_rows if r.get("auc") is not None]

    # Build the complete post-run report before publishing the downloadable
    # W&B/DVC bundle. HPO selection rows intentionally have no test pair dump,
    # so they skip this report until the final holdout-scoring lane.
    if args.plot and aucs and not _REMOTE_TRAINING:
        pair_paths = sorted(RESULTS.glob(f"train_{model_tag}_{run_tag}_fold*_pairs.csv"))
        if pair_paths:
            try:
                from training.generate_training_report import generate_report

                report_dir = RESULTS / f"report_{run_tag}"
                generate_report(
                    out,
                    pair_paths,
                    report_dir,
                    sorted(RESULTS.glob(f"train_{model_tag}_{run_tag}_fold*_train_scores.csv")),
                    sorted(RESULTS.glob(f"train_{model_tag}_{run_tag}_fold*_random_easy_scores.csv")),
                )
                print(f"[report] complete report -> {report_dir}", flush=True)
            except Exception:
                report_dir.mkdir(parents=True, exist_ok=True)
                report_trace = traceback.format_exc()
                (report_dir / "report_error.txt").write_text(
                    report_trace, encoding="utf-8"
                )
                _wandb.log_config(
                    {
                        "report_status": "failed",
                        "report_error_file": str(report_dir / "report_error.txt"),
                    }
                )
                print(
                    "[report] FAILED (full traceback saved as report_error.txt):\n"
                    + report_trace,
                    flush=True,
                )

    # ── mlflow: params + per-fold nested runs (local backend by default) ──
    _mlf.log_params(
        {
            "model": args.model,
            "split": args.split,
            "payload": args.payload,
            "epochs": args.epochs,
            "lr": args.lr,
            "loss": args.loss,
            "mask_frac": args.mask_frac,
            "band": str(band),
            "sample": args.sample or "full",
        }
    )
    from training.hpo_metrics import numeric_calibration_metrics

    for r in ok_rows:
        with _mlf.nested:
            _mlf.log_params({"fold": r.get("fold")})
            fold_metrics = {
                k: r[k]
                for k in (
                    "auc",
                    "auc_cross",
                    "acc_at_thr",
                    "youden_thr",
                    "pr_auc",
                    "hits_at_1",
                    "best_dev_ap",
                    "final_train_loss",
                    "s_per_step",
                    # fixed-threshold metric names are config-derived
                    # (f"f1_at_{thr:g}") — match by prefix
                    *[
                        k
                        for k in r
                        if k.startswith(
                            ("f1_at_", "precision_at_", "recall_at_")
                        )
                    ],
                )
                if r.get(k) is not None
            }
            fold_metrics.update(numeric_calibration_metrics(r))
            _mlf.log_metrics(fold_metrics)
            calibration_metrics = numeric_calibration_metrics(r)
            if r.get("traceback"):
                continue
            _wandb.log_metrics(
                {
                    **{
                        f"fold_{r.get('fold')}_{k}": r[k]
                        for k in (
                            "auc", "auc_cross", "acc_at_thr", "pr_auc", "hits_at_1",
                            "best_dev_ap", "final_train_loss", "n_random_easy_neg",
                            "random_easy_available", "n_masked_pos",
                            "n_masked_hard_negatives",
                            *[key for key in r if key.startswith(("f1_at_", "precision_at_", "recall_at_"))],
                        )
                        if r.get(k) is not None
                    },
                    **{
                        f"fold_{r.get('fold')}_{key}": value
                        for key, value in calibration_metrics.items()
                    },
                }
            )
    if aucs:
        _mlf.log_metrics(
            {"mean_auc": float(np.mean(aucs)), "std_auc": float(np.std(aucs))}
        )
    calibration_rows = [
        r
        for r in ok_rows
        if np.isfinite(r.get("calibration_rand_index", float("nan")))
    ]
    if calibration_rows:
        _mlf.log_metrics(
            {
                "mean_calibration_rand_index": float(
                    np.mean([r["calibration_rand_index"] for r in calibration_rows])
                ),
                "mean_calibration_adjusted_rand": float(
                    np.mean(
                        [r["calibration_adjusted_rand"] for r in calibration_rows]
                    )
                ),
            }
        )
    if not _REMOTE_TRAINING:
        _mlf.log_artifact(out)
        _wandb.log_artifact(out, "fold-metrics")
        _log_run_artifacts_to_wandb(
            _wandb,
            run_tag=run_tag,
            model_tag=model_tag,
            metrics_path=out,
            rows=all_rows,
        )
    _wandb.set_summary(
        {
            "mean_auc": float(np.mean(aucs)) if aucs else None,
            "mean_calibration_rand_index": (
                float(np.mean([r["calibration_rand_index"] for r in calibration_rows]))
                if calibration_rows
                else None
            ),
            "mean_calibration_adjusted_rand": (
                float(
                    np.mean(
                        [r["calibration_adjusted_rand"] for r in calibration_rows]
                    )
                )
                if calibration_rows
                else None
            ),
            "n_folds": len(ok_rows),
            "mask_copies": len(mask_audit),
            "fold_metrics_csv": out.name,
        }
    )
    if aucs:
        print(
            f"\n[done] {len(ok_rows)} folds in {elapsed / 60:.1f} min | "
            f"test AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}",
            flush=True,
        )
    else:
        print("\n[done] no folds produced metrics", flush=True)
    print(f"[artifacts] {out}", flush=True)
    # pair stats for the run log
    print(json.dumps(s, indent=2), flush=True)
    if args.plot and aucs and not _REMOTE_TRAINING:
        from training.plots import report_plots, training_loss_plot

        report_plots(out, args)
        training_loss_plot(out, args)
        for png in RESULTS.glob("train_*.png"):
            _mlf.log_artifact(png)
        for png in RESULTS.glob("training_loss_*.png"):
            _mlf.log_artifact(png)

    if args.rerank:
        from training.rerank import rerank_stage

        rerank_stage(
            args,
            rows,
            run_tag,
            row_bc,
            payload,
            df,
            dev_override,
            folds_override,
            SEED,
            payload_variant=args.payload,
        )


if __name__ == "__main__":
    main()
