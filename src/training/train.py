"""train.py — the TRAIN_GPU training entry point (the ONE callable in
TRAIN_GPU root; the training internals live in src/training/).

Trains on the OFFICIAL clean-sku scheme (DATA_PIPE.pairs):
  positive : (clean_sku_text, canonical_gtin)
  negative : (clean_sku_text, canonical_other_gtin) — gate hard-no pairs
Run:  python train.py --model models/<name> [--split holdout|cv ...]
"""

from __future__ import annotations
import traceback
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
    collapse_guardrail_cfg,
    load_config,
    load_dataset_deduped,
    load_local_sentence_transformer,
    masking_cfg,
    plot_dpi,
    recall_column_suffix,
    runtime,
    set_determinism,
    training_cfg,
)
from core.common import SSOT_LOSS as _SSOT_LOSS
from core.common import SSOT_CONTRASTIVE_MARGIN as _SSOT_CONTRASTIVE_MARGIN
from core.run_log import RunLogger
from core.step_trace import send, timed, trace_step
from training.folds import component_folds, derive_holdout
from training.training import ES_PATIENCE, ES_THRESHOLD, train_one_config

_LOG = RunLogger(__name__)

# Colab workers should spend GPU time only on training and the score exports
# needed by the local post-processing lane.  Reports, plots, artifact bundles,
# and final DVC publishing are CPU/network work and are intentionally deferred
# until the worker outputs have been downloaded locally.
_REMOTE_TRAINING = os.environ.get("EUROMONITOR_REMOTE_TRAINING") == "1"


@timed
def _uniformity_cfg() -> dict:
    """Read at call time so refresh_training_config() is never shadowed."""
    return load_config()["training"]["uniformity_regularization"]

# thresholds live in config/paths.yaml (pairs.proceed_sim_threshold /
# pairs.hardneg_sim_threshold) and are read by pipeline.build_training_data
# directly — no duplicate constants here (they were dead globals: set but
# never read anywhere).


@timed
def load_training_data(df: pd.DataFrame, payload_variant: str = "full") -> dict:
    """OFFICIAL pair construction (DATA_PIPE.pairs): payload = clean sku
    text per row + canonical per GTIN; pos = (sku, own canonical), neg =
    (sku, other canonical) from gate hard-no pairs."""
    from training.base_data import load_base_data

    return load_base_data(df, payload_variant=payload_variant)


@timed
def _write_hard_negative_mask_trace(
    audit: list[dict],
    *,
    run_tag: str,
    sample: bool,
    enabled: bool,
) -> None:
    """Persist each dynamic negative-mask presentation with its extent."""
    if not enabled or not audit:
        return
    from core.common import write_visibility_log

    write_visibility_log(
        pd.DataFrame(audit),
        "mask_hard_negative_presentations.csv",
        run_tag,
        sample,
    )


@timed
def results_pointer(rows, *, run_tag, args, metrics_csv_name):
    """Run-level results pointer (owner Q21): names the latest full-run
    artifacts and states whether fold metrics are present or deferred.

    GPU-only runs (ER_GPU_TRAINING_ONLY=1) defer calibration/test
    reporting to the local CPU lane: their fold rows carry per-row
    deferred_local markers and no metric columns. Record the deferral
    at run level here instead of letting the marker live only in the
    per-fold rows, and never average over rows that lack the field."""
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    deferred_rows = [r for r in ok_rows if r.get("calibration_status") == "deferred_local"]
    return {
        "run_tag": run_tag,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "split": args.split,
        "payload": args.payload,
        "loss": args.loss,
        "train_frac": args.train_frac,
        "mask_frac": args.mask_frac,
        "fold_metrics_csv": metrics_csv_name,
        "visibility_log_dir": f"logs/{run_tag}",
        "n_folds_ok": len(ok_rows),
        "metrics_status": (
            None if not ok_rows
            else "deferred_local" if len(deferred_rows) == len(ok_rows)
            else "partial_deferred_local" if deferred_rows
            else "available"
        ),
        "n_folds_metrics_deferred": len(deferred_rows),
        "mean_auc": (
            float(np.mean([r["auc"] for r in ok_rows]))
            if ok_rows and all("auc" in r for r in ok_rows)
            else None
        ),
        "mean_pr_auc": (
            float(np.mean([r["pr_auc"] for r in ok_rows]))
            if ok_rows and all("pr_auc" in r for r in ok_rows)
            else None
        ),
        "mean_calibration_rand_index": (
            float(np.mean([r["calibration_rand_index"] for r in ok_rows]))
            if ok_rows
            and all("calibration_rand_index" in r for r in ok_rows)
            else None
        ),
        "mean_calibration_adjusted_rand": (
            float(np.mean([r["calibration_adjusted_rand"] for r in ok_rows]))
            if ok_rows
            and all("calibration_adjusted_rand" in r for r in ok_rows)
            else None
        ),
    }


@timed
def _emit_07_series(ok_rows: list[dict], args) -> None:
    """Write report aggregates using the entry point's configured dependencies."""
    from training.report_rows import emit_07_series

    emit_07_series(
        ok_rows, args, report_paths=F, seed=SEED, train_config=training_cfg,
        recall_suffix=recall_column_suffix, append_csv=_append_csv,
    )


# Preserve the entry module's helper import surface for existing callers.
from training.report_rows import (
    _append_csv,
    _nonnumeric_calibration_fields,
    _trace_nonnumeric_calibration_fields,
)


@timed
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

    for row in _LOG.progress(rows, desc="wandb_fold_artifacts", unit="fold",
                             total=len(rows)):
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


@timed
def main() -> None:
    """Train with W&B telemetry and local run artifacts."""
    RunLogger.configure_console()
    from core.wandb_ctx import WandbCtx

    run_name = "hpo" if "--hpo" in sys.argv else "train_gpu"
    with WandbCtx(run_name) as _wandb:
        _main_inner(_wandb)


@timed
def _main_inner(_wandb) -> None:
    """One run, threaded phase by phase (the former _main_inner body)."""
    _TrainerDriver(_wandb).run()


_THREADED = [
    "cfg", "tr", "mining_cfg", "mining_profile", "ann_cfg", "attr_cfg", "cross_cfg", "ann_mining_enabled", "attribute_conflict_enabled", "cross_brand_enabled", "mining_enabled", "mask_cfg", "collapse_cfg", "split_cfg", "args", "dev_share", "test_share", "mask_hard_negatives", "mask_hard_negative_frac", "mask_prob", "hard_negative_mask_prob", "hard_negative_mask_lo", "hard_negative_mask_hi", "swap_value_frac", "hard_negative_swap_value_frac", "swap_max_donor_overlap", "counterfactual_frac", "swap_max_field_share", "swap_max_value_share", "field_quota_shares", "attribute_augment", "cross_retailer_donors", "dropout_frac", "_dropout_cfg", "on_cuda", "band", "df", "data", "timing", "payload", "structured_features", "row_bc", "pos", "neg", "run_tag", "model_tag", "frac_tag", "payload_retailers", "entity_keys", "targeted_attribute_neg", "cross_brand_neg", "s", "uniformity", "country", "neg_sources", "mask_audit", "hard_negative_mask_audit", "train_neg", "train_neg_sources", "balanced_policy", "balanced_coverage", "frozen_holdout", "donor_scope", "mint_rejections", "folds_override", "dev_override", "hard_positives_enabled", "hp_pairs", "emb0", "rows", "elapsed",
]


class _TrainerDriver:
    """The training entry's phase pipeline.

    One thread of run state, one single-responsibility phase method per
    stage of the former _main_inner body: config scale, CLI args, masking
    knobs, dataset/tag prep, entity clusters, W&B config publication, the
    augmentation lanes, the split, the negative supply, country padding,
    the bundle/HPO/train dispatch, the mask-extent audit and results
    publication. Statements in each phase are the verbatim former body.
    """

    def __init__(self, _wandb) -> None:
        self._wandb = _wandb
        for attribute in _THREADED:
            setattr(self, attribute, None)

    def run(self) -> None:
        """Phase pipeline in a fixed order."""
        self.resolve_config()
        self.resolve_arguments()
        self.resolve_masking()
        self.load_run_data()
        self.build_entity_clusters()
        self.log_run_configs()
        self.augment()
        self.resolve_split()
        self.attach_supply()
        self.pad_country()
        self.assemble_run_data()
        if self.write_bundle():
            return
        if self.run_hpo_lanes():
            return
        self.train_folds()
        self.mask_effect_audit()
        self.publish_results()


    @timed
    def resolve_config(self) -> None:
        """Union of the scale: determinism, sections, mining lanes."""
        # determinism FIRST (2026-10-06): one call before any model/data
        # randomness — masking augmentation, zero-shot encode, fold carving
        # and the grid/tpe sweep lanes (dispatched below, same entry) all
        # ride this seed. The component split keeps passing SEED explicitly
        # (its own contract); the GLOBAL RNGs (random/numpy/torch/cudnn)
        # are pinned here, once, at entry.
        with trace_step('training._main_inner.config'):
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
                    "cross_brand": {
                        **mining_cfg["cross_brand"],
                        "enabled": bool(profile_cfg["cross_brand_enabled"]),
                    },
                }
            ann_cfg = mining_cfg["ann"]
            attr_cfg = mining_cfg["attribute_conflict"]
            cross_cfg = mining_cfg["cross_brand"]
            ann_mining_enabled = bool(ann_cfg["enabled"])
            attribute_conflict_enabled = bool(attr_cfg["enabled"])
            cross_brand_enabled = bool(cross_cfg["enabled"])
            mining_enabled = ann_mining_enabled or attribute_conflict_enabled
            mask_cfg = cfg["masking"]
            collapse_cfg = cfg["collapse_guardrail"]
            split_cfg = cfg["split"]
        self.cfg = cfg
        self.tr = tr
        self.mining_cfg = mining_cfg
        self.mining_profile = mining_profile
        self.ann_cfg = ann_cfg
        self.attr_cfg = attr_cfg
        self.cross_cfg = cross_cfg
        self.ann_mining_enabled = ann_mining_enabled
        self.attribute_conflict_enabled = attribute_conflict_enabled
        self.cross_brand_enabled = cross_brand_enabled
        self.mining_enabled = mining_enabled
        self.mask_cfg = mask_cfg
        self.collapse_cfg = collapse_cfg
        self.split_cfg = split_cfg

    @timed
    def resolve_arguments(self) -> None:
        """Parse CLI args with config-owned defaults."""
        cfg = self.cfg
        tr = self.tr
        ann_cfg = self.ann_cfg
        split_cfg = self.split_cfg
        mask_cfg = self.mask_cfg
        collapse_cfg = self.collapse_cfg
        # Trainer models resolve through the project-owned registry. Resolution is
        # local-only: a missing DVC bundle fails before data preparation begins.
        from core.common import resolve_model

        default_model = str(tr["base_model"])
        default_band = str(ann_cfg["band"])

        with trace_step('training._main_inner.args'):
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
            # 05-01: the holdout shares are CONFIG-owned, so the help text reports the
            # configured shares instead of restating literals. argparse %-formats help
            # strings — a bare percent must be doubled or --help raises.
            dev_share = float(split_cfg["dev_fraction"])
            test_share = float(split_cfg["test_fraction"])
            _share_pct = (
                f"{1.0 - dev_share - test_share:.0%}/{dev_share:.0%}/{test_share:.0%}"
            ).replace("%", "%%")
            ap.add_argument(
                "--split",
                choices=["holdout", "cv"],
                default=str(split_cfg["mode"]),
                help=f"holdout: ONE component-aware {_share_pct} "
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
                help="registry key or local path to the bundled model directory",
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
                "--dataset",
                type=Path,
                default=None,
                help="validated deduped source CSV override (used by prepared Colab bundles)",
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
            ap.add_argument(
                "--masking-profile",
                default=str(mask_cfg["profile"]),
                help="config/training.yaml masking_profiles entry",
            )
            ap.add_argument(
                "--collapse-guardrail-profile",
                default=str(collapse_cfg["profile"]),
                help="config/training.yaml collapse_guardrail_profiles entry",
            )
            ap.add_argument(
                "--prepare-bundle",
                type=Path,
                default=None,
                help="write the fully prepared local training bundle and exit",
            )
            args = ap.parse_args()

        mask_cfg = masking_cfg(args.masking_profile)
        cfg["collapse_guardrail"] = collapse_guardrail_cfg(
            args.collapse_guardrail_profile
        )

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
        self.args = args
        self.dev_share = dev_share
        self.test_share = test_share

    @timed
    def resolve_masking(self) -> None:
        """Resolve the masking profile knobs (hard indexing)."""
        args = self.args
        cfg = self.cfg
        mask_cfg = self.mask_cfg
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
        hard_negative_mask_prob = mask_cfg["hard_negative_mask_prob"]
        hard_negative_mask_prob = (
            float(hard_negative_mask_prob)
            if hard_negative_mask_prob is not None
            else None
        )
        hard_negative_mask_lo = float(mask_cfg["hard_negative_mask_lo"])
        hard_negative_mask_hi = float(mask_cfg["hard_negative_mask_hi"])
        # Static pre-training value swaps (donor transplant, symmetric for
        # positives / anchor-side for hard negatives). Same hard indexing.
        swap_value_frac = float(mask_cfg["swap_value_frac"])
        hard_negative_swap_value_frac = float(mask_cfg["hard_negative_swap_value_frac"])
        swap_max_donor_overlap = float(mask_cfg["swap_max_donor_overlap"])
        # Counterfactual twins: minimal agreed-field flips of positive anchors,
        # labeled 0 into the negative pool. Same hard indexing.
        counterfactual_frac = float(mask_cfg["counterfactual_frac"])
        # Anti-dominance caps for the swap lanes (soft field cap, hard value
        # cap). Same hard indexing.
        swap_max_field_share = float(mask_cfg["swap_max_field_share"])
        swap_max_value_share = float(mask_cfg["swap_max_value_share"])
        # Measured-variation wiring (duplicate variation census 2026-10-01):
        # per-field slot quotas for the swap lanes (measured same-GTIN conflict
        # rates), cross-retailer donor precedence, declaration-dropout lane.
        # The dict from masking_cfg() is validated through the MaskingSpec
        # contract first: a missing/untyped YAML block dies on schema
        # validation, never on a silent default.
        from core.schemas import MaskingSpec

        _mask_spec = MaskingSpec(**mask_cfg)
        field_quota_shares = dict(_mask_spec.field_quota_shares)
        attribute_augment = {
            k: v.model_dump() for k, v in _mask_spec.attribute_augment.items()
        }
        cross_retailer_donors = bool(_mask_spec.cross_retailer_donors)
        _dropout_cfg = _mask_spec.declaration_dropout
        dropout_frac = float(_dropout_cfg.frac) if _dropout_cfg is not None else 0.0
        self.mask_cfg = mask_cfg
        self.mask_hard_negatives = mask_hard_negatives
        self.mask_hard_negative_frac = mask_hard_negative_frac
        self.mask_prob = mask_prob
        self.hard_negative_mask_prob = hard_negative_mask_prob
        self.hard_negative_mask_lo = hard_negative_mask_lo
        self.hard_negative_mask_hi = hard_negative_mask_hi
        self.swap_value_frac = swap_value_frac
        self.hard_negative_swap_value_frac = hard_negative_swap_value_frac
        self.swap_max_donor_overlap = swap_max_donor_overlap
        self.counterfactual_frac = counterfactual_frac
        self.swap_max_field_share = swap_max_field_share
        self.swap_max_value_share = swap_max_value_share
        self.field_quota_shares = field_quota_shares
        self.attribute_augment = attribute_augment
        self.cross_retailer_donors = cross_retailer_donors
        self.dropout_frac = dropout_frac
        self._dropout_cfg = _dropout_cfg
        self.cfg = cfg

    @timed
    def load_run_data(self) -> None:
        """Load (or override/sample) the dataset and derive data pairs."""
        args = self.args
        import torch

        on_cuda = torch.cuda.is_available()
        print(f"device: {'cuda' if on_cuda else 'cpu'}", flush=True)

        lo, hi = (float(x) for x in args.band.split("-"))
        band = (lo, hi)

        with trace_step('training._main_inner.data_load'):
            from core.timing import Timing

            timing = Timing("train.data_path")

            if args.dataset is None:
                df = load_dataset_deduped()
            else:
                dataset_path = args.dataset.expanduser().resolve()
                if not dataset_path.is_file():
                    raise FileNotFoundError(f"training dataset override is missing: {dataset_path}")
                df = load_dataset_deduped(dataset_path)
                print(f"[dataset] override={dataset_path} rows={len(df):,}", flush=True)
            if args.sample:
                df = df.head(args.sample).reset_index(drop=True)
                print(f"SAMPLE MODE: first {args.sample} rows", flush=True)
            timing.mark("dataset_load")
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
            timing.mark("base_data")
        self.on_cuda = on_cuda
        self.band = band
        self.df = df
        self.data = data
        self.timing = timing
        self.payload = payload
        self.structured_features = structured_features
        self.row_bc = row_bc
        self.pos = pos
        self.neg = neg
        self.run_tag = run_tag
        self.model_tag = model_tag
        self.frac_tag = frac_tag

    @timed
    def build_entity_clusters(self) -> None:
        """Entity keys for donor-disjointness over the payload."""
        df = self.df
        payload = self.payload
        row_bc = self.row_bc
        pos = self.pos
        # Retailer per payload row, for the cross-retailer donor precedence the
        # duplicate census asked for (91.4% of real same-GTIN duplicates are
        # cross-retailer). Sku rows carry their import retailer; the canonical
        # tail carries "" (mixed-provenance slots). Only pool-range indices are
        # consulted by the donor loops, so the suffix needs no bookkeeping.
        _ret_sku = df["retailer"].fillna("").astype(str).to_numpy()
        _n_ret_sku = min(len(_ret_sku), len(payload))
        payload_retailers = np.array(
            [str(x) for x in _ret_sku[:_n_ret_sku]]
            + [""] * (len(payload) - _n_ret_sku),
            dtype=object,
        )
        # Entity keys for donor-disjointness (one per ORIGINAL payload row;
        # augmentation copies inherit via audit lineage, so this never grows).
        # Transitive-closure clusters over ground-truth positive pairs plus
        # shared gtins: multi-hop duplicates (retailer-feed variants, GTIN
        # length fragments) land in one CLUSTER_xxxxxx even when no single
        # gtin matches. Unmapped/isolated rows get unique per-row keys and
        # never block each other.
        import pandas as _pd

        from training.masking import (
            build_entity_cluster_map as _cluster_map,
        )
        from training.masking import normalize_entity_key as _entity_key

        _n_df_rows = len(df)
        _record_ids = [
            str(df["sku_id"].iloc[idx])
            if idx < _n_df_rows and "sku_id" in df.columns
            else str(row_bc[idx]).strip()
            for idx in range(len(payload))
        ]
        _pool_gtins = [
            _entity_key(str(row_bc[idx]).strip() if idx < len(row_bc) else "", "")
            for idx in range(len(payload))
        ]
        _donor_pool = _pd.DataFrame({
            "record_id": _record_ids,
            "gtin": _pool_gtins,
        })
        _truth_pairs = _pd.DataFrame({
            "anchor_id": [_record_ids[int(a)] for a, _b in np.asarray(pos, dtype=int)],
            "pair_id": [_record_ids[int(b)] for _a, b in np.asarray(pos, dtype=int)],
        })
        _clusters = _cluster_map(_truth_pairs, _donor_pool)
        from training.masking import check_cluster_sizes as _check_clusters

        _cluster_cfg = load_config()["entity_clusters"]
        _cluster_stats = _check_clusters(
            _clusters,
            max_component_size=int(_cluster_cfg["max_component_size"]),
            max_giant_ratio=float(_cluster_cfg["max_giant_ratio"]),
            population_size=len(_record_ids),
        )
        entity_keys = [
            str(_clusters.get(record, f"row:{idx}")) for idx, record in enumerate(_record_ids)
        ]
        _clustered = sum(1 for record in _record_ids if record in _clusters)
        print(
            f"[entity-clusters] {_clustered:,}/{len(_record_ids):,} payload rows "
            f"in {len(set(_clusters.values())):,} clusters "
            f"({_clustered / max(len(_record_ids), 1):.1%} covered; "
            f"max_size={_cluster_stats['max_size']}, "
            f"giant_ratio={_cluster_stats['giant_ratio']:.4f})",
            flush=True,
        )
        self.payload_retailers = payload_retailers
        self.entity_keys = entity_keys

    def log_run_configs(self) -> None:
        """Publish the run scale to W&B and print the pair census."""
        _wandb = self._wandb
        args = self.args
        data = self.data
        cfg = self.cfg
        mask_cfg = self.mask_cfg
        mask_hard_negatives = self.mask_hard_negatives
        mask_hard_negative_frac = self.mask_hard_negative_frac
        mask_prob = self.mask_prob
        hard_negative_mask_prob = self.hard_negative_mask_prob
        hard_negative_mask_lo = self.hard_negative_mask_lo
        hard_negative_mask_hi = self.hard_negative_mask_hi
        tr = self.tr
        ann_cfg = self.ann_cfg
        attr_cfg = self.attr_cfg
        mining_profile = self.mining_profile
        ann_mining_enabled = self.ann_mining_enabled
        attribute_conflict_enabled = self.attribute_conflict_enabled
        df = self.df
        pos = self.pos
        timing = self.timing
        targeted_attribute_neg = np.asarray(
            data.get("targeted_attribute_neg", np.empty((0, 2), dtype=int)),
            dtype=int,
        ).reshape(-1, 2)
        cross_brand_neg = np.asarray(
            data.get("cross_brand_neg", np.empty((0, 2), dtype=int)),
            dtype=int,
        ).reshape(-1, 2)
        s = data["stats"]
        uniformity = _uniformity_cfg()
        _wandb.log_config(
            {
                "model": args.model,
                "split": args.split,
                "payload": args.payload,
                "loss": args.loss,
                "contrastive_margin": float(_SSOT_CONTRASTIVE_MARGIN),
                "architecture": runtime("architecture"),
                "mask_frac": args.mask_frac,
                "masking_profile": str(mask_cfg["profile"]),
                "masking_enabled": bool(args.mask_frac > 0),
                "mask_hard_negatives": mask_hard_negatives,
                "mask_hard_negative_frac": mask_hard_negative_frac,
                "mask_frac_negatives": mask_hard_negative_frac,
                "mask_prob": mask_prob,
                "mask_lo": float(mask_cfg["mask_lo"]),
                "mask_hi": float(mask_cfg["mask_hi"]),
                "hard_negative_mask_prob": hard_negative_mask_prob,
                "hard_negative_mask_lo": hard_negative_mask_lo,
                "hard_negative_mask_hi": hard_negative_mask_hi,
                "mask_track_visibility": bool(mask_cfg["track_visibility"]),
                "mask_track_per_epoch": bool(mask_cfg["track_per_epoch"]),
                "uniformity_regularization_enabled": bool(
                    uniformity["enabled"]
                ),
                "uniformity_regularization_weight": float(
                    uniformity["weight"]
                ),
                "uniformity_temperature": float(uniformity["temperature"]),
                "late_epoch_lr_decay_enabled": bool(
                    runtime("late_epoch_lr_decay")["enabled"]
                ),
                "late_epoch_lr_decay_start_epoch_fraction": float(
                    runtime("late_epoch_lr_decay")["start_epoch_fraction"]
                ),
                "late_epoch_lr_decay_multiplier": float(
                    runtime("late_epoch_lr_decay")["multiplier"]
                ),
                "collapse_guardrail_profile": str(cfg["collapse_guardrail"]["profile"]),
                "collapse_operating_threshold": float(
                    cfg["collapse_guardrail"]["operating_threshold"]
                ),
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
                "vs dataset_deduped.csv gtins"
            )

        timing.mark("pair_census")
        self.targeted_attribute_neg = targeted_attribute_neg
        self.cross_brand_neg = cross_brand_neg
        self.s = s
        self.uniformity = uniformity
        self.country = country

    @timed
    def augment(self) -> None:
        """Append the static augmentation lanes (mask, swaps, twins)."""
        args = self.args
        split_cfg = self.split_cfg
        mask_cfg = self.mask_cfg
        mask_prob = self.mask_prob
        mask_hard_negatives = self.mask_hard_negatives
        mask_hard_negative_frac = self.mask_hard_negative_frac
        hard_negative_mask_prob = self.hard_negative_mask_prob
        hard_negative_mask_lo = self.hard_negative_mask_lo
        hard_negative_mask_hi = self.hard_negative_mask_hi
        swap_value_frac = self.swap_value_frac
        hard_negative_swap_value_frac = self.hard_negative_swap_value_frac
        swap_max_donor_overlap = self.swap_max_donor_overlap
        counterfactual_frac = self.counterfactual_frac
        swap_max_field_share = self.swap_max_field_share
        swap_max_value_share = self.swap_max_value_share
        field_quota_shares = self.field_quota_shares
        attribute_augment = self.attribute_augment
        entity_keys = self.entity_keys
        payload_retailers = self.payload_retailers
        run_tag = self.run_tag
        df = self.df
        pos = self.pos
        neg = self.neg
        payload = self.payload
        row_bc = self.row_bc
        structured_features = self.structured_features
        neg_sources = self.neg_sources
        timing = self.timing
        data = self.data
        attribute_conflict_enabled = self.attribute_conflict_enabled
        cross_brand_enabled = self.cross_brand_enabled
        targeted_attribute_neg = self.targeted_attribute_neg
        cross_brand_neg = self.cross_brand_neg
        dropout_frac = self.dropout_frac
        _dropout_cfg = self._dropout_cfg
        # ── masking augmentation (src/training/masking.py, config-driven) ──
        # Owner's masking augmentation, corrected for MNRL semantics: the original
        # the original masking script fed label=0.0 hard-negative pairs to MNRL — MNRL
        # IGNORES labels and would train them as POSITIVES (different products
        # pulled together). Here positives get TWO label-preserving views per
        # anchor — config-band random token masks AND agreed-surface swaps (the
        # anchor takes the counterpart's surface form where parsed values agree;
        # disagreed fields are never touched) — appended as EXTRA positives (same
        # pair semantics, noised anchor). Hard negatives get the same two views
        # and stay label 0: the masked/swapped anchor remains paired with its
        # different-product target. Masked/swapped texts are NEW payload entries
        # (row_bc = same gtin), so folds/components are unaffected.
        with trace_step('training._main_inner.augmentation'):
            mask_audit: list[dict] = []
            hard_negative_mask_audit: list[dict] = []
            # Augmented hard-negative copies join BOTH the eval (neg) and training
            # (train_neg) pools with an "<source>+aug" provenance label so the diet
            # gate (scripts/diet_manifest.py) sees the same augmented views the
            # trainer presents. Labels are untouched: every copy stays label 0.
            # The dynamic per-presentation path (training.py) is separate and unchanged.
            # Keep provenance aligned with every negative row before any fold split.
            # The baseline resolved gate population is immutable; supplemental
            # miners append their own source label rather than collapsing into a
            # generic ``hard_negative`` bucket.
            # Provenance per negative row. Lane mode ships its own populations
            # (base_negative / real_partner); the gate path keeps "gate".
            neg_sources = (
                np.array(data["neg_source"], dtype=object)
                if data.get("neg_source")
                else np.full(len(neg), "gate", dtype=object)
            )
            # Join all static real-negative populations BEFORE augmentation. Leaving
            # the targeted/cross-brand lanes until after masking made their rows
            # bypass every negative augmentation lane and diluted the actual MNRL
            # presentation diet. Miners already resolve original payload endpoints;
            # augmentation inherits their identity/fold lineage and source label.
            for enabled, candidates, source in (
                (attribute_conflict_enabled, targeted_attribute_neg, "targeted_attribute_conflict"),
                (cross_brand_enabled, cross_brand_neg, "cross_brand_conflict"),
            ):
                if enabled and len(candidates):
                    neg = np.vstack([neg, candidates]) if len(neg) else candidates.copy()
                    neg_sources = np.concatenate([
                        neg_sources, np.full(len(candidates), source, dtype=object)
                    ])
            train_neg = neg
            train_neg_sources = neg_sources.copy()
            # Freeze identity components before selecting any attribute donor. Copies
            # inherit existing entities and cannot redefine the parent split.
            frozen_holdout = None
            donor_scope = None
            canonical_scope = set(range(len(df), len(payload)))
            from collections import Counter as _MintCounter
            mint_rejections = _MintCounter()
            if args.split == "holdout":
                train_bc, dev_bc, test_bc = derive_holdout(pos, row_bc, split_cfg, seed=SEED)
                frozen_holdout = {role: sorted(values) for role, values in
                                  zip(("train", "dev", "test"), (train_bc, dev_bc, test_bc))}
                donor_scope = {i for i,value in enumerate(row_bc) if str(value) in train_bc}
            elif args.mask_frac > 0 and (swap_value_frac or hard_negative_swap_value_frac or counterfactual_frac):
                raise ValueError("static attribute transplants require a frozen holdout; prepare each CV fold separately")
            from core.schemas import BalancedAugmentationSpec
            balanced_policy = BalancedAugmentationSpec.model_validate(mask_cfg['balanced_augmentation'])
            if args.sample and balanced_policy.sample_counts is not None:
                balanced_policy = balanced_policy.model_copy(update={'counts':balanced_policy.sample_counts})
            cohort_tag = os.environ.get('ER_COHORT_TAG', '')
            sample_selected = args.sample and balanced_policy.sample_counts is not None
            if not sample_selected and cohort_tag and cohort_tag in balanced_policy.cohort_counts:
                balanced_policy = balanced_policy.model_copy(
                    update={'counts': balanced_policy.cohort_counts[cohort_tag]})
            if args.mask_frac > 0 and balanced_policy.enabled:
                from training.balanced_augmentation import augment_balanced
                from core.manifest import atomic_write_json
                before_neg = len(neg)
                pos, neg, payload, row_bc, structured_features, mask_audit, hard_negative_mask_audit, balanced_coverage = augment_balanced(
                    pos=pos, neg=neg, payload=payload, row_bc=row_bc, features=structured_features,
                    df=df, train_indices=donor_scope, canonical_indices=canonical_scope,
                    spec=balanced_policy, seed=SEED)
                added_sources = np.full(len(neg)-before_neg, 'counterfactual', dtype=object)
                neg_sources = np.concatenate([neg_sources, added_sources])
                train_neg, train_neg_sources = neg, neg_sources.copy()
                coverage_path = RESULTS / 'training' / run_tag / 'balanced_augmentation.json'
                coverage_path.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_json(balanced_coverage.model_dump(mode='json'), coverage_path)
                print(f'[balanced-augmentation] {balanced_coverage.model_dump(mode="json")} -> {coverage_path}', flush=True)
            if args.mask_frac > 0 and not balanced_policy.enabled:
                from training.masking import (
                    augment_hard_negatives,
                    augment_positives,
                    augment_value_swaps,
                )

                n_pre_mask_pos = len(pos)
                pos, payload, row_bc, n_added, mask_audit = augment_positives(
                    pos, payload, row_bc, frac=args.mask_frac, mask_prob=mask_prob, seed=SEED
                )
                # One shared donor-value counter across all three swap lanes, passed
                # in fixed call order (pos values, neg values, twins): the footprint
                # cap binds the whole bundle, not one lane. The cap budget is the
                # three lanes' combined picks, so shares mean the same everywhere.
                # (len(neg) here equals the later n_pre_mask_neg: pos augmentation
                # never touches the neg array.) Deterministic.
                from collections import Counter as _Counter

                _shared_value_counts: _Counter = _Counter()
                _swap_pick_total = (
                    int(n_pre_mask_pos * min(swap_value_frac, 1.0))
                    + int(len(neg) * min(hard_negative_swap_value_frac, 1.0))
                    + int(n_pre_mask_pos * min(counterfactual_frac, 1.0))
                )
                # Static value swaps (coconut -> lime) sample the same ORIGINAL
                # prefix. Positives are rewritten on BOTH sides from an agreeing
                # donor pair, so a match stays a match; each such audit appends TWO
                # payload rows (anchor copy + counterpart copy).
                pos, payload, row_bc, n_value_added, value_audit = augment_value_swaps(
                    pos,
                    payload,
                    row_bc,
                    frac=swap_value_frac,
                    seed=SEED + 3,
                    population="positive",
                    pool_size=n_pre_mask_pos,
                    symmetric=True,
                    entity_keys=entity_keys,
                    allowed_payload_indices=donor_scope,
                    coherent_prose=True,
                    rejection_counts=mint_rejections,
                    max_field_share=swap_max_field_share,
                    max_value_share=swap_max_value_share,
                    shared_value_counts=_shared_value_counts,
                    cap_base=_swap_pick_total,
                    max_donor_overlap=swap_max_donor_overlap,
                    field_quota_shares=field_quota_shares or None,
                    row_retailer=payload_retailers,
                    attribute_augment=attribute_augment or None,
                )
                mask_audit.extend(value_audit)
                from training.masking import extend_augmented_features

                structured_features = extend_augmented_features(
                    structured_features, payload, mask_audit
                )
                # ── declaration-dropout lane (duplicate census 2026-10-01) ──
                # Attribute cells differ in 100% of same-GTIN groups: every retailer
                # declares a different partial key subset. Mint that shape: copy the
                # anchor, remove 1..3 random declared structured groups, pair with
                # the unchanged positive. Label-safe (removing evidence cannot
                # contradict identity); this is also the lane that reaches the
                # registry-only weak spots no token group serves.
                n_dropout_added = 0
                if dropout_frac > 0:
                    from training.masking import augment_declaration_dropout as _aug_drop

                    _pos_pre_drop = len(pos)
                    pos, payload, row_bc, n_dropout_added, _drop_audit = _aug_drop(
                        pos,
                        payload,
                        row_bc,
                        frac=dropout_frac,
                        seed=SEED + 6,
                        pool_size=n_pre_mask_pos,
                        min_drop=int(_dropout_cfg.min_drop),
                        max_drop=int(_dropout_cfg.max_drop),
                        max_value_share=swap_max_value_share,
                    )
                    if n_dropout_added:
                        mask_audit.extend(_drop_audit)
                        structured_features = extend_augmented_features(
                            structured_features, payload, _drop_audit
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
                # high/low halves of the extent distribution — the split point is
                # the MIDPOINT of the config extent band (masking.mask_lo..
                # mask_hi), derived here so a band change can never leave the
                # buckets misaligned with the distribution (was inline 0.10).
                # swap_values copies carry realized_extent measuring the replaced
                # fraction — a different quantity — so that mode alone is excluded
                # from the extent halves; the full audit (masked + swapped) is
                # what the visibility CSV keeps.
                _mid = (float(mask_cfg["mask_lo"]) + float(mask_cfg["mask_hi"])) / 2.0
                _ma_masked = _ma[~_ma.target_mode.isin(["swap_values"])] if len(_ma) else _ma
                if len(_ma_masked):
                    _hi = _ma_masked[_ma_masked.realized_extent >= _mid]
                    _lo = _ma_masked[_ma_masked.realized_extent < _mid]
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
                if n_value_added:
                    print(
                        f"[masking] +{n_value_added:,} swap-values positives "
                        f"(frac={swap_value_frac:.0%}, symmetric donor transplant)",
                        flush=True,
                    )

                # ── negative augmentation (bundle path; labels stay 0) ──
                # Random-band masks (config hard-negative extent band) plus
                # agreed-surface swaps plus anchor-side value swaps, all sampled
                # from the ORIGINAL negative prefix only (pool_size = pre-
                # augmentation neg count), so nothing compounds on masked copies.
                # New rows join BOTH neg and train_neg with an "<source>+aug"
                # provenance label; the payload / structured tail grows with the
                # anchor rows. Labels and the loss mapping are untouched: every
                # copy stays label 0.
                n_pre_mask_neg = len(neg)
                _neg_pair_source = {
                    (int(a), int(b)): str(s)
                    for (a, b), s in zip(neg.tolist(), neg_sources.tolist())
                }
                neg, payload, row_bc, n_neg_added, _neg_mask_audit = augment_hard_negatives(
                    neg,
                    payload,
                    row_bc,
                    frac=mask_hard_negative_frac,
                    seed=SEED + 1,
                )
                neg, payload, row_bc, n_neg_value_added, _neg_value_audit = augment_value_swaps(
                    neg,
                    payload,
                    row_bc,
                    frac=hard_negative_swap_value_frac,
                    seed=SEED + 4,
                    population="hard_negative",
                    pool_size=n_pre_mask_neg,
                    symmetric=False,
                    entity_keys=entity_keys,
                    allowed_payload_indices=donor_scope,
                    coherent_prose=True,
                    rejection_counts=mint_rejections,
                    max_field_share=swap_max_field_share,
                    max_value_share=swap_max_value_share,
                    shared_value_counts=_shared_value_counts,
                    cap_base=_swap_pick_total,
                    max_donor_overlap=swap_max_donor_overlap,
                    field_quota_shares=field_quota_shares or None,
                    row_retailer=payload_retailers,
                    attribute_augment=attribute_augment or None,
                )
                _neg_new_audit = _neg_mask_audit + _neg_value_audit
                if _neg_new_audit:
                    train_neg = neg
                    _aug_sources = np.array(
                        [
                            _neg_pair_source[
                                (int(row["anchor_payload_idx"]), int(row["pair_payload_idx"]))
                            ]
                            + "+aug"
                            for row in _neg_new_audit
                        ],
                        dtype=object,
                    )
                    neg_sources = np.concatenate([neg_sources, _aug_sources])
                    train_neg_sources = np.concatenate([train_neg_sources, _aug_sources])
                    structured_features = extend_augmented_features(
                        structured_features, payload, _neg_new_audit
                    )
                hard_negative_mask_audit.extend(_neg_new_audit)
                # ── TIER 1(a): counterpart positives for anchor-side value swaps ──
                # An anchor-only transplant invalidates the source positive, so the
                # triple builder omitted every swap copy and it never trained. Replay
                # the same donor transplant onto the source's own positive so the pair
                # (copy, counterpart) is genuinely positive. Feature lineage for the new
                # counterpart rows is derived from the source positive row, so the
                # already-extended swap copy is not re-claimed.
                from training.masking import mint_swap_counterpart_positives as _mint_cfp

                _cfp_pairs, _cfp_payload, _cfp_bc, _cfp_audit = _mint_cfp(
                    _neg_value_audit, payload, row_bc, pos
                )
                if _cfp_pairs:
                    payload.extend(_cfp_payload)
                    row_bc = np.concatenate(
                        [row_bc, np.array(_cfp_bc, dtype=row_bc.dtype)]
                    )
                    pos = (
                        np.vstack([pos, np.array(_cfp_pairs, dtype=pos.dtype)])
                        if len(pos)
                        else np.array(_cfp_pairs, dtype=pos.dtype)
                    )
                    structured_features = extend_augmented_features(
                        structured_features, payload, _cfp_audit
                    )
                    hard_negative_mask_audit.extend(_cfp_audit)
                    print(
                        f"[masking] +{len(_cfp_pairs):,} swap counterpart positives "
                        f"(anchor-side transplant replayed onto the source positive)",
                        flush=True,
                    )
                # ── counterfactual twins (minimal-flip negatives from positives) ──
                # Sampled from the SAME original positive prefix (pool_size =
                # n_pre_mask_pos), so twins never compound on masked/swapped copies.
                # Each twin breaks exactly one previously-agreed field and is labeled
                # 0 by construction; twins join BOTH neg and train_neg with a
                # "counterfactual" provenance label (registered in the datapoint
                # population spec, so coverage stays exact).
                from training.masking import augment_counterfactual_twins as _aug_cf

                _cf_pos_len = len(pos)
                _cf_full, payload, row_bc, n_cf_added, _cf_audit = _aug_cf(
                    pos,
                    payload,
                    row_bc,
                    frac=counterfactual_frac,
                    seed=SEED + 5,
                    pool_size=n_pre_mask_pos,
                    entity_keys=entity_keys,
                    allowed_payload_indices=donor_scope,
                    coherent_prose=True,
                    canonical_indices=canonical_scope,
                    rejection_counts=mint_rejections,
                    max_field_share=swap_max_field_share,
                    max_value_share=swap_max_value_share,
                    shared_value_counts=_shared_value_counts,
                    cap_base=_swap_pick_total,
                    max_donor_overlap=swap_max_donor_overlap,
                    field_quota_shares=field_quota_shares or None,
                    row_retailer=payload_retailers,
                    attribute_augment=attribute_augment or None,
                )
                _cf_new = np.asarray(_cf_full, dtype=int)[_cf_pos_len:]
                if n_cf_added:
                    if len(_cf_new) != n_cf_added:
                        raise RuntimeError(
                            "counterfactual twin row accounting did not close: "
                            f"{len(_cf_new)} != {n_cf_added}"
                        )
                    neg = np.vstack([neg, _cf_new])
                    train_neg = np.vstack([train_neg, _cf_new])
                    _cf_sources = np.full(n_cf_added, "counterfactual", dtype=object)
                    neg_sources = np.concatenate([neg_sources, _cf_sources])
                    train_neg_sources = np.concatenate([train_neg_sources, _cf_sources])
                    structured_features = extend_augmented_features(
                        structured_features, payload, _cf_audit
                    )
                    hard_negative_mask_audit.extend(_cf_audit)
                    print(
                        f"[masking] +{n_cf_added:,} counterfactual twins "
                        f"(frac={counterfactual_frac:.0%}, agreed-field flip, label=0)",
                        flush=True,
                    )
                if len(structured_features) != len(payload):
                    raise RuntimeError(
                        "structured feature/payload length mismatch after negative "
                        f"augmentation: {len(structured_features)} != {len(payload)}"
                    )
                if len(neg_sources) != len(neg) or len(train_neg_sources) != len(train_neg):
                    raise RuntimeError(
                        "negative source provenance length mismatch after augmentation: "
                        f"eval={len(neg_sources)}/{len(neg)} "
                        f"train={len(train_neg_sources)}/{len(train_neg)}"
                    )
                if n_neg_added + n_neg_value_added:
                    print(
                        f"[masking] +{n_neg_added:,} masked hard negatives "
                        f"(frac={mask_hard_negative_frac:.0%}) "
                        f"+{n_neg_value_added:,} swap-values hard negatives "
                        f"(frac={hard_negative_swap_value_frac:.0%}, label=0)",
                        flush=True,
                    )

            timing.mark("augmentation")
        self.pos = pos
        self.neg = neg
        self.payload = payload
        self.row_bc = row_bc
        self.structured_features = structured_features
        self.mask_audit = mask_audit
        self.hard_negative_mask_audit = hard_negative_mask_audit
        self.neg_sources = neg_sources
        self.train_neg = train_neg
        self.train_neg_sources = train_neg_sources
        self.balanced_policy = balanced_policy
        self.balanced_coverage = balanced_coverage
        self.frozen_holdout = frozen_holdout
        self.donor_scope = donor_scope
        self.mint_rejections = mint_rejections
        self.timing = timing

    def resolve_split(self) -> None:
        """Resolve the component-aware split (holdout or cv)."""
        args = self.args
        frozen_holdout = self.frozen_holdout
        pos = self.pos
        row_bc = self.row_bc
        dev_share = self.dev_share
        test_share = self.test_share
        # ── COMPONENT-AWARE SPLITS ──────────────────────────────────────────────
        # UNEXPECTED-BEHAVIOR FIX: pipeline positives connect TWO DIFFERENT
        # gtins, so gtin-level folds straddle pairs (one endpoint per
        # side) and pairs_in_set silently drops them — measured 7,489
        # positives → ~1,500 straddling per fold boundary, test sets at ~130
        # pairs. The split unit is the CONNECTED COMPONENT of the positive-pair
        # graph; the dev boundary must ALSO be component-aligned (a gtin-level
        # rng carve splits 7,808 of 37,445 train-side pair-uses between train/dev).
        # holdout = ONE component-aware split, derived by folds.derive_holdout:
        # test = the LAST component group, dev = the one before it, train = the
        # rest — no hardcoded quarter indices (05-01/05-02). The helper raises
        # unless 1/n_folds equals the configured dev/test_fraction, so a knob that
        # cannot be honoured fails loudly instead of quietly building a 60/20/20
        # split under the banner of another share; the banner below prints the
        # CONFIGURED shares.
        if args.split == "holdout":
            train_bc, dev_bc, test_bc = (set(frozen_holdout[role]) for role in ("train", "dev", "test"))
            folds_override = test_bc  # single-set: train = all others
            dev_override = dev_bc
            from core.hard_negatives import pairs_in_set

            n_tr = len(train_bc)
            tr_pos = len(pos[pairs_in_set(pos, row_bc, train_bc)])
            print(
                f"[holdout {1.0 - dev_share - test_share:.0%}/{dev_share:.0%}/"
                f"{test_share:.0%}] train≈{n_tr:,} / dev {len(dev_bc):,} / "
                f"test {len(test_bc):,} gtins | train_pos {tr_pos:,} (component-aware)",
                flush=True,
            )
        else:
            folds_override = component_folds(pos, row_bc, args.folds, SEED)
            dev_override = None
            print(f"[cv] {args.folds} component folds", flush=True)
        self.folds_override = folds_override
        self.dev_override = dev_override

    @timed
    def attach_supply(self) -> None:
        """Hard positives, supplemental miners, minted lane, balance."""
        _wandb = self._wandb
        args = self.args
        df = self.df
        data = self.data
        pos = self.pos
        neg = self.neg
        neg_sources = self.neg_sources
        train_neg = self.train_neg
        train_neg_sources = self.train_neg_sources
        mask_cfg = self.mask_cfg
        balanced_policy = self.balanced_policy
        run_tag = self.run_tag
        mint_rejections = self.mint_rejections
        hard_negative_mask_audit = self.hard_negative_mask_audit
        attr_cfg = self.attr_cfg
        targeted_attribute_neg = self.targeted_attribute_neg
        cross_brand_neg = self.cross_brand_neg
        attribute_conflict_enabled = self.attribute_conflict_enabled
        cross_brand_enabled = self.cross_brand_enabled
        mining_enabled = self.mining_enabled
        payload = self.payload
        row_bc = self.row_bc
        timing = self.timing
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
        if not mining_enabled:
            print("[mining] disabled by config; skipping ANN and supplemental mining", flush=True)

        # Supplemental dynamic mining uses the same dimension contract after a
        # fine-tuned embedding population exists.
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
                # The conflict verdict must use the training gate's own volume
                # tolerance, or this miner can emit a pair the gate already
                # labelled compatible as a hard negative (see the SSOT note in
                # core.hard_negatives).
                volume_relative_tolerance=float(
                    load_config()["gate"]["vol_tolerance"]
                ),
                # Both cuts: relative-only rejects small-volume pairs the gate
                # accepts, which would mine a true match as a hard negative.
                volume_absolute_tolerance_ml=float(
                    load_config()["gate"]["vol_abs_tolerance"]
                ),
            )
        elif attribute_conflict_enabled:
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
        # Lane mode: minted partners are TRAINING-ONLY (owner ruling 2026-10-03).
        # They append to train_neg (never to the eval `neg`), so evaluation stays
        # real-pairs-only and the augmentation stages above — which read `neg` —
        # never re-edit a record that was already minted.
        neg_minted = np.asarray(
            data.get("neg_minted", np.empty((0, 2), dtype=int)), dtype=int
        )
        if len(neg_minted):
            train_neg = (
                np.vstack([train_neg, neg_minted])
                if len(train_neg) else neg_minted.copy()
            )
            train_neg_sources = np.concatenate(
                [train_neg_sources, np.full(len(neg_minted), "minted", dtype=object)]
            )
        if len(neg_sources) != len(neg) or len(train_neg_sources) != len(train_neg):
            raise RuntimeError(
                "negative source provenance length mismatch before fold split: "
                f"eval={len(neg_sources)}/{len(neg)} "
                f"train={len(train_neg_sources)}/{len(train_neg)}"
            )
        balance_train_classes = bool(load_config()["pairs"]["balance_train_classes"])
        n_class_balance_shortfall = 0
        n_class_balance_discarded_base = 0
        n_class_balance_discarded_aug = 0
        if balance_train_classes and not balanced_policy.enabled:
            target = len(pos)
            if target and not len(train_neg):
                raise RuntimeError(
                    "class balancing requested but no training negatives are available"
                )
            if target:
                from training.training import select_balanced_negatives

                # WITHOUT replacement (defect fix, measured at live scale): this
                # sampler used to draw `replace=len(train_neg) < target`, padding a
                # short negative pool by DUPLICATING rows — 7,952 of 11,686 train
                # negatives on the pre-change population and 6,564 after the
                # cross-brand lane joined — while every downstream counter (the
                # per-fold coverage audit's distinct_pairs_presented, the source
                # census) reported those rows as distinct pairs. A duplicated row
                # is not new data: it silently re-weights one pair. The pool is now
                # kept WHOLE when it is smaller than the positive class, and the
                # shortfall is reported as its own number instead.
                # Base-first retention (2026-09-28): augmented copies are trimmed
                # before real base pairs, so the 1.000 ratio never costs organic
                # data. Discards are counted by kind, not hidden.
                neg_copy_anchors = {
                    int(row["copy_payload_idx"])
                    for row in hard_negative_mask_audit
                    if row.get("copy_payload_idx") is not None
                }
                train_neg, train_neg_sources, n_disc_base, n_disc_aug = (
                    select_balanced_negatives(
                        train_neg, train_neg_sources, neg_copy_anchors,
                        target, SEED + 91_003,
                    )
                )
                n_class_balance_discarded_base = int(n_disc_base)
                n_class_balance_discarded_aug = int(n_disc_aug)
                n_class_balance_shortfall = max(0, target - len(train_neg))
            print(
                f"[class-balance] training positives={len(pos):,} "
                f"negatives={len(train_neg):,} ratio="
                f"{len(train_neg) / max(len(pos), 1):.3f} "
                f"(without replacement; unavailable rows={n_class_balance_shortfall:,}, "
                f"discarded_base={n_class_balance_discarded_base:,}, "
                f"discarded_aug={n_class_balance_discarded_aug:,}, "
                "duplicated rows=0)",
                flush=True,
            )
        _wandb.log_config(
            {
                "n_gate_negative_pairs": int(np.sum(neg_sources == "gate")),
                "n_attribute_conflict_negative_pairs": int(
                    np.sum(
                        np.isin(
                            neg_sources,
                            ["attribute_conflict", "targeted_attribute_conflict"],
                        )
                    )
                ),
                "n_targeted_attribute_negative_pairs": int(
                    np.sum(neg_sources == "targeted_attribute_conflict")
                ),
                "n_cross_brand_negative_pairs": int(
                    np.sum(neg_sources == "cross_brand_conflict")
                ),
            }
        )
        _wandb.log_config(
            {
                "n_attribute_conflict_negatives": int(len(_attr_neg)),
                "n_cross_brand_negatives": int(len(cross_brand_neg)),
                "n_hard_negative_training_pairs": int(len(train_neg)),
                "n_hard_negative_eval_pairs": int(len(neg)),
                "balance_train_classes": balance_train_classes,
                "n_class_balance_shortfall": int(n_class_balance_shortfall),
                "n_class_balance_duplicated_rows": 0,
                "n_class_balance_discarded_base": int(n_class_balance_discarded_base),
                "n_class_balance_discarded_aug": int(n_class_balance_discarded_aug),
                "n_training_positive_pairs": int(len(pos)),
                "n_training_negative_pairs": int(len(train_neg)),
            }
        )
        print(
            f"[attribute-conflicts] +{len(targeted_attribute_neg):,} targeted static, "
            f"+{len(_attr_neg):,} supplemental dynamic label-0 pairs "
            f"(baseline gate hard-negatives preserved; total {len(neg):,})",
            flush=True,
        )
        print(
            f"[cross-brand] +{len(cross_brand_neg):,} cross-brand static label-0 pairs "
            f"(enabled={cross_brand_enabled}; "
            f"{np.sum(neg_sources == 'cross_brand_conflict'):,} carry the "
            "cross_brand_conflict source in this fold's negative pool)",
            flush=True,
        )
        print(f"[mint-quality] rejected edits: {dict(mint_rejections)}", flush=True)
        if args.mask_frac > 0 and bool(mask_cfg["track_visibility"]) and not balanced_policy.enabled:
            import pandas as _pd

            from core.common import write_visibility_log as _wvl

            _wvl(_pd.DataFrame([{"reason": key, "count": value} for key, value in mint_rejections.items()]),
                 "mint_rejections.csv", run_tag, bool(args.sample))
        timing.mark("splits_mining_balance")
        self.hard_positives_enabled = hard_positives_enabled
        self.hp_pairs = hp_pairs
        self.emb0 = emb0
        self.neg = neg
        self.neg_sources = neg_sources
        self.train_neg = train_neg
        self.train_neg_sources = train_neg_sources
        self.timing = timing
        self.mint_rejections = mint_rejections

    def pad_country(self) -> None:
        """Cover the extended payload rows with country evidence."""
        country = self.country
        payload = self.payload
        row_bc = self.row_bc
        timing = self.timing
        # masked anchors extend the row-index space beyond df; every df-indexed
        # side array must cover them (row_bc/payload/emb0 already do; country
        # doesn't — pad with the anchor's own country)
        # AUDIT FIX 2026-09-08 (agent finding 7): padding "" for EVERYTHING past
        # df made cross_mask ≈ 'sku side has a country' — 85% of positives
        # flagged cross-country and auc_cross ≈ auc. The canonical block gets
        # the REAL country of its GTIN's df rows (mode over the rows that map
        # to it); masked copies inherit the ANCHOR's country via their gtin.
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
                f"(canonicals by their GTIN's rows; masked copies by gtin)",
                flush=True,
            )

        timing.mark("country_pad")
        self.country = country
        self.timing = timing

    def assemble_run_data(self) -> None:
        """The data tuple the trainer/HPO lanes consume."""
        df = self.df
        payload = self.payload
        structured_features = self.structured_features
        row_bc = self.row_bc
        country = self.country
        pos = self.pos
        hp_pairs = self.hp_pairs
        emb0 = self.emb0
        data = (df, payload, structured_features, row_bc, country, pos, hp_pairs, emb0)
        self.data = data

    @timed
    def write_bundle(self) -> None:
        """Write the prepared local training bundle (short-circuits)."""
        args = self.args
        df = self.df
        payload = self.payload
        structured_features = self.structured_features
        row_bc = self.row_bc
        country = self.country
        pos = self.pos
        hp_pairs = self.hp_pairs
        emb0 = self.emb0
        neg = self.neg
        train_neg = self.train_neg
        neg_sources = self.neg_sources
        train_neg_sources = self.train_neg_sources
        mask_audit = self.mask_audit
        hard_negative_mask_audit = self.hard_negative_mask_audit
        frozen_holdout = self.frozen_holdout
        balanced_policy = self.balanced_policy
        balanced_coverage = self.balanced_coverage
        run_tag = self.run_tag
        mask_cfg = self.mask_cfg
        timing = self.timing
        if args.prepare_bundle is not None:
            from training.prepared_bundle import write_prepared_bundle

            manifest = write_prepared_bundle(
                args.prepare_bundle,
                df=df,
                payload=payload,
                structured_features=structured_features,
                row_bc=row_bc,
                country=country,
                pos=pos,
                hp_pairs=hp_pairs,
                emb0=emb0,
                neg=neg,
                train_neg=train_neg,
                neg_sources=neg_sources,
                train_neg_sources=train_neg_sources,
                mask_audit=mask_audit,
                hard_negative_mask_audit=hard_negative_mask_audit,
                labeled_pairs_csv=(RESULTS / F["labeled_pairs"]).read_bytes(),
                canonical_records_csv=(RESULTS / F["canonical_records"]).read_bytes(),
                gate_results_csv=(RESULTS / F["gate_results"]).read_bytes(),
                payload_variant=args.payload,
                holdout_populations=frozen_holdout,
                augmentation_coverage=(balanced_coverage.model_dump(mode='json')
                    if args.mask_frac > 0 and balanced_policy.enabled else None),
                masking_profile=str(mask_cfg["profile"]),
                token_checkpoint=str(args.model),
                plan_loss=args.loss,
                plan_train_frac=args.train_frac,
                plan_sample=bool(args.sample),
            )
            print(
                f"[prepared-bundle] wrote {args.prepare_bundle} "
                f"({manifest.n_df:,} source rows, {manifest.n_payload:,} payload rows, "
                f"{manifest.n_pos:,} positives, {manifest.n_neg:,} negatives)",
                flush=True,
            )
            timing.mark("bundle_write")
            timing.dump_if_requested()
            return True

    @timed
    def run_hpo_lanes(self) -> None:
        """Dispatch the grid/TPE sweep lanes (short-circuits)."""
        _wandb = self._wandb
        args = self.args
        data = self.data
        mask_cfg = self.mask_cfg
        folds_override = self.folds_override
        dev_override = self.dev_override
        neg = self.neg
        train_neg = self.train_neg
        neg_sources = self.neg_sources
        train_neg_sources = self.train_neg_sources
        mask_hard_negatives = self.mask_hard_negatives
        mask_hard_negative_frac = self.mask_hard_negative_frac
        hard_negative_mask_prob = self.hard_negative_mask_prob
        hard_negative_mask_lo = self.hard_negative_mask_lo
        hard_negative_mask_hi = self.hard_negative_mask_hi
        mask_audit = self.mask_audit
        hard_negative_mask_audit = self.hard_negative_mask_audit
        run_tag = self.run_tag
        # ── HPO lanes: second07 fixed grid / second08 optuna TPE ───────────────
        if args.grid or args.hpo:
            from training.hpo import run_grid, run_tpe

            # TEST-LEAK WIRING (2026-09-12): both sweep lanes ride the SAME
            # component split the main lane just built — holdout mode passes
            # the test quarter (folds_override) + dev quarter (dev_override)
            # so the sweep trains q0+q1 and selects on q2; cv mode passes the
            # fold list. Never let a sweep rebuild its own folds over ALL
            # gtins: that put the dev/test quarters into the sweep's train
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
                    dynamic_mask_prob=hard_negative_mask_prob,
                    dynamic_mask_lo=hard_negative_mask_lo,
                    dynamic_mask_hi=hard_negative_mask_hi,
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
                    dynamic_mask_prob=hard_negative_mask_prob,
                    dynamic_mask_lo=hard_negative_mask_lo,
                    dynamic_mask_hi=hard_negative_mask_hi,
                    mask_audit=mask_audit,
                    hard_negative_mask_audit=hard_negative_mask_audit,
                    wandb_ctx=_wandb,
                )
            _write_hard_negative_mask_trace(
                hard_negative_mask_audit,
                run_tag=run_tag,
                sample=bool(args.sample),
                enabled=bool(mask_cfg["track_visibility"]),
            )
            return True

    @timed
    def train_folds(self) -> None:
        """Run the trainer over the resolved split."""
        _wandb = self._wandb
        args = self.args
        band = self.band
        data = self.data
        on_cuda = self.on_cuda
        hard_positives_enabled = self.hard_positives_enabled
        hp_pairs = self.hp_pairs
        neg = self.neg
        train_neg = self.train_neg
        neg_sources = self.neg_sources
        train_neg_sources = self.train_neg_sources
        mask_hard_negatives = self.mask_hard_negatives
        mask_hard_negative_frac = self.mask_hard_negative_frac
        hard_negative_mask_prob = self.hard_negative_mask_prob
        hard_negative_mask_lo = self.hard_negative_mask_lo
        hard_negative_mask_hi = self.hard_negative_mask_hi
        mask_audit = self.mask_audit
        hard_negative_mask_audit = self.hard_negative_mask_audit
        ann_mining_enabled = self.ann_mining_enabled
        attribute_conflict_enabled = self.attribute_conflict_enabled
        run_tag = self.run_tag
        mask_cfg = self.mask_cfg
        folds_override = self.folds_override
        dev_override = self.dev_override
        # src/training/training's DEFAULT_CFG key set (train_one_config reads these)

        # NO FALLBACK (owner Q27): every optimizer knob comes from the SSOT
        # training: block via runtime() — the inline 0.05/0.01/linear/1.0
        # literals duplicated config/training.yaml and could silently diverge.
        uniformity = _uniformity_cfg()
        with trace_step('training._main_inner.training'):
            cfg = {
                "architecture": runtime("architecture"),
                "epochs": args.epochs,
                "lr": args.lr,
                "warmup_ratio": args.warmup_ratio,
                "weight_decay": args.weight_decay,
                "projection_dropout": runtime("projection_dropout"),
                "label_smoothing": runtime("label_smoothing"),
                "random_easy_enabled": bool(runtime("random_easy_negatives")["enabled"]),
                "random_easy_ratio_to_hard": float(
                    runtime("random_easy_negatives")["ratio_to_hard"]
                ),
                "random_easy_candidate_pool_size": int(
                    runtime("random_easy_negatives")["candidate_pool_size"]
                ),
                "lr_scheduler": runtime("lr_scheduler"),
                "max_grad_norm": runtime("max_grad_norm"),
                "patience": ES_PATIENCE,
                "es_threshold": ES_THRESHOLD,
                "uniformity_weight": (
                    float(uniformity["weight"])
                    if bool(uniformity["enabled"])
                    else 0.0
                ),
                "late_epoch_decay_enabled": bool(
                    runtime("late_epoch_lr_decay")["enabled"]
                ),
                "late_epoch_decay_start_fraction": float(
                    runtime("late_epoch_lr_decay")["start_epoch_fraction"]
                ),
                "late_epoch_decay_multiplier": float(
                    runtime("late_epoch_lr_decay")["multiplier"]
                ),
            }
            t0 = time.perf_counter()
            # run_tag carries EVERY varying axis (owner ruling): the 07-series
            # ablation sweep ran 12 variants into the SAME train_fold_metrics.csv
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
                dynamic_mask_prob=hard_negative_mask_prob,
                dynamic_mask_lo=hard_negative_mask_lo,
                dynamic_mask_hi=hard_negative_mask_hi,
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
            _write_hard_negative_mask_trace(
                hard_negative_mask_audit,
                run_tag=run_tag,
                sample=bool(args.sample),
                enabled=bool(mask_cfg["track_visibility"]),
            )
            elapsed = time.perf_counter() - t0
        self.rows = rows
        self.elapsed = elapsed

    @timed
    def mask_effect_audit(self) -> None:
        """Score masked copies under the trained model."""
        _wandb = self._wandb
        rows = self.rows
        hard_negative_mask_audit = self.hard_negative_mask_audit
        args = self.args
        mask_audit = self.mask_audit
        payload = self.payload
        mask_cfg = self.mask_cfg
        model_tag = self.model_tag
        run_tag = self.run_tag
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
                _model = load_local_sentence_transformer(str(_src), device="cpu")
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
                        "gtin": [m["gtin"] for m in mask_audit],
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
        self.rows = rows

    @timed
    def publish_results(self) -> None:
        """Fold metrics, pointer, 07 emission, report, W&B, plots."""
        _wandb = self._wandb
        rows = self.rows
        args = self.args
        mask_audit = self.mask_audit
        hard_negative_mask_audit = self.hard_negative_mask_audit
        model_tag = self.model_tag
        frac_tag = self.frac_tag
        run_tag = self.run_tag
        s = self.s
        payload = self.payload
        row_bc = self.row_bc
        df = self.df
        dev_override = self.dev_override
        folds_override = self.folds_override
        elapsed = self.elapsed
        # persist metrics — SUFFIXED per run (see run_tag note): the fixed
        # F["fold_metrics"] name meant the 07-series ablation left only
        # the LAST variant's fold metrics on disk. Per-run copy keeps every
        # variant; the shared name stays for the "latest run" consumer
        # (report_plots greps train_*_fold_metrics.csv).
        # SAMPLE COLLISION FIX (audit round 2): this per-run name lacked the
        # sample axis — a --sample run still overwrote the full run's suffixed
        # metrics CSV (the pointer at F["fold_metrics"] was already protected;
        # this was the last unprotected surface of the collision class).
        _metrics_tag = (
            f"train_{model_tag}_{args.split}_{args.payload}_{frac_tag}"
            + (f"_sample{args.sample}" if args.sample else "")
        )
        out = RESULTS / f"{_metrics_tag}_fold_metrics.csv"
        all_rows = []
        for r in _LOG.progress(rows, desc="fold_metric_rows", unit="fold",
                               total=len(rows)):
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
            pointer = results_pointer(all_rows, run_tag=run_tag, args=args, metrics_csv_name=out.name)
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
        #     a 07-series sweep writes full + title_only into ONE csv)
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
                    report_payload = generate_report(
                        out,
                        pair_paths,
                        report_dir,
                        sorted(RESULTS.glob(f"train_{model_tag}_{run_tag}_fold*_train_scores.csv")),
                        sorted(RESULTS.glob(f"train_{model_tag}_{run_tag}_fold*_random_easy_scores.csv")),
                        data_path=F["dataset_deduped"],
                        canonical_path=F["canonical_records"],
                    )
                    robust = report_payload.get("robust_validation", {})
                    robust_metrics: dict[str, float] = {}
                    for metric, values in robust.get("aggregate", {}).items():
                        if not isinstance(values, dict):
                            continue
                        for statistic in ("mean", "std"):
                            value = values.get(statistic)
                            if isinstance(value, (int, float)) and np.isfinite(value):
                                robust_metrics[f"validation/robust/{metric}_{statistic}"] = float(value)
                    for operating_point, values in robust.get("operating_aggregate", {}).items():
                        if not isinstance(values, dict):
                            continue
                        safe_point = str(operating_point).replace("/", "_")
                        for metric, value in values.items():
                            if metric.endswith("_mean") or metric.endswith("_std"):
                                if isinstance(value, (int, float)) and np.isfinite(value):
                                    robust_metrics[
                                        f"validation/robust_operating/{safe_point}/{metric}"
                                    ] = float(value)
                    slice_aggregate = robust.get("slice_aggregate", {})
                    if isinstance(slice_aggregate, dict):
                        for slice_key, values in slice_aggregate.items():
                            if not isinstance(values, dict):
                                continue
                            safe_key = "_".join(
                                part for part in str(slice_key).replace("/", "_").split("::") if part
                            )
                            for metric in ("error_rate_mean", "f1_mean", "roc_auc_mean"):
                                value = values.get(metric)
                                if isinstance(value, (int, float)) and np.isfinite(value):
                                    robust_metrics[f"validation/robust_slice/{safe_key}/{metric}"] = float(value)
                    if robust_metrics:
                        _wandb.log_metrics(robust_metrics)
                        _wandb.log_config(
                            {"robust_validation": robust.get("config", {})}
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

        from training.hpo_metrics import numeric_calibration_metrics

        for r in _LOG.progress(ok_rows, desc="fold_telemetry", unit="fold",
                               total=len(ok_rows)):
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
        if not _REMOTE_TRAINING:
            _wandb.log_artifact(out, "fold-metrics")
            _log_run_artifacts_to_wandb(
                _wandb,
                run_tag=run_tag,
                model_tag=model_tag,
                metrics_path=out,
                rows=all_rows,
            )
        calibration_rows = [
            row for row in ok_rows if row.get("calibration_rand_index") is not None
        ]
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
