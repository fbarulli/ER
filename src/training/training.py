"""src/training/training.py — solid GPU-ready fine-tune pipeline for the
second-series lane.

Rewrite of the training path with everything the 07-series left out:

  TRACEBACKS      every failure prints the full chain (traceback.format_exc()
                  into the run log AND the CSV — no silent folds, no
                  "AUC=NaN, moving on").
  OPTUNA          HPO mode (--hpo): TPE over epochs/lr/warmup/band, each trial
                  recorded trial, best config reported + persisted.
  EARLY STOPPING  HF EarlyStoppingCallback on the dev AUC (patience
                  configurable); load_best_model_at_end so the reported
                  metric is the best checkpoint, not the last.
  FULL SETTINGS    warmup_ratio, weight decay, lr scheduler, grad clipping,
                  native mixed precision (on CUDA), checkpoints + save_total_limit, seeded
                  everything, save_best_model, per-fold dev/test split.

Group-aware splits throughout: gtin-level (no product straddles a fold),
and within each fold the train gtins are split again into train/dev for
early stopping (dev NEVER touches test).

Device: CPU here, CUDA on the Colab VM unchanged — the pipeline reads
torch.cuda.is_available() and flips precision/batch-size guidance, nothing else.

Usage:
  python src/training/train.py --loss contrastive     (entry; src/training/ is a package)
  python src/training/train.py --hpo --n-trials 20    # optuna TPE sweep
  python src/training/train.py --loss triplet --band 0.45-0.80


Artifacts (results/, SSOT via config/paths.yaml):
  train_<model>_fold_metrics.csv  one row per fold per config (incl. failures
                               with traceback column)
  train_<model>_hpo_best.json    best config + its metrics (HPO mode)
"""

from __future__ import annotations

import json
import os
import time
import traceback
import fcntl
from pathlib import Path

import numpy as np
import pandas as pd

# DEFAULT_MODEL retired: the entry (train.py) resolves the trainer base
# from the config SSOT (models.multilingual_l12); the lib.cache.MODEL_NAME
# indirection was dead code with a phantom docstring.
from transformers import EarlyStoppingCallback, TrainerCallback

from core.common import (
    F,
    RESULTS,
    SEED,
    artifact,
    ensure_parent,
    kfold_gtins,
    load_config,
    config_section,
    load_local_sentence_transformer,
    metadata_text,
    pair_auc,
    pair_similarity,
    plot_dpi,
    recall_column_suffix,
    row_metadata_text,
    runtime,
    training_cfg,
    trace_artifact,
)
from core.schemas import check_labeled_pairs_frame
from core.common import SSOT_CONTRASTIVE_MARGIN as _SSOT_MARGIN
from core.common import runtime as _runtime
from core.timing import emit_timing
from core.step_trace import send, timed, trace_step
from core.tracing import (
    SCOPE_ENTITY,
    SCOPE_GROUP,
    SCOPE_RUN,
    flush_stage_trace,
    stage_trace,
)
from core.perf_switches import perf_enabled

# D7 telemetry: wall seconds spent inside load_config's deepcopy, keyed by
# call-site label (fold{n}.* for per-fold sites); aggregated per fold and
# emitted by train_one_config before it returns.
_CFG_DEEPCOPY_TOTALS: dict[str, float] = {}


# ═══════════════════════════════════════════════════════════════════════════
# CONSOLIDATED TRACE — the "training" stage (owner contract: core/tracing.py)
# ═══════════════════════════════════════════════════════════════════════════
# Why a stage-level owner here: core.tracing commits ONE stage per (run, stage)
# and REPLACES that stage's rows in place (see its "run identity" doctrine), so
# a stage must have exactly one writer. train.py is the run entry point; this
# module is where the folds/epochs/batches actually happen, so the writer lives
# here and train.py flushes it once when the run is over. Every row this module
# adds therefore lands in the SAME commit, in flow order, joined to the
# data-prep rows by the run id core.tracing already resolves (EUROMONITOR_RUN_ID
# / content fingerprint) — never a second id scheme and never a second file.
TRAINING_STAGE = "training"

#: The training side's trace-writer slot: the shared shim's (see
#: :func:`core.tracing.stage_trace`), ``None`` until first use. The ONE
#: reset-on-flush variant in the tree (``flush_training_trace``).
_TRAINING_TRACE = None


def training_trace(stage: str | None = None):
    """The ONE writer for the training-side stage of the current run.

    The laziness and the run id resolution live in the shared shim
    (:func:`core.tracing.stage_trace`); this module owns only its slot.

    ``stage`` pins the process's stage (the --prepare-bundle lane runs under
    the orchestrator's ``full_bundle`` stage, not ``training``). Pinning is
    all-or-nothing: an already-created writer may not be re-labelled, because
    rows already recorded carry the old stage and core.tracing commits a
    writer's rows as ONE stage.
    """
    global _TRAINING_TRACE
    _TRAINING_TRACE = stage_trace(TRAINING_STAGE, _TRAINING_TRACE, pinned=stage)
    return _TRAINING_TRACE


def flush_training_trace():
    """Commit this process's training-side stage rows once (no-op when empty).

    Called by the run entry point (train.py) after the run is over. This is the
    ONE reset-on-flush variant in the tree (see :func:`core.tracing.stage_trace`):
    the writer is released, so a later phase opens a fresh one for the next run
    id instead of re-committing stale rows.
    """
    global _TRAINING_TRACE
    path = flush_stage_trace(_TRAINING_TRACE)
    if path is not None:
        _TRAINING_TRACE = None
    return path


@timed
def _timed_load_config(key: str) -> dict:
    """Thin delegate: the measured load_config lives in _LaneContext."""
    return _LaneContext._load_config_measured(key)

_SSOT_HP = bool(_runtime("hard_positives"))  # no-fallback SSOT
from core.hard_negatives import mine_hard_negatives, pairs_in_set
from core.worker_telemetry import write_worker_live_status

# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

# ── runtime knobs: SSOT ONLY (config/paths.yaml training: block via
# lib.common.runtime). No duplicated literals here — changing batch size or
# cadence happens in the config, once, for every script.
# NO FALLBACKS (owner Q27): split.cv_folds is hard-indexed — a missing key
# crashes at import, never a silent 5.
CV_FOLDS = int(config_section("split", "cv_folds"))
DEV_FRACTION = runtime("dev_fraction")
BATCH_SIZE_CPU = runtime("batch_size_cpu")
BATCH_SIZE_CUDA = runtime("batch_size_cuda")
MAX_TRIPLES = runtime("max_triples")
EVAL_STEPS_PER_EPOCH = runtime("eval_steps_per_epoch")
ES_PATIENCE = runtime("es_patience")
ES_THRESHOLD = runtime("es_threshold")
_ANN_MINING_CFG = config_section("mining", "ann")
N_TARGET_MINING = int(_ANN_MINING_CFG["target"])
ANN_MINING_ENABLED = bool(_ANN_MINING_CFG["enabled"])
MASK_TRACK_PER_EPOCH = bool(config_section("masking", "track_per_epoch"))
TRACK_DATAPOINT_USAGE = bool(config_section("training", "track_datapoint_usage"))
# Contrastive per-step telemetry is skipped only when the run has disabled
# datapoint tracking (the consumers of that telemetry) AND the switch is on.
# With the default config (track_datapoint_usage: true) the exact telemetry
# artifacts are preserved; ER_PERF_LEGACY=1 always collects.
_COLLECT_CONTRASTIVE_TELEMETRY = not (
    perf_enabled("text.contrastive_telemetry_gate") and not TRACK_DATAPOINT_USAGE
)
_SHARE_TEXT_HASHES = perf_enabled("text.share_sampler_hashes")
_UNIFORMITY_CFG = config_section("training", "uniformity_regularization")
# ── DATAPOINT POPULATION REGISTRY (SSOT for the coverage audit) ────────────
# DERIVED FROM THE PRODUCERS, not hand-kept beside them (audit A4-2). Every
# tag a producer can write into a pair-population list or a negative-source
# array is declared here together with the emitter that writes it, so a tag
# cannot be "known" without naming whoever produces it, and a registered
# population cannot be forgotten by the coverage audit.
#
#   role == "positive"        -> emitted by _training_pair_populations
#   role == "negative_source" -> emitted into neg_sources/train_neg_sources
#   role == "presented_label" -> only ever a PRESENTATION label (a negative
#                                whose text was replaced), never a source tag
#   dynamic == True           -> mining-refresh population: "configured" only
#                                while its producer is enabled by config
#
# Two guards keep this table honest, because the defect was a SILENT omission:
#   * runtime — _write_datapoint_usage raises
#     UnregisteredDatapointPopulationError at the point of use when a producer
#     emits a tag this table does not declare (the coverage artifact is still
#     written first, so the evidence survives the failure);
#   * static — tests/test_datapoint_coverage.py scans the producers named below
#     and fails if the scanned tag set and this table ever diverge.
DATAPOINT_POPULATION_SPEC: dict[str, dict[str, object]] = {
    "gate_positive": {
        "emitter": "training.training._training_pair_populations",
        "role": "positive",
        "dynamic": False,
    },
    "hard_positive": {
        "emitter": "training.training._training_pair_populations",
        "role": "positive",
        "dynamic": False,
    },
    "masked_positive": {
        "emitter": "training.training._training_pair_populations",
        "role": "positive",
        "dynamic": False,
    },
    "gate": {
        "emitter": "training.train: neg_sources = np.full(len(neg), 'gate')",
        "role": "negative_source",
        "dynamic": False,
    },
    # Static targeted-attribute negatives (train.py: np.full(
    # len(targeted_attribute_neg), "targeted_attribute_conflict")). Missing
    # from this registry until audit A4-2, which is exactly why 350
    # negatives/fold could be presented with no coverage row and no
    # present/selected/backprop counters.
    "targeted_attribute_conflict": {
        "emitter": "training.train: np.full(len(targeted_attribute_neg), ...)",
        "role": "negative_source",
        "dynamic": False,
    },
    # Static cross-brand negatives (train.py: np.full(len(cross_brand_neg),
    # "cross_brand_conflict")). The ONLY negatives whose two sides carry
    # DIFFERENT brands: without them brand agreement is 100% in both training
    # classes and measured brand separation is exactly 0.000.
    "cross_brand_conflict": {
        "emitter": "training.train: np.full(len(cross_brand_neg), 'cross_brand_conflict')",
        "role": "negative_source",
        "dynamic": False,
    },
    # Counterfactual twins (train.py: np.full(len(cf_pairs),
    # "counterfactual")). Minimal single-agreed-field flips of positive
    # anchors, labeled 0 by construction and appended to the negative pool.
    "counterfactual": {
        "emitter": "training.train: counterfactual twin minting + training.masking.augment_counterfactual_twins",
        "role": "negative_source",
        "dynamic": False,
    },
    "attribute_conflict": {
        "emitter": "training.train: np.full(len(_attr_neg), 'attribute_conflict')",
        "role": "negative_source",
        "dynamic": True,
    },
    # Negative-supply lane (owner ruling 2026-10-03). Emitted by
    # training.negative_supply (base_negative, real_partner) and appended
    # TRAINING-ONLY by train.py (minted); the lane never enters the eval pool,
    # so the coverage audit sees these populations only in training.
    "base_negative": {
        "emitter": "training.negative_supply.build_lane_training_data (neg_source)",
        "role": "negative_source",
        "dynamic": False,
    },
    "real_partner": {
        "emitter": "training.negative_supply.mine_real_partners (neg_source)",
        "role": "negative_source",
        "dynamic": False,
    },
    "minted": {
        "emitter": "training.train: np.full(len(neg_minted), 'minted')",
        "role": "negative_source",
        "dynamic": False,
    },
    "random_easy": {
        "emitter": "training.training._mix_random_easy_training_negatives",
        "role": "negative_source",
        "dynamic": False,
    },
    "ann_finetuned": {
        "emitter": "training.ann_refresh.refresh_finetuned_ann",
        "role": "presented_label",
        "dynamic": True,
    },
}
KNOWN_DATAPOINT_POPULATIONS = tuple(DATAPOINT_POPULATION_SPEC)
# Populations whose producer is config-gated: they are "configured" only while
# the producer reports itself enabled (was the local literal {"ann_finetuned",
# "attribute_conflict"} at the coverage site — derived now).
DYNAMIC_DATAPOINT_POPULATIONS = frozenset(
    name for name, spec in DATAPOINT_POPULATION_SPEC.items() if spec["dynamic"]
)
# Populations that can appear as a NEGATIVE SOURCE tag in the train fold.
# Derived, so the per-fold negative-source census enumerates every producer
# instead of a hardcoded sub-list that silently skipped a population.
NEGATIVE_SOURCE_DATAPOINT_POPULATIONS = tuple(
    name
    for name, spec in DATAPOINT_POPULATION_SPEC.items()
    if spec["role"] == "negative_source"
)
# Fallback labels the same code paths emit when a source array is absent or
# short. They are NOT registered populations: the audit rejects them LOUDLY
# (they mean pair provenance was lost), and this set exists so the failure
# message can name the emitter. Scanned by the divergence test as well.
DATAPOINT_FALLBACK_TAGS = frozenset({"hard_negative", "hard_neg", "unknown"})
EVAL_ONLY_DATAPOINT_POPULATIONS: set[str] = set()

# train.py mints static masked/value-swapped copies of hard negatives into
# the neg and train_neg pools with an "<source>+aug" provenance label
# (f-string concat — invisible to the static producer scan). The copy's
# SOURCE is still the base population; the minting mode is carried by the
# pair lineage (target_mode from hard_negative_mask_audit). Compound tags
# are normalized to their base population wherever a registry check
# happens, so the registry never grows a combinatorial "<base>+aug" family.
AUG_SOURCE_SUFFIX = "+aug"


def _base_population_tag(source: str) -> str:
    """Thin delegate: ownership lives in _LaneContext."""
    return _LaneContext._base_population_tag(source)


class _LaneContext:
    """Small SR owner of the lane's context knobs: the '+aug' tag owner
    (`_base_population_tag`), the deepcopy-measured config loader
    (`_load_config_measured`, behind the pinned `_timed_load_config`), and
    the registry-violation signal the coverage audit raises."""

    # Re-exported for the coverage-audit message consumers; the class stays
    # module-level below (tests import it from training.training).
    Unregistered = None  # bound after UnregisteredDatapointPopulationError is defined

    @staticmethod
    def _base_population_tag(source: str) -> str:
        """Strip the static-copy '+aug' suffix minted by train.py's augmentation."""
        source = str(source)
        if source.endswith(AUG_SOURCE_SUFFIX):
            return source[: -len(AUG_SOURCE_SUFFIX)]
        return source

    @staticmethod
    def _load_config_measured(key: str) -> dict:
        """Behavior-identical load_config with a measured deepcopy component."""
        started = time.perf_counter()
        value = load_config()
        _CFG_DEEPCOPY_TOTALS[key] = _CFG_DEEPCOPY_TOTALS.get(key, 0.0) + (
            time.perf_counter() - started
        )
        return value


class UnregisteredDatapointPopulationError(RuntimeError):
    """A producer emitted a datapoint population the registry does not know.

    Raised at the point of use by the per-fold coverage audit, AFTER the
    coverage artifact for that fold has been written, so an undeclared
    population is both loud and still inspectable. Silence was the defect
    (audit A4-2): a tag outside the registry used to be dropped from the
    coverage rows with no warning at all.
    """

# bind the violation signal after its definition (class-body ordering).
_LaneContext.Unregistered = UnregisteredDatapointPopulationError

# DEFAULT_CFG REMOVED (audit 2026-09-09): zero readers since the entry
# (train.py) constructs its own cfg dict; a stale epochs=2 default here
# contradicted the SSOT epochs=10 and was pure dead-code risk.

# TPE search space, SSOT: config/training.yaml hpo.tpe_space (validated by
# HpoSpaceSpec at load — lo < hi per knob). The dict literal was a second
# declaration the config could not steer (audit 2026-09-09, owner Q27).
from core.common import hpo_cfg as _hpo_cfg_load

_HPO_SETTINGS = _hpo_cfg_load()
HPO_SPACE = {
    k: tuple(v)
    for k, v in _HPO_SETTINGS["tpe_space"].items()
    if isinstance(v, (list, tuple)) and len(v) == 2
}
# TASK B item 5: optional categorical scheduler choices (null => not swept).
HPO_SCHEDULERS = _HPO_SETTINGS["tpe_space"].get("lr_scheduler") or None

# HPO objective protocol, SSOT: hpo.objective /
# hpo.selection_skip_test_eval (validated by HpoSpec/ObjectiveSpec at load).
# Both modes now rank trials on the calibrated direct-assignment Rand proxy;
# holdout selection still skips the test quarter entirely.
_HPO_OBJ_TABLE = _HPO_SETTINGS["objective"]
HPO_OBJECTIVE_HOLDOUT = _HPO_OBJ_TABLE["holdout"]
HPO_OBJECTIVE_CV = _HPO_OBJ_TABLE["cv"]
HPO_SKIP_TEST_EVAL = bool(_HPO_SETTINGS["selection_skip_test_eval"])


class RequiredCalibrationError(RuntimeError):
    """Signal that normal training cannot complete without Rand calibration."""


class CalibrationEvaluatorError(RuntimeError):
    """Signal an unexpected calibration evaluator failure with fold context."""


class FoldExecutionError(RuntimeError):
    """Signal that a selection lane received incomplete fold evidence."""

    def __init__(self, lane: str, incomplete_rows: list[dict]) -> None:
        self.lane = lane
        self.incomplete_rows = incomplete_rows
        # Compatibility for the first failure-path oracle and existing callers.
        self.failed_rows = incomplete_rows
        details = "; ".join(
            f"fold {row.get('fold', '<unknown>')}: "
            f"status={row.get('status', 'missing')}; "
            f"{str(row.get('traceback', 'no traceback')).splitlines()[-1]}"
            for row in incomplete_rows
        )
        super().__init__(
            f"{lane} cannot select from incomplete fold(s): {details}"
        )


@timed
def require_no_failed_folds(rows: list[dict], *, lane: str) -> None:
    """Make incomplete calibration evidence fatal before selection aggregation."""
    _CalibrationEvaluator._fold_evidence_complete(rows, lane=lane)


class _CalibrationEvaluator:
    """Small SR owner of calibration evaluation discipline.

    Holds the selection-requirement gate (`_fold_evidence_complete`) and one
    fold's holdout/CV calibration scoring (`_evaluate_fold_calibration`) —
    the loud RequiredCalibrationError / CalibrationEvaluatorError handling —
    plus the one _precision_at_recall consumption contract. The module-level
    `require_no_failed_folds` stays the pinned API for the sweep lanes.
    """

    @staticmethod
    def _fold_evidence_complete(rows: list[dict], *, lane: str) -> None:
        """Make incomplete calibration evidence fatal before selection aggregation."""
        incomplete_rows = [
            row
            for row in rows
            if row.get("status") != "ok"
            or not np.isfinite(row.get("calibration_rand_index", float("nan")))
        ]
        if not rows or incomplete_rows:
            raise FoldExecutionError(lane, incomplete_rows or [{"status": "missing"}])

    @staticmethod
    def _evaluate_fold_calibration(
        calibration_config,
        *,
        fold_i: int,
        sample: bool,
        model,
        df,
        payload,
        structured_features,
        calibration_pos: np.ndarray,
        calibration_neg: np.ndarray,
        row_bc: np.ndarray,
        structured_feature_weight: float,
    ) -> dict[str, object]:
        """Score one fold's Rand calibration; unavailable stays loud.

        Every lane uses the same component-safe calibration/Rand
        computation. The holdout population remains isolated for
        final reporting and is never used by HPO selection.
        """
        from training.hpo_metrics import (
            CALIBRATION_REASON_EMPTY_SPLIT,
            evaluate_calibration_trial,
            unavailable_calibration_metrics,
        )

        if len(calibration_pos) == 0 or len(calibration_neg) == 0:
            calibration_metrics = _CalibrationEvaluator._empty_split_result(
                fold_i,
                sample,
                calibration_pos,
                calibration_neg,
                reason_code=CALIBRATION_REASON_EMPTY_SPLIT,
                unavailable=unavailable_calibration_metrics,
            )
        else:
            calibration_metrics = _CalibrationEvaluator._evaluate_available(
                calibration_config,
                fold_i=fold_i,
                model=model,
                df=df,
                payload=payload,
                structured_features=structured_features,
                calibration_pos=calibration_pos,
                calibration_neg=calibration_neg,
                row_bc=row_bc,
                structured_feature_weight=structured_feature_weight,
            )
        return calibration_metrics

    @staticmethod
    def _evaluate_available(
        calibration_config,
        *,
        fold_i: int,
        model,
        df,
        payload,
        structured_features,
        calibration_pos: np.ndarray,
        calibration_neg: np.ndarray,
        row_bc: np.ndarray,
        structured_feature_weight: float,
    ) -> dict[str, object]:
        """The available-population branch: score + wrapped evaluator failure.

        Invariant: the try/except must stay glued to `from exc` — an evaluator
        exception keeps its traceback instead of becoming a prunable result.
        """
        from training.hpo_metrics import evaluate_calibration_trial

        try:
            return evaluate_calibration_trial(
                model=model,
                df=df,
                payload=payload,
                structured_features=structured_features,
                pos_pairs=calibration_pos,
                neg_pairs=calibration_neg,
                row_bc=row_bc,
                structured_weight=structured_feature_weight,
                batch_size=runtime("batch_size_eval"),
                config=calibration_config,
                include_collapse_guardrail=bool(
                    calibration_config["collapse_guardrail"]["enabled"]
                ),
            )
        except Exception as exc:
            raise CalibrationEvaluatorError(
                f"calibration evaluator failed on fold {fold_i}"
            ) from exc

    @staticmethod
    def _empty_split_result(
        fold_i: int,
        sample: bool,
        calibration_pos: np.ndarray,
        calibration_neg: np.ndarray,
        *,
        reason_code: str,
        unavailable,
    ):
        """The empty-split branch: unavailable record + loud failure gate."""
        calibration_metrics = unavailable(
            reason_code=reason_code,
            reason=(
                "empty calibration split — Rand threshold calibration "
                f"needs pos={len(calibration_pos)}, neg={len(calibration_neg)}"
            ),
            positive_pairs=len(calibration_pos),
            negative_pairs=len(calibration_neg),
        )
        print(
            f"  [calibration] fold {fold_i}: {'sample' if sample else 'REQUIRED'} calibration "
            f"unavailable; {calibration_metrics['calibration_reason']}",
            flush=True,
        )
        # Chain-check samples intentionally do not reserve a Rand
        # calibration population: their job is to prove training,
        # checkpointing, and inference wiring on a bounded input.
        # Full runs must still fail loudly rather than publish an
        # uncalibrated threshold.
        if not sample:
            raise RequiredCalibrationError(
                calibration_metrics["calibration_reason"]
            )
        return calibration_metrics

    @staticmethod
    def _precision_at_recall(y: np.ndarray, scores: np.ndarray, recall_target: float):
        """07-series precision/recall/threshold at a target recall.

        Threshold = the LOWEST score still achieving recall_target (any
        higher cut drops below it); precision at that cut with the FP count
        implied. Deterministic: sorted order, ties resolved by score value.
        """
        order = np.argsort(-scores, kind="stable")
        y_sorted = y[order]
        s_sorted = scores[order]
        n_pos = int((y == 1).sum())
        if n_pos == 0 or len(scores) == 0:
            return float("nan"), float("nan"), float("nan")
        tp_cum = np.cumsum(y_sorted == 1)
        # first rank where recall >= target
        k = int(np.searchsorted(tp_cum, int(np.ceil(recall_target * n_pos))))
        k = min(k, len(s_sorted) - 1)
        thr = float(s_sorted[k])
        tp = int(tp_cum[k])
        fp = int((k + 1) - tp)
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / n_pos
        return float(prec), float(rec), thr

    @staticmethod
    def _precision_at_recall_audit(
        y: np.ndarray, scores: np.ndarray, recall_target: float
    ):
        """One consumer contract: the 07-schema audit triple beside the
        (precision, recall, threshold) tuple — TP/FP at the same cut."""
        _prec90, _rec90, _thr90 = _CalibrationEvaluator._precision_at_recall(
            y, scores, recall_target
        )
        _tp90 = int(((scores >= _thr90) & (y == 1)).sum())
        _fp90 = int(((scores >= _thr90) & (y == 0)).sum())
        return _prec90, _rec90, _tp90, _fp90, _thr90


# ═══════════════════════════════════════════════════════════════════════════

from core.ranking_metrics import (
    added_encode_rows,
    build_evaluation_pool,
    competitors_per_query,
    ranking_at_k_by_query,
    ranking_coverage,
    youden_threshold,
)

# ═══════════════════════════════════════════════════════════════════════════
# Training (ST 6 modern Trainer path with HF early stopping)
# ═══════════════════════════════════════════════════════════════════════════


@timed
def _align_model_token_ids(model: SentenceTransformer) -> None:
    """Thin delegate: token-ID alignment lives in _CheckpointPublisher."""
    _CheckpointPublisher._align_model_token_ids(model)


@timed
def _configure_projection_dropout(model, probability: float) -> bool:
    """Thin delegate: projection-dropout wiring lives in _CheckpointPublisher."""
    return _CheckpointPublisher._configure_projection_dropout(model, probability)


@timed
def _make_checkpoint_tokenizer_portable(checkpoint: Path) -> None:
    """Thin delegate: tokenizer portability lives in _CheckpointPublisher."""
    _CheckpointPublisher._make_checkpoint_tokenizer_portable(checkpoint)


@timed
def _write_checkpoint_manifest(
    checkpoint: Path,
    *,
    epoch,
    global_step: int,
    model,
    optimizer,
    scheduler,
    scaler,
    trainer_state,
    trainer_control,
    training_args,
) -> None:
    """Thin delegate: the resume manifest lives in _CheckpointPublisher."""
    _CheckpointPublisher._write_checkpoint_manifest(
        checkpoint,
        epoch=epoch,
        global_step=global_step,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        trainer_state=trainer_state,
        trainer_control=trainer_control,
        training_args=training_args,
    )


def _resume_component_filename(name: str) -> str:
    """Resolve a native-HF resume component filename through the config SSOT.

    The checkpoint layout owner is ``training_cfg().bundle.resume_only_filenames``;
    every surface reads the name from there instead of re-spelling it. A
    filename the config does not declare is reported, never invented.
    """
    bundle = training_cfg().bundle
    for filename in bundle.resume_only_filenames:
        if filename == name:
            return filename
    raise RuntimeError(
        f"resume component filename {name!r} is not declared in "
        f"bundle.resume_only_filenames={list(bundle.resume_only_filenames)}"
    )


def _trainer_state_filename() -> str:
    """The native-HF trainer-state filename (config SSOT, not resume-only)."""
    return training_cfg().bundle.trainer_state_file


def _trainer_best_key() -> str:
    """The trainer-state JSON key naming the selected checkpoint (config SSOT)."""
    return training_cfg().bundle.trainer_best_key


def _required_resume_filenames() -> tuple[str, ...]:
    """The resume preflight set for ``on_save`` and the fold resume path.

    Optimizer/scheduler/RNG are the native-HF resume state; the manifest and the
    trainer-state file complete the set. ``scaler.pt`` and ``training_args.bin``
    are deliberately NOT required here: they exist only under AMP / for arg
    replay, so the preflight stays byte-identical to the historical check.
    """
    return (
        _resume_component_filename("optimizer.pt"),
        _resume_component_filename("scheduler.pt"),
        _resume_component_filename("rng_state.pth"),
        training_cfg().colab.checkpoint_manifest_name,
        _trainer_state_filename(),
    )


class _CheckpointPublisher:
    """Small SR owner of checkpoint publication prerequisites.

    * `_align_model_token_ids`        tokenizer special-token IDs -> model config
    * `_configure_projection_dropout` idempotent serializable dropout module
    * `_make_checkpoint_tokenizer_portable`
                 Transformers 5 -> older HF runtime tokenizer saves
    * `_write_checkpoint_manifest`    native HF resume snapshot description
    * `_publication_deferred`         the one env flag parser owned by the
    run finisher. Module-level `_align_model_token_ids`,
    `_configure_projection_dropout`, `_make_checkpoint_tokenizer_portable`
    and `_write_checkpoint_manifest` stay the pinned (@timed) call surface.
    """

    @staticmethod
    def _align_model_token_ids(model: SentenceTransformer) -> None:
        """Make tokenizer special-token IDs the single source of truth.

        Transformers can load a tokenizer whose PAD/BOS/EOS IDs differ from the
        IDs serialized in the base model config.  It repairs that mismatch in
        memory, but relying on that implicit repair leaves checkpoint contents
        dependent on the loader version.  Align both configs explicitly before
        the trainer starts; the model config is then serialized with each saved
        checkpoint.  Sentence-transformers models are encoder-only, so the
        generation config is normally unused, but align it when Transformers
        exposes one as well.
        """
        model = getattr(model, 'module', model)  # duck-typed DataParallel/DDP unwrap
        tokenizer = model.tokenizer
        auto_model = model[0].auto_model

        token_ids = {
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        changed: dict[str, tuple[object, int]] = {}
        for name, token_id in token_ids.items():
            if token_id is None:
                continue
            old_value = getattr(auto_model.config, name, None)
            if old_value != token_id:
                changed[name] = (old_value, token_id)
            setattr(auto_model.config, name, token_id)

            generation_config = getattr(auto_model, "generation_config", None)
            if generation_config is not None:
                setattr(generation_config, name, token_id)

        if changed:
            details = ", ".join(
                f"{name}={old!r}->{new!r}" for name, (old, new) in changed.items()
            )
            print(f"    [tokens] aligned tokenizer IDs in model config: {details}", flush=True)

        unresolved = {
            name: (getattr(auto_model.config, name, None), token_id)
            for name, token_id in token_ids.items()
            if token_id is not None and getattr(auto_model.config, name, None) != token_id
        }
        if unresolved:
            raise RuntimeError(f"tokenizer/model token-ID alignment failed: {unresolved}")

    @staticmethod
    def _configure_projection_dropout(model, probability: float) -> bool:
        """Append serializable dropout after pooling, idempotently.

        Existing regularized checkpoints already contain the SentenceTransformers
        dropout module. In that case update its probability instead of appending a
        second layer. Returns whether a new module was added.
        """
        probability = float(probability)
        if not 0.0 <= probability < 1.0:
            raise ValueError("projection dropout must be in [0, 1)")

        from sentence_transformers.sentence_transformer.modules import Dropout

        existing = [module for module in model.children() if isinstance(module, Dropout)]
        if len(existing) > 1:
            raise RuntimeError("model contains multiple SentenceTransformer dropout modules")
        if existing:
            existing[0].dropout = probability
            existing[0].dropout_layer.p = probability
            return False
        if probability == 0.0:
            return False
        model.add_module("projection_dropout", Dropout(dropout=probability))
        return True

    @staticmethod
    def _make_checkpoint_tokenizer_portable(checkpoint: Path) -> None:
        """Keep Transformers 5 tokenizer saves loadable by older HF runtimes.

        Transformers 5 may serialize the fast tokenizer as ``TokenizersBackend``.
        That name is not an AutoTokenizer class in the older runtime used by some
        Colab images, even though the accompanying ``tokenizer.json`` is valid.
        The generic fast-tokenizer class reads the same file and preserves the
        already aligned special-token IDs.
        """
        config_path = checkpoint / "tokenizer_config.json"
        tokenizer_path = checkpoint / "tokenizer.json"
        if not config_path.is_file() or not tokenizer_path.is_file():
            raise RuntimeError(
                f"checkpoint is missing tokenizer assets: {checkpoint}"
            )

        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("tokenizer_class") == "TokenizersBackend":
            config["tokenizer_class"] = "PreTrainedTokenizerFast"
            config_path.write_text(
                json.dumps(config, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

        special_tokens = {
            name: config[name]
            for name in ("bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token")
            if name in config
        }
        special_map_path = checkpoint / "special_tokens_map.json"
        if special_tokens and not special_map_path.exists():
            special_map_path.write_text(
                json.dumps(special_tokens, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    @staticmethod
    def _model_files(checkpoint: Path) -> list[str]:
        """The serialized weight files of this checkpoint (safetensors first)."""
        model_files = sorted(
            path.name
            for path in checkpoint.glob("model.safetensors*")
            if path.is_file()
        )
        if not model_files:
            model_files = sorted(
                path.name
                for path in checkpoint.glob("pytorch_model*.bin*")
                if path.is_file()
            )
        return model_files

    @staticmethod
    def _token_id_snapshot(tokenizer, auto_model, token_names) -> dict:
        """The three tokenizer/model/generation token-ID blocks of the manifest."""
        return {
            "tokenizer_token_ids": {
                name: getattr(tokenizer, name, None) for name in token_names
            },
            "model_config_token_ids": {
                name: getattr(auto_model.config, name, None) for name in token_names
            },
            "generation_config_token_ids": {
                name: getattr(getattr(auto_model, "generation_config", None), name, None)
                for name in token_names
            },
        }

    @staticmethod
    def _files_block(checkpoint: Path, optimizer, scheduler, scaler) -> dict:
        """The exact components of the requested checkpoint dict.

        Invariant: they remain in their native HF files so model/optimizer
        tensors are not serialized a second time into a multi-GB sidecar.
        """
        return {
            "model_state_dict": _CheckpointPublisher._model_files(checkpoint),
            "optimizer_state_dict": _resume_component_filename("optimizer.pt") if optimizer is not None else None,
            "scheduler_state_dict": _resume_component_filename("scheduler.pt") if scheduler is not None else None,
            "scaler_state_dict": _resume_component_filename("scaler.pt") if scaler is not None else None,
            "rng_state": _resume_component_filename("rng_state.pth"),
            "trainer_state": _trainer_state_filename(),
            "training_args": _resume_component_filename("training_args.bin"),
        }

    @staticmethod
    def _resume_block() -> dict:
        """The fixed native-HF resume component map manifest field."""
        state_file = _trainer_state_filename()
        return {
            "native_hf_resume": {
                "trainer_state": state_file,
                "trainer_control": f"{state_file}:control",
                "training_args": _resume_component_filename("training_args.bin"),
                "optimizer": _resume_component_filename("optimizer.pt"),
                "scheduler": _resume_component_filename("scheduler.pt"),
                "rng": _resume_component_filename("rng_state.pth"),
            }
        }

    @staticmethod
    def _write_checkpoint_manifest(
        checkpoint: Path,
        *,
        epoch,
        global_step: int,
        model,
        optimizer,
        scheduler,
        scaler,
        trainer_state,
        trainer_control,
        training_args,
    ) -> None:
        """Describe the complete native HF resume snapshot without duplicating it."""
        from core.model_input import model_input_composition

        log_history = getattr(trainer_state, "log_history", []) or []
        losses = [entry["eval_loss"] for entry in log_history if "eval_loss" in entry]
        tokenizer = getattr(model, "tokenizer", None)
        auto_model = model[0].auto_model
        token_names = ("pad_token_id", "bos_token_id", "eos_token_id")
        manifest = {
            "format": "euromonitor-hf-resume-v1",
            "epoch": epoch,
            "global_step": global_step,
            "best_loss": float(min(losses)) if losses else None,
            # The encoder TEXT this checkpoint was trained on. Weights are only
            # comparable, and only reusable at scoring time, together with the
            # composition that produced them — a checkpoint trained on one
            # composition is not interchangeable with another.
            "model_input": model_input_composition().model_dump(),
            "files": _CheckpointPublisher._files_block(
                checkpoint, optimizer, scheduler, scaler
            ),
            **_CheckpointPublisher._token_id_snapshot(tokenizer, auto_model, token_names),
            **_CheckpointPublisher._resume_block(),
        }
        with trace_step('training.write_checkpoint_manifest'):
            (checkpoint / training_cfg().colab.checkpoint_manifest_name).write_text(
                json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )

    @staticmethod
    def _publication_deferred() -> bool:
        """One flag parser for checkpoint publication owned by the run finisher."""
        return os.environ.get("EUROMONITOR_DISABLE_DVC_CHECKPOINTS", "0").lower() in {"1", "true", "yes"}


# _auc/_cos -> _common SSOT (see GATES_MAP.md)
_auc = pair_auc


_cos = pair_similarity


@timed
def _split_safe_random_negative_pairs(
    df: pd.DataFrame,
    row_bc: np.ndarray,
    split_gtins: set[str],
    *,
    seed: int,
    n_neg: int,
) -> np.ndarray:
    """Thin delegate: ownership lives in _PopulationBuilders."""
    return _PopulationBuilders._split_safe_random_negative_pairs(
        df, row_bc, split_gtins, seed=seed, n_neg=n_neg
    )


@timed
def _mix_random_easy_training_negatives(
    hard_pairs: np.ndarray,
    hard_sources: np.ndarray,
    *,
    df: pd.DataFrame,
    row_bc: np.ndarray,
    train_gtins: set[str],
    seed: int,
    enabled: bool,
    ratio_to_hard: float,
    candidate_pool_size: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Thin delegate: ownership lives in _PopulationBuilders."""
    return _PopulationBuilders._mix_random_easy_training_negatives(
        hard_pairs,
        hard_sources,
        df=df,
        row_bc=row_bc,
        train_gtins=train_gtins,
        seed=seed,
        enabled=enabled,
        ratio_to_hard=ratio_to_hard,
        candidate_pool_size=candidate_pool_size,
    )


@timed
def _mnrl_training_triples_with_populations(
    train_pos: np.ndarray,
    train_neg: np.ndarray,
    *,
    mask_audit: list[dict] | None,
    hard_negative_mask_audit: list[dict] | None,
) -> list[tuple[tuple[int, int, int], str]]:
    """Thin delegate: ownership lives in _PopulationBuilders."""
    return _PopulationBuilders._mnrl_training_triples_with_populations(
        train_pos,
        train_neg,
        mask_audit=mask_audit,
        hard_negative_mask_audit=hard_negative_mask_audit,
    )


@timed
def _build_mnrl_training_triples(
    train_pos: np.ndarray,
    train_neg: np.ndarray,
    *,
    mask_audit: list[dict] | None,
    hard_negative_mask_audit: list[dict] | None,
) -> list[tuple[int, int, int]]:
    """Thin delegate: ownership lives in _PopulationBuilders."""
    return _PopulationBuilders._build_mnrl_training_triples(
        train_pos,
        train_neg,
        mask_audit=mask_audit,
        hard_negative_mask_audit=hard_negative_mask_audit,
    )


@timed
def _build_mnrl_triple_populations(
    train_pos: np.ndarray,
    train_neg: np.ndarray,
    *,
    mask_audit: list[dict] | None,
    hard_negative_mask_audit: list[dict] | None,
) -> list[str]:
    """Thin delegate: ownership lives in _PopulationBuilders."""
    return _PopulationBuilders._build_mnrl_triple_populations(
        train_pos,
        train_neg,
        mask_audit=mask_audit,
        hard_negative_mask_audit=hard_negative_mask_audit,
    )


@timed
def _mnrl_shared_positive_gtin_rows(
    triples: list[tuple[int, int, int]], row_bc: np.ndarray
) -> int:
    """Thin delegate: ownership lives in _PopulationBuilders."""
    return _PopulationBuilders._mnrl_shared_positive_gtin_rows(triples, row_bc)


@timed
def select_balanced_negatives(
    train_neg: np.ndarray,
    train_neg_sources: np.ndarray,
    neg_copy_anchors: set[int],
    target: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Thin delegate: ownership lives in _PopulationBuilders."""
    return _PopulationBuilders.select_balanced_negatives(
        train_neg,
        train_neg_sources,
        neg_copy_anchors,
        target,
        seed,
    )


class _PopulationBuilders:
    """Small SR owner of the training population/negative builders.

    Contrastive random/easy lane:
      * `_split_safe_random_negative_pairs` split-local random negatives
      * `_mix_random_easy_training_negatives` hard + deterministic easy mix

    MNRL triple lane (+ negative-population owners):
      * `_mnrl_training_triples_with_populations` triples with base/masked/
        twin population tags (the single source of both projections below)
      * `_build_mnrl_training_triples` / `_build_mnrl_triple_populations`
        the two projections consumers pin
      * `_mnrl_shared_positive_gtin_rows` positive-GTIN repeat exposure
      * `select_balanced_negatives` base-first negative subsampling

    Module-level names stay the pinned (@timed) call surface; the bodies
    here are behavior-identical (verified row-for-row).
    """

    @staticmethod
    def _sample_split_negatives(
        subset, seed: int, n_neg: int, split_rows: np.ndarray
    ) -> np.ndarray:
        """Halve the request on sampling misses; keep 1 alive once.

        Invariant: a one-pair request must survive the final feasibility
        check — target //= 2 used to turn 1 into 0 and silently discard the
        random/easy population after one sampling miss.
        """
        from core.blocking import build_pairs

        pairs_cfg = training_cfg().pairs
        target = min(int(n_neg), len(subset) * 4)
        while target:
            try:
                _, local_neg = build_pairs(
                    subset,
                    seed=seed,
                    max_pos_per_group=int(pairs_cfg.max_pos_per_group),
                    n_neg=target,
                )
                return split_rows[local_neg]
            except RuntimeError:
                if target == 1:
                    break
                target = max(1, target // 2)
        print(
            "[random-easy] WARNING: no split-safe negatives could be sampled "
            f"(requested={n_neg}, split_rows={len(split_rows)})",
            flush=True,
        )
        return np.empty((0, 2), dtype=int)

    @staticmethod
    def _split_safe_random_negative_pairs(
        df: pd.DataFrame,
        row_bc: np.ndarray,
        split_gtins: set[str],
        *,
        seed: int,
        n_neg: int,
    ) -> np.ndarray:
        """Build known-different random negatives using only one split.

        ``build_pairs`` owns the gtin validity/title-difference rules. This
        wrapper restricts its input to the requested split first, then maps the
        returned local row indices back to the training payload indices.
        """
        split_rows = np.flatnonzero(
            np.isin(row_bc[: len(df)], np.asarray(sorted(split_gtins), dtype=str))
        )
        if len(split_rows) < 2 or n_neg <= 0:
            if n_neg > 0:
                print(
                    "[random-easy] WARNING: split-safe negative sampling skipped "
                    f"(requested={n_neg}, split_rows={len(split_rows)})",
                    flush=True,
                )
            return np.empty((0, 2), dtype=int)

        return _PopulationBuilders._sample_split_negatives(
            df.iloc[split_rows].reset_index(drop=True), seed, n_neg, split_rows
        )

    @staticmethod
    def _mix_random_easy_training_negatives(
        hard_pairs: np.ndarray,
        hard_sources: np.ndarray,
        *,
        df: pd.DataFrame,
        row_bc: np.ndarray,
        train_gtins: set[str],
        seed: int,
        enabled: bool,
        ratio_to_hard: float,
        candidate_pool_size: int,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Retain hard negatives and deterministically add split-local easy ones.

        When the unique easy pool is smaller than the ratio target, deterministic
        sampling with replacement replenishes it. The returned integer is the
        unique candidate count before replenishment, useful for telemetry.
        """
        cls = _PopulationBuilders
        hard_pairs, hard_sources, target = cls._mix_inputs(
            hard_pairs, hard_sources, ratio_to_hard, candidate_pool_size
        )
        if not enabled or target == 0:
            return hard_pairs, hard_sources, 0

        candidates = _split_safe_random_negative_pairs(
            df,
            row_bc,
            train_gtins,
            seed=seed,
            n_neg=min(target, int(candidate_pool_size)),
        )
        if not len(candidates):
            return hard_pairs, hard_sources, 0
        cls._assert_still_inside_train_split(candidates, row_bc, train_gtins)

        unique_candidates = cls._unique_easy_candidates(candidates, hard_pairs)
        if not len(unique_candidates):
            print(
                "[random-easy] WARNING: candidate pool only duplicated hard negatives",
                flush=True,
            )
            return hard_pairs, hard_sources, 0
        chosen = cls._replenish(rng_seed=seed, unique_candidates=unique_candidates, target=target)
        mixed_pairs = np.vstack([hard_pairs, chosen])
        mixed_sources = np.concatenate(
            [hard_sources, np.full(target, "random_easy", dtype=object)]
        )
        return mixed_pairs, mixed_sources, len(unique_candidates)

    @staticmethod
    def _mix_inputs(hard_pairs, hard_sources, ratio_to_hard, candidate_pool_size):
        """Coerce + validate the mix inputs; return the easy-diversity target."""
        hard_pairs = np.asarray(hard_pairs, dtype=int).reshape(-1, 2)
        hard_sources = np.asarray(hard_sources, dtype=object)
        if len(hard_sources) != len(hard_pairs):
            raise ValueError("hard negative/source lengths differ")
        ratio_to_hard = float(ratio_to_hard)
        if ratio_to_hard < 0:
            raise ValueError("random/easy to hard ratio must be non-negative")
        if int(candidate_pool_size) < 1:
            raise ValueError("random/easy candidate pool size must be positive")
        return hard_pairs, hard_sources, int(np.ceil(len(hard_pairs) * ratio_to_hard))

    @staticmethod
    def _assert_still_inside_train_split(candidates, row_bc, train_gtins) -> None:
        """The mixed population may never cross the train component boundary."""
        if not pairs_in_set(candidates, row_bc, train_gtins).all():
            raise RuntimeError("random/easy training negatives crossed the train split")

    @staticmethod
    def _unique_easy_candidates(candidates: np.ndarray, hard_pairs: np.ndarray) -> np.ndarray:
        """Deduplicated candidate rows minus the ones already hard negatives."""
        normalized_hard = {tuple(pair) for pair in np.sort(hard_pairs, axis=1)}
        return np.asarray(
            [
                pair
                for pair in np.unique(np.sort(candidates, axis=1), axis=0)
                if tuple(pair) not in normalized_hard
            ],
            dtype=int,
        ).reshape(-1, 2)

    @staticmethod
    def _replenish(*, rng_seed: int, unique_candidates: np.ndarray, target: int) -> np.ndarray:
        """Deterministic (with-replacement) sampling up to the ratio target."""
        rng = np.random.default_rng(rng_seed + 1)
        return unique_candidates[
            rng.choice(
                len(unique_candidates),
                size=target,
                replace=len(unique_candidates) < target,
            )
        ]

    @staticmethod
    @staticmethod
    def _positives_by_anchor(train_pos: np.ndarray) -> dict[int, set[int]]:
        """Anchor -> its explicit positive endpoints."""
        positives_by_anchor: dict[int, set[int]] = {}
        for anchor, positive in np.asarray(train_pos, dtype=int).reshape(-1, 2):
            positives_by_anchor.setdefault(int(anchor), set()).add(int(positive))
        return positives_by_anchor

    @staticmethod
    def _copy_lineage_maps(hard_negative_mask_audit: list[dict] | None):
        """The copy->original-anchor and copy->mode maps; conflicting lineage dies."""
        original_by_copy_pair: dict[tuple[int, int], int] = {}
        mode_by_copy_pair: dict[tuple[int, int], str] = {}
        for audit in hard_negative_mask_audit or []:
            key = (int(audit["copy_payload_idx"]), int(audit["pair_payload_idx"]))
            original = int(audit["anchor_payload_idx"])
            if key in original_by_copy_pair and original_by_copy_pair[key] != original:
                raise ValueError(f"conflicting hard-negative augmentation lineage: {key}")
            original_by_copy_pair[key] = original
            mode_by_copy_pair[key] = str(audit.get("target_mode", ""))
        return original_by_copy_pair, mode_by_copy_pair

    @staticmethod
    def _negatives_by_anchor(
        train_neg: np.ndarray,
        selected_edges: set,
        hard_negative_mask_audit: list[dict] | None,
        positives_by_anchor: dict[int, set[int]],
    ) -> dict[int, list[int]]:
        """Anchor -> explicit negative endpoints, plus the counterfactual twin addendum."""
        negatives_by_anchor: dict[int, list[int]] = {}
        for anchor, negative in np.asarray(train_neg, dtype=int).reshape(-1, 2):
            negatives_by_anchor.setdefault(int(anchor), []).append(int(negative))
        for audit in hard_negative_mask_audit or []:
            if audit.get('target_mode') == 'counterfactual':
                source, pair, copy = (int(audit[key]) for key in ('anchor_payload_idx','pair_payload_idx','copy_payload_idx'))
                if (copy,pair) in selected_edges and pair in positives_by_anchor.get(source,set()):
                    negatives_by_anchor.setdefault(source,[]).append(copy)
        return negatives_by_anchor

    @staticmethod
    def _base_mnrl_candidates(
        train_neg: np.ndarray,
        positives_by_anchor: dict[int, set[int]],
        original_by_copy_pair: dict[tuple[int, int], int],
        mode_by_copy_pair: dict[tuple[int, int], str],
    ) -> list[tuple[tuple[int, int, int], str]]:
        """Source-anchored triples for every train negative with a positive.

        Counterfactual twin rows are skipped here (source-anchored branches
        below); swap_values replays the transplant onto the source positive
        and registers the result against the COPY anchor."""
        candidates: list[tuple[tuple[int, int, int], str]] = []
        for anchor, negative in np.asarray(train_neg, dtype=int).reshape(-1, 2):
            anchor_i, negative_i = int(anchor), int(negative)
            mode = mode_by_copy_pair.get((anchor_i, negative_i))
            if mode == "counterfactual":
                # Counterfactual twins get source-anchored triples below; treating
                # the twin copy as an anchor would inherit an incompatible positive.
                continue
            if mode == "swap_values":
                # An anchor-only transplant CONTRADICTS the source's unchanged
                # positive, so that positive is not a valid target for this row.
                # TIER 1(a) replays the same transplant onto the source positive
                # and registers the result against the COPY anchor, so the copy
                # has a compatible positive of its own. With no counterpart the row
                # is still omitted rather than trained against a false match.
                positives = sorted(positives_by_anchor.get(anchor_i, ()))
            else:
                source_anchor = original_by_copy_pair.get(
                    (anchor_i, negative_i), anchor_i
                )
                positives = sorted(positives_by_anchor.get(source_anchor, ()))
            positive_i = positives[0] if positives else None
            if positive_i is None or positive_i == negative_i:
                continue
            triple = (anchor_i, positive_i, negative_i)
            # A hard-negative copy anchor means the triple's negative is a
            # masked/swap augmentation; otherwise it is an organic base triple.
            population = (
                "masked"
                if (anchor_i, negative_i) in original_by_copy_pair
                else "base"
            )
            candidates.append((triple, population))
        return candidates

    @staticmethod
    def _masked_counterpart_candidates(
        mask_audit: list[dict] | None,
        train_pos: np.ndarray,
        negatives_by_anchor: dict[int, list[int]],
    ) -> list[tuple[tuple[int, int, int], str]]:
        """Triples anchored on a masked-copy anchor with its own positive.

        Symmetric value swaps append a counterpart copy alongside the anchor
        copy: the copy's positive side is that counterpart copy, not the
        original pair side (older audits fall back to the original pair
        side). Fold membership is checked on both edges independently."""
        candidates: list[tuple[tuple[int, int, int], str]] = []
        train_positive_pairs = {
            (int(anchor), int(positive))
            for anchor, positive in np.asarray(train_pos, dtype=int).reshape(-1, 2)
        }
        for audit in mask_audit or []:
            source_i = int(audit["anchor_payload_idx"])
            copy_i = int(audit["copy_payload_idx"])
            source_positive_i = int(audit["pair_payload_idx"])
            positive_i = int(
                audit.get("copy_pair_payload_idx")
                if audit.get("copy_pair_payload_idx") is not None
                else source_positive_i
            )
            # Fold membership is checked on both edges independently: the
            # original source pair licenses the augmentation, while a symmetric
            # swap's generated counterpart must itself survive in this fold.
            if (source_i, source_positive_i) not in train_positive_pairs:
                continue
            if (copy_i, positive_i) not in train_positive_pairs:
                continue
            negative_i = next(
                (
                    negative
                    for negative in negatives_by_anchor.get(source_i, [])
                    if negative not in {source_positive_i, positive_i}
                ),
                None,
            )
            if negative_i is None:
                continue
            candidates.append(((copy_i, positive_i, negative_i), "masked"))
        return candidates

    @staticmethod
    def _twin_candidates(
        selected_negative_pairs: set,
        positives_by_anchor: dict[int, set[int]],
        hard_negative_mask_audit: list[dict] | None,
    ) -> list[tuple[tuple[int, int, int], str]]:
        """Counterfactual twin triples: source trains against its own flip.

        Twins never survive _base_mnrl_candidates: a twin row is (copy,
        pair-side) with label 0, and its source's positive IS the pair side,
        so positive_i == negative_i skips it — silently dropping every twin
        from training (they would linger in eval/diet only). Twins train as
        explicit negatives of their own source: (source, pair-side, copy),
        i.e. "the original matches its canonical better than its one-flip
        twin". That is the counterfactual pressure; without this branch the
        twin lane mints evaluation rows that never see a gradient."""
        candidates: list[tuple[tuple[int, int, int], str]] = []
        for audit in hard_negative_mask_audit or []:
            if audit.get("target_mode") != "counterfactual":
                continue
            source_i = int(audit["anchor_payload_idx"])
            copy_i = int(audit["copy_payload_idx"])
            pair_i = int(audit["pair_payload_idx"])
            # Twins are eligible only if the twin edge survived negative
            # balancing/filtering and the exact counterpart survived positives.
            if (copy_i, pair_i) not in selected_negative_pairs:
                continue
            if pair_i not in positives_by_anchor.get(source_i, set()):
                continue
            if audit.get('copy_source_payload_idx') == pair_i:
                # Compare a canonical-derived twin to its clean canonical parent.
                # The listing remains the licensed positive, while vendor wording
                # cannot dilute the minimal attribute distinction on the negative.
                candidates.append(((pair_i, source_i, copy_i), 'twin'))
                continue
            for positive_i in sorted(positives_by_anchor.get(source_i, set())):
                candidates.append(((source_i, positive_i, copy_i), "twin"))
        return candidates

    @staticmethod
    def _mnrl_training_triples_with_populations(
        train_pos: np.ndarray,
        train_neg: np.ndarray,
        *,
        mask_audit: list[dict] | None,
        hard_negative_mask_audit: list[dict] | None,
    ) -> list[tuple[tuple[int, int, int], str]]:
        """Join explicit negatives to positives without losing augmented anchors.

        Masked/swapped copies have new payload indices; audit rows identify the
        original anchor. Keep the *copy* as the MNRL anchor. Positive copies use
        one of their source's explicit negatives, if available. Never infer a
        positive or negative from a gtin alone: that can silently mislabel.

        Each triple is paired with its training population so train-time
        per-subset loss monitoring can attribute loss to the population that
        generated the negative pressure:
          * ``base``   — ordinary source-anchored triples (no augmentation copy)
          * ``masked`` — triples anchored on a masked/swap hard-negative copy
          * ``twin``   — counterfactual twin negatives trained against their source

        Invariant: the four generators emit candidates in EXACTLY the old
        in-loop order; the single dedup pass below reproduces the old
        `if triple not in seen` behavior row-for-row (seen was only updated
        on append, and no candidate generation depended on it).
        """
        cls = _PopulationBuilders
        positives_by_anchor = cls._positives_by_anchor(train_pos)
        original_by_copy_pair, mode_by_copy_pair = cls._copy_lineage_maps(
            hard_negative_mask_audit
        )
        selected_edges = set(
            map(tuple, np.asarray(train_neg, dtype=int).reshape(-1, 2).tolist())
        )
        negatives_by_anchor = cls._negatives_by_anchor(
            train_neg, selected_edges, hard_negative_mask_audit, positives_by_anchor
        )
        candidates = [
            *cls._base_mnrl_candidates(
                train_neg, positives_by_anchor, original_by_copy_pair, mode_by_copy_pair
            ),
            *cls._masked_counterpart_candidates(
                mask_audit, train_pos, negatives_by_anchor
            ),
            *cls._twin_candidates(
                selected_edges, positives_by_anchor, hard_negative_mask_audit
            ),
        ]
        triples: list[tuple[tuple[int, int, int], str]] = []
        seen: set[tuple[int, int, int]] = set()
        for candidate in candidates:
            if candidate[0] not in seen:
                seen.add(candidate[0])
                triples.append(candidate)
        return triples

    @staticmethod
    def _build_mnrl_training_triples(
        train_pos: np.ndarray,
        train_neg: np.ndarray,
        *,
        mask_audit: list[dict] | None,
        hard_negative_mask_audit: list[dict] | None,
    ) -> list[tuple[int, int, int]]:
        """Join explicit negatives to positives without losing augmented anchors."""
        return [
            triple
            for triple, _population in _mnrl_training_triples_with_populations(
                train_pos,
                train_neg,
                mask_audit=mask_audit,
                hard_negative_mask_audit=hard_negative_mask_audit,
            )
        ]

    @staticmethod
    def _build_mnrl_triple_populations(
        train_pos: np.ndarray,
        train_neg: np.ndarray,
        *,
        mask_audit: list[dict] | None,
        hard_negative_mask_audit: list[dict] | None,
    ) -> list[str]:
        """Per-triple training population (base/masked/twin) for MNRL monitoring.

        Returned in the same order as ``_build_mnrl_training_triples`` so the
        i-th population tag attributes the i-th triple.
        """
        return [
            population
            for _triple, population in _mnrl_training_triples_with_populations(
                train_pos,
                train_neg,
                mask_audit=mask_audit,
                hard_negative_mask_audit=hard_negative_mask_audit,
            )
        ]

    @staticmethod
    def _mnrl_shared_positive_gtin_rows(
        triples: list[tuple[int, int, int]], row_bc: np.ndarray
    ) -> int:
        """Count triple rows whose positive GTIN occurs in another triple.

        The no-duplicate-text sampler cannot protect nonidentical payloads for
        the same product from becoming in-batch negatives. This is an exposure
        count, not the number actually colliding in a shuffled batch.
        """
        from collections import Counter

        gtins = [str(row_bc[positive]) for _, positive, _ in triples]
        counts = Counter(gtins)
        return sum(counts[gtin] > 1 for gtin in gtins)

    @staticmethod
    def select_balanced_negatives(
        train_neg: np.ndarray,
        train_neg_sources: np.ndarray,
        neg_copy_anchors: set[int],
        target: int,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        """Subsample negatives to ``target`` rows, keeping every base row first.

        Augmented copies are trimmed before real base pairs — never the reverse:
        a neat 1.000 ratio must not cost organic data. Only when the base pool
        alone exceeds the target is the base itself trimmed (reported, not
        silent). Deterministic in ``seed``. Returns (selected, sources,
        n_discarded_base, n_discarded_aug).
        """
        pairs = np.asarray(train_neg, dtype=int).reshape(-1, 2)
        sources = np.asarray(train_neg_sources, dtype=object)
        if len(pairs) != len(sources):
            raise ValueError("negative/source lengths differ")
        if target < 0:
            raise ValueError("balance target must be non-negative")
        if target >= len(pairs):
            return pairs, sources, 0, 0
        rng = np.random.default_rng(seed)
        base_idx = np.array(
            [i for i, (a, _b) in enumerate(pairs) if int(a) not in neg_copy_anchors],
            dtype=int,
        )
        aug_idx = np.array(
            [i for i, (a, _b) in enumerate(pairs) if int(a) in neg_copy_anchors],
            dtype=int,
        )
        if len(base_idx) > target:
            keep_base = rng.choice(base_idx, size=target, replace=False)
            keep_aug: np.ndarray = np.empty(0, dtype=int)
        else:
            keep_base = base_idx
            need = target - len(keep_base)
            keep_aug = (
                rng.choice(aug_idx, size=min(need, len(aug_idx)), replace=False)
                if need > 0 and len(aug_idx)
                else np.empty(0, dtype=int)
            )
        keep = np.concatenate([keep_base, keep_aug])
        return (
            pairs[keep],
            sources[keep],
            int(len(base_idx) - len(keep_base)),
            int(len(aug_idx) - len(keep_aug)),
        )



@timed
def _make_loss(
    model,
    loss: str,
    *,
    margin: float | None = None,
    structured_feature_weight: float,
    uniformity_weight: float,
    uniformity_temperature: float,
    uniformity_min_batch_size: int,
    label_smoothing: float,
    mnrl_monitoring_enabled: bool = False,
    twin_warmup_enabled: bool = False,
    twin_warmup_epochs: int = 2,
    twin_weight: float = 0.25,
    contrastive_telemetry_enabled: bool = True,
):
    """Loss factory (SSOT knobs: training.loss / training.contrastive_margin).

    mnrl  — MultipleNegativesRankingLoss: (anchor, positive[, negative])
            column dataset; in-batch negatives; ignores labels. Wrapped in
            ``_tracking_mnrl_loss`` when per-population monitoring or the
            twin-loss warmup is enabled (both disabled by default -> the
            installed loss is returned unchanged, byte-for-byte identical).
    contrastive — OnlineContrastiveLoss: (sentence1, sentence2, label=0/1)
            pairs; PER BATCH it selects hard positives (farthest pos pairs)
            and hard negatives (closest neg pairs) and computes the margin
            hinge loss only on those — hard-pair training is native to the
            loss, not a mining layer bolted on. (Owner ruling 2026-09-07:
            the lane's default objective.)
    triplet — TripletLoss (legacy mined-triplets path).
    """
    from sentence_transformers.sentence_transformer import losses

    if loss == "mnrl":
        if mnrl_monitoring_enabled or twin_warmup_enabled:
            return _tracking_mnrl_loss(
                model,
                monitoring_enabled=mnrl_monitoring_enabled,
                warmup_enabled=twin_warmup_enabled,
                warmup_epochs=int(twin_warmup_epochs),
                twin_weight=float(twin_weight),
            )
        return losses.MultipleNegativesRankingLoss(model)
    if loss == "contrastive":
        m = float(
            margin
            if margin is not None
            else _SSOT_MARGIN
        )
        return _tracking_contrastive_loss(
            model,
            margin=m,
            structured_feature_weight=structured_feature_weight,
            uniformity_weight=uniformity_weight,
            uniformity_temperature=uniformity_temperature,
            uniformity_min_batch_size=uniformity_min_batch_size,
            label_smoothing=label_smoothing,
            telemetry_enabled=contrastive_telemetry_enabled,
        )
    return losses.TripletLoss(model)


# Compatible imports for callers of the original orchestration module.
from training.losses import (
    _smoothed_contrastive_losses,
    _tracking_contrastive_loss,
    _tracking_mnrl_loss,
)


@timed
def _runtime_telemetry() -> dict[str, float | int]:
    """Thin delegate: collection lives in _RuntimeTelemetry."""
    return _RuntimeTelemetry._collect()


def _format_telemetry(values: dict[str, float | int]) -> str:
    """Thin delegate: formatting lives in _RuntimeTelemetry."""
    return _RuntimeTelemetry._format(values)


def _wandb_memory_metrics(values: dict[str, float | int]) -> dict[str, float]:
    """Thin delegate: the W&B projection lives in _RuntimeTelemetry."""
    return _RuntimeTelemetry._memory_metrics(values)


class _RuntimeTelemetry:
    """Small SR owner of system telemetry facts and their two projections.

    `_collect` — process/CUDA facts (pid, rss, host memory, GPU allocator);
    `_format` — the pinned heartbeat console fragment;
    `_memory_metrics` — the ONLY memory fields allowed into W&B.
    """

    @staticmethod
    def _collect() -> dict[str, float | int]:
        """Cheap process and CUDA facts emitted with each training heartbeat."""
        telemetry: dict[str, float | int] = {"pid": os.getpid()}
        _RuntimeTelemetry._proc_rss(telemetry)
        _RuntimeTelemetry._host_memory(telemetry)
        _RuntimeTelemetry._cuda_fact(telemetry)
        _RuntimeTelemetry._gpu_utilization(telemetry)
        return telemetry

    @staticmethod
    def _gpu_utilization(telemetry: dict[str, float | int]) -> None:
        """TASK B item 16: NVML util/power/clocks, config-gated (off default)."""
        try:
            from core.common import training_cfg

            if not bool(training_cfg().advanced.telemetry.nvml):
                return
        except Exception:
            return
        from training.advanced import collect_nvml_telemetry

        telemetry.update(collect_nvml_telemetry())

    @staticmethod
    def _proc_rss(telemetry: dict[str, float | int]) -> None:
        """Resident-set size of this worker process (absent on OSError)."""
        try:
            for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    telemetry["rss_mb"] = round(int(line.split()[1]) / 1024, 1)
                    break
        except OSError:
            pass

    @staticmethod
    def _host_memory(telemetry: dict[str, float | int]) -> None:
        """Host total/available/used memory in MB (best effort)."""
        try:
            memory = {}
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                key, value = line.split(":", 1)
                if key in {"MemTotal", "MemAvailable"}:
                    memory[key] = int(value.strip().split()[0])
            if "MemTotal" in memory and "MemAvailable" in memory:
                telemetry.update(
                    memory_total_mb=round(memory["MemTotal"] / 1024, 1),
                    memory_available_mb=round(memory["MemAvailable"] / 1024, 1),
                    memory_used_mb=round(
                        (memory["MemTotal"] - memory["MemAvailable"]) / 1024, 1
                    ),
                )
        except (OSError, ValueError):
            pass

    @staticmethod
    def _cuda_fact(telemetry: dict[str, float | int]) -> None:
        """CUDA allocator stats — never interrupting training on failure."""
        try:
            import torch

            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                telemetry.update(
                    gpu_allocated_gb=round(torch.cuda.memory_allocated() / 1e9, 2),
                    gpu_reserved_gb=round(torch.cuda.memory_reserved() / 1e9, 2),
                    gpu_peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                    gpu_free_gb=round(free / 1e9, 2),
                    gpu_total_gb=round(total / 1e9, 2),
                )
        except Exception as exc:  # telemetry must never interrupt training
            print(f"    [telemetry] CUDA query failed: {exc}", flush=True)

    @staticmethod
    def _format(values: dict[str, float | int]) -> str:
        pieces = [f"pid={values['pid']}"]
        if "rss_mb" in values:
            pieces.append(f"rss={values['rss_mb']:.0f}MB")
        if "gpu_allocated_gb" in values:
            pieces.append(
                f"gpu={values['gpu_allocated_gb']:.2f}G alloc/"
                f"{values['gpu_reserved_gb']:.2f}G reserved/"
                f"{values['gpu_free_gb']:.2f}G free"
            )
        return " | ".join(pieces)

    @staticmethod
    def _memory_metrics(values: dict[str, float | int]) -> dict[str, float]:
        """Return the only system telemetry allowed into W&B."""
        names = {
            "rss_mb": "memory/worker_rss_mb",
            "memory_used_mb": "memory/total_used_mb",
            "memory_available_mb": "memory/total_available_mb",
            # Present only when advanced.telemetry.nvml is enabled.
            "gpu_util_pct": "gpu/util_pct",
            "power_w": "gpu/power_w",
            "sm_clock_mhz": "gpu/sm_clock_mhz",
            "mem_clock_mhz": "gpu/mem_clock_mhz",
            "temperature_c": "gpu/temperature_c",
            "gpu_memory_used_mb": "gpu/memory_used_mb",
        }
        return {
            target: float(values[source])
            for source, target in names.items()
            if source in values
        }


def _loss_trace_path(run_tag: str, fold_i: int) -> Path:
    """This fold's loss/backprop CSV (ProgressCallback's own trace target)."""
    return RESULTS / "logs" / run_tag / f"loss_backprop_fold{fold_i}.csv"


def _collapse_pair_trace_path(run_tag: str, fold_i: int) -> Path:
    """The per-pair collapse CSV the live diagnostic writes for this fold.

    Same derivation as ``_CollapseReporter._metrics`` (the loss trace's stem
    plus ``_collapse_pairs``), from the same run tag, so the consolidated trace
    can never read a different file than the diagnostic wrote.
    """
    loss = _loss_trace_path(run_tag, fold_i)
    return loss.with_name(f"{loss.stem}_collapse_pairs.csv")


class _BatchStepTrace(TrainerCallback):
    """Per-optimizer-step capture for the consolidated trace (read-only).

    The owner directive asks to follow a run at BATCH grain, not only at
    stage/step grain. HF exposes a step's geometry at ``on_step_end`` (epoch,
    step index, batch size — one optimizer step == one batch at the configured
    batch size) but NOT its loss, so this collector reads the loss from the one
    hook every SentenceTransformers loss in this lane computes through,
    ``compute_loss_from_embeddings``, by wrapping that method ON THE LOSS
    INSTANCE. The wrapper returns the original tensor untouched: it observes
    the batch loss, it never recomputes, rescales or replaces it.

    grad_norm is not handed to callbacks at all; it is recorded at log cadence
    in the per-epoch rows, and the batch rows say so explicitly instead of
    carrying a fabricated value.

    The collector does NOT write the trace (one writer per stage): it
    accumulates rows that ``train_one_config`` hands to
    ``TraceRun.add_entities`` once per call, where core.tracing's sampling caps
    bound what reaches the file.
    """

    # The trace's own caps (core/tracing.py) are the ONE volume policy; the
    # collector never picks its own budget.
    def __init__(self, *, fold_i: int, optimizer=None) -> None:
        self.fold = int(fold_i)
        self._optimizer = optimizer
        self.loss_hook_attached = False
        self.rows: list[dict[str, object]] = []
        self._step_losses: list[float] = []

    def attach(self, loss_fn) -> bool:
        """Observe each batch loss through the loss' own embeddings hook."""
        hook = getattr(loss_fn, "compute_loss_from_embeddings", None)
        if not callable(hook):
            return False

        def observed(*args, **kwargs):
            value = hook(*args, **kwargs)
            self._observe_loss(value)
            return value

        loss_fn.compute_loss_from_embeddings = observed
        self.loss_hook_attached = True
        return True

    def _observe_loss(self, value) -> None:
        """Keep the scalar the trainer actually computed; refuse anything else."""
        detached = getattr(value, "detach", None)
        try:
            item = float(detached() if callable(detached) else value)
        except (TypeError, ValueError):
            return
        if np.isfinite(item):
            self._step_losses.append(item)

    def on_step_end(self, args, state, control, **kwargs):
        """One row per completed optimizer step (== one batch of this run)."""
        losses = self._step_losses
        self._step_losses = []
        optimizer = kwargs.get("optimizer") or self._optimizer
        learning_rate = None
        if optimizer is not None and getattr(optimizer, "param_groups", None):
            learning_rate = float(optimizer.param_groups[0]["lr"])
        epoch = float(state.epoch or 0.0)
        # HF's convention: state.epoch is 1.0 at the END of the first epoch, so
        # the zero-based index of the epoch in progress is ceil(epoch) - 1.
        self.rows.append(
            {
                "fold": self.fold,
                "epoch": epoch,
                "epoch_index": max(0, int(np.ceil(epoch)) - 1),
                "global_step": int(state.global_step),
                "max_steps": int(state.max_steps or 0),
                "batch_size": int(getattr(args, "per_device_train_batch_size", 0) or 0),
                "micro_batches": len(losses),
                "loss": (float(np.mean(losses)) if losses else None),
                "loss_source": (
                    "loss.compute_loss_from_embeddings (observed, unchanged)"
                    if self.loss_hook_attached
                    else "unavailable: loss exposes no embeddings hook"
                ),
                "learning_rate": learning_rate,
            }
        )
        return control


def _emit_batch_rows(
    trace, rows: list[dict], *, source: str, total_cap: int
) -> dict[str, object]:
    """Publish the per-batch grain through the trace's sampling contract.

    ``add_entities`` censuses every bucket EXACTLY (one group row per
    fold/epoch with the real batch count) and publishes a bounded stratified
    sample of the batch rows; the accompanying ``sample_budget`` row records
    the caps that were spent. ``total_cap=0`` keeps the exact census and
    withholds the entity sample (used for sweep/selection folds, whose
    per-batch volume is bounded by the census alone).
    """
    from core.tracing import ENTITY_SAMPLE_PER_REASON, ENTITY_ROW_CAP

    cap = int(ENTITY_ROW_CAP if total_cap is None else total_cap)
    return trace.add_entities(
        "batch.record",
        rows,
        key_of=lambda r: f"fold{r['fold']}/step{r['global_step']}",
        reason_of=lambda r: f"fold{r['fold']}/epoch{r['epoch_index']}",
        detail_of=lambda r: {
            "fold": r["fold"],
            "epoch": round(float(r["epoch"]), 4),
            "global_step": r["global_step"],
            "max_steps": r["max_steps"],
            "batch_size": r["batch_size"],
            "micro_batches": r["micro_batches"],
            "loss": None if r["loss"] is None else round(float(r["loss"]), 6),
            "loss_source": r["loss_source"],
            "learning_rate": r["learning_rate"],
            "grad_norm": None,
            "grad_norm_note": (
                "the HF callback contract does not expose grad_norm per step; "
                "it is recorded at log cadence in the step/epoch rows"
            ),
        },
        source=source,
        per_reason=int(ENTITY_SAMPLE_PER_REASON),
        total_cap=cap,
    )


# ── collapse evidence: the EXACT sample and the responsible attribute ──────
_COLLAPSE_ATTRIBUTION_UNKNOWN = "attribution_unknown"


def _shared_payload_tokens(left: object, right: object) -> list[str]:
    """Tokens present in BOTH payloads (the collapse candidates)."""
    from training.uniformity import _tokens

    return sorted(_tokens(left) & _tokens(right))


def _token_document_frequency(payload: list[str], tokens: set[str]) -> dict[str, int]:
    """How many source payloads carry each of ``tokens`` (real df, no estimate)."""
    from training.uniformity import _tokens

    counts = {token: 0 for token in tokens}
    if not tokens:
        return counts
    for text in payload:
        for token in _tokens(text) & tokens:
            counts[token] += 1
    return counts


def _collapse_attribution(
    left: object,
    right: object,
    *,
    shared_attributes: dict[str, str],
    document_frequency: dict[str, int],
    frequency_limit: float,
) -> tuple[str, dict[str, object]]:
    """Name the cause of one collapse, or say plainly that it is unknown.

    Selection requires different brand/category and excludes every token whose
    document frequency exceeds ``max_token_frequency * n``, so a pair that WAS
    selected can only share tokens above that limit: those shared tokens are
    the real, checkable candidates for the collapse. When a breaching pair has
    no shared token and no shared attribute value, the cause is NOT invented.
    """
    shared = _shared_payload_tokens(left, right)
    if shared:
        # Highest document frequency wins (the most boilerplate token is the
        # strongest candidate); alphabetical tie-break keeps it deterministic.
        driver = max(shared, key=lambda token: (document_frequency.get(token, 0), token))
        detail = {
            "attribution_reason": f"shared_high_frequency_token:{driver}",
            "shared_tokens": [
                {
                    "token": token,
                    "document_frequency": int(document_frequency.get(token, 0)),
                    "frequency_limit": float(frequency_limit),
                    "above_limit": bool(document_frequency.get(token, 0) > frequency_limit),
                }
                for token in shared
            ],
        }
        return detail["attribution_reason"], detail
    if shared_attributes:
        column = sorted(shared_attributes)[0]
        detail = {
            "attribution_reason": f"shared_attribute:{column}",
            "shared_tokens": [],
            "shared_attributes": shared_attributes,
        }
        return detail["attribution_reason"], detail
    return (
        _COLLAPSE_ATTRIBUTION_UNKNOWN,
        {
            "attribution_reason": _COLLAPSE_ATTRIBUTION_UNKNOWN,
            "shared_tokens": [],
            "shared_attributes": {},
            "attribution_note": (
                "no shared payload token and no shared attribute value between "
                "the two sides; the breach is real (cosine >= operating "
                "threshold) but this diagnostic cannot attribute it to a "
                "surface token/attribute"
            ),
        },
    )


def collapse_pair_records(
    df: pd.DataFrame,
    payload,
    *,
    pair_trace_path,
    guardrail: dict,
    fold=None,
) -> list[dict]:
    """One record per BREACHING unrelated pair: exact sample + responsible cause.

    Reads the per-pair CSV the live collapse diagnostic already writes
    (``training.uniformity._write_pair_trace``: one row per scored pair with
    its cosine, its evaluation step, and BOTH sides' full source metadata) and
    returns the pairs whose cosine crossed
    ``collapse_guardrail.operating_threshold``. Each record carries the exact
    sample identity (both ``sku_id``s, both row indices, both payloads), the
    accounting (cosine vs threshold, per-observation cosines), and the
    attribution produced by :func:`_collapse_attribution`.

    Purely additive: this reads the diagnostic's own evidence, it does not
    re-encode, re-score or change any accept/reject decision.
    """
    path = Path(pair_trace_path)
    if not path.is_file():
        return []
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    threshold = float(guardrail["operating_threshold"])
    max_token_frequency = float(guardrail["max_token_frequency"])
    from training.uniformity import aligned_payload_for_diagnostics

    aligned = aligned_payload_for_diagnostics(df, payload)
    n_valid = len(aligned) or 1
    frequency_limit = max_token_frequency * n_valid

    # Aggregate every observation of a pair across the fold's evaluation steps:
    # a breach is the WORST cosine it ever reached, and the observation list
    # shows when it started collapsing.
    grouped: dict[tuple[str, str], dict] = {}
    for row in frame.to_dict("records"):
        try:
            cosine = float(row["cosine"])
        except (KeyError, TypeError, ValueError):
            continue
        left_row, right_row = row.get("a_row_index", ""), row.get("b_row_index", "")
        identity = (str(left_row), str(right_row))
        observation = {
            "evaluation_step": _collapse_step_value(row.get("evaluation_step")),
            "cosine": round(cosine, 6),
        }
        entry = grouped.setdefault(identity, {"observations": []})
        entry["observations"].append(observation)
        if "worst" not in entry or cosine > entry["worst"]:
            entry["worst"] = cosine
            entry["row"] = row
    records: list[dict] = []
    for entry in grouped.values():
        if float(entry["worst"]) < threshold:
            continue
        row = entry["row"]
        left = row.get("a_payload", "")
        right = row.get("b_payload", "")
        attributes = {}
        for column in df.columns:
            key = f"a_{column}"
            other = f"b_{column}"
            if key not in row or other not in row or column == "sku_id":
                continue
            value, peer = str(row[key]).strip(), str(row[other]).strip()
            if value and value == peer:
                attributes[str(column)] = value
        shared_tokens = _shared_payload_tokens(left, right)
        document_frequency = _token_document_frequency(aligned, set(shared_tokens))
        reason, attribution = _collapse_attribution(
            left,
            right,
            shared_attributes=attributes,
            document_frequency=document_frequency,
            frequency_limit=frequency_limit,
        )
        sku_a = str(row.get("a_sku_id", "")) or f"row{entry['row'].get('a_row_index', '')}"
        sku_b = str(row.get("b_sku_id", "")) or f"row{entry['row'].get('b_row_index', '')}"
        records.append(
            {
                "key": f"{sku_a}|{sku_b}",
                "reason": reason,
                "fold": fold,
                "sku_id1": sku_a,
                "sku_id2": sku_b,
                "row_index1": _collapse_step_value(row.get("a_row_index")),
                "row_index2": _collapse_step_value(row.get("b_row_index")),
                "cosine": round(float(entry["worst"]), 6),
                "operating_threshold": threshold,
                "crossed": True,
                "observations": entry["observations"],
                "n_observations": len(entry["observations"]),
                "payload1": left,
                "payload2": right,
                **attribution,
            }
        )
    return records


def _collapse_step_value(value):
    """Integral CSV cells as ints, anything else verbatim (never fabricated)."""
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return text


def _emit_collapse_rows(
    trace, records: list[dict], *, guardrail: dict, source: str, evidence_folds: int
) -> int:
    """Publish the collapse grain: exact census per cause + sampled entity rows.

    Reuses core.tracing's entity contract (SCOPE_ENTITY rows sampled by
    ``reason``, bounded by ENTITY_SAMPLE_PER_REASON/ENTITY_ROW_CAP), so a flood
    of collapse events stays readable while the per-cause census stays exact.
    """
    from core.tracing import ENTITY_ROW_CAP, ENTITY_SAMPLE_PER_REASON

    summary = trace.add_entities(
        "collapse.pairs",
        records,
        key_of=lambda r: r["key"],
        reason_of=lambda r: r["reason"],
        detail_of=lambda r: {
            "fold": r["fold"],
            "sku_id1": r["sku_id1"],
            "sku_id2": r["sku_id2"],
            "row_index1": r["row_index1"],
            "row_index2": r["row_index2"],
            "cosine": r["cosine"],
            "operating_threshold": r["operating_threshold"],
            "crossed": r["crossed"],
            "attribution_reason": r["reason"],
            "shared_tokens": r.get("shared_tokens", []),
            "shared_attributes": r.get("shared_attributes", {}),
            "attribution_note": r.get("attribution_note", ""),
            "n_observations": r["n_observations"],
            "observations": r["observations"],
            "payload1": r["payload1"],
            "payload2": r["payload2"],
        },
        source=source,
        per_reason=int(ENTITY_SAMPLE_PER_REASON),
        total_cap=int(ENTITY_ROW_CAP),
    )
    trace.add(
        "collapse",
        "breach_summary",
        in_count=None,
        out_count=len(records),
        reason=(
            "one entity row per breaching unrelated pair (cosine >= "
            "collapse_guardrail.operating_threshold); the census rows above "
            "name every cause bucket exactly"
        ),
        detail={
            "operating_threshold": float(guardrail["operating_threshold"]),
            "breaching_pairs": len(records),
            "buckets": summary["per_reason"],
            "sampled": summary["sampled"],
            "omitted": summary["omitted"],
            "folds_with_pair_evidence": int(evidence_folds),
            "evidence_note": (
                "read from each fold's diagnostic pair CSV "
                "(loss_backprop_fold<i>_collapse_pairs.csv); a fold with no "
                "pair evidence under the enabled guardrail was never "
                "diagnosed, which is why it is counted here explicitly"
            ),
        },
        source=source,
    )
    return len(records)


def _fold_step_events(hist: list[dict]) -> list[dict]:
    """The trainer's own log events: one per logging/eval step, with epoch+step.

    This is the source the early-stopper watched, so the trace reads it rather
    than re-deriving metrics. Its volume is bounded by the run's own
    ``logging_steps``/``eval_steps`` cadence (EVAL_STEPS_PER_EPOCH per epoch),
    not by the batch count.
    """
    events = []
    for event in hist:
        if event.get("epoch") is None:
            continue
        if event.get("loss") is None and event.get("eval_loss") is None and not any(
            key.startswith("eval_dev_cosine_") for key in event
        ):
            continue
        events.append(event)
    return events


def _trace_fold_outcome(
    trace,
    *,
    fold_i,
    hist: list[dict],
    trainer_state,
    cfg: dict,
    planned_steps: int,
    best_metric_key: str | None,
    guardrail: dict,
    collapse_records: list[dict],
) -> dict[str, object]:
    """Per-fold training rows: logging-step metrics, checkpoint, early stop.

    Returns the fold summary (real numbers, reused by the caller); every count
    comes from the trainer's own state/log history — nothing is estimated.
    """
    planned_epochs = int(cfg["epochs"])
    executed = int(trainer_state.global_step or 0)
    max_steps = int(trainer_state.max_steps or 0)
    planned = int(planned_steps or max_steps or 0)
    epochs_run = 0.0
    for event in hist:
        if event.get("epoch") is not None:
            epochs_run = max(epochs_run, float(event["epoch"]))
    events = _fold_step_events(hist)
    for event in events:
        step = event.get("step")
        eval_metrics = {
            key: number
            for key, value in event.items()
            if key.startswith("eval_")
            for number in (_optional_float(value),)
            if number is not None
        }
        trace.add(
            "step",
            "metrics",
            scope=SCOPE_ENTITY,
            key=f"fold{fold_i}/step{step}",
            in_count=None,
            out_count=None,
            reason=(
                "the trainer's own log event (the source the early-stopper "
                "watched); one row per logging/eval step"
            ),
            detail={
                "fold": int(fold_i),
                "step": None if step is None else int(step),
                "epoch": round(float(event["epoch"]), 4),
                "train_loss": _optional_float(event.get("loss")),
                "learning_rate": _optional_float(event.get("learning_rate")),
                "grad_norm": _optional_float(event.get("grad_norm")),
                **eval_metrics,
            },
            source="hf trainer state.log_history",
        )
    epochs_saved = max(0.0, float(planned_epochs) - epochs_run)
    # Counts for the two epoch/step funnels. ``ceil`` of HF's fractional epoch
    # is the completed-epoch count; the entrance is the LARGER of planned and
    # observed so a `dropped_count` can never go negative (core.schemas.TraceRow
    # requires ge=0 and the derived drop must mean attrition, never overshoot).
    epochs_completed = int(np.ceil(epochs_run)) if epochs_run else 0
    epoch_funnel_in = max(planned_epochs, epochs_completed)
    step_funnel_in = max(planned, executed)
    trace.add(
        "fold",
        "early_stop",
        scope=SCOPE_ENTITY,
        key=fold_i,
        in_count=epoch_funnel_in,
        out_count=epochs_completed,
        reason=(
            "epochs planned vs epochs actually run; HF EarlyStoppingCallback "
            "on the dev metric ends the fold when patience is exhausted"
        ),
        detail={
            "patience": cfg.get("patience"),
            "es_threshold": cfg.get("es_threshold"),
            "epochs_planned": planned_epochs,
            "epochs_run": round(epochs_run, 4),
            "epochs_saved": round(epochs_saved, 4),
            "stopped_early": int(epochs_run < planned_epochs),
            "metric_for_best_model": best_metric_key,
        },
        source="training.training ResumableSentenceTransformerTrainer",
    )
    best_checkpoint = getattr(trainer_state, "best_model_checkpoint", None)
    trace.add(
        "checkpoint",
        "select",
        scope=SCOPE_ENTITY,
        key=fold_i,
        in_count=max(epochs_completed, 1 if best_checkpoint else 0),
        out_count=1 if best_checkpoint else 0,
        reason=(
            "checkpoint selection: load_best_model_at_end keeps the epoch that "
            "maximised the dev metric; the remaining checkpoints stay on disk"
        ),
        detail={
            "best_model_checkpoint": None if best_checkpoint is None else str(best_checkpoint),
            "best_metric": _optional_float(getattr(trainer_state, "best_metric", None)),
            "metric_for_best_model": best_metric_key,
            "global_step": executed,
            "last_epoch": round(epochs_run, 4),
        },
        source="hf trainer state.best_model_checkpoint",
    )
    trace.add(
        "fold",
        "complete",
        scope=SCOPE_ENTITY,
        key=fold_i,
        in_count=step_funnel_in,
        out_count=executed,
        reason=(
            "planned optimizer steps vs executed steps; the difference is "
            "exactly what early stopping saved"
        ),
        detail={
            "planned_steps": planned,
            "executed_steps": executed,
            "max_steps": max_steps,
            "steps_saved": max(0, planned - executed),
            "epochs_run": round(epochs_run, 4),
            "collapse_breaches": len(collapse_records),
            "collapse_breach_reasons": _collapse_reason_counts(collapse_records),
            "operating_threshold": float(guardrail["operating_threshold"]),
        },
        source="hf trainer state.global_step",
    )
    return {
        "fold": int(fold_i),
        "epochs_run": round(epochs_run, 4),
        "executed_steps": executed,
        "planned_steps": planned,
        "best_metric": _optional_float(getattr(trainer_state, "best_metric", None)),
        "breaches": len(collapse_records),
    }


def _optional_float(value) -> float | None:
    """A finite float, or None — never a fabricated number."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _optional_int(value) -> int | None:
    """An integral count, or None — a missing count is never a zero."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _collapse_reason_counts(records: list[dict]) -> dict[str, int]:
    """Per-cause census of a fold's breaching pairs (exact counts)."""
    counts: dict[str, int] = {}
    for record in records:
        reason = str(record["reason"])
        counts[reason] = counts.get(reason, 0) + 1
    return counts


class _CollapseReporter:
    """Owner of the collapse diagnostic wiring (uniformity lane).

    `_metrics` runs the shared unrelated-pair diagnostic on the current
    model; `_wandb_metrics` projects the result onto the live/ W&B keys —
    the two pieces ProgressCallback used to inline as private methods.
    """

    def __init__(
        self,
        *,
        model=None,
        df=None,
        payload=None,
        config=None,
        batch_size=None,
        trace_path=None,
    ):
        self.model = model
        self.df = df
        self.payload = payload
        self.config = config
        self.batch_size = batch_size
        self.trace_path = trace_path

    def _metrics(self, evaluation_step: int) -> dict[str, float | int | str]:
        """Run the shared unrelated-pair diagnostic on the current model."""
        if self.model is None:
            return {}
        from training.uniformity import collapse_diagnostics

        return collapse_diagnostics(
            model=self.model,
            df=self.df,
            payload=self.payload,
            config=self.config,
            batch_size=int(self.batch_size),
            trace_path=(
                self.trace_path.with_name(f"{self.trace_path.stem}_collapse_pairs.csv")
                if self.trace_path is not None
                else None
            ),
            evaluation_step=evaluation_step,
        )

    @staticmethod
    def _wandb_metrics(
        metrics: dict[str, float | int | str],
    ) -> dict[str, float | int | str]:
        return {
            f"live/{key}": value
            for key, value in metrics.items()
            if key != "collapse_status" and isinstance(value, (float, int))
        } | (
            {"live/collapse_status": metrics["collapse_status"]}
            if "collapse_status" in metrics
            else {}
        )


class _LiveStatusWriter:
    """Owner of the atomic worker heartbeat the Colab launcher polls."""

    def __init__(self, *, wandb_ctx=None):
        self.wandb_ctx = wandb_ctx

    def _write(self, state, event: str, **values) -> None:
        """Atomically expose a compact worker heartbeat to the Colab launcher."""
        write_worker_live_status(
            target=RESULTS / "live_status.json",
            event=event,
            step=int(state.global_step),
            max_steps=int(state.max_steps),
            epoch=float(state.epoch or 0.0),
            wandb_run_id=getattr(self.wandb_ctx, "run_id", None),
            wandb_url=getattr(self.wandb_ctx, "run_url", None),
            **values,
        )


class _LossTraceJournal:
    """Owner of the loss/trace CSV: one row per real step or evaluation event."""

    def __init__(self, trace_path: Path | None):
        self.trace_path = trace_path
        self._rows: list[dict[str, object]] = []

    def record_loss_step(
        self, state, loss: float, grad_norm, loss_stats: dict
    ) -> None:
        self._rows.append(
            {
                "step": float(state.global_step),
                "epoch": float(state.epoch or 0.0),
                "train_loss": loss,
                "grad_norm": grad_norm,
                **loss_stats,
            }
        )

    def record_collapse_event(
        self, state, collapse_metrics: dict[str, float | int | str]
    ) -> None:
        self._rows.append(
            {
                "step": float(state.global_step),
                "epoch": float(state.epoch or 0.0),
                "event": "evaluation",
                **collapse_metrics,
            }
        )

    def flush(self) -> None:
        """The on_train_end write: write-now-rotate keeps reruns honest."""
        if self.trace_path is not None and self._rows:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            # A fold owns this file; write mode keeps reruns from appending
            # stale optimizer telemetry from an earlier attempt.
            pd.DataFrame(self._rows).to_csv(self.trace_path, index=False, mode="w")
            print(f"    [loss-trace] wrote {self.trace_path}", flush=True)


class _TrainLogDispatcher:
    """Owner of the on_log body: train-loss presentation, W&B, heartbeat.

    `dev_accuracy` arrives from the evaluate presenter so both hooks read
    the exact same shared state the original callback held.
    """

    def __init__(
        self,
        *,
        tracked_loss=None,
        journal: _LossTraceJournal,
        live: _LiveStatusWriter,
        wandb_ctx=None,
    ):
        self.tracked_loss = tracked_loss
        self.journal = journal
        self.live = live
        self.wandb_ctx = wandb_ctx
        self.latest_train_loss: float | None = None

    def pop_tracking_stats(self):
        """The tracked loss's per-batch stats, when the loss opts into tracking."""
        return (
            self.tracked_loss.pop_tracking_stats()
            if self.tracked_loss is not None
            and hasattr(self.tracked_loss, "pop_tracking_stats")
            else {}
        )

    def dispatch(self, args, state, logs=None, dev_accuracy=None, **kwargs) -> None:
        if not logs or not state.is_world_process_zero:
            return
        if "loss" in logs:
            loss, loss_stats, grad_norm, grad_norm_wire = self._record(state, logs)
            telemetry = _runtime_telemetry()
            self._console(args, state, loss, dev_accuracy, telemetry)
            self._wandb_log(state, loss, loss_stats, grad_norm_wire, telemetry)
            self._heartbeat(state, loss, dev_accuracy, loss_stats, grad_norm_wire, telemetry)

    def _record(self, state, logs) -> tuple[float, dict, float, float | None]:
        """Persist the step into the journal; return its loss facts."""
        loss = float(logs["loss"])
        self.latest_train_loss = loss
        loss_stats = self.pop_tracking_stats()
        grad_norm = (
            float(logs["grad_norm"])
            if logs.get("grad_norm") is not None
            else float("nan")
        )
        grad_norm_wire = (
            float(logs["grad_norm"])
            if logs.get("grad_norm") is not None
            else None
        )
        self.journal.record_loss_step(state, loss, grad_norm, loss_stats)
        return loss, loss_stats, grad_norm, grad_norm_wire

    @staticmethod
    def _console(args, state, loss, dev_accuracy, telemetry) -> None:
        """The pinned train-loss epoch line."""
        total_epochs = float(args.num_train_epochs)
        accuracy = (
            f" | dev_acc {dev_accuracy:.4f}"
            if dev_accuracy is not None
            else ""
        )
        print(
            f"    [epoch {state.epoch:>5.2f}/{total_epochs:g} | step {state.global_step:>4}/"
            f"{state.max_steps:<4}] train_loss {loss:.4f}{accuracy} | {_format_telemetry(telemetry)}",
            flush=True,
        )

    def _wandb_log(self, state, loss, loss_stats, grad_norm_wire, telemetry) -> None:
        """The live/ W&B projection of a training log event."""
        if self.wandb_ctx is None:
            return
        self.wandb_ctx.log_metrics(
            {
                "live/train_loss": loss,
                "live/epoch": float(state.epoch or 0.0),
                "live/grad_norm": grad_norm_wire,
                **{
                    f"live/loss_{key}": value
                    for key, value in loss_stats.items()
                },
                **_wandb_memory_metrics(telemetry),
            },
        )

    def _heartbeat(self, state, loss, dev_accuracy, loss_stats, grad_norm_wire, telemetry) -> None:
        """The shared worker live-status event for a training log event."""
        self.live._write(
            state,
            "train",
            train_loss=loss,
            dev_accuracy=dev_accuracy,
            grad_norm=grad_norm_wire,
            **{f"loss_{key}": value for key, value in loss_stats.items()},
            **telemetry,
        )


class _DevEvaluatePresenter:
    """Owner of the on_evaluate body: the dev evaluator's metric contract,
    its presentation line, the collapse projection, W&B and the heartbeat.

    `_DEV_METRICS` comes straight from the callback class so both surfaces
    keep one declaration.
    """

    def __init__(self, *, dev_metrics: dict, journal: _LossTraceJournal, live: _LiveStatusWriter, wandb_ctx=None):
        self._DEV_METRICS = dev_metrics
        self.journal = journal
        self.live = live
        self.wandb_ctx = wandb_ctx
        self.latest_dev_accuracy: float | None = None
        self.latest_collapse_metrics: dict[str, float | int | str] = {}

    def _contract_values(self, metrics: dict) -> tuple[float, float, float, float, float]:
        """Validate the structured evaluator contract, then unpack its values."""
        missing = [key for key in self._DEV_METRICS.values() if key not in metrics]
        if missing:
            raise RuntimeError(
                "structured dev evaluator violated its metric contract; missing "
                + ", ".join(missing)
            )
        accuracy = float(metrics[self._DEV_METRICS["accuracy"]])
        ap = float(metrics[self._DEV_METRICS["average_precision"]])
        f1 = float(metrics[self._DEV_METRICS["f1"]])
        precision = float(metrics[self._DEV_METRICS["precision"]])
        recall = float(metrics[self._DEV_METRICS["recall"]])
        return accuracy, ap, f1, precision, recall

    def present(
        self,
        args,
        state,
        metrics: dict | None,
        *, collapse, latest_train_loss,
        **kwargs,
    ) -> None:
        if not metrics or not state.is_world_process_zero:
            return
        accuracy, ap, f1, precision, recall = self._contract_values(metrics)
        self.latest_dev_accuracy = accuracy
        collapse_metrics = collapse._metrics(int(state.global_step))
        self.latest_collapse_metrics = dict(collapse_metrics)
        if collapse_metrics:
            self.journal.record_collapse_event(state, collapse_metrics)
        telemetry = _runtime_telemetry()
        self._console(args, state, ap, accuracy, f1, collapse_metrics, latest_train_loss, telemetry)
        self._wandb_evaluation(state, metrics, accuracy, ap, f1, precision, recall, collapse_metrics, telemetry)
        self._heartbeat(state, metrics, latest_train_loss, accuracy, ap, f1, precision, recall, collapse_metrics, telemetry)

    @staticmethod
    def _console(args, state, ap, accuracy, f1, collapse_metrics, latest_train_loss, telemetry) -> None:
        """The pinned dev-evaluation epoch line (+ collapse status when run)."""
        parts = [
            f"dev_ap {ap:.4f}",
            f"dev_acc {accuracy:.4f}",
            f"dev_f1 {f1:.4f}",
        ]
        if collapse_metrics:
            parts.append(
                "collapse "
                f"status={collapse_metrics['collapse_status']} "
                f"median={collapse_metrics.get('collapse_median_cosine', float('nan')):.4f} "
                f"p90={collapse_metrics.get('collapse_p90_cosine', float('nan')):.4f} "
                f"std={collapse_metrics.get('collapse_cosine_std', float('nan')):.4f} "
                f"healthy={collapse_metrics['collapse_healthy']}"
            )
        parts.append(_format_telemetry(telemetry))
        if parts:
            total_epochs = float(args.num_train_epochs)
            loss = (
                f"train_loss {latest_train_loss:.4f} | "
                if latest_train_loss is not None
                else ""
            )
            print(
                f"    [epoch {state.epoch:>5.2f}/{total_epochs:g} | step "
                f"{state.global_step:>4}/{state.max_steps:<4}] {loss}" + " | ".join(parts),
                flush=True,
            )

    def _wandb_evaluation(
        self, state, metrics, accuracy, ap, f1, precision, recall,
        collapse_metrics, telemetry,
    ) -> None:
        """The live/ W&B projection of an evaluation event."""
        if self.wandb_ctx is None:
            return
        self.wandb_ctx.log_metrics(
            {
                "live/dev_loss": float(metrics["eval_loss"])
                if metrics.get("eval_loss") is not None else None,
                "live/dev_accuracy": accuracy,
                "live/dev_average_precision": ap,
                "live/dev_f1": f1,
                "live/dev_precision": precision,
                "live/dev_recall": recall,
                "live/epoch": float(state.epoch or 0.0),
                **_CollapseReporter._wandb_metrics(collapse_metrics),
                **_wandb_memory_metrics(telemetry),
            },
        )

    def _heartbeat(
        self, state, metrics, latest_train_loss, accuracy, ap, f1, precision, recall,
        collapse_metrics, telemetry,
    ) -> None:
        """The shared worker live-status event for an evaluation event."""
        self.live._write(
            state,
            "evaluation",
            train_loss=latest_train_loss,
            dev_loss=float(metrics["eval_loss"]) if metrics.get("eval_loss") is not None else None,
            dev_average_precision=ap,
            dev_accuracy=self.latest_dev_accuracy,
            dev_f1=f1,
            dev_precision=precision,
            dev_recall=recall,
            **collapse_metrics,
            **telemetry,
        )


class _EpochAdvance:
    """Owner of the on_epoch_begin body: the tracked loss's epoch wiring."""

    def __init__(self, *, tracked_loss=None):
        self.tracked_loss = tracked_loss

    def advance(self, state, control):
        epoch = int((state.epoch or 0.0)) + 1
        if self.tracked_loss is not None and hasattr(self.tracked_loss, "set_epoch"):
            self.tracked_loss.set_epoch(epoch)
        dynamic_ref = getattr(self.tracked_loss, "_dynamic_epoch_ref", None)
        if dynamic_ref is not None:
            dynamic_ref["epoch"] = epoch
        return control


class _TrainingStartHeartbeat:
    """Owner of the on_train_begin body: first heartbeat with telemetry."""

    def __init__(self, *, live: _LiveStatusWriter, wandb_ctx=None):
        self.live = live
        self.wandb_ctx = wandb_ctx

    def begin(self, state, control) -> None:
        if state.is_world_process_zero:
            telemetry = _runtime_telemetry()
            print(f"    [telemetry] training-started | {_format_telemetry(telemetry)}", flush=True)
            if self.wandb_ctx is not None:
                self.wandb_ctx.log_metrics(_wandb_memory_metrics(telemetry))
            self.live._write(state, "training-started", **telemetry)


class _LateEpochLrDecayPolicy:
    """Owns applying the SSOT late-epoch LR reduction exactly once.

    The HF scheduler still owns its normal warmup/linear schedule. At the
    configured later-epoch boundary we scale both optimizer and scheduler
    base LRs, so the reduction survives subsequent scheduler steps and is
    preserved in resumable optimizer state. The callback hooks delegate here;
    all observable state (applied / resume fingerprint / LR snapshots) lives
    on the policy and is exposed through the callback's properties.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        start_epoch_fraction: float,
        multiplier: float,
    ):
        self.enabled = bool(enabled)
        self.start_epoch_fraction = float(start_epoch_fraction)
        self.multiplier = float(multiplier)
        self.applied = False
        self.applied_epoch: float | None = None
        self.learning_rates_before: list[float] = []
        self.learning_rates_after: list[float] = []
        self.resumed_start = False
        self.global_step_at_resume = 0

    def record_resume(self, state, control):
        # D2 telemetry: whether this run STARTED from an existing checkpoint.
        self.resumed_start = bool(getattr(state, "global_step", 0))
        self.global_step_at_resume = int(getattr(state, "global_step", 0))
        return control

    def _boundary_reached(self, args, state) -> tuple[bool, float | None]:
        """Small SR gates: enabled + once + at or past the epoch boundary."""
        if not self.enabled or self.applied:
            return False, None
        current_epoch = float(state.epoch or 0.0)
        boundary = float(args.num_train_epochs) * self.start_epoch_fraction
        if current_epoch + 1e-9 < boundary:
            return False, None
        return True, boundary

    @staticmethod
    def _scale_optimizer(optimizer, multiplier: float) -> list[float]:
        before = [float(group["lr"]) for group in optimizer.param_groups]
        for group in optimizer.param_groups:
            group["lr"] = float(group["lr"]) * multiplier
            if "initial_lr" in group:
                group["initial_lr"] = float(group["initial_lr"]) * multiplier
        return before

    @staticmethod
    def _scale_scheduler(scheduler, multiplier: float) -> None:
        if scheduler is not None and hasattr(scheduler, "base_lrs"):
            scheduler.base_lrs = [
                float(lr) * multiplier for lr in scheduler.base_lrs
            ]

    def apply_at_boundary(self, args, state, control, **kwargs):
        reached, boundary = self._boundary_reached(args, state)
        if not reached:
            return control
        optimizer = kwargs.get("optimizer")
        scheduler = kwargs.get("lr_scheduler")
        if optimizer is None:
            raise RuntimeError(
                "late-epoch LR decay reached its boundary without an optimizer"
            )
        current_epoch = float(state.epoch or 0.0)
        self.learning_rates_before = self._scale_optimizer(optimizer, self.multiplier)
        self._scale_scheduler(scheduler, self.multiplier)
        self.learning_rates_after = [
            float(group["lr"]) for group in optimizer.param_groups
        ]
        self.applied = True
        self.applied_epoch = current_epoch
        print(
            f"    [optim] late-epoch LR decay applied at epoch {current_epoch:.3f} "
            f"(boundary={boundary:.3f}, multiplier={self.multiplier:.3f})",
            flush=True,
        )
        emit_timing(
            "[timing] training.late_epoch_lr applied "
            f"applied_epoch={current_epoch:.3f} boundary={boundary:.3f} "
            f"multiplier={self.multiplier:.3f} resumed_start={self.resumed_start} "
            f"global_step_at_resume={self.global_step_at_resume} "
            f"lr_before_min={min(self.learning_rates_before):.8g} "
            f"lr_after_min={min(self.learning_rates_after):.8g}"
        )
        return control


class ProgressCallback(TrainerCallback):
    """Thin hook shell over the presentation owners; behavior pinned.

    The modern Trainer path replaces 07b's log_steps=True (which wrapped the
    loss module's forward to print every batch). This is the equivalent on the
    HF contract: on_log fires at logging_steps and carries the running train
    loss; on_evaluate fires at eval_steps and carries the dev metrics the
    early-stopper is actually watching. The hook bodies delegate to
    `_TrainingStartHeartbeat` / `_EpochAdvance` / `_TrainLogDispatcher` /
    `_DevEvaluatePresenter` / `_CollapseReporter` / `_LiveStatusWriter` /
    `_LossTraceJournal`.
    """

    # The structured dev evaluator below is the sole source of these values.
    # Keep its names explicit: searching metric suffixes made a renamed or
    # incomplete evaluator look like a successful evaluation.
    _DEV_METRICS = {
        "accuracy": "eval_dev_cosine_accuracy",
        "average_precision": "eval_dev_cosine_ap",
        "f1": "eval_dev_cosine_f1",
        "precision": "eval_dev_cosine_precision",
        "recall": "eval_dev_cosine_recall",
    }

    def __init__(
        self,
        wandb_ctx=None,
        tracked_loss=None,
        trace_path=None,
        *,
        collapse_model=None,
        collapse_df=None,
        collapse_payload=None,
        collapse_config=None,
        collapse_batch_size=None,
    ):
        self.wandb_ctx = wandb_ctx
        self.tracked_loss = tracked_loss
        self.trace_path = Path(trace_path) if trace_path is not None else None
        collapse_values = (
            collapse_model,
            collapse_df,
            collapse_payload,
            collapse_config,
            collapse_batch_size,
        )
        if any(value is not None for value in collapse_values) and not all(
            value is not None for value in collapse_values
        ):
            raise ValueError(
                "collapse monitoring requires model, dataframe, payload, "
                "config, and batch size together"
            )
        self._journal = _LossTraceJournal(self.trace_path)
        self._collapse = _CollapseReporter(
            model=collapse_model,
            df=collapse_df,
            payload=collapse_payload,
            config=collapse_config,
            batch_size=collapse_batch_size,
            trace_path=self.trace_path,
        )
        self._live = _LiveStatusWriter(wandb_ctx=wandb_ctx)
        self._start = _TrainingStartHeartbeat(live=self._live, wandb_ctx=wandb_ctx)
        self._advance = _EpochAdvance(tracked_loss=tracked_loss)
        self._log = _TrainLogDispatcher(
            tracked_loss=tracked_loss,
            journal=self._journal,
            live=self._live,
            wandb_ctx=wandb_ctx,
        )
        self._evaluate = _DevEvaluatePresenter(
            dev_metrics=self._DEV_METRICS,
            journal=self._journal,
            live=self._live,
            wandb_ctx=wandb_ctx,
        )

    @property
    def latest_train_loss(self):
        return self._log.latest_train_loss

    @property
    def latest_dev_accuracy(self):
        return self._evaluate.latest_dev_accuracy

    @property
    def latest_collapse_metrics(self):
        return self._evaluate.latest_collapse_metrics

    def _collapse_metrics(self, evaluation_step: int) -> dict[str, float | int | str]:
        return self._collapse._metrics(evaluation_step)

    @staticmethod
    def _collapse_wandb_metrics(
        metrics: dict[str, float | int | str],
    ) -> dict[str, float | int | str]:
        return _CollapseReporter._wandb_metrics(metrics)

    def _write_live_status(self, state, event: str, **values) -> None:
        self._live._write(state, event, **values)

    def on_train_begin(self, args, state, control, **kwargs):
        self._start.begin(state, control)
        return control

    def on_epoch_begin(self, args, state, control, **kwargs):
        return self._advance.advance(state, control)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not state.is_world_process_zero:
            return
        self._log.dispatch(
            args, state, logs=logs, dev_accuracy=self._evaluate.latest_dev_accuracy
        )
        return None

    def on_train_end(self, args, state, control, **kwargs):
        self._journal.flush()
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or not state.is_world_process_zero:
            return
        self._evaluate.present(
            args,
            state,
            metrics,
            collapse=self._collapse,
            latest_train_loss=self._log.latest_train_loss,
        )
        return None


class LateEpochLrDecayCallback(TrainerCallback):
    """Thin hook shell over `_LateEpochLrDecayPolicy`; state is delegated.

    The policy owns the apply-once boundary logic and the D2 resume
    fingerprint; observable state (applied, applied_epoch, LR snapshots,
    resumed_start, global_step_at_resume) is exposed through properties so
    the fold-metrics reader and the D2 oracle keep their interface.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        start_epoch_fraction: float,
        multiplier: float,
    ):
        self._policy = _LateEpochLrDecayPolicy(
            enabled=enabled,
            start_epoch_fraction=start_epoch_fraction,
            multiplier=multiplier,
        )
        self.enabled = self._policy.enabled

    # Policy state surface (was direct attributes on this callback).
    @property
    def applied(self) -> bool:
        return self._policy.applied

    @property
    def applied_epoch(self):
        return self._policy.applied_epoch

    @property
    def learning_rates_before(self) -> list[float]:
        return self._policy.learning_rates_before

    @property
    def learning_rates_after(self) -> list[float]:
        return self._policy.learning_rates_after

    @property
    def resumed_start(self) -> bool:
        return self._policy.resumed_start

    @property
    def global_step_at_resume(self) -> int:
        return self._policy.global_step_at_resume

    def on_train_begin(self, args, state, control, **kwargs):
        return self._policy.record_resume(state, control)

    def on_epoch_begin(self, args, state, control, **kwargs):
        return self._policy.apply_at_boundary(args, state, control, **kwargs)


def checkpoint_publication_deferred() -> bool:
    """Thin delegate: the flag parser lives in _CheckpointPublisher."""
    return _CheckpointPublisher._publication_deferred()


class DvcCheckpointCallback(TrainerCallback):
    """Stage immutable checkpoints and publish them together at train end."""

    def __init__(self):
        self._pending: list[tuple[Path, str, Path]] = []
        self._stage_futures = []
        self._stage_executor = None

    @staticmethod
    def _snapshot(checkpoint: Path) -> Path:
        """Hard-link an immutable checkpoint before Trainer rotation can delete it."""
        import hashlib
        import shutil

        staging = RESULTS / "_checkpoint_upload_staging"
        staging.mkdir(parents=True, exist_ok=True)
        # Optuna runs trial callbacks concurrently.  Steps are only unique
        # inside one Trainer output directory, so two trials can both save
        # ``checkpoint-459``.  Preserve that output-root identity in staging
        # instead of flattening every checkpoint into one shared directory.
        key = hashlib.sha256(str(checkpoint.parent.resolve()).encode()).hexdigest()[:16]
        snapshot = staging / key / checkpoint.name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if snapshot.exists():
            shutil.rmtree(snapshot)
        try:
            shutil.copytree(checkpoint, snapshot, copy_function=os.link)
        except OSError:
            # Different filesystems cannot hard-link.  Correctness matters more
            # than the copy cost in that unusual case.
            shutil.copytree(checkpoint, snapshot)
        return snapshot

    def on_save(self, args, state, control, **kwargs):
        import os

        if not state.is_world_process_zero:
            return control
        if checkpoint_publication_deferred():
            print("    [checkpoint-dvc] deferred: publication handled after run completion", flush=True)
            return control
        if not os.environ.get("DVC_API_KEY"):
            print("    [checkpoint-dvc] skipped: DVC_API_KEY absent", flush=True)
            return control
        checkpoint_root = Path(args.output_dir)
        checkpoint = checkpoint_root / f"checkpoint-{state.global_step}"
        _make_checkpoint_tokenizer_portable(checkpoint)
        required = _required_resume_filenames()
        missing = [name for name in required if not (checkpoint / name).is_file()]
        if missing:
            raise RuntimeError(
                f"checkpoint is not resumable: {checkpoint}; "
                f"missing {', '.join(missing)}"
            )
        snapshot = self._snapshot(checkpoint)
        if os.environ.get('ER_INCREMENTAL_DVC') == '1':
            from model_tracks.incremental import ArtifactPublisher
            if not hasattr(self, '_incremental_publisher'):
                self._incremental_publisher = ArtifactPublisher(RESULTS)
            self._incremental_publisher.submit(f'checkpoint-{state.global_step}', [snapshot])
            import shutil
            shutil.rmtree(snapshot)
            return control
        from concurrent.futures import ThreadPoolExecutor
        from training.dvc_store import stage_checkpoint

        if self._stage_executor is None:
            self._stage_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="dvc-stage"
            )
        # Snapshot creation is synchronous so Trainer rotation cannot remove
        # the checkpoint. Hashing/staging runs off the training thread.
        self._stage_futures.append(
            self._stage_executor.submit(stage_checkpoint, RESULTS, snapshot)
        )
        import hashlib
        key = hashlib.sha256(str(checkpoint.parent.resolve()).encode()).hexdigest()[:16]
        self._pending.append((snapshot, f"{checkpoint.name}--{key}", checkpoint))
        print(f"    [checkpoint-dvc] added locally at step {state.global_step}; upload deferred", flush=True)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        """Push the complete checkpoint batch before reporting success."""
        if hasattr(self, '_incremental_publisher'):
            self._incremental_publisher.close()
            return control
        import shutil
        from training.dvc_store import publish_checkpoints

        pending = self._pending
        try:
            # Preserve the durability guarantee: no checkpoint batch push can
            # begin until every asynchronous local dvc add has completed.
            for future in self._stage_futures:
                future.result()
            for pointer in publish_checkpoints(RESULTS, pending, already_staged=True):
                print(f"    [checkpoint-dvc] verified -> {pointer.relative_to(RESULTS)}", flush=True)
        finally:
            for snapshot, _, _ in pending:
                native_pointer = snapshot.with_name(f"{snapshot.name}.dvc")
                shutil.rmtree(snapshot, ignore_errors=True)
                native_pointer.unlink(missing_ok=True)
                try:
                    snapshot.parent.rmdir()
                except OSError:
                    pass
            self._pending = []
            if self._stage_executor is not None:
                self._stage_executor.shutdown(wait=True)
                self._stage_executor = None
            self._stage_futures = []
        return control


class _WeightEmaCallback(TrainerCallback):
    """TASK B item 1: epoch-wise weight EMA, persisted beside each checkpoint.

    The EMA arithmetic lives in the tested ``training.advanced.EmaTracker``.
    The callback folds the model's state dict in at each epoch boundary and
    writes ``ema_state.pt`` into every checkpoint directory and the run output
    dir, so the EMA model is checkpointed and resumable. (Best-model SELECTION
    on the EMA dev AP is left to the caller; the online metric stays the
    selection signal unless EMA selection is explicitly wired.)
    """

    def __init__(self, decay: float, warmup_updates: int = 0):
        self._decay = float(decay)
        self._warmup_updates = int(warmup_updates)
        self._tracker = None

    def _unwrap(self, model):
        return getattr(model, "module", model)

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None or not state.is_world_process_zero:
            return control
        from training.advanced import EmaTracker

        if self._tracker is None:
            self._tracker = EmaTracker(
                decay=self._decay, warmup_updates=self._warmup_updates
            )
        self._tracker.update(dict(self._unwrap(model).state_dict()))
        return control

    def _write(self, destination: Path) -> None:
        if self._tracker is None:
            return
        import torch

        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._tracker.state_dict(), destination)

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self._write(
                Path(args.output_dir) / f"checkpoint-{state.global_step}" / "ema_state.pt"
            )
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self._write(Path(args.output_dir) / "ema_state.pt")
        return control

    def state_dict(self):
        return None if self._tracker is None else self._tracker.state_dict()


class FineTunedAnnRefreshCallback(TrainerCallback):
    """Refresh ANN negatives from the live fine-tuned model after saves.

    The callback starts only after a checkpoint exists.  Its pair map is read
    by the contrastive dataset transform, so refreshed pairs replace the
    reserved negative slots on subsequent batches without ever creating a
    zero-shot embedding pool.
    """

    def __init__(
        self,
        *,
        df,
        payload,
        row_gtins,
        structured_features,
        train_gtins,
        existing,
        ann_state,
        slot_ids,
        fold_i,
        run_tag,
        batch_size,
        max_seq_length,
        model,
        wandb_ctx,
    ):
        self.df = df
        self.payload = payload
        self.row_gtins = row_gtins
        self.structured_features = structured_features
        self.train_gtins = train_gtins
        self.existing = existing
        self.ann_state = ann_state
        self.slot_ids = list(slot_ids)
        self.fold_i = int(fold_i)
        self.run_tag = str(run_tag)
        self.batch_size = int(batch_size)
        self.max_seq_length = int(max_seq_length)
        self.model = model
        self.wandb_ctx = wandb_ctx
        self.last_epoch = 0

    def _record_ann_refire(self, state, completed_epoch: int, cadence: int) -> None:
        """D2 telemetry: every ANN-refresh fire with its resume fingerprint.

        A fire with ``last_epoch=0`` while ``global_step`` is past the first
        epochs is the resume-refire signature; the run timing log is the
        surface the resume decision reads.
        """
        emit_timing(
            "[timing] training.ann_refresh fired "
            f"fold={self.fold_i} epoch={completed_epoch} "
            f"last_epoch={int(self.last_epoch)} cadence={cadence} "
            f"global_step={int(getattr(state, 'global_step', 0))}"
        )

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control
        model = self.model
        if model is None:
            raise RuntimeError(
                "FineTunedAnnRefreshCallback was created without the live model"
            )
        ann_cfg = config_section("mining", "ann")
        attr_cfg = _timed_load_config(f"ann_refresh.fold{self.fold_i}")["mining"]["attribute_conflict"]
        if not bool(ann_cfg["refresh_enabled"]) and not bool(attr_cfg["enabled"]):
            return control
        epoch = float(state.epoch or 0.0)
        cadence = int(ann_cfg["refresh_every_epochs"])
        completed_epoch = int(np.floor(epoch + 1e-8))
        if completed_epoch < self.last_epoch + cadence:
            return control
        self._record_ann_refire(state, completed_epoch, cadence)
        from training.ann_refresh import (
            encode_finetuned_embeddings,
            refresh_finetuned_ann,
        )

        lo, hi = (float(x) for x in str(ann_cfg["band"]).split("-"))
        qlo, qhi = (
            float(x) for x in str(ann_cfg["score_quantiles"]).split("-")
        )
        audit_path = (
            RESULTS
            / "logs"
            / self.run_tag
            / f"ann_refresh_fold{self.fold_i}_step{state.global_step}.csv"
        )
        # Refresh cost is a plan-required operational measurement. It spans the
        # fine-tuned encode, both mining passes and the audit rewrite, so it is
        # timed around the whole refresh rather than inside any single helper.
        refresh_started = time.monotonic()
        fine_tuned_emb = encode_finetuned_embeddings(
            model,
            self.payload,
            self.structured_features,
            batch_size=self.batch_size,
            max_seq_length=self.max_seq_length,
        )
        if bool(ann_cfg["refresh_enabled"]):
            pairs, stats = refresh_finetuned_ann(
                model,
                self.df,
                self.payload,
                self.row_gtins,
                structured_features=self.structured_features,
                train_gtins=self.train_gtins,
                existing=self.existing,
                step=int(state.global_step),
                epoch=epoch,
                output_path=audit_path,
                target=int(ann_cfg["target"]),
                configured_band=(lo, hi),
                band_mode=str(ann_cfg["band_mode"]),
                k=int(ann_cfg["k"]),
                candidate_multiplier=int(ann_cfg["candidate_multiplier"]),
                score_quantiles=(qlo, qhi),
                max_per_canonical=int(ann_cfg["max_per_canonical"]),
                max_per_brand=int(ann_cfg["max_per_brand"]),
                batch_size=self.batch_size,
                max_seq_length=self.max_seq_length,
                exclude_conflicting=bool(ann_cfg["exclude_conflicting"]),
                embeddings=fine_tuned_emb,
            )
        else:
            pairs = np.empty((0, 2), dtype=int)
            stats = {
                "band_lo": lo,
                "band_hi": hi,
                "band_overlap_pct": 0.0,
                "candidate_count": 0.0,
                "candidate_median": float("nan"),
                "scores": [],
                "band_mode": str(ann_cfg["band_mode"]),
            }
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                columns=[
                    "step", "epoch", "row_a", "row_b", "gtin_a", "gtin_b",
                    "cosine", "band_lo", "band_hi", "band_mode", "source",
                ]
            ).to_csv(audit_path, index=False, mode="w")
        from core.hard_negatives import mine_attribute_conflict_negatives

        attr_lo, attr_hi = (float(x) for x in str(attr_cfg["band"]).split("-"))
        if bool(attr_cfg["enabled"]):
            attr_pairs, attr_scores = mine_attribute_conflict_negatives(
                self.df,
                self.payload,
                self.row_gtins,
                fine_tuned_emb,
                existing=self.existing,
                n_target=int(attr_cfg["target"]),
                cosine_lo=attr_lo,
                cosine_hi=attr_hi,
            )
        else:
            attr_pairs = np.empty((0, 2), dtype=int)
            attr_scores = np.empty((0,), dtype=float)
        if len(attr_pairs):
            attr_keep = pairs_in_set(
                attr_pairs, self.row_gtins, set(self.train_gtins)
            )
            attr_pairs, attr_scores = attr_pairs[attr_keep], attr_scores[attr_keep]

        # ANN and attribute-conflict candidates share the reserved dynamic
        # negative slots. Allocate those slots by configured target share,
        # then fill any unused capacity from the remaining highest scores.
        capacity = len(self.slot_ids)
        ann_scores = stats.get("scores", [])
        n_ann_pairs = len(pairs.tolist())
        if ann_scores and len(ann_scores) != n_ann_pairs:
            raise ValueError(
                "ANN refresh score count must match pair count (or be absent "
                f"for audit recovery): {len(ann_scores)} != {n_ann_pairs}"
            )
        candidates: dict[str, list[tuple[float, int, int]]] = {
            "ann_finetuned": [
                (float(score), int(a), int(b))
                for (a, b), score in zip(pairs.tolist(), ann_scores)
            ],
            "attribute_conflict": [
                (float(score), int(a), int(b))
                for (a, b), score in zip(attr_pairs.tolist(), attr_scores.tolist(), strict=True)
            ],
        }
        # refresh_finetuned_ann intentionally returns only pairs plus summary;
        # recover ANN scores from the audit output so source allocation remains
        # deterministic without encoding the model a second time.
        if candidates["ann_finetuned"] and not stats.get("scores"):
            ann_audit = pd.read_csv(audit_path)
            candidates["ann_finetuned"] = [
                (float(row.cosine), int(row.row_a), int(row.row_b))
                for row in ann_audit.itertuples(index=False)
            ]
        for source in candidates:
            candidates[source].sort(reverse=True)
        target_by_source = {
            "ann_finetuned": int(ann_cfg["target"])
            if bool(ann_cfg["refresh_enabled"])
            else 0,
            "attribute_conflict": int(attr_cfg["target"])
            if bool(attr_cfg["enabled"])
            else 0,
        }
        requested = sum(target_by_source.values())
        take_by_source = {
            source: min(
                len(candidates[source]),
                int(round(capacity * target_by_source[source] / requested))
                if requested and capacity
                else 0,
            )
            for source in candidates
        }
        selected: list[tuple[str, float, int, int]] = []
        used: set[tuple[int, int]] = set()
        for source in ("ann_finetuned", "attribute_conflict"):
            for score, a, b in candidates[source][: take_by_source[source]]:
                key = (min(a, b), max(a, b))
                if key not in used:
                    used.add(key)
                    selected.append((source, score, a, b))
        remaining = [
            (source, score, a, b)
            for source, values in candidates.items()
            for score, a, b in values
            if (min(a, b), max(a, b)) not in used
        ]
        remaining.sort(key=lambda item: -item[1])
        selected.extend(remaining[: max(0, capacity - len(selected))])
        selected = selected[:capacity]
        pairs = (
            np.asarray([(a, b) for _, _, a, b in selected], dtype=int).reshape(-1, 2)
            if selected
            else np.empty((0, 2), dtype=int)
        )
        selected_sources = [source for source, _, _, _ in selected]
        attr_audit_path = (
            RESULTS
            / "logs"
            / self.run_tag
            / f"attribute_conflict_refresh_fold{self.fold_i}_step{state.global_step}.csv"
        )
        attr_audit_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "step": int(state.global_step),
                    "epoch": float(epoch),
                    "row_a": int(a),
                    "row_b": int(b),
                    "gtin_a": str(self.row_gtins[a]),
                    "gtin_b": str(self.row_gtins[b]),
                    "cosine": float(score),
                    "source": source,
                }
                for source, score, a, b in selected
                if source == "attribute_conflict"
            ],
            columns=[
                "step", "epoch", "row_a", "row_b", "gtin_a", "gtin_b",
                "cosine", "source",
            ],
        ).to_csv(attr_audit_path, index=False, mode="w")
        # Only slots that exist in this fold can be presented. The miner may
        # find more rows than the current fold's negative population.
        pair_map = self.ann_state["pairs"]
        pair_map.clear()
        # Miner may find fewer pairs than slot capacity: the assigned subset
        # is explicit (slice to the available count) and never silently zipped.
        pair_map.update({
            int(slot): (str(self.payload[a]), str(self.payload[b]))
            for slot, (a, b) in zip(
                self.slot_ids[: len(pairs)], pairs.tolist(), strict=True
            )
        })
        feature_map = self.ann_state["structured_features"]
        feature_map.clear()
        feature_map.update(
            {
                int(slot): [
                    self.structured_features[int(a)].tolist(),
                    self.structured_features[int(b)].tolist(),
                ]
                for slot, (a, b) in zip(
                    self.slot_ids[: len(pairs)], pairs.tolist(), strict=True
                )
            }
        )
        source_map = self.ann_state["sources"]
        source_map.clear()
        source_map.update(
            {
                int(slot): source
                for slot, source in zip(
                    self.slot_ids[: len(selected_sources)],
                    selected_sources,
                    strict=True,
                )
            }
        )
        self.ann_state["version"] = int(state.global_step)
        self.ann_state["count"] = int(len(pairs))
        self.last_epoch = completed_epoch
        refresh_seconds = time.monotonic() - refresh_started
        refresh_timings = (
            RESULTS
            / "logs"
            / self.run_tag
            / f"refresh_timings_fold{self.fold_i}.json"
        )
        timings = []
        if refresh_timings.is_file():
            from core.performance import load_refresh_timings
            timings = load_refresh_timings(refresh_timings)
        timings.append(
            {
                "step": int(state.global_step),
                "epoch": float(epoch),
                "refresh_seconds": float(refresh_seconds),
                "pairs": int(len(pairs)),
                "ann_pairs": int(
                    selected_sources.count("ann_finetuned")
                ),
                "attribute_conflict_pairs": int(
                    selected_sources.count("attribute_conflict")
                ),
            }
        )
        refresh_timings.parent.mkdir(parents=True, exist_ok=True)
        refresh_timings.write_text(json.dumps(timings, indent=2) + "\n")
        print(
            f"    [ann-refresh] fold {self.fold_i}: step={state.global_step} "
            f"epoch={epoch:.2f} pairs={len(pairs):,} "
            f"ann={selected_sources.count('ann_finetuned'):,} "
            f"attribute_conflict={selected_sources.count('attribute_conflict'):,} "
            f"band={stats.get('band_lo', lo):.4f}-{stats.get('band_hi', hi):.4f} "
            f"refresh_seconds={refresh_seconds:.3f} "
            f"audit={audit_path} attr_audit={attr_audit_path}",
            flush=True,
        )
        if self.wandb_ctx is not None:
            self.wandb_ctx.log_metrics(
                {
                    "ann_refresh/step": float(state.global_step),
                    "ann_refresh/epoch": epoch,
                    "ann_refresh/pairs": float(len(pairs)),
                    "ann_refresh/ann_selected_pairs": float(
                        selected_sources.count("ann_finetuned")
                    ),
                    "ann_refresh/attribute_conflict_selected_pairs": float(
                        selected_sources.count("attribute_conflict")
                    ),
                    "ann_refresh/attribute_conflict_candidates": float(
                        len(attr_pairs)
                    ),
                    "ann_refresh/candidate_count": stats.get("candidate_count"),
                    "ann_refresh/candidate_median": stats.get("candidate_median"),
                    "ann_refresh/band_overlap_pct": stats.get("band_overlap_pct"),
                    "ann_refresh/band_lo": stats.get("band_lo"),
                    "ann_refresh/band_hi": stats.get("band_hi"),
                    "ann_refresh/refresh_seconds": float(refresh_seconds),
                },
                step=int(state.global_step),
            )
        return control


@timed
def retain_hpo_champion(
    *, model_id: str, run_tag: str, value: float, folds: list[int]
) -> bool:
    """Thin delegate: champion retention lives in _HpoStream (one owner)."""
    return _HpoStream._retain_hpo_champion(
        model_id=model_id, run_tag=run_tag, value=value, folds=folds
    )


@timed
def _discriminative_groups(
    model, base_lr: float, layer_decay: float | None = None
) -> list[dict]:
    """Per-layer LR groups: bottom embeddings get base_lr*decay^n, each encoder
    layer rises geometrically to base_lr at the top; head/pooler at full base_lr.

    Dedup by tensor id — shared/tied weights (embeddings<->pooler etc.) must
    appear in exactly one group or AdamW raises. Returns groups bottom->top.
    """
    if layer_decay is None:
        layer_decay = float(runtime("layer_decay"))  # SSOT training.layer_decay
    encoder = model[0].auto_model  # ST transformer wraps HF encoder
    n_total = encoder.config.num_hidden_layers

    def lr_at(layer_i: int) -> float:
        return base_lr * (layer_decay ** (n_total - layer_i))

    seen: set[int] = set()
    groups: list[dict] = []

    def add(params, lr: float) -> None:
        keep = [p for p in params if id(p) not in seen]
        for p in keep:
            seen.add(id(p))
        if keep:
            groups.append({"params": keep, "lr": lr})

    add(encoder.embeddings.parameters(), lr_at(0))
    for i, layer in enumerate(encoder.encoder.layer):
        add(layer.parameters(), lr_at(i + 1))
    # head = everything outside the encoder (ST pooling etc.) at full LR
    add((p for nm, p in model.named_parameters() if "auto_model" not in nm), base_lr)
    # encoder leftovers not in embeddings/layers (pooler etc.) at full LR
    add(
        (
            p
            for nm, p in encoder.named_parameters()
            if not nm.startswith("embeddings.") and ".layer." not in nm
        ),
        base_lr,
    )
    return groups


# Per-process memo of the two read-only CSV artifacts. Keyed by (path, mtime,
# size) like folds.LabeledPairsCache, so a rebound F (tests, prepared-bundle
# materialization) or an in-place regeneration still loads fresh.
_ARTIFACT_CACHE = perf_enabled("text.artifact_cache")
_CANONICAL_METADATA_CACHE: dict[tuple, dict] = {}
_GATE_LOOKUP_CACHE: dict[tuple, dict] = {}


def _artifact_cache_key(path: Path) -> tuple:
    if not _ARTIFACT_CACHE:
        return (str(path),)
    try:
        stat = Path(path).stat()
        return (str(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return (str(path),)


@timed
def _load_canonical_metadata() -> dict[str, dict]:
    cache_key = _artifact_cache_key(F["canonical_records"])
    if _ARTIFACT_CACHE:
        cached = _CANONICAL_METADATA_CACHE.get(cache_key)
        if cached is not None:
            return cached
    records = pd.read_csv(
        F["canonical_records"], dtype=str, keep_default_na=False
    )
    required = {
        "gtin",
        "canonical",
        "mode_brand",
        "mode_type",
        "mode_flavor",
        "volume_set",
        "pack_set",
        "package_type_set",
        "volume_confidence",
        "pack_confidence",
    }
    missing = required - set(records.columns)
    if missing:
        raise ValueError(
            f"canonical metadata missing columns: {sorted(missing)}"
        )
    if records["gtin"].duplicated().any():
        raise ValueError("canonical_records.csv contains duplicate GTIN rows")
    result = {
        str(row["gtin"]): row.to_dict()
        for _, row in records.iterrows()
    }
    if _ARTIFACT_CACHE:
        _CANONICAL_METADATA_CACHE[cache_key] = result
    return result


def _sku_payload_metadata(index: int, row, gtin: str, text: str) -> dict:
    from core.attribute_conflicts import sku_attribute_info

    attributes = row_metadata_text(row, "attribute", "attr")
    info = sku_attribute_info(
        row_metadata_text(row, "sku_name_eng"), attributes,
        row_metadata_text(row, "description_short_eng", "description_short_eng"),
    )
    return {
        "payload_idx": index,
        "point_kind": "sku",
        "source_payload_idx": index,
        "sku_id": row_metadata_text(row, "sku_id", "SKU_ID"),
        "gtin": gtin,
        "brand": row_metadata_text(row, "brand"),
        "sku_name_eng": row_metadata_text(row, "sku_name_eng"),
        "attribute": attributes,
        "country": row_metadata_text(row, "country"),
        "category": row_metadata_text(row, "category", "breadcrumbs_eng"),
        "volume": sorted(info["volume"]),
        "pack": sorted(info["pack"]),
        "package_type": sorted(info["package_type"]),
        "flavor": str(info["flavor"]),
        "carbonation": sorted(info["carbonation"]),
        "sweetener": sorted(info["sweetener"]),
        "pulp": sorted(info["pulp"]),
        "volume_confidence": "",
        "pack_confidence": "",
        "text": text,
    }


def _canonical_payload_metadata(
    index: int, record: dict, gtin: str, text: str
) -> dict:
    from core.attribute_conflicts import canonical_attribute_info

    info = canonical_attribute_info(record)
    return {
        "payload_idx": index,
        "point_kind": "canonical",
        "source_payload_idx": index,
        "sku_id": "",
        "gtin": gtin,
        "brand": metadata_text(record["mode_brand"]),
        "sku_name_eng": metadata_text(record["canonical"]),
        "attribute": "",
        "country": "",
        "category": metadata_text(record["mode_type"]),
        "volume": sorted(info["volume"]),
        "pack": sorted(info["pack"]),
        "package_type": sorted(info["package_type"]),
        "flavor": str(info["flavor"]),
        "carbonation": sorted(info["carbonation"]),
        "sweetener": sorted(info["sweetener"]),
        "pulp": sorted(info["pulp"]),
        "volume_confidence": metadata_text(record["volume_confidence"]),
        "pack_confidence": metadata_text(record["pack_confidence"]),
        "text": text,
    }


@timed
def _load_gate_lookup() -> dict[tuple[str, str], dict[str, object]]:
    gate_path = F["gate_results"]
    cache_key = _artifact_cache_key(gate_path)
    if _ARTIFACT_CACHE:
        cached = _GATE_LOOKUP_CACHE.get(cache_key)
        if cached is not None:
            return cached
    if not gate_path.is_file():
        raise FileNotFoundError(f"gate metadata is missing: {gate_path}")
    gates = pd.read_csv(gate_path, dtype=str, keep_default_na=False)
    required = {"gtin1", "gtin2", "gate_decision", "gate_reason", "similarity"}
    missing = required - set(gates.columns)
    if missing:
        raise ValueError(f"gate metadata missing columns: {sorted(missing)}")
    lookup: dict[tuple[str, str], dict[str, object]] = {}
    for _, row in gates.iterrows():
        value = {
            "gate_decision": str(row["gate_decision"]),
            "gate_reason": str(row["gate_reason"]),
            "gate_similarity": str(row["similarity"]),
        }
        left, right = str(row["gtin1"]), str(row["gtin2"])
        for key in ((left, right), (right, left)):
            if key in lookup and lookup[key] != value:
                raise ValueError(f"conflicting gate metadata for GTIN pair: {key}")
            lookup[key] = value
    if _ARTIFACT_CACHE:
        _GATE_LOOKUP_CACHE[cache_key] = lookup
    return lookup


@timed
def _build_payload_metadata(
    df: pd.DataFrame,
    payload: list[str],
    row_bc: np.ndarray,
    *,
    mask_audit: list[dict] | None,
    hard_negative_mask_audit: list[dict] | None,
) -> tuple[list[dict], dict[tuple[str, str], dict[str, object]]]:
    """Build endpoint metadata keyed by the payload lineage index."""
    if len(payload) != len(row_bc):
        raise ValueError(
            f"payload metadata alignment failure: {len(payload)} != {len(row_bc)}"
        )
    canonical_map = _load_canonical_metadata()
    copy_sources = {
        int(item["copy_payload_idx"]): int(item.get("copy_source_payload_idx") if item.get("copy_source_payload_idx") is not None else item["anchor_payload_idx"])
        for item in list(mask_audit or []) + list(hard_negative_mask_audit or [])
        if item.get("copy_payload_idx") is not None
        and item.get("anchor_payload_idx") is not None
    }
    for item in list(mask_audit or []) + list(hard_negative_mask_audit or []):
        if item.get("copy_pair_payload_idx") is not None:
            copy_sources[int(item["copy_pair_payload_idx"])] = int(
                item["pair_payload_idx"]
            )
    metadata: list[dict] = []
    for index, value in enumerate(row_bc):
        gtin = str(value)
        source_index = copy_sources.get(index)
        if source_index is not None:
            source = dict(metadata[source_index])
            source.update(
                payload_idx=index,
                point_kind="masked_copy",
                source_payload_idx=source_index,
                text=str(payload[index]),
            )
            metadata.append(source)
        elif index < len(df):
            metadata.append(
                _sku_payload_metadata(index, df.iloc[index], gtin, str(payload[index]))
            )
        else:
            if gtin not in canonical_map:
                raise ValueError(
                    "payload canonical has no canonical metadata: "
                    f"{gtin}"
                )
            metadata.append(
                _canonical_payload_metadata(
                    index,
                    canonical_map[gtin],
                    gtin,
                    str(payload[index]),
                )
            )
    return metadata, _load_gate_lookup()


def _pair_metadata(
    left_index: int,
    right_index: int,
    metadata: list[dict],
    gate_lookup: dict[tuple[str, str], dict[str, object]],
) -> dict:
    """Flatten endpoint and gate metadata into one traceable pair record."""
    left = metadata[int(left_index)]
    right = metadata[int(right_index)]
    fields = {
        "endpoint_a_payload_idx": int(left_index),
        "endpoint_b_payload_idx": int(right_index),
    }
    for prefix, endpoint in (("a", left), ("b", right)):
        for key in (
            "point_kind",
            "source_payload_idx",
            "sku_id",
            "gtin",
            "brand",
            "sku_name_eng",
            "attribute",
            "country",
            "category",
            "volume",
            "pack",
            "package_type",
            "flavor",
            "carbonation",
            "sweetener",
            "pulp",
            "volume_confidence",
            "pack_confidence",
        ):
            if key not in endpoint:
                raise ValueError(f"payload metadata is missing field: {key}")
            value = endpoint[key]
            fields[f"{prefix}_{key}"] = json.dumps(value) if isinstance(value, list) else value
    gate = gate_lookup.get((str(left["gtin"]), str(right["gtin"])))
    if gate is None and str(left["gtin"]) == str(right["gtin"]):
        gate = {
            "gate_decision": "same_gtin",
            "gate_reason": "same_gtin_identity",
            "gate_similarity": "",
        }
    fields.update(gate or {
        "gate_decision": "missing_gate_lookup",
        "gate_reason": "pair_not_present_in_gate_results",
        "gate_similarity": "",
    })
    return fields


@timed
def _dump_train_visibility(
    fold_i,
    s1,
    s2,
    lab,
    train_all,
    tr_negs,
    *,
    hp_in_train,
    tr_neg_sources=None,
    payload,
    row_bc,
    payload_metadata,
    gate_lookup,
    run_tag="main",
    sample=False,
) -> None:
    """EXACT train-row dump (owner directive 2026-09-07): every row the
    model ingests for this fold — literal texts, labels, gtins, and
    provenance (gate-pos / hard-positive / hard-negative). Rewritten per
    run at results/logs/{run_tag}/train_rows_fold{N}.csv (+ latest pointer
    for non-sample runs)."""
    import pandas as pd

    hp_set = {(int(a), int(b)) for a, b in hp_in_train} if hp_in_train is not None else set()

    def prov(k: int) -> str:
        if lab[k] == 1:
            a, b = int(train_all[k][0]), int(train_all[k][1])
            return "hard_positive" if (a, b) in hp_set else "gate_pos"
        return str(tr_neg_sources[k - len(train_all)]) if tr_neg_sources is not None else "hard_neg"

    rows = []
    for k, (t1, t2, l) in enumerate(zip(s1, s2, lab, strict=True)):
        a = int(train_all[k][0]) if l == 1 else int(tr_negs[k - len(train_all)][0])
        b = int(train_all[k][1]) if l == 1 else int(tr_negs[k - len(train_all)][1])
        rows.append(
            {
                "fold": fold_i,
                "row": k,
                "sentence1": t1,
                "sentence2": t2,
                "label": l,
                "provenance": prov(k),
                "gtin_a": row_bc[a],
                "gtin_b": row_bc[b],
                **_pair_metadata(a, b, payload_metadata, gate_lookup),
            }
        )
    from core.common import write_visibility_log

    write_visibility_log(
        pd.DataFrame(rows), f"train_rows_fold{fold_i}.csv", run_tag, sample
    )
    n_pos = sum(1 for r in rows if r["label"] == 1)
    n_neg = len(rows) - n_pos
    n_hp = sum(1 for r in rows if r["provenance"] == "hard_positive")
    print(
        f"    [train-visibility] fold {fold_i}: {len(rows):,} rows dumped "
        f"({n_pos:,} pos [{n_hp:,} hard-pos + {n_pos - n_hp:,} gate-pos] / "
        f"{n_neg:,} hard-neg) -> results/logs/train_rows_fold{fold_i}.csv",
        flush=True,
    )


@timed
def _build_pair_lineage(
    train_pos: np.ndarray,
    train_neg: np.ndarray,
    *,
    train_neg_sources: np.ndarray | None,
    mask_audit: list[dict] | None,
    hard_negative_mask_audit: list[dict] | None,
    payload_metadata: list[dict],
    gate_lookup: dict[tuple[str, str], dict[str, object]],
) -> list[dict]:
    """Map training pair IDs to original/masked source-pair lineage."""
    lookup: dict[tuple[int, int, int], dict] = {}
    audits = list(mask_audit or []) + list(hard_negative_mask_audit or [])
    for audit in audits:
        population = str(audit.get("population", "positive"))
        label = 1 if population == "positive" else 0
        anchor = int(audit["anchor_payload_idx"])
        target = int(audit["pair_payload_idx"])
        lineage_id = f"{population}:{anchor}:{target}"
        base = {
            "population": population,
            "lineage_id": lineage_id,
            "source_anchor_payload_idx": anchor,
            "source_pair_payload_idx": target,
            "is_masked_copy": 0,
            # static minting mode of this copy (train.py's "<source>+aug"
            # copies): carried into the usage row's lineage block so the
            # mode is recoverable per row without a training run.
            "target_mode": str(audit.get("target_mode", "")),
        }
        lookup[(label, anchor, target)] = base
        lookup[(label, int(audit["copy_payload_idx"]), target)] = {
            **base,
            "is_masked_copy": 1,
        }
        if audit.get("copy_pair_payload_idx") is not None:
            lookup[
                (
                    label,
                    int(audit["copy_payload_idx"]),
                    int(audit["copy_pair_payload_idx"]),
                )
            ] = {
                **base,
                "is_masked_copy": 1,
            }

    rows: list[dict] = []
    for label, pairs in ((1, train_pos), (0, train_neg)):
        for pair_index, (a, b) in enumerate(pairs):
            a, b = int(a), int(b)
            row = lookup.get((label, a, b))
            if row is None:
                population = (
                    "positive"
                    if label
                    else str(train_neg_sources[pair_index])
                    if train_neg_sources is not None
                    else "hard_negative"
                )
                row = {
                    "population": population,
                    "lineage_id": f"{population}:{a}:{b}",
                    "source_anchor_payload_idx": a,
                    "source_pair_payload_idx": b,
                    "is_masked_copy": 0,
                }
            rows.append(
                {
                    **dict(row),
                    **_pair_metadata(a, b, payload_metadata, gate_lookup),
                }
            )
    return rows


@timed
def _training_pair_populations(
    train_pos: np.ndarray,
    train_neg: np.ndarray,
    *,
    train_neg_sources: np.ndarray | None,
    hp_in_train: np.ndarray | None,
    mask_audit: list[dict] | None,
) -> list[str]:
    """Return the population label for every contrastive dataset pair."""
    hp_set = {
        (int(a), int(b)) for a, b in (hp_in_train if hp_in_train is not None else [])
    }
    masked_ids = {
        int(item["copy_payload_idx"])
        for item in (mask_audit or [])
        if item.get("copy_payload_idx") is not None
    }
    populations: list[str] = []
    for a, b in train_pos:
        pair = (int(a), int(b))
        if int(a) in masked_ids:
            populations.append("masked_positive")
        elif pair in hp_set:
            populations.append("hard_positive")
        else:
            populations.append("gate_positive")
    for i, _pair in enumerate(train_neg):
        populations.append(
            _base_population_tag(str(train_neg_sources[i]))
            if train_neg_sources is not None and i < len(train_neg_sources)
            else "hard_negative"
        )
    return populations


def _usage_row(
    *,
    fold_i: int,
    epoch: int,
    pair_id: int,
    population: str,
    augmentation: str,
    ann_version: int,
    presentations: int,
    lineage: dict,
) -> dict:
    """Build one datapoint_usage row.

    Shared by the presented and the synthetic ``not_presented`` rows so the
    two key sets cannot drift apart: lineage fields may only FILL a key the
    base row does not already declare (``if key not in detail``). Spreading
    lineage over the base dict instead would let its mask-audit ``population``
    ("positive"/"negative") overwrite the real pair population label.
    """
    detail = {
        "fold": int(fold_i),
        "epoch": int(epoch),
        "pair_id": int(pair_id),
        "population": str(population),
        "augmentation": str(augmentation),
        "ann_version": int(ann_version),
        "presentations": int(presentations),
        "lineage_id": lineage.get("lineage_id", ""),
        "source_anchor_payload_idx": lineage.get("source_anchor_payload_idx", ""),
        "source_pair_payload_idx": lineage.get("source_pair_payload_idx", ""),
        "is_masked_copy": int(lineage.get("is_masked_copy", 0)),
    }
    detail.update(
        {key: value for key, value in lineage.items() if key not in detail}
    )
    return detail


@timed
def _write_datapoint_usage(
    *,
    fold_i: int,
    pair_populations: list[str],
    presentation_counts: dict[tuple, int],
    pair_lineage: list[dict] | None,
    dynamic_populations: set[str] | None,
    run_tag: str,
    sample: bool,
) -> dict[str, int]:
    """Persist per-pair presentation counts and verify source coverage.

    A configured source with no rows is recorded as ``unavailable``. A
    dynamic source that has not had a chance to run before early stopping is
    recorded as ``not_reached``. A non-empty training population that receives
    zero presentations is recorded as ``missing``, warned about and counted in
    the returned ``n_missing_datapoint_populations``; the fold continues
    (audit A4 — this is no longer a hard failure).
    """
    from collections import Counter
    from core.common import write_visibility_log

    def _lineage_at(pair_id: int) -> dict:
        if pair_lineage is not None and pair_id < len(pair_lineage):
            return pair_lineage[pair_id]
        return {}

    expected = Counter(pair_populations)
    dynamic_populations = set(dynamic_populations or ())
    observed: Counter[str] = Counter()
    detailed: list[dict] = []
    by_population: dict[str, dict[str, object]] = {}
    seen_pair_ids: set[int] = set()
    for key, count in sorted(presentation_counts.items(), key=lambda item: str(item[0])):
        epoch, pair_id, population, augmentation, ann_version = key
        pair_id = int(pair_id)
        if pair_id < 0 or pair_id >= len(pair_populations):
            raise ValueError(
                f"presentation count pair_id {pair_id} is outside the pair population"
            )
        seen_pair_ids.add(pair_id)
        observed[str(population)] += int(count)
        # Coverage counters describe PRESENTATIONS, so they are aggregated
        # here and never from the synthetic zero-presentation rows below
        # (01-2: those rows made distinct_pairs_presented count pairs that
        # were never presented — "0 presentations, 1 pair presented").
        coverage = by_population.setdefault(
            str(population), {"presentations": 0, "pair_ids": set()}
        )
        coverage["presentations"] += int(count)
        coverage["pair_ids"].add(pair_id)
        detailed.append(
            _usage_row(
                fold_i=fold_i,
                epoch=epoch,
                pair_id=pair_id,
                population=population,
                augmentation=augmentation,
                ann_version=ann_version,
                presentations=count,
                lineage=_lineage_at(pair_id),
            )
        )
    for pair_id, population in enumerate(pair_populations):
        if pair_id in seen_pair_ids:
            continue
        detailed.append(
            _usage_row(
                fold_i=fold_i,
                epoch=-1,
                pair_id=pair_id,
                population=population,
                augmentation="not_presented",
                ann_version=0,
                presentations=0,
                lineage=_lineage_at(pair_id),
            )
        )
    write_visibility_log(
        pd.DataFrame(detailed),
        f"datapoint_usage_fold{fold_i}.csv",
        run_tag,
        sample,
    )
    coverage_rows: list[dict] = []
    missing: list[str] = []
    unregistered: list[str] = []
    # Derived from the registry spec: a dynamic population is "configured" only
    # while its producer says it is enabled.
    dynamic_names = set(DYNAMIC_DATAPOINT_POPULATIONS)
    configured_populations = (
        (set(KNOWN_DATAPOINT_POPULATIONS) - dynamic_names) | dynamic_populations
    )
    all_populations = configured_populations | set(expected) | set(by_population)
    for population in sorted(all_populations):
        expected_rows = int(expected.get(population, 0))
        item = by_population.get(population, {"presentations": 0, "pair_ids": set()})
        presentations = int(item["presentations"])
        # The load-bearing ladder below is unchanged (ok/missing/eval_only/
        # not_reached/unavailable). The ONLY added outcome is "unregistered",
        # and it never masks an existing signal: a name the registry does not
        # know may not be reported as "ok"/"unavailable" as if it were a
        # declared population, while `missing` and `eval_only` keep their
        # exact meaning.
        if expected_rows > 0:
            status = "ok" if presentations > 0 else "missing"
        elif population in EVAL_ONLY_DATAPOINT_POPULATIONS:
            status = "eval_only"
        elif population in dynamic_populations:
            status = "ok" if presentations > 0 else "not_reached"
        else:
            status = "ok" if presentations > 0 else "unavailable"
        registered = population in KNOWN_DATAPOINT_POPULATIONS
        if not registered:
            unregistered.append(population)
            if status not in {"missing", "eval_only"}:
                status = "unregistered"
        if status == "missing":
            missing.append(population)
        coverage_rows.append(
            {
                "fold": int(fold_i),
                "population": population,
                "registered": bool(registered),
                "expected_pairs": expected_rows,
                "presentations": presentations,
                "distinct_pairs_presented": len(item["pair_ids"]),
                "status": status,
            }
        )
    write_visibility_log(
        pd.DataFrame(coverage_rows),
        f"datapoint_type_coverage_fold{fold_i}.csv",
        run_tag,
        sample,
    )
    _assert_datapoint_coverage_identity(
        fold_i=fold_i,
        coverage_rows=coverage_rows,
        presentation_counts=presentation_counts,
        observed=observed,
        pair_populations=pair_populations,
        unregistered=unregistered,
        by_population=by_population,
    )
    if missing:
        print(
            "    [datapoint-coverage] WARNING: non-empty populations received "
            f"zero presentations: {', '.join(missing)}",
            flush=True,
        )
    print(
        f"    [datapoint-coverage] fold {fold_i}: "
        + ", ".join(
            f"{row['population']}={row['presentations']:,}"
            for row in coverage_rows
        ),
        flush=True,
    )
    return {
        **{f"n_presented_{key}": int(value) for key, value in observed.items()},
        "n_missing_datapoint_populations": int(len(missing)),
        "n_unregistered_datapoint_populations": int(len(unregistered)),
        "n_coverage_populations": int(len(coverage_rows)),
        "n_coverage_registered_populations": int(
            sum(1 for row in coverage_rows if row["registered"])
        ),
        "n_coverage_ambiguous_pair_attributions": _ambiguous_pair_attributions(
            by_population
        ),
    }


@timed
def _ambiguous_pair_attributions(by_population: dict[str, dict[str, object]]) -> int:
    """Count pairs attributed to more than one population in the same fold.

    A pair presented under two labels is not a coverage hole (both labels are
    reported), but it is a TRACEABILITY fact: the ANN refresh rewrites a
    negative's text between epochs, so one negative can be presented as
    ``attribute_conflict`` early and as ``ann_finetuned`` later. The per-fold
    identity therefore reconciles PRESENTATIONS exactly and reports pair
    attribution exclusivity as a separate, visible number instead of silently
    assuming it.
    """
    seen: dict[int, str] = {}
    ambiguous: set[int] = set()
    for population, item in by_population.items():
        for pair_id in item.get("pair_ids", set()):
            pair_id = int(pair_id)
            if seen.setdefault(pair_id, population) != population:
                ambiguous.add(pair_id)
    return len(ambiguous)


@timed
def _assert_datapoint_coverage_identity(
    *,
    fold_i: int,
    coverage_rows: list[dict],
    presentation_counts: dict[tuple, int],
    observed: dict,
    pair_populations: list[str],
    unregistered: list[str],
    by_population: dict[str, dict[str, object]],
) -> None:
    """Assert the per-fold accounting identities and fail LOUDLY on breach.

    Identity 1 (closure): the presentations booked by the trainer
    (``presentation_counts``) equal the presentations attributed by the
    coverage rows, and equal ``observed``. A row that quietly failed to
    aggregate, or a presentation keyed under a population outside the rows,
    breaks this sum.

    Identity 2 (attribution): every population carrying presentations, and
    every population the fold expected, is a REGISTERED producer population.
    An unregistered name here means provenance was lost between a producer and
    the audit, so the fold raises ``UnregisteredDatapointPopulationError``
    after the artifact is on disk.

    Identity 3 (full census): the rows cover every pair of the fold dataset —
    ``expected_pairs`` sums to ``len(pair_populations)`` — so no pair can be
    absent from the audit.
    """
    total_recorded = int(sum(int(count) for count in presentation_counts.values()))
    total_rows = int(sum(int(row["presentations"]) for row in coverage_rows))
    total_observed = int(sum(int(count) for count in observed.values()))
    if not (total_recorded == total_rows == total_observed):
        raise ValueError(
            f"datapoint-coverage identity 1 (presentation closure) broken in "
            f"fold {fold_i}: presentation_counts={total_recorded} "
            f"coverage_rows={total_rows} observed={total_observed}"
        )
    total_expected = int(sum(int(row["expected_pairs"]) for row in coverage_rows))
    if total_expected != len(pair_populations):
        raise ValueError(
            f"datapoint-coverage identity 3 (full census) broken in fold "
            f"{fold_i}: sum(expected_pairs)={total_expected} != "
            f"len(pair_populations)={len(pair_populations)}"
        )
    if unregistered:
        by_row = {str(row["population"]): row for row in coverage_rows}
        detail = "; ".join(
            f"{name!r} (expected_pairs="
            f"{int(by_row.get(name, {}).get('expected_pairs', 0))}, "
            f"presentations="
            f"{int(by_row.get(name, {}).get('presentations', 0))})"
            for name in sorted(unregistered)
        )
        raise UnregisteredDatapointPopulationError(
            f"fold {fold_i}: producer-emitted datapoint population(s) outside "
            f"DATAPOINT_POPULATION_SPEC: {detail}. Register the tag in "
            "training.training.DATAPOINT_POPULATION_SPEC (with its emitter) or "
            "fix the producer: an undeclared population is invisible to the "
            "coverage audit, and silence here is the defect. The fold coverage "
            "artifact was written before this failure so the evidence survives."
        )
    ambiguous = _ambiguous_pair_attributions(by_population)
    if ambiguous:
        print(
            f"    [datapoint-coverage] WARNING: fold {fold_i}: {ambiguous:,} "
            "pair(s) presented under more than one population label "
            "(ANN refresh re-attribution); presentations still reconcile "
            "exactly, pair-level exclusivity does not hold for these.",
            flush=True,
        )


# Roles that may legitimately label a label-0 (negative) row in the loss's
# per-pair usage rows. Derived from the registry, never a literal sub-list:
# the previous hardcoded {"gate", "attribute_conflict", "random_easy"} skipped
# every other producer and silently under-counted present/selected/backprop.
ATTRIBUTABLE_NEGATIVE_USAGE_ROLES = frozenset({"negative_source", "presented_label"})


@timed
def _negative_source_accounting(
    *,
    fold_i: int,
    tr_negs: np.ndarray,
    tr_neg_sources: np.ndarray,
    usage_rows: list[dict],
) -> dict[str, int]:
    """Per-fold census of every negative SOURCE that reached the train fold.

    Producer-side completeness: the source array (``tr_neg_sources``) is the
    authoritative provenance of the fold's negatives, so it is enumerated in
    full with ``np.unique`` — every producer tag, registered or not.

    Identities asserted here (all non-tautological):

    * N1 (closure) — the per-source counts sum to ``len(tr_negs)`` AND to
      ``len(tr_neg_sources)``. A provenance array shorter than the pair array
      would break this, so a truncated source array can no longer hide.
    * N2 (funnel) — per source, ``backprop <= selected <= present <= total``.
    * N3 (registry conformance) — every source tag is declared in
      ``DATAPOINT_POPULATION_SPEC``; a fallback tag (``hard_negative`` etc.)
      means provenance was lost. Breach raises
      ``UnregisteredDatapointPopulationError`` instead of silently dropping the
      tag from the census.
    """
    sources = sorted(
        {_base_population_tag(source) for source in np.unique(tr_neg_sources)}
    )
    totals = {source: 0 for source in sources}
    for raw_source in np.unique(tr_neg_sources):
        base = _base_population_tag(str(raw_source))
        totals[base] += int(np.sum(tr_neg_sources == raw_source))
    unregistered = sorted(
        source for source in sources if source not in KNOWN_DATAPOINT_POPULATIONS
    )
    source_coverage: dict[str, int] = {}
    for source in sources:
        source_coverage[f"n_train_neg_source_{source}"] = totals[source]

    # A usage row is attributable when its population is a registered
    # negative-source tag or a registered presentation label (ann_finetuned:
    # a negative whose TEXT was replaced by the ANN refresh, whose SOURCE
    # remains the pair's original tag). Registered positives cannot label a
    # negative row, and a fallback tag means provenance was lost.
    source_names = NEGATIVE_SOURCE_DATAPOINT_POPULATIONS
    attributable = {
        name
        for name, spec in DATAPOINT_POPULATION_SPEC.items()
        if spec["role"] in ATTRIBUTABLE_NEGATIVE_USAGE_ROLES
    }
    usage_counts: dict[str, dict[str, int]] = {
        name: {"present": 0, "selected": 0, "backprop": 0}
        for name in attributable
    }
    usage_by_label: dict[str, int] = {}
    unattributed_usage = 0
    for usage in usage_rows:
        label = str(usage.get("population", "unknown"))
        if label not in attributable:
            unattributed_usage += 1
            usage_by_label[label] = usage_by_label.get(label, 0) + 1
            continue
        # ROW counts with the same additive semantics the previous literal
        # sub-list used (one usage row per pair_id from pair_usage_rows), so
        # an existing counter can only gain producers, never change meaning.
        usage_counts[label]["present"] += int(bool(usage.get("present_count", 0)))
        usage_counts[label]["selected"] += int(bool(usage.get("hard_selected_count", 0)))
        usage_counts[label]["backprop"] += int(bool(usage.get("backprop_count", 0)))

    # Registered negative sources are always in the census (explicit zero),
    # including one whose producer is config-enabled but contributed nothing.
    for name in source_names:
        source_coverage.setdefault(f"n_train_neg_source_{name}", 0)

    denominator = int(len(tr_negs))
    funnel_breaches: list[str] = []
    for name in sorted(attributable):
        counts = usage_counts[name]
        for suffix, key in (
            ("_present", "present"),
            ("_selected", "selected"),
            ("_backprop", "backprop"),
        ):
            source_coverage[f"n_train_neg_source_{name}{suffix}"] = counts[key]
        # The funnel bound only applies to a real SOURCE tag: a presentation
        # label (ann_finetuned) replaces a negative's text without moving it
        # between sources, so its total is 0 by construction.
        bound = (
            source_coverage.setdefault(f"n_train_neg_source_{name}", 0)
            if name in source_names
            else denominator
        )
        for key in ("present", "selected", "backprop"):
            if counts[key] > bound:
                funnel_breaches.append(f"{name}: {key}={counts[key]} > total={bound}")
        if counts["selected"] > counts["present"]:
            funnel_breaches.append(
                f"{name}: selected={counts['selected']} > present={counts['present']}"
            )
        if counts["backprop"] > counts["selected"]:
            funnel_breaches.append(
                f"{name}: backprop={counts['backprop']} > selected={counts['selected']}"
            )

    for name in source_names:
        source_coverage[f"pct_train_neg_source_{name}"] = (
            source_coverage[f"n_train_neg_source_{name}"] / denominator
            if denominator
            else 0.0
        )
    source_coverage["n_train_neg_source_total"] = denominator
    source_coverage["n_train_neg_source_registered"] = sum(
        totals.get(name, 0) for name in source_names
    )
    source_coverage["n_train_neg_source_unregistered"] = sum(
        totals[name] for name in unregistered
    )
    source_coverage["n_train_neg_usage_rows_unattributed"] = int(unattributed_usage)

    if unregistered or unattributed_usage:
        # Checked BEFORE the numeric identities: when provenance is missing the
        # closure sum breaks as a CONSEQUENCE, and naming the undeclared tag is
        # the actionable cause.
        raise UnregisteredDatapointPopulationError(
            f"fold {fold_i}: negative provenance outside "
            f"DATAPOINT_POPULATION_SPEC. unregistered source tag(s)="
            f"{ {name: totals[name] for name in unregistered} }; "
            f"usage-row population label(s) not attributable to a registered "
            f"negative source={usage_by_label}. Register the tag in "
            "training.training.DATAPOINT_POPULATION_SPEC (with its emitter) or "
            "fix the producer: such rows were previously dropped from the "
            "present/selected/backprop census with no warning."
        )
    if source_coverage["n_train_neg_source_registered"] != denominator:
        raise ValueError(
            f"negative-source identity N1 (closure) broken in fold {fold_i}: "
            f"registered sources sum to "
            f"{source_coverage['n_train_neg_source_registered']} but the fold "
            f"has {denominator} negatives; unregistered="
            f"{ {name: totals[name] for name in unregistered} }"
        )
    if len(tr_neg_sources) != denominator:
        raise ValueError(
            f"negative-source identity N1 (closure) broken in fold {fold_i}: "
            f"{len(tr_neg_sources)} provenance tags for {denominator} pairs"
        )
    if funnel_breaches:
        raise ValueError(
            f"negative-source identity N2 (funnel) broken in fold {fold_i}: "
            + "; ".join(funnel_breaches)
        )

    print(
        f"    [negative-source] fold {fold_i}: "
        + " | ".join(
            f"{name}={source_coverage[f'n_train_neg_source_{name}']:,} "
            f"({source_coverage[f'pct_train_neg_source_{name}']:.2%})"
            for name in source_names
        )
        + f" of {denominator:,} train-fold negatives (no unattributed source)",
        flush=True,
    )
    return source_coverage


def _dynamic_mask_negative_transform(
    batch,
    *,
    rng,
    frac: float,
    mask_prob: float | None,
    mask_lo: float,
    mask_hi: float,
    counts: dict[int, int],
    counts_by_epoch: dict[int, dict[int, int]],
    stats_by_epoch: dict[int, dict[str, float]],
    epoch_ref: dict[str, int],
    ann_pairs: dict[int, tuple[str, str]] | None = None,
    ann_structured_features: dict[int, list[list[float]]] | None = None,
    ann_sources: dict[int, str] | None = None,
    ann_state: dict[str, object] | None = None,
    pair_populations: list[str] | None = None,
    presentation_counts: dict[tuple, int] | None = None,
    mask_audit: list[dict] | None = None,
    fold: int | None = None,
    token_lookup=None,
):
    """Freshly mask selected label-0 anchors whenever a batch is materialized."""
    from training.masking import mask_text

    transformed = {key: list(values) for key, values in batch.items()}
    ann_version = int(ann_state.get("version", 0)) if ann_state else 0
    epoch_now = int(epoch_ref["epoch"])
    for i, label in enumerate(batch["label"]):
        pair_id = int(batch["pair_id"][i])
        base_population = (
            str(pair_populations[pair_id])
            if pair_populations is not None and pair_id < len(pair_populations)
            else ("positive" if int(label) else "hard_negative")
        )
        augmentation = "none"
        if int(label) == 0 and ann_pairs:
            replacement = ann_pairs.get(pair_id)
            if replacement is not None:
                transformed["sentence1"][i] = replacement[0]
                transformed["sentence2"][i] = replacement[1]
                if ann_structured_features is not None:
                    feature_replacement = ann_structured_features.get(pair_id)
                    if feature_replacement is not None and "structured_features" in transformed:
                        transformed["structured_features"][i] = feature_replacement
                base_population = (
                    ann_sources.get(pair_id, "ann_finetuned")
                    if ann_sources is not None
                    else "ann_finetuned"
                )
        if int(label) != 0:
            if base_population == "masked_positive":
                augmentation = "static_mask"
            if presentation_counts is not None:
                key = (epoch_now, pair_id, base_population, augmentation, ann_version)
                presentation_counts[key] = presentation_counts.get(key, 0) + 1
            continue
        epoch_stats = stats_by_epoch.setdefault(
            epoch_now,
            {"negative_presented": 0.0, "masked_count": 0.0, "extent_sum": 0.0},
        )
        epoch_stats["negative_presented"] += 1.0
        if rng.random() >= frac:
            if presentation_counts is not None:
                key = (epoch_now, pair_id, base_population, augmentation, ann_version)
                presentation_counts[key] = presentation_counts.get(key, 0) + 1
            continue
        original_text = str(transformed["sentence1"][i])
        masked, _extent = mask_text(
            original_text,
            mask_prob,
            rng,
            lo=mask_lo,
            hi=mask_hi,
        )
        transformed["sentence1"][i] = masked
        if token_lookup is not None:
            token_lookup.register_generated(masked)
        augmentation = "dynamic_mask"
        epoch_stats["masked_count"] += 1.0
        epoch_stats["extent_sum"] += float(_extent)
        counts[pair_id] = counts.get(pair_id, 0) + 1
        epoch_counts = counts_by_epoch.setdefault(epoch_now, {})
        epoch_counts[pair_id] = epoch_counts.get(pair_id, 0) + 1
        if presentation_counts is not None:
            key = (epoch_now, pair_id, base_population, augmentation, ann_version)
            presentation_counts[key] = presentation_counts.get(key, 0) + 1
        if mask_audit is not None:
            mask_audit.append(
                {
                    "fold": fold,
                    "epoch": epoch_now,
                    "pair_id": pair_id,
                    "population": base_population,
                    "augmentation": augmentation,
                    "anchor_text": original_text,
                    "masked_text": masked,
                    "realized_extent": round(float(_extent), 4),
                    "configured_mask_lo": float(mask_lo),
                    "configured_mask_hi": float(mask_hi),
                    "mask_prob": mask_prob,
                }
            )
    return transformed


@timed
def _load_labeled_different_positive_pairs(
    *,
    eval_pos: np.ndarray,
    row_bc: np.ndarray,
    n_source_rows: int,
) -> np.ndarray:
    """Load one valid different-GTIN positive per source GTIN.

    ``eval_pos`` already contains the implicit exact-GTIN positives. The
    labeled-pairs artifact supplies the separately investigated
    different-GTIN positives. Calibration's proxy contract permits one truth
    candidate per SKU, so keep one deterministic labeled target per source
    GTIN; callers replace that source row's exact-GTIN pair when the pair is
    admitted to a fold-local DEV pool.

    Deterministic selection rule: source rows use the lowest source-row index
    for each source GTIN; target GTINs use lexicographic order in the sorted
    labeled artifact. This is reproducible across reruns and does not pretend
    the gate similarity is independent evidence (the artifact has no separate
    quality signal beyond its gate-derived positive label).
    """
    selection_rule = _timed_load_config("plan.rand_matching")["rand_matching"][
        "calibration_different_gtin_selection"
    ]
    if selection_rule != "lowest_source_row_lexicographic_target":
        raise ValueError(
            "unsupported calibration different-GTIN selection rule: "
            f"{selection_rule!r}"
        )
    labeled_path = RESULTS / F["labeled_pairs"]
    if not labeled_path.is_file():
        raise FileNotFoundError(
            "labeled-pairs calibration input is missing: " f"{labeled_path}"
        )
    if _ARTIFACT_CACHE:
        from training.folds import load_labeled_pairs

        labeled = check_labeled_pairs_frame(load_labeled_pairs(labeled_path))
    else:
        labeled = check_labeled_pairs_frame(
            pd.read_csv(
                labeled_path,
                dtype={"gtin1": str, "gtin2": str},
                keep_default_na=False,
            )
        )
    labels = pd.to_numeric(labeled["true_label"], errors="raise").astype(int)
    positives = labeled.loc[
        (labels == 1)
        & labeled["gtin1"].astype(str).str.strip().ne(
            labeled["gtin2"].astype(str).str.strip()
        )
    ].copy()
    positives["gtin1"] = positives["gtin1"].astype(str).str.strip()
    positives["gtin2"] = positives["gtin2"].astype(str).str.strip()
    positives = positives.sort_values(["gtin1", "gtin2"], kind="stable")

    source_by_gtin: dict[str, int] = {}
    eligible_source_rows = {
        int(source) for source in eval_pos[:, 0]
    } if len(eval_pos) else set()
    for source in sorted(eligible_source_rows):
        gtin = str(row_bc[source]).strip()
        if gtin and source < n_source_rows and gtin not in source_by_gtin:
            source_by_gtin[gtin] = source

    canonical_by_gtin: dict[str, int] = {}
    for index in range(n_source_rows, len(row_bc)):
        gtin = str(row_bc[index]).strip()
        if gtin and gtin not in canonical_by_gtin:
            canonical_by_gtin[gtin] = index

    available_source_gtins = {
        str(row.gtin1)
        for row in positives.itertuples(index=False)
        if str(row.gtin1) in source_by_gtin
        and str(row.gtin2) in canonical_by_gtin
    }

    selected: list[tuple[int, int]] = []
    selected_sources: set[int] = set()
    for row in positives.itertuples(index=False):
        source = source_by_gtin.get(str(row.gtin1))
        target = canonical_by_gtin.get(str(row.gtin2))
        if source is None or target is None or source in selected_sources:
            continue
        selected.append((source, target))
        selected_sources.add(source)

    pairs = np.asarray(selected, dtype=int).reshape(-1, 2)
    print(
        f"[calibration-positives] labeled_pairs different-GTIN rows="
        f"{len(positives):,}; source-GTINs with usable candidates="
        f"{len(available_source_gtins):,}; selected one/source-GTIN="
        f"{len(pairs):,}; selection={selection_rule}",
        flush=True,
    )
    return pairs


@timed
def _merge_different_calibration_positives(
    dev_pos: np.ndarray,
    labeled_pairs: np.ndarray,
    row_bc: np.ndarray,
    dev_bc: set[str],
) -> np.ndarray:
    """Add fold-local different-GTIN positives without duplicate truths."""
    if len(labeled_pairs) == 0:
        return dev_pos
    eligible = labeled_pairs[pairs_in_set(labeled_pairs, row_bc, dev_bc)]
    if len(eligible) == 0:
        return dev_pos

    # A source SKU already has an exact-GTIN positive in ``dev_pos``. Replace
    # that exact candidate for the selected source rows so _candidate_frame's
    # one-truth-per-SKU contract remains valid.
    replace_sources = {int(pair[0]) for pair in eligible}
    keep = np.asarray(
        [int(pair[0]) not in replace_sources for pair in dev_pos], dtype=bool
    )
    merged = np.vstack([dev_pos[keep], eligible]) if len(dev_pos[keep]) else eligible
    print(
        f"[calibration-positives] DEV merge: different-GTIN={len(eligible):,} "
        f"exact-GTIN replacements={int((~keep).sum()):,} total={len(merged):,}",
        flush=True,
    )
    return merged


@timed
def _evaluation_negative_mask(pairs: np.ndarray, n_source: int, copy_ids: set[int]) -> np.ndarray:
    """Calibration queries must be real SKU rows, with unaugmented endpoints."""
    source_rows = (pairs[:, 0] >= 0) & (pairs[:, 0] < n_source)
    return source_rows & ~np.isin(pairs, list(copy_ids)).any(axis=1)


@timed
def _prepare_objective_plan(*, loss, payload, structured_features, train_all, tr_negs,
                            tr_neg_sources, hp_pairs, row_bc, tr_bc, use_hp,
                            mask_audit, hard_negative_mask_audit, hard_train, seed, fold_i):
    """Freeze objective rows and deterministic epoch presentation indices locally."""
    from datasets import Dataset
    from training.sampler import ControlledBatchSampler, resolve_composition
    hp_tracking = hp_pairs[pairs_in_set(hp_pairs, row_bc, tr_bc)] if use_hp and hp_pairs is not None and len(hp_pairs) else None
    objective = {}
    if loss == "contrastive":
        if not len(tr_negs):
            raise ValueError("contrastive loss needs labeled training negatives")
        populations = _training_pair_populations(train_all, tr_negs, train_neg_sources=tr_neg_sources,
                                                  hp_in_train=hp_tracking, mask_audit=mask_audit)
        all_pairs = list(train_all) + list(tr_negs)
        dataset = {
            "sentence1": [payload[int(a)] for a, b in all_pairs],
            "sentence2": [payload[int(b)] for a, b in all_pairs],
            "label": [1] * len(train_all) + [0] * len(tr_negs),
            "pair_id": list(range(len(all_pairs))), "pair_population": populations,
            "structured_features": [[structured_features[int(a)].tolist(), structured_features[int(b)].tolist()]
                                     for a, b in all_pairs],
        }
        sampler_populations = ["masked_positive" if pop == "masked_positive" else
                               ("gate_positive" if label else "hard_negative")
                               for pop, label in zip(populations, dataset["label"], strict=True)]
    elif loss == "mnrl":
        triples_with_populations = _mnrl_training_triples_with_populations(
            train_all, tr_negs, mask_audit=mask_audit,
            hard_negative_mask_audit=hard_negative_mask_audit,
        )
        triples = [triple for triple, _population in triples_with_populations]
        if not triples:
            raise ValueError("MNRL needs anchor-positive-negative training triples")
        populations = [
            population for _triple, population in triples_with_populations
        ]
        balanced_policy = training_cfg().masking.balanced_augmentation
        if balanced_policy.enabled:
            from training.balanced_augmentation import balance_objective
            copy_ids = {int(row[key]) for row in [*(mask_audit or []),*(hard_negative_mask_audit or [])]
                        for key in ('copy_payload_idx','copy_pair_payload_idx') if row.get(key) is not None}
            triples, populations, balance_coverage = balance_objective(
                triples,populations,copy_ids,balanced_policy.original_objective_share)
            objective['balance_coverage'] = balance_coverage.model_dump(mode='json')
        dataset = {"anchor": [payload[a] for a, b, c in triples],
                   "positive": [payload[b] for a, b, c in triples],
                   "negative": [payload[c] for a, b, c in triples],
                   "pair_id": list(range(len(triples))), "population": populations}
        objective["triples"] = triples
        objective["shared_gtin_rows"] = _mnrl_shared_positive_gtin_rows(triples, row_bc)
        sampler_populations = populations
    else:
        from core.hard_negatives import build_triplets
        examples = build_triplets(train_all, hard_train, payload, seed=seed + fold_i, max_triples=MAX_TRIPLES)
        if not examples:
            raise ValueError("triplet training needs checkpoint-dependent mined negative inputs")
        dataset = {"anchor": [example.texts[0] for example in examples],
                   "positive": [example.texts[1] for example in examples],
                   "negative": [example.texts[2] for example in examples],
                   "pair_population": ["triplet"] * len(examples)}
        sampler_populations = dataset["pair_population"]
    objective["dataset"] = dataset
    bs_cfg = _timed_load_config(f"fold{fold_i}.bs_cfg")["training"]["batch_sampler"]
    ds = Dataset.from_dict(dataset)
    epochs = int(_timed_load_config(f"fold{fold_i}.epochs")["training"]["epochs"])
    packed = {}
    grouped_ds = None
    shared_text_hashes = None
    for device, batch_size in (("cpu", BATCH_SIZE_CPU), ("cuda", BATCH_SIZE_CUDA)):
        device_started = time.perf_counter()
        print(f"[plan-sampler] start device={device} batch_size={batch_size} rows={len(ds):,} epochs={epochs}", flush=True)
        if bs_cfg["enabled"]:
            weights = bs_cfg.get("compositions_by_loss", {}).get(loss, bs_cfg["composition"])
            composition = resolve_composition(weights, sampler_populations, batch_size)
            if grouped_ds is None:
                grouped_ds = ds.add_column("sampler_population", sampler_populations)
            if _SHARE_TEXT_HASHES:
                if shared_text_hashes is None:
                    from training.sampler import _row_text_hashes
                    shared_text_hashes = _row_text_hashes(grouped_ds)
                text_hashes = shared_text_hashes
            else:
                text_hashes = None
            sampler = ControlledBatchSampler(grouped_ds, batch_size, composition, seed=int(bs_cfg["seed"]),
                                              population_column="sampler_population",
                                              text_hashes=text_hashes)
        else:
            import torch
            from sentence_transformers.base.sampler import NoDuplicatesBatchSampler, DefaultBatchSampler
            generator = torch.Generator().manual_seed(seed + fold_i)
            if loss == "mnrl":
                # Native duplicate checks inspect every non-label column.
                # Population telemetry would otherwise force all same-population
                # rows into separate batches and destroy the in-batch pool.
                from training.token_inputs import TEXT_COLUMNS
                text_ds = ds.select_columns([name for name in ds.column_names if name in TEXT_COLUMNS])
                sampler = NoDuplicatesBatchSampler(text_ds, batch_size=batch_size, drop_last=False,
                                                   valid_label_columns=["label"], generator=generator, seed=seed+fold_i)
            else:
                sampler = DefaultBatchSampler(torch.utils.data.RandomSampler(ds, generator=generator),
                                                batch_size=batch_size, drop_last=False)
        epoch_batches = []
        for epoch in range(epochs):
            epoch_started = time.perf_counter()
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            batches = [list(map(int, batch)) for batch in sampler]
            if sorted(index for batch in batches for index in batch) != list(range(len(ds))):
                raise ValueError("local objective plan sampler did not account for every training row")
            epoch_batches.append(batches)
            print(f"[plan-sampler] device={device} epoch={epoch + 1}/{epochs} batches={len(batches):,} seconds={time.perf_counter() - epoch_started:.2f}", flush=True)
        print(f"[plan-sampler] complete device={device} seconds={time.perf_counter() - device_started:.2f}", flush=True)
        packed[device] = {"batch_size": batch_size, "epochs": epoch_batches}
    objective["sampler"] = packed
    return {"objective": objective}


@timed
def prepare_fixed_training_inputs(
    cfg, *, loss, model_id, use_hp, band, data, seed, cv_folds=None,
    folds_override=None, dev_fraction=None, dev_override=None,
    neg_pairs=None, train_neg_pairs=None, neg_pair_sources=None,
    train_neg_pair_sources=None, mask_audit=None, hard_negative_mask_audit=None,
    train_frac=None, sample=False, selection_mode=False
):
    """The single CPU implementation for fixed fold inputs and objective rows."""
    calibration_config = _timed_load_config("plan.calibration")
    _train_neg_source = (
        train_neg_pairs if train_neg_pairs is not None else neg_pairs
    )
    _eval_neg_sources = (
        np.asarray(neg_pair_sources, dtype=object)
        if neg_pair_sources is not None
        else (np.full(len(neg_pairs), "unknown", dtype=object)
              if neg_pairs is not None else np.empty(0, dtype=object))
    )
    _train_neg_sources = (
        np.asarray(train_neg_pair_sources, dtype=object)
        if train_neg_pair_sources is not None
        else (_eval_neg_sources.copy() if train_neg_pairs is None
              else np.full(len(train_neg_pairs), "unknown", dtype=object))
    )
    if neg_pairs is not None and len(_eval_neg_sources) != len(neg_pairs):
        raise ValueError(
            "neg_pair_sources must align with neg_pairs: "
            f"{len(_eval_neg_sources)} != {len(neg_pairs)}"
        )
    if _train_neg_source is not None and len(_train_neg_sources) != len(_train_neg_source):
        raise ValueError(
            "train_neg_pair_sources must align with train_neg_pairs: "
            f"{len(_train_neg_sources)} != {len(_train_neg_source)}"
        )


    df, payload, structured_features, row_bc, country, pos, hp_pairs, emb0 = data
    if len(payload) < len(df):
        raise ValueError(
            "training payload is shorter than source SKU dataframe: "
            f"payload={len(payload)} rows={len(df)}"
        )
    # The training payload intentionally appends canonical and masked-copy
    # entries after the source SKU rows. The shared uniformity boundary owns
    # the source-row alignment before selecting unrelated pairs.
    with trace_step('training.prepare_inputs.payload_metadata'):
        payload_metadata, gate_lookup = _build_payload_metadata(
            df,
            payload,
            row_bc,
            mask_audit=mask_audit,
            hard_negative_mask_audit=hard_negative_mask_audit,
        )
        # Masked positive copies are augmentation for training only.  Splits are
        # gtin-based, so passing the augmented array directly into dev/test
        # would silently put those copies into evaluation even though they carry
        # the same gtin as the original SKU.  Keep the augmented ``pos`` for
        # train-side selection, but remove copy endpoints from evaluation pools.
        _masked_copy_ids = {
            int(row["copy_payload_idx"])
            for row in (mask_audit or []) + (hard_negative_mask_audit or [])
            if row.get("copy_payload_idx") is not None
        }
        eval_pos = (
            pos[~np.isin(pos[:, 0], np.fromiter(_masked_copy_ids, dtype=int))]
            if _masked_copy_ids and len(pos)
            else pos
        )
        if _masked_copy_ids:
            print(
                f"    [masking] excluded {len(pos) - len(eval_pos):,} masked "
                "positive copies from dev/holdout evaluation; training retains them",
                flush=True,
            )
        if neg_pairs is not None and len(neg_pairs):
            # Preserve _train_neg_source and its source labels above: synthetic
            # copies train normally, but calibration needs a real source SKU.
            eval_mask = _evaluation_negative_mask(neg_pairs, len(df), _masked_copy_ids)
            excluded = int((~eval_mask).sum())
            neg_pairs = neg_pairs[eval_mask]
            _eval_neg_sources = _eval_neg_sources[eval_mask]
            if excluded:
                print(f"    [masking] excluded {excluded:,} synthetic negative views from evaluation; training retains them", flush=True)
        labeled_different_pos = _load_labeled_different_positive_pairs(
            eval_pos=eval_pos,
            row_bc=row_bc,
            n_source_rows=len(df),
        )
    all_gtin_set = set(row_bc.tolist())

    # country must cover every payload entry (canonicals + masked copies
    # appended after the sku rows); pad with "" so the cross-country mask
    # never IndexErrors no matter which lane built the data tuple
    if len(country) < len(payload):
        pad = np.full(len(payload) - len(country), "", dtype=country.dtype)
        country = np.concatenate([country, pad])
        data = (df, payload, structured_features, row_bc, country, pos, hp_pairs, emb0)

    # BOUNDARY CONTRACT (lib.schemas.DataTuple): the 8-tuple is the widest
    # crossing in the lane — payload/structured_features/row_bc/country locked, every
    # pos/hp index in range, emb0 rows == payload. Validated ONCE per
    # train_one_config call; a shape break dies here with a named field
    # instead of an IndexError three stack frames into a fold.
    from core.schemas import DataTuple as _DataTuple

    with trace_step('training.prepare_inputs.contracts'):
        _DataTuple(
            n_df=len(df),
            payload=payload,
            structured_features=structured_features,
            row_bc=row_bc,
            country=country,
            pos=pos,
            hp_pairs=hp_pairs,
            emb0=emb0,
        )

        if folds_override is not None:
            # folds_override contract: EITHER one gtin-set (holdout mode: that
            # set is the single test fold; train = every other gtin) OR a list
            # of sets (explicit CV folds — second11's connected-component split;
            # each set is one fold, train = union of the others).
            if isinstance(folds_override, (set, frozenset)):
                folds = [set(folds_override)]
            else:
                folds = list(folds_override)
        else:
            # AUDIT FIX (round 2 F10, round 3): --folds N now builds N folds.
            # The old code ALWAYS dealt CV_FOLDS (5) and sliced [:n_folds], so
            # --folds 8 silently trained 5 — a cap no one asked for. Behavior
            # for n_folds <= CV_FOLDS is IDENTICAL: kfold_gtins deals the
            # same strided permutation split, and the --quick prefix slice is
            # unchanged.
            n_folds = cv_folds if cv_folds is not None else CV_FOLDS
            all_folds = kfold_gtins(df, n_folds, SEED)
            # --quick trains on the first n_folds of the SAME split (folds stay comparable)
            folds = all_folds[:n_folds]

    # The training band is selected by config/CLI/HPO; the evaluation band
    # has its own fixed config contract. Share the implementation and reuse
    # results only when those contracts agree, so retuning training still works.
    if ANN_MINING_ENABLED and emb0.size:
        from core.common import band as _band_helper
        from core.schemas import BandSpec
        train_band = BandSpec.model_validate(band)
        eval_band = BandSpec.model_validate(_band_helper("eval_mining"))

        def mine(selected: BandSpec):
            pairs, _ = mine_hard_negatives(
                df, emb0, seed=seed, n_target=N_TARGET_MINING,
                cosine_lo=selected.lo, cosine_hi=selected.hi,
            )
            return pairs

        hard_train_all = mine(train_band)
        hard_eval = hard_train_all if train_band == eval_band else mine(eval_band)
    else:
        hard_train_all = np.empty((0, 2), dtype=int)
        hard_eval = np.empty((0, 2), dtype=int)

    # ═══════════════════════════════════════════════════════════════════════
    # RETRIEVAL-POOL INPUTS (ER-346) — computed ONCE, before the fold loop.
    # ═══════════════════════════════════════════════════════════════════════
    # The holdout ranking metric used to be evaluated on a pool that was one
    # candidate wide for 90.5% of queries (5,292 of 5,847 holdout queries saw
    # ONLY their own positive; the widest pool any query saw was 6 against
    # max(ks)=10).  A perfect oracle and an informationless constant scorer
    # therefore produced IDENTICAL numbers, so the metric measured nothing.
    # The pool below gives every query a genuine ranking task.  Its inputs are
    # derived here from the SAME graph, the SAME payload space, and the SAME
    # fixed artifacts the fold itself uses.
    with trace_step('training.prepare_inputs.retrieval_pool_inputs'):
        from core.ranking_metrics import component_index

        # Component ids over the positive-pair graph: the unit
        # training.folds.component_folds splits on.  Recomputed here (the split
        # helper returns fold membership, not component identity) and asserted
        # fold-pure below, so a competitor drawn from ANOTHER component of the
        # SAME fold provably shares no positive-pair chain with the query.
        _retrieval_row_component = component_index(pos, row_bc)

        def _holdout_true_match_gtin_pairs() -> frozenset[tuple[str, str]]:
            """Known same-product relations — never a competing candidate.

            A competitor that the lane's own evidence says IS the query's product
            would be scored as a non-relevant candidate and turn a correct
            ranking into a recorded miss.  Three sources of that evidence exist
            and all three are excluded: the labeled-pairs ground truth, the
            gate's own ``proceed`` decision (the relation the canonicals are built
            from), and an identical canonical identity string.
            """
            from pipeline import load_canonical_map

            pairs: set[tuple[str, str]] = set()

            def add(left: object, right: object) -> None:
                a, b = str(left).strip(), str(right).strip()
                if a and b and a != b:
                    pairs.add((a, b))
                    pairs.add((b, a))

            if _ARTIFACT_CACHE:
                from training.folds import load_labeled_pairs

                labeled = check_labeled_pairs_frame(
                    load_labeled_pairs(RESULTS / F["labeled_pairs"])
                )
            else:
                labeled = check_labeled_pairs_frame(
                    pd.read_csv(
                        RESULTS / F["labeled_pairs"],
                        dtype={"gtin1": str, "gtin2": str},
                        keep_default_na=False,
                    )
                )
            labels = pd.to_numeric(labeled["true_label"], errors="raise").astype(int)
            positives = labeled.loc[labels == 1]
            for left, right in zip(positives["gtin1"], positives["gtin2"], strict=True):
                add(left, right)
            gates = pd.read_csv(
                RESULTS / F["gate_results"],
                dtype={"gtin1": str, "gtin2": str},
                keep_default_na=False,
            )
            proceeds = gates.loc[gates["gate_decision"] == "proceed"]
            for left, right in zip(proceeds["gtin1"], proceeds["gtin2"], strict=True):
                add(left, right)
            by_canonical: dict[str, list[str]] = {}
            for gtin, canonical in load_canonical_map().items():
                by_canonical.setdefault(str(canonical), []).append(str(gtin))
            for group in by_canonical.values():
                if len(group) > 1:
                    for left in group:
                        for right in group:
                            add(left, right)
            return frozenset(pairs)

        from training.prepared_bundle import canonical_payload_rows
        _retrieval_canonical_rows = canonical_payload_rows(len(df), payload, row_bc)
        _retrieval_true_match_pairs = _holdout_true_match_gtin_pairs()
        print(
            f"[retrieval-pool] canonical competitor universe={len(_retrieval_canonical_rows):,} "
            f"payload rows | known true-match gtin pairs excluded="
            f"{len(_retrieval_true_match_pairs) // 2:,} (labeled positives + gate "
            f"proceed + identical canonical identity)",
            flush=True,
        )

    rows: list[dict] = []
    fold_plans = []
    for fold_i, test_bc in enumerate(folds):
        try:
            t_fold = time.perf_counter()
            with trace_step('training.prepare_inputs.fold_split'):
                if len(folds) > 1:
                    train_bc = set().union(*[f for j, f in enumerate(folds) if j != fold_i])
                else:
                    # single holdout fold: train side = every gtin NOT in the
                    # test fold (dev_override carves dev out of this below)
                    train_bc = all_gtin_set - folds[0]

                # split train gtins into train/dev (early stopping target).
                # dev_override: caller-supplied component-aware dev boundary
                # (skips the rng carve — a gtin-level carve SPLITS positive
                # pairs between train and dev, silently dropping them from both:
                # measured 7,808 of 37,445 pair-uses in 5-fold CV).
                if dev_override is not None:
                    dev_bc = set(dev_override) & train_bc
                    tr_bc = train_bc - dev_bc
                else:
                    rng = np.random.default_rng(seed + fold_i)
                    train_bcs = np.array(sorted(train_bc))
                    perm = rng.permutation(len(train_bcs))
                    dev_frac = dev_fraction if dev_fraction is not None else DEV_FRACTION
                    n_dev = max(1, int(len(train_bcs) * dev_frac))
                    dev_bc = set(train_bcs[perm[:n_dev]])
                    tr_bc = set(train_bcs[perm[n_dev:]])
            # ── HOLDOUT-SELECTION DISCIPLINE (test-leak fix, 2026-09-12) ──
            # In selection mode the fold's job is to RANK hyperparameters,
            # and the ranking signal must come from DEV only. Hard asserts
            # (fold dies loudly — FAILED row + traceback, never a silent
            # leak) when the boundary is wrong: test gtins in
            # train/dev, or dev gtins in the test fold, would leak the
            # holdout into the very signal that picks the config.
            if selection_mode:
                assert dev_override is not None, (
                    "[hpo] selection mode requires an explicit component-"
                    "aware dev boundary (dev_override) — an rng carve cannot "
                    "guarantee the selection signal is dev-only"
                )
                _dev_o = set(dev_override)
                _dev_in_test = _dev_o & test_bc
                assert not _dev_in_test, (
                    f"[hpo] LEAK: {len(_dev_in_test)} dev_override gtins "
                    f"are in the test fold (e.g. {sorted(_dev_in_test)[:3]}) "
                    "— they would be silently dropped from dev while the "
                    "split claims to be clean"
                )
                _leak = test_bc & (tr_bc | dev_bc)
                assert not _leak, (
                    f"[hpo] LEAK: {len(_leak)} test gtins in train/dev "
                    f"(e.g. {sorted(_leak)[:3]})"
                )
                assert dev_bc, (
                    "[hpo] LEAK: empty dev set — nothing to select on"
                )
                assert dev_bc == _dev_o & train_bc, (
                    "[hpo] LEAK: dev boundary != dev_override ∩ train side"
                )
                print(
                    f"[hpo] objective={HPO_OBJECTIVE_HOLDOUT} | "
                    f"train={len(tr_bc):,} dev={len(dev_bc):,} "
                    f"test={len(test_bc):,} gtins | test quarter "
                    f"excluded from training+selection",
                    flush=True,
                )

            with trace_step('training.prepare_inputs.fold_pools'):
                test_pos = eval_pos[pairs_in_set(eval_pos, row_bc, test_bc)]
                # ── RETRIEVAL-POOL FOLD PURITY (ER-346) ──────────────────────
                # The competitor rule excludes the query's own COMPONENT, which is
                # only fold-safe if the split really deals whole components: a
                # component straddling the boundary would let a competitor carry a
                # positive relationship across it.  Checked, not assumed — the
                # component ids and the split are derived from the same graph.
                _bc_in_test = np.asarray(
                    [row_bc[i] in test_bc for i in range(len(row_bc))], dtype=bool
                )
                _test_components = np.unique(
                    _retrieval_row_component[_bc_in_test]
                )
                _impure = int(
                    np.sum(
                        np.isin(_retrieval_row_component, _test_components)
                        & ~_bc_in_test
                        & (_retrieval_row_component >= 0)
                    )
                )
                assert not _impure, (
                    f"LEAK: {_impure} payload rows belong to a component that "
                    "intersects the test fold but is not contained in it — the "
                    "retrieval competitor rule assumes the split deals WHOLE "
                    "components, so excluding own-component competitors would not "
                    "be fold-safe"
                )
                # Competitor universe for THIS fold: canonical payload rows whose
                # gtin belongs to the test fold, so every query's ranking task
                # stays inside the fold it is scored on.
                _fold_canonical_rows = _retrieval_canonical_rows[
                    _bc_in_test[_retrieval_canonical_rows]
                ]
                train_pos = pos[pairs_in_set(pos, row_bc, tr_bc)]
                dev_pos = eval_pos[pairs_in_set(eval_pos, row_bc, dev_bc)]
                dev_pos = _merge_different_calibration_positives(
                    dev_pos,
                    labeled_different_pos,
                    row_bc,
                    dev_bc,
                )
                hard_train = hard_train_all[pairs_in_set(hard_train_all, row_bc, tr_bc)]
                hard_dev = hard_eval[pairs_in_set(hard_eval, row_bc, dev_bc)]
                hard_test = hard_eval[pairs_in_set(hard_eval, row_bc, test_bc)]
                _train_neg_mask = (
                    pairs_in_set(_train_neg_source, row_bc, tr_bc)
                    if _train_neg_source is not None and len(_train_neg_source)
                    else np.zeros(0, dtype=bool)
                )
                tr_negs = (
                    _train_neg_source[_train_neg_mask]
                    if _train_neg_source is not None and len(_train_neg_source)
                    else np.empty((0, 2), dtype=int)
                )
                tr_neg_sources = (
                    _train_neg_sources[_train_neg_mask]
                    if len(_train_neg_mask)
                    else np.empty(0, dtype=object)
                )
            n_train_hard_neg = len(tr_negs)
            random_easy_unique_candidates = 0
            if loss == "contrastive":
                tr_negs, tr_neg_sources, random_easy_unique_candidates = (
                    _mix_random_easy_training_negatives(
                        tr_negs,
                        tr_neg_sources,
                        df=df,
                        row_bc=row_bc,
                        train_gtins=tr_bc,
                        seed=seed + fold_i + 20_003,
                        enabled=bool(cfg["random_easy_enabled"]),
                        ratio_to_hard=float(cfg["random_easy_ratio_to_hard"]),
                        candidate_pool_size=int(
                            cfg["random_easy_candidate_pool_size"]
                        ),
                    )
                )
            n_train_random_easy_neg = int(
                np.sum(tr_neg_sources == "random_easy")
            )
            train_neg_source_counts = {
                str(source): int(np.sum(tr_neg_sources == source))
                for source in np.unique(tr_neg_sources)
            }
            # caller's explicit negatives (gate hard-negs): same eval pools,
            # same boundary rules — they reinforce dev early-stopping and the
            # test AUC with the "text-similar, different size/pack" class
            if neg_pairs is not None and len(neg_pairs):
                hard_dev = (
                    np.vstack(
                        [hard_dev, neg_pairs[pairs_in_set(neg_pairs, row_bc, dev_bc)]]
                    )
                    if len(neg_pairs[pairs_in_set(neg_pairs, row_bc, dev_bc)])
                    else hard_dev
                )
                hard_test = (
                    np.vstack(
                        [hard_test, neg_pairs[pairs_in_set(neg_pairs, row_bc, test_bc)]]
                    )
                    if len(neg_pairs[pairs_in_set(neg_pairs, row_bc, test_bc)])
                    else hard_test
                )
            if sample:
                # A small chain-check sample need not contain all component
                # populations required for threshold calibration.  Preserve
                # its complete DEV pool for early stopping and record an
                # explicitly unavailable calibration result after training;
                # full runs retain the strict component-safe reservation.
                calibration_pos = np.empty((0, 2), dtype=int)
                calibration_neg = np.empty((0, 2), dtype=int)
                print(
                    f"  [calibration] fold {fold_i}: sample mode — "
                    "strict calibration reservation skipped",
                    flush=True,
                )
            else:
                from training.folds import derive_calibration_carve

                dev_pos, calibration_pos, hard_dev, calibration_neg = (
                    derive_calibration_carve(
                        dev_pos,
                        hard_dev,
                        row_bc,
                        calibration_config["split"],
                        seed=seed,
                        fold_index=fold_i,
                    )
                )
            if len(test_pos) == 0 or len(hard_test) == 0:
                rows.append(
                    {
                        "fold": fold_i,
                        "status": "skipped",
                        "reason": f"empty eval (pos={len(test_pos)}, neg={len(hard_test)})",
                    }
                )
                continue
            train_all = train_pos
            n_gate_kept = len(train_pos)  # gate rows actually in train_all
            if train_frac is not None and train_frac < 1.0 and len(train_all):
                rng_f = np.random.default_rng(seed + fold_i + 7)
                keep = rng_f.random(len(train_all)) < train_frac
                train_all = train_all[keep] if keep.any() else train_all[:1]
                n_gate_kept = len(train_all)
            if use_hp:
                hp_train = hp_pairs[pairs_in_set(hp_pairs, row_bc, tr_bc)]
                # frac subsetting applies to the gate positives ONLY; the
                # volume-verified cross-country pairs are the rare class the
                # lane is trying to ADD — subsampling them away would defeat
                # their purpose (the old code overwrote train_all with the
                # UNSAMPLED train_pos, silently discarding the frac knob).
                if train_frac is not None and train_frac < 1.0 and len(train_all):
                    # re-derive the sampled gate set: train_all IS the sampled
                    # train_pos here (same rng, same order) — vstack on top
                    train_all = (
                        np.vstack([train_all, hp_train])
                        if len(hp_train)
                        else train_all
                    )
                else:
                    train_all = (
                        np.vstack([train_pos, hp_train])
                        if len(hp_train)
                        else train_pos
                    )
                    n_gate_kept = len(train_pos)
                if len(hp_train):
                    print(
                        f"    [hard-positives] fold {fold_i}: +{len(hp_train):,} "
                        f"volume-verified pairs in train "
                        f"({len(train_all) - len(hp_train):,} gate + {len(hp_train):,} "
                        f"cross-country)",
                        flush=True,
                    )

            # Static masked-positive copies are present in every epoch of the
            # training dataset. Keep their per-fold denominator beside the
            # dynamic hard-negative mask telemetry so masking percentages are
            # interpretable rather than just raw counts.
            train_gtins = set(row_bc[train_all[:, 0]].tolist()) if len(train_all) else set()
            static_masked_pos = sum(
                1 for item in (mask_audit or [])
                if str(item.get("gtin", "")) in train_gtins
            )
            static_positive_pct = (
                static_masked_pos / len(train_all) if len(train_all) else 0.0
            )

            # dev evaluator needs pos/neg pairs as texts
            dev_pairs = [(payload[a], payload[b]) for a, b in dev_pos]
            dev_neg_pairs = [(payload[a], payload[b]) for a, b in hard_dev]
            dev_structured = [
                [structured_features[int(a)].tolist(), structured_features[int(b)].tolist()]
                for a, b in list(dev_pos) + list(hard_dev)
            ]
            if len(dev_pairs) == 0 or len(dev_neg_pairs) == 0:
                rows.append(
                    {
                        "fold": fold_i,
                        "status": "skipped",
                        "reason": "empty dev split — early stopping needs pos and neg dev pairs",
                    }
                )
                continue

            with trace_step('training.prepare_inputs.fold_plan'):
                fixed = {
                    'fold_i': fold_i,
                    'test_bc': test_bc,
                    'tr_bc': tr_bc,
                    'test_pos': test_pos,
                    'train_pos': train_pos,
                    'dev_pos': dev_pos,
                    'hard_train': hard_train,
                    'hard_dev': hard_dev,
                    'hard_test': hard_test,
                    'tr_negs': tr_negs,
                    'tr_neg_sources': tr_neg_sources,
                    'n_train_hard_neg': n_train_hard_neg,
                    'random_easy_unique_candidates': random_easy_unique_candidates,
                    'n_train_random_easy_neg': n_train_random_easy_neg,
                    'train_neg_source_counts': train_neg_source_counts,
                    'calibration_pos': calibration_pos,
                    'calibration_neg': calibration_neg,
                    'train_all': train_all,
                    'n_gate_kept': n_gate_kept,
                    'static_masked_pos': static_masked_pos,
                    'static_positive_pct': static_positive_pct,
                    'dev_pairs': dev_pairs,
                    'dev_neg_pairs': dev_neg_pairs,
                    'dev_structured': dev_structured,
                    '_fold_canonical_rows': _fold_canonical_rows,
                }
                fixed.update(_prepare_objective_plan(
                    loss=loss, payload=payload, structured_features=structured_features,
                    train_all=train_all, tr_negs=tr_negs, tr_neg_sources=tr_neg_sources,
                    hp_pairs=hp_pairs, row_bc=row_bc, tr_bc=tr_bc, use_hp=use_hp,
                    mask_audit=mask_audit, hard_negative_mask_audit=hard_negative_mask_audit,
                    hard_train=hard_train, seed=seed, fold_i=fold_i,
                ))
                fixed["random_neg_pairs"] = _split_safe_random_negative_pairs(
                    df, row_bc, set(test_bc), seed=SEED + fold_i + 1000,
                    n_neg=int(_timed_load_config(f"fold{fold_i}.random_neg_pairs")["pairs"]["n_neg"]),
                )
                retrieval_ks = tuple(_timed_load_config(f"fold{fold_i}.retrieval_ks")["evaluation"]["retrieval_ks"])
                fixed["retrieval_pool"] = build_evaluation_pool(
                    test_pos, np.asarray([str(df["sku_id"].iloc[int(i)]) for i in test_pos[:, 0]], dtype=str),
                    competitor_rows=_fold_canonical_rows, row_component=_retrieval_row_component,
                    row_bc=row_bc, n_competitors=competitors_per_query(retrieval_ks),
                    seed=seed + fold_i * 1009 + 340346, ks=retrieval_ks,
                    priority_pairs=hard_test, excluded_gtin_pairs=_retrieval_true_match_pairs,
                )
            fold_plans.append(fixed)
        except Exception:
            rows.append({"fold": fold_i, "status": "failed", "traceback": traceback.format_exc()})
    return {"shared": {
        'payload_metadata': payload_metadata,
        'gate_lookup': gate_lookup,
        '_retrieval_row_component': _retrieval_row_component,
        '_retrieval_true_match_pairs': _retrieval_true_match_pairs,
        '_train_neg_source': _train_neg_source,
        'country': country,
    }, "folds": fold_plans, "skipped": rows}


@timed
def train_one_config(
    cfg: dict,
    *,
    loss: str,
    model_id: str,
    use_hp: bool,
    band: tuple[float, float],
    data,
    seed: int,
    on_cuda: bool,
    cv_folds: int | None = None,
    run_tag: str = "main",
    folds_override: list[set[str]] | set[str] | None = None,
    dev_fraction: float | None = None,
    # dev_override: explicit dev gtin set (component-aware splits pass it;
    # when set, the rng carve is skipped — the caller owns the boundary)
    dev_override: set[str] | None = None,
    # neg_pairs: (N,2) row pairs used as EXPLICIT negatives — appended to the
    # mined dev/test eval pools and, for MNRL, the 3rd dataset column (the
    # gate hard-negatives: same-brand, text-similar, different size/pack)
    neg_pairs: np.ndarray | None = None,
    # training-only negative population; may include masked label-0 copies.
    # neg_pairs remains the immutable dev/test evaluation population.
    train_neg_pairs: np.ndarray | None = None,
    # Source provenance aligned row-for-row with the negative arrays. These
    # labels must travel with the pairs through the component fold boundary.
    neg_pair_sources: np.ndarray | None = None,
    train_neg_pair_sources: np.ndarray | None = None,
    # dynamic hard-negative masking: each training dataset presentation gets
    # a fresh masked anchor; no static negative copies are added.
    dynamic_mask_hard_negatives: bool = False,
    dynamic_mask_frac: float = 0.0,
    dynamic_mask_prob: float | None = None,
    dynamic_mask_lo: float | None = None,
    dynamic_mask_hi: float | None = None,
    mask_audit: list[dict] | None = None,
    hard_negative_mask_audit: list[dict] | None = None,
    prepared_tokens: dict | None = None,
    prepared_plan: dict | None = None,
    ann_refresh_enabled: bool = False,
    attribute_conflict_refresh_enabled: bool = False,
    # 07d data-scaling: keep only this fraction of TRAIN pairs (dev/test
    # pools untouched). Subsampled AFTER the split, seeded per fold.
    train_frac: float | None = None,
    # sample run (chain validation): visibility dumps write into the
    # run-tag dir but never move the shared latest-pointer (same
    # discipline as the fold-metrics pointer in train.py)
    sample: bool = False,
    resume: bool = False,
    # selection mode (test-leak fix, 2026-09-12): HPO/grid lanes in
    # HOLDOUT split call with True — the fold trains on q0+q1, early-stops
    # and is SELECTED on calibration metrics from dev (q2), and the test quarter's
    # eval block (pair_auc/PR-AUC/Youden/pair dump) is SKIPPED entirely:
    # the test quarter is read exactly once, by the main train lane, so
    # hyperparameters can never be fitted on it. Skipped rows carry
    # test_eval="skipped_selection_mode" — loud, never a silent NaN.
    selection_mode: bool = False,
    # Explicit callers can withhold test evaluation independently of HPO settings.
    skip_test_eval: bool = False,
    wandb_ctx=None,
    # Consolidated trace (core/tracing.py stage "training"): the caller may pass
    # the run's single stage writer; the default is this module's process-scoped
    # owner, so a caller that does not know about the trace (hpo lanes, model
    # tracks) still lands its folds in the same stage. Nothing here WRITES the
    # trace — train.py flushes the stage once, which is what keeps one commit
    # per stage and the flow order intact.
    trace=None,
) -> list[dict]:
    """Train cfg across the group-aware folds. Returns fold metric rows
    (failures included, with traceback)."""
    import torch
    trace = trace if trace is not None else training_trace()
    # Per-batch rows are published for the lanes a reader follows step by step;
    # a SWEEP lane (grid/tpe trial, or an explicit selection-mode caller) keeps
    # the exact per-epoch batch CENSUS but withholds the entity sample, whose
    # volume is bounded by the trial count rather than by the run. The policy is
    # stated in the sample_budget row and in the batch.capture row below.
    batch_entity_cap = 0 if (selection_mode or skip_test_eval) else None
    with trace_step('training.train_one_config.config_validation'):
        if dynamic_mask_lo is None or dynamic_mask_hi is None:
            raise ValueError(
                "dynamic hard-negative masking requires its configured extent band"
            )
        dynamic_mask_lo = float(dynamic_mask_lo)
        dynamic_mask_hi = float(dynamic_mask_hi)
        # BOUNDARY CONTRACT (lib.schemas.TrainConfig): the optimizer/early-stop
        # dict — every key validated (epochs >= 1, lr > 0, warmup in [0,1]...)
        # before a single fold runs. A missing/illegal knob dies HERE with the
        # field named, not inside the HF Trainer mid-epoch.
        from core.schemas import TrainConfig as _TrainConfig

        _TrainConfig.model_validate(cfg)
        if cfg["architecture"] != "two_tower":  # schema keeps this exhaustive
            raise ValueError(f"unsupported training architecture: {cfg['architecture']}")
        _CFG_DEEPCOPY_TOTALS.clear()
        calibration_config = _timed_load_config("run.calibration")

    with trace_step('training.train_one_config.fixed_inputs'):
        df, payload, structured_features, row_bc, country, pos, hp_pairs, emb0 = data
        if prepared_plan is None:
            fixed_inputs = prepare_fixed_training_inputs(
                cfg, loss=loss, model_id=model_id, use_hp=use_hp, band=band, data=data,
                seed=seed, cv_folds=cv_folds, folds_override=folds_override,
                dev_fraction=dev_fraction, dev_override=dev_override,
                neg_pairs=neg_pairs, train_neg_pairs=train_neg_pairs,
                neg_pair_sources=neg_pair_sources, train_neg_pair_sources=train_neg_pair_sources,
                mask_audit=mask_audit, hard_negative_mask_audit=hard_negative_mask_audit,
                train_frac=train_frac, sample=sample, selection_mode=selection_mode,
            )
        else:
            fixed_inputs = prepared_plan["inputs"]
        payload_metadata = fixed_inputs["shared"]['payload_metadata']
        gate_lookup = fixed_inputs["shared"]['gate_lookup']
        _retrieval_row_component = fixed_inputs["shared"]['_retrieval_row_component']
        _retrieval_true_match_pairs = fixed_inputs["shared"]['_retrieval_true_match_pairs']
        _train_neg_source = fixed_inputs["shared"]['_train_neg_source']
        country = fixed_inputs["shared"]['country']
        rows = list(fixed_inputs["skipped"])
        _canon_attrs: dict[str, dict] | None = None
        if prepared_tokens is not None:
            from training.token_inputs import payload_sha256

            prepared_payload_digest = payload_sha256(payload)
        else:
            prepared_payload_digest = None
    # ── CONSOLIDATED TRACE: one config-level row + the per-grain accumulators
    # that are published once, after the fold loop, so the stage keeps ONE
    # commit (train.py) and the row volume stays inside core.tracing's caps.
    _batch_records: list[dict] = []
    _collapse_records: list[dict] = []
    _collapse_evidence_folds = 0
    trace.add(
        "run",
        "train_config",
        in_count=None,
        out_count=None,
        reason=(
            "one train_one_config call: the config scale + folds trained "
            "(failures included in the returned rows). No in/out counts: the "
            "inputs (source rows) and the outputs (folds) are different "
            "units, so any derived dropped_count would be meaningless"
        ),
        detail={
            "loss": loss,
            "model_id": model_id,
            "run_tag": run_tag,
            "sample": bool(sample),
            "selection_mode": bool(selection_mode),
            "skip_test_eval": bool(skip_test_eval),
            "epochs": cfg.get("epochs"),
            "n_source_rows": int(len(df)),
            "n_folds": int(len(fixed_inputs["folds"])),
            "learning_rate": cfg.get("lr"),
            "batch_size_cpu": BATCH_SIZE_CPU,
            "batch_size_cuda": BATCH_SIZE_CUDA,
            "n_train_pos": int(len(pos)),
            "n_hp_pairs": int(len(hp_pairs)) if hp_pairs is not None else 0,
            "n_neg_pairs": int(len(neg_pairs)) if neg_pairs is not None else 0,
            "train_frac": train_frac,
            "cv_folds": cv_folds,
            "folds": [int(fold["fold_i"]) for fold in fixed_inputs["folds"]],
            "batch_entity_cap": batch_entity_cap,
            "batch_rows_policy": (
                "entity sample withheld (sweep/selection lane): the exact "
                "per-fold-epoch batch census is still emitted"
                if batch_entity_cap == 0
                else "entity sample emitted under core.tracing's ENTITY caps"
            ),
        },
        source="training.train_one_config",
    )
    for fold_inputs in fixed_inputs["folds"]:
        fold_i = fold_inputs["fold_i"]
        try:
            with trace_step('training.train_one_config.fold_setup'):
                t_fold = time.perf_counter()
                test_bc = fold_inputs['test_bc']
                tr_bc = fold_inputs['tr_bc']
                test_pos = fold_inputs['test_pos']
                train_pos = fold_inputs['train_pos']
                dev_pos = fold_inputs['dev_pos']
                hard_train = fold_inputs['hard_train']
                hard_dev = fold_inputs['hard_dev']
                hard_test = fold_inputs['hard_test']
                tr_negs = fold_inputs['tr_negs']
                tr_neg_sources = fold_inputs['tr_neg_sources']
                n_train_hard_neg = fold_inputs['n_train_hard_neg']
                random_easy_unique_candidates = fold_inputs['random_easy_unique_candidates']
                n_train_random_easy_neg = fold_inputs['n_train_random_easy_neg']
                train_neg_source_counts = fold_inputs['train_neg_source_counts']
                calibration_pos = fold_inputs['calibration_pos']
                calibration_neg = fold_inputs['calibration_neg']
                train_all = fold_inputs['train_all']
                n_gate_kept = fold_inputs['n_gate_kept']
                static_masked_pos = fold_inputs['static_masked_pos']
                static_positive_pct = fold_inputs['static_positive_pct']
                dev_pairs = fold_inputs['dev_pairs']
                dev_neg_pairs = fold_inputs['dev_neg_pairs']
                dev_structured = fold_inputs['dev_structured']
                _fold_canonical_rows = fold_inputs['_fold_canonical_rows']
                objective_plan = fold_inputs["objective"]
                checkpoint_dir = artifact(
                    "checkpoint_repo",
                    {
                        "model_tag": model_id.rstrip("/").rsplit("/", 1)[-1],
                        "run_tag": run_tag,
                        "fold": fold_i,
                        "step": 0,
                    },
                ).parent
                if resume and not any(checkpoint_dir.glob(f"checkpoint-*/{_trainer_state_filename()}")):
                    if checkpoint_publication_deferred():
                        raise FileNotFoundError(f"resume requires downloaded local trainer checkpoints: {checkpoint_dir}")
                    from training.dvc_store import restore_checkpoint

                    restore_checkpoint(RESULTS, checkpoint_dir)
                    print(f"    [resume] restored {checkpoint_dir} from DVC", flush=True)

                ensure_parent(checkpoint_dir)
                # ── CONSOLIDATED TRACE: the fold's input populations ──────────
                # The fold boundary is where the split becomes real, so it is
                # recorded per fold (entity scope, key = fold) with the counts
                # the objective actually receives: train pairs enter it, the
                # dev/test pairs are HELD OUT (a different destiny, stated in
                # words — never silently counted as a drop).
                trace.add(
                    "fold",
                    "inputs",
                    scope=SCOPE_ENTITY,
                    key=fold_i,
                    in_count=(
                        int(len(train_all)) + int(len(dev_pos)) + int(len(test_pos))
                    ),
                    out_count=int(len(train_all)),
                    reason=(
                        "fold split: train pairs enter the objective; dev/test "
                        "pairs are held out for early stopping / evaluation"
                    ),
                    detail={
                        "n_train": int(len(train_all)),
                        "n_dev": int(len(dev_pos)),
                        "n_test": int(len(test_pos)),
                        "n_train_pos": int(len(train_pos)),
                        "n_gate_kept": int(n_gate_kept),
                        "n_train_hard_neg": int(n_train_hard_neg),
                        "n_train_random_easy_neg": int(n_train_random_easy_neg),
                        "n_train_random_easy_unique_candidates": int(
                            random_easy_unique_candidates
                        ),
                        "n_train_neg": int(len(tr_negs)),
                        "n_test_gate_neg": int(len(hard_test)),
                        "static_masked_positives": int(static_masked_pos),
                        "static_positive_pct": float(static_positive_pct),
                        "train_neg_sources": dict(train_neg_source_counts),
                        "held_out_dev": int(len(dev_pos)),
                        "held_out_test": int(len(test_pos)),
                    },
                    source="training.prepare_fixed_training_inputs",
                )
            batch_trace = None

            # Tied-weight two-tower retrieval model: the trainer receives
            # (SKU text, canonical text) pairs; each side is encoded on its
            # own before cosine/loss comparison.  CrossEncoder is optional
            # only in rerank.py after retrieval, never this default path.
            with trace_step('training.train_one_config.fold_model_and_dataset'):
                model = load_local_sentence_transformer(
                    model_id, device="cuda" if on_cuda else "cpu"
                )
                _align_model_token_ids(model)
                dropout_added = _configure_projection_dropout(
                    model, float(cfg["projection_dropout"])
                )
                print(
                    f"    [regularization] weight_decay={cfg['weight_decay']:.4g} | "
                    f"projection_dropout={cfg['projection_dropout']:.4g} "
                    f"({'added' if dropout_added else 'configured'}) | "
                    f"label_smoothing={cfg['label_smoothing']:.4g}",
                    flush=True,
                )
                if loss == "contrastive":
                    print(
                        f"    [random-easy-train] hard={n_train_hard_neg:,} | "
                        f"random_easy={n_train_random_easy_neg:,} "
                        f"({random_easy_unique_candidates:,} unique candidates) | "
                        f"ratio={n_train_random_easy_neg / n_train_hard_neg if n_train_hard_neg else 0.0:.3f}",
                        flush=True,
                    )
                model.max_seq_length = runtime("max_seq_length")  # SSOT, no literal
                from core.encoding_inputs import enable_zero_truncation
                enable_zero_truncation(model)
                # TASK B item 6: TF32 + text torch.compile behind advanced.accel.*
                # (both default OFF; compile also respects accel.compile perf gate).
                _accel = training_cfg().advanced.accel
                if on_cuda and _accel.tf32:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    print("    [accel] TF32 matmul/cudnn enabled", flush=True)
                if _accel.compile:
                    from core.fast_kernels import compile_model
                    model[0].auto_model = compile_model(
                        model[0].auto_model,
                        name="text.encoder",
                        mode=str(_accel.compile_mode),
                    )
                    print(
                        f"    [accel] text encoder compile requested mode={_accel.compile_mode}",
                        flush=True,
                    )
                token_lookup = None
                if prepared_tokens is not None:
                    from training.token_inputs import PreparedTokenLookup
                    token_lookup = PreparedTokenLookup(
                        model, prepared_tokens, payload, payload_digest=prepared_payload_digest
                    )

                # ── build the training dataset FIRST (steps derive from it) ──
                from datasets import Dataset

                examples = None
                ann_refresh_state: dict[str, object] = {
                    "pairs": {},
                    "structured_features": {},
                    "sources": {},
                    "version": 0,
                    "count": 0,
                }
                # MNRL/triplet do not install the dynamic contrastive dataset
                # transform, but post-training telemetry is shared by every loss.
                # Keep its empty state defined for those lanes.
                dynamic_mask_stats_by_epoch: dict[int, dict[str, float]] = {}
                if loss == "contrastive":
                    # OnlineContrastiveLoss (owner ruling 2026-09-07): paired
                    # (sentence1, sentence2, label) rows. POSITIVES = train_all
                    # (sku, own canonical); NEGATIVES = the gate hard-no pairs —
                    # text-similar, gate-proven different size/pack/flavor —
                    # restricted to TRAIN gtins (component boundary holds:
                    # pairs_in_set filters by tr_bc). The loss itself then picks
                    # the hard subset per batch (farthest positives, closest
                    # negatives) — hard-pair training at both layers.
                    if len(tr_negs) == 0:
                        rows.append(
                            {
                                "fold": fold_i,
                                "status": "skipped",
                                "reason": "contrastive loss needs labeled negatives "
                                "(gate hard-no pairs) — none resolved in train",
                            }
                        )
                        continue
                    fixed_dataset = objective_plan["dataset"]
                    s1, s2, lab = (fixed_dataset[key] for key in ("sentence1", "sentence2", "label"))
                    pair_populations = fixed_dataset["pair_population"]
                    presentation_counts: dict[tuple, int] = {}
                    train_ds = Dataset.from_dict(fixed_dataset)
                    dynamic_mask_counts: dict[int, int] = {}
                    dynamic_mask_counts_by_epoch: dict[int, dict[int, int]] = {}
                    dynamic_epoch_ref = {"epoch": 0}
                    dynamic_mask_audit: list[dict] = []
                    if (
                        (dynamic_mask_hard_negatives and dynamic_mask_frac > 0)
                        or ann_refresh_enabled
                        or attribute_conflict_refresh_enabled
                        or TRACK_DATAPOINT_USAGE
                    ):
                        import random as _random
                        from functools import partial

                        _mask_rng = _random.Random(seed + fold_i + 100_003)
                        train_ds.set_transform(
                            partial(
                                _dynamic_mask_negative_transform,
                                rng=_mask_rng,
                                frac=dynamic_mask_frac if dynamic_mask_hard_negatives else 0.0,
                                mask_prob=dynamic_mask_prob,
                                mask_lo=dynamic_mask_lo,
                                mask_hi=dynamic_mask_hi,
                                counts=dynamic_mask_counts,
                                counts_by_epoch=dynamic_mask_counts_by_epoch,
                                stats_by_epoch=dynamic_mask_stats_by_epoch,
                                epoch_ref=dynamic_epoch_ref,
                                ann_pairs=ann_refresh_state["pairs"],
                                ann_structured_features=ann_refresh_state[
                                    "structured_features"
                                ],
                                ann_sources=ann_refresh_state["sources"],
                                ann_state=ann_refresh_state,
                                pair_populations=pair_populations,
                                presentation_counts=(
                                    presentation_counts
                                    if TRACK_DATAPOINT_USAGE
                                    else None
                                ),
                                mask_audit=dynamic_mask_audit,
                                fold=fold_i,
                                token_lookup=token_lookup,
                            )
                        )
                    # ── TRAIN VISIBILITY (owner directive 2026-09-07): the
                    # EXACT rows the model ingests for this fold — sentence1,
                    # sentence2, label, both gtins, pos/hp/neg provenance.
                    # Rewritten per fold (last fold wins; fold metrics CSV
                    # keeps per-fold counts).
                    _dump_train_visibility(
                        fold_i,
                        s1,
                        s2,
                        lab,
                        train_all,
                        tr_negs,
                        tr_neg_sources=tr_neg_sources,
                        hp_in_train=(
                            hp_pairs[pairs_in_set(hp_pairs, row_bc, tr_bc)]
                            if use_hp and hp_pairs is not None and len(hp_pairs)
                            else None
                        ),
                        payload=payload,
                        row_bc=row_bc,
                        payload_metadata=payload_metadata,
                        gate_lookup=gate_lookup,
                        run_tag=run_tag,
                        sample=sample,
                    )
                elif loss == "mnrl":
                    # MNRL's third column is an explicit negative for *that same
                    # anchor*, not an arbitrary text sampled from a global pool.
                    # Preserve the gate/attribute-conflict evidence by joining
                    # every source-side hard negative to its source's positive
                    # canonical pair. The loss also continues to use the other
                    # positives in a batch as in-batch negatives.
                    triples = objective_plan["triples"]
                    if not triples:
                        rows.append(
                            {
                                "fold": fold_i,
                                "status": "skipped",
                                "reason": "MNRL needs anchor-positive-negative triples; "
                                "none survived the train component boundary",
                            }
                        )
                        continue
                    shared_gtin_rows = objective_plan["shared_gtin_rows"]
                    # Twin exposure (point A watch-item): counterfactual copies
                    # share 90%+ tokens with their source positive, so their
                    # denominator pressure is the sharpest in the batch. Report
                    # the share every fold; gradient spikes in epochs 1-2 point
                    # here first. Existing guards: max_grad_norm=1.0,
                    # warmup_ratio=0.05, dev-AP early stopping, and ~63
                    # in-batch natural negatives per anchor at batch 64.
                    twin_copies = {
                        int(audit["copy_payload_idx"])
                        for audit in hard_negative_mask_audit or []
                        if audit.get("target_mode") == "counterfactual"
                        and audit.get("copy_payload_idx") is not None
                    }
                    twin_triples = sum(1 for _, _, n in triples if n in twin_copies)
                    print(
                        f"    [mnrl-pairs] triples={len(triples):,} | "
                        f"twin_negatives={twin_triples:,} "
                        f"({twin_triples / max(len(triples), 1):.1%}) | "
                        f"positive-GTIN repeat exposure={shared_gtin_rows:,} "
                        "(different texts may still share product identity)",
                        flush=True,
                    )
                    # Per-triple population tags (base/masked/twin) parallel the
                    # triples so train-time MNRL subset monitoring can attribute
                    # loss per population. pair_id is the triple index threaded
                    # through PairIdDataCollator to the loss; both columns are
                    # stripped before tokenization and never affect the loss value.
                    triple_populations = objective_plan["dataset"]["population"]
                    train_ds = Dataset.from_dict(objective_plan["dataset"])
                else:
                    train_ds = Dataset.from_dict(objective_plan["dataset"])
                    examples = list(range(len(train_ds)))

            with trace_step('training.train_one_config.fold_evaluator'):
                from training.sampler import FrozenBatchSampler
                device_key = "cuda" if on_cuda else "cpu"
                fixed_sampler = objective_plan["sampler"][device_key]
                if cfg["epochs"] > len(fixed_sampler["epochs"]):
                    raise ValueError("requested training epochs exceed locally prepared presentation plan; rebuild locally")
                batch_size = runtime("batch_size_cuda" if on_cuda else "batch_size_cpu")
                if sample:
                    # Smoke tests exercise the saved CPU/CUDA presentation plan.
                    batch_size = fixed_sampler["batch_size"]
                elif fixed_sampler["batch_size"] != batch_size:
                    raise ValueError("local presentation batch size differs from configured runtime; rebuild locally")
                if loss in {"contrastive", "mnrl"}:
                    if "pair_id" not in train_ds.column_names or list(train_ds["pair_id"]) != list(range(len(train_ds))):
                        raise ValueError("local objective pair IDs must map every training row in order; rebuild locally")
                controlled_sampler = FrozenBatchSampler(
                    fixed_sampler["epochs"], expected_rows=len(train_ds), batch_size=batch_size
                )
                n_steps_per_epoch = max(1, len(controlled_sampler))
                warmup_steps = int(sum(len(batches) for batches in fixed_sampler["epochs"][:cfg["epochs"]]) * cfg["warmup_ratio"])
                eval_steps = max(1, n_steps_per_epoch // EVAL_STEPS_PER_EPOCH)

                # dev evaluator: pos pairs vs hard negatives, binary AUC-style
                from sentence_transformers.evaluation import BinaryClassificationEvaluator

                class StructuredBinaryClassificationEvaluator(BinaryClassificationEvaluator):
                    """Binary evaluator using the same fused score as final reports."""

                    def __init__(self, *args, structured_features, feature_weight, **kwargs):
                        super().__init__(*args, **kwargs)
                        self.structured_features = np.asarray(
                            structured_features, dtype=np.float32
                        )
                        self.feature_weight = float(feature_weight)

                    def compute_metrics(self, model):
                        from sklearn.metrics import average_precision_score, matthews_corrcoef
                        from sentence_transformers.util import pairwise_cos_sim

                        emb1 = self.embed_inputs(model, self.sentences1)
                        emb2 = self.embed_inputs(model, self.sentences2)
                        from core.structured_features import fuse_torch

                        n = len(self.sentences1)
                        if self.structured_features.shape[0] != n or self.structured_features.shape[1] != 2:
                            raise RuntimeError(
                                "structured evaluator feature count mismatch: "
                                f"{self.structured_features.shape} != ({n}, 2, feature_dim)"
                            )
                        feat = torch.tensor(self.structured_features, dtype=torch.float32)
                        emb1 = fuse_torch(
                            torch.as_tensor(emb1), feat[:, 0, :], self.feature_weight
                        )
                        emb2 = fuse_torch(
                            torch.as_tensor(emb2), feat[:, 1, :], self.feature_weight
                        )
                        scores = pairwise_cos_sim(emb1, emb2).detach().cpu().numpy()
                        labels_np = np.asarray(self.labels)
                        acc, acc_threshold = self.find_best_acc_and_threshold(
                            scores, labels_np, True
                        )
                        f1, precision, recall, f1_threshold = self.find_best_f1_and_threshold(
                            scores, labels_np, True
                        )
                        predicted = scores >= f1_threshold
                        return {
                            "cosine": {
                                "accuracy": acc,
                                "accuracy_threshold": acc_threshold,
                                "f1": f1,
                                "f1_threshold": f1_threshold,
                                "precision": precision,
                                "recall": recall,
                                "ap": average_precision_score(labels_np, scores),
                                "mcc": matthews_corrcoef(labels_np, predicted),
                            }
                        }

                sentences1 = [a for a, _ in dev_pairs] + [a for a, _ in dev_neg_pairs]
                sentences2 = [b for _, b in dev_pairs] + [b for _, b in dev_neg_pairs]
                labels = [1] * len(dev_pairs) + [0] * len(dev_neg_pairs)
                _sf_cfg = _timed_load_config(
                    f"fold{fold_i}.structured_features"
                )["training"]["structured_features"]
                structured_feature_weight = (
                    float(_sf_cfg["embedding_weight"])
                    if bool(_sf_cfg["enabled"]) and bool(_sf_cfg["feed_to_loss"])
                    else 0.0
                )
                evaluator = StructuredBinaryClassificationEvaluator(
                    sentences1,
                    sentences2,
                    labels,
                    name="dev",
                    show_progress_bar=False,
                    structured_features=np.asarray(dev_structured, dtype=np.float32),
                    feature_weight=structured_feature_weight,
                )

            # ── VALIDATION-LOSS DATASET (owner directive 2026-09-10) ──────
            # eval_dataset in the trainer's OWN column shape -> HF computes
            # eval_loss per eval step natively (log_history gains "eval_loss"),
            # the curve the train-vs-val loss plot draws. Same population as
            # the dev evaluator above (pos dev pairs + dev hard negatives),
            # so the loss and the AP describe the SAME dev data. Contrastive:
            # (sentence1, sentence2, label); mnrl: (anchor, positive) — the
            # same shapes the train datasets use. Triplet lane: no eval
            # dataset (build_triplets on dev would double the mining cost
            # for a lane that no longer runs; val loss stays absent and the
            # plot shows the train curve + dev-AP vline only).
            eval_ds = None
            if loss == "contrastive":
                eval_ds = Dataset.from_dict(
                    {
                        "sentence1": sentences1,
                        "sentence2": sentences2,
                        "label": labels,
                        "structured_features": dev_structured,
                    }
                )
            elif loss == "mnrl":
                eval_ds = Dataset.from_dict(
                    {
                        "anchor": [a for a, _ in dev_pairs],
                        "positive": [b for _, b in dev_pairs],
                    }
                )

            # ── modern Trainer path: real early stopping ──────────────────
            # ST 6's legacy fit() builds an HF Trainer internally but exposes
            # none of its knobs (no metric_for_best_model / load_best_model_at_end
            # / callbacks). SentenceTransformerTrainer gives us the full HF
            # contract: EarlyStoppingCallback on the dev evaluator's AP.
            # Checkpoint retention follows the shared runtime policy.
            # caps disk at ~2x model size (~1 GB L12 / ~180 MB L6) — no explosion,
            # and load_best_model_at_end restores the best epoch. Unique subdir
            # per run_tag so parallel trials never collide.
            with trace_step('training.train_one_config.fold_training'):
                from sentence_transformers import SentenceTransformerTrainer
                from sentence_transformers import (
                    SentenceTransformerTrainingArguments as STArgs,
                )
                from sentence_transformers.sentence_transformer.training_args import (
                    BatchSamplers,
                )
                from training.sampler import ControlledBatchSampler

                from training.token_inputs import ObjectiveDataCollator as PairIdDataCollator

                class ResumableSentenceTransformerTrainer(SentenceTransformerTrainer):
                    """HF Trainer plus an explicit manifest of all resume state."""

                    def add_model_card_callback(self, default_args_dict):
                        if prepared_tokens is None:
                            return super().add_model_card_callback(default_args_dict)
                        from training.token_inputs import model_card_text_dataset
                        original_train, original_eval = self.train_dataset, self.eval_dataset
                        try:
                            self.train_dataset = model_card_text_dataset(original_train)
                            self.eval_dataset = model_card_text_dataset(original_eval)
                            return super().add_model_card_callback(default_args_dict)
                        finally:
                            self.train_dataset, self.eval_dataset = original_train, original_eval

                    def get_batch_sampler(self, dataset, batch_size, drop_last, **kwargs):
                        if "pair_id" in dataset.column_names or loss == "triplet":
                            return FrozenBatchSampler(
                                fixed_sampler["epochs"], expected_rows=len(dataset),
                                batch_size=fixed_sampler["batch_size"],
                            )
                        return super().get_batch_sampler(dataset, batch_size, drop_last, **kwargs)

                    def compute_loss(
                        self,
                        model,
                        inputs,
                        return_outputs=False,
                        num_items_in_batch=None,
                    ):
                        pair_ids = inputs.pop("pair_id", None)
                        structured = inputs.pop("structured_features", None)
                        loss_fn = self.loss
                        if pair_ids is not None and hasattr(loss_fn, "set_batch_pair_ids"):
                            loss_fn.set_batch_pair_ids(pair_ids)
                        if structured is not None and hasattr(
                            loss_fn, "set_batch_structured_features"
                        ):
                            loss_fn.set_batch_structured_features(structured)
                        return super().compute_loss(
                            model,
                            inputs,
                            return_outputs=return_outputs,
                            num_items_in_batch=num_items_in_batch,
                        )

                    def _load_optimizer_and_scheduler(self, checkpoint):
                        super()._load_optimizer_and_scheduler(checkpoint)
                        optimizer_policy.validate_restored(self.optimizer)

                    def _save_checkpoint(self, model, trial):
                        if hasattr(self.loss, 'flush_tracking'):
                            self.loss.flush_tracking()
                        super()._save_checkpoint(model, trial)
                        # Duck-typed unwrap: HF wraps the model in DataParallel/DDP
                        # when multiple GPUs are visible, which makes `model`
                        # unsubscriptable and crashes the manifest writer below.
                        # A bare SentenceTransformer never exposes `.module`.
                        manifest_model = getattr(model, 'module', model)
                        checkpoint = (
                            Path(self._get_output_dir(trial=trial))
                            / f"checkpoint-{self.state.global_step}"
                        )
                        _make_checkpoint_tokenizer_portable(checkpoint)
                        _write_checkpoint_manifest(
                            checkpoint,
                            epoch=self.state.epoch,
                            global_step=self.state.global_step,
                            model=manifest_model,
                            optimizer=self.optimizer,
                            scheduler=self.lr_scheduler,
                            scaler=getattr(self.accelerator, "scaler", None),
                            trainer_state=self.state,
                            trainer_control=self.control,
                            training_args=self.args,
                        )
                        trace_artifact("checkpoint_repo", checkpoint, producer="training.training")

                # T4 can emulate BF16, but has native FP16 tensor cores. Avoid
                # selecting emulated BF16 from PyTorch's permissive default probe.
                native_bf16 = on_cuda and torch.cuda.is_bf16_supported(including_emulation=False)
                model_tag = str(model_id).rstrip("/").rsplit("/", 1)[-1]
                args_hf = STArgs(
                    output_dir=str(checkpoint_dir),
                    per_device_train_batch_size=batch_size,
                    num_train_epochs=cfg["epochs"],
                    learning_rate=cfg["lr"],
                    warmup_steps=warmup_steps,
                    weight_decay=cfg["weight_decay"],
                    lr_scheduler_type=cfg["lr_scheduler"],
                    max_grad_norm=cfg["max_grad_norm"],
                    # TASK B item 7: micro-batch accumulation (SSOT, default 1).
                    gradient_accumulation_steps=int(
                        training_cfg().advanced.gradient_accumulation_steps
                    ),
                    bf16=native_bf16,
                    fp16=on_cuda and not native_bf16,
                    # early stopping: eval every eval_steps, stop on plateau,
                    # restore the best checkpoint at the end
                    eval_strategy="steps",
                    eval_steps=eval_steps,
                    # eval batch = the SSOT encode batch (loss on dev pairs is
                    # gradient-free; same batch the dev evaluator uses)
                    per_device_eval_batch_size=runtime("batch_size_eval"),
                    metric_for_best_model="eval_dev_cosine_ap",
                    greater_is_better=True,
                    load_best_model_at_end=True,
                    save_strategy="steps",
                    save_steps=eval_steps,
                    save_total_limit=training_cfg().training.save_total_limit,  # Validated nullable SSOT; retain all when None
                    # A resumable checkpoint must retain optimizer, scheduler,
                    # RNG, and trainer state.  Model-only snapshots cannot pick
                    # up a stopped run faithfully.
                    save_only_model=False,
                    logging_strategy="steps",
                    logging_steps=eval_steps,
                    remove_unused_columns=False,
                    report_to=[],
                    seed=seed + fold_i,
                    use_cpu=not on_cuda,
                    # Controlled batch sampler (owner): offline prep
                    # determines composition; default HF sampling is
                    # used when batch_sampler.enabled is false.
                    batch_sampler=(
                        BatchSamplers.NO_DUPLICATES
                        if loss == "mnrl"
                        else BatchSamplers.BATCH_SAMPLER
                    ),
                )
                # discriminative LRs: bottom layers hold pretrained knowledge ->
                # smaller LR; top layers + pooling head adapt to the task -> full
                # LR. Per-layer multiplicative decay (0.9^k), bottom to top.
                # Dedup by tensor id: some architectures tie/share weights
                # (embeddings<->pooler etc.) — AdamW REJECTS a param in two groups.
                base_lr = cfg["lr"]
                # AUDIT FIX (round 2 F09, round 3): the single-LR fallback stays
                # (sanctioned: it is visible in stdout and the run continues),
                # but it is now QUERYABLE downstream — `lr_groups` lands in the
                # fold-metrics row ("discriminative" normal / "single"
                # fallback) so a degraded fold is distinguishable without
                # scraping stdout.
                lr_groups = "discriminative"
                try:
                    groups = _discriminative_groups(model, base_lr)
                    n_g = len(groups)
                    print(
                        f"    [optim] discriminative LR: {n_g} groups, "
                        f"bottom {groups[0]['lr']:.2e} .. top {base_lr:.2e}",
                        flush=True,
                    )
                except Exception as exc:
                    lr_groups = "single"
                    print(
                        f"    [optim] single LR fallback ({exc}) "
                        f"[lr_groups=single — recorded in the fold-metrics row]",
                        flush=True,
                    )
                    groups = [{"params": model.parameters(), "lr": base_lr}]

                from torch import optim

                from core.gpu_execution import OptimizerExecution
                optimizer_policy = OptimizerExecution(backend=training_cfg().training.optimizer_backend)
                optimizer = optim.AdamW(
                    groups, weight_decay=cfg["weight_decay"], lr=base_lr,
                    **optimizer_policy.kwargs('cuda' if on_cuda else 'cpu'),
                )
                optimizer_backend = optimizer_policy.resolved_backend('cuda' if on_cuda else 'cpu')
                print(f'    [optim] policy={optimizer_policy.backend} backend={optimizer_backend}', flush=True)

                mnrl_cfg = training_cfg().training
                loss_fn = _make_loss(
                    model,
                    loss,
                    structured_feature_weight=structured_feature_weight,
                    uniformity_weight=float(cfg["uniformity_weight"]),
                    uniformity_temperature=float(_UNIFORMITY_CFG["temperature"]),
                    uniformity_min_batch_size=int(_UNIFORMITY_CFG["min_batch_size"]),
                    label_smoothing=float(cfg["label_smoothing"]),
                    mnrl_monitoring_enabled=bool(
                        mnrl_cfg.mnrl_monitoring.enabled
                    ),
                    twin_warmup_enabled=bool(mnrl_cfg.twin_loss_warmup.enabled),
                    twin_warmup_epochs=int(mnrl_cfg.twin_loss_warmup.warmup_epochs),
                    twin_weight=float(mnrl_cfg.twin_loss_warmup.twin_weight),
                    contrastive_telemetry_enabled=_COLLECT_CONTRASTIVE_TELEMETRY,
                )
                if loss == "mnrl" and hasattr(loss_fn, "set_triple_populations"):
                    loss_fn.set_triple_populations(triple_populations)
                pair_lineage = _build_pair_lineage(
                    train_all,
                    tr_negs,
                    train_neg_sources=tr_neg_sources,
                    mask_audit=mask_audit,
                    hard_negative_mask_audit=hard_negative_mask_audit,
                    payload_metadata=payload_metadata,
                    gate_lookup=gate_lookup,
                )
                if hasattr(loss_fn, "set_pair_lineage"):
                    loss_fn.set_pair_lineage(pair_lineage)
                    loss_fn._dynamic_mask_counts = dynamic_mask_counts
                    loss_fn._dynamic_mask_counts_by_epoch = dynamic_mask_counts_by_epoch
                    loss_fn._dynamic_mask_stats_by_epoch = dynamic_mask_stats_by_epoch
                    loss_fn._dynamic_epoch_ref = dynamic_epoch_ref
                if hasattr(loss_fn, "set_total_negative_pairs"):
                    loss_fn.set_total_negative_pairs(len(tr_negs))

                callbacks = [
                    ProgressCallback(
                        wandb_ctx,
                        tracked_loss=loss_fn,
                        trace_path=_loss_trace_path(run_tag, fold_i),
                        collapse_model=model,
                        collapse_df=df,
                        collapse_payload=payload,
                        collapse_config=calibration_config,
                        collapse_batch_size=runtime("batch_size_eval"),
                    ),
                    LateEpochLrDecayCallback(
                        enabled=bool(cfg["late_epoch_decay_enabled"]),
                        start_epoch_fraction=float(
                            cfg["late_epoch_decay_start_fraction"]
                        ),
                        multiplier=float(cfg["late_epoch_decay_multiplier"]),
                    ),
                    EarlyStoppingCallback(
                        early_stopping_patience=cfg["patience"],
                        early_stopping_threshold=cfg["es_threshold"],
                    ),
                ]
                # TASK B item 1: weight EMA (default OFF).
                _ema_cfg = training_cfg().advanced.ema
                if _ema_cfg.enabled:
                    callbacks.append(
                        _WeightEmaCallback(
                            decay=float(_ema_cfg.decay),
                            warmup_updates=int(_ema_cfg.warmup_updates),
                        )
                    )
                # BATCH GRAIN (consolidated trace): one row per optimizer step.
                # Observer only — the loss hook returns the original tensor, and
                # the collector never writes the trace itself (train_one_config
                # publishes its rows through the trace's sampling caps).
                batch_trace = _BatchStepTrace(fold_i=fold_i, optimizer=optimizer)
                batch_trace.attach(loss_fn)
                callbacks.append(batch_trace)
                if not checkpoint_publication_deferred():
                    callbacks.append(DvcCheckpointCallback())
                if (
                    (ann_refresh_enabled or attribute_conflict_refresh_enabled)
                    and loss == "contrastive"
                ):
                    ann_cfg = config_section("mining", "ann")
                    callbacks.append(
                        FineTunedAnnRefreshCallback(
                            df=df,
                            payload=payload,
                            row_gtins=row_bc,
                            structured_features=structured_features,
                            train_gtins=set(tr_bc),
                            existing=tr_negs,
                            ann_state=ann_refresh_state,
                            slot_ids=range(len(train_all), len(train_all) + len(tr_negs)),
                            fold_i=fold_i,
                            run_tag=run_tag,
                            batch_size=runtime("batch_size_embed"),
                            max_seq_length=runtime("max_seq_length"),
                            model=model,
                            wandb_ctx=wandb_ctx,
                        )
                    )
                from core.training_profiler import TrainingProfiler
                training_profile = TrainingProfiler(RESULTS / 'profiles' / run_tag / f'fold{fold_i}',str(next(model.parameters()).device.type))
                if training_profile.enabled:
                    callbacks.append(training_profile.callback())
                trainer = ResumableSentenceTransformerTrainer(
                    model=model,
                    args=args_hf,
                    train_dataset=train_ds,
                    eval_dataset=eval_ds,
                    evaluator=evaluator,
                    data_collator=PairIdDataCollator(
                        preprocess_fn=model.preprocess,
                        router_mapping=args_hf.router_mapping,
                        prompts=args_hf.prompts,
                    ),
                    loss=loss_fn,
                    optimizers=(optimizer, None),  # prebuilt AdamW with
                    # discriminative LRs; scheduler=None -> HF builds warmup+linear
                    # from args, scaling our per-group LRs
                    callbacks=callbacks,
                )
                resume_checkpoint = None
                if resume:
                    candidates = sorted(
                        checkpoint_dir.glob("checkpoint-*"),
                        key=lambda path: int(path.name.removeprefix("checkpoint-")),
                    )
                    if candidates:
                        latest = candidates[-1]
                        required = _required_resume_filenames()
                        missing = [name for name in required if not (latest / name).is_file()]
                        if missing:
                            raise RuntimeError(
                                f"cannot resume {latest}: checkpoint lacks trainer state; "
                                f"missing {', '.join(missing)}. Start a new run once "
                                "to create resumable checkpoints."
                            )
                        from core.model_input import model_input_composition
                        checkpoint_manifest = json.loads((latest / training_cfg().colab.checkpoint_manifest_name).read_text())
                        if checkpoint_manifest.get("model_input") != model_input_composition().model_dump():
                            raise ValueError("resume checkpoint model input composition mismatch")
                        # Trainer state contains absolute paths from the original
                        # VM. Rebase only to the selected sibling in this restored
                        # checkpoint tree, never to another run's checkpoint.
                        state_path = latest / _trainer_state_filename()
                        restored_state = json.loads(state_path.read_text())
                        selected = restored_state.get(_trainer_best_key())
                        if selected:
                            local_selected = checkpoint_dir / Path(selected).name
                            if not local_selected.is_dir():
                                raise FileNotFoundError(f"resume selected checkpoint missing: {local_selected}")
                            restored_state[_trainer_best_key()] = str(local_selected.resolve())
                            state_path.write_text(json.dumps(restored_state, indent=2) + "\n")
                        resume_checkpoint = str(latest)
                        print(f"    [resume] fold {fold_i}: {resume_checkpoint}", flush=True)
                    else:
                        print(f"    [resume] fold {fold_i}: no checkpoint found; starting fresh", flush=True)
                try:
                    trainer.train(resume_from_checkpoint=resume_checkpoint)
                finally:
                    training_profile.close()
            with trace_step('training.train_one_config.fold_telemetry'):
                progress_callback = next(
                    callback
                    for callback in callbacks
                    if isinstance(callback, ProgressCallback)
                )
                late_lr_callback = next(
                    callback
                    for callback in callbacks
                    if isinstance(callback, LateEpochLrDecayCallback)
                )
                datapoint_coverage: dict[str, int] = {}
                if loss == "contrastive" and TRACK_DATAPOINT_USAGE:
                    enabled_dynamic_populations = {
                        population
                        for population, enabled in (
                            ("ann_finetuned", ann_refresh_enabled),
                            ("attribute_conflict", attribute_conflict_refresh_enabled),
                        )
                        if enabled
                    }
                    print(
                        "    [datapoint-sources] "
                        + ", ".join(
                            f"{name}={'enabled' if name in enabled_dynamic_populations else 'disabled_by_config'}"
                            for name in ("ann_finetuned", "attribute_conflict")
                        ),
                        flush=True,
                    )
                    datapoint_coverage = _write_datapoint_usage(
                        fold_i=fold_i,
                        pair_populations=pair_populations,
                        presentation_counts=presentation_counts,
                        pair_lineage=pair_lineage,
                        dynamic_populations=enabled_dynamic_populations,
                        run_tag=run_tag,
                        sample=sample,
                    )
                    if wandb_ctx is not None and datapoint_coverage:
                        wandb_ctx.log_metrics(
                            {
                                f"datapoint_coverage/{key}": float(value)
                                for key, value in datapoint_coverage.items()
                            }
                        )
                        wandb_ctx.set_summary(
                            {
                                f"datapoint_coverage/{key}": float(value)
                                for key, value in datapoint_coverage.items()
                            }
                        )
                if MASK_TRACK_PER_EPOCH and dynamic_mask_stats_by_epoch:
                    mask_epoch_rows = []
                    for epoch, stats in sorted(dynamic_mask_stats_by_epoch.items()):
                        presented = float(stats["negative_presented"])
                        masked_count = float(stats["masked_count"])
                        mask_epoch_rows.append(
                            {
                                "fold": fold_i,
                                "epoch": int(epoch),
                                "negative_presented": int(presented),
                                "masked_count": int(masked_count),
                                "masked_pct": masked_count / presented if presented else 0.0,
                                "mean_realized_extent": (
                                    float(stats["extent_sum"]) / masked_count
                                    if masked_count else 0.0
                                ),
                                "configured_mask_lo": float(dynamic_mask_lo),
                                "configured_mask_hi": float(dynamic_mask_hi),
                                "static_positive_masked": int(static_masked_pos),
                                "static_positive_total": int(len(train_all)),
                                "static_positive_masked_pct": float(static_positive_pct),
                            }
                        )
                    from core.common import write_visibility_log

                    write_visibility_log(
                        pd.DataFrame(mask_epoch_rows),
                        f"masking_per_epoch_fold{fold_i}.csv",
                        run_tag,
                        sample,
                    )
                    write_visibility_log(
                        pd.DataFrame(mask_epoch_rows),
                        "mask_hard_negative_visibility.csv",
                        run_tag,
                        sample,
                    )
                    if wandb_ctx is not None:
                        for row in mask_epoch_rows:
                            wandb_ctx.log_metrics(
                                {
                                    "masking/epoch": float(row["epoch"]),
                                    "masking/dynamic_negative_presented": float(row["negative_presented"]),
                                    "masking/dynamic_negative_masked": float(row["masked_count"]),
                                    "masking/dynamic_negative_masked_pct": float(row["masked_pct"]),
                                    "masking/dynamic_negative_mean_realized_extent": float(row["mean_realized_extent"]),
                                },
                            )
                if (
                    loss == "mnrl"
                    and mnrl_cfg.mnrl_monitoring.enabled
                    and hasattr(loss_fn, "mnrl_subset_rows_by_epoch")
                ):
                    mnrl_subset_rows = loss_fn.mnrl_subset_rows_by_epoch()
                    if mnrl_subset_rows:
                        from core.common import write_visibility_log

                        write_visibility_log(
                            pd.DataFrame(mnrl_subset_rows),
                            f"mnrl_subset_loss_by_epoch_fold{fold_i}.csv",
                            run_tag,
                            sample,
                        )
                        if wandb_ctx is not None:
                            for row in mnrl_subset_rows:
                                wandb_ctx.log_metrics(
                                    {
                                        "mnrl_subset/epoch": float(row["epoch"]),
                                        f"mnrl_subset/loss_{row['population']}": float(
                                            row["mean_loss"]
                                        ),
                                        f"mnrl_subset/count_{row['population']}": float(
                                            row["triple_count"]
                                        ),
                                    }
                                )
                usage_rows: list[dict] = []
                if hasattr(loss_fn, "pair_usage_rows"):
                    usage_rows = loss_fn.pair_usage_rows()
                if hasattr(loss_fn, "pair_usage_rows_by_epoch"):
                    usage_epoch_rows = loss_fn.pair_usage_rows_by_epoch()
                    if usage_epoch_rows:
                        for usage in usage_epoch_rows:
                            pair_id = int(usage["pair_id"])
                            usage["dynamic_mask_count"] = int(
                                dynamic_mask_counts_by_epoch.get(
                                    int(usage["epoch"]), {}
                                ).get(pair_id, 0)
                            )
                        from core.common import write_visibility_log

                        write_visibility_log(
                            pd.DataFrame(usage_epoch_rows),
                            f"pair_backprop_fold{fold_i}.csv",
                            run_tag,
                            sample,
                        )
            # Source accounting is computed after the fold boundary and from
            # the same pair IDs used by the loss. This directly answers how
            # many gate / targeted-attribute / attribute-conflict / random-easy
            # negatives landed in the training fold and how many received
            # selection/backprop. The source list is DERIVED from
            # DATAPOINT_POPULATION_SPEC (never a literal sub-list), and the
            # helper asserts the closure + funnel identities and raises
            # UnregisteredDatapointPopulationError on an undeclared producer
            # tag instead of silently omitting it (audit A4-2).
            source_coverage = _negative_source_accounting(
                fold_i=fold_i,
                tr_negs=tr_negs,
                tr_neg_sources=tr_neg_sources,
                usage_rows=usage_rows,
            )
            if wandb_ctx is not None:
                wandb_ctx.log_metrics(
                    {
                        f"negative_source/{key}": float(value)
                        for key, value in source_coverage.items()
                    }
                )
                wandb_ctx.set_summary(
                    {
                        f"negative_source/{key}": float(value)
                        for key, value in source_coverage.items()
                    }
                )
            coverage = (
                loss_fn.coverage_stats()
                if hasattr(loss_fn, "coverage_stats")
                else {}
            )
            if coverage:
                print(
                    f"    [loss-coverage] fold {fold_i}: "
                    f"hard-selected {int(coverage['n_train_neg_hard_selected_unique']):,}/"
                    f"{int(coverage['n_train_neg_total']):,} "
                    f"({coverage['train_neg_hard_selection_coverage']:.2%}), "
                    f"margin-active {int(coverage['n_train_neg_margin_active_unique']):,}/"
                    f"{int(coverage['n_train_neg_total']):,} "
                    f"({coverage['train_neg_margin_active_coverage']:.2%})",
                    flush=True,
                )

            # final training loss + best dev AP from the trainer's own log
            # history (the source the early-stopper actually used)
            hist = trainer.state.log_history
            # ── CONSOLIDATED TRACE: this fold's training steps, checkpoint
            # selection and early stop — plus the per-batch rows the collector
            # captured (published after the fold loop, under the trace's caps)
            # and the collapse breaches this fold's diagnostic evidenced.
            _fold_collapse_records = collapse_pair_records(
                df,
                payload,
                pair_trace_path=_collapse_pair_trace_path(run_tag, fold_i),
                guardrail=calibration_config["collapse_guardrail"],
                fold=int(fold_i),
            )
            if _collapse_pair_trace_path(run_tag, fold_i).is_file():
                _collapse_evidence_folds += 1
            _collapse_records.extend(_fold_collapse_records)
            _trace_fold_outcome(
                trace,
                fold_i=fold_i,
                hist=hist,
                trainer_state=trainer.state,
                cfg=cfg,
                planned_steps=n_steps_per_epoch * cfg["epochs"],
                best_metric_key=getattr(args_hf, "metric_for_best_model", None),
                guardrail=calibration_config["collapse_guardrail"],
                collapse_records=_fold_collapse_records,
            )
            if batch_trace is not None and batch_trace.rows:
                _batch_records.extend(batch_trace.rows)
            train_losses = [e["loss"] for e in hist if "loss" in e]
            dev_aps = [
                e["eval_dev_cosine_ap"] for e in hist if "eval_dev_cosine_ap" in e
            ]
            # validation loss curve (owner directive 2026-09-10): the dev
            # evaluator's own loss per eval step — the pair the train-vs-val
            # plot draws. Absent only when no eval ran (skipped/failed fold).
            dev_losses = [e["eval_loss"] for e in hist if "eval_loss" in e]
            dev_metric_events = [
                e for e in hist
                if e.get("epoch") is not None
                and any(
                    e.get(key) is not None
                    for key in (
                        "eval_dev_cosine_ap",
                        "eval_dev_cosine_auc",
                        "eval_dev_cosine_f1",
                        "eval_dev_cosine_precision",
                        "eval_dev_cosine_recall",
                    )
                )
            ]
            dev_aucs = [
                e["eval_dev_cosine_auc"]
                for e in dev_metric_events
                if e.get("eval_dev_cosine_auc") is not None
            ]
            dev_precisions = [
                e["eval_dev_cosine_precision"]
                for e in dev_metric_events
                if e.get("eval_dev_cosine_precision") is not None
            ]
            dev_recalls = [
                e["eval_dev_cosine_recall"]
                for e in dev_metric_events
                if e.get("eval_dev_cosine_recall") is not None
            ]
            dev_f1s = [
                e["eval_dev_cosine_f1"]
                for e in dev_metric_events
                if e.get("eval_dev_cosine_f1") is not None
            ]
            final_train_loss = train_losses[-1] if train_losses else float("nan")
            best_dev_ap = max(dev_aps) if dev_aps else float("nan")

            # W&B curve tracking: log the train/dev loss trajectory at the
            # trainer's global steps so an overfit shape is visible, rather
            # than inferring it from one final loss snapshot. The test split
            # is intentionally excluded; dev is the validation signal used
            # for early stopping and remains holdout-safe.
            if wandb_ctx is not None:
                curve_prefix = f"curve/{run_tag}/fold_{fold_i}"
                # Keep W&B bounded: one point per integer epoch, selected by
                # nearest true event epoch. Logging every sub-epoch event made
                # long HPO runs noisy without adding a useful curve.
                curve_events = [
                    event
                    for event in hist
                    if event.get("epoch") is not None
                        and any(
                            event.get(key) is not None
                            for key in (
                                "loss",
                                "eval_loss",
                                "eval_dev_cosine_ap",
                                "eval_dev_cosine_f1",
                            )
                        )
                ]
                if curve_events:
                    max_epoch = int(
                        np.ceil(max(float(event["epoch"]) for event in curve_events))
                    )
                    for epoch in range(1, max_epoch + 1):
                        nearest = min(
                            curve_events,
                            key=lambda event: abs(float(event["epoch"]) - epoch),
                        )
                        point = {f"{curve_prefix}/epoch": float(epoch)}
                        for source, target in (
                            ("loss", "train_loss"),
                            ("eval_loss", "dev_loss"),
                            ("eval_dev_cosine_ap", "dev_average_precision"),
                            ("eval_dev_cosine_f1", "dev_f1"),
                        ):
                            if nearest.get(source) is not None:
                                point[f"{curve_prefix}/{target}"] = float(
                                    nearest[source]
                                )
                        wandb_ctx.log_metrics(point)
                if train_losses and dev_losses:
                    dev_min_i = int(np.argmin(dev_losses))
                    overfit_signature = int(
                        train_losses[-1] < train_losses[0]
                        and len(dev_losses) > dev_min_i + 1
                        and dev_losses[-1] > dev_losses[dev_min_i]
                    )
                    wandb_ctx.set_summary(
                        {
                            f"{curve_prefix}/train_loss_first": float(train_losses[0]),
                            f"{curve_prefix}/train_loss_final": float(train_losses[-1]),
                            f"{curve_prefix}/dev_loss_min": float(min(dev_losses)),
                            f"{curve_prefix}/dev_loss_final": float(dev_losses[-1]),
                            f"{curve_prefix}/overfit_signature": overfit_signature,
                        }
                    )
                train_curve = [
                    (float(event["epoch"]), float(event["loss"]))
                    for event in hist
                    if event.get("epoch") is not None and event.get("loss") is not None
                ]
                dev_curve = [
                    (float(event["epoch"]), float(event["eval_loss"]))
                    for event in hist
                    if event.get("epoch") is not None and event.get("eval_loss") is not None
                ]
                if (
                    (train_curve or dev_curve)
                    and os.environ.get("EUROMONITOR_REMOTE_TRAINING") != "1"
                ):
                    import matplotlib

                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(7, 4.5))
                    if train_curve:
                        ax.plot(
                            [point[0] for point in train_curve],
                            [point[1] for point in train_curve],
                            marker="o",
                            label="train loss",
                        )
                    if dev_curve:
                        ax.plot(
                            [point[0] for point in dev_curve],
                            [point[1] for point in dev_curve],
                            marker="o",
                            label="dev loss",
                        )
                    ax.set(
                        xlabel="epoch",
                        ylabel="loss",
                        title="Train vs DEV loss by epoch",
                    )
                    ax.grid(alpha=0.25)
                    ax.legend()
                    fig.tight_layout()
                    curve_path = RESULTS / f"wandb_loss_by_epoch_{run_tag}_fold{fold_i}.png"
                    fig.savefig(curve_path, dpi=plot_dpi())
                    plt.close(fig)
                    wandb_ctx.log_image(
                        curve_path,
                        f"{curve_prefix}/loss_by_epoch",
                    )

            dynamic_negative_presented_total = int(
                sum(stats["negative_presented"] for stats in dynamic_mask_stats_by_epoch.values())
            )
            dynamic_negative_masked_total = int(
                sum(stats["masked_count"] for stats in dynamic_mask_stats_by_epoch.values())
            )

            if os.environ.get('ER_GPU_TRAINING_ONLY') == '1':
                rows.append({
                    'fold': fold_i, 'status': 'ok',
                    'best_model_checkpoint': trainer.state.best_model_checkpoint,
                    'best_metric': trainer.state.best_metric,
                    'global_step': trainer.state.global_step,
                    'calibration_status': 'deferred_local',
                    'test_eval': 'deferred_local',
                })
                print(f'[train] fold {fold_i}: GPU training complete; reporting deferred to local CPU', flush=True)
                continue

            # Every lane uses the same component-safe calibration/Rand
            # computation (_CalibrationEvaluator owns the scoring + loud
            # failures). The holdout population below remains isolated for
            # final reporting and is never used by HPO selection.
            calibration_metrics: dict[str, object] = _CalibrationEvaluator._evaluate_fold_calibration(
                calibration_config,
                fold_i=fold_i,
                sample=sample,
                model=model,
                df=df,
                payload=payload,
                structured_features=structured_features,
                calibration_pos=calibration_pos,
                calibration_neg=calibration_neg,
                row_bc=row_bc,
                structured_feature_weight=structured_feature_weight,
            )

            # ── SELECTION-MODE EXIT (test-leak fix, 2026-09-12) ───────────
            # Holdout HPO/grid folds STOP HERE: the config is ranked on
            # calibration Rand and the test quarter's eval block is never
            # entered — no pair_auc, no PR-AUC, no Youden, no pair dump,
            # not even an encode. The test quarter is read exactly once, by
            # the main train lane, so no per-config test metric can ever
            # exist to select on. Recorded LOUDLY (explicit field + print),
            # never as a silent NaN.
            if skip_test_eval or (selection_mode and HPO_SKIP_TEST_EVAL):
                print(
                    f"  [hpo] fold {fold_i}: test-side eval SKIPPED "
                    f"(selection mode — test read exactly once)",
                    flush=True,
                )
                # fold-local latency (no test encode ran: fold time IS
                # train time; same formulas as the main path's latency
                # block, minus the encode terms)
                _steps_run = trainer.state.global_step
                _steps_full = n_steps_per_epoch * cfg["epochs"]
                rows.append(
                    {
                        "fold": fold_i,
                        "status": (
                            "ok"
                            if sample or calibration_metrics.get("calibration_status") == "available"
                            else "calibration_unavailable"
                        ),
                        "test_eval": "skipped_selection_mode",
                        "objective": HPO_OBJECTIVE_HOLDOUT,
                        "best_dev_ap": best_dev_ap,
                        "final_train_loss": final_train_loss,
                        "train_loss_hist": json.dumps(
                            [round(x, 4) for x in train_losses]
                        ),
                        "train_epoch_hist": json.dumps(
                            [round(float(e["epoch"]), 4) for e in hist if "loss" in e and e.get("epoch") is not None]
                        ),
                        "dev_ap_hist": json.dumps([round(x, 4) for x in dev_aps]),
                        "dev_loss_hist": json.dumps(
                            [round(x, 4) for x in dev_losses]
                        ),
                        "dev_epoch_hist": json.dumps(
                            [round(float(e["epoch"]), 4) for e in hist if "eval_loss" in e and e.get("epoch") is not None]
                        ),
                        "n_dev_pos": len(dev_pos),
                        "n_dev_neg": len(hard_dev),
                        **coverage,
                        **datapoint_coverage,
                        "n_negative_presented": dynamic_negative_presented_total,
                        "n_masked_hard_negatives": dynamic_negative_masked_total,
                        "masked_hard_negative_pct": (
                            dynamic_negative_masked_total / dynamic_negative_presented_total
                            if dynamic_negative_presented_total
                            else 0.0
                        ),
                        "n_train": (
                            len(train_ds)
                            if loss in ("mnrl", "contrastive")
                            else len(examples or [])
                        ),
                        "s_per_step": round(
                            (time.perf_counter() - t_fold) / _steps_run
                            if _steps_run
                            else float("nan"),
                            3,
                        ),
                        "es_saved_pct": round(
                            100 * (1 - _steps_run / _steps_full)
                            if _steps_full
                            else float("nan"),
                            1,
                        ),
                        "fold_s": round(time.perf_counter() - t_fold, 1),
                        **calibration_metrics,
                    }
                )
                continue

            # eval on test (timed: encode latency is a first-class metric)
            with trace_step('training.train_one_config.holdout_eval'):
                from core.structured_features import fuse_numpy

                post_train_embedding_cache: dict[int, np.ndarray] = {}

                def _encode_fused_rows(rows: np.ndarray) -> np.ndarray:
                    """Encode each post-train payload row once per fold."""
                    unique_rows = np.unique(np.asarray(rows, dtype=int))
                    missing_rows = np.asarray(
                        [
                            row
                            for row in unique_rows
                            if int(row) not in post_train_embedding_cache
                        ],
                        dtype=int,
                    )
                    if len(missing_rows):
                        encoded = model.encode(
                            [payload[int(row)] for row in missing_rows],
                            batch_size=runtime("batch_size_eval"),
                            normalize_embeddings=True,
                            show_progress_bar=False,
                        )
                        fused = fuse_numpy(
                            encoded,
                            structured_features[missing_rows],
                            structured_feature_weight,
                        )
                        post_train_embedding_cache.update(
                            {
                                int(row): fused[position]
                                for position, row in enumerate(missing_rows)
                            }
                        )
                    return np.asarray(
                        [post_train_embedding_cache[int(row)] for row in rows]
                    )

                t_encode = time.perf_counter()
                eval_rows = np.unique(np.r_[test_pos.ravel(), hard_test.ravel()])
                row_to_idx = {int(r): i for i, r in enumerate(eval_rows)}
                tp_idx = np.array([row_to_idx[int(r)] for r in test_pos.ravel()]).reshape(
                    -1, 2
                )
                hn_idx = np.array([row_to_idx[int(r)] for r in hard_test.ravel()]).reshape(
                    -1, 2
                )
                emb = _encode_fused_rows(eval_rows)
                encode_s = time.perf_counter() - t_encode
                pos_s = _cos(emb, tp_idx)
                neg_s = _cos(emb, hn_idx)
                cross_mask = country[test_pos[:, 0]] != country[test_pos[:, 1]]

                # Split-safe random/easy negatives: construct from TEST rows only,
                # score with this fine-tuned model, and keep the population label
                # separate from the gate-mined hard-negative dump.
                random_neg_pairs = fold_inputs["random_neg_pairs"]
                random_easy_s = np.empty(0, dtype=float)
                random_easy_score_path = RESULTS / (
                    f"train_{model_tag}_{run_tag}_fold{fold_i}_random_easy_scores.csv"
                )
                if len(random_neg_pairs):
                    random_rows = np.unique(random_neg_pairs.ravel())
                    random_row_to_idx = {int(r): i for i, r in enumerate(random_rows)}
                    random_idx = np.array(
                        [random_row_to_idx[int(r)] for r in random_neg_pairs.ravel()]
                    ).reshape(-1, 2)
                    random_emb = _encode_fused_rows(random_rows)
                    random_easy_s = _cos(random_emb, random_idx)
                random_easy_status = "ok" if len(random_neg_pairs) else "empty"
                pd.DataFrame(
                    [
                        *(
                            {
                                "fold": fold_i,
                                "population": "holdout_pos",
                                "label": 1,
                                "score": float(score),
                            }
                            for score in pos_s
                        ),
                        *(
                            {
                                "fold": fold_i,
                                "population": "random_neg",
                                "label": 0,
                                "score": float(score),
                            }
                            for score in random_easy_s
                        ),
                    ]
                ).to_csv(random_easy_score_path, index=False)

                # ── HOLDOUT DISCIPLINE (owner audit 2026-09-07) ────────────────
                # Youden threshold is picked on DEV and applied to TEST. The old
                # code computed the operating point on the test scores itself —
                # an optimistic leak (the reported acc_at_thr was fitted on the
                # very pairs it scored). Dev-side rows are already encoded here:
                # reuse the SAME eval payload block for the dev pairs.
                dev_rows = np.unique(np.r_[dev_pos.ravel(), hard_dev.ravel()])
                dev_row_to_idx = {int(r): i for i, r in enumerate(dev_rows)}
                dev_tp_idx = np.array(
                    [dev_row_to_idx[int(r)] for r in dev_pos.ravel()]
                ).reshape(-1, 2)
                dev_hn_idx = np.array(
                    [dev_row_to_idx[int(r)] for r in hard_dev.ravel()]
                ).reshape(-1, 2)
                dev_emb = _encode_fused_rows(dev_rows)
                dev_pos_s = _cos(dev_emb, dev_tp_idx)
                dev_neg_s = _cos(dev_emb, dev_hn_idx)

                # Diagnostic train-side score population for class-overlap plots.
                # It is never used for threshold fitting or HPO selection. For
                # contrastive training, negatives are the exact gate negatives
                # seen by the loss; for other losses, the mined hard-train set is
                # the comparable negative population.
                train_neg_eval_pairs = (
                    _train_neg_source[
                        pairs_in_set(_train_neg_source, row_bc, tr_bc)
                    ]
                    if _train_neg_source is not None and len(_train_neg_source)
                    else hard_train
                )
                train_rows = np.unique(
                    np.r_[train_all.ravel(), train_neg_eval_pairs.ravel()]
                )
                train_row_to_idx = {int(r): i for i, r in enumerate(train_rows)}
                train_pos_idx = np.array(
                    [train_row_to_idx[int(r)] for r in train_all.ravel()]
                ).reshape(-1, 2)
                train_neg_idx = np.array(
                    [train_row_to_idx[int(r)] for r in train_neg_eval_pairs.ravel()]
                ).reshape(-1, 2)
                train_emb = _encode_fused_rows(train_rows)
                train_pos_s = _cos(train_emb, train_pos_idx)
                train_neg_s = _cos(train_emb, train_neg_idx)

                # ── latency metrics ──────────────────────────────────────────
                train_s = time.perf_counter() - t_fold - encode_s
                steps_run = trainer.state.global_step
                s_per_step = train_s / steps_run if steps_run else float("nan")
                texts_per_s = len(eval_rows) / encode_s if encode_s else float("nan")
                # steps saved by early stopping (epochs requested vs run)
                steps_full = n_steps_per_epoch * cfg["epochs"]
                es_saved_pct = (
                    100 * (1 - steps_run / steps_full) if steps_full else float("nan")
                )

                # ── GPU usage (0 on CPU) ──────────────────────────────────────
                gpu_vram_gb = gpu_peak_gb = float("nan")
                if on_cuda:
                    gpu_vram_gb = torch.cuda.memory_allocated() / 1e9
                    gpu_peak_gb = torch.cuda.max_memory_allocated() / 1e9
                    torch.cuda.reset_peak_memory_stats()

                # ── Youden threshold: picked on DEV, applied to TEST ───────────
                # (J = TPR - FPR). The threshold the fold would SHIP with is
                # chosen on validation, never fitted on the holdout it scores.
                _all = np.r_[pos_s, neg_s]
                _y = np.r_[np.ones(len(pos_s)), np.zeros(len(neg_s))]
                _dev_all = np.r_[dev_pos_s, dev_neg_s]
                _dev_y = np.r_[
                    np.ones(len(dev_pos_s)), np.zeros(len(dev_neg_s))
                ]

                _thr = youden_threshold(_dev_all, _dev_y)
                _pred_at_thr = (_all >= _thr).astype(int)
                _acc = float(
                    ((_y == 1) & (_pred_at_thr == 1)).sum()
                    + ((_y == 0) & (_pred_at_thr == 0)).sum()
                ) / len(_y) if len(_y) else float("nan")

                # PR-AUC + F1 at the FIXED operating threshold (SSOT):
                # ROC AUC alone hides class imbalance and threshold choice;
                # the lane reports PR-AUC + the F1 the pipeline would ship with
                from sklearn.metrics import average_precision_score, f1_score

                from core.common import load_config as _lc

                _pr_auc = float(average_precision_score(_y, _all))
                # TASK B item 2: post-hoc temperature scaling + ECE/Brier +
                # reliability artifact. The temperature is fitted on the DEV
                # carve ONLY; the test quarter is scored with it, never used to
                # fit (leakage discipline). Default OFF.
                _calibration_fields: dict[str, float] = {}
                _cal_cfg = training_cfg().advanced.calibration
                if _cal_cfg.enabled and len(_dev_all):
                    from training.advanced import calibration_report

                    _cal = calibration_report(
                        _dev_all,
                        _dev_y.astype(int),
                        _all,
                        _y.astype(int),
                        n_bins=int(_cal_cfg.n_bins),
                        min_temperature=float(_cal_cfg.min_temperature),
                        max_temperature=float(_cal_cfg.max_temperature),
                        fit=bool(_cal_cfg.temperature_scaling),
                    )
                    _calibration_fields = {
                        f"calibration_{key}": value
                        for key, value in _cal.items()
                        if key != "reliability"
                    }
                    _reliability_path = (
                        RESULTS / "logs" / run_tag / f"reliability_fold{fold_i}.json"
                    )
                    _reliability_path.parent.mkdir(parents=True, exist_ok=True)
                    _reliability_path.write_text(
                        json.dumps(_cal["reliability"], sort_keys=True), encoding="utf-8"
                    )
                # ══════════════════════════════════════════════════════════════
                # HOLDOUT RETRIEVAL (ER-346) — two protocols, both reported
                # ══════════════════════════════════════════════════════════════
                # OLD PROTOCOL (retained, renamed, and PROVEN degenerate by its own
                # coverage record below): the pool was np.vstack([test_pos,
                # hard_test]) grouped by source sku_id.  Negatives are anchored
                # only at the single representative row per gtin, so 5,292 of
                # 5,847 holdout queries saw EXACTLY their own positive and no query
                # ever saw more than 6 candidates (< max(ks)=10).  With the positive
                # first in a stable sort, a perfect oracle and a constant scorer
                # both read Hits@1 = Precision@1 = Recall@1 = Recall@5 = Recall@10
                # = 1.0.  Its coverage fields (share_queries_pool_le_max_k = 1.0,
                # trustworthy = 0) are what make that visible in the CSV.
                _ks = tuple(_lc()["evaluation"]["retrieval_ks"])
                _eval_pairs = np.vstack([test_pos, hard_test])
                _query_ids = np.asarray(
                    [str(df["sku_id"].iloc[int(i)]) for i in _eval_pairs[:, 0]],
                    dtype=str,
                )
                _old_protocol = {
                    f"old_protocol_{name}": value
                    for name, value in ranking_at_k_by_query(
                        _y, _all, _query_ids, _ks
                    ).items()
                }
                _old_protocol.update(
                    ranking_coverage(
                        _y,
                        _query_ids,
                        _ks,
                        prefix="old_protocol_",
                        tie_break="stable_input_order_positive_first",
                    )
                )

                # NEW PROTOCOL: every query gets its OWN positive plus
                # ``competitors_per_query(ks)`` competing canonicals from OTHER
                # components of the SAME test fold.  N = 10 * max(ks) - 1 = 99 for
                # ks=[1,5,10] — the floor is N >= max(ks) (pool strictly larger than
                # the largest K) and the chosen N puts an informationless scorer's
                # Recall@10 at 10/100 = 0.10, one decade of usable range below 1.0.
                # The correctness floor for the FIVE bare schema-pinned columns
                # (hits_at_1 / precision_at_k / recall_at_k) now carries these
                # corrected values.
                _n_competitors = competitors_per_query(_ks)
                _retrieval_pool = fold_inputs["retrieval_pool"]
                _t_pool = time.perf_counter()
                _pool_rows = np.unique(_retrieval_pool.pairs.ravel())
                # Pool rows are payload indices encoded through the SAME fused
                # cache the rest of the fold uses, so the added cost is the number
                # of UNIQUE rows (bounded by the fold's canonical count), never
                # queries x N.
                _pool_added_rows = added_encode_rows(_pool_rows, eval_rows)
                _pool_emb = _encode_fused_rows(_pool_rows)
                _pool_idx = np.searchsorted(_pool_rows, _retrieval_pool.pairs).reshape(-1, 2)
                _pool_scores = _cos(_pool_emb, _pool_idx)
                _pool_encode_s = time.perf_counter() - _t_pool
                _retrieval = ranking_at_k_by_query(
                    _retrieval_pool.labels,
                    _pool_scores,
                    _retrieval_pool.query_keys,
                    _ks,
                )
                _retrieval.update(
                    ranking_coverage(
                        _retrieval_pool.labels,
                        _retrieval_pool.query_keys,
                        _ks,
                        prefix="retrieval_",
                        tie_break="seeded_within_query_permutation",
                    )
                )
                _retrieval.update(
                    {
                        f"retrieval_{name}": value
                        for name, value in _retrieval_pool.coverage.items()
                    }
                )
                _retrieval["retrieval_pool_unique_rows"] = len(_pool_rows)
                _retrieval["retrieval_pool_added_encode_rows"] = int(_pool_added_rows)
                _retrieval["retrieval_pool_encode_s"] = round(_pool_encode_s, 1)
                _top_k = max(_ks)
                print(
                    f"  [retrieval] fold {fold_i}: {_retrieval_pool.coverage['pool_queries_evaluated']:,}"
                    f"/{_retrieval_pool.coverage['pool_queries_requested']:,} queries pooled | "
                    f"pool sizes {_retrieval_pool.coverage['pool_size_min']}-"
                    f"{_retrieval_pool.coverage['pool_size_max']} "
                    f"(target 1+{_n_competitors}) | unique rows +{_pool_added_rows:,}"
                    f" | Recall@{_top_k} {_retrieval[f'recall_at_{_top_k}']:.4f} vs chance "
                    f"{_retrieval[f'retrieval_chance_recall_at_{_top_k}']:.4f} | "
                    f"old protocol Recall@{_top_k} "
                    f"{_old_protocol[f'old_protocol_recall_at_{_top_k}']:.4f} "
                    f"(share_pool_le_max_k="
                    f"{_old_protocol['old_protocol_share_queries_pool_le_max_k']:.4f})",
                    flush=True,
                )

                _fixed_thr = float(_lc()["split"]["fixed_threshold"])
                _pred = (_all >= _fixed_thr).astype(int)
                _f1_fixed = float(f1_score(_y, _pred, zero_division=0))
                _prec_fixed = float(
                    (_y[_pred == 1] == 1).mean() if (_pred == 1).any() else 0.0
                )
                _rec_fixed = float((_pred[_y == 1] == 1).mean() if (_y == 1).any() else 0.0)
                # metric column names follow the SSOT threshold (the historical
                # "f1_at_0.55" names hardcoded 0.55 while the value was already
                # config-driven — a threshold change would have made every CSV
                # header lie). Consumers key on f"*_at_{thr:g}".
                _thr_key = f"{_fixed_thr:g}"

                # ── 07-series schema fields (owner ruling): the report plots read
                # precision at 90% recall with its audit triple (TP/FP/threshold)
                # from 07b/07c/07d CSVs; compute them from the SAME score
                # population as pr_auc so those CSVs regenerate from src/training/train runs.
                _target_recall = float(
                    calibration_config["rand_matching"]["target_recall"]
                )
                _prec90, _rec90, _tp90, _fp90, _thr90 = _CalibrationEvaluator._precision_at_recall_audit(_y, _all, _target_recall)
                # Same SSOT doctrine as _thr_key above: the recall-tied KEYS must
                # follow the configured target_recall, not a literal "90pct" —
                # otherwise a retune writes a 95%-recall number under a 90% header.
                _recall_key = recall_column_suffix(_target_recall)

                row = {
                    "fold": fold_i,
                    "status": "ok",
                    "auc": _auc(pos_s, neg_s),
                    # dev-picked Youden applied to test (holdout discipline);
                    # youden_thr_test_descriptive = the threshold argmax ON test
                    # scores — reported ONLY as the leak diagnostic (how much
                    # the old protocol flattered itself), never as the ship point
                    "youden_thr": _thr,
                    "youden_thr_test_descriptive": youden_threshold(_all, _y),
                    "acc_at_thr": _acc,
                    "pr_auc": _pr_auc,
                    # 07-schema: AP under the same name the plots expect
                    "average_precision": _pr_auc,
                    # TASK B item 2 fields (dev-fit temperature, dev/test ECE,
                    # test Brier); empty when advanced.calibration is OFF.
                    **_calibration_fields,
                    # The five bare, schema-pinned retrieval columns
                    # (hits_at_1/precision_at_k/recall_at_k) now carry the
                    # CORRECTED per-query pool; the degenerate historical pool is
                    # retained under old_protocol_* so the improvement is auditable
                    # in the same row, and BOTH carry their own coverage record.
                    **_retrieval,
                    **_old_protocol,
                    f"precision_at_{_recall_key}_recall": _prec90,
                    f"tp_at_{_recall_key}_recall": _tp90,
                    f"fp_at_{_recall_key}_recall": _fp90,
                    f"threshold_at_{_recall_key}_recall": _thr90,
                    f"f1_at_{_thr_key}": _f1_fixed,
                    f"precision_at_{_thr_key}": _prec_fixed,
                    f"recall_at_{_thr_key}": _rec_fixed,
                    "auc_cross": _auc(pos_s[cross_mask], neg_s)
                    if cross_mask.any()
                    else float("nan"),
                    "final_train_loss": final_train_loss,
                    "best_dev_ap": best_dev_ap,
                    # full curves for the train-vs-val loss plot (json: csv-column-safe)
                    "train_loss_hist": json.dumps([round(x, 4) for x in train_losses]),
                    "train_epoch_hist": json.dumps(
                        [round(float(e["epoch"]), 4) for e in hist if "loss" in e and e.get("epoch") is not None]
                    ),
                    "dev_ap_hist": json.dumps([round(x, 4) for x in dev_aps]),
                    "dev_auc_hist": json.dumps([round(x, 4) for x in dev_aucs]),
                    "dev_precision_hist": json.dumps([round(x, 4) for x in dev_precisions]),
                    "dev_recall_hist": json.dumps([round(x, 4) for x in dev_recalls]),
                    "dev_f1_hist": json.dumps([round(x, 4) for x in dev_f1s]),
                    "dev_metric_epoch_hist": json.dumps(
                        [round(float(e["epoch"]), 4) for e in dev_metric_events]
                    ),
                    "dev_loss_hist": json.dumps([round(x, 4) for x in dev_losses]),
                    "dev_epoch_hist": json.dumps(
                        [round(float(e["epoch"]), 4) for e in hist if "eval_loss" in e and e.get("epoch") is not None]
                    ),
                    # ── pair accounting (failure-analysis ground) ────────────
                    "n_pos": len(test_pos),
                    "n_neg": len(hard_test),
                    "n_random_easy_neg": len(random_neg_pairs),
                    "random_easy_status": random_easy_status,
                    "random_easy_available": int(bool(len(random_neg_pairs))),
                    "random_easy_score_csv": str(random_easy_score_path),
                    **coverage,
                    **source_coverage,
                    **datapoint_coverage,
                    "n_train_pos": len(train_pos),
                    # hp rows in train = total minus the GATE rows actually kept.
                    # Subtracting the UNSAMPLED train_pos went NEGATIVE under
                    # train_frac<1 (measured -426 on the frac0.25 run).
                    "n_train_hp": int(len(train_all) - n_gate_kept),
                    # contrastive: labeled negatives = gate hard-no pairs in
                    # train gtins (the label=0 half of the dataset); mnrl:
                    # in-batch only (counted separately below); triplet: mined
                    "n_train_neg": (
                        len(tr_negs)
                        if loss == "contrastive"
                        else (0 if loss == "mnrl" else len(hard_train))
                    ),
                    "n_train_hard_neg": n_train_hard_neg,
                    "n_train_random_easy_neg": n_train_random_easy_neg,
                    "n_train_random_easy_unique_candidates": (
                        random_easy_unique_candidates
                    ),
                    "train_random_easy_to_hard_ratio": (
                        n_train_random_easy_neg / n_train_hard_neg
                        if n_train_hard_neg
                        else 0.0
                    ),
                    "n_hp_in_train": (
                        len(hp_pairs[pairs_in_set(hp_pairs, row_bc, tr_bc)])
                        if use_hp and hp_pairs is not None and len(hp_pairs)
                        else 0
                    ),
                    # MNRL negatives are in-batch: each anchor sees every other
                    # example's positive as a negative -> (batch_size - 1) per
                    # anchor, ~batch*n_train per epoch
                    "n_mnrl_neg_per_anchor": (
                        BATCH_SIZE_CUDA if on_cuda else BATCH_SIZE_CPU
                    )
                    - 1
                    if loss == "mnrl"
                    else 0,
                    "n_dev_pos": len(dev_pos),
                    "n_dev_neg": len(hard_dev),
                    "n_negative_presented": dynamic_negative_presented_total,
                    "n_masked_hard_negatives": dynamic_negative_masked_total,
                    "masked_hard_negative_pct": (
                        dynamic_negative_masked_total / dynamic_negative_presented_total
                        if dynamic_negative_presented_total
                        else 0.0
                    ),
                    "n_train": (
                        len(train_ds)
                        if loss in ("mnrl", "contrastive")
                        else len(examples or [])
                    ),
                    # optimizer geometry actually used (F09): "discriminative"
                    # = per-layer groups; "single" = the visible single-LR
                    # fallback after a _discriminative_groups failure
                    "lr_groups": lr_groups,
                    "warmup_steps": warmup_steps,
                    "weight_decay": float(cfg["weight_decay"]),
                    "projection_dropout": float(cfg["projection_dropout"]),
                    "label_smoothing": (
                        float(cfg["label_smoothing"])
                        if loss == "contrastive"
                        else 0.0
                    ),
                    "uniformity_weight": float(cfg["uniformity_weight"]),
                    "late_epoch_decay_enabled": bool(
                        cfg["late_epoch_decay_enabled"]
                    ),
                    "late_epoch_decay_start_fraction": float(
                        cfg["late_epoch_decay_start_fraction"]
                    ),
                    "late_epoch_decay_multiplier": float(
                        cfg["late_epoch_decay_multiplier"]
                    ),
                    "late_epoch_decay_applied": int(late_lr_callback.applied),
                    "late_epoch_decay_applied_epoch": (
                        float(late_lr_callback.applied_epoch)
                        if late_lr_callback.applied_epoch is not None
                        else float("nan")
                    ),
                    **{
                        f"train_{key}": value
                        for key, value in progress_callback.latest_collapse_metrics.items()
                    },
                    # latency
                    "s_per_step": round(s_per_step, 3),
                    "texts_per_s_encode": round(texts_per_s, 1),
                    "encode_s": round(encode_s, 1),
                    "train_s": round(train_s, 1),
                    "es_saved_pct": round(es_saved_pct, 1),
                    # device
                    "gpu_vram_gb": round(gpu_vram_gb, 2),
                    "gpu_peak_gb": round(gpu_peak_gb, 2),
                    "fold_s": round(time.perf_counter() - t_fold, 1),
                    **calibration_metrics,
                }
                rows.append(row)
                # ── CONSOLIDATED TRACE: the fold's final evaluation, as reported
                trace.add(
                    "fold",
                    "test_metrics",
                    scope=SCOPE_ENTITY,
                    key=fold_i,
                    in_count=_optional_int(row.get("n_pos")) + _optional_int(row.get("n_neg"))
                    if row.get("n_pos") is not None or row.get("n_neg") is not None
                    else None,
                    out_count=None,
                    reason=(
                        "final per-fold evaluation: the test-side pairs scored at "
                        "the dev-picked operating threshold (test read once)"
                    ),
                    detail={
                        "fold": int(fold_i),
                        "status": row.get("status"),
                        "auc": _optional_float(row.get("auc")),
                        "pr_auc": _optional_float(row.get("pr_auc")),
                        "average_precision": _optional_float(row.get("average_precision")),
                        "acc_at_thr": _optional_float(row.get("acc_at_thr")),
                        "youden_thr": _optional_float(row.get("youden_thr")),
                        "n_pos": _optional_int(row.get("n_pos")),
                        "n_neg": _optional_int(row.get("n_neg")),
                        "best_dev_ap": _optional_float(row.get("best_dev_ap")),
                        "final_train_loss": _optional_float(row.get("final_train_loss")),
                        "lr_groups": row.get("lr_groups"),
                        "es_saved_pct": _optional_float(row.get("es_saved_pct")),
                        "fold_s": _optional_float(row.get("fold_s")),
                        "calibration_status": row.get("calibration_status"),
                        "test_eval": row.get("test_eval"),
                        "train_collapse_median_cosine": _optional_float(
                            row.get("train_collapse_median_cosine")
                        ),
                        "train_collapse_crossing_rate": _optional_float(
                            row.get("train_collapse_crossing_rate")
                        ),
                        "train_collapse_healthy": _optional_int(
                            row.get("train_collapse_healthy")
                        ),
                    },
                    source="training.training fold metrics row",
                )

            # ── per-fold pair dump: the failure-analysis ground truth ─────
            # every scored pair with its sku_ids, score, label and stratum —
            # this is what makes "why did fold 2 miss 300 cross-country
            # positives" answerable post-hoc instead of guesswork.
            # payload layout: [0, len(df)) = sku rows, then canonicals
            # (one per GTIN, contiguous), then masked-anchor copies. The
            # copy block starts right after the canonical block, and every
            # masked copy appears as a FIRST endpoint of a pos pair (its
            # anchor is a df row) — so the smallest pos-first-endpoint
            # >= len(df) marks the canonical/masked boundary.
            with trace_step('training.train_one_config.pair_dump'):
                _masked_firsts = [
                    int(i) for i in pos[:, 0] if i >= len(df)
                ]
                _canon_end = (
                    min(_masked_firsts) if _masked_firsts else len(payload)
                )
                n_canon_entries = max(0, _canon_end - len(df))
                pair_records = []
                _attribute_cache: dict[int, dict[str, object]] = {}

                from core.attribute_conflicts import (
                    canonical_attribute_info,
                    conflict_columns,
                    sku_attribute_info,
                )
                from core.common import F as _F

                if _canon_attrs is None:
                    _canon_frame = pd.read_csv(
                        _F["canonical_records"], dtype=str, keep_default_na=False
                    )
                    _canon_attrs = {
                        str(record["gtin"]): canonical_attribute_info(record)
                        for record in _canon_frame.to_dict("records")
                    }

                def _attribute_info(index: int) -> dict[str, object]:
                    """Use structured canonical attributes for canonical endpoints."""
                    if index in _attribute_cache:
                        return _attribute_cache[index]
                    if index < len(df):
                        info = sku_attribute_info(
                            df["sku_name_eng"].iloc[index], df["attribute"].iloc[index],
                            df["description_short_eng"].iloc[index]
                            if "description_short_eng" in df else "",
                        )
                    else:
                        gtin = str(row_bc[index])
                        if gtin not in _canon_attrs:
                            raise KeyError(
                                f"payload endpoint {index} has gtin {gtin!r} "
                                "but no canonical attribute record"
                            )
                        info = _canon_attrs[gtin]
                    _attribute_cache[index] = info
                    return info

                def _attribute_conflicts(a: int, b: int) -> dict[str, object]:
                    return conflict_columns(_attribute_info(a), _attribute_info(b))

                df_sku_ids = df["sku_id"].tolist()
                df_retailers = df["retailer"].tolist()

                def _sku_id(i, _n_canon=n_canon_entries):
                    if i < len(df):
                        return str(df_sku_ids[i])
                    bc_i = str(row_bc[i]) if i < len(row_bc) else ""
                    if i < len(df) + _n_canon:
                        return f"canon#{bc_i or i}"
                    return f"masked#{i}"

                def _retailer(i):
                    return str(df_retailers[i]) if i < len(df) else "-"

                for pairs, scores, label, a_col, b_col in (
                    (test_pos, pos_s, 1, None, None),
                    (hard_test, neg_s, 0, None, None),
                ):
                    for k in range(len(pairs)):
                        a, b = int(pairs[k, 0]), int(pairs[k, 1])
                        # payload space: [0, len(df)) = sku rows, then canonicals
                        # (one per GTIN), then masked-anchor copies. The old dump
                        # labeled EVERYTHING past df as "masked#N" — canonical
                        # targets (the majority, ~85% of pos endpoints) were
                        # mislabeled. Label by what the entry actually is.
                        pair_records.append(
                            {
                                "fold": fold_i,
                                "label": label,
                                "sku_id_a": _sku_id(a),
                                "sku_id_b": _sku_id(b),
                                "score": float(scores[k]),
                                "cross_country": bool(country[a] != country[b]),
                                "retailer_a": _retailer(a),
                                "retailer_b": _retailer(b),
                                **_attribute_conflicts(a, b),
                            }
                        )
                model_tag = model_id.split("/")[-1]
                # pair dump carries run_tag (owner audit 2026-09-07): the bare
                # model_tag name collided across runs — a 1k --sample run
                # overwrote a 3h full run's pair dump (14,414 rows -> 1,079).
                pd.DataFrame(pair_records).to_csv(
                    RESULTS / f"train_{model_tag}_{run_tag}_fold{fold_i}_pairs.csv",
                    index=False,
                )
                pd.DataFrame(
                    [
                        *({"fold": fold_i, "label": 1, "score": float(s)} for s in train_pos_s),
                        *({"fold": fold_i, "label": 0, "score": float(s)} for s in train_neg_s),
                    ]
                ).to_csv(
                    RESULTS / f"train_{model_tag}_{run_tag}_fold{fold_i}_train_scores.csv",
                    index=False,
                )

            # Track the train-versus-holdout geometry directly in W&B. The
            # CSVs remain the source of truth, but these summaries and the
            # overlay make a generalization gap visible without downloading
            # the artifact first.
            score_plot_path = RESULTS / (
                f"wandb_score_distributions_{run_tag}_fold{fold_i}.png"
            )
            score_groups = {
                "train/label_0": train_neg_s,
                "train/label_1": train_pos_s,
                "holdout/label_0": neg_s,
                "holdout/label_1": pos_s,
            }
            score_metrics = {
                f"scores/fold_{fold_i}/{name.replace('/', '_')}_median": float(
                    np.median(values)
                )
                for name, values in score_groups.items()
                if len(values)
            }
            score_metrics.update(
                {
                    f"scores/fold_{fold_i}/{name.replace('/', '_')}_mean": float(
                        np.mean(values)
                    )
                    for name, values in score_groups.items()
                    if len(values)
                }
            )
            # Final decision metrics are separate from the score-distribution
            # medians so W&B exposes the business operating point explicitly.
            score_metrics.update(
                {
                    f"metrics/fold_{fold_i}/auc": float(_auc(pos_s, neg_s)),
                    f"metrics/fold_{fold_i}/average_precision": _pr_auc,
                    f"metrics/fold_{fold_i}/accuracy_at_youden": _acc,
                    f"metrics/fold_{fold_i}/f1_at_{_thr_key}": _f1_fixed,
                    f"metrics/fold_{fold_i}/precision_at_{_thr_key}": _prec_fixed,
                    f"metrics/fold_{fold_i}/recall_at_{_thr_key}": _rec_fixed,
                    f"metrics/fold_{fold_i}/precision_at_{_recall_key}_recall": _prec90,
                    f"metrics/fold_{fold_i}/threshold_at_{_recall_key}_recall": _thr90,
                }
            )
            if wandb_ctx is not None and score_metrics:
                wandb_ctx.log_metrics(score_metrics)
                wandb_ctx.set_summary(score_metrics)

            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            finite_groups = {
                name: np.asarray(values, dtype=float)[
                    np.isfinite(np.asarray(values, dtype=float))
                ]
                for name, values in score_groups.items()
            }
            nonempty = [values for values in finite_groups.values() if len(values)]
            if (
                nonempty
                and os.environ.get("EUROMONITOR_REMOTE_TRAINING") != "1"
            ):
                all_scores = np.concatenate(nonempty)
                lo, hi = float(np.min(all_scores)), float(np.max(all_scores))
                bins = np.linspace(lo, hi, 31) if hi > lo else 30
                fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
                for ax, split in zip(axes, ("train", "holdout"), strict=True):
                    for label, color in ((0, "tab:orange"), (1, "tab:blue")):
                        values = finite_groups[f"{split}/label_{label}"]
                        if len(values):
                            ax.hist(
                                values,
                                bins=bins,
                                alpha=0.55,
                                density=True,
                                label=f"label {label}",
                                color=color,
                            )
                    ax.set_title(split)
                    ax.set_xlabel("cosine similarity")
                    ax.grid(alpha=0.25)
                    ax.legend()
                axes[0].set_ylabel("density")
                fig.suptitle("Train vs holdout score distributions")
                fig.tight_layout()
                fig.savefig(score_plot_path, dpi=plot_dpi())
                plt.close(fig)
                if wandb_ctx is not None:
                    wandb_ctx.log_image(
                        score_plot_path,
                        f"scores/fold_{fold_i}/train_vs_holdout_distributions",
                    )

            dev = f"gpu {gpu_peak_gb:.1f}GB peak" if on_cuda else "cpu"
            print(
                f"  fold {fold_i}: loss={row['final_train_loss']:.4f} "
                f"acc@dev-youden{row['youden_thr']:.2f}={row['acc_at_thr']:.4f} "
                f"AUC={row['auc']:.4f} cross={row['auc_cross']:.4f} "
                f"PR-AUC={row['pr_auc']:.4f} "
                f"F1@{_fixed_thr:g}={_f1_fixed:.4f} "
                f"P@{_fixed_thr:g}={_prec_fixed:.4f} "
                f"R@{_fixed_thr:g}={_rec_fixed:.4f} "
                f"| best_dev_ap={row['best_dev_ap']:.4f} "
                f"| {row['s_per_step']:.2f}s/step {row['texts_per_s_encode']:.0f} txt/s "
                f"ES-saved {row['es_saved_pct']:.0f}% [{dev}] ({row['fold_s']}s)",
                flush=True,
            )
        except Exception as exc:
            tb = traceback.format_exc()
            if "out of memory" in str(exc).lower():
                print(f"  [cuda-oom] fold {fold_i} | {_format_telemetry(_runtime_telemetry())}\n{tb}", flush=True)
            print(f"  fold {fold_i}: FAILED\n{tb}", flush=True)
            if selection_mode or isinstance(
                exc, (RequiredCalibrationError, CalibrationEvaluatorError)
            ):
                raise
            # ── CONSOLIDATED TRACE: a failed fold is part of the run's story
            trace.add(
                "fold",
                "failed",
                scope=SCOPE_ENTITY,
                key=fold_i,
                in_count=None,
                out_count=None,
                reason="fold raised; the failure is recorded, never silent",
                detail={
                    "fold": int(fold_i),
                    "error": f"{type(exc).__name__}: {exc}",
                    "cuda_oom": int("out of memory" in str(exc).lower()),
                },
                source="training.train_one_config fold loop",
            )
            rows.append({"fold": fold_i, "status": "failed", "traceback": tb})

    # ── CONSOLIDATED TRACE: publish the two bounded grains ONCE, so the stage
    # keeps a single commit and the volume is governed by core.tracing's caps
    # rather than by the fold count. Order = flow order: batches (inside the
    # folds) then the collapse evidence they produced.
    _batch_summary = _emit_batch_rows(
        trace,
        _batch_records,
        source=f"training.training fold batches (run_tag={run_tag})",
        total_cap=batch_entity_cap,
    )
    trace.add(
        "batch",
        "capture",
        in_count=len(_batch_records),
        out_count=None,
        reason=(
            "one row per optimizer step (== one batch at the configured batch "
            "size), observed read-only at on_step_end; published through "
            "core.tracing's entity sampling so a 38k-step run stays readable"
        ),
        detail={
            "captured_rows": len(_batch_records),
            "census_buckets": _batch_summary["per_reason"],
            "sampled": _batch_summary["sampled"],
            "omitted": _batch_summary["omitted"],
            "entity_cap": batch_entity_cap,
            "sampling_contract": "core.tracing ENTITY_ROW_CAP / ENTITY_SAMPLE_PER_REASON",
            "loss_observed_by": "loss.compute_loss_from_embeddings (returns unchanged)",
            "grad_norm": None,
            "grad_norm_note": (
                "not exposed by the HF step callback; recorded at log cadence "
                "in the step.metrics rows"
            ),
        },
        source="training.training _BatchStepTrace",
    )
    guardrail_cfg = calibration_config["collapse_guardrail"]
    if bool(guardrail_cfg["enabled"]):
        _emit_collapse_rows(
            trace,
            _collapse_records,
            guardrail=guardrail_cfg,
            source=f"training.uniformity collapse diagnostic (run_tag={run_tag})",
            evidence_folds=_collapse_evidence_folds,
        )
    else:
        trace.add(
            "collapse",
            "disabled",
            in_count=None,
            out_count=0,
            reason=(
                "collapse_guardrail.enabled is false, so no unrelated-pair "
                "diagnostic ran and there is no per-sample collapse evidence "
                "to publish (this row states that explicitly)"
            ),
            detail={
                "enabled": False,
                "operating_threshold": float(guardrail_cfg["operating_threshold"]),
            },
            source="config collapse_guardrail",
        )
    # D7 telemetry: aggregate load_config deepcopy cost per fold, one line each.
    _fold_aggregate: dict[str, float] = {}
    for _key, _seconds in _CFG_DEEPCOPY_TOTALS.items():
        _prefix = _key.split(".", 1)[0]
        _fold_aggregate[_prefix] = _fold_aggregate.get(_prefix, 0.0) + _seconds
    for _prefix in sorted(_fold_aggregate):
        emit_timing(
            f"[timing] training.config_deepcopy fold={_prefix} "
            f"seconds={_fold_aggregate[_prefix]:.6f}"
        )
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# Optuna
# ═══════════════════════════════════════════════════════════════════════════
class _HpoStream:
    """One owner for the TPE sweep stream (OPUNA mode --hpo --n-trials).

    Small SR methods, each verbatim-from-run_hpo in behavior:
      * `_retain_hpo_champion`  per-model champion artifact retention
      * `_resolve_study`        study/storage/control-plane resolution
      * `_optimize`             resume-safe prior-trial accounting + optimize
      * `_publish_best`         train_<model>_hpo_best.json publication
    `run_hpo` stays the pinned API surface; this class owns the steps.
    """

    @staticmethod
    def _retention_artifacts(
        tag: str, model_tag: str, fold_numbers: list[int]
    ) -> list[Path]:
        """One trial's bulky local artifacts: fold logs + checkpoints + dumps."""
        paths = [RESULTS / "logs" / tag]
        paths.extend(
            artifact(
                "checkpoint_repo",
                {"model_tag": model_tag, "run_tag": tag, "fold": fold, "step": 0},
            ).parent
            for fold in fold_numbers
        )
        paths.extend(RESULTS.glob(f"train_{model_tag}_{tag}_fold*_pairs.csv"))
        return paths

    @staticmethod
    def _retention_remove(paths: list[Path]) -> None:
        import shutil

        for path in paths:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)

    @staticmethod
    def _retain_hpo_champion(
        *, model_id: str, run_tag: str, value: float, folds: list[int]
    ) -> bool:
        """Keep exactly one completed HPO trial's bulky local artifacts per model.

        Optuna may complete two trials concurrently.  The per-model lock makes
        comparison, removal of the previous champion, and champion-record update
        one transaction (invariant: compare -> prune/replace -> write record
        under ONE external lock, so a concurrent trial can never observe a
        record-less or two-champion state).  This mode deliberately retains
        local artifacts only; DVC checkpoint publishing is disabled by the HPO
        launcher.
        """
        import tempfile

        model_tag = model_id.rstrip("/").rsplit("/", 1)[-1]
        record = RESULTS / f"hpo_{model_tag}_champion.json"
        lock_path = RESULTS / f".hpo-{model_tag}-retention.lock"
        def artifacts(tag, fold_numbers):
            return _HpoStream._retention_artifacts(tag, model_tag, fold_numbers)

        remove = _HpoStream._retention_remove

        with lock_path.open("w", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                previous = json.loads(record.read_text(encoding="utf-8")) if record.is_file() else None
                if previous is not None and float(previous["value"]) >= value:
                    remove(artifacts(run_tag, folds))
                    print(
                        f"[hpo-retention] pruned trial {run_tag}: {value:.6f} "
                        f"<= champion {previous['value']:.6f}",
                        flush=True,
                    )
                    return False
                if previous is not None:
                    remove(artifacts(previous["run_tag"], list(previous["folds"])))
                payload = {"model": model_id, "run_tag": run_tag, "value": value, "folds": folds}
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=RESULTS,
                    prefix=f".{record.name}.", suffix=".tmp", delete=False,
                ) as handle:
                    json.dump(payload, handle, indent=2, sort_keys=True)
                    temporary = Path(handle.name)
                os.replace(temporary, record)
                print(f"[hpo-retention] champion {run_tag}: {value:.6f}", flush=True)
                return True
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _control_plane_storage():
        """PostgreSQL study name + storage when OPTUNA_STORAGE_URL is set."""
        if not os.environ.get("OPTUNA_STORAGE_URL"):
            return None, None
        from training.hpo_control_plane import (
            create_storage,
            generation_study_name,
            storage_from_environment,
        )

        generation_id = os.environ.get("EUROMONITOR_HPO_GENERATION_ID", "").strip()
        model_key = os.environ.get("EUROMONITOR_HPO_MODEL_KEY", "").strip()
        if not generation_id or not model_key:
            raise RuntimeError(
                "PostgreSQL HPO requires EUROMONITOR_HPO_GENERATION_ID and "
                "EUROMONITOR_HPO_MODEL_KEY"
            )
        study_name = generation_study_name(
            generation_id=generation_id, model_key=model_key
        )
        control_plane = create_storage(storage_from_environment())
        print(f"[hpo-control] PostgreSQL study={study_name}", flush=True)
        return study_name, control_plane

    @staticmethod
    def _restore_study_db(args, study_db: Path) -> None:
        """Resume precondition: a downloaded sqlite study, or restore from DVC."""
        if not args.resume:
            return
        if checkpoint_publication_deferred():
            if not study_db.is_file():
                raise FileNotFoundError(f"resume requires downloaded local Optuna study: {study_db}")
            print(f"[resume] using local Optuna study: {study_db.name}", flush=True)
        else:
            from training.dvc_store import restore_checkpoint

            restore_checkpoint(RESULTS, study_db)
            print(f"[resume] restored Optuna study from DVC: {study_db.name}", flush=True)

    @staticmethod
    def _resolve_study(args):
        """Create/load the TPE study (SSOT space, sqlite or control plane)."""
        import optuna

        with trace_step('training.run_hpo.study_creation'):
            sampler = optuna.samplers.TPESampler(seed=SEED)
            control_study_name, control_plane = _HpoStream._control_plane_storage()
            # sqlite storage: the sweep SURVIVES session loss — re-running with the same
            # --study resumes; every trial's params/value persist (the essential record)
            study_name = control_study_name or f"second08-{args.model.split('/')[-1]}-dlr"
            # dlr suffix: discriminative-LR trials form a NEW objective surface —
            # never mixed into the pre-dlr TPE history (its surrogate would be poisoned
            # by trials whose values came from single-LR training)
            study_db = RESULTS / f"second08-{args.model.split('/')[-1]}-dlr.optuna.db"
            _HpoStream._restore_study_db(args, study_db)
            storage = control_plane or f"sqlite:///{study_db}"
            study = optuna.create_study(
                direction="maximize",
                sampler=sampler,
                study_name=study_name,
                storage=storage,
                load_if_exists=True,
            )
            if control_plane is not None:
                from training.hpo_control_plane import fail_stale_trials

                fail_stale_trials(study)
        return study, control_plane, study_db

    @staticmethod
    def _optimize(study, objective, *, n_trials: int, n_jobs: int, wandb_ctx, study_db: Path) -> None:
        """Resume-safe optimize: count prior trials, run only what remains."""
        with trace_step('training.run_hpo.optimize'):
            prior = len(
                [t for t in study.trials if t.state.name in ("COMPLETE", "PRUNED", "FAIL")]
            )
            remaining = max(0, n_trials - prior)
            print(
                f"HPO: {prior} prior trials on record, running {remaining} more (n_jobs={n_jobs})",
                flush=True,
            )
            if remaining:
                def _persist_study(*_args) -> None:
                    _HpoStream._persist_study(study_db)

                study.optimize(
                    objective,
                    n_trials=remaining,
                    n_jobs=n_jobs,
                    callbacks=[_optuna_tracking_cb(wandb_ctx), _persist_study],
                )

    @staticmethod
    def _persist_study(study_db: Path) -> None:
        """Push the study db after each committed trial (opt-in environment)."""
        if checkpoint_publication_deferred():
            return
        if not os.environ.get("DVC_API_KEY"):
            return
        from training.dvc_store import publish_checkpoint

        publish_checkpoint(RESULTS, study_db)

    @staticmethod
    def _write_decision_trail(study, args) -> Path:
        """Every trial's params + value on disk for an auditable decision trail."""
        trials_df = study.trials_dataframe(
            attrs=("number", "state", "value", "params", "user_attrs")
        )
        model_tag = args.model.split("/")[-1]
        era = "-dlr"  # discriminative-LR sweep era (see study_name above)
        trials_path = artifact("hpo_trials", {"model": model_tag, "era": era})
        ensure_parent(trials_path)
        trials_df.to_csv(trials_path, index=False)
        trace_artifact("hpo_trials", trials_path, producer="training.training")
        return trials_path

    @staticmethod
    def _write_best_record(study, args, *, selection_mode: bool) -> tuple[dict, Path]:
        """The train_<model>_hpo_best.json payload: config + ranked signal."""
        model_tag = args.model.split("/")[-1]
        era = "-dlr"
        best = {
            "config": study.best_params,
            "value": study.best_value,
            "n_trials": len(study.trials),
            "model": args.model,
            "objective": f"discriminative-LR ({_runtime('layer_decay')}^k per-layer groups)",
            # which signal ranked the trials (test-leak fix, 2026-09-12):
            # calibration Rand in both holdout and CV selection modes
            "selection": (
                HPO_OBJECTIVE_HOLDOUT if selection_mode else HPO_OBJECTIVE_CV
            ),
        }
        out_path = artifact("hpo_best", {"model": model_tag, "era": era})
        ensure_parent(out_path)
        with open(out_path, "w") as f:
            json.dump(best, f, indent=2)
        trace_artifact("hpo_best", out_path, producer="training.training")
        return best, out_path

    @staticmethod
    def _publish_best(study, args, *, selection_mode: bool, wandb_ctx) -> None:
        """Persist the sweep's decision trail and its best config + metrics."""
        with trace_step('training.run_hpo.best_config'):
            if not [
                trial
                for trial in study.trials
                if trial.state.name == "COMPLETE" and trial.value is not None
            ]:
                raise FoldExecutionError(
                    "Optuna selection",
                    [{"status": "no_completed_trials"}],
                )

            trials_path = _HpoStream._write_decision_trail(study, args)
            best, out_path = _HpoStream._write_best_record(
                study, args, selection_mode=selection_mode
            )
            if (
                wandb_ctx is not None
                and os.environ.get("EUROMONITOR_REMOTE_TRAINING") != "1"
            ):
                wandb_ctx.log_artifact(trials_path, "hpo-trials")
                wandb_ctx.log_artifact(out_path, "hpo-best")
                wandb_ctx.set_summary(
                    {"hpo_best_objective": study.best_value, "hpo_completed_trials": len(study.trials)}
                )
            _sel = best["selection"]
            print(f"\nBEST: {study.best_params} -> {_sel} {study.best_value:.4f}", flush=True)
            # AUDIT FIX (round 2, F02): print the path ACTUALLY written — the old
            # line named train_hpo_best.json, a file never written by this lane.
            print(f"wrote {out_path}", flush=True)


class OptunaObjectiveOwner:
    """One TPE trial, end to end, owned: configuration sampling from
    HPO_SPACE (hpo.tpe_space), protocol-row training, trajectory evidence,
    calibrated-Rand ranking + guardrails, and champion retention.

    run_hpo holds this owner so the objective contract (optuna.Trial -> float
    with TrialPruned semantics) stays byte-identical for the sweep consumers.
    """

    def __init__(
        self,
        args,
        data,
        *,
        cv_folds,
        folds_override,
        dev_fraction,
        dev_override,
        selection_mode: bool,
        neg_pairs,
        train_neg_pairs,
        neg_pair_sources,
        train_neg_pair_sources,
        dynamic_mask_hard_negatives: bool,
        dynamic_mask_prob,
        dynamic_mask_lo,
        dynamic_mask_hi,
        mask_audit,
        hard_negative_mask_audit,
        wandb_ctx,
    ) -> None:
        # Boundary plumbing only: kwargs -> instance state; no logic of its
        # own (the trial contract is owned by the methods), so no extraction
        # is possible without hiding the fields behind a second indirection.
        self.args = args
        self.data = data
        self.cv_folds = cv_folds
        self.folds_override = folds_override
        self.dev_fraction = dev_fraction
        self.dev_override = dev_override
        self.selection_mode = selection_mode
        self.neg_pairs = neg_pairs
        self.train_neg_pairs = train_neg_pairs
        self.neg_pair_sources = neg_pair_sources
        self.train_neg_pair_sources = train_neg_pair_sources
        self.dynamic_mask_hard_negatives = dynamic_mask_hard_negatives
        self.dynamic_mask_prob = dynamic_mask_prob
        self.dynamic_mask_lo = dynamic_mask_lo
        self.dynamic_mask_hi = dynamic_mask_hi
        self.mask_audit = mask_audit
        self.hard_negative_mask_audit = hard_negative_mask_audit
        self.wandb_ctx = wandb_ctx
        # Set once the study stream resolved its control plane (PostgreSQL
        # mode); None means local champion retention stays enabled.
        self.control_plane = None

    def _suggest_configuration(self, trial) -> dict:
        """Sample one config from HPO_SPACE, SSOT-fixed knobs included."""
        cfg = self._fixed_ssot_knobs()
        cfg.update(self._suggest_searchable(trial))
        return cfg

    @staticmethod
    def _fixed_ssot_knobs() -> dict:
        """Knobs the trial cannot steer: runtime/SSOT values, never literals."""
        return {
            "architecture": _runtime("architecture"),
            "projection_dropout": _runtime("projection_dropout"),
            "label_smoothing": _runtime("label_smoothing"),
            "random_easy_enabled": bool(
                _runtime("random_easy_negatives")["enabled"]
            ),
            "random_easy_ratio_to_hard": float(
                _runtime("random_easy_negatives")["ratio_to_hard"]
            ),
            "random_easy_candidate_pool_size": int(
                _runtime("random_easy_negatives")["candidate_pool_size"]
            ),
            "lr_scheduler": _runtime("lr_scheduler"),  # SSOT
            "max_grad_norm": _runtime("max_grad_norm"),  # SSOT
            "patience": ES_PATIENCE,
            "es_threshold": ES_THRESHOLD,
            "late_epoch_decay_enabled": bool(
                _runtime("late_epoch_lr_decay")["enabled"]
            ),
            "late_epoch_decay_start_fraction": float(
                _runtime("late_epoch_lr_decay")["start_epoch_fraction"]
            ),
            "late_epoch_decay_multiplier": float(
                _runtime("late_epoch_lr_decay")["multiplier"]
            ),
        }

    @staticmethod
    def _suggest_searchable(trial) -> dict:
        """The six TPE suggest calls, in the original (pinned) order.

        Invariant: suggest-call ORDER defines the TPE search layout — keep it
        identical even though the fixed-knob keys were interleaved in the old
        dict literal (they run no suggest calls).
        """
        suggested = {
            "epochs": trial.suggest_int("epochs", *HPO_SPACE["epochs"]),
            "lr": trial.suggest_float("lr", *HPO_SPACE["lr"], log=True),
            "warmup_ratio": trial.suggest_float(
                "warmup_ratio", *HPO_SPACE["warmup_ratio"]
            ),
            "weight_decay": trial.suggest_float(
                "weight_decay", *HPO_SPACE["weight_decay"]
            ),
            "negative_mask_frac": trial.suggest_float(
                "negative_mask_frac", *HPO_SPACE["negative_mask_frac"]
            ),
            "uniformity_weight": trial.suggest_float(
                "uniformity_weight", *HPO_SPACE["uniformity_weight"]
            ),
        }
        # TASK B item 5: appended LAST so the existing TPE suggest layout is
        # byte-identical when hpo.tpe_space.lr_scheduler is null.
        if HPO_SCHEDULERS:
            suggested["lr_scheduler"] = trial.suggest_categorical(
                "lr_scheduler", list(HPO_SCHEDULERS)
            )
        return suggested

    def _run_trial(self, trial, cfg: dict) -> list[dict]:
        """Train one protocol configuration over the selection folds."""
        import torch

        return train_one_config(
            cfg,
            loss=self.args.loss,
            model_id=self.args.model,
            use_hp=_SSOT_HP,
            band=_band_tuple(self.args.band),
            data=self.data,
            seed=SEED,
            on_cuda=torch.cuda.is_available(),
            cv_folds=self.cv_folds,
            run_tag=f"{self.args.model.split('/')[-1]}_t{trial.number}",
            folds_override=self.folds_override,
            dev_fraction=self.dev_fraction,
            dev_override=self.dev_override,
            selection_mode=self.selection_mode,
            neg_pairs=self.neg_pairs,
            train_neg_pairs=self.train_neg_pairs,
            neg_pair_sources=self.neg_pair_sources,
            train_neg_pair_sources=self.train_neg_pair_sources,
            dynamic_mask_hard_negatives=self.dynamic_mask_hard_negatives,
            dynamic_mask_frac=cfg["negative_mask_frac"],
            dynamic_mask_prob=self.dynamic_mask_prob,
            dynamic_mask_lo=self.dynamic_mask_lo,
            dynamic_mask_hi=self.dynamic_mask_hi,
            mask_audit=self.mask_audit,
            hard_negative_mask_audit=self.hard_negative_mask_audit,
            wandb_ctx=self.wandb_ctx,
        )

    def _record_loss_trajectory(self, trial, ok_rows: list[dict]) -> None:
        """Persist the actual trial evidence in Optuna as user attrs."""
        _trial_loss = [r.get("final_train_loss") for r in ok_rows if np.isfinite(r.get("final_train_loss", float("nan")))]
        if _trial_loss:
            trial.set_user_attr("mean_final_train_loss", float(np.mean(_trial_loss)))
        self._set_trajectory_attrs(trial, self._loss_histories(ok_rows))

    @staticmethod
    def _loss_histories(ok_rows: list[dict]) -> tuple[list, list]:
        """Parse per-fold dev/train loss histories (skips unparsable rows)."""
        _dev_loss_histories = []
        _train_loss_histories = []
        for row in ok_rows:
            try:
                _dev_loss_histories.append(json.loads(row.get("dev_loss_hist", "[]")))
                _train_loss_histories.append(json.loads(row.get("train_loss_hist", "[]")))
            except (TypeError, json.JSONDecodeError):
                continue
        return _dev_loss_histories, _train_loss_histories

    def _set_trajectory_attrs(self, trial, histories: tuple[list, list]) -> None:
        """Best/final dev loss + overfit signature summaries, when present."""
        _dev_loss_histories, _train_loss_histories = histories
        _best_dev_losses = [min(v) for v in _dev_loss_histories if v]
        _final_dev_losses = [v[-1] for v in _dev_loss_histories if v]
        _overfit_flags = [
            int(bool(t) and bool(d) and t[-1] < t[0] and d[-1] > min(d))
            for t, d in zip(_train_loss_histories, _dev_loss_histories, strict=True)
        ]
        if _best_dev_losses:
            trial.set_user_attr("mean_best_dev_loss", float(np.mean(_best_dev_losses)))
        if _final_dev_losses:
            trial.set_user_attr("mean_final_dev_loss", float(np.mean(_final_dev_losses)))
        if _overfit_flags:
            trial.set_user_attr("overfit_signature_rate", float(np.mean(_overfit_flags)))

    def _rank_on_calibration_proxy(self, ok_rows: list[dict], guardrail: dict):
        """Rank trials on the calibrated direct-assignment Rand proxy.

        Returns (value, proxy_rows, mean_rand, mean_penalty). Trials with no
        finite proxy row are pruned; the collapse guardrail prunes on either
        configured breach before the objective value becomes meaningful.
        """
        import optuna

        proxy_rows = [
            r for r in ok_rows
            if np.isfinite(r.get("calibration_rand_index", float("nan")))
        ]
        if not proxy_rows:
            raise optuna.TrialPruned(
                "no fold produced a finite calibrated Rand Index proxy"
            )
        self._reject_on_collapse_guardrail(proxy_rows, guardrail)
        mean_rand = float(np.mean([r["calibration_rand_index"] for r in proxy_rows]))
        mean_penalty = float(np.mean([r["collapse_penalty"] for r in proxy_rows]))
        value = mean_rand - mean_penalty
        return value, proxy_rows, mean_rand, mean_penalty

    @staticmethod
    def _reject_on_collapse_guardrail(proxy_rows: list[dict], guardrail: dict) -> None:
        """Prune a trial whose median cosine or crossing rate breaches its
        configured ceiling (the objective value must stay meaningful)."""
        import optuna

        collapse_medians = [
            float(r["collapse_median_cosine"])
            for r in proxy_rows
            if np.isfinite(r.get("collapse_median_cosine", float("nan")))
        ]
        collapse_crossing_rates = [
            float(r["collapse_crossing_rate"])
            for r in proxy_rows
            if np.isfinite(r.get("collapse_crossing_rate", float("nan")))
        ]
        reject_median = bool(
            collapse_medians and max(collapse_medians) > float(guardrail["reject_median"])
        )
        reject_crossing = bool(
            collapse_crossing_rates
            and max(collapse_crossing_rates) > float(guardrail["crossing_rate_ceiling"])
        )
        if reject_median or reject_crossing:
            # ── CONSOLIDATED TRACE: the trial-pruning decision, with the
            # aggregate breach that caused it. The PER-SAMPLE evidence for the
            # same trial is in the `collapse.pairs` entity rows its folds emit.
            training_trace().add(
                "hpo",
                "guardrail_reject",
                in_count=len(proxy_rows),
                out_count=0,
                reason=(
                    "trial pruned: collapse guardrail breached before the "
                    "objective value became meaningful"
                ),
                detail={
                    "median_max": max(collapse_medians) if collapse_medians else None,
                    "median_ceiling": float(guardrail["reject_median"]),
                    "median_breach": int(reject_median),
                    "crossing_rate_max": (
                        max(collapse_crossing_rates) if collapse_crossing_rates else None
                    ),
                    "crossing_rate_ceiling": float(guardrail["crossing_rate_ceiling"]),
                    "crossing_rate_breach": int(reject_crossing),
                    "folds": [int(r["fold"]) for r in proxy_rows if "fold" in r],
                    "per_sample_evidence": "collapse.pairs entity rows for this trial's folds",
                },
                source="training.training _HpoStream collapse guardrail",
            )
        if reject_median:
            raise optuna.TrialPruned(
                "collapse guardrail rejected trial: "
                f"median_cosine={max(collapse_medians):.4f}"
            )
        if reject_crossing:
            raise optuna.TrialPruned(
                "collapse guardrail rejected trial: "
                f"crossing_rate={max(collapse_crossing_rates):.4f} "
                f"ceiling={float(guardrail['crossing_rate_ceiling']):.4f}"
            )

    def _record_proxy_summary(self, trial, proxy_rows, value: float, mean_rand: float, mean_penalty: float, guardrail: dict) -> dict:
        """Mirror every proxy aggregate onto the trial as user attrs."""
        proxy_summary = {
            **self._rand_summary(proxy_rows, mean_rand, mean_penalty),
            **self._collapse_summary(proxy_rows, guardrail),
            **self._diagnostic_summary(proxy_rows),
        }
        trial.set_user_attr("rand_index_objective", value)
        for key, metric in proxy_summary.items():
            trial.set_user_attr(key, metric)
        return proxy_summary

    @staticmethod
    def _rand_summary(proxy_rows: list[dict], mean_rand: float, mean_penalty: float) -> dict:
        """Calibrated Rand means over the proxy rows (+ penalty)."""
        return {
            "mean_calibration_rand_index": mean_rand,
            "mean_calibration_adjusted_rand": float(
                np.mean([r["calibration_adjusted_rand"] for r in proxy_rows])
            ),
            "mean_calibration_precision_at_threshold": float(
                np.mean([r["calibration_precision_at_threshold"] for r in proxy_rows])
            ),
            "mean_calibration_recall_at_threshold": float(
                np.mean([r["calibration_recall_at_threshold"] for r in proxy_rows])
            ),
            "mean_calibration_over_merge_rate": float(
                np.mean([r["calibration_over_merge_rate"] for r in proxy_rows])
            ),
            "mean_calibration_under_merge_rate": float(
                np.mean([r["calibration_under_merge_rate"] for r in proxy_rows])
            ),
            "mean_calibration_threshold_stable": float(
                np.mean([r["calibration_threshold_stable"] for r in proxy_rows])
            ),
            "mean_collapse_penalty": mean_penalty,
        }

    @staticmethod
    def _collapse_summary(proxy_rows: list[dict], guardrail: dict) -> dict:
        """Collapse-distribution means + the crossing-rate ceiling used."""
        return {
            "mean_collapse_median_cosine": float(
                np.mean([r["collapse_median_cosine"] for r in proxy_rows])
            ),
            "mean_collapse_p90_cosine": float(
                np.mean([r["collapse_p90_cosine"] for r in proxy_rows])
            ),
            "mean_collapse_cosine_std": float(
                np.mean([r["collapse_cosine_std"] for r in proxy_rows])
            ),
            "mean_collapse_crossing_rate": float(
                np.mean([r["collapse_crossing_rate"] for r in proxy_rows])
            ),
            "collapse_crossing_rate_ceiling": float(
                guardrail["crossing_rate_ceiling"]
            ),
        }

    @staticmethod
    def _diagnostic_summary(proxy_rows: list[dict]) -> dict:
        """Plain diagnostic means (bridge edges, attribute-conflict rate)."""
        return {
            "mean_diagnostic_bridge_edge_count": float(
                np.mean([r["diagnostic_bridge_edge_count"] for r in proxy_rows])
            ),
            "mean_attribute_conflict_error_rate": float(
                np.nanmean([r["attribute_conflict_error_rate"] for r in proxy_rows])
            ),
        }

    def _retain_champion(self, trial, value: float, proxy_rows: list[dict]) -> None:
        """Local champion retention unless the control plane owns promotion."""
        if self.control_plane is None:
            retain_hpo_champion(
                model_id=self.args.model,
                run_tag=f"{self.args.model.split('/')[-1]}_t{trial.number}",
                value=value,
                folds=[int(r["fold"]) for r in proxy_rows],
            )

    def objective(self, trial) -> float:
        """The pinned Optuna objective: one TPE trial, complete."""
        cfg = self._suggest_configuration(trial)
        rows = self._run_trial(trial, cfg)
        require_no_failed_folds(rows, lane=f"HPO trial {trial.number}")
        ok_rows = rows
        self._record_loss_trajectory(trial, ok_rows)
        guardrail = _timed_load_config("hpo.guardrail")["collapse_guardrail"]
        value, proxy_rows, mean_rand, mean_penalty = (
            self._rank_on_calibration_proxy(ok_rows, guardrail)
        )
        self._record_proxy_summary(
            trial, proxy_rows, value, mean_rand, mean_penalty, guardrail
        )
        # PostgreSQL mode promotes only from the controller after Optuna
        # commits COMPLETE and a sealed artifact snapshot is READY.
        self._retain_champion(trial, value, proxy_rows)
        return value

@timed
def run_hpo(
    args,
    data,
    cv_folds: int | None = None,
    folds_override: list[set[str]] | set[str] | None = None,
    dev_fraction: float | None = None,
    dev_override: set[str] | None = None,
    selection_mode: bool = False,
    # gate hard-no pairs — the contrastive SSOT loss needs labeled
    # negatives; the sweep lanes pass them exactly like the main lane
    # (train_one_config filters them to the fold's train side itself)
    neg_pairs: np.ndarray | None = None,
    train_neg_pairs: np.ndarray | None = None,
    neg_pair_sources: np.ndarray | None = None,
    train_neg_pair_sources: np.ndarray | None = None,
    dynamic_mask_hard_negatives: bool = False,
    dynamic_mask_frac: float = 0.0,
    dynamic_mask_prob: float | None = None,
    dynamic_mask_lo: float | None = None,
    dynamic_mask_hi: float | None = None,
    mask_audit: list[dict] | None = None,
    hard_negative_mask_audit: list[dict] | None = None,
    wandb_ctx=None,
) -> None:
    """Thin orchestrator over `_HpoStream` + `OptunaObjectiveOwner`: the TPE
    HPO mode (--hpo / --n-trials surface, protocol rows, best-config
    publication). Behavior is byte-identical to the pre-owner monolith."""
    import optuna

    stream = _HpoStream()
    owner = OptunaObjectiveOwner(
        args,
        data,
        cv_folds=cv_folds,
        folds_override=folds_override,
        dev_fraction=dev_fraction,
        dev_override=dev_override,
        selection_mode=selection_mode,
        neg_pairs=neg_pairs,
        train_neg_pairs=train_neg_pairs,
        neg_pair_sources=neg_pair_sources,
        train_neg_pair_sources=train_neg_pair_sources,
        dynamic_mask_hard_negatives=dynamic_mask_hard_negatives,
        dynamic_mask_prob=dynamic_mask_prob,
        dynamic_mask_lo=dynamic_mask_lo,
        dynamic_mask_hi=dynamic_mask_hi,
        mask_audit=mask_audit,
        hard_negative_mask_audit=hard_negative_mask_audit,
        wandb_ctx=wandb_ctx,
    )

    def objective(trial: optuna.Trial) -> float:
        return owner.objective(trial)

    study, control_plane, study_db = stream._resolve_study(args)
    owner.control_plane = control_plane
    stream._optimize(
        study,
        objective,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        wandb_ctx=wandb_ctx,
        study_db=study_db,
    )
    stream._publish_best(study, args, selection_mode=selection_mode, wandb_ctx=wandb_ctx)


def _optuna_tracking_cb(wandb_ctx):
    """Record every completed Optuna trial in local and optional remote logs."""
    def cb(study, trial):
        if trial.state.name == "COMPLETE" and trial.value is not None:
            if wandb_ctx is not None:
                wandb_ctx.log_metrics(
                    {
                        "hpo/trial": float(trial.number),
                        "hpo_objective": trial.value,
                        **{f"hpo_{k}": v for k, v in trial.params.items()},
                        **{f"hpo_{k}": v for k, v in trial.user_attrs.items()},
                    },
                )
    return cb


def _band_tuple(band: str) -> tuple[float, float]:
    lo, hi = (float(x) for x in band.split("-"))
    return lo, hi
