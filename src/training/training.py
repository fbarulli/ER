"""src/training/training.py — solid GPU-ready fine-tune pipeline for the
second-series lane.

Rewrite of the training path with everything the 07-series left out:

  TRACEBACKS      every failure prints the full chain (traceback.format_exc()
                  into the run log AND the CSV — no silent folds, no
                  "AUC=NaN, moving on").
  MLFLOW          one parent run per invocation; every fold/arm is a nested
                  run with params + metrics + the fold CSV artifact; the best
                  config is registered as a tagged child run.
  OPTUNA          HPO mode (--hpo): TPE over epochs/lr/warmup/band, each trial
                  a nested MLflow run, best config reported + persisted.
  EARLY STOPPING  HF EarlyStoppingCallback on the dev AUC (patience
                  configurable); load_best_model_at_end so the reported
                  metric is the best checkpoint, not the last.
  FULL SETTINGS    warmup_ratio, weight decay, lr scheduler, grad clipping,
                  bf16 (on CUDA), checkpoints + save_total_limit, seeded
                  everything, save_best_model, per-fold dev/test split.

Group-aware splits throughout: barcode-level (no product straddles a fold),
and within each fold the train barcodes are split again into train/dev for
early stopping (dev NEVER touches test).

Device: CPU here, CUDA on the Colab VM unchanged — the pipeline reads
torch.cuda.is_available() and flips bf16/batch-size guidance, nothing else.

Usage:
  python src/training/train.py --loss contrastive     (entry; src/training/ is a package)
  python src/training/train.py --hpo --n-trials 20    # optuna TPE sweep
  python src/training/train.py --loss triplet --band 0.45-0.80
  MLFLOW_TRACKING_URI=... uv run ...      # local sqlite default; =off disables
                                            # (SSOT: src/core/mlflow_ctx.py)

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
    kfold_barcodes,
    load_config,
    load_local_sentence_transformer,
    metadata_text,
    pair_auc,
    pair_similarity,
    plot_dpi,
    recall_column_suffix,
    row_metadata_text,
    runtime,
    trace_artifact,
)
from core.schemas import check_labeled_pairs_frame
from core.common import SSOT_CONTRASTIVE_MARGIN as _SSOT_MARGIN
from core.common import runtime as _runtime

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
CV_FOLDS = int(load_config()["split"]["cv_folds"])
DEV_FRACTION = runtime("dev_fraction")
BATCH_SIZE_CPU = runtime("batch_size_cpu")
BATCH_SIZE_CUDA = runtime("batch_size_cuda")
MAX_TRIPLES = runtime("max_triples")
EVAL_STEPS_PER_EPOCH = runtime("eval_steps_per_epoch")
ES_PATIENCE = runtime("es_patience")
ES_THRESHOLD = runtime("es_threshold")
_ANN_MINING_CFG = load_config()["mining"]["ann"]
N_TARGET_MINING = int(_ANN_MINING_CFG["target"])
ANN_MINING_ENABLED = bool(_ANN_MINING_CFG["enabled"])
MASK_TRACK_PER_EPOCH = bool(load_config()["masking"]["track_per_epoch"])
TRACK_DATAPOINT_USAGE = bool(load_config()["training"]["track_datapoint_usage"])
_UNIFORMITY_CFG = load_config()["training"]["uniformity_regularization"]
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
    "attribute_conflict": {
        "emitter": "training.train: np.full(len(_attr_neg), 'attribute_conflict')",
        "role": "negative_source",
        "dynamic": True,
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


class UnregisteredDatapointPopulationError(RuntimeError):
    """A producer emitted a datapoint population the registry does not know.

    Raised at the point of use by the per-fold coverage audit, AFTER the
    coverage artifact for that fold has been written, so an undeclared
    population is both loud and still inspectable. Silence was the defect
    (audit A4-2): a tag outside the registry used to be dropped from the
    coverage rows with no warning at all.
    """

# DEFAULT_CFG REMOVED (audit 2026-09-09): zero readers since the entry
# (train.py) constructs its own cfg dict; a stale epochs=2 default here
# contradicted the SSOT epochs=10 and was pure dead-code risk.

# TPE search space, SSOT: config/training.yaml hpo.tpe_space (validated by
# HpoSpaceSpec at load — lo < hi per knob). The dict literal was a second
# declaration the config could not steer (audit 2026-09-09, owner Q27).
from core.common import hpo_cfg as _hpo_cfg_load

HPO_SPACE = {k: (lo, hi) for k, (lo, hi) in _hpo_cfg_load()["tpe_space"].items()}

# HPO objective protocol, SSOT: hpo.objective /
# hpo.selection_skip_test_eval (validated by HpoSpec/ObjectiveSpec at load).
# Both modes now rank trials on the calibrated direct-assignment Rand proxy;
# holdout selection still skips the test quarter entirely.
_HPO_OBJ_TABLE = _hpo_cfg_load()["objective"]
HPO_OBJECTIVE_HOLDOUT = _HPO_OBJ_TABLE["holdout"]
HPO_OBJECTIVE_CV = _HPO_OBJ_TABLE["cv"]
HPO_SKIP_TEST_EVAL = bool(_hpo_cfg_load()["selection_skip_test_eval"])


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


def require_no_failed_folds(rows: list[dict], *, lane: str) -> None:
    """Make incomplete calibration evidence fatal before selection aggregation."""
    incomplete_rows = [
        row
        for row in rows
        if row.get("status") != "ok"
        or not np.isfinite(row.get("calibration_rand_index", float("nan")))
    ]
    if not rows or incomplete_rows:
        raise FoldExecutionError(lane, incomplete_rows or [{"status": "missing"}])


# ═══════════════════════════════════════════════════════════════════════════
# MLflow — SSOT src/core/mlflow_ctx (audit 2026-09-09: this module used to carry
# its own MlflowCtx duplicate with CONFLICTING semantics — "off unless
# MLFLOW_TRACKING_URI is set" — while the owner mandate (training logs
# available locally) is lib's: local sqlite by default, =off to disable.
# The duplicate shadowed the mandate; re-exported here for hpo.py's import.)
# ═══════════════════════════════════════════════════════════════════════════

from core.mlflow_ctx import MlflowCtx
from core.ranking_metrics import (
    added_encode_rows,
    build_evaluation_pool,
    competitors_per_query,
    ranking_at_k_by_query,
    ranking_coverage,
)

# ═══════════════════════════════════════════════════════════════════════════
# Training (ST 6 modern Trainer path with HF early stopping)
# ═══════════════════════════════════════════════════════════════════════════


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
    log_history = getattr(trainer_state, "log_history", []) or []
    losses = [entry["eval_loss"] for entry in log_history if "eval_loss" in entry]
    tokenizer = getattr(model, "tokenizer", None)
    auto_model = model[0].auto_model
    token_names = ("pad_token_id", "bos_token_id", "eos_token_id")
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
    manifest = {
        "format": "euromonitor-hf-resume-v1",
        "epoch": epoch,
        "global_step": global_step,
        "best_loss": float(min(losses)) if losses else None,
        # These are the exact components of the requested checkpoint dict.
        # They remain in their native HF files so model/optimizer tensors are
        # not serialized a second time into a multi-GB sidecar.
        "files": {
            "model_state_dict": model_files,
            "optimizer_state_dict": "optimizer.pt" if optimizer is not None else None,
            "scheduler_state_dict": "scheduler.pt" if scheduler is not None else None,
            "scaler_state_dict": "scaler.pt" if scaler is not None else None,
            "rng_state": "rng_state.pth",
            "trainer_state": "trainer_state.json",
            "training_args": "training_args.bin",
        },
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
        "native_hf_resume": {
            "trainer_state": "trainer_state.json",
            "trainer_control": "trainer_state.json:control",
            "training_args": "training_args.bin",
            "optimizer": "optimizer.pt",
            "scheduler": "scheduler.pt",
            "rng": "rng_state.pth",
        },
    }
    (checkpoint / "checkpoint_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


# _auc/_cos -> _common SSOT (see GATES_MAP.md)
_auc = pair_auc


_cos = pair_similarity


def _split_safe_random_negative_pairs(
    df: pd.DataFrame,
    row_bc: np.ndarray,
    split_barcodes: set[str],
    *,
    seed: int,
    n_neg: int,
) -> np.ndarray:
    """Build known-different random negatives using only one split.

    ``build_pairs`` owns the barcode validity/title-difference rules. This
    wrapper restricts its input to the requested split first, then maps the
    returned local row indices back to the training payload indices.
    """
    from core.blocking import build_pairs
    from core.common import training_cfg

    split_rows = np.flatnonzero(
        np.isin(row_bc[: len(df)], np.asarray(sorted(split_barcodes), dtype=str))
    )
    if len(split_rows) < 2 or n_neg <= 0:
        if n_neg > 0:
            print(
                "[random-easy] WARNING: split-safe negative sampling skipped "
                f"(requested={n_neg}, split_rows={len(split_rows)})",
                flush=True,
            )
        return np.empty((0, 2), dtype=int)

    subset = df.iloc[split_rows].reset_index(drop=True)
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
            # Keep a one-pair request alive for the final feasibility check;
            # target //= 2 used to turn 1 into 0 and silently discard the
            # random/easy population after one sampling miss.
            if target == 1:
                break
            target = max(1, target // 2)
    print(
        "[random-easy] WARNING: no split-safe negatives could be sampled "
        f"(requested={n_neg}, split_rows={len(split_rows)})",
        flush=True,
    )
    return np.empty((0, 2), dtype=int)


def _mix_random_easy_training_negatives(
    hard_pairs: np.ndarray,
    hard_sources: np.ndarray,
    *,
    df: pd.DataFrame,
    row_bc: np.ndarray,
    train_barcodes: set[str],
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
    hard_pairs = np.asarray(hard_pairs, dtype=int).reshape(-1, 2)
    hard_sources = np.asarray(hard_sources, dtype=object)
    if len(hard_sources) != len(hard_pairs):
        raise ValueError("hard negative/source lengths differ")
    ratio_to_hard = float(ratio_to_hard)
    if ratio_to_hard < 0:
        raise ValueError("random/easy to hard ratio must be non-negative")
    if int(candidate_pool_size) < 1:
        raise ValueError("random/easy candidate pool size must be positive")
    target = int(np.ceil(len(hard_pairs) * ratio_to_hard))
    if not enabled or target == 0:
        return hard_pairs, hard_sources, 0

    candidates = _split_safe_random_negative_pairs(
        df,
        row_bc,
        train_barcodes,
        seed=seed,
        n_neg=min(target, int(candidate_pool_size)),
    )
    if not len(candidates):
        return hard_pairs, hard_sources, 0
    if not pairs_in_set(candidates, row_bc, train_barcodes).all():
        raise RuntimeError("random/easy training negatives crossed the train split")

    normalized_hard = {tuple(pair) for pair in np.sort(hard_pairs, axis=1)}
    unique_candidates = np.asarray(
        [
            pair
            for pair in np.unique(np.sort(candidates, axis=1), axis=0)
            if tuple(pair) not in normalized_hard
        ],
        dtype=int,
    ).reshape(-1, 2)
    if not len(unique_candidates):
        print(
            "[random-easy] WARNING: candidate pool only duplicated hard negatives",
            flush=True,
        )
        return hard_pairs, hard_sources, 0
    rng = np.random.default_rng(seed + 1)
    chosen = unique_candidates[
        rng.choice(
            len(unique_candidates),
            size=target,
            replace=len(unique_candidates) < target,
        )
    ]
    mixed_pairs = np.vstack([hard_pairs, chosen])
    mixed_sources = np.concatenate(
        [hard_sources, np.full(target, "random_easy", dtype=object)]
    )
    return mixed_pairs, mixed_sources, len(unique_candidates)


def _precision_at_recall(y: np.ndarray, scores: np.ndarray, recall_target: float):
    """Precision/recall/threshold at a target recall (07-series schema).

    Threshold = the LOWEST score still achieving recall_target (any higher
    cut drops below it); precision at that cut with the FP count implied.
    Deterministic: sorted order, ties resolved by score value.
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


def _youden_thr(scores: np.ndarray, labels: np.ndarray) -> float:
    """Youden-optimal threshold (J = TPR - FPR) over a labeled score set.

    HOLDOUT DISCIPLINE: call this on the DEV scores, then apply the returned
    threshold verbatim to the TEST scores — never on the scores it rates
    (that is the optimistic leak the owner audit removed; the fold rows keep
    youden_thr_test_descriptive as the leak diagnostic only).
    """
    order = np.argsort(-scores)
    tps = np.cumsum(labels[order])
    fps = np.cumsum(1 - labels[order])
    tpr = tps / max(int((labels == 1).sum()), 1)
    fpr = fps / max(int((labels == 0).sum()), 1)
    j = tpr - fpr
    k = int(np.argmax(j))
    return float(scores[order][k])


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
):
    """Loss factory (SSOT knobs: training.loss / training.contrastive_margin).

    mnrl  — MultipleNegativesRankingLoss: (anchor, positive[, negative])
            column dataset; in-batch negatives; ignores labels.
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
        )
    return losses.TripletLoss(model)


def _smoothed_contrastive_losses(
    positive_pairs,
    negative_pairs,
    *,
    margin: float,
    label_smoothing: float,
):
    """Return positive/negative OnlineContrastiveLoss terms with smoothing."""
    import torch.nn.functional as F

    smoothing = float(label_smoothing)
    if not 0.0 <= smoothing < 0.5:
        raise ValueError("contrastive label smoothing must be in [0, 0.5)")
    positive_hinge = F.relu(float(margin) - positive_pairs)
    negative_hinge = F.relu(float(margin) - negative_pairs)
    positive_loss = (
        (1.0 - smoothing) * positive_pairs.pow(2)
        + smoothing * positive_hinge.pow(2)
    ).sum()
    negative_loss = (
        (1.0 - smoothing) * negative_hinge.pow(2)
        + smoothing * negative_pairs.pow(2)
    ).sum()
    return positive_loss, negative_loss, negative_hinge


def _tracking_contrastive_loss(
    model,
    *,
    margin: float,
    structured_feature_weight: float,
    uniformity_weight: float,
    uniformity_temperature: float,
    uniformity_min_batch_size: int,
    label_smoothing: float,
):
    """Return OnlineContrastiveLoss with selection/backprop telemetry.

    The implementation preserves the installed loss's hard-pair selection
    and arithmetic. It only accumulates detached counters and loss-component
    values during gradient-enabled forwards; ProgressCallback drains them at
    Trainer logging steps.
    """
    import torch
    import torch.nn.functional as F
    from sentence_transformers.sentence_transformer import losses

    class _TrackedOnlineContrastiveLoss(losses.OnlineContrastiveLoss):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._tracking_totals: dict[str, float] = {}
            self._tracking_batches = 0
            self._batch_pair_ids = None
            self._batch_structured_features = None
            self._total_negative_pairs = 0
            self._seen_hard_negative_ids: set[int] = set()
            self._seen_margin_active_negative_ids: set[int] = set()
            self._negative_present_counts: dict[int, int] = {}
            self._negative_selected_counts: dict[int, int] = {}
            self._negative_backprop_counts: dict[int, int] = {}
            self._per_epoch_counts: dict[int, dict[int, dict[str, int]]] = {}
            self._pair_lineage: list[dict] = []
            self._current_epoch = 0

        def _uniformity_penalty(self, embeddings):
            if uniformity_weight <= 0:
                return embeddings[0].sum() * 0.0
            vectors = torch.cat(embeddings, dim=0)
            if len(vectors) < uniformity_min_batch_size:
                return vectors.sum() * 0.0
            vectors = F.normalize(vectors, p=2, dim=1)
            distances = torch.pdist(vectors, p=2).pow(2)
            if not len(distances):
                return vectors.sum() * 0.0
            return torch.logsumexp(
                -uniformity_temperature * distances,
                dim=0,
            ) - torch.log(
                torch.as_tensor(
                    len(distances),
                    dtype=distances.dtype,
                    device=distances.device,
                )
            )

        def set_batch_pair_ids(self, pair_ids) -> None:
            self._batch_pair_ids = pair_ids.detach().cpu()

        def set_batch_structured_features(self, features) -> None:
            self._batch_structured_features = features.detach().cpu()

        def set_total_negative_pairs(self, count: int) -> None:
            self._total_negative_pairs = int(count)

        def set_pair_lineage(self, pair_lineage: list[dict]) -> None:
            self._pair_lineage = pair_lineage

        def set_epoch(self, epoch: int) -> None:
            self._current_epoch = int(epoch)

        def compute_loss_from_embeddings(self, embeddings, labels):
            if not self._checked_labels:
                self._checked_labels = True
                if labels.ne(0).logical_and(labels.ne(1)).any().item():
                    import warnings

                    warnings.warn(
                        "OnlineContrastiveLoss expects binary labels (0 or 1). "
                        "Pairs with any other label are ignored, since they "
                        "match neither the positive nor the negative set.",
                        UserWarning,
                        stacklevel=4,
                    )

            structured = self._batch_structured_features
            self._batch_structured_features = None
            if structured is not None and structured_feature_weight > 0:
                from core.structured_features import fuse_torch

                pair_features = structured.to(device=embeddings[0].device)
                embeddings = [
                    fuse_torch(
                        embedding,
                        pair_features[:, side, :],
                        structured_feature_weight,
                    )
                    for side, embedding in enumerate(embeddings)
                ]
            distance_matrix = self.distance_metric(embeddings[0], embeddings[1])
            negs = distance_matrix[labels == 0]
            poss = distance_matrix[labels == 1]
            batch_pair_ids = self._batch_pair_ids
            self._batch_pair_ids = None

            # This is the installed sentence-transformers selection rule.
            negative_selection = negs < (
                poss.max() if len(poss) > 1 else negs.mean()
            )
            negative_pairs = negs[
                negative_selection
            ]
            positive_selection = poss > (
                negs.min() if len(negs) > 1 else poss.mean()
            )
            positive_pairs = poss[
                positive_selection
            ]
            # Binary label smoothing mixes a small amount of the opposite
            # class objective into each selected pair. At smoothing=0 this
            # is exactly the installed OnlineContrastiveLoss arithmetic.
            smoothing = float(label_smoothing)
            positive_loss, negative_loss, negative_hinge = (
                _smoothed_contrastive_losses(
                    positive_pairs,
                    negative_pairs,
                    margin=self.margin,
                    label_smoothing=smoothing,
                )
            )
            uniformity_loss = self._uniformity_penalty(embeddings)
            anti_collapse_loss = uniformity_weight * uniformity_loss
            loss_value = (
                positive_loss
                + negative_loss
                + anti_collapse_loss
            )

            # Evaluator forwards are no-grad; only optimizer-facing forwards
            # belong to the backprop attribution window.
            if torch.is_grad_enabled():
                if batch_pair_ids is not None:
                    labels_cpu = labels.detach().cpu()
                    negative_ids = batch_pair_ids[labels_cpu == 0]
                    selected_negative_ids = negative_ids[
                        negative_selection.detach().cpu()
                    ]
                    margin_active_negative_ids = selected_negative_ids[
                        (negative_hinge > 0).detach().cpu()
                    ]
                    backprop_negative_ids = (
                        selected_negative_ids
                        if smoothing > 0
                        else margin_active_negative_ids
                    )
                    for value in negative_ids.tolist():
                        key = int(value)
                        self._negative_present_counts[key] = (
                            self._negative_present_counts.get(key, 0) + 1
                        )
                        self._per_epoch_counts.setdefault(self._current_epoch, {}).setdefault(
                            key, {"present_count": 0, "hard_selected_count": 0, "backprop_count": 0}
                        )["present_count"] += 1
                    for value in selected_negative_ids.tolist():
                        key = int(value)
                        self._negative_selected_counts[key] = (
                            self._negative_selected_counts.get(key, 0) + 1
                        )
                        self._per_epoch_counts.setdefault(self._current_epoch, {}).setdefault(
                            key, {"present_count": 0, "hard_selected_count": 0, "backprop_count": 0}
                        )["hard_selected_count"] += 1
                    for value in backprop_negative_ids.tolist():
                        key = int(value)
                        self._negative_backprop_counts[key] = (
                            self._negative_backprop_counts.get(key, 0) + 1
                        )
                        self._per_epoch_counts.setdefault(self._current_epoch, {}).setdefault(
                            key, {"present_count": 0, "hard_selected_count": 0, "backprop_count": 0}
                        )["backprop_count"] += 1
                    self._seen_hard_negative_ids.update(
                        int(value) for value in selected_negative_ids.tolist()
                    )
                    self._seen_margin_active_negative_ids.update(
                        int(value) for value in margin_active_negative_ids.tolist()
                    )
                    source_sets = {
                        "present": negative_ids,
                        "selected": selected_negative_ids,
                        "backprop": backprop_negative_ids,
                    }
                    for event, ids in source_sets.items():
                        for pair_id in ids.tolist():
                            source = str(
                                self._pair_lineage[int(pair_id)].get(
                                    "population", "unknown"
                                )
                            )
                            metric = f"negative_source_{source}_{event}_count"
                            self._tracking_totals[metric] = (
                                self._tracking_totals.get(metric, 0.0) + 1.0
                            )
                values = {
                    "hard_positive_count": float(len(positive_pairs)),
                    "hard_negative_count": float(len(negative_pairs)),
                    "margin_active_negative_count": float(
                        (negative_hinge > 0).sum().item()
                    ),
                    "all_positive_count": float(len(poss)),
                    "all_negative_count": float(len(negs)),
                    "positive_loss": float(positive_loss.detach().item()),
                    "negative_loss": float(negative_loss.detach().item()),
                    "uniformity_loss": float(uniformity_loss.detach().item()),
                    "anti_collapse_loss": float(anti_collapse_loss.detach().item()),
                }
                for key, value in values.items():
                    self._tracking_totals[key] = (
                        self._tracking_totals.get(key, 0.0) + value
                    )
                self._tracking_batches += 1

            return loss_value

        def pop_tracking_stats(self) -> dict[str, float]:
            batches = self._tracking_batches
            totals = self._tracking_totals
            self._tracking_totals = {}
            self._tracking_batches = 0
            if not batches:
                return {}
            result = {
                "hard_positive_count": totals.get("hard_positive_count", 0.0),
                "hard_negative_count": totals.get("hard_negative_count", 0.0),
                "margin_active_negative_count": totals.get(
                    "margin_active_negative_count", 0.0
                ),
                "all_positive_count": totals.get("all_positive_count", 0.0),
                "all_negative_count": totals.get("all_negative_count", 0.0),
                "positive_loss": totals.get("positive_loss", 0.0),
                "negative_loss": totals.get("negative_loss", 0.0),
                "uniformity_loss": totals.get("uniformity_loss", 0.0),
                "anti_collapse_loss": totals.get("anti_collapse_loss", 0.0),
                "tracking_batches": float(batches),
            }
            result.update(
                {
                    key: value
                    for key, value in totals.items()
                    if key.startswith("negative_source_")
                }
            )
            result["margin_active_negative_fraction"] = (
                result["margin_active_negative_count"]
                / result["hard_negative_count"]
                if result["hard_negative_count"]
                else 0.0
            )
            total_loss = result["positive_loss"] + result["negative_loss"]
            result["negative_loss_fraction"] = (
                result["negative_loss"] / total_loss if total_loss else 0.0
            )
            return result

        def coverage_stats(self) -> dict[str, float]:
            selected = len(self._seen_hard_negative_ids)
            active = len(self._seen_margin_active_negative_ids)
            total = self._total_negative_pairs
            return {
                "contrastive_margin": float(self.margin),
                "label_smoothing": float(label_smoothing),
                "negative_cosine_target": float(1.0 - self.margin),
                "n_train_neg_total": float(total),
                "n_train_neg_hard_selected_unique": float(selected),
                "n_train_neg_margin_active_unique": float(active),
                "train_neg_hard_selection_coverage": selected / total if total else 0.0,
                "train_neg_margin_active_coverage": active / total if total else 0.0,
                "n_train_neg_present_unique": float(len(self._negative_present_counts)),
                "n_train_neg_backprop_unique": float(len(self._negative_backprop_counts)),
                "train_neg_backprop_events": float(sum(self._negative_backprop_counts.values())),
            }

        def pair_usage_rows(self) -> list[dict]:
            """Return cumulative per-pair usage and gradient attribution."""
            ids = set(self._negative_present_counts)
            ids.update(self._negative_selected_counts)
            ids.update(self._negative_backprop_counts)
            rows = []
            for pair_id in sorted(ids):
                lineage = (
                    self._pair_lineage[pair_id]
                    if pair_id < len(self._pair_lineage)
                    else {}
                )
                rows.append(
                    {
                        "pair_id": pair_id,
                        "present_count": self._negative_present_counts.get(pair_id, 0),
                        "hard_selected_count": self._negative_selected_counts.get(pair_id, 0),
                        "backprop_count": self._negative_backprop_counts.get(pair_id, 0),
                        **lineage,
                    }
                )
            return rows

        def pair_usage_rows_by_epoch(self) -> list[dict]:
            """Return per-epoch pair presentation/selection/backprop counts."""
            rows = []
            for epoch in sorted(self._per_epoch_counts):
                for pair_id in sorted(self._per_epoch_counts[epoch]):
                    lineage = (
                        self._pair_lineage[pair_id]
                        if pair_id < len(self._pair_lineage)
                        else {}
                    )
                    rows.append(
                        {
                            "epoch": epoch,
                            "pair_id": pair_id,
                            **self._per_epoch_counts[epoch][pair_id],
                            **lineage,
                        }
                    )
            return rows

    return _TrackedOnlineContrastiveLoss(model, margin=margin)


def _runtime_telemetry() -> dict[str, float | int]:
    """Cheap process and CUDA facts emitted with each training heartbeat."""
    telemetry: dict[str, float | int] = {"pid": os.getpid()}
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                telemetry["rss_mb"] = round(int(line.split()[1]) / 1024, 1)
                break
    except OSError:
        pass
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
    return telemetry


def _format_telemetry(values: dict[str, float | int]) -> str:
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


def _wandb_memory_metrics(values: dict[str, float | int]) -> dict[str, float]:
    """Return the only system telemetry allowed into W&B."""
    names = {
        "rss_mb": "memory/worker_rss_mb",
        "memory_used_mb": "memory/total_used_mb",
        "memory_available_mb": "memory/total_available_mb",
    }
    return {
        target: float(values[source])
        for source, target in names.items()
        if source in values
    }


class ProgressCallback(TrainerCallback):
    """Live per-step display of train loss + dev AP/AUC during training.

    The modern Trainer path replaces 07b's log_steps=True (which wrapped the
    loss module's forward to print every batch). This is the equivalent on the
    HF contract: on_log fires at logging_steps and carries the running train
    loss; on_evaluate fires at eval_steps and carries the dev metrics the
    early-stopper is actually watching.
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
        self._trace_rows: list[dict[str, object]] = []
        self.latest_train_loss: float | None = None
        self.latest_dev_accuracy: float | None = None
        self.latest_collapse_metrics: dict[str, float | int | str] = {}
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
        self.collapse_model = collapse_model
        self.collapse_df = collapse_df
        self.collapse_payload = collapse_payload
        self.collapse_config = collapse_config
        self.collapse_batch_size = collapse_batch_size

    def _collapse_metrics(self, evaluation_step: int) -> dict[str, float | int | str]:
        """Run the shared unrelated-pair diagnostic on the current model."""
        if self.collapse_model is None:
            return {}
        from training.uniformity import collapse_diagnostics

        return collapse_diagnostics(
            model=self.collapse_model,
            df=self.collapse_df,
            payload=self.collapse_payload,
            config=self.collapse_config,
            batch_size=int(self.collapse_batch_size),
            trace_path=(
                self.trace_path.with_name(f"{self.trace_path.stem}_collapse_pairs.csv")
                if self.trace_path is not None
                else None
            ),
            evaluation_step=evaluation_step,
        )

    @staticmethod
    def _collapse_wandb_metrics(
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

    def _write_live_status(self, state, event: str, **values) -> None:
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

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            telemetry = _runtime_telemetry()
            print(f"    [telemetry] training-started | {_format_telemetry(telemetry)}", flush=True)
            if self.wandb_ctx is not None:
                self.wandb_ctx.log_metrics(_wandb_memory_metrics(telemetry))
            self._write_live_status(state, "training-started", **telemetry)
        return control

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = int((state.epoch or 0.0)) + 1
        if self.tracked_loss is not None and hasattr(self.tracked_loss, "set_epoch"):
            self.tracked_loss.set_epoch(epoch)
        dynamic_ref = getattr(self.tracked_loss, "_dynamic_epoch_ref", None)
        if dynamic_ref is not None:
            dynamic_ref["epoch"] = epoch
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not state.is_world_process_zero:
            return
        if "loss" in logs:
            loss = float(logs["loss"])
            self.latest_train_loss = loss
            loss_stats = (
                self.tracked_loss.pop_tracking_stats()
                if self.tracked_loss is not None
                and hasattr(self.tracked_loss, "pop_tracking_stats")
                else {}
            )
            self._trace_rows.append(
                {
                    "step": float(state.global_step),
                    "epoch": float(state.epoch or 0.0),
                    "train_loss": loss,
                    "grad_norm": float(logs["grad_norm"])
                    if logs.get("grad_norm") is not None
                    else float("nan"),
                    **loss_stats,
                }
            )
            telemetry = _runtime_telemetry()
            total_epochs = float(args.num_train_epochs)
            accuracy = (
                f" | dev_acc {self.latest_dev_accuracy:.4f}"
                if self.latest_dev_accuracy is not None
                else ""
            )
            print(
                f"    [epoch {state.epoch:>5.2f}/{total_epochs:g} | step {state.global_step:>4}/"
                f"{state.max_steps:<4}] train_loss {loss:.4f}{accuracy} | {_format_telemetry(telemetry)}",
                flush=True,
            )
            if self.wandb_ctx is not None:
                self.wandb_ctx.log_metrics(
                    {
                        "live/train_loss": loss,
                        "live/epoch": float(state.epoch or 0.0),
                        "live/grad_norm": float(logs["grad_norm"])
                        if logs.get("grad_norm") is not None
                        else None,
                        **{
                            f"live/loss_{key}": value
                            for key, value in loss_stats.items()
                        },
                        **_wandb_memory_metrics(telemetry),
                    },
                )
            self._write_live_status(
                state,
                "train",
                train_loss=loss,
                dev_accuracy=self.latest_dev_accuracy,
                grad_norm=(
                    float(logs["grad_norm"])
                    if logs.get("grad_norm") is not None
                    else None
                ),
                **{f"loss_{key}": value for key, value in loss_stats.items()},
                **telemetry,
            )

    def on_train_end(self, args, state, control, **kwargs):
        if self.trace_path is not None and self._trace_rows:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            # A fold owns this file; write mode keeps reruns from appending
            # stale optimizer telemetry from an earlier attempt.
            pd.DataFrame(self._trace_rows).to_csv(self.trace_path, index=False, mode="w")
            print(f"    [loss-trace] wrote {self.trace_path}", flush=True)
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or not state.is_world_process_zero:
            return
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
        self.latest_dev_accuracy = accuracy
        collapse_metrics = self._collapse_metrics(int(state.global_step))
        self.latest_collapse_metrics = dict(collapse_metrics)
        if collapse_metrics:
            self._trace_rows.append(
                {
                    "step": float(state.global_step),
                    "epoch": float(state.epoch or 0.0),
                    "event": "evaluation",
                    **collapse_metrics,
                }
            )
        telemetry = _runtime_telemetry()
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
                f"train_loss {self.latest_train_loss:.4f} | "
                if self.latest_train_loss is not None
                else ""
            )
            print(
                f"    [epoch {state.epoch:>5.2f}/{total_epochs:g} | step "
                f"{state.global_step:>4}/{state.max_steps:<4}] {loss}" + " | ".join(parts),
                flush=True,
            )
        if self.wandb_ctx is not None:
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
                    **self._collapse_wandb_metrics(collapse_metrics),
                    **_wandb_memory_metrics(telemetry),
                },
            )
        self._write_live_status(
            state,
            "evaluation",
            train_loss=self.latest_train_loss,
            dev_loss=float(metrics["eval_loss"]) if metrics.get("eval_loss") is not None else None,
            dev_average_precision=ap,
            dev_accuracy=self.latest_dev_accuracy,
            dev_f1=f1,
            dev_precision=precision,
            dev_recall=recall,
            **collapse_metrics,
            **telemetry,
        )


class LateEpochLrDecayCallback(TrainerCallback):
    """Apply the SSOT late-epoch LR reduction exactly once.

    The HF scheduler still owns its normal warmup/linear schedule. At the
    configured later-epoch boundary we scale both optimizer and scheduler
    base LRs, so the reduction survives subsequent scheduler steps and is
    preserved in resumable optimizer state.
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

    def on_epoch_begin(self, args, state, control, **kwargs):
        if not self.enabled or self.applied:
            return control
        current_epoch = float(state.epoch or 0.0)
        boundary = float(args.num_train_epochs) * self.start_epoch_fraction
        if current_epoch + 1e-9 < boundary:
            return control

        optimizer = kwargs.get("optimizer")
        scheduler = kwargs.get("lr_scheduler")
        if optimizer is None:
            raise RuntimeError(
                "late-epoch LR decay reached its boundary without an optimizer"
            )
        self.learning_rates_before = [
            float(group["lr"]) for group in optimizer.param_groups
        ]
        for group in optimizer.param_groups:
            group["lr"] = float(group["lr"]) * self.multiplier
            if "initial_lr" in group:
                group["initial_lr"] = float(group["initial_lr"]) * self.multiplier
        if scheduler is not None and hasattr(scheduler, "base_lrs"):
            scheduler.base_lrs = [
                float(lr) * self.multiplier for lr in scheduler.base_lrs
            ]
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
        return control


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
        if os.environ.get("EUROMONITOR_DISABLE_DVC_CHECKPOINTS"):
            print("    [checkpoint-dvc] skipped: disabled for HPO retention mode", flush=True)
            return control
        if not os.environ.get("DVC_API_KEY"):
            print("    [checkpoint-dvc] skipped: DVC_API_KEY absent", flush=True)
            return control
        checkpoint_root = Path(args.output_dir)
        checkpoint = checkpoint_root / f"checkpoint-{state.global_step}"
        _make_checkpoint_tokenizer_portable(checkpoint)
        required = (
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            "checkpoint_manifest.json",
            "trainer_state.json",
        )
        missing = [name for name in required if not (checkpoint / name).is_file()]
        if missing:
            raise RuntimeError(
                f"checkpoint is not resumable: {checkpoint}; "
                f"missing {', '.join(missing)}"
            )
        snapshot = self._snapshot(checkpoint)
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
        row_barcodes,
        structured_features,
        train_barcodes,
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
        self.row_barcodes = row_barcodes
        self.structured_features = structured_features
        self.train_barcodes = train_barcodes
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

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control
        model = self.model
        if model is None:
            raise RuntimeError(
                "FineTunedAnnRefreshCallback was created without the live model"
            )
        ann_cfg = load_config()["mining"]["ann"]
        attr_cfg = load_config()["mining"]["attribute_conflict"]
        if not bool(ann_cfg["refresh_enabled"]) and not bool(attr_cfg["enabled"]):
            return control
        epoch = float(state.epoch or 0.0)
        cadence = int(ann_cfg["refresh_every_epochs"])
        completed_epoch = int(np.floor(epoch + 1e-8))
        if completed_epoch < self.last_epoch + cadence:
            return control
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
                self.row_barcodes,
                structured_features=self.structured_features,
                train_barcodes=self.train_barcodes,
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
                    "step", "epoch", "row_a", "row_b", "barcode_a", "barcode_b",
                    "cosine", "band_lo", "band_hi", "band_mode", "source",
                ]
            ).to_csv(audit_path, index=False, mode="w")
        from core.hard_negatives import mine_attribute_conflict_negatives

        attr_lo, attr_hi = (float(x) for x in str(attr_cfg["band"]).split("-"))
        if bool(attr_cfg["enabled"]):
            attr_pairs, attr_scores = mine_attribute_conflict_negatives(
                self.df,
                self.payload,
                self.row_barcodes,
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
                attr_pairs, self.row_barcodes, set(self.train_barcodes)
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
                    "barcode_a": str(self.row_barcodes[a]),
                    "barcode_b": str(self.row_barcodes[b]),
                    "cosine": float(score),
                    "source": source,
                }
                for source, score, a, b in selected
                if source == "attribute_conflict"
            ],
            columns=[
                "step", "epoch", "row_a", "row_b", "barcode_a", "barcode_b",
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
        print(
            f"    [ann-refresh] fold {self.fold_i}: step={state.global_step} "
            f"epoch={epoch:.2f} pairs={len(pairs):,} "
            f"ann={selected_sources.count('ann_finetuned'):,} "
            f"attribute_conflict={selected_sources.count('attribute_conflict'):,} "
            f"band={stats.get('band_lo', lo):.4f}-{stats.get('band_hi', hi):.4f} "
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
                },
                step=int(state.global_step),
            )
        return control


def retain_hpo_champion(
    *, model_id: str, run_tag: str, value: float, folds: list[int]
) -> bool:
    """Keep exactly one completed HPO trial's bulky local artifacts per model.

    Optuna may complete two trials concurrently.  The per-model lock makes
    comparison, removal of the previous champion, and champion-record update
    one transaction.  This mode deliberately retains local artifacts only;
    DVC checkpoint publishing is disabled by the HPO launcher.
    """
    import shutil
    import tempfile

    model_tag = model_id.rstrip("/").rsplit("/", 1)[-1]
    record = RESULTS / f"hpo_{model_tag}_champion.json"
    lock_path = RESULTS / f".hpo-{model_tag}-retention.lock"

    def artifacts(tag: str, fold_numbers: list[int]) -> list[Path]:
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

    def remove(paths: list[Path]) -> None:
        for path in paths:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)

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


def _load_canonical_metadata() -> dict[str, dict]:
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
    return {
        str(row["gtin"]): row.to_dict()
        for _, row in records.iterrows()
    }


def _sku_payload_metadata(index: int, row, barcode: str, text: str) -> dict:
    from core.attribute_conflicts import sku_attribute_info

    attributes = row_metadata_text(row, "attributes", "attr")
    info = sku_attribute_info(row_metadata_text(row, "title"), attributes)
    return {
        "payload_idx": index,
        "point_kind": "sku",
        "source_payload_idx": index,
        "sku_id": row_metadata_text(row, "product_id", "SKU_ID"),
        "gtin": barcode,
        "brand": row_metadata_text(row, "brand"),
        "title": row_metadata_text(row, "title"),
        "attributes": attributes,
        "country": row_metadata_text(row, "country"),
        "category": row_metadata_text(row, "category", "category_path"),
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
    index: int, record: dict, barcode: str, text: str
) -> dict:
    from core.attribute_conflicts import canonical_attribute_info

    info = canonical_attribute_info(record)
    return {
        "payload_idx": index,
        "point_kind": "canonical",
        "source_payload_idx": index,
        "sku_id": "",
        "gtin": barcode,
        "brand": metadata_text(record["mode_brand"]),
        "title": metadata_text(record["canonical"]),
        "attributes": "",
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


def _load_gate_lookup() -> dict[tuple[str, str], dict[str, object]]:
    gate_path = F["gate_results"]
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
    return lookup


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
        int(item["copy_payload_idx"]): int(item["anchor_payload_idx"])
        for item in list(mask_audit or []) + list(hard_negative_mask_audit or [])
        if item.get("copy_payload_idx") is not None
        and item.get("anchor_payload_idx") is not None
    }
    metadata: list[dict] = []
    for index, value in enumerate(row_bc):
        barcode = str(value)
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
                _sku_payload_metadata(index, df.iloc[index], barcode, str(payload[index]))
            )
        else:
            if barcode not in canonical_map:
                raise ValueError(
                    "payload canonical has no canonical metadata: "
                    f"{barcode}"
                )
            metadata.append(
                _canonical_payload_metadata(
                    index,
                    canonical_map[barcode],
                    barcode,
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
            "title",
            "attributes",
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
    model ingests for this fold — literal texts, labels, barcodes, and
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
                "barcode_a": row_bc[a],
                "barcode_b": row_bc[b],
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
        }
        lookup[(label, anchor, target)] = base
        lookup[(label, int(audit["copy_payload_idx"]), target)] = {
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
            str(train_neg_sources[i])
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
    sources = [str(source) for source in np.unique(tr_neg_sources)]
    totals = {
        source: int(np.sum(tr_neg_sources == source)) for source in sources
    }
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
):
    """Freshly mask selected label-0 anchors whenever a batch is materialized."""
    from training.masking import mask_text

    transformed = {key: list(values) for key, values in batch.items()}
    for i, label in enumerate(batch["label"]):
        pair_id = int(batch["pair_id"][i])
        base_population = (
            str(pair_populations[pair_id])
            if pair_populations is not None and pair_id < len(pair_populations)
            else ("positive" if int(label) else "hard_negative")
        )
        ann_version = int(ann_state.get("version", 0)) if ann_state else 0
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
                key = (int(epoch_ref["epoch"]), pair_id, base_population, augmentation, ann_version)
                presentation_counts[key] = presentation_counts.get(key, 0) + 1
            continue
        epoch_stats = stats_by_epoch.setdefault(
            int(epoch_ref["epoch"]),
            {"negative_presented": 0.0, "masked_count": 0.0, "extent_sum": 0.0},
        )
        epoch_stats["negative_presented"] += 1.0
        if rng.random() >= frac:
            if presentation_counts is not None:
                key = (int(epoch_ref["epoch"]), pair_id, base_population, augmentation, ann_version)
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
        augmentation = "dynamic_mask"
        epoch_stats["masked_count"] += 1.0
        epoch_stats["extent_sum"] += float(_extent)
        counts[pair_id] = counts.get(pair_id, 0) + 1
        epoch_counts = counts_by_epoch.setdefault(int(epoch_ref["epoch"]), {})
        epoch_counts[pair_id] = epoch_counts.get(pair_id, 0) + 1
        if presentation_counts is not None:
            key = (int(epoch_ref["epoch"]), pair_id, base_population, augmentation, ann_version)
            presentation_counts[key] = presentation_counts.get(key, 0) + 1
        if mask_audit is not None:
            mask_audit.append(
                {
                    "fold": fold,
                    "epoch": int(epoch_ref["epoch"]),
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


def _partition_calibration_pairs(
    positive_pairs: np.ndarray,
    negative_pairs: np.ndarray,
    row_bc: np.ndarray,
    calibration_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reserve component-safe calibration pairs from early-stop DEV."""
    from training.folds import partition_component_pairs

    return partition_component_pairs(
        positive_pairs,
        negative_pairs,
        row_bc,
        calibration_fraction,
        seed,
        ensure_different_gtin=True,
    )


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
    selection_rule = load_config()["rand_matching"][
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
    # dev_override: explicit dev barcode set (component-aware splits pass it;
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
    wandb_ctx=None,
) -> list[dict]:
    """Train cfg across the group-aware folds. Returns fold metric rows
    (failures included, with traceback)."""
    import torch
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
    calibration_config = load_config()
    calibration_fraction = float(
        calibration_config["split"]["calibration_dev_fraction"]
    )
    calibration_seed_offset = int(
        calibration_config["split"]["calibration_seed_offset"]
    )

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
    payload_metadata, gate_lookup = _build_payload_metadata(
        df,
        payload,
        row_bc,
        mask_audit=mask_audit,
        hard_negative_mask_audit=hard_negative_mask_audit,
    )
    # Masked positive copies are augmentation for training only.  Splits are
    # barcode-based, so passing the augmented array directly into dev/test
    # would silently put those copies into evaluation even though they carry
    # the same barcode as the original SKU.  Keep the augmented ``pos`` for
    # train-side selection, but remove copy endpoints from evaluation pools.
    _masked_copy_ids = {
        int(row["copy_payload_idx"])
        for row in (mask_audit or [])
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
    labeled_different_pos = _load_labeled_different_positive_pairs(
        eval_pos=eval_pos,
        row_bc=row_bc,
        n_source_rows=len(df),
    )
    all_barcode_set = set(row_bc.tolist())

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
        # folds_override contract: EITHER one barcode-set (holdout mode: that
        # set is the single test fold; train = every other barcode) OR a list
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
        # for n_folds <= CV_FOLDS is IDENTICAL: kfold_barcodes deals the
        # same strided permutation split, and the --quick prefix slice is
        # unchanged.
        n_folds = cv_folds if cv_folds is not None else CV_FOLDS
        all_folds = kfold_barcodes(df, n_folds, SEED)
        # --quick trains on the first n_folds of the SAME split (folds stay comparable)
        folds = all_folds[:n_folds]

    # mine ONCE outside the fold loop (band is fixed per config); the EVAL
    # mining band comes from the config SSOT (bands.eval_mining) — was a
    # hardcoded (0.35, 0.90) inline, a second declaration the config could
    # not steer. NO FALLBACK (owner Q27): band() raises when the key is
    # missing — the old "if present else (0.35, 0.90)" branch silently
    # resurrected the inline literal.
    # NOTE: the `band` PARAMETER (tuple) shadows lib.common.band() in this
    # function scope — alias the import.
    # Profile-selected masking-only runs pass an empty embedding matrix.
    # Guard on the actual input as well as the import-time default so a
    # worker profile cannot invoke even an empty ANN audit.
    if ANN_MINING_ENABLED and emb0.size:
        from core.common import band as _band_helper

        _eval_band = _band_helper("eval_mining")
        hard_train_all, _ = mine_hard_negatives(
            df, emb0, n_target=N_TARGET_MINING, cosine_lo=band[0], cosine_hi=band[1]
        )
        hard_eval, _ = mine_hard_negatives(
            df, emb0, n_target=N_TARGET_MINING,
            cosine_lo=_eval_band[0], cosine_hi=_eval_band[1],
        )
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
    from core.ranking_metrics import component_index

    # Component ids over the positive-pair graph: the unit
    # training.folds.component_folds splits on.  Recomputed here (the split
    # helper returns fold membership, not component identity) and asserted
    # fold-pure below, so a competitor drawn from ANOTHER component of the
    # SAME fold provably shares no positive-pair chain with the query.
    _retrieval_row_component = component_index(pos, row_bc)

    def _canonical_payload_rows() -> np.ndarray:
        """Payload rows that are CANONICAL entries (the competitor universe).

        EXACT, not a boundary heuristic: ``pipeline.build_training_data``
        appends one canonical per GTIN in sorted order immediately after the
        source rows, so the canonical block is ``[len(df), len(df) + n_gtins)``
        and ``n_gtins`` is the canonical map's own size.  Masked-anchor copies
        are appended AFTER that block and are therefore never candidates —
        which the masked-copy boundary heuristic gets wrong when only
        hard-negative augmentation (no positive augmentation) is enabled,
        because then no copy appears as a positive's first endpoint at all.
        The barcode set of the block is asserted to equal the canonical map,
        so a payload-layout change fails loudly instead of silently shifting
        the candidate universe.
        """
        from pipeline import load_canonical_map

        canon_map = load_canonical_map()
        end = len(df) + len(canon_map)
        if end > len(payload):
            raise ValueError(
                f"canonical block [{len(df)}, {end}) exceeds the payload "
                f"({len(payload)} entries) — the payload layout changed"
            )
        rows = np.arange(len(df), end, dtype=int)
        if {str(b) for b in row_bc[rows]} != set(canon_map):
            raise ValueError(
                "canonical payload block does not carry the canonical map's "
                "GTINs — refusing to build the competitor universe from rows "
                "that are not canonicals"
            )
        return rows

    def _holdout_true_match_barcode_pairs() -> frozenset[tuple[str, str]]:
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

    _retrieval_canonical_rows = _canonical_payload_rows()
    _retrieval_true_match_pairs = _holdout_true_match_barcode_pairs()
    print(
        f"[retrieval-pool] canonical competitor universe={len(_retrieval_canonical_rows):,} "
        f"payload rows | known true-match barcode pairs excluded="
        f"{len(_retrieval_true_match_pairs) // 2:,} (labeled positives + gate "
        f"proceed + identical canonical identity)",
        flush=True,
    )

    rows: list[dict] = []
    for fold_i, test_bc in enumerate(folds):
        try:
            t_fold = time.perf_counter()
            if len(folds) > 1:
                train_bc = set().union(*[f for j, f in enumerate(folds) if j != fold_i])
            else:
                # single holdout fold: train side = every barcode NOT in the
                # test fold (dev_override carves dev out of this below)
                train_bc = all_barcode_set - folds[0]

            # split train barcodes into train/dev (early stopping target).
            # dev_override: caller-supplied component-aware dev boundary
            # (skips the rng carve — a barcode-level carve SPLITS positive
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
            # leak) when the boundary is wrong: test barcodes in
            # train/dev, or dev barcodes in the test fold, would leak the
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
                    f"[hpo] LEAK: {len(_dev_in_test)} dev_override barcodes "
                    f"are in the test fold (e.g. {sorted(_dev_in_test)[:3]}) "
                    "— they would be silently dropped from dev while the "
                    "split claims to be clean"
                )
                _leak = test_bc & (tr_bc | dev_bc)
                assert not _leak, (
                    f"[hpo] LEAK: {len(_leak)} test barcodes in train/dev "
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
                    f"test={len(test_bc):,} barcodes | test quarter "
                    f"excluded from training+selection",
                    flush=True,
                )

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
            # barcode belongs to the test fold, so every query's ranking task
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
                        train_barcodes=tr_bc,
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
                dev_pos, calibration_pos, hard_dev, calibration_neg = (
                    _partition_calibration_pairs(
                        dev_pos,
                        hard_dev,
                        row_bc,
                        calibration_fraction,
                        seed + fold_i + calibration_seed_offset,
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
            train_barcodes = set(row_bc[train_all[:, 0]].tolist()) if len(train_all) else set()
            static_masked_pos = sum(
                1 for item in (mask_audit or [])
                if str(item.get("barcode", "")) in train_barcodes
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

            checkpoint_dir = artifact(
                "checkpoint_repo",
                {
                    "model_tag": model_id.rstrip("/").rsplit("/", 1)[-1],
                    "run_tag": run_tag,
                    "fold": fold_i,
                    "step": 0,
                },
            ).parent
            if resume:
                from training.dvc_store import restore_checkpoint

                restore_checkpoint(RESULTS, checkpoint_dir)
                print(f"    [resume] restored {checkpoint_dir} from DVC", flush=True)

            ensure_parent(checkpoint_dir)

            # Tied-weight two-tower retrieval model: the trainer receives
            # (SKU text, canonical text) pairs; each side is encoded on its
            # own before cosine/loss comparison.  CrossEncoder is optional
            # only in rerank.py after retrieval, never this default path.
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
                # restricted to TRAIN barcodes (component boundary holds:
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
                s1 = [payload[a] for a, b in train_all] + [
                    payload[a] for a, b in tr_negs
                ]
                s2 = [payload[b] for a, b in train_all] + [
                    payload[b] for a, b in tr_negs
                ]
                lab = [1] * len(train_all) + [0] * len(tr_negs)
                hp_train_for_tracking = (
                    hp_pairs[pairs_in_set(hp_pairs, row_bc, tr_bc)]
                    if use_hp and hp_pairs is not None and len(hp_pairs)
                    else None
                )
                pair_populations = _training_pair_populations(
                    train_all,
                    tr_negs,
                    train_neg_sources=tr_neg_sources,
                    hp_in_train=hp_train_for_tracking,
                    mask_audit=mask_audit,
                )
                presentation_counts: dict[tuple, int] = {}
                train_ds = Dataset.from_dict(
                    {
                        "sentence1": s1,
                        "sentence2": s2,
                        "label": lab,
                        "pair_id": list(range(len(s1))),
                        "structured_features": [
                            [structured_features[int(a)].tolist(), structured_features[int(b)].tolist()]
                            for a, b in list(train_all) + list(tr_negs)
                        ],
                    }
                )
                dynamic_mask_counts: dict[int, int] = {}
                dynamic_mask_counts_by_epoch: dict[int, dict[int, int]] = {}
                dynamic_epoch_ref = {"epoch": 0}
                if (
                    (dynamic_mask_hard_negatives and dynamic_mask_frac > 0)
                    or ann_refresh_enabled
                    or TRACK_DATAPOINT_USAGE
                ):
                    import random as _random
                    from functools import partial

                    _mask_rng = _random.Random(seed + fold_i + 100_003)
                    train_ds.set_transform(
                        partial(
                            _dynamic_mask_negative_transform,
                            rng=_mask_rng,
                            frac=dynamic_mask_frac,
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
                            mask_audit=hard_negative_mask_audit,
                            fold=fold_i,
                        )
                    )
                # ── TRAIN VISIBILITY (owner directive 2026-09-07): the
                # EXACT rows the model ingests for this fold — sentence1,
                # sentence2, label, both barcodes, pos/hp/neg provenance.
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
                positive_by_anchor: dict[int, int] = {}
                for anchor, positive in train_all:
                    positive_by_anchor.setdefault(int(anchor), int(positive))
                triples: list[tuple[int, int, int]] = []
                seen_triples: set[tuple[int, int, int]] = set()
                for anchor, negative in tr_negs:
                    anchor_i, negative_i = int(anchor), int(negative)
                    positive_i = positive_by_anchor.get(anchor_i)
                    if positive_i is None or positive_i == negative_i:
                        continue
                    triple = (anchor_i, positive_i, negative_i)
                    if triple not in seen_triples:
                        seen_triples.add(triple)
                        triples.append(triple)
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
                train_ds = Dataset.from_dict(
                    {
                        "anchor": [payload[a] for a, _, _ in triples],
                        "positive": [payload[b] for _, b, _ in triples],
                        "negative": [payload[c] for _, _, c in triples],
                    }
                )
            else:
                from core.hard_negatives import build_triplets

                examples = build_triplets(
                    train_all,
                    hard_train,
                    payload,
                    seed=seed + fold_i,
                    max_triples=MAX_TRIPLES,
                )
                if not examples:
                    raise RuntimeError("no triples built for fold")
                train_ds = Dataset.from_dict(
                    {
                        "anchor": [ex.texts[0] for ex in examples],
                        "positive": [ex.texts[1] for ex in examples],
                        "negative": [ex.texts[2] for ex in examples],
                    }
                )

            batch_size = BATCH_SIZE_CUDA if on_cuda else BATCH_SIZE_CPU
            n_steps_per_epoch = max(1, len(train_ds) // batch_size)
            warmup_steps = int(n_steps_per_epoch * cfg["epochs"] * cfg["warmup_ratio"])
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
            _sf_cfg = load_config()["training"]["structured_features"]
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
            # Checkpoints KEPT but bounded: save_only_model + save_total_limit=2
            # caps disk at ~2x model size (~1 GB L12 / ~180 MB L6) — no explosion,
            # and load_best_model_at_end restores the best epoch. Unique subdir
            # per run_tag so parallel trials never collide.
            from sentence_transformers import SentenceTransformerTrainer
            from sentence_transformers.sentence_transformer.data_collator import (
                SentenceTransformerDataCollator,
            )
            from sentence_transformers import (
                SentenceTransformerTrainingArguments as STArgs,
            )

            class PairIdDataCollator(SentenceTransformerDataCollator):
                """Keep telemetry and structured features out of tokenization."""

                def __call__(self, features):
                    text_features = [
                        {
                            key: value
                            for key, value in row.items()
                            if key not in {"pair_id", "structured_features"}
                        }
                        for row in features
                    ]
                    batch = super().__call__(text_features)
                    # Training rows carry pair_id; evaluator rows do not.
                    # Detect the field structurally rather than treating a
                    # list of optional values as a valid batch. If a training
                    # row is malformed, direct indexing raises loudly.
                    if features and "pair_id" in features[0]:
                        batch["pair_id"] = torch.tensor(
                            [row["pair_id"] for row in features], dtype=torch.long
                        )
                    if features and "structured_features" in features[0]:
                        batch["structured_features"] = torch.tensor(
                            [row["structured_features"] for row in features],
                            dtype=torch.float32,
                        )
                    return batch

            class ResumableSentenceTransformerTrainer(SentenceTransformerTrainer):
                """HF Trainer plus an explicit manifest of all resume state."""

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

                def _save_checkpoint(self, model, trial):
                    super()._save_checkpoint(model, trial)
                    checkpoint = (
                        Path(self._get_output_dir(trial=trial))
                        / f"checkpoint-{self.state.global_step}"
                    )
                    _make_checkpoint_tokenizer_portable(checkpoint)
                    _write_checkpoint_manifest(
                        checkpoint,
                        epoch=self.state.epoch,
                        global_step=self.state.global_step,
                        model=model,
                        optimizer=self.optimizer,
                        scheduler=self.lr_scheduler,
                        scaler=getattr(self.accelerator, "scaler", None),
                        trainer_state=self.state,
                        trainer_control=self.control,
                        training_args=self.args,
                    )
                    trace_artifact("checkpoint_repo", checkpoint, producer="training.training")

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
                bf16=on_cuda and torch.cuda.is_bf16_supported(),
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
                save_total_limit=int(runtime("save_total_limit")),  # SSOT
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

            optimizer = optim.AdamW(
                groups, weight_decay=cfg["weight_decay"], lr=base_lr
            )

            loss_fn = _make_loss(
                model,
                loss,
                structured_feature_weight=structured_feature_weight,
                uniformity_weight=float(cfg["uniformity_weight"]),
                uniformity_temperature=float(_UNIFORMITY_CFG["temperature"]),
                uniformity_min_batch_size=int(_UNIFORMITY_CFG["min_batch_size"]),
                label_smoothing=float(cfg["label_smoothing"]),
            )
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
                    trace_path=RESULTS
                    / "logs"
                    / run_tag
                    / f"loss_backprop_fold{fold_i}.csv",
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
                DvcCheckpointCallback(),
                EarlyStoppingCallback(
                    early_stopping_patience=cfg["patience"],
                    early_stopping_threshold=cfg["es_threshold"],
                ),
            ]
            if (
                (ann_refresh_enabled or attribute_conflict_refresh_enabled)
                and loss == "contrastive"
            ):
                ann_cfg = load_config()["mining"]["ann"]
                callbacks.append(
                    FineTunedAnnRefreshCallback(
                        df=df,
                        payload=payload,
                        row_barcodes=row_bc,
                        structured_features=structured_features,
                        train_barcodes=set(tr_bc),
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
                    required = (
                        "optimizer.pt",
                        "scheduler.pt",
                        "rng_state.pth",
                        "checkpoint_manifest.json",
                        "trainer_state.json",
                    )
                    missing = [name for name in required if not (latest / name).is_file()]
                    if missing:
                        raise RuntimeError(
                            f"cannot resume {latest}: checkpoint lacks trainer state; "
                            f"missing {', '.join(missing)}. Start a new run once "
                            "to create resumable checkpoints."
                        )
                    resume_checkpoint = str(latest)
                    print(f"    [resume] fold {fold_i}: {resume_checkpoint}", flush=True)
                else:
                    print(f"    [resume] fold {fold_i}: no checkpoint found; starting fresh", flush=True)
            trainer.train(resume_from_checkpoint=resume_checkpoint)
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

            # Every lane uses the same component-safe calibration/Rand
            # computation. The holdout population below remains isolated for
            # final reporting and is never used by HPO selection.
            from training.hpo_metrics import (
                CALIBRATION_REASON_EMPTY_SPLIT,
                evaluate_calibration_trial,
                unavailable_calibration_metrics,
            )

            calibration_metrics: dict[str, object]
            if len(calibration_pos) == 0 or len(calibration_neg) == 0:
                calibration_metrics = unavailable_calibration_metrics(
                    reason_code=CALIBRATION_REASON_EMPTY_SPLIT,
                    reason=(
                        "empty calibration split — Rand threshold calibration "
                        f"needs pos={len(calibration_pos)}, neg={len(calibration_neg)}"
                    ),
                    positive_pairs=len(calibration_pos),
                    negative_pairs=len(calibration_neg),
                )
                print(
                    f"  [calibration] fold {fold_i}: REQUIRED calibration "
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
            else:
                # The explicit empty-population branch above is the only
                # expected unavailable-calibration condition.  An exception
                # from the evaluator is a programming/data-contract failure,
                # including in selection mode, and must retain its traceback
                # instead of becoming a prunable/unavailable result.
                try:
                    calibration_metrics = evaluate_calibration_trial(
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

            # ── SELECTION-MODE EXIT (test-leak fix, 2026-09-12) ───────────
            # Holdout HPO/grid folds STOP HERE: the config is ranked on
            # calibration Rand and the test quarter's eval block is never
            # entered — no pair_auc, no PR-AUC, no Youden, no pair dump,
            # not even an encode. The test quarter is read exactly once, by
            # the main train lane, so no per-config test metric can ever
            # exist to select on. Recorded LOUDLY (explicit field + print),
            # never as a silent NaN.
            if selection_mode and HPO_SKIP_TEST_EVAL:
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
                            if calibration_metrics.get("calibration_status") == "available"
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
            random_neg_pairs = _split_safe_random_negative_pairs(
                df,
                row_bc,
                set(test_bc),
                seed=SEED + fold_i + 1000,
                n_neg=int(load_config()["pairs"]["n_neg"]),
            )
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

            _thr = _youden_thr(_dev_all, _dev_y)
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
            # ══════════════════════════════════════════════════════════════
            # HOLDOUT RETRIEVAL (ER-346) — two protocols, both reported
            # ══════════════════════════════════════════════════════════════
            # OLD PROTOCOL (retained, renamed, and PROVEN degenerate by its own
            # coverage record below): the pool was np.vstack([test_pos,
            # hard_test]) grouped by source product_id.  Negatives are anchored
            # only at the single representative row per barcode, so 5,292 of
            # 5,847 holdout queries saw EXACTLY their own positive and no query
            # ever saw more than 6 candidates (< max(ks)=10).  With the positive
            # first in a stable sort, a perfect oracle and a constant scorer
            # both read Hits@1 = Precision@1 = Recall@1 = Recall@5 = Recall@10
            # = 1.0.  Its coverage fields (share_queries_pool_le_max_k = 1.0,
            # trustworthy = 0) are what make that visible in the CSV.
            _ks = tuple(_lc()["evaluation"]["retrieval_ks"])
            _eval_pairs = np.vstack([test_pos, hard_test])
            _query_ids = np.asarray(
                [str(df["product_id"].iloc[int(i)]) for i in _eval_pairs[:, 0]],
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
            _retrieval_pool = build_evaluation_pool(
                test_pos,
                np.asarray(
                    [str(df["product_id"].iloc[int(i)]) for i in test_pos[:, 0]],
                    dtype=str,
                ),
                competitor_rows=_fold_canonical_rows,
                row_component=_retrieval_row_component,
                row_bc=row_bc,
                n_competitors=_n_competitors,
                seed=seed + fold_i * 1009 + 340346,
                ks=_ks,
                # the fold's mined hard negatives are seated FIRST so the
                # hardest distractors stay inside the ranking comparison
                priority_pairs=hard_test,
                excluded_barcode_pairs=_retrieval_true_match_pairs,
            )
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
            _prec90, _rec90, _thr90 = _precision_at_recall(_y, _all, _target_recall)
            _tp90 = int(((_all >= _thr90) & (_y == 1)).sum())
            _fp90 = int(((_all >= _thr90) & (_y == 0)).sum())
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
                "youden_thr_test_descriptive": _youden_thr(_all, _y),
                "acc_at_thr": _acc,
                "pr_auc": _pr_auc,
                # 07-schema: AP under the same name the plots expect
                "average_precision": _pr_auc,
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
                # train barcodes (the label=0 half of the dataset); mnrl:
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
                        df["title"].iloc[index], df["attributes"].iloc[index]
                    )
                else:
                    gtin = str(row_bc[index])
                    if gtin not in _canon_attrs:
                        raise KeyError(
                            f"payload endpoint {index} has barcode {gtin!r} "
                            "but no canonical attribute record"
                        )
                    info = _canon_attrs[gtin]
                _attribute_cache[index] = info
                return info

            def _attribute_conflicts(a: int, b: int) -> dict[str, object]:
                return conflict_columns(_attribute_info(a), _attribute_info(b))

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
                    def _sku_id(i, _n_canon=n_canon_entries):
                        if i < len(df):
                            return str(df["product_id"].iloc[i])
                        # canonical entries carry the GTIN as their barcode
                        bc_i = str(row_bc[i]) if i < len(row_bc) else ""
                        if i < len(df) + _n_canon:
                            return f"canon#{bc_i or i}"
                        return f"masked#{i}"

                    def _retailer(i):
                        return str(df["retailer"].iloc[i]) if i < len(df) else "-"

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
            rows.append({"fold": fold_i, "status": "failed", "traceback": tb})

    return rows


# ═══════════════════════════════════════════════════════════════════════════
# Optuna
# ═══════════════════════════════════════════════════════════════════════════


def run_hpo(
    args,
    data,
    mlf: MlflowCtx,
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
    import optuna
    import torch

    def objective(trial: optuna.Trial) -> float:
        cfg = {
            "architecture": _runtime("architecture"),
            "epochs": trial.suggest_int("epochs", *HPO_SPACE["epochs"]),
            "lr": trial.suggest_float("lr", *HPO_SPACE["lr"], log=True),
            "warmup_ratio": trial.suggest_float(
                "warmup_ratio", *HPO_SPACE["warmup_ratio"]
            ),
            "weight_decay": trial.suggest_float(
                "weight_decay", *HPO_SPACE["weight_decay"]
            ),
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
            "negative_mask_frac": trial.suggest_float(
                "negative_mask_frac", *HPO_SPACE["negative_mask_frac"]
            ),
            "uniformity_weight": trial.suggest_float(
                "uniformity_weight", *HPO_SPACE["uniformity_weight"]
            ),
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
        with mlf.nested:
            mlf.log_params(
                {
                    **cfg,
                    "loss": args.loss,
                    # hard-positive lane: SSOT knob (training.hard_positives);
                    # the legacy --no-hard-positives flag no longer exists
                    "hard_pos": _SSOT_HP,  # training.hard_positives SSOT
                    "band": args.band,
                }
            )
            rows = train_one_config(
                cfg,
                loss=args.loss,
                model_id=args.model,
                use_hp=_SSOT_HP,
                band=_band_tuple(args.band),
                data=data,
                seed=SEED,
                on_cuda=torch.cuda.is_available(),
                cv_folds=cv_folds,
                run_tag=f"{args.model.split('/')[-1]}_t{trial.number}",
                folds_override=folds_override,
                dev_fraction=dev_fraction,
                dev_override=dev_override,
                selection_mode=selection_mode,
                neg_pairs=neg_pairs,
                train_neg_pairs=train_neg_pairs,
                neg_pair_sources=neg_pair_sources,
                train_neg_pair_sources=train_neg_pair_sources,
                dynamic_mask_hard_negatives=dynamic_mask_hard_negatives,
                dynamic_mask_frac=cfg["negative_mask_frac"],
                dynamic_mask_prob=dynamic_mask_prob,
                dynamic_mask_lo=dynamic_mask_lo,
                dynamic_mask_hi=dynamic_mask_hi,
                mask_audit=mask_audit,
                hard_negative_mask_audit=hard_negative_mask_audit,
                wandb_ctx=wandb_ctx,
            )
            require_no_failed_folds(rows, lane=f"HPO trial {trial.number}")
            ok_rows = rows
            # Persist the actual trial evidence in Optuna. The callback below
            # mirrors these values to W&B after the trial has committed.
            _trial_loss = [r.get("final_train_loss") for r in ok_rows if np.isfinite(r.get("final_train_loss", float("nan")))]
            if _trial_loss:
                trial.set_user_attr("mean_final_train_loss", float(np.mean(_trial_loss)))
            _dev_loss_histories = []
            _train_loss_histories = []
            for row in ok_rows:
                try:
                    _dev_loss_histories.append(json.loads(row.get("dev_loss_hist", "[]")))
                    _train_loss_histories.append(json.loads(row.get("train_loss_hist", "[]")))
                except (TypeError, json.JSONDecodeError):
                    continue
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
            proxy_rows = [
                r for r in ok_rows
                if np.isfinite(r.get("calibration_rand_index", float("nan")))
            ]
            if not proxy_rows:
                raise optuna.TrialPruned(
                    "no fold produced a finite calibrated Rand Index proxy"
                )
            mean_rand = float(np.mean([r["calibration_rand_index"] for r in proxy_rows]))
            mean_penalty = float(np.mean([r["collapse_penalty"] for r in proxy_rows]))
            value = mean_rand - mean_penalty
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
            guardrail = load_config()["collapse_guardrail"]
            if collapse_medians and max(collapse_medians) > float(guardrail["reject_median"]):
                raise optuna.TrialPruned(
                    "collapse guardrail rejected trial: "
                    f"median_cosine={max(collapse_medians):.4f}"
                )
            if collapse_crossing_rates and max(collapse_crossing_rates) > float(
                guardrail["crossing_rate_ceiling"]
            ):
                raise optuna.TrialPruned(
                    "collapse guardrail rejected trial: "
                    f"crossing_rate={max(collapse_crossing_rates):.4f} "
                    f"ceiling={float(guardrail['crossing_rate_ceiling']):.4f}"
                )
            proxy_summary = {
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
                "mean_diagnostic_bridge_edge_count": float(
                    np.mean([r["diagnostic_bridge_edge_count"] for r in proxy_rows])
                ),
                "mean_attribute_conflict_error_rate": float(
                    np.nanmean([r["attribute_conflict_error_rate"] for r in proxy_rows])
                ),
            }
            trial.set_user_attr("rand_index_objective", value)
            for key, metric in proxy_summary.items():
                trial.set_user_attr(key, metric)
            mlf.log_metrics({"hpo_objective": value, **proxy_summary})
            # PostgreSQL mode promotes only from the controller after Optuna
            # commits COMPLETE and a sealed artifact snapshot is READY.
            if control_plane is None:
                retain_hpo_champion(
                    model_id=args.model,
                    run_tag=f"{args.model.split('/')[-1]}_t{trial.number}",
                    value=value,
                    folds=[int(r["fold"]) for r in proxy_rows],
                )
            return value

    sampler = optuna.samplers.TPESampler(seed=SEED)
    # sqlite storage: the sweep SURVIVES session loss — re-running with the same
    # --study resumes; every trial's params/value persist (the essential record)
    # dlr suffix: discriminative-LR trials form a NEW objective surface —
    # never mixed into the pre-dlr TPE history (its surrogate would be poisoned
    # by trials whose values came from single-LR training)
    study_name = f"second08-{args.model.split('/')[-1]}-dlr"
    study_db = RESULTS / f"{study_name}.optuna.db"
    control_plane = None
    if os.environ.get("OPTUNA_STORAGE_URL"):
        from training.hpo_control_plane import (
            create_storage,
            fail_stale_trials,
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
    if args.resume and control_plane is None:
        from training.dvc_store import restore_checkpoint

        restore_checkpoint(RESULTS, study_db)
        print(f"[resume] restored Optuna study from DVC: {study_db.name}", flush=True)
    storage = control_plane or f"sqlite:///{study_db}"
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
    if control_plane is not None:
        fail_stale_trials(study)
    # resume-safe: count prior trials, run only what remains
    prior = len(
        [t for t in study.trials if t.state.name in ("COMPLETE", "PRUNED", "FAIL")]
    )
    remaining = max(0, args.n_trials - prior)
    print(
        f"HPO: {prior} prior trials on record, running {remaining} more (n_jobs={args.n_jobs})",
        flush=True,
    )
    if remaining:
        def _persist_study(*_args) -> None:
            if os.environ.get("EUROMONITOR_DISABLE_DVC_CHECKPOINTS"):
                return
            if not os.environ.get("DVC_API_KEY"):
                return
            from training.dvc_store import publish_checkpoint

            publish_checkpoint(RESULTS, study_db)

        study.optimize(
            objective,
            n_trials=remaining,
            n_jobs=args.n_jobs,
            callbacks=[_optuna_tracking_cb(mlf, wandb_ctx), _persist_study],
        )

    completed_trials = [
        trial
        for trial in study.trials
        if trial.state.name == "COMPLETE" and trial.value is not None
    ]
    if not completed_trials:
        raise FoldExecutionError(
            "Optuna selection",
            [{"status": "no_completed_trials"}],
        )

    # every trial's params + value, on disk (optuna keeps them in the study;
    # the CSV makes the sweep's decision trail auditable without re-loading)
    trials_df = study.trials_dataframe(
        attrs=("number", "state", "value", "params", "user_attrs")
    )
    model_tag = args.model.split("/")[-1]
    era = "-dlr"  # discriminative-LR sweep era (see study_name above)
    trials_path = artifact("hpo_trials", {"model": model_tag, "era": era})
    ensure_parent(trials_path)
    trials_df.to_csv(trials_path, index=False)
    trace_artifact("hpo_trials", trials_path, producer="training.training")
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


def _optuna_mlflow_cb(mlf: MlflowCtx):
    def cb(study, trial):
        if trial.state.name == "COMPLETE" and trial.value is not None:
            mlf.log_metrics({f"trial_{trial.number}_auc": trial.value})

    return cb


def _optuna_tracking_cb(mlf: MlflowCtx, wandb_ctx):
    """Record every completed Optuna trial in local and optional remote logs."""
    def cb(study, trial):
        if trial.state.name == "COMPLETE" and trial.value is not None:
            mlf.log_metrics({f"trial_{trial.number}_objective": trial.value})
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
