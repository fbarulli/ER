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
    RESULTS,
    SEED,
    kfold_barcodes,
    load_config,
    pair_auc,
    pair_similarity,
    runtime,
)
from core.common import SSOT_CONTRASTIVE_MARGIN as _SSOT_MARGIN
from core.common import runtime as _runtime

_SSOT_HP = bool(_runtime("hard_positives"))  # no-fallback SSOT
from core.hard_negatives import mine_hard_negatives, pairs_in_set

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
N_TARGET_MINING = runtime("n_target_mining")

# DEFAULT_CFG REMOVED (audit 2026-09-09): zero readers since the entry
# (train.py) constructs its own cfg dict; a stale epochs=2 default here
# contradicted the SSOT epochs=10 and was pure dead-code risk.

# TPE search space, SSOT: config/training.yaml hpo.tpe_space (validated by
# HpoSpaceSpec at load — lo < hi per knob). The dict literal was a second
# declaration the config could not steer (audit 2026-09-09, owner Q27).
from core.common import hpo_cfg as _hpo_cfg_load

HPO_SPACE = {k: (lo, hi) for k, (lo, hi) in _hpo_cfg_load()["tpe_space"].items()}

# selection protocol (test-leak fix, 2026-09-12), SSOT: hpo.objective /
# hpo.selection_skip_test_eval (validated by HpoSpec/ObjectiveSpec at
# load). The table is PINNED per split mode — holdout sweeps select on the
# dev quarter's best_dev_ap (the test quarter's eval is skipped entirely);
# cv sweeps select on mean fold auc (fold test sides are validation folds
# there). selection_mode=True in train_one_config means the holdout rule
# is in force, so the selection row carries the holdout entry.
_HPO_OBJ_TABLE = _hpo_cfg_load()["objective"]
HPO_OBJECTIVE_HOLDOUT = _HPO_OBJ_TABLE["holdout"]
HPO_OBJECTIVE_CV = _HPO_OBJ_TABLE["cv"]
HPO_SKIP_TEST_EVAL = bool(_hpo_cfg_load()["selection_skip_test_eval"])


# ═══════════════════════════════════════════════════════════════════════════
# MLflow — SSOT src/core/mlflow_ctx (audit 2026-09-09: this module used to carry
# its own MlflowCtx duplicate with CONFLICTING semantics — "off unless
# MLFLOW_TRACKING_URI is set" — while the owner mandate (training logs
# available locally) is lib's: local sqlite by default, =off to disable.
# The duplicate shadowed the mandate; re-exported here for hpo.py's import.)
# ═══════════════════════════════════════════════════════════════════════════

from core.mlflow_ctx import MlflowCtx
from core.ranking_metrics import ranking_at_k

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


def _make_loss(model, loss: str, margin: float | None = None):
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
        return losses.OnlineContrastiveLoss(model, margin=m)
    return losses.TripletLoss(model)


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


class ProgressCallback(TrainerCallback):
    """Live per-step display of train loss + dev AP/AUC during training.

    The modern Trainer path replaces 07b's log_steps=True (which wrapped the
    loss module's forward to print every batch). This is the equivalent on the
    HF contract: on_log fires at logging_steps and carries the running train
    loss; on_evaluate fires at eval_steps and carries the dev metrics the
    early-stopper is actually watching.
    """

    def __init__(self, wandb_ctx=None):
        self.wandb_ctx = wandb_ctx
        self.latest_train_loss: float | None = None
        self.latest_dev_accuracy: float | None = None

    def _write_live_status(self, state, event: str, **values) -> None:
        """Atomically expose a compact worker heartbeat to the Colab launcher."""
        import tempfile

        payload = {
            "updated_at": time.time(),
            "event": event,
            "step": int(state.global_step),
            "max_steps": int(state.max_steps),
            "epoch": float(state.epoch or 0.0),
            "wandb_run_id": getattr(self.wandb_ctx, "run_id", None),
            "wandb_url": getattr(self.wandb_ctx, "run_url", None),
            **{key: value for key, value in values.items() if value is not None},
        }
        target = RESULTS / "live_status.json"
        # Parallel Optuna trials share a model worker's RESULTS directory.
        # A fixed ``live_status.json.tmp`` lets one thread replace (remove)
        # the other thread's temporary file before it reaches os.replace.
        # Keep the final target shared (latest heartbeat wins), but give every
        # atomic write a private same-filesystem temporary path.
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            temporary = Path(handle.name)
        os.replace(temporary, target)

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            telemetry = _runtime_telemetry()
            print(f"    [telemetry] training-started | {_format_telemetry(telemetry)}", flush=True)
            self._write_live_status(state, "training-started", **telemetry)
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not state.is_world_process_zero:
            return
        if "loss" in logs:
            loss = float(logs["loss"])
            self.latest_train_loss = loss
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
                    },
                )
            self._write_live_status(
                state,
                "train",
                train_loss=loss,
                dev_accuracy=self.latest_dev_accuracy,
                **telemetry,
            )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or not state.is_world_process_zero:
            return
        ap = metrics.get("eval_dev_cosine_ap")
        auc_key = next((k for k in metrics if k.endswith("_auc")), None)
        acc_key = next((k for k in metrics if k.endswith("_cosine_accuracy")), None)
        if acc_key is not None:
            self.latest_dev_accuracy = float(metrics[acc_key])
        telemetry = _runtime_telemetry()
        parts = [f"dev_ap {float(ap):.4f}"] if ap is not None else []
        if auc_key is not None:
            parts.append(f"dev_auc {float(metrics[auc_key]):.4f}")
        if acc_key is not None:
            parts.append(f"dev_acc {float(metrics[acc_key]):.4f}")
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
                    "live/dev_accuracy": float(metrics[acc_key])
                    if acc_key is not None else None,
                    "live/dev_average_precision": float(ap)
                    if ap is not None else None,
                    "live/dev_auc": float(metrics[auc_key])
                    if auc_key is not None else None,
                    "live/epoch": float(state.epoch or 0.0),
                },
            )
        self._write_live_status(
            state,
            "evaluation",
            train_loss=self.latest_train_loss,
            dev_loss=float(metrics["eval_loss"]) if metrics.get("eval_loss") is not None else None,
            dev_average_precision=float(ap) if ap is not None else None,
            dev_auc=float(metrics[auc_key]) if auc_key is not None else None,
            dev_accuracy=self.latest_dev_accuracy,
            **telemetry,
        )


class DvcCheckpointCallback(TrainerCallback):
    """Queue immutable Trainer checkpoints for verified DVC persistence."""

    def __init__(self):
        from concurrent.futures import ThreadPoolExecutor

        # One publisher per worker prevents concurrent DVC metadata mutations.
        # Training itself continues while this worker performs network I/O.
        self._publisher = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="checkpoint-dvc"
        )
        self._futures = []

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

    @staticmethod
    def _publish(snapshot: Path, checkpoint: Path) -> Path:
        import hashlib
        import shutil
        from training.dvc_store import publish_checkpoint

        try:
            key = hashlib.sha256(str(checkpoint.parent.resolve()).encode()).hexdigest()[:16]
            return publish_checkpoint(
                RESULTS,
                snapshot,
                # Likewise, resume pointers must not collide between two
                # trial output roots that happen to save at the same step.
                resume_name=f"{checkpoint.name}--{key}",
                restore_root=checkpoint,
            )
        finally:
            native_pointer = snapshot.with_name(f"{snapshot.name}.dvc")
            shutil.rmtree(snapshot, ignore_errors=True)
            native_pointer.unlink(missing_ok=True)
            try:
                snapshot.parent.rmdir()
            except OSError:
                pass

    def _raise_publish_errors(self) -> None:
        remaining = []
        for future in self._futures:
            if future.done():
                pointer = future.result()
                print(f"    [checkpoint-dvc] verified -> {pointer.relative_to(RESULTS)}", flush=True)
            else:
                remaining.append(future)
        self._futures = remaining

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
        self._raise_publish_errors()
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
        self._futures.append(self._publisher.submit(self._publish, snapshot, checkpoint))
        print(f"    [checkpoint-dvc] queued step {state.global_step}", flush=True)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        """Never report a successful worker exit with uploads still pending."""
        for future in self._futures:
            pointer = future.result()
            print(f"    [checkpoint-dvc] verified -> {pointer.relative_to(RESULTS)}", flush=True)
        self._futures = []
        self._publisher.shutdown(wait=True)
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
    checkpoint_base = RESULTS / "_checkpoints" / model_tag
    record = RESULTS / f"hpo_{model_tag}_champion.json"
    lock_path = RESULTS / f".hpo-{model_tag}-retention.lock"

    def artifacts(tag: str, fold_numbers: list[int]) -> list[Path]:
        paths = [RESULTS / "logs" / tag]
        paths.extend(
            checkpoint_base / f"r{tag}_f{fold}" for fold in fold_numbers
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


def _dump_train_visibility(
    fold_i,
    s1,
    s2,
    lab,
    train_all,
    tr_negs,
    *,
    hp_in_train,
    payload,
    row_bc,
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
        return "hard_neg"

    rows = []
    for k, (t1, t2, l) in enumerate(zip(s1, s2, lab)):
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
    # and is SELECTED on dev (q2, best_dev_ap), and the test quarter's
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
    from sentence_transformers import SentenceTransformer

    # BOUNDARY CONTRACT (lib.schemas.TrainConfig): the optimizer/early-stop
    # dict — every key validated (epochs >= 1, lr > 0, warmup in [0,1]...)
    # before a single fold runs. A missing/illegal knob dies HERE with the
    # field named, not inside the HF Trainer mid-epoch.
    from core.schemas import TrainConfig as _TrainConfig

    _TrainConfig.model_validate(cfg)
    if cfg["architecture"] != "two_tower":  # schema keeps this exhaustive
        raise ValueError(f"unsupported training architecture: {cfg['architecture']}")

    df, payload, row_bc, country, pos, hp_pairs, emb0 = data
    all_barcode_set = set(row_bc.tolist())

    # country must cover every payload entry (canonicals + masked copies
    # appended after the sku rows); pad with "" so the cross-country mask
    # never IndexErrors no matter which lane built the data tuple
    if len(country) < len(payload):
        pad = np.full(len(payload) - len(country), "", dtype=country.dtype)
        country = np.concatenate([country, pad])
        data = (df, payload, row_bc, country, pos, hp_pairs, emb0)

    # BOUNDARY CONTRACT (lib.schemas.DataTuple): the 7-tuple is the widest
    # crossing in the lane — payload/row_bc/country length-locked, every
    # pos/hp index in range, emb0 rows == payload. Validated ONCE per
    # train_one_config call; a shape break dies here with a named field
    # instead of an IndexError three stack frames into a fold.
    from core.schemas import DataTuple as _DataTuple

    _DataTuple(
        n_df=len(df),
        payload=payload,
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
    from core.common import band as _band_helper

    _eval_band = _band_helper("eval_mining")
    hard_train_all, _ = mine_hard_negatives(
        df, emb0, n_target=N_TARGET_MINING, cosine_lo=band[0], cosine_hi=band[1]
    )
    hard_eval, _ = mine_hard_negatives(
        df, emb0, n_target=N_TARGET_MINING,
        cosine_lo=_eval_band[0], cosine_hi=_eval_band[1],
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

            test_pos = pos[pairs_in_set(pos, row_bc, test_bc)]
            train_pos = pos[pairs_in_set(pos, row_bc, tr_bc)]
            dev_pos = pos[pairs_in_set(pos, row_bc, dev_bc)]
            hard_train = hard_train_all[pairs_in_set(hard_train_all, row_bc, tr_bc)]
            hard_dev = hard_eval[pairs_in_set(hard_eval, row_bc, dev_bc)]
            hard_test = hard_eval[pairs_in_set(hard_eval, row_bc, test_bc)]
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

            # dev evaluator needs pos/neg pairs as texts
            dev_pairs = [(payload[a], payload[b]) for a, b in dev_pos]
            dev_neg_pairs = [(payload[a], payload[b]) for a, b in hard_dev]
            if len(dev_pairs) == 0 or len(dev_neg_pairs) == 0:
                rows.append(
                    {
                        "fold": fold_i,
                        "status": "skipped",
                        "reason": "empty dev split — early stopping needs pos and neg dev pairs",
                    }
                )
                continue

            checkpoint_dir = (
                RESULTS / "_checkpoints" / model_id.rstrip("/").rsplit("/", 1)[-1]
                / f"r{run_tag}_f{fold_i}"
            )
            if resume:
                from training.dvc_store import restore_checkpoint

                restore_checkpoint(RESULTS, checkpoint_dir)
                print(f"    [resume] restored {checkpoint_dir} from DVC", flush=True)

            # Tied-weight two-tower retrieval model: the trainer receives
            # (SKU text, canonical text) pairs; each side is encoded on its
            # own before cosine/loss comparison.  CrossEncoder is optional
            # only in rerank.py after retrieval, never this default path.
            model = SentenceTransformer(model_id, device="cuda" if on_cuda else "cpu")
            _align_model_token_ids(model)
            model.max_seq_length = runtime("max_seq_length")  # SSOT, no literal

            # ── build the training dataset FIRST (steps derive from it) ──
            from datasets import Dataset

            examples = None
            if loss == "contrastive":
                # OnlineContrastiveLoss (owner ruling 2026-09-07): paired
                # (sentence1, sentence2, label) rows. POSITIVES = train_all
                # (sku, own canonical); NEGATIVES = the gate hard-no pairs —
                # text-similar, gate-proven different size/pack/flavor —
                # restricted to TRAIN barcodes (component boundary holds:
                # pairs_in_set filters by tr_bc). The loss itself then picks
                # the hard subset per batch (farthest positives, closest
                # negatives) — hard-pair training at both layers.
                tr_negs = (
                    neg_pairs[pairs_in_set(neg_pairs, row_bc, tr_bc)]
                    if neg_pairs is not None and len(neg_pairs)
                    else np.empty((0, 2), dtype=int)
                )
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
                train_ds = Dataset.from_dict(
                    {"sentence1": s1, "sentence2": s2, "label": lab}
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
                    hp_in_train=(
                        hp_pairs[pairs_in_set(hp_pairs, row_bc, tr_bc)]
                        if use_hp and hp_pairs is not None and len(hp_pairs)
                        else None
                    ),
                    payload=payload,
                    row_bc=row_bc,
                    run_tag=run_tag,
                    sample=sample,
                )
            elif loss == "mnrl":
                # 3rd column when caller passes neg_pairs: ST's MNRL treats
                # extra columns as explicit in-batch negatives per anchor —
                # the gate hard-negatives ("similar text, different size/
                # pack/flavor") enter training here. Rows are paired with a
                # shuffled neg pool (dedup guard: skip negs that ARE the
                # positive text).
                neg_texts: list[str] | None = None
                if neg_pairs is not None and len(neg_pairs):
                    tr_negs = neg_pairs[pairs_in_set(neg_pairs, row_bc, tr_bc)]
                    if len(tr_negs):
                        rng_n = np.random.default_rng(seed + fold_i + 1)
                        pool = [payload[a] for a, b in tr_negs] + [
                            payload[b] for a, b in tr_negs
                        ]
                        neg_texts = [
                            pool[k % len(pool)]
                            for k in rng_n.permutation(len(train_all))[: len(train_all)]
                        ]
                train_ds = Dataset.from_dict(
                    {
                        "anchor": [payload[a] for a, b in train_all],
                        "positive": [payload[b] for a, b in train_all],
                        **({"negative": neg_texts} if neg_texts else {}),
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

            sentences1 = [a for a, _ in dev_pairs] + [a for a, _ in dev_neg_pairs]
            sentences2 = [b for _, b in dev_pairs] + [b for _, b in dev_neg_pairs]
            labels = [1] * len(dev_pairs) + [0] * len(dev_neg_pairs)
            evaluator = BinaryClassificationEvaluator(
                sentences1, sentences2, labels, name="dev", show_progress_bar=False
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
            from sentence_transformers import (
                SentenceTransformerTrainingArguments as STArgs,
            )

            class ResumableSentenceTransformerTrainer(SentenceTransformerTrainer):
                """HF Trainer plus an explicit manifest of all resume state."""

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

            trainer = ResumableSentenceTransformerTrainer(
                model=model,
                args=args_hf,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                evaluator=evaluator,
                loss=_make_loss(model, loss),
                optimizers=(optimizer, None),  # prebuilt AdamW with
                # discriminative LRs; scheduler=None -> HF builds warmup+linear
                # from args, scaling our per-group LRs
                callbacks=[
                    ProgressCallback(wandb_ctx),
                    DvcCheckpointCallback(),
                    EarlyStoppingCallback(
                        early_stopping_patience=cfg["patience"],
                        early_stopping_threshold=cfg["es_threshold"],
                    ),
                ],
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
            final_train_loss = train_losses[-1] if train_losses else float("nan")
            best_dev_ap = max(dev_aps) if dev_aps else float("nan")

            # ── SELECTION-MODE EXIT (test-leak fix, 2026-09-12) ───────────
            # Holdout HPO/grid folds STOP HERE: the config is ranked on
            # best_dev_ap and the test quarter's eval block is never
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
                        "status": "ok",
                        "test_eval": "skipped_selection_mode",
                        "objective": HPO_OBJECTIVE_HOLDOUT,
                        "best_dev_ap": best_dev_ap,
                        "final_train_loss": final_train_loss,
                        "train_loss_hist": json.dumps(
                            [round(x, 4) for x in train_losses]
                        ),
                        "dev_ap_hist": json.dumps([round(x, 4) for x in dev_aps]),
                        "dev_loss_hist": json.dumps(
                            [round(x, 4) for x in dev_losses]
                        ),
                        "n_dev_pos": len(dev_pos),
                        "n_dev_neg": len(hard_dev),
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
                    }
                )
                continue

            # eval on test (timed: encode latency is a first-class metric)
            t_encode = time.perf_counter()
            eval_rows = np.unique(np.r_[test_pos.ravel(), hard_test.ravel()])
            row_to_idx = {int(r): i for i, r in enumerate(eval_rows)}
            tp_idx = np.array([row_to_idx[int(r)] for r in test_pos.ravel()]).reshape(
                -1, 2
            )
            hn_idx = np.array([row_to_idx[int(r)] for r in hard_test.ravel()]).reshape(
                -1, 2
            )
            eval_payload = [payload[r] for r in eval_rows]
            emb = model.encode(
                eval_payload,
                batch_size=runtime("batch_size_eval"),
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            encode_s = time.perf_counter() - t_encode
            pos_s = _cos(emb, tp_idx)
            neg_s = _cos(emb, hn_idx)
            cross_mask = country[test_pos[:, 0]] != country[test_pos[:, 1]]

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
            dev_payload = [payload[r] for r in dev_rows]
            dev_emb = model.encode(
                dev_payload,
                batch_size=runtime("batch_size_eval"),
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            dev_pos_s = _cos(dev_emb, dev_tp_idx)
            dev_neg_s = _cos(dev_emb, dev_hn_idx)

            # ── latency metrics ──────────────────────────────────────────
            train_s = time.perf_counter() - t_fold - encode_s
            steps_run = trainer.state.global_step
            s_per_step = train_s / steps_run if steps_run else float("nan")
            texts_per_s = len(eval_payload) / encode_s if encode_s else float("nan")
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
            _ranking = ranking_at_k(
                _y, _all, tuple(_lc()["evaluation"]["retrieval_ks"])
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
            _prec90, _rec90, _thr90 = _precision_at_recall(_y, _all, 0.90)
            _tp90 = int(((_all >= _thr90) & (_y == 1)).sum())
            _fp90 = int(((_all >= _thr90) & (_y == 0)).sum())

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
                **_ranking,
                "precision_at_90pct_recall": _prec90,
                "tp_at_90pct_recall": _tp90,
                "fp_at_90pct_recall": _fp90,
                "threshold_at_90pct_recall": _thr90,
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
                "dev_ap_hist": json.dumps([round(x, 4) for x in dev_aps]),
                "dev_loss_hist": json.dumps([round(x, 4) for x in dev_losses]),
                # ── pair accounting (failure-analysis ground) ────────────
                "n_pos": len(test_pos),
                "n_neg": len(hard_test),
                "n_train_pos": len(train_pos),
                # hp rows in train = total minus the GATE rows actually kept.
                # Subtracting the UNSAMPLED train_pos went NEGATIVE under
                # train_frac<1 (measured -426 on the frac0.25 run).
                "n_train_hp": int(len(train_all) - n_gate_kept),
                # contrastive: labeled negatives = gate hard-no pairs in
                # train barcodes (the label=0 half of the dataset); mnrl:
                # in-batch only (counted separately below); triplet: mined
                "n_train_neg": (
                    len(
                            neg_pairs[
                                pairs_in_set(neg_pairs, row_bc, tr_bc)
                            ]
                        )
                    if loss == "contrastive"
                    and neg_pairs is not None
                    and len(neg_pairs)
                    else (0 if loss == "mnrl" else len(hard_train))
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
            "lr_scheduler": _runtime("lr_scheduler"),  # SSOT
            "max_grad_norm": _runtime("max_grad_norm"),  # SSOT
            "patience": ES_PATIENCE,
            "es_threshold": ES_THRESHOLD,
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
                wandb_ctx=wandb_ctx,
            )
            ok_rows = [r for r in rows if r.get("status") == "ok"]
            if not ok_rows:
                raise optuna.TrialPruned("no fold completed")
            # Persist the actual trial evidence in Optuna. The callback below
            # mirrors these values to W&B after the trial has committed.
            _trial_loss = [r.get("final_train_loss") for r in ok_rows if np.isfinite(r.get("final_train_loss", float("nan")))]
            if _trial_loss:
                trial.set_user_attr("mean_final_train_loss", float(np.mean(_trial_loss)))
            if selection_mode:
                # HOLDOUT RULE (test-leak fix, 2026-09-12): rank the trial
                # on the dev quarter's best_dev_ap ONLY. Selection-mode
                # folds carry NO test metric (test_eval=
                # skipped_selection_mode) — the test quarter is read
                # exactly once, by the main train lane, so there is no
                # per-trial test number to select on even by accident.
                dev_aps = [
                    r["best_dev_ap"]
                    for r in ok_rows
                    if np.isfinite(r.get("best_dev_ap", float("nan")))
                ]
                if not dev_aps:
                    raise optuna.TrialPruned("no fold produced a finite dev AP")
                value = float(np.mean(dev_aps))
                retain_hpo_champion(
                    model_id=args.model,
                    run_tag=f"{args.model.split('/')[-1]}_t{trial.number}",
                    value=value,
                    folds=[int(r["fold"]) for r in ok_rows],
                )
                trial.set_user_attr("dev_selection_ap", value)
                mlf.log_metrics(
                    {
                        "mean_dev_ap": value,
                        **{
                            f"fold_{r['fold']}_best_dev_ap": r["best_dev_ap"]
                            for r in ok_rows
                        },
                    }
                )
                return value
            # CV RULE: fold test sides are validation folds — mean fold auc
            # is the legitimate selection signal there (HPO_OBJECTIVE_CV).
            aucs = [
                r["auc"]
                for r in rows
                if r.get("status") == "ok" and np.isfinite(r.get("auc", float("nan")))
            ]
            if not aucs:
                raise optuna.TrialPruned("no fold produced a finite AUC")
            mean_auc = float(np.mean(aucs))
            retain_hpo_champion(
                model_id=args.model,
                run_tag=f"{args.model.split('/')[-1]}_t{trial.number}",
                value=mean_auc,
                folds=[int(r["fold"]) for r in ok_rows],
            )
            trial.set_user_attr("mean_auc", mean_auc)
            mlf.log_metrics(
                {
                    "mean_auc": mean_auc,
                    **{
                        f"fold_{r['fold']}_auc": r["auc"]
                        for r in rows
                        if r.get("status") == "ok"
                    },
                }
            )
            return mean_auc

    sampler = optuna.samplers.TPESampler(seed=SEED)
    # sqlite storage: the sweep SURVIVES session loss — re-running with the same
    # --study resumes; every trial's params/value persist (the essential record)
    # dlr suffix: discriminative-LR trials form a NEW objective surface —
    # never mixed into the pre-dlr TPE history (its surrogate would be poisoned
    # by trials whose values came from single-LR training)
    study_name = f"second08-{args.model.split('/')[-1]}-dlr"
    study_db = RESULTS / f"{study_name}.optuna.db"
    if args.resume:
        from training.dvc_store import restore_checkpoint

        restore_checkpoint(RESULTS, study_db)
        print(f"[resume] restored Optuna study from DVC: {study_db.name}", flush=True)
    storage = f"sqlite:///{study_db}"
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
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

    # every trial's params + value, on disk (optuna keeps them in the study;
    # the CSV makes the sweep's decision trail auditable without re-loading)
    trials_df = study.trials_dataframe(
        attrs=("number", "state", "value", "params", "user_attrs")
    )
    model_tag = args.model.split("/")[-1]
    era = "-dlr"  # discriminative-LR sweep era (see study_name above)
    trials_df.to_csv(RESULTS / f"train_{model_tag}{era}_hpo_trials.csv", index=False)
    best = {
        "config": study.best_params,
        "value": study.best_value,
        "n_trials": len(study.trials),
        "model": args.model,
        "objective": f"discriminative-LR ({_runtime('layer_decay')}^k per-layer groups)",
        # which signal ranked the trials (test-leak fix, 2026-09-12):
        # best_dev_ap in holdout selection mode, mean fold auc in cv
        "selection": (
            HPO_OBJECTIVE_HOLDOUT if selection_mode else HPO_OBJECTIVE_CV
        ),
    }
    out_path = RESULTS / f"train_{model_tag}{era}_hpo_best.json"
    with open(out_path, "w") as f:
        json.dump(best, f, indent=2)
    if wandb_ctx is not None:
        trials_path = RESULTS / f"train_{model_tag}{era}_hpo_trials.csv"
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
                        "hpo_objective": trial.value,
                        **{f"hpo_{k}": v for k, v in trial.params.items()},
                        **{f"hpo_{k}": v for k, v in trial.user_attrs.items()},
                    },
                    step=trial.number,
                )
    return cb


def _band_tuple(band: str) -> tuple[float, float]:
    lo, hi = (float(x) for x in band.split("-"))
    return lo, hi
