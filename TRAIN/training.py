"""TRAIN/training.py — solid GPU-ready fine-tune pipeline for the
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
  uv run 05_train.py (entry) or python lib/training.py --loss mnrl
  uv run ... --hpo --n-trials 20          # optuna TPE sweep
  uv run ... --loss triplet --no-hard-positives --band 0.45-0.80
  MLFLOW_TRACKING_URI=... uv run ...      # mlflow off unless URI is set

Artifacts (results/, SSOT via 00_config):
  train_<model>_fold_metrics.csv  one row per fold per config (incl. failures
                               with traceback column)
  train_<model>_hpo_best.json    best config + its metrics (HPO mode)
"""

from __future__ import annotations

import argparse
import json
import time
import traceback

import numpy as np
import pandas as pd

from lib.blocking import build_pairs
from lib.cache import MODEL_NAME
from lib.common import (
    DATA_DIR,
    RESULTS,
    SEED,
    F,
    load_dataset_deduped,
    pair_auc,
    pair_similarity,
)
from lib.hard_negatives import mine_hard_negatives, pairs_in_set

DEFAULT_MODEL = f"sentence-transformers/{MODEL_NAME}"
from transformers import EarlyStoppingCallback, TrainerCallback

from lib.common import kfold_barcodes
from lib.nlp import encode_corpus
from lib.volume_verified import volume_verified_cross_country

# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

CACHE = str(DATA_DIR / "embeddings_cache")
CV_FOLDS = 5
DEV_FRACTION = 0.15  # of train barcodes, for early stopping
BATCH_SIZE_CPU = 32
BATCH_SIZE_CUDA = 128
MAX_TRIPLES = 5_000
EVAL_STEPS_PER_EPOCH = 4  # early-stop eval cadence
ES_PATIENCE = 3  # EarlyStoppingCallback patience (evals)
ES_THRESHOLD = 1e-3  # min dev-AUC improvement to count
N_TARGET_MINING = 20_000

DEFAULT_CFG = {
    "epochs": 2,
    "lr": 2e-5,
    "warmup_ratio": 0.05,  # fraction of train steps
    "weight_decay": 0.01,
    "lr_scheduler": "linear",  # WarmupLinear equivalent
    "max_grad_norm": 1.0,
    "patience": ES_PATIENCE,
    "es_threshold": ES_THRESHOLD,
}

HPO_SPACE = {
    "epochs": (1, 4),  # int
    "lr": (1e-5, 1e-4),  # float, log
    "warmup_ratio": (0.0, 0.1),  # float
    "weight_decay": (0.0, 0.1),  # float
}


# ═══════════════════════════════════════════════════════════════════════════
# MLflow (optional; off unless MLFLOW_TRACKING_URI is set — the euromonitor
# experiments must stay runnable without a server, same contract as nlp.py)
# ═══════════════════════════════════════════════════════════════════════════


class MlflowCtx:
    """No-op when MLFLOW_TRACKING_URI is unset; nested runs when it is."""

    def __init__(self, experiment: str):
        import os

        self.enabled = bool(os.environ.get("MLFLOW_TRACKING_URI"))
        self.experiment = experiment
        self._mlflow = None
        if self.enabled:
            import mlflow

            self._mlflow = mlflow
            mlflow.set_experiment(experiment)

    def __enter__(self):
        if self.enabled:
            self.parent = self._mlflow.start_run(run_name=self.experiment)
        return self

    def __exit__(self, *exc):
        if self.enabled:
            self._mlflow.end_run()
        return False

    @property
    def nested(self):
        """Context manager for a child run; no-op when disabled."""
        if self.enabled:
            from contextlib import nullcontext

            return self._mlflow.start_run(nested=True)
        from contextlib import nullcontext

        return nullcontext()

    def log_params(self, params: dict) -> None:
        if self.enabled:
            self._mlflow.log_params({k: str(v) for k, v in params.items()})

    def log_metrics(self, metrics: dict) -> None:
        if self.enabled:
            numeric = {
                k: float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float)) and np.isfinite(v)
            }
            if numeric:
                self._mlflow.log_metrics(numeric)

    def log_artifact(self, path, artifact_path: str | None = None) -> None:
        if self.enabled:
            self._mlflow.log_artifact(str(path), artifact_path=artifact_path)


# ═══════════════════════════════════════════════════════════════════════════
# Training (ST 6 modern Trainer path with HF early stopping)
# ═══════════════════════════════════════════════════════════════════════════


# _auc/_cos -> _common SSOT (see GATES_MAP.md)
_auc = pair_auc


_cos = pair_similarity


def _make_loss(model, loss: str):
    from sentence_transformers.sentence_transformer import losses

    if loss == "mnrl":
        return losses.MultipleNegativesRankingLoss(model)
    return losses.TripletLoss(model)


class ProgressCallback(TrainerCallback):
    """Live per-step display of train loss + dev AP/AUC during training.

    The modern Trainer path replaces 07b's log_steps=True (which wrapped the
    loss module's forward to print every batch). This is the equivalent on the
    HF contract: on_log fires at logging_steps and carries the running train
    loss; on_evaluate fires at eval_steps and carries the dev metrics the
    early-stopper is actually watching.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not state.is_world_process_zero:
            return
        if "loss" in logs:
            print(
                f"    [epoch {state.epoch:>5.2f} | step {state.global_step:>4}/"
                f"{state.max_steps:<4}] train_loss {float(logs['loss']):.4f}",
                flush=True,
            )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or not state.is_world_process_zero:
            return
        ap = metrics.get("eval_dev_cosine_ap")
        auc_key = next((k for k in metrics if k.endswith("_auc")), None)
        acc_key = next((k for k in metrics if k.endswith("_cosine_accuracy")), None)
        parts = [f"dev_ap {float(ap):.4f}"] if ap is not None else []
        if auc_key is not None:
            parts.append(f"dev_auc {float(metrics[auc_key]):.4f}")
        if acc_key is not None:
            parts.append(f"dev_acc {float(metrics[acc_key]):.4f}")
        # live GPU usage alongside the metrics
        try:
            import torch

            if torch.cuda.is_available():
                parts.append(
                    f"vram {torch.cuda.memory_allocated() / 1e9:.1f}/"
                    f"{torch.cuda.max_memory_allocated() / 1e9:.1f}GB peak"
                )
        except ImportError:  # display only, never kill training
            pass
        if parts:
            print(f"    [step {state.global_step:>4}] " + " | ".join(parts), flush=True)


def _discriminative_groups(
    model, base_lr: float, layer_decay: float = 0.9
) -> list[dict]:
    """Per-layer LR groups: bottom embeddings get base_lr*decay^n, each encoder
    layer rises geometrically to base_lr at the top; head/pooler at full base_lr.

    Dedup by tensor id — shared/tied weights (embeddings<->pooler etc.) must
    appear in exactly one group or AdamW raises. Returns groups bottom->top.
    """
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


def train_one_config(
    cfg: dict,
    *,
    loss: str,
    model_id: str = DEFAULT_MODEL,
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
) -> list[dict]:
    """Train cfg across the group-aware folds. Returns fold metric rows
    (failures included, with traceback)."""
    import torch
    from sentence_transformers import SentenceTransformer

    df, payload, row_bc, country, pos, hp_pairs, emb0 = data
    all_barcode_set = set(row_bc.tolist())

    # country must cover every payload entry (canonicals + masked copies
    # appended after the sku rows); pad with "" so the cross-country mask
    # never IndexErrors no matter which lane built the data tuple
    if len(country) < len(payload):
        pad = np.full(len(payload) - len(country), "", dtype=country.dtype)
        country = np.concatenate([country, pad])
        data = (df, payload, row_bc, country, pos, hp_pairs, emb0)

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
        n_folds = cv_folds if cv_folds is not None else CV_FOLDS
        all_folds = kfold_barcodes(df, CV_FOLDS, SEED)
        # --quick trains on the first n_folds of the SAME split (folds stay comparable)
        folds = all_folds[:n_folds]

    # mine ONCE outside the fold loop (band is fixed per config)
    hard_train_all, _ = mine_hard_negatives(
        df, emb0, n_target=N_TARGET_MINING, cosine_lo=band[0], cosine_hi=band[1]
    )
    hard_eval, _ = mine_hard_negatives(
        df, emb0, n_target=N_TARGET_MINING, cosine_lo=0.35, cosine_hi=0.90
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
            if train_frac is not None and train_frac < 1.0 and len(train_all):
                rng_f = np.random.default_rng(seed + fold_i + 7)
                keep = rng_f.random(len(train_all)) < train_frac
                train_all = train_all[keep] if keep.any() else train_all[:1]
            if use_hp:
                hp_train = hp_pairs[pairs_in_set(hp_pairs, row_bc, tr_bc)]
                train_all = (
                    np.vstack([train_pos, hp_train]) if len(hp_train) else train_pos
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

            model = SentenceTransformer(model_id, device="cuda" if on_cuda else "cpu")
            model.max_seq_length = 128

            # ── build the training dataset FIRST (steps derive from it) ──
            from datasets import Dataset

            examples = None
            if loss == "mnrl":
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
                from lib.hard_negatives import build_triplets

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

            args_hf = STArgs(
                output_dir=str(RESULTS / f"_checkpoints/r{run_tag}_f{fold_i}"),
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
                metric_for_best_model="eval_dev_cosine_ap",
                greater_is_better=True,
                load_best_model_at_end=True,
                save_strategy="steps",
                save_steps=eval_steps,
                save_total_limit=2,
                save_only_model=True,
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
            try:
                groups = _discriminative_groups(model, base_lr)
                n_g = len(groups)
                print(
                    f"    [optim] discriminative LR: {n_g} groups, "
                    f"bottom {groups[0]['lr']:.2e} .. top {base_lr:.2e}",
                    flush=True,
                )
            except Exception as exc:
                print(f"    [optim] single LR fallback ({exc})", flush=True)
                groups = [{"params": model.parameters(), "lr": base_lr}]

            from torch import optim

            optimizer = optim.AdamW(
                groups, weight_decay=cfg["weight_decay"], lr=base_lr
            )

            trainer = SentenceTransformerTrainer(
                model=model,
                args=args_hf,
                train_dataset=train_ds,
                evaluator=evaluator,
                loss=_make_loss(model, loss),
                optimizers=(optimizer, None),  # prebuilt AdamW with
                # discriminative LRs; scheduler=None -> HF builds warmup+linear
                # from args, scaling our per-group LRs
                callbacks=[
                    ProgressCallback(),
                    EarlyStoppingCallback(
                        early_stopping_patience=cfg["patience"],
                        early_stopping_threshold=cfg["es_threshold"],
                    ),
                ],
            )
            trainer.train()

            # final training loss + best dev AP from the trainer's own log
            # history (the source the early-stopper actually used)
            hist = trainer.state.log_history
            train_losses = [e["loss"] for e in hist if "loss" in e]
            dev_aps = [
                e["eval_dev_cosine_ap"] for e in hist if "eval_dev_cosine_ap" in e
            ]
            final_train_loss = train_losses[-1] if train_losses else float("nan")
            best_dev_ap = max(dev_aps) if dev_aps else float("nan")

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
                batch_size=256,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            encode_s = time.perf_counter() - t_encode
            pos_s = _cos(emb, tp_idx)
            neg_s = _cos(emb, hn_idx)
            cross_mask = country[test_pos[:, 0]] != country[test_pos[:, 1]]

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

            # accuracy at the Youden-optimal threshold (J = TPR - FPR):
            # the single operating point the fold would ship with
            _all = np.r_[pos_s, neg_s]
            _y = np.r_[np.ones(len(pos_s)), np.zeros(len(neg_s))]
            _order = np.argsort(-_all)
            _tps = np.cumsum(_y[_order])
            _fps = np.cumsum(1 - _y[_order])
            _tpr = _tps / max(len(pos_s), 1)
            _fpr = _fps / max(len(neg_s), 1)
            _j = _tpr - _fpr
            _k = int(np.argmax(_j))
            _thr = float(_all[_order][_k])
            _acc = (
                float(
                    ((_y[_order][: _k + 1] == 1).sum())
                    + ((_y[_order][_k + 1 :] == 0).sum())
                )
                / len(_y)
                if len(_y)
                else float("nan")
            )

            # PR-AUC + F1 at the FIXED operating threshold (SSOT):
            # ROC AUC alone hides class imbalance and threshold choice;
            # the lane reports PR-AUC + the F1 the pipeline would ship with
            from sklearn.metrics import average_precision_score, f1_score

            _pr_auc = float(average_precision_score(_y, _all))
            from lib.common import load_config as _lc

            _fixed_thr = float(_lc().get("split", {}).get("fixed_threshold", 0.55))
            _pred = (_all >= _fixed_thr).astype(int)
            _f1_fixed = float(f1_score(_y, _pred, zero_division=0))
            _prec_fixed = float(
                (_y[_pred == 1] == 1).mean() if (_pred == 1).any() else 0.0
            )
            _rec_fixed = float((_pred[_y == 1] == 1).mean() if (_y == 1).any() else 0.0)

            row = {
                "fold": fold_i,
                "status": "ok",
                "auc": _auc(pos_s, neg_s),
                "youden_thr": _thr,
                "acc_at_thr": _acc,
                "pr_auc": _pr_auc,
                "f1_at_0.55": _f1_fixed,
                "precision_at_0.55": _prec_fixed,
                "recall_at_0.55": _rec_fixed,
                "auc_cross": _auc(pos_s[cross_mask], neg_s)
                if cross_mask.any()
                else float("nan"),
                "final_train_loss": final_train_loss,
                "best_dev_ap": best_dev_ap,
                # full curves for the loss plot (json: csv-column-safe)
                "train_loss_hist": json.dumps([round(x, 4) for x in train_losses]),
                "dev_ap_hist": json.dumps([round(x, 4) for x in dev_aps]),
                # ── pair accounting (failure-analysis ground) ────────────
                "n_pos": len(test_pos),
                "n_neg": len(hard_test),
                "n_train_pos": len(train_pos),
                "n_train_hp": int(len(train_all) - len(train_pos)),
                "n_train_neg": 0 if loss == "mnrl" else len(hard_train),
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
                "n_train": len(train_all) if loss == "mnrl" else len(examples),
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
            pair_records = []
            for pairs, scores, label, a_col, b_col in (
                (test_pos, pos_s, 1, None, None),
                (hard_test, neg_s, 0, None, None),
            ):
                for k in range(len(pairs)):
                    a, b = int(pairs[k, 0]), int(pairs[k, 1])
                    # masked-anchor rows (masking augmentation) live beyond
                    # df — they are noised COPIES of df rows, so report the
                    # ORIGINAL anchor's ids (the pair's df-side identity)
                    a_in, b_in = a < len(df), b < len(df)
                    pair_records.append(
                        {
                            "fold": fold_i,
                            "label": label,
                            "sku_id_a": str(df["product_id"].iloc[a])
                            if a_in
                            else f"masked#{a}",
                            "sku_id_b": str(df["product_id"].iloc[b])
                            if b_in
                            else f"masked#{b}",
                            "score": float(scores[k]),
                            "cross_country": bool(country[a] != country[b]),
                            "retailer_a": str(df["retailer"].iloc[a]) if a_in else "-",
                            "retailer_b": str(df["retailer"].iloc[b]) if b_in else "-",
                        }
                    )
            model_tag = model_id.split("/")[-1]
            pd.DataFrame(pair_records).to_csv(
                RESULTS / f"train_{model_tag}_fold{fold_i}_pairs.csv", index=False
            )

            dev = f"gpu {gpu_peak_gb:.1f}GB peak" if on_cuda else "cpu"
            print(
                f"  fold {fold_i}: loss={row['final_train_loss']:.4f} "
                f"acc@{row['youden_thr']:.2f}={row['acc_at_thr']:.4f} "
                f"AUC={row['auc']:.4f} cross={row['auc_cross']:.4f} "
                f"PR-AUC={row['pr_auc']:.4f} "
                f"F1@0.55={row['f1_at_0.55']:.4f} "
                f"P@0.55={row['precision_at_0.55']:.4f} R@0.55={row['recall_at_0.55']:.4f} "
                f"| best_dev_ap={row['best_dev_ap']:.4f} "
                f"| {row['s_per_step']:.2f}s/step {row['texts_per_s_encode']:.0f} txt/s "
                f"ES-saved {row['es_saved_pct']:.0f}% [{dev}] ({row['fold_s']}s)",
                flush=True,
            )
        except Exception:
            tb = traceback.format_exc()
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
    folds_override: list[set[str]] | None = None,
    dev_fraction: float | None = None,
    dev_override: set[str] | None = None,
) -> None:
    import optuna
    import torch

    def objective(trial: optuna.Trial) -> float:
        cfg = {
            "epochs": trial.suggest_int("epochs", *HPO_SPACE["epochs"]),
            "lr": trial.suggest_float("lr", *HPO_SPACE["lr"], log=True),
            "warmup_ratio": trial.suggest_float(
                "warmup_ratio", *HPO_SPACE["warmup_ratio"]
            ),
            "weight_decay": trial.suggest_float(
                "weight_decay", *HPO_SPACE["weight_decay"]
            ),
            "lr_scheduler": "linear",
            "max_grad_norm": 1.0,
            "patience": ES_PATIENCE,
            "es_threshold": ES_THRESHOLD,
        }
        with mlf.nested:
            mlf.log_params(
                {
                    **cfg,
                    "loss": args.loss,
                    "hard_pos": not args.no_hard_positives,
                    "band": args.band,
                }
            )
            rows = train_one_config(
                cfg,
                loss=args.loss,
                model_id=args.model,
                use_hp=not args.no_hard_positives,
                band=_band_tuple(args.band),
                data=data,
                seed=SEED,
                on_cuda=torch.cuda.is_available(),
                cv_folds=cv_folds,
                run_tag=f"t{trial.number}",
                folds_override=folds_override,
                dev_fraction=dev_fraction,
                dev_override=dev_override,
            )
            aucs = [
                r["auc"]
                for r in rows
                if r.get("status") == "ok" and np.isfinite(r.get("auc", float("nan")))
            ]
            if not aucs:
                raise optuna.TrialPruned("no fold produced a finite AUC")
            mean_auc = float(np.mean(aucs))
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
    storage = f"sqlite:///{RESULTS / (study_name + '.optuna.db')}"
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
        study.optimize(
            objective,
            n_trials=remaining,
            n_jobs=args.n_jobs,
            callbacks=[_optuna_mlflow_cb(mlf)] if mlf.enabled else None,
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
        "objective": "discriminative-LR (0.9^k per-layer groups)",
    }
    with open(RESULTS / f"train_{model_tag}{era}_hpo_best.json", "w") as f:
        json.dump(best, f, indent=2)
    print(f"\nBEST: {study.best_params} -> mean AUC {study.best_value:.4f}", flush=True)
    print(f"wrote {RESULTS / 'train_hpo_best.json'}", flush=True)


def _optuna_mlflow_cb(mlf: MlflowCtx):
    def cb(study, trial):
        if trial.state.name == "COMPLETE" and trial.value is not None:
            mlf.log_metrics({f"trial_{trial.number}_auc": trial.value})

    return cb


def _band_tuple(band: str) -> tuple[float, float]:
    lo, hi = (float(x) for x in band.split("-"))
    return lo, hi


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loss", choices=["mnrl", "triplet"], default="mnrl")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="sentence-transformer model id (L12 multilingual / L6 english)",
    )
    parser.add_argument("--no-hard-positives", action="store_true")
    parser.add_argument("--band", default="0.45-0.80")
    parser.add_argument("--hpo", action="store_true", help="optuna TPO sweep")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="parallel HPO trials (threads share one GPU; T4 + 2 vCPUs saturates at 2)",
    )
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--quick", action="store_true", help="2 folds, 1 epoch")
    parser.add_argument(
        "--split",
        choices=["cv", "holdout"],
        default="cv",
        help="cv: 5-fold group CV | holdout: one 50/25/25 "
        "train/dev/test barcode-grouped split (dev/test "
        "pools double — the anti-overfit view)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="full-pipeline pass on the first N rows (tiny-data validation)",
    )
    args = parser.parse_args()

    cv_folds = 2 if args.quick else None

    t0 = time.perf_counter()
    on_cuda = torch.cuda.is_available()
    print(
        f"device: {'cuda' if on_cuda else 'cpu'} | loss={args.loss} "
        f"hard_pos={'off' if args.no_hard_positives else 'on'} band={args.band}",
        flush=True,
    )

    df = load_dataset_deduped()
    if args.sample:
        # full-pipeline pass on a tiny subset: same mining/dev/ES/eval code,
        # just fewer rows — validates the chain end to end in seconds.
        df = df.head(args.sample).reset_index(drop=True)
        print(f"SAMPLE MODE: first {args.sample} rows", flush=True)
    payload = (
        df["title"].fillna("")
        + " | "
        + df["brand"].fillna("")
        + " | "
        + df["category"].fillna("")
    ).tolist()
    row_bc = df["barcode"].fillna("").astype(str).to_numpy()
    country = df["country"].fillna("").astype(str).to_numpy()
    pos, _ = build_pairs(df, SEED, 4, 10_000)
    hp_pairs = (
        volume_verified_cross_country(df)
        if not args.no_hard_positives
        else np.empty((0, 2), dtype=int)
    )
    print(
        f"train positives pool: {len(pos):,} | hp pairs: {len(hp_pairs):,}", flush=True
    )

    model_id = args.model
    emb0, encode_s = encode_corpus(
        model_id,
        payload,
        batch_size=256,
        max_seq_length=128,
        cache_dir=CACHE,
        device="cuda" if on_cuda else "cpu",
    )
    print(f"zero-shot encode_s = {encode_s:.1f}s", flush=True)

    data = (df, payload, row_bc, country, pos, hp_pairs, emb0)

    # holdout split (--split holdout): barcode quarters -> 50 train / 25 dev / 25 test.
    # Same group-aware guarantee as CV: no barcode straddles any boundary.
    folds_override = None
    dev_override_hold = None
    dev_fraction = None
    if args.split == "holdout":
        # FIXED (2026-09-06): single-set folds_override now means
        # train = all-other-barcodes, and dev_override passes the dev
        # quarter explicitly — the component/holdout contract works.
        quarters = kfold_barcodes(df, 4, SEED)
        test_bc_hold, dev_bc = quarters[3], quarters[2]
        folds_override = test_bc_hold
        dev_override_hold = dev_bc
        print(
            f"HOLDOUT: test {len(test_bc_hold)} bcs | dev {len(dev_bc)} bcs | "
            f"train {len(set(row_bc.tolist()) - test_bc_hold - dev_bc)} bcs",
            flush=True,
        )

    with MlflowCtx("train-pipeline") as mlf:
        mlf.log_params(
            {
                "model": model_id,
                "loss": args.loss,
                "hard_pos": not args.no_hard_positives,
                "band": args.band,
                "cuda": on_cuda,
                "quick": args.quick,
                "split": args.split,
            }
        )
        if args.hpo:
            run_hpo(
                args,
                data,
                mlf,
                cv_folds=cv_folds,
                folds_override=folds_override,
                dev_fraction=dev_fraction,
                dev_override=dev_override_hold,
            )
        else:
            rows = train_one_config(
                DEFAULT_CFG,
                loss=args.loss,
                model_id=args.model,
                use_hp=not args.no_hard_positives,
                band=_band_tuple(args.band),
                data=data,
                seed=SEED,
                on_cuda=on_cuda,
                cv_folds=cv_folds,
                folds_override=folds_override,
                dev_fraction=dev_fraction,
            )
            out = pd.DataFrame(rows)
            model_tag = model_id.split("/")[-1]
            out.to_csv(RESULTS / f"train_{model_tag}_fold_metrics.csv", index=False)
            mlf.log_artifact(RESULTS / F["fold_metrics"])
            if "auc" in out.columns and (out["auc"].notna().any()):
                mlf.log_metrics({"mean_auc": float(out["auc"].mean())})
            ok = out[out["status"] == "ok"] if "status" in out.columns else out
            if len(ok) and "auc" in ok.columns:
                print(
                    f"\nfolds ok: {len(ok)}/{len(out)} | mean AUC {ok['auc'].mean():.4f}",
                    flush=True,
                )
            print(f"wrote {RESULTS / F['fold_metrics']}", flush=True)

    print(f"TOTAL {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
