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
  python TRAIN/train.py --loss contrastive     (entry; TRAIN/ is a package)
  python TRAIN/train.py --hpo --n-trials 20    # optuna TPE sweep
  python TRAIN/train.py --loss triplet --band 0.45-0.80
  MLFLOW_TRACKING_URI=... uv run ...      # local sqlite default; =off disables
                                            # (SSOT: lib/mlflow_ctx.py)

Artifacts (results/, SSOT via 00_config):
  train_<model>_fold_metrics.csv  one row per fold per config (incl. failures
                               with traceback column)
  train_<model>_hpo_best.json    best config + its metrics (HPO mode)
"""

from __future__ import annotations

import json
import time
import traceback

import numpy as np
import pandas as pd

# DEFAULT_MODEL retired: the entry (train.py) resolves the trainer base
# from the config SSOT (models.multilingual_l12); the lib.cache.MODEL_NAME
# indirection was dead code with a phantom docstring.
from transformers import EarlyStoppingCallback, TrainerCallback

from lib.common import (
    RESULTS,
    SEED,
    kfold_barcodes,
    load_config,
    pair_auc,
    pair_similarity,
    runtime,
)
from lib.common import SSOT_CONTRASTIVE_MARGIN as _SSOT_MARGIN
from lib.common import runtime as _runtime

_SSOT_HP = bool(_runtime("hard_positives"))  # no-fallback SSOT
from lib.hard_negatives import mine_hard_negatives, pairs_in_set

# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

# ── runtime knobs: SSOT ONLY (00_config.yaml training: block via
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

# TPE search space, SSOT: TRAIN/training.yaml hpo.tpe_space (validated by
# HpoSpaceSpec at load — lo < hi per knob). The dict literal was a second
# declaration the config could not steer (audit 2026-09-09, owner Q27).
from lib.common import hpo_cfg as _hpo_cfg_load

HPO_SPACE = {k: (lo, hi) for k, (lo, hi) in _hpo_cfg_load()["tpe_space"].items()}


# ═══════════════════════════════════════════════════════════════════════════
# MLflow — SSOT lib/mlflow_ctx (audit 2026-09-09: this module used to carry
# its own MlflowCtx duplicate with CONFLICTING semantics — "off unless
# MLFLOW_TRACKING_URI is set" — while the owner mandate (training logs
# available locally) is lib's: local sqlite by default, =off to disable.
# The duplicate shadowed the mandate; re-exported here for hpo.py's import.)
# ═══════════════════════════════════════════════════════════════════════════

from lib.mlflow_ctx import MlflowCtx

# ═══════════════════════════════════════════════════════════════════════════
# Training (ST 6 modern Trainer path with HF early stopping)
# ═══════════════════════════════════════════════════════════════════════════


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
    from lib.common import write_visibility_log

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
) -> list[dict]:
    """Train cfg across the group-aware folds. Returns fold metric rows
    (failures included, with traceback)."""
    import torch
    from sentence_transformers import SentenceTransformer

    # BOUNDARY CONTRACT (lib.schemas.TrainConfig): the optimizer/early-stop
    # dict — every key validated (epochs >= 1, lr > 0, warmup in [0,1]...)
    # before a single fold runs. A missing/illegal knob dies HERE with the
    # field named, not inside the HF Trainer mid-epoch.
    from lib.schemas import TrainConfig as _TrainConfig

    _TrainConfig.model_validate(cfg)

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
    from lib.schemas import DataTuple as _DataTuple

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
    from lib.common import band as _band_helper

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

            model = SentenceTransformer(model_id, device="cuda" if on_cuda else "cpu")
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
                # eval batch = the SSOT encode batch (loss on dev pairs is
                # gradient-free; same batch the dev evaluator uses)
                per_device_eval_batch_size=runtime("batch_size_eval"),
                metric_for_best_model="eval_dev_cosine_ap",
                greater_is_better=True,
                load_best_model_at_end=True,
                save_strategy="steps",
                save_steps=eval_steps,
                save_total_limit=int(runtime("save_total_limit")),  # SSOT
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

            trainer = SentenceTransformerTrainer(
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
            # validation loss curve (owner directive 2026-09-10): the dev
            # evaluator's own loss per eval step — the pair the train-vs-val
            # plot draws. Absent only when no eval ran (skipped/failed fold).
            dev_losses = [e["eval_loss"] for e in hist if "eval_loss" in e]
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

            _pr_auc = float(average_precision_score(_y, _all))
            from lib.common import load_config as _lc

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
            # population as pr_auc so those CSVs regenerate from TRAIN/train runs.
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
        "objective": f"discriminative-LR ({_runtime('layer_decay')}^k per-layer groups)",
    }
    out_path = RESULTS / f"train_{model_tag}{era}_hpo_best.json"
    with open(out_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"\nBEST: {study.best_params} -> mean AUC {study.best_value:.4f}", flush=True)
    # AUDIT FIX (round 2, F02): print the path ACTUALLY written — the old
    # line named train_hpo_best.json, a file never written by this lane.
    print(f"wrote {out_path}", flush=True)


def _optuna_mlflow_cb(mlf: MlflowCtx):
    def cb(study, trial):
        if trial.state.name == "COMPLETE" and trial.value is not None:
            mlf.log_metrics({f"trial_{trial.number}_auc": trial.value})

    return cb


def _band_tuple(band: str) -> tuple[float, float]:
    lo, hi = (float(x) for x in band.split("-"))
    return lo, hi

