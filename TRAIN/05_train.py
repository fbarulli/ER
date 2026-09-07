"""05_train.py — the TRAIN_GPU training entry point (the ONE callable in
TRAIN_GPU root; the training internals live in TRAIN/).

Trains on the OFFICIAL clean-sku scheme (DATA_PIPE.pairs):
  positive : (clean_sku_text, canonical_gtin)
  negative : (clean_sku_text, canonical_other_gtin) — gate hard-no pairs
Run:  python 05_train.py --model models/<name> [--split holdout|cv ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import argparse
import json
import time

import numpy as np
import pandas as pd

from lib.common import RESULTS, SEED, F, load_config, load_dataset_deduped
from TRAIN.folds import component_folds
from TRAIN.training import ES_PATIENCE, ES_THRESHOLD, train_one_config

# thresholds: config is SSOT (00_config pairs.*); these are fallback docs —
# positives: proceed & sim>=0.80 | hard-negatives: hard_no & sim>=0.80
PROCEED_SIM_THRESHOLD = 0.80
HARDNEG_SIM_THRESHOLD = 0.80


def load_training_data(df: pd.DataFrame, payload_variant: str = "full") -> dict:
    """OFFICIAL pair construction (DATA_PIPE.pairs): payload = clean sku
    text per row + canonical per GTIN; pos = (sku, own canonical), neg =
    (sku, other canonical) from gate hard-no pairs."""
    from data_pipe import build_training_data

    return build_training_data(df, payload_variant=payload_variant)


def main() -> None:
    """Entry: everything runs inside one mlflow parent run (local backend
    + artifact store at artifacts/mlruns — browsable with
    `mlflow ui --backend-store-uri file:artifacts/mlruns`; MLFLOW_TRACKING_URI
    overrides, =off disables)."""
    from lib.mlflow_ctx import MlflowCtx

    with MlflowCtx("train_gpu") as _mlf:
        _main_inner(_mlf)


def _main_inner(_mlf) -> None:
    cfg = load_config()
    tr = cfg.get("training", {})
    pairs_cfg = cfg.get("pairs", {})
    mining_cfg = cfg.get("mining", {})
    mask_cfg = cfg.get("masking", {})

    # thresholds: config is SSOT; module constants stay as fallback docs
    global PROCEED_SIM_THRESHOLD, HARDNEG_SIM_THRESHOLD
    PROCEED_SIM_THRESHOLD = float(
        pairs_cfg.get("proceed_sim_threshold", PROCEED_SIM_THRESHOLD)
    )
    HARDNEG_SIM_THRESHOLD = float(
        pairs_cfg.get("hardneg_sim_threshold", HARDNEG_SIM_THRESHOLD)
    )
    default_model = "sentence-transformers/" + cfg.get("models", {}).get(
        "multilingual_l12", "paraphrase-multilingual-MiniLM-L12-v2"
    )
    default_band = str(mining_cfg.get("band", "0.45-0.80"))

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=int(tr.get("epochs", 3)))
    ap.add_argument("--lr", type=float, default=float(tr.get("lr", 2e-5)))
    ap.add_argument(
        "--split",
        choices=["holdout", "cv"],
        default=str(cfg.get("split", {}).get("mode", "holdout")),
        help="holdout: ONE component-aware 50/25/25 "
        "train/dev/test split (the owner's spec) | "
        "cv: k component folds (research mode)",
    )
    ap.add_argument(
        "--folds",
        type=int,
        default=int(cfg.get("split", {}).get("cv_folds", 5)),
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
        "--n-target-mining", type=int, default=int(mining_cfg.get("n_target", 20_000))
    )
    ap.add_argument("--dev-fraction", type=float, default=0.15)
    ap.add_argument(
        "--sample",
        type=int,
        default=None,
        help="debug: cap dataset rows (full chain, tiny data)",
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
        choices=["mnrl", "triplet"],
        default="mnrl",
        help="training loss (mnrl default; triplet = mined hard negatives)",
    )
    ap.add_argument(
        "--no-hard-positives",
        action="store_true",
        help="disable the volume-verified hard-positive lane",
    )
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
    ap.add_argument("--n-trials", type=int, default=20, help="tpe lane: trial budget")
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
        help="masking augmentation (00_config masking.*): "
        "fraction of positive anchors to mask. Default "
        "comes from the config — enabled:false -> 0 "
        "(off), enabled:true -> masking.frac. Explicit "
        "CLI value always wins.",
    )
    args = ap.parse_args()

    # masking resolution: CLI flag > config; config default OFF
    if args.mask_frac is None:
        args.mask_frac = (
            float(mask_cfg.get("frac", 0.15)) if mask_cfg.get("enabled") else 0.0
        )
    mask_prob = mask_cfg.get("mask_prob", None)
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
    # 07c field ablation — payload variants (only the TEXT the encoder sees
    # changes, so fold metrics are comparable):
    #   full       = clean sku (title + attributes, the owner's cleaning)
    #                — pairs (sku, own canonical)
    #   title_only = clean sku from title alone
    data = load_training_data(df, payload_variant=args.payload)
    payload, row_bc, pos, neg = (
        data["payload"],
        data["row_bc"],
        data["pos"],
        data["neg"],
    )
    s = data["stats"]
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
    if args.payload != "full":
        print(f"[07c] payload variant: {args.payload}", flush=True)
    country = df["country"].fillna("").astype(str).to_numpy()
    if len(pos) == 0:
        raise SystemExit(
            "no positives resolved — check canonical_records.csv GTINs "
            "vs dataset_deduped.csv barcodes"
        )

    # ── masking augmentation (TRAIN/masking.py, on/off via 00_config) ──
    # Owner's masking augmentation, corrected for MNRL semantics: the original
    # the original masking script fed label=0.0 hard-negative pairs to MNRL — MNRL
    # IGNORES labels and would train them as POSITIVES (different products
    # pulled together). Here masking augments POSITIVES only: for a fraction
    # of pairs, the ANCHOR text gets 15% random token masking and the masked
    # pair is appended as an EXTRA positive (same pair semantics, noised
    # anchor). Masked texts are NEW payload entries (row_bc = same barcode),
    # so folds/components are unaffected.
    if args.mask_frac > 0:
        from TRAIN.masking import augment_positives

        pos, payload, row_bc, n_added = augment_positives(
            pos, payload, row_bc, frac=args.mask_frac, mask_prob=mask_prob, seed=SEED
        )
        print(
            f"[masking] +{n_added:,} masked-anchor positive pairs "
            f"(frac={args.mask_frac:.0%}, extent={'5-15% variable' if mask_prob is None else mask_prob})",
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
        quarters = component_folds(pos, row_bc, 4, SEED)
        test_bc, dev_bc = quarters[3], quarters[2]
        folds_override = test_bc  # single-set: train = all others
        dev_override = dev_bc
        from lib.hard_negatives import pairs_in_set

        n_tr = len({b for b in row_bc if b and b not in test_bc and b not in dev_bc})
        tr_pos = len(
            pos[
                pairs_in_set(
                    pos, row_bc, None or (set(row_bc.tolist()) - test_bc - dev_bc)
                )
            ]
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

    hp_pairs = np.empty((0, 2), dtype=int)  # hard-positive lane off: the
    # new pipeline's proceed-pairs ARE the volume-verified signal now.

    from lib.nlp import encode_corpus

    emb0, encode_s = encode_corpus(
        args.model,
        payload,
        batch_size=256,
        max_seq_length=128,
        device="cuda" if on_cuda else "cpu",
    )
    print(f"zero-shot encode_s = {encode_s:.1f}s", flush=True)

    # masked anchors extend the row-index space beyond df; every df-indexed
    # side array must cover them (row_bc/payload/emb0 already do; country
    # doesn't — pad with the anchor's own country)
    if len(country) < len(payload):
        pad = np.full(len(payload) - len(country), "", dtype=country.dtype)
        country = np.concatenate([country, pad])

    data = (df, payload, row_bc, country, pos, hp_pairs, emb0)

    # ── HPO lanes: second07 fixed grid / second08 optuna TPE ───────────────
    if args.grid or args.hpo:
        from TRAIN.hpo import run_grid, run_tpe

        if args.grid:
            run_grid(args, data, mask_cfg)
        else:
            run_tpe(args, data, mask_cfg)
        return

    # TRAIN/training's DEFAULT_CFG key set (train_one_config reads these)

    cfg = {
        "epochs": args.epochs,
        "lr": args.lr,
        "warmup_ratio": 0.05,
        "weight_decay": 0.01,
        "lr_scheduler": "linear",
        "max_grad_norm": 1.0,
        "patience": ES_PATIENCE,
        "es_threshold": ES_THRESHOLD,
    }
    t0 = time.perf_counter()
    run_tag = f"train-{args.split}-{args.payload}".replace("/", "-")
    rows = train_one_config(
        cfg,
        loss="mnrl",
        model_id=args.model,
        use_hp=False,
        band=band,
        data=data,
        seed=SEED,
        on_cuda=on_cuda,
        cv_folds=None,
        folds_override=folds_override,
        dev_fraction=args.dev_fraction,
        dev_override=dev_override,
        neg_pairs=neg,
        train_frac=args.train_frac if args.train_frac < 1.0 else None,
        run_tag=run_tag,
    )
    elapsed = time.perf_counter() - t0

    # persist metrics
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / F["fold_metrics"]
    all_rows = []
    for r in rows:
        row = dict(r)
        row["model"] = args.model
        all_rows.append(row)
    pd.DataFrame(all_rows).to_csv(out, index=False)

    ok_rows = [r for r in rows if r.get("status") != "skipped"]
    aucs = [r.get("auc") for r in ok_rows if r.get("auc") is not None]

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
    for r in ok_rows:
        with _mlf.nested:
            _mlf.log_params({"fold": r.get("fold")})
            _mlf.log_metrics(
                {
                    k: r[k]
                    for k in (
                        "auc",
                        "auc_cross",
                        "acc_at_thr",
                        "youden_thr",
                        "pr_auc",
                        "f1_at_0.55",
                        "precision_at_0.55",
                        "recall_at_0.55",
                        "best_dev_ap",
                        "final_train_loss",
                        "s_per_step",
                    )
                    if r.get(k) is not None
                }
            )
            if r.get("traceback"):
                continue
    if aucs:
        _mlf.log_metrics(
            {"mean_auc": float(np.mean(aucs)), "std_auc": float(np.std(aucs))}
        )
    _mlf.log_artifact(out)
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
    if args.plot and aucs:
        from TRAIN.plots import report_plots, training_loss_plot

        report_plots(out, args)
        training_loss_plot(out, args)
        for png in RESULTS.glob("train_*.png"):
            _mlf.log_artifact(png)
        for png in RESULTS.glob("training_loss_*.png"):
            _mlf.log_artifact(png)

    if args.rerank:
        from TRAIN.rerank import rerank_stage

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
