"""TRAIN/hpo.py — hyperparameter sweeps (owner's second07 semantics + masking).

Two lanes, both on the OFFICIAL clean-sku pair scheme:

  grid: fixed 11-config grid
        (epochs x lr x warmup), fold AUC + sd + cross-country AUC per
        config, config_mean summary block, --quick 3-config smoke. Every
        config trains WITH masking augmentation (variable 5-15% extent)
        when enabled — "it must be like this, but with masking as well".

  tpe   (second08 lane): optuna TPE over HPO_SPACE, resume-safe sqlite
        study, per-fold mean AUC objective — TRAIN/training.run_hpo.

Writes (results/, names via 00_config files:):
  hpo_grid.csv       one row per config per fold + config_mean summary
  hpo_tpe_best.json  best config + metrics (tpe lane)
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from lib.common import RESULTS, SEED, F
from TRAIN.masking import augment_positives
from TRAIN.training import ES_PATIENCE, ES_THRESHOLD, train_one_config

# second07's grid, verbatim: epochs x lr x warmup% — small by design
GRID = [
    {"epochs": 1, "lr": 2e-5, "warmup": 0},
    {"epochs": 1, "lr": 5e-5, "warmup": 0},
    {"epochs": 1, "lr": 1e-4, "warmup": 0},
    {"epochs": 2, "lr": 2e-5, "warmup": 0},
    {"epochs": 2, "lr": 5e-5, "warmup": 0},
    {"epochs": 2, "lr": 1e-4, "warmup": 0},
    {"epochs": 4, "lr": 2e-5, "warmup": 0},
    {"epochs": 4, "lr": 5e-5, "warmup": 0},
    {"epochs": 4, "lr": 1e-4, "warmup": 0},
    {"epochs": 1, "lr": 2e-5, "warmup": 10},
    {"epochs": 2, "lr": 5e-5, "warmup": 10},
]

QUICK = [
    {"epochs": 1, "lr": 2e-5, "warmup": 0},
    {"epochs": 2, "lr": 5e-5, "warmup": 0},
    {"epochs": 2, "lr": 5e-5, "warmup": 10},
]


def run_grid(args, data, mask_cfg) -> None:
    """Fixed-grid sweep (second07 semantics) with masking in every config."""
    import torch

    grid = QUICK if args.quick else GRID
    df, payload, row_bc, country, pos, hp_pairs, emb0 = data

    # side arrays must cover canonical entries appended to the payload
    # (same pad 05_train.main does before building the data tuple)
    import numpy as _np

    if len(country) < len(payload):
        country = _np.concatenate(
            [country, _np.full(len(payload) - len(country), "", dtype=country.dtype)]
        )

    # masking augmentation applied ONCE, identically for every config —
    # the sweep measures optimizer knobs, not augmentation noise
    n_added = 0
    if mask_cfg.get("enabled") or (args.mask_frac or 0) > 0:
        frac = args.mask_frac if args.mask_frac else float(mask_cfg.get("frac", 0.15))
        pos, payload, row_bc, n_added = augment_positives(
            pos, payload, row_bc, frac=frac, mask_prob=None, seed=SEED
        )
        print(
            f"[hpo-grid] masking on: +{n_added:,} masked positives in every config",
            flush=True,
        )
    data = (df, payload, row_bc, country, pos, hp_pairs, emb0)

    rows: list[dict] = []
    t0 = time.perf_counter()
    for cfg in grid:
        cfg_id = f"e{cfg['epochs']}_lr{cfg['lr']:.0e}_w{cfg['warmup']}"
        tcfg = {
            "epochs": cfg["epochs"],
            "lr": cfg["lr"],
            "warmup_ratio": cfg["warmup"] / 100.0,
            "weight_decay": 0.01,
            "lr_scheduler": "linear",
            "max_grad_norm": 1.0,
            "patience": ES_PATIENCE,
            "es_threshold": ES_THRESHOLD,
        }
        fold_rows = train_one_config(
            tcfg,
            loss="mnrl",
            model_id=args.model,
            use_hp=False,
            band=_band_tuple(args.band),
            data=data,
            seed=SEED,
            on_cuda=torch.cuda.is_available(),
            cv_folds=None,
            folds_override=_grid_folds(args, data),
            run_tag=f"g{cfg_id}",
        )
        aucs = [
            r["auc"]
            for r in fold_rows
            if r.get("status") == "ok" and np.isfinite(r.get("auc", np.nan))
        ]
        for r in fold_rows:
            row = dict(r)
            row["config"] = cfg_id
            rows.append(row)
        if aucs:
            print(
                f"{cfg_id}: AUC {np.mean(aucs):.4f} (sd {np.std(aucs):.4f})",
                flush=True,
            )
        # config_mean summary row (second07's summary block)
        rows.append(
            {
                "config": cfg_id,
                "fold": "mean",
                "auc": round(float(np.mean(aucs)), 4) if aucs else float("nan"),
                "auc_sd": round(float(np.std(aucs)), 4) if aucs else float("nan"),
                "n_masked_pos": n_added,
            }
        )

    out = RESULTS / F["hpo_grid_csv"]
    pd.DataFrame(rows).to_csv(out, index=False)
    print(
        f"\nwrote {out} ({len(rows)} rows) in {time.perf_counter() - t0:.0f}s",
        flush=True,
    )


def _band_tuple(band: str) -> tuple[float, float]:
    lo, hi = (float(x) for x in band.split("-"))
    return (lo, hi)


def _grid_folds(args, data):
    """CV folds for the grid (second07 used k-fold; component folds here)."""
    from TRAIN.folds import component_folds

    row_bc = data[2]
    pos = data[4]
    return component_folds(pos, row_bc, args.folds, SEED)


def run_tpe(args, data, mask_cfg) -> None:
    """Optuna TPE lane with masking — delegates to
    TRAIN.training.run_hpo after applying the same one-time augmentation."""
    from TRAIN.training import run_hpo

    df, payload, row_bc, country, pos, hp_pairs, emb0 = data
    if mask_cfg.get("enabled") or (args.mask_frac or 0) > 0:
        frac = args.mask_frac if args.mask_frac else float(mask_cfg.get("frac", 0.15))
        pos, payload, row_bc, n_added = augment_positives(
            pos, payload, row_bc, frac=frac, mask_prob=None, seed=SEED
        )
        print(
            f"[hpo-tpe] masking on: +{n_added:,} masked positives in every trial",
            flush=True,
        )
    run_hpo(
        args,
        (df, payload, row_bc, country, pos, hp_pairs, emb0),
        _mlf_null(),
        cv_folds=None,
        folds_override=None,
        dev_fraction=args.dev_fraction,
    )


def _mlf_null():
    from TRAIN.training import MlflowCtx

    return MlflowCtx("hpo")
