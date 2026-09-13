"""src/training/hpo.py — hyperparameter sweeps (owner's second07 semantics + masking).

Two lanes, both on the OFFICIAL clean-sku pair scheme:

  grid: fixed 11-config grid
        (epochs x lr x warmup), fold AUC + sd + cross-country AUC per
        config, config_mean summary block, --quick 3-config smoke. Every
        config trains WITH masking augmentation (variable configured extent,
        U(mask_lo, mask_hi) per masked copy — config/training.yaml band)
        when enabled — "it must be like this, but with masking as well".

  tpe   (second08 lane): optuna TPE over HPO_SPACE, resume-safe sqlite
        study, per-fold mean AUC objective — src/training/training.run_hpo.

Writes (results/):
  hpo_grid.csv                          one row per config per fold + config_mean
                                       summary (name via F["hpo_grid_csv"])
  train_<model><era>_hpo_best.json     best config + metrics (tpe lane,
  train_<model><era>_hpo_trials.csv    run-tagged names built inline in
                                       src/training/training.run_hpo, which prints
                                       the actual out_path)
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from core.common import SEED, F, set_determinism
from core.common import SSOT_LOSS as _SSOT_LOSS  # no-fallback SSOT

# second07's grid semantics (epochs x lr x warmup%), SSOT: config/training.yaml
# hpo.grid / hpo.quick — module-level literal lists were a second
# declaration the config could not steer (audit 2026-09-09, owner Q27).
from core.common import hpo_cfg as _hpo_cfg
from core.common import runtime as _runtime  # SSOT optimizer knobs (F06)
from training.training import ES_PATIENCE, ES_THRESHOLD, _band_tuple, train_one_config

_HPO = _hpo_cfg()
GRID = _HPO["grid"]    # the full 11-config sweep
QUICK = _HPO["quick"]  # the --quick 3-config smoke subset


def run_grid(
    args,
    data,
    mask_cfg,
    folds_override=None,
    dev_override=None,
    n_masked=0,
    # gate hard-no pairs: the contrastive SSOT loss needs labeled
    # negatives — same channel the main lane uses (train_one_config
    # filters them to each fold's train side via pairs_in_set)
    neg_pairs=None,
    # training-only augmented negatives; neg_pairs remains eval-only
    train_neg_pairs=None,
    neg_pair_sources=None,
    train_neg_pair_sources=None,
    dynamic_mask_hard_negatives=False,
    dynamic_mask_frac=0.0,
    dynamic_mask_prob=None,
    mask_audit=None,
    hard_negative_mask_audit=None,
) -> None:
    """Fixed-grid sweep (second07 semantics) over the ENTRY-augmented tuple.

    Holdout split (test-leak fix, 2026-09-12): every config trains on
    q0+q1, early-stops + is RANKED on the dev quarter q2's calibration Rand;
    the per-config test-side eval is SKIPPED (train_one_config
    selection_mode). CV split keeps component folds — fold test sides are
    validation folds there, calibration Rand stays the reported metric.

    Masking: applied ONCE by the entry lane (src/training/train.py resolves
    CLI > config and augments BEFORE the zero-shot encode), so the tuple
    arriving here is augmented AND emb0-aligned; every config trains on
    the IDENTICAL augmented data. The lane used to re-run
    augment_positives here — extending payload past emb0 and dying at
    the DataTuple contract (emb0 rows != payload) on every
    masking-enabled run (the shipped config has masking on), and it
    silently re-masked after an explicit --mask-frac 0. The sweep
    inherits the entry's augmentation; n_masked is the entry's count
    for the summary row.
    """
    # determinism at LANE entry (2026-10-06): see run_tpe — pin before
    # any per-config training randomness (the entry lane already did).
    set_determinism(SEED)
    import torch

    grid = QUICK if args.quick else GRID
    df, payload, structured_features, row_bc, country, pos, hp_pairs, emb0 = data

    # side arrays must cover canonical entries appended to the payload
    # (same pad src/training/train.main does before building the data tuple)
    import numpy as _np

    if len(country) < len(payload):
        country = _np.concatenate(
            [country, _np.full(len(payload) - len(country), "", dtype=country.dtype)]
        )
    data = (df, payload, structured_features, row_bc, country, pos, hp_pairs, emb0)

    # masking provenance: the augmentation itself lives in the entry lane
    # (identical for every config — the sweep measures optimizer knobs,
    # not augmentation noise). NO FALLBACK (owner Q27): the state comes
    # from the entry's CLI>config resolution, never re-derived here.
    print(
        f"[hpo-grid] masking: +{n_masked:,} masked positives in every config "
        f"(applied once by the entry lane)",
        flush=True,
    )

    # holdout lanes pass folds_override=[test_bc] + dev_override=q2 from
    # the SAME component split the main lane uses; only cv lanes fall
    # back to building component folds here. NO FALLBACK (owner Q27): a
    # holdout grid without the explicit boundary is a config error, not
    # something to paper over with a fold rebuild (that rebuild read the
    # dev/test quarters into the sweep's train side — the leak this fixes).
    _holdout = getattr(args, "split", None) == "holdout"
    if _holdout:
        assert folds_override is not None and dev_override is not None, (
            "[hpo-grid] holdout split requires the component split's "
            "folds_override (test quarter) + dev_override (dev quarter) — "
            "rebuilding folds over all barcodes leaks test into the sweep"
        )
        folds = [set(folds_override)] if isinstance(folds_override, (set, frozenset)) else list(folds_override)
        print(
            f"[hpo-grid] holdout selection: {len(folds)} test fold(s), "
            f"dev_override={len(dev_override):,} barcodes — per-config "
            f"test eval SKIPPED (test read exactly once)",
            flush=True,
        )
    else:
        folds = _grid_folds(args, data)
        print(f"[hpo-grid] cv: {len(folds)} component folds", flush=True)

    rows: list[dict] = []
    t0 = time.perf_counter()
    for cfg in grid:
        cfg_id = f"e{cfg['epochs']}_lr{cfg['lr']:.0e}_w{cfg['warmup']}"
        tcfg = {
            "architecture": _runtime("architecture"),
            "epochs": cfg["epochs"],
            "lr": cfg["lr"],
            "warmup_ratio": cfg["warmup"] / 100.0,
            # AUDIT FIX (round 2 F06, round 3): these three knobs read the
            # SSOT (training: block) via runtime() — same pattern as
            # src/training/train.py's cfg build. The old inline 0.01/"linear"/1.0
            # literals matched the config TODAY but nothing kept them
            # aligned, so the grid could train under different
            # regularization than the lane it tunes.
            "weight_decay": _runtime("weight_decay"),
            "lr_scheduler": _runtime("lr_scheduler"),
            "max_grad_norm": _runtime("max_grad_norm"),
            "patience": ES_PATIENCE,
            "es_threshold": ES_THRESHOLD,
            "uniformity_weight": (
                float(_runtime("uniformity_regularization") ["weight"])
                if bool(_runtime("uniformity_regularization") ["enabled"])
                else 0.0
            ),
        }
        # Both split modes rank on the same component-safe calibration Rand
        # proxy. Holdout calibration is dev-side; CV calibration is
        # validation-side. Neither path uses the final holdout test score.
        _sig, _sd_key = "calibration_rand_index", "calibration_rand_index_sd"
        fold_rows = train_one_config(
            tcfg,
            # SSOT: training.loss (owner ruling: contrastive default)
            loss=_SSOT_LOSS,  # training.loss from config/paths.yaml — no fallback (owner Q27)
            model_id=args.model,
            use_hp=False,
            band=_band_tuple(args.band),
            data=data,
            seed=SEED,
            on_cuda=torch.cuda.is_available(),
            cv_folds=None,
            # folds built ONCE above the loop (cv: component folds;
            # holdout: the passed test-quarter boundary) — rebuilding
            # _grid_folds per config re-dealt the CV split every iteration
            # (same seed => same deal, but a second component_folds pass
            # over the full graph per config, and a second place to drift).
            folds_override=folds,
            run_tag=f"g{cfg_id}",
            # holdout selection protocol: train q0+q1, select on q2, skip
            # the test eval (train_one_config asserts the boundary)
            dev_override=dev_override if _holdout else None,
            selection_mode=_holdout,
            # gate hard-negatives ride the SAME explicit channel the main
            # lane uses — without them the contrastive loss (SSOT) has no
            # labeled negatives and every fold skips
            neg_pairs=neg_pairs,
            train_neg_pairs=train_neg_pairs,
            neg_pair_sources=neg_pair_sources,
            train_neg_pair_sources=train_neg_pair_sources,
            dynamic_mask_hard_negatives=dynamic_mask_hard_negatives,
            dynamic_mask_frac=dynamic_mask_frac,
            dynamic_mask_prob=dynamic_mask_prob,
            mask_audit=mask_audit,
            hard_negative_mask_audit=hard_negative_mask_audit,
        )
        vals = [
            r[_sig]
            for r in fold_rows
            if r.get("status") == "ok"
            and np.isfinite(r.get(_sig, float("nan")))
        ]
        for r in fold_rows:
            row = dict(r)
            row["config"] = cfg_id
            rows.append(row)
        if vals:
            _mean = float(np.mean(vals))
            _sd = float(np.std(vals))
            _label = "calibration Rand"
            print(
                f"{cfg_id}: {_label} {_mean:.4f} (sd {_sd:.4f})",
                flush=True,
            )
        else:
            _mean = _sd = float("nan")
        # config_mean summary row (second07's summary block). The metric is
        # identical in holdout and CV so grid and TPE consume one schema.
        rows.append(
            {
                "config": cfg_id,
                "fold": "mean",
                "objective": _sig,
                _sig: round(_mean, 4),
                _sd_key: round(_sd, 4),
                "n_masked_pos": n_masked,
            }
        )

    out = F["hpo_grid_csv"]
    pd.DataFrame(rows).to_csv(out, index=False)
    print(
        f"\nwrote {out} ({len(rows)} rows) in {time.perf_counter() - t0:.0f}s",
        flush=True,
    )


def _grid_folds(
    args: argparse.Namespace, data: tuple
) -> list[set[str]]:
    """CV folds for the grid (second07 used k-fold; component folds here)."""
    from training.folds import component_folds

    row_bc = data[3]
    pos = data[5]
    return component_folds(pos, row_bc, args.folds, SEED)


def run_tpe(
    args: argparse.Namespace,
    data: tuple,
    mask_cfg: dict,
    folds_override: list[set[str]] | set[str] | None = None,
    dev_override: set[str] | None = None,
    n_masked: int = 0,
    # gate hard-no pairs (same channel as the main lane — see run_grid)
    neg_pairs: np.ndarray | None = None,
    # training-only augmented negatives; neg_pairs remains eval-only
    train_neg_pairs: np.ndarray | None = None,
    neg_pair_sources: np.ndarray | None = None,
    train_neg_pair_sources: np.ndarray | None = None,
    dynamic_mask_hard_negatives: bool = False,
    dynamic_mask_frac: float = 0.0,
    dynamic_mask_prob: float | None = None,
    mask_audit: list[dict] | None = None,
    hard_negative_mask_audit: list[dict] | None = None,
    wandb_ctx=None,
    mlf_ctx=None,
) -> None:
    """Optuna TPE lane over the ENTRY-augmented tuple — delegates to
    TRAIN.training.run_hpo.

    Holdout split (test-leak fix, 2026-09-12): the component split's
    boundary (folds_override=test quarter, dev_override=q2) replaces the
    old all-barcode CV folds + per-fold rng carve — trials train q0+q1,
    rank on calibration Rand, never see a test-side metric.

    Masking: applied ONCE by the entry lane BEFORE the zero-shot encode
    (see run_grid) — re-running augment_positives here extended payload
    past emb0 and died at the DataTuple contract on masking-enabled
    runs. The lane inherits the entry's augmentation (n_masked is its
    count, printed for provenance).
    """
    from training.training import run_hpo

    # determinism at LANE entry (2026-10-06): both sweep lanes get the
    # same pin even when invoked directly — the entry lane (src/training/train
    # .py _main_inner) already seeds; this makes a standalone run_tpe
    # call identical, before any optuna/trial randomness.
    set_determinism(SEED)
    df, payload, structured_features, row_bc, country, pos, hp_pairs, emb0 = data
    _holdout = getattr(args, "split", None) == "holdout"
    if _holdout:
        # NO FALLBACK (owner Q27): holdout without the explicit boundary is
        # a wiring error — defaulting to all-barcode CV folds would put the
        # test quarter's barcodes in the trials' train/dev sides (the leak
        # this fix closes).
        assert folds_override is not None and dev_override is not None, (
            "[hpo-tpe] holdout split requires the component split's "
            "folds_override (test quarter) + dev_override (dev quarter) — "
            "rebuilding folds over all barcodes leaks test into the sweep"
        )
        folds = [set(folds_override)] if isinstance(folds_override, (set, frozenset)) else list(folds_override)
        print(
            f"[hpo-tpe] holdout selection: {len(folds)} test fold(s), "
            f"dev_override={len(dev_override):,} barcodes — per-trial "
            f"test eval SKIPPED (test read exactly once)",
            flush=True,
        )
    else:
        # cv: the component fold list from the main lane (masking appends
        # same-barcode payload entries, so the passed boundary stays valid)
        folds = folds_override
    # masking provenance (same discipline as run_grid): the entry lane
    # applied the augmentation once, pre-encode; no re-augment here.
    print(
        f"[hpo-tpe] masking: +{n_masked:,} masked positives in every trial "
        f"(applied once by the entry lane)",
        flush=True,
    )
    # both boundaries pass through UNCONDITIONALLY (never mode-gated): the
    # holdout lane gets [test quarter] + q2, the cv lane gets the
    # component folds (dev_override=None there -> the same per-fold rng
    # carve the main cv lane uses). run_hpo ->
    # train_one_config(folds_override=None) would rebuild folds over ALL
    # barcodes — q3 included — and the sweep would train on the test
    # quarter; the assert above is what keeps that from happening quietly.
    run_hpo(
        args,
        (df, payload, structured_features, row_bc, country, pos, hp_pairs, emb0),
        mlf_ctx if mlf_ctx is not None else _mlf_null(),
        cv_folds=None,
        folds_override=folds,
        dev_fraction=args.dev_fraction,
        dev_override=dev_override,
        selection_mode=_holdout,
        neg_pairs=neg_pairs,
        train_neg_pairs=train_neg_pairs,
        neg_pair_sources=neg_pair_sources,
        train_neg_pair_sources=train_neg_pair_sources,
        dynamic_mask_hard_negatives=dynamic_mask_hard_negatives,
        dynamic_mask_frac=dynamic_mask_frac,
        dynamic_mask_prob=dynamic_mask_prob,
        mask_audit=mask_audit,
        hard_negative_mask_audit=hard_negative_mask_audit,
        wandb_ctx=wandb_ctx,
    )


def _mlf_null():
    # NOTE (audit 2026-09-09): TRAIN.training re-exports the SSOT
    # src/core/mlflow_ctx.MlflowCtx — local sqlite by default, =off disables.
    # The old duplicate (off unless URI set) shadowed the owner mandate.
    from training.training import MlflowCtx

    return MlflowCtx("hpo")
