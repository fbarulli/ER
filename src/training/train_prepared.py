"""GPU-only trainer for a bundle prepared on the local machine.

This entrypoint deliberately has no dataset, pair-building, masking,
country-padding, or calibration-input generation path. Those inputs are
validated in the local prepared bundle before this process starts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from core.common import (
    F,
    RESULTS,
    SEED,
    collapse_guardrail_cfg,
    load_config,
    masking_cfg,
    resolve_model,
    runtime,
    set_determinism,
)
from core.mlflow_ctx import MlflowCtx
from core.wandb_ctx import WandbCtx
from training.folds import holdout_split
from training.training import ES_PATIENCE, ES_THRESHOLD, train_one_config
from training.prepared_bundle import load_prepared_bundle


def _parse_args() -> argparse.Namespace:
    cfg = load_config()
    tr = cfg["training"]
    split = cfg["split"]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--model", default=str(tr["base_model"]))
    ap.add_argument("--epochs", type=int, default=int(tr["epochs"]))
    ap.add_argument("--lr", type=float, default=float(tr["lr"]))
    ap.add_argument("--train-frac", type=float, default=1.0)
    ap.add_argument("--split", choices=["holdout"], default=str(split["mode"]))
    ap.add_argument("--payload", choices=["full", "title_only"], default="full")
    ap.add_argument("--loss", choices=["contrastive", "mnrl", "triplet"], default=str(tr["loss"]))
    ap.add_argument("--band", default=str(cfg["mining"]["ann"]["band"]))
    ap.add_argument("--masking-profile", default=None)
    ap.add_argument("--collapse-guardrail-profile", default=None)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--mask-effect", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--run-tag", default=os.environ.get("EUROMONITOR_RUN_ID", "prepared"))
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    run_name = str(args.run_tag)
    # This is the remote training entrypoint, not a local-only convenience
    # path. A normal Colab run must create both tracking contexts; proceeding
    # without W&B would make a missing injected credential look successful.
    with MlflowCtx(run_name), WandbCtx(run_name) as wandb_ctx:
        if not wandb_ctx.enabled:
            raise RuntimeError(
                "prepared remote training requires WANDB_API_KEY; refusing "
                "a local-only tracking fallback"
            )
        _main(args, wandb_ctx)


def _main(args: argparse.Namespace, wandb_ctx: WandbCtx) -> None:
    set_determinism(SEED)
    cfg = load_config()
    manifest, bundle = load_prepared_bundle(args.bundle)
    if args.payload != manifest.payload_variant:
        raise ValueError(
            f"bundle payload={manifest.payload_variant!r} but CLI payload={args.payload!r}"
        )
    if args.masking_profile and args.masking_profile != manifest.masking_profile:
        raise ValueError(
            f"bundle masking profile={manifest.masking_profile!r} "
            f"but CLI profile={args.masking_profile!r}"
        )
    model_id = resolve_model(args.model)
    profile = masking_cfg(manifest.masking_profile)
    collapse_cfg = collapse_guardrail_cfg(str(cfg["collapse_guardrail"]["profile"]))
    df = bundle["df"]
    payload = bundle["payload"]
    structured_features = np.asarray(bundle["structured_features"], dtype=np.float32)
    row_bc = np.asarray(bundle["row_bc"])
    country = np.asarray(bundle["country"])
    pos = np.asarray(bundle["pos"], dtype=int)
    hp_pairs = np.asarray(bundle["hp_pairs"], dtype=int)
    emb0 = np.asarray(bundle["emb0"], dtype=np.float32)
    neg = np.asarray(bundle["neg"], dtype=int)
    train_neg = np.asarray(bundle["train_neg"], dtype=int)
    neg_sources = np.asarray(bundle["neg_sources"], dtype=object)
    train_neg_sources = np.asarray(bundle["train_neg_sources"], dtype=object)
    mask_audit = list(bundle["mask_audit"])
    hard_negative_mask_audit = list(bundle["hard_negative_mask_audit"])
    frozen_inputs = {
        "labeled_pairs": "labeled_pairs_csv",
        "canonical_records": "canonical_records_csv",
        "gate_results": "gate_results_csv",
    }
    for file_key, bundle_key in frozen_inputs.items():
        destination = RESULTS / F[file_key]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(bundle[bundle_key])
        print(
            f"[prepared-bundle] materialized {file_key}={destination} "
            f"bytes={len(bundle[bundle_key]):,}",
            flush=True,
        )

    if args.split != "holdout":
        raise ValueError("prepared GPU training currently supports the SSOT holdout split only")
    train_bc, dev_bc, test_bc = holdout_split(
        pos,
        row_bc,
        n_folds=int(cfg["split"]["holdout_component_folds"]),
        seed=SEED,
        dev_fraction=float(cfg["split"]["dev_fraction"]),
        test_fraction=float(cfg["split"]["test_fraction"]),
    )
    print(
        f"[prepared-bundle] loaded {args.bundle} "
        f"sha256={manifest.sha256} profile={manifest.masking_profile} "
        f"rows={manifest.n_df:,} payload={manifest.n_payload:,}",
        flush=True,
    )
    print(
        f"[prepared-bundle] holdout train={len(train_bc):,} "
        f"dev={len(dev_bc):,} test={len(test_bc):,}; "
        "remote data preparation=disabled",
        flush=True,
    )

    cfg_train = {
        "architecture": runtime("architecture"),
        "epochs": args.epochs,
        "lr": args.lr,
        "warmup_ratio": runtime("warmup_ratio"),
        "weight_decay": runtime("weight_decay"),
        "lr_scheduler": runtime("lr_scheduler"),
        "max_grad_norm": runtime("max_grad_norm"),
        "patience": ES_PATIENCE,
        "es_threshold": ES_THRESHOLD,
        "uniformity_weight": (
            float(cfg["training"]["uniformity_regularization"]["weight"])
            if bool(cfg["training"]["uniformity_regularization"]["enabled"])
            else 0.0
        ),
        "late_epoch_decay_enabled": bool(cfg["training"]["late_epoch_lr_decay"]["enabled"]),
        "late_epoch_decay_start_fraction": float(
            cfg["training"]["late_epoch_lr_decay"]["start_epoch_fraction"]
        ),
        "late_epoch_decay_multiplier": float(
            cfg["training"]["late_epoch_lr_decay"]["multiplier"]
        ),
    }
    data = (df, payload, structured_features, row_bc, country, pos, hp_pairs, emb0)
    rows = train_one_config(
        cfg_train,
        loss=args.loss,
        model_id=model_id,
        use_hp=bool(cfg["training"]["hard_positives"]) and len(hp_pairs) > 0,
        band=tuple(float(x) for x in args.band.split("-")),
        data=data,
        seed=SEED,
        on_cuda=__import__("torch").cuda.is_available(),
        folds_override=test_bc,
        dev_fraction=float(cfg["training"]["dev_fraction"]),
        dev_override=dev_bc,
        neg_pairs=neg,
        train_neg_pairs=train_neg,
        neg_pair_sources=neg_sources,
        train_neg_pair_sources=train_neg_sources,
        dynamic_mask_hard_negatives=bool(profile["mask_hard_negatives"]),
        dynamic_mask_frac=float(profile["hard_negative_frac"]),
        dynamic_mask_prob=(
            float(profile["hard_negative_mask_prob"])
            if profile["hard_negative_mask_prob"] is not None
            else None
        ),
        dynamic_mask_lo=float(profile["hard_negative_mask_lo"]),
        dynamic_mask_hi=float(profile["hard_negative_mask_hi"]),
        mask_audit=mask_audit,
        hard_negative_mask_audit=hard_negative_mask_audit,
        ann_refresh_enabled=False,
        attribute_conflict_refresh_enabled=False,
        train_frac=args.train_frac if args.train_frac < 1.0 else None,
        run_tag=args.run_tag,
        sample=False,
        resume=False,
        wandb_ctx=wandb_ctx,
    )
    out = RESULTS / f"train_{Path(str(model_id)).name}_holdout_{manifest.payload_variant}_fold_metrics.csv"
    rows_out = []
    for row in rows:
        row_out = dict(row)
        row_out.update(
            {
                "model": model_id,
                "payload": manifest.payload_variant,
                "train_frac": args.train_frac,
                "prepared_bundle": str(args.bundle),
                "prepared_bundle_sha256": manifest.sha256,
            }
        )
        rows_out.append(row_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_out).to_csv(out, index=False)
    ok = [row for row in rows_out if row.get("status") == "ok"]
    print(
        f"[prepared-bundle] complete folds_ok={len(ok)}/{len(rows_out)} "
        f"metrics={out}",
        flush=True,
    )
    print(json.dumps({"bundle": manifest.model_dump(), "metrics": str(out)}, indent=2))


if __name__ == "__main__":
    main()
