"""Trainer for a bundle prepared on the local machine.

This entrypoint deliberately has no dataset, pair-building, masking,
country-padding, or calibration-input generation path. Those inputs are
validated in the local prepared bundle before this process starts.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core.common import (
    F,
    RESULTS,
    SEED,
    load_config,
    masking_cfg,
    resolve_model,
    runtime,
    set_determinism,
)
from core.timing import emit_timing
from core.wandb_ctx import WandbCtx
from training.attestation import (
    TrainingAttestation,
    read_attestation,
    verify_attestation,
    verify_plan_identity,
)
from training.training import ES_PATIENCE, ES_THRESHOLD, train_one_config
from training.prepared_bundle import load_prepared_bundle, prepared_holdout


def _parse_args() -> argparse.Namespace:
    cfg = load_config()
    tr = cfg["training"]
    split = cfg["split"]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument('--shared-training-data', type=Path)
    ap.add_argument('--training-binding', type=Path)
    ap.add_argument('--allow-unshared-supervision', action='store_true',
                    help='train without the shared supervision binding (recorded, off by default)')
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
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--report-test", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--resume", action="store_true", help="restore native trainer checkpoints in this run")
    ap.add_argument("--mask-effect", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--attestation", type=Path, default=None,
                    help="skip re-validation with this handoff attestation "
                         "or run handoff.json (env: ER_TRAINING_ATTESTATION)")
    ap.add_argument("--run-tag", default=os.environ.get("EUROMONITOR_RUN_ID", "prepared"))
    return ap.parse_args()


def _attestation_gate(args: argparse.Namespace, bundle_path: Path) -> TrainingAttestation | None:
    raw = getattr(args, "attestation", None) or os.environ.get("ER_TRAINING_ATTESTATION")
    if not raw:
        return None
    attestation = read_attestation(Path(raw), bundle_path=bundle_path)
    verify_attestation(attestation, bundle_path=bundle_path)
    verify_plan_identity(attestation, loss=args.loss,
                         train_frac=args.train_frac, sample=bool(args.sample))
    emit_timing(f"[timing] training.attestation verified bundle_sha256={attestation.bundle_sha256}")
    return attestation


def main() -> None:
    args = _parse_args()
    run_name = str(args.run_tag)
    # W&B is supplementary telemetry.  A checkout-native Colab smoke must
    # remain runnable with the signed-in Colab account alone: its artifacts
    # are collected by the launcher either way.
    with WandbCtx(run_name) as wandb_ctx:
        if not wandb_ctx.enabled:
            print("[wandb] disabled; continuing with Colab result collection", flush=True)
        _main(args, wandb_ctx)


def _main(args: argparse.Namespace, wandb_ctx: WandbCtx) -> None:
    set_determinism(SEED)
    cfg = load_config()
    if (
        args.collapse_guardrail_profile is not None
        and args.collapse_guardrail_profile != str(cfg["collapse_guardrail"]["profile"])
    ):
        raise ValueError(
            "prepared training uses the configured collapse guardrail profile; "
            f"CLI profile={args.collapse_guardrail_profile!r} differs from "
            f"active profile={cfg['collapse_guardrail']['profile']!r}. "
            "Set the profile in config/training.yaml before preparing and launching."
        )
    attestation = _attestation_gate(args, args.bundle)
    # Check-free path: the attestation (bundle sha256 + boundary report) owns
    # verification, so the load must not re-force the full digest/array
    # validation through the data-gate default. Without an attestation the
    # loader's own gate decision applies, unchanged.
    manifest, bundle = load_prepared_bundle(
        args.bundle, verify_inputs=False) if attestation else load_prepared_bundle(args.bundle)
    shared_path = getattr(args, 'shared_training_data', None)
    binding_path = getattr(args, 'training_binding', None)
    if bool(shared_path) != bool(binding_path):
        raise ValueError('shared training data and track binding must be supplied together')
    if not shared_path and not args.allow_unshared_supervision:
        # The suite always supplies the binding (model_tracks.worker). Training
        # without it means the supervision is not the shared population, which
        # is a deliberate, recorded decision — never a default. Checked before
        # any other bundle interpretation so the contract fails fast.
        raise ValueError(
            'prepared training requires --shared-training-data with '
            '--training-binding so the supervision is the shared population; '
            'pass --allow-unshared-supervision to train without it on purpose')
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
    if "training_tokens" not in bundle:
        raise ValueError("prepared bundle lacks local training tokens; rebuild locally before GPU training")
    if args.loss != "contrastive" and bool(cfg["mining"]["ann"]["refresh_enabled"]):
        raise ValueError("live ANN refresh is currently supported only for contrastive training")
    profile = masking_cfg(manifest.masking_profile)
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
        # Each prepared worker owns its frozen-input copy. F is a process-local
        # mapping, so downstream split/calibration readers use this copy without
        # modifying checkout inputs shared with graph/hybrid workers.
        destination = RESULTS / "_prepared_inputs" / Path(F[file_key]).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(bundle[bundle_key])
        F[file_key] = destination
        print(
            f"[prepared-bundle] materialized {file_key}={destination} "
            f"bytes={len(bundle[bundle_key]):,}",
            flush=True,
        )

    if args.split != "holdout":
        raise ValueError("prepared GPU training currently supports the SSOT holdout split only")
    if "training_plan" not in bundle:
        raise ValueError("prepared bundle lacks local training row plan; rebuild locally before GPU training")
    if attestation is not None:
        # The handoff boundary attested this exact bundle's plan identity and
        # epoch coverage; re-proving the bundle bytes is the only live check.
        plan = bundle["training_plan"]
    else:
        from training.run_plan import validate_run_plan

        _run_plan_started = time.perf_counter()
        plan = validate_run_plan(bundle, bundle["training_plan"], loss=args.loss,
                                 train_frac=args.train_frac, sample=bool(args.sample), seed=SEED)
        emit_timing(
            f"[timing] training.run_plan data_digest_revalidation: "
            f"{time.perf_counter() - _run_plan_started:.3f}s"
        )
    if shared_path:
        from model_tracks.training_data import SharedTrainingData, TrackTrainingBinding, from_bundle
        shared = SharedTrainingData.model_validate_json(shared_path.read_text())
        binding = TrackTrainingBinding.model_validate_json(binding_path.read_text())
        if binding.track != 'text' or from_bundle(bundle).fingerprint != shared.fingerprint:
            raise ValueError('text frozen objective differs from shared training data')
        binding.validate_data(shared)
        print(f'[shared-training/text] examples={len(shared.examples)} endpoints={len(shared.endpoints)} '
              f'sha256={shared.fingerprint}', flush=True)
    train_bc, dev_bc, test_bc = (plan["holdout"][key] for key in ("train", "dev", "test"))
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

    from training.run_plan import training_config
    import torch

    cfg_train = training_config(epochs=args.epochs, lr=args.lr)
    data = (df, payload, structured_features, row_bc, country, pos, hp_pairs, emb0)
    rows = train_one_config(
        cfg_train,
        loss=args.loss,
        model_id=model_id,
        use_hp=bool(cfg["training"]["hard_positives"]) and len(hp_pairs) > 0,
        band=tuple(float(x) for x in args.band.split("-")),
        data=data,
        seed=SEED,
        on_cuda=(args.device == "cuda" or
                 (args.device == "auto" and torch.cuda.is_available())),
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
        ann_refresh_enabled=bool(cfg["mining"]["ann"]["refresh_enabled"]),
        attribute_conflict_refresh_enabled=(args.loss == "contrastive" and bool(cfg["mining"]["attribute_conflict"]["enabled"])),
        prepared_tokens=bundle["training_tokens"],
        prepared_plan=plan,
        train_frac=args.train_frac if args.train_frac < 1.0 else None,
        run_tag=args.run_tag,
        sample=bool(args.sample),
        resume=bool(getattr(args, "resume", False)),
        selection_mode=not args.report_test,
        skip_test_eval=not args.report_test,
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
    if len(ok) != len(rows_out):
        raise SystemExit(
            f"prepared training failed: successful_folds={len(ok)}/{len(rows_out)}\n"
            + '\n'.join(str(row.get('traceback', row)) for row in rows_out if row.get('status') != 'ok')
        )


if __name__ == "__main__":
    main()
