"""Eval-path speed classes for the laya perf patch (injected VERBATIM).

Same injection contract as `core.laya_controls`: cohesive classes, no
module-level torch import, unit-tested on the host and exec'd byte-for-byte
into the staged kernels (which cannot import the repo). Listed in
`cli.laya_train_patch_ddp.FINETUNE_CONTROL_LOGIC_SOURCE`.

Why this module exists: laya's `calibration_records` hardcodes
`_forward(amp=False)` and `batch_size=16`, so EVERY eval-records call (the
per-epoch dev eval, the finetune before/after evals, the held-out eval) paid
fp32 forwards on the T4 — 441 s/epoch for the 27,305-item dev split in the
`laya_hpo_1kbaseline_20261010T165809Z` runs (W&B `timing/laya/dev_eval`).
"""
from __future__ import annotations


class EvalRecordsPolicy:
    """Per-trial eval-records speed policy (fp16 autocast + batch override).

    Values come from the training-control block (SSOT `FinetuneSpec`
    `eval_amp` / `eval_batch`); `eval_batch == 0` keeps laya's own default.
    """

    def __init__(self, eval_amp: bool = False, eval_batch: int = 0):
        self.eval_amp = bool(eval_amp)
        self.eval_batch = max(0, int(eval_batch))

    def update(self, control) -> "EvalRecordsPolicy":
        self.eval_amp = bool(control.get("eval_amp"))
        self.eval_batch = max(0, int(control.get("eval_batch") or 0))
        return self


class EvalRecordsHook:
    """Batched, fp16 eval records: wraps `calibration_records` + `_forward`.

    The `_forward` wrapper flips `amp` on ONLY for grad-disabled CUDA
    forwards — that is exactly laya's eval path (`calibration_records` is
    `@torch.no_grad`); training forwards run with grad enabled and keep
    their flag, so the recipe is untouched. The `calibration_records`
    wrapper injects the control-selected batch size as a default, so an
    explicit caller batch (the held-out `HELD_OUT_BATCH`) always wins.

    Install is idempotent (one wrapper per process); a re-install only
    refreshes the policy, so per-trial `apply_perf_patch` calls never stack
    wrappers.
    """

    INSTALLED = "_er_eval_records_installed"

    @classmethod
    def install(cls, laya_train, policy: EvalRecordsPolicy):
        if getattr(laya_train, cls.INSTALLED, False):
            return policy
        original_records = laya_train.calibration_records
        original_forward = laya_train._forward

        def calibration_records(model, tok, items, device, *args, **kwargs):
            if policy.eval_batch > 0:
                kwargs.setdefault("batch_size", policy.eval_batch)
            return original_records(model, tok, items, device, *args, **kwargs)

        def _forward(model, batch, device, amp, detach_encoder):
            import torch

            if (not amp and policy.eval_amp and not torch.is_grad_enabled()
                    and getattr(device, "type", "") == "cuda"):
                amp = True
            return original_forward(model, batch, device, amp, detach_encoder)

        laya_train.calibration_records = calibration_records
        laya_train._forward = _forward
        setattr(laya_train, cls.INSTALLED, True)
        return policy


class DevEvaluator:
    """Rank-0 dev scoring + end-of-run calibration/confusion artifacts."""

    @staticmethod
    def metrics(laya_train, model, tok, dev_items, device, max_len,
                head_max_len, parallel, control, annotate=None):
        records = DevEvaluator._records(laya_train, model, tok, dev_items,
                                        device, max_len, head_max_len,
                                        parallel, annotate)
        metrics = laya_train.evaluate_records(records)
        return DevReport.from_metrics(
            metrics, records=records,
            confidence_threshold=control.get("abstain_confidence"))

    @staticmethod
    def artifacts(laya_train, model, tok, dev_items, device, max_len,
                  head_max_len, parallel, control, annotate=None):
        records = DevEvaluator._records(laya_train, model, tok, dev_items,
                                        device, max_len, head_max_len,
                                        parallel, annotate)
        metrics = laya_train.evaluate_records(records)
        confusion = {}
        for qtype, logits, target, k in records:
            try:
                import numpy as np
                z = np.asarray(logits, dtype=float)[:int(k)]
                pred = int(z.argmax())
                gold = int(np.asarray(target, dtype=float)[:int(k)].argmax())
            except Exception:
                continue
            bucket = confusion.setdefault(str(qtype), {})
            row = bucket.setdefault(str(gold), {})
            row[str(pred)] = row.get(str(pred), 0) + 1
        temperature = None
        if control.get("temperature_scale"):
            try:
                temperature = laya_train.fit_temperature_map(
                    records).get("temperature")
            except Exception as error:
                print("[perf-patch] temperature fit skipped: "
                      + str(error)[:120], flush=True)
        return {
            "items": len(records),
            "accuracy": metrics.get("accuracy"),
            "ece": metrics.get("ece"),
            "brier": metrics.get("brier"),
            "brier_top1": metrics.get("brier_top1"),
            "confusion": confusion,
            "temperature": temperature,
        }

    @staticmethod
    def _records(laya_train, model, tok, dev_items, device, max_len,
                 head_max_len, parallel, annotate=None):
        import contextlib

        scope = annotate("calibration") if annotate else contextlib.nullcontext()
        with scope:
            return laya_train.calibration_records(
                model, tok, dev_items, device, max_len, head_max_len,
                parallel=parallel)
