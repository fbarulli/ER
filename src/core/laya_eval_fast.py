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

    def update(self, control) -> "EvalRecordsPolicy":  # noqa: UP037 -- also injected without future annotations
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


class EncodeMemo:
    """Process-wide memoization of laya's ``encode_item`` (ONE cache for the
    training loop AND every eval path).

    laya re-tokenizes from the raw item on EVERY ``calibration_records``
    call, so the 27,305-item dev split paid tokenization every epoch inside
    dev_eval, and the finetune before/after evals paid it again. The loop's
    old per-call dict covered TRAIN items only; this memo wraps the module
    function once, so every caller hits the same cache. Keyed by
    ``(id(item), args, kwargs)`` — items are immutable dicts owned by the
    trial; ``reset()`` per trial (apply_perf_patch) bounds the cache to ONE
    trial's items, the same lifetime the loop-local dict had.
    """

    def __init__(self):
        self._cache = {}
        self.hits = 0
        self.lookups = 0

    @staticmethod
    def _key(item, args, kwargs):
        def part(value):
            return tuple(value) if isinstance(value, list) else value
        return ((id(item),) + tuple(part(value) for value in args)
                + tuple(sorted((name, part(value))
                               for name, value in kwargs.items())))

    def lookup(self, item, args, kwargs):
        self.lookups += 1
        return self._cache.get(self._key(item, args, kwargs))

    def store(self, item, args, kwargs, encoded):
        self._cache[self._key(item, args, kwargs)] = encoded

    def reset(self):
        self._cache.clear()
        self.hits = 0
        self.lookups = 0
        return self

    @classmethod
    def install(cls, laya_train, memo):
        current = laya_train.encode_item
        if getattr(current, "_er_memo", None) is memo:
            return memo
        original = getattr(current, "_er_original", current)

        def encode_item(tok, item, *args, **kwargs):
            encoded = memo.lookup(item, args, kwargs)
            if encoded is None:
                encoded = original(tok, item, *args, **kwargs)
                memo.store(item, args, kwargs, encoded)
            else:
                memo.hits += 1
            return encoded

        encode_item._er_memo = memo
        encode_item._er_original = original
        laya_train.encode_item = encode_item
        return memo


class CollateMemo:
    """Memoize laya's ``collate_items`` for the FIXED dev set (offender #6).

    ``calibration_records`` re-collates the SAME dev items every epoch (the
    train loop shuffles; dev does not), paying the Python collate again each
    epoch (CPU proxy: ~8.5 s per 7093-item dev pass). With ``EncodeMemo``
    returning stable encoded objects, the collate input ``id``s are stable
    across epochs, so keying on them caches the collated CPU batch and dev
    collate runs ONCE per trial.

    Scoped by the ``_active`` gate (this instance is its own context manager):
    ONLY a dev-eval window caches, so the train loop's ever-changing chunks
    never pollute or grow the cache. Reuses laya's ``collate_items`` — never a
    reimplementation. Numerically identical (collate is a pure function of the
    encoded items). Reset per trial alongside ``EncodeMemo``.
    """

    def __init__(self):
        self._cache = {}
        self._active = False
        self.hits = 0
        self.lookups = 0

    @staticmethod
    def _key(chunks, pad_token_id):
        return (pad_token_id,
                tuple(id(enc) for chunk in chunks for enc in chunk))

    def reset(self):
        self._cache.clear()
        self.hits = 0
        self.lookups = 0
        return self

    def __enter__(self):
        self._active = True
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._active = False
        return False

    @classmethod
    def install(cls, laya_train, memo):
        current = laya_train.collate_items
        if getattr(current, "_er_collate_memo", None) is memo:
            return memo
        original = getattr(current, "_er_original", current)

        def collate_items(chunks, pad_token_id):
            if not memo._active:
                return original(chunks, pad_token_id)
            memo.lookups += 1
            key = memo._key(chunks, pad_token_id)
            hit = memo._cache.get(key)
            if hit is not None:
                memo.hits += 1
                return hit
            out = original(chunks, pad_token_id)
            memo._cache[key] = out
            return out

        collate_items._er_collate_memo = memo
        collate_items._er_original = original
        laya_train.collate_items = collate_items
        return memo


class DevEvaluator:
    """Rank-0 dev scoring + end-of-run calibration/confusion artifacts."""

    @staticmethod
    def metrics(laya_train, model, tok, dev_items, device, max_len,
                head_max_len, parallel, control, collate_memo=None):
        records = DevEvaluator._records(laya_train, model, tok, dev_items,
                                        device, max_len, head_max_len,
                                        parallel, collate_memo)
        metrics = laya_train.evaluate_records(records)
        return DevReport.from_metrics(  # noqa: F821 -- DevReport co-injects (core.laya_controls) into the kernel namespace
            metrics, records=records,
            confidence_threshold=control.get("abstain_confidence"))

    @staticmethod
    def artifacts(laya_train, model, tok, dev_items, device, max_len,
                  head_max_len, parallel, control, collate_memo=None):
        records = DevEvaluator._records(laya_train, model, tok, dev_items,
                                        device, max_len, head_max_len,
                                        parallel, collate_memo)
        metrics = laya_train.evaluate_records(records)
        confusion = {}
        for qtype, logits, target, k in records:
            try:
                import numpy as np
                z = np.asarray(logits, dtype=float)[:int(k)]
                pred = int(z.argmax())
                gold = int(np.asarray(target, dtype=float)[:int(k)].argmax())
            except Exception:  # noqa: BLE001,S112 - a malformed record never fails the artifacts
                continue
            bucket = confusion.setdefault(str(qtype), {})
            row = bucket.setdefault(str(gold), {})
            row[str(pred)] = row.get(str(pred), 0) + 1
        temperature = None
        if control.get("temperature_scale"):
            try:
                temperature = laya_train.fit_temperature_map(
                    records).get("temperature")
            except Exception as error:  # noqa: BLE001 - the fit is optional, recorded and skipped
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
                 head_max_len, parallel, collate_memo=None):
        if collate_memo is None:
            return laya_train.calibration_records(
                model, tok, dev_items, device, max_len, head_max_len,
                parallel=parallel)
        with collate_memo:
            return laya_train.calibration_records(
                model, tok, dev_items, device, max_len, head_max_len,
                parallel=parallel)
