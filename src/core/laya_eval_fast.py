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

from typing import Any

from pydantic import BaseModel

from core.laya_controls import DevReport
from core.laya_train_fast import AutocastCacheFactory


class EvalRecordsCall(BaseModel, frozen=True):
    """One immutable Laya evaluation request; retain identities for safe reuse."""

    model: object
    tokenizer: object
    items: object
    device: object
    args: tuple[object, ...]
    kwargs: dict[str, object]

    def matches(self, other: "EvalRecordsCall") -> bool:  # noqa: UP037 -- injected without future annotations
        return (self.model is other.model and self.tokenizer is other.tokenizer
                and self.items is other.items and self.device == other.device
                and self.args == other.args and self.kwargs == other.kwargs)

    def run(self, original: "Any", amp: bool) -> object:  # noqa: UP037 -- injected without module imports
        import torch

        with AutocastCacheFactory.build(torch, self.device, amp):
            return original(self.model, self.tokenizer, self.items, self.device,
                            *self.args, **self.kwargs)


class EvalRecordsResult(BaseModel):
    """Laya owns the opaque records representation; reuse it without conversion."""

    call: EvalRecordsCall
    records: object


class EvalRecordsReuse:
    """Bounded to the final calibration tail, where weights no longer mutate."""

    def __init__(self) -> None:
        self._latest: "EvalRecordsResult | None" = None  # noqa: UP037 -- injected without future annotations

    def records(self, call: "EvalRecordsCall", original: "Any", amp: bool) -> object:  # noqa: UP037 -- injected without module imports
        latest = self._latest
        if latest is None or not latest.call.matches(call):
            latest = EvalRecordsResult(call=call, records=call.run(original, amp))
            self._latest = latest
        return latest.records


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
    REUSE = "_er_eval_records_reuse"

    @classmethod
    def reuse_after_training(cls, laya_train: "Any") -> None:  # noqa: UP037 -- injected without module imports
        setattr(laya_train, cls.REUSE, EvalRecordsReuse())

    @classmethod
    def release_after_training(cls, laya_train: "Any") -> None:  # noqa: UP037 -- injected without module imports
        setattr(laya_train, cls.REUSE, None)

    @classmethod
    def install(cls, laya_train: "Any", policy: "EvalRecordsPolicy") -> "EvalRecordsPolicy":  # noqa: UP037 -- injected without module imports
        cls.release_after_training(laya_train)
        installed = getattr(laya_train, cls.INSTALLED, None)
        if installed is not None:
            setattr(laya_train, cls.INSTALLED, policy)
            return policy
        original_records = laya_train.calibration_records
        original_forward = laya_train._forward

        def calibration_records(model: "Any", tok: "Any", items: "Any",  # noqa: UP037 -- injected without module imports
                                device: "Any", *args: "Any", **kwargs: "Any") -> "Any":  # noqa: UP037
            policy = getattr(laya_train, cls.INSTALLED)
            if policy.eval_batch > 0:
                kwargs.setdefault("batch_size", policy.eval_batch)
            call = EvalRecordsCall(model=model, tokenizer=tok, items=items,
                                   device=device, args=args, kwargs=kwargs)
            reuse = getattr(laya_train, cls.REUSE)
            if reuse is not None:
                return reuse.records(call, original_records, policy.eval_amp)
            return call.run(original_records, policy.eval_amp)

        def _forward(model: "Any", batch: "Any", device: "Any", amp: bool,  # noqa: UP037 -- injected without module imports
                     detach_encoder: bool) -> "Any":  # noqa: UP037
            import torch

            policy = getattr(laya_train, cls.INSTALLED)
            if (not amp and policy.eval_amp and not torch.is_grad_enabled()
                    and getattr(device, "type", "") == "cuda"):
                amp = True
            return original_forward(model, batch, device, amp, detach_encoder)

        laya_train.calibration_records = calibration_records
        laya_train._forward = _forward
        setattr(laya_train, cls.INSTALLED, policy)
        return policy


class EncodeMemo:
    """Process-wide memoization of laya's ``encode_item`` (ONE cache for the
    training loop AND every eval path).

    laya re-tokenizes from the raw item on EVERY ``calibration_records``
    call, so the 27,305-item dev split paid tokenization every epoch inside
    dev_eval, and the finetune before/after evals paid it again. The loop's
    old per-call dict covered TRAIN items only; this memo wraps the module
    function once, so every caller hits the same cache. Keyed by
    ``(tokenizer identity, item identity, args, kwargs)`` — items are immutable dicts owned by the
    trial; ``reset()`` per trial (apply_perf_patch) bounds the cache to ONE
    trial's items, the same lifetime the loop-local dict had. Cache entries retain
    both inputs so object-id reuse cannot return another input's encoding.
    """

    class Entry:
        """Retain input identities so Python cannot reuse a cached object's id."""

        def __init__(self, tokenizer: "Any", item: "Any", encoded: "Any") -> None:  # noqa: UP037 -- injected without module imports
            self.tokenizer = tokenizer
            self.item = item
            self.encoded = encoded

    def __init__(self) -> None:
        self._cache: "dict[tuple[Any, ...], EncodeMemo.Entry]" = {}  # noqa: UP037
        self.hits = 0
        self.lookups = 0

    @staticmethod
    def _part(value: "Any") -> "Any":  # noqa: UP037 -- injected without module imports
        return tuple(value) if isinstance(value, list) else value

    @classmethod
    def _key(cls, tok: "Any", item: "Any", args: "tuple[Any, ...]",  # noqa: UP037 -- injected without module imports
             kwargs: "dict[str, Any]") -> "tuple[Any, ...]":  # noqa: UP037
        return ((id(tok), id(item)) + tuple(cls._part(value) for value in args)
                + tuple(sorted((name, cls._part(value))
                               for name, value in kwargs.items())))

    def reset(self) -> "EncodeMemo":  # noqa: UP037 -- injected without module imports
        self._cache.clear()
        self.hits = 0
        self.lookups = 0
        return self

    @classmethod
    def install(cls, laya_train: "Any", memo: "EncodeMemo") -> "EncodeMemo":  # noqa: UP037 -- injected without module imports
        current = laya_train.encode_item
        if getattr(current, "_er_memo", None) is memo:
            return memo
        original = getattr(current, "_er_original", current)

        def encode_item(tok: "Any", item: "Any", *args: "Any", **kwargs: "Any") -> "Any":  # noqa: UP037 -- injected without module imports
            memo.lookups += 1
            key = cls._key(tok, item, args, kwargs)
            entry = memo._cache.get(key)
            if entry is None:
                entry = cls.Entry(tok, item, original(tok, item, *args, **kwargs))
                memo._cache[key] = entry
            else:
                memo.hits += 1
            return entry.encoded

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


class DevArtifacts(BaseModel):
    """The shared artifact shape at the evaluator-to-training-result boundary."""

    items: int
    accuracy: float | None
    ece: float | None
    brier: float | None
    brier_top1: float | None
    confusion: dict[str, dict[str, dict[str, int]]]
    temperature: object


class DevEvaluator:
    """Rank-0 dev scoring + end-of-run calibration/confusion artifacts."""

    @staticmethod
    def metrics(laya_train: "Any", model: "Any", tok: "Any",  # noqa: UP037 -- injected without module imports
                dev_items: "Any", device: "Any", max_len: int,  # noqa: UP037
                head_max_len: int, parallel: bool, control: "dict[str, Any]",  # noqa: UP037
                collate_memo: "CollateMemo | None" = None) -> "DevReport":  # noqa: UP037
        records = DevEvaluator._records(laya_train, model, tok, dev_items,
                                        device, max_len, head_max_len,
                                        parallel, collate_memo)
        metrics = laya_train.evaluate_records(records)
        return DevReport.from_metrics(
            metrics, records=records,
            confidence_threshold=control.get("abstain_confidence"))

    @staticmethod
    def artifacts(laya_train: "Any", model: "Any", tok: "Any",  # noqa: UP037 -- injected without module imports
                  dev_items: "Any", device: "Any", max_len: int,  # noqa: UP037
                  head_max_len: int, parallel: bool, control: "Any",  # noqa: UP037
                  collate_memo: "CollateMemo | None" = None) -> "DevArtifacts":  # noqa: UP037
        records = DevEvaluator._records(laya_train, model, tok, dev_items,
                                        device, max_len, head_max_len,
                                        parallel, collate_memo)
        metrics = laya_train.evaluate_records(records)
        confusion = {}
        import numpy as np

        for qtype, logits, target, k in records:
            pred = int(np.asarray(logits, dtype=float)[:int(k)].argmax())
            gold = int(np.asarray(target, dtype=float)[:int(k)].argmax())
            bucket = confusion.setdefault(str(qtype), {})
            row = bucket.setdefault(str(gold), {})
            row[str(pred)] = row.get(str(pred), 0) + 1
        temperature = None
        if control.get("temperature_scale"):
            try:
                temperature = laya_train.fit_temperature_map(
                    records).get("temperature")
            except Exception:  # noqa: BLE001 - optional fit, recorded in full
                import traceback

                print("[perf-patch] temperature fit failed: "
                      + traceback.format_exc(), flush=True)
        return DevArtifacts(
            items=len(records), accuracy=metrics.get("accuracy"),
            ece=metrics.get("ece"), brier=metrics.get("brier"),
            brier_top1=metrics.get("brier_top1"), confusion=confusion,
            temperature=temperature)

    @staticmethod
    def _records(laya_train: "Any", model: "Any", tok: "Any",  # noqa: UP037 -- injected without module imports
                 dev_items: "Any", device: "Any", max_len: int,  # noqa: UP037
                 head_max_len: int, parallel: bool,
                 collate_memo: "CollateMemo | None" = None) -> "Any":  # noqa: UP037
        from contextlib import nullcontext

        with collate_memo if collate_memo is not None else nullcontext():
            return laya_train.calibration_records(
                model, tok, dev_items, device, max_len, head_max_len,
                parallel=parallel)
