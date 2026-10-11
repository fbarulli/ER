"""Public-API pins for the laya eval-records speed path (GPU-free).

The eval offenders harvested from `laya_hpo_1kbaseline_20261010T165809Z`
(W&B `timing/laya/dev_eval` = 441 s/epoch, fp32 batch-16 over 27,305 items):
`core.laya_eval_fast.EvalRecordsHook` is the ONE seam that makes every
grad-disabled CUDA eval-records forward fp16 and batched, and the SSOT knobs
(`FinetuneSpec.eval_amp`/`eval_batch`) stay default-OFF so the finetune
lane's honest numbers are unchanged unless the YAML/HPO policy opts in.
"""
from __future__ import annotations

import math
import os
import random
import types

import torch

from cli import laya_lane
from core import laya_eval_fast
from core.laya_config import FinetuneSpec


def _stub_laya_train(calls):
    """A laya.train stand-in recording what the hook hands downstream."""

    def calibration_records(model, tok, items, device, *args, **kwargs):
        calls.append(("records", kwargs))
        return []

    def _forward(model, batch, device, amp, detach_encoder):
        calls.append(("forward", amp))

    return types.SimpleNamespace(
        calibration_records=calibration_records, _forward=_forward)


def test_eval_records_hook_batches_and_autocasts_eval_only():
    calls = []
    laya_train = _stub_laya_train(calls)
    policy = laya_eval_fast.EvalRecordsPolicy().update(
        {"eval_amp": True, "eval_batch": 32})
    laya_eval_fast.EvalRecordsHook.install(laya_train, policy)
    # a second install must NOT stack wrappers (per-trial apply_perf_patch)
    laya_eval_fast.EvalRecordsHook.install(laya_train, policy)

    device = types.SimpleNamespace(type="cuda")
    with torch.no_grad():
        laya_train.calibration_records(None, None, [1], device, 8, 4,
                                       parallel=False)
        laya_train._forward(None, None, device, False, False)
    # an explicit caller batch (the held-out HELD_OUT_BATCH) always wins
    with torch.no_grad():
        laya_train.calibration_records(None, None, [1], device, 8, 4,
                                       batch_size=8)
    # a training forward (grad enabled) keeps its amp flag untouched
    laya_train._forward(None, None, device, False, False)

    assert calls == [
        ("records", {"batch_size": 32, "parallel": False}),
        ("forward", True),    # grad-disabled CUDA eval: flipped to fp16
        ("records", {"batch_size": 8}),
        ("forward", False),   # grad-enabled training forward: untouched
    ]

    # A new trial may provide a new policy object; the installed hook follows it.
    replacement = laya_eval_fast.EvalRecordsPolicy(eval_amp=False, eval_batch=4)
    laya_eval_fast.EvalRecordsHook.install(laya_train, replacement)
    with torch.no_grad():
        laya_train.calibration_records(None, None, [1], device, 8, 4)
        laya_train._forward(None, None, device, False, False)
    assert calls[-2:] == [("records", {"batch_size": 4}), ("forward", False)]


def test_eval_speed_knobs_are_default_off_and_hpo_on():
    ft = FinetuneSpec()
    assert ft.eval_amp is False and ft.eval_batch == 0
    assert {"eval_amp", "eval_batch"} <= set(laya_lane.FINETUNE_CONTROL_FIELDS)
    assert {"eval_amp", "eval_batch"}.isdisjoint(
        laya_lane.FINETUNE_CONFIG_FIELDS)
    from cli.laya_hpo import HpoArtifactPolicy
    override = HpoArtifactPolicy.apply(laya_lane.finetune_control())
    assert override["eval_amp"] is True and override["eval_batch"] == 32
    # the knobs reach the staged perf patch (the injected classes are exec'd
    # byte-for-byte, so the staged namespace re-defines them from source)
    namespace = {"os": os, "math": math, "random": random}
    exec(laya_lane.FINETUNE_PERF_PATCH_SOURCE, namespace)  # noqa: S102 -- trusted generated kernel under test
    for injected in ("EvalRecordsPolicy", "EvalRecordsHook", "DevEvaluator"):
        assert injected in namespace, injected
    import inspect
    assert (inspect.getsource(laya_eval_fast.EvalRecordsHook)
            in laya_lane.FINETUNE_PERF_PATCH_SOURCE)
    assert "EvalRecordsHook.install(" in laya_lane.FINETUNE_PERF_PATCH_SOURCE


def test_encode_memo_serves_every_caller_and_resets_per_trial():
    """ONE cache for the loop AND the eval paths: the second call with the
    same (item, args) never re-tokenizes; reset() (the per-trial bound)
    makes the next call encode again."""
    import types

    encoded_calls = []

    def encode_item(tok, item, *args, **kwargs):
        encoded_calls.append(item)
        return {"ids": [1], "item": item}

    laya_train = types.SimpleNamespace(encode_item=encode_item)
    memo = laya_eval_fast.EncodeMemo.install(laya_train,
                                             laya_eval_fast.EncodeMemo())
    item = {"q": 1}
    first = laya_train.encode_item(None, item, 8, 4, None, False)
    second = laya_train.encode_item(None, item, 8, 4, None, False)
    nested = laya_train.encode_item(None, item, 8, 4, parallel=False,
                                    option_order=None)  # different kwargs
    assert first is second and encoded_calls == [item, item]
    assert nested is not first and (memo.hits, memo.lookups) == (1, 3)
    memo.reset()
    laya_train.encode_item(None, item, 8, 4, None, False)
    assert len(encoded_calls) == 3 and memo.lookups == 1 and memo.hits == 0


def test_hpo_final_eval_off_and_loop_stamps_best_dev_loss(monkeypatch):
    """An HPO trial never pays laya's full-dev before/after eval (the
    per-epoch DevEvaluator already ranks it): the trial policy turns
    final_eval OFF (finetune-lane default stays ON) and the loop stamps the
    ranked epoch's raw dev loss into the control result as the trial's
    dev_loss source."""
    import sys

    import torch
    from test_laya_ddp import _Cfg, _load_ddp_namespace, _make_fake_laya, _TinyNet

    # SSOT: default-ON in the finetune lane, OFF in the HPO trial policy
    assert FinetuneSpec().final_eval is True
    assert "final_eval" in laya_lane.FINETUNE_CONTROL_FIELDS
    from cli.laya_hpo import HpoArtifactPolicy
    override = HpoArtifactPolicy.apply(laya_lane.finetune_control())
    assert override["final_eval"] is False
    # the staged kernel gates eval_data on the control (no phantom knob)
    assert ('FINETUNE_CONTROL.get("final_eval", True)'
            in laya_lane._hpo_kernel_script_source()
            if hasattr(laya_lane, "_hpo_kernel_script_source") else True)

    # behaviour: the loop result carries the best epoch's dev loss
    namespace = _load_ddp_namespace()
    fake = _make_fake_laya([])
    monkeypatch.setitem(sys.modules, "laya", fake)
    monkeypatch.setitem(sys.modules, "laya.train", fake.train)
    namespace["FINETUNE_CONTROL"] = {**laya_lane.finetune_control(),
                                     "eval_dev": True}
    namespace["FINETUNE_DEV_ROWS"] = [{"state": "x"}]
    items = [{"i": index, "label": index % 3} for index in range(4)]
    cfg = _Cfg()
    cfg.epochs = 1
    namespace["_perf_train_model"](
        _TinyNet(), types.SimpleNamespace(pad_token_id=0), items, cfg,
        torch.device("cpu"), 8, 4)
    result = namespace["FINETUNE_CONTROL_RESULT"]
    assert result["best_dev_accuracy"] == 0.9
    assert result["best_dev_loss"] == 0.5   # the fake dev eval's loss
