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
import sys
import types
import weakref
from typing import Any

import pytest
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

    class Tokenizer:
        """An identity-only tokenizer for the cache lifetime contract."""

    first_tokenizer, second_tokenizer = Tokenizer(), Tokenizer()
    left = laya_train.encode_item(first_tokenizer, item, 8, 4, None, False)
    right = laya_train.encode_item(second_tokenizer, item, 8, 4, None, False)
    assert left is not right
    assert laya_train.encode_item(first_tokenizer, item, 8, 4, None, False) is left
    assert len(encoded_calls) == 5
    retained = weakref.ref(first_tokenizer)
    del first_tokenizer
    assert retained() is not None
    memo.reset()
    assert retained() is None


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


def test_calibration_tail_reuses_logits_across_temperature_fitting_and_resets() -> None:
    from laya.train import evaluate_records

    calls = []
    model = torch.nn.Linear(2, 1, bias=False)

    def records(model: torch.nn.Linear, tok: object, items: list[object],
                device: torch.device, *args: object,
                **kwargs: object) -> list[tuple[int, Any, list[float], int]]:
        calls.append((items, args, kwargs))
        return [(0, model.weight.detach().numpy().reshape(-1).copy(), [1.0, 0.0], 2)]

    train = types.SimpleNamespace(calibration_records=records, _forward=lambda *args: None)
    policy = laya_eval_fast.EvalRecordsPolicy()
    laya_eval_fast.EvalRecordsHook.install(train, policy)
    items, tokenizer, device = [object()], object(), torch.device("cpu")
    # Training/baseline evaluations remain uncached while weights can change.
    train.calibration_records(model, tokenizer, items, device, 8, 4)
    train.calibration_records(model, tokenizer, items, device, 8, 4)
    laya_eval_fast.EvalRecordsHook.reuse_after_training(train)
    first = train.calibration_records(model, tokenizer, items, device, 8, 4)
    second = train.calibration_records(model, tokenizer, items, device, 8, 4)
    assert first is second and len(calls) == 3
    assert evaluate_records(first, [2.0]) == evaluate_records(second, [2.0])
    laya_eval_fast.EvalRecordsHook.release_after_training(train)
    train.calibration_records(model, tokenizer, items, device, 8, 4)
    assert len(calls) == 4
    train.calibration_records(model, tokenizer, items, device, 16, 4)
    assert len(calls) == 5
    # Reinstall starts a new trial, even when the policy object is unchanged.
    with torch.no_grad():
        model.weight.add_(1.0)
    laya_eval_fast.EvalRecordsHook.install(train, policy)
    third = train.calibration_records(model, tokenizer, items, device, 8, 4)
    assert len(calls) == 6
    assert not (first[0][1] == third[0][1]).all()


def test_error_artifact_failure_reports_the_full_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_laya_ddp import _Cfg, _load_ddp_namespace, _make_fake_laya, _TinyNet

    namespace = _load_ddp_namespace()
    fake = _make_fake_laya([])

    def records(*args: object, **kwargs: object) -> list[tuple[int, list[str], list[float], int]]:
        return [(0, ["invalid-logit"], [1.0], 1)]

    fake.train.calibration_records = records
    monkeypatch.setitem(sys.modules, "laya", fake)
    monkeypatch.setitem(sys.modules, "laya.train", fake.train)
    namespace["FINETUNE_CONTROL"] = {
        **laya_lane.finetune_control(), "save_each_epoch": False,
        "resume": False, "write_error_artifacts": True,
    }
    namespace["FINETUNE_DEV_ROWS"] = [{"state": "x"}]
    config = _Cfg()
    config.epochs = 1
    namespace["_perf_train_model"](
        _TinyNet(), types.SimpleNamespace(pad_token_id=0),
        [{"i": i, "label": i % 3} for i in range(4)], config,
        torch.device("cpu"), 8, 4)
    result = namespace["FINETUNE_CONTROL_RESULT"]
    assert "error_artifacts" not in result
    assert "Traceback (most recent call last)" in result["error_artifacts_error"]
    assert "invalid-logit" in result["error_artifacts_error"]


def test_host_dev_scoring_returns_the_shared_report() -> None:
    from test_laya_ddp import _make_fake_laya

    fake = _make_fake_laya([])
    report = laya_eval_fast.DevEvaluator.metrics(
        fake.train, torch.nn.Linear(2, 1), object(), [{"state": "x"}],
        torch.device("cpu"), 8, 4, False, laya_lane.finetune_control())
    assert report.accuracy == 0.9 and report.loss == 0.5
