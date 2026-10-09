"""Offline pins for the laya finetune torch.profiler integration (no GPU).

The profiler is DEFAULT-ON per owner but auto-disables without CUDA (CPU
profiling is near-useless and slows training); under DDP it profiles rank 0
only. The bounded schedule records a slice of an epoch, writes a chrome trace
per profiled epoch under `<output_dir>/<profile_dir>/epoch_<n>.json`, and feeds
a top-ops table to an optional sink. Every profiler path is fail-soft.

The session logic is the `ProfilerSession` class in `core.laya_controls`; these
tests exercise its public methods with a fake torch module (no GPU, no real
profiler).
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import types
from pathlib import Path

import pytest

from core import laya_controls
from core.laya_config import FinetuneSpec
from cli import laya_lane


def _device(kind):
    return types.SimpleNamespace(type=kind)


class _Event:
    key = "aten::matmul"
    self_cuda_time_total = 2500.0
    self_cpu_time_total = 100.0
    count = 7


class _FakeProfiler:
    """Minimal torch.profiler stand-in that fires on_trace_ready on step()."""

    def __init__(self, **kwargs):
        self._on_trace_ready = kwargs["on_trace_ready"]

    def export_chrome_trace(self, path):
        Path(path).write_text("{}", encoding="utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def step(self):
        self._on_trace_ready(self)

    def key_averages(self):
        return [_Event()]


class _FakeActivity:
    CPU = "cpu"
    CUDA = "cuda"


def _fake_torch(profile_cls=_FakeProfiler):
    import contextlib

    profiler = types.SimpleNamespace(
        ProfilerActivity=_FakeActivity,
        schedule=lambda **kwargs: kwargs,
        profile=profile_cls,
        record_function=lambda name: contextlib.nullcontext())
    return types.SimpleNamespace(profiler=profiler)


def _profiling_control():
    """The baked control with profiling forced ON (default is now OFF)."""
    return {**laya_lane.finetune_control(), "profile": True}


def test_profile_default_off_and_schedule_knobs():
    ft = FinetuneSpec()
    assert ft.profile is False
    assert ft.profile_dir == "profiler"
    assert ft.profile_schedule == {"wait": 1, "warmup": 1, "active": 1,
                                   "repeat": 1}
    control = laya_lane.finetune_control()
    assert control["profile"] is False
    assert control["profile_schedule"]["active"] == 1


def test_profile_schedule_rejects_unknown_keys_and_bad_dir():
    with pytest.raises(Exception):
        FinetuneSpec(profile_schedule={"wait": 1, "nope": 2})
    with pytest.raises(Exception):
        FinetuneSpec(profile_dir="../escape")


def test_profiler_session_enabled_only_for_cuda_rank0(monkeypatch):
    control = {"profile": True}
    assert laya_controls.ProfilerSession.enabled_for(
        control, _device("cpu"), True) is False
    assert laya_controls.ProfilerSession.enabled_for(
        {"profile": False}, _device("cuda"), True) is False
    assert laya_controls.ProfilerSession.enabled_for(
        control, _device("cuda"), False) is False
    assert laya_controls.ProfilerSession.enabled_for(
        control, _device("cuda"), True) is True


def test_profiler_session_for_training_resolves_trace_dir(tmp_path):
    session = laya_controls.ProfilerSession.for_training(
        _fake_torch(), _profiling_control(), _device("cuda"), True,
        str(tmp_path))
    assert session.enabled is True
    assert session._trace_dir == os.path.join(str(tmp_path), "profiler")
    # CPU auto-disables and never resolves a trace dir
    disabled = laya_controls.ProfilerSession.for_training(
        _fake_torch(), _profiling_control(), _device("cpu"), True,
        str(tmp_path))
    assert disabled.enabled is False


def test_profiler_session_writes_trace_and_feeds_sink(tmp_path):
    captured = []
    torch_module = _fake_torch()
    session = laya_controls.ProfilerSession.for_training(
        torch_module, _profiling_control(), _device("cuda"), True,
        str(tmp_path), on_metrics=lambda rows, epoch, path:
        captured.append((rows, epoch, path)))
    assert session.start() is True
    session.epoch = 3
    session.step()
    session.close()
    trace = tmp_path / "profiler" / "epoch_3.json"
    assert trace.is_file()
    assert captured and captured[0][1] == 3
    rows = captured[0][0]
    assert rows[0][0] == "aten::matmul" and rows[0][1] == 2.5
    # disabled session is a no-op
    off = laya_controls.ProfilerSession.for_training(
        torch_module, _profiling_control(), _device("cpu"), True,
        str(tmp_path))
    assert off.start() is False
    off.step()
    off.close()


def test_profiler_phase_gates_on_an_active_profiler(tmp_path):
    inactive = laya_controls.ProfilerSession.for_training(
        _fake_torch(), _profiling_control(), _device("cpu"), True, str(tmp_path))
    with inactive.phase("forward"):   # no record_function emit on CPU
        pass

    active = laya_controls.ProfilerSession.for_training(
        _fake_torch(), _profiling_control(), _device("cuda"), True, str(tmp_path))
    assert active.start() is True
    calls = []
    import contextlib
    active._torch.profiler.record_function = \
        lambda name: calls.append(name) or contextlib.nullcontext()
    with active.phase("forward"):
        pass
    assert calls == ["forward"]
    active.close()


def test_disabled_profiler_still_emits_phase_timings(tmp_path, monkeypatch,
                                                     capsys):
    """P1-A: wall-clock phases are recorded even when the torch profiler is
    off (now the default); flush_epoch streams them and merges the JSON."""
    out = tmp_path / "timings.json"
    log = tmp_path / "timings.log"
    monkeypatch.setenv("ER_TIMING_OUT", str(out))
    monkeypatch.setenv("ER_TIMING_LOG", str(log))
    session = laya_controls.ProfilerSession.for_training(
        _fake_torch(), _profiling_control(), _device("cpu"), True, str(tmp_path))
    with session.phase("forward"):
        pass
    session.flush_epoch(1)
    line = "[timing] laya forward epoch=1 elapsed_seconds="
    assert line in capsys.readouterr().out
    assert line in log.read_text(encoding="utf-8")
    document = json.loads(out.read_text(encoding="utf-8"))
    assert document["components"]["laya"]["label"] == "laya"


def test_profiler_session_is_fail_soft(tmp_path):
    class _Broken(_FakeProfiler):
        def export_chrome_trace(self, path):
            raise RuntimeError("boom")

    session = laya_controls.ProfilerSession.for_training(
        _fake_torch(_Broken), _profiling_control(), _device("cuda"),
        True, str(tmp_path))
    assert session.start() is True
    session.step()   # trace handler error must not raise
    session.close()

    class _NoEnter(_FakeProfiler):
        def __enter__(self):
            raise RuntimeError("no profiler")

    failing = laya_controls.ProfilerSession.for_training(
        _fake_torch(_NoEnter), _profiling_control(), _device("cuda"),
        True, str(tmp_path))
    assert failing.start() is False


def test_rendered_kernel_annotates_phases_and_schedules_profiler():
    spec = laya_lane._spec()
    values = {
        "LAYA_PACKAGE": laya_lane.FINETUNE_LAYA_PACKAGE,
        "BASE_MODEL_ARCHIVE": spec.base_model_archive,
        "BASE_MODEL_DIR": spec.base_model_dir,
        "RUN_TAG": "gpu_test",
        "TRAIN_JSONL": "train.jsonl",
        "DEV_JSONL": "dev.jsonl",
        "TEST_JSONL": "test.jsonl",
        "FINETUNE_CONFIG": repr(laya_lane.finetune_config()),
        "FINETUNE_CONTROL": repr(laya_lane.finetune_control()),
        "FINETUNE_DEVICE": "auto",
        "HELD_OUT_BATCH": "8",
        "RECEIPT_NAME": "laya_finetune.receipt.json",
        "WANDB_API_KEY": "",
        "WANDB_PROJECT": "e-r",
        "REPOSITORY": "anomalyco/er",
        "BRANCH": "main",
        "REVISION": "deadbeef",
        "DEVICE_PATCH": laya_lane.FINETUNE_DEVICE_PATCH_SOURCE,
        "PERF_PATCH": laya_lane.FINETUNE_PERF_PATCH_SOURCE,
    }
    preflight = laya_lane._template(laya_lane.FINETUNE_RUNTIME_PREFLIGHT, values)
    script = laya_lane._template(
        laya_lane.FINETUNE_KERNEL_SCRIPT,
        {**values, "RUNTIME_PREFLIGHT": preflight})
    laya_lane._kernel_script_gate(script)
    laya_lane._module_scope_gate(script)
    assert not re.search(r"@[A-Z][A-Z0-9_]*@", script)
    for phase in ("data.encode", "collate+batch_move", "forward", "loss",
                  "backward", "optimizer_step", "dev_eval",
                  "checkpoint_save"):
        assert 'phase("%s")' % phase in script, phase
    # calibration is annotated inside DevEvaluator (gated via `annotate`)
    assert 'annotate("calibration")' in script
    assert "class ProfilerSession" in script
    assert "def phase(self, name)" in script
    assert "torch.profiler.profile(" in script
    assert "torch.profiler.schedule(" in script
    assert "profile_memory=True" in script
    assert "with_stack=False" in script
    assert "record_shapes=False" in script
    assert "export_chrome_trace" in script
    assert "key_averages()" in script
    # P1 instrumentation surfaces: one per-epoch flush, pre-train calibration
    # timing and the previously-unlogged training knobs.
    assert "flush_epoch(epoch + 1)" in script
    assert 'phase("calibration_records")' in script
    assert "amp_dtype=%s" in script and "grad_accum" in script
