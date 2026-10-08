"""Offline pins for the laya finetune torch.profiler integration (no GPU).

The profiler is DEFAULT-ON per owner but auto-disables without CUDA (CPU
profiling is near-useless and slows training); under DDP it profiles rank 0
only. The bounded schedule records only a slice of an epoch, writes a chrome
trace per profiled epoch under `<output_dir>/<profile_dir>/epoch_<n>.json`,
and prints/logs a top-ops table. Every profiler path is fail-soft.
"""
from __future__ import annotations

import math
import os
import random
import re
import types
from pathlib import Path

import torch

from core import laya_config
from cli import laya_lane


def _perf_namespace():
    namespace = {"os": os, "math": math, "random": random}
    exec(laya_lane.FINETUNE_PERF_PATCH_SOURCE, namespace)
    return namespace


def _device(kind):
    return types.SimpleNamespace(type=kind)


def test_profile_default_on_and_schedule_knobs():
    ft = laya_config.FinetuneSpec()
    assert ft.profile is True
    assert ft.profile_dir == "profiler"
    assert ft.profile_schedule == {"wait": 1, "warmup": 1, "active": 1,
                                   "repeat": 1}
    control = laya_lane.finetune_control()
    assert control["profile"] is True
    assert control["profile_schedule"]["active"] == 1


def test_profile_schedule_rejects_unknown_keys_and_bad_dir():
    from pydantic import ValidationError

    with __import__("pytest").raises(ValidationError):
        laya_config.FinetuneSpec(profile_schedule={"wait": 1, "nope": 2})
    with __import__("pytest").raises(ValidationError):
        laya_config.FinetuneSpec(profile_dir="../escape")


def test_profile_enabled_gates_cuda_and_rank(monkeypatch):
    namespace = _perf_namespace()
    enabled = namespace["_profile_enabled"]
    # off when the knob is off
    assert enabled({"profile": False}, _device("cuda")) is False
    # off on CPU (near-useless + slows training)
    assert enabled({"profile": True}, _device("cpu")) is False
    # on for cuda rank 0
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert enabled({"profile": True}, _device("cuda")) is True
    # off for a non-zero DDP rank
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    assert enabled({"profile": True}, _device("cuda")) is False


def test_profile_trace_and_top_ops_are_fail_soft(tmp_path):
    namespace = _perf_namespace()
    namespace["WANDB_RUN"] = None
    namespace["FINETUNE_OUTPUT_DIR"] = str(tmp_path)

    class _Event:
        key = "aten::matmul"
        self_cuda_time_total = 2500.0
        self_cpu_time_total = 100.0
        count = 7

    class _Profiler:
        def export_chrome_trace(self, path):
            Path(path).write_text("{}", encoding="utf-8")

        def key_averages(self):
            return [_Event()]

    namespace["_profile_top_ops"](
        torch, _Profiler(), {"epoch": 3}, {"profile_dir": "profiler"})
    assert (tmp_path / "profiler" / "epoch_3.json").is_file()

    class _Broken:
        def export_chrome_trace(self, path):
            raise RuntimeError("boom")

        def key_averages(self):
            raise RuntimeError("boom")

    # a profiler error must never raise out of the handler
    namespace["_profile_top_ops"](
        torch, _Broken(), {"epoch": 4}, {"profile_dir": "profiler"})


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
                  "checkpoint_save", "calibration"):
        assert 'record_function("%s")' % phase in script, phase
    assert "torch.profiler.profile(" in script
    assert "torch.profiler.schedule(" in script
    assert "profile_memory=True" in script
    assert "with_stack=False" in script
    assert "record_shapes=False" in script
    assert 'export_chrome_trace' in script
    assert "key_averages()" in script
