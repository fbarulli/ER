"""Offline pins for the two lane fixes.

Fix 1 (embed kernel slug) is report-only: no embed kernel exists on the
account, so there is nothing to repoint — pinned in the fix note, not here.

Fix 2 (laya finetune CUDA device mismatch): `laya<=0.4.0` `finetune()`
evaluates the base checkpoint through `calibration_records()` BEFORE
`train_model()` calls `model.to(device)`, so `load_checkpoint()`'s CPU model
meets cuda `input_ids` and `index_select` raises "index is on cuda:0,
different from other tensors on cpu". The finetune kernel now injects a
runtime device patch (`FINETUNE_DEVICE_PATCH_SOURCE`); these tests pin the
patch function and its embedding in the staged payload. Live T4 verification
still needs a kernel run (this box has no cuda).
"""
from __future__ import annotations

import re
import sys
import types

import torch
import torch.nn as nn

from cli import laya_lane


class _StandIn(nn.Module):
    """Tiny model with params, nested modules and a non-persistent buffer."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(5, 4)
        self.head = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        self.register_buffer("temperature", torch.ones(3))
        self.register_buffer("rope", torch.arange(4), persistent=False)


def _patch_namespace():
    namespace: dict = {}
    exec(laya_lane.FINETUNE_DEVICE_PATCH_SOURCE, namespace)
    return namespace


def _render_finetune_script() -> str:
    """Render the finetune kernel the way stage_finetune_kernel does.

    The trainer surface is config-driven: the FULL ``laya.train.TrainConfig``
    kwargs come from ``finetune_config()`` and are baked as one repr literal
    (``FINETUNE_CONFIG``); the device AND perf monkeypatches ride their own
    ``@DEVICE_PATCH@`` / ``@PERF_PATCH@`` markers.
    """
    values = {
        "LAYA_PACKAGE": laya_lane.FINETUNE_LAYA_PACKAGE,
        "BASE_MODEL_ARCHIVE": "convaiinnovations-laya.tar.zst",
        "BASE_MODEL_DIR": "convaiinnovations-laya",
        "RUN_TAG": "gpu_test",
        "TRAIN_JSONL": "train.jsonl",
        "DEV_JSONL": "dev.jsonl",
        "TEST_JSONL": "test.jsonl",
        "FINETUNE_CONFIG": repr(laya_lane.finetune_config()),
        "FINETUNE_DEVICE": "auto",
        "HELD_OUT_BATCH": str(laya_lane._spec().laya_decision_batch_size),
        "REPOSITORY": "anomalyco/er",
        "BRANCH": "main",
        "REVISION": "deadbeef",
        # The ONE structural size helper, injected by the same staging
        # helper the kernel stager uses (never hand-copied: it cannot drift).
        "SIZE_OF": laya_lane._file_bytes_of_source(),
        "DEVICE_PATCH": laya_lane.FINETUNE_DEVICE_PATCH_SOURCE,
        "PERF_PATCH": laya_lane.FINETUNE_PERF_PATCH_SOURCE,
        "SESSION_REPORT": laya_lane._session_report_helper(),
    }
    preflight = laya_lane._template(laya_lane.FINETUNE_RUNTIME_PREFLIGHT, values)
    return laya_lane._template(
        laya_lane.FINETUNE_KERNEL_SCRIPT,
        {**values, "RUNTIME_PREFLIGHT": preflight})


def test_force_model_to_device_moves_every_module_and_buffer():
    """The patch moves every param and every registered buffer — including a
    non-persistent one — onto the target device (meta proves the same code
    path that runs on cuda)."""
    force_model_to_device = _patch_namespace()["force_model_to_device"]
    model = _StandIn()
    assert all(p.device.type == "cpu" for p in model.parameters())
    assert all(b.device.type == "cpu" for b in model.buffers())
    assert "rope" in model._non_persistent_buffers_set

    force_model_to_device(model, "meta")

    assert list(model.parameters())
    assert list(model.buffers())
    assert all(p.device.type == "meta" for p in model.parameters())
    assert all(b.device.type == "meta" for b in model.buffers())
    assert all(m.weight.device.type == "meta"
               for m in model.modules() if getattr(m, "weight", None) is not None)


def test_apply_device_patch_wraps_both_forward_entrypoints(monkeypatch):
    """`apply_device_patch` wraps `laya.train.calibration_records` (the actual
    crash: pre-training base eval) and `train_model`, moving the model before
    either forward runs, and still calls the originals."""
    calls = {"calibration": 0, "train": 0}

    fake_train = types.ModuleType("laya.train")

    def calibration_records(model, tok, items, device, *args, **kwargs):
        calls["calibration"] += 1
        return "calibration-result"

    def train_model(model, tok, items, config, device, *args, **kwargs):
        calls["train"] += 1
        return "train-result"

    fake_train.calibration_records = calibration_records
    fake_train.train_model = train_model

    fake_laya = types.ModuleType("laya")
    fake_laya.train = fake_train

    monkeypatch.setitem(sys.modules, "laya", fake_laya)
    monkeypatch.setitem(sys.modules, "laya.train", fake_train)

    apply_device_patch = _patch_namespace()["apply_device_patch"]
    apply_device_patch()

    assert fake_train.calibration_records is not calibration_records
    assert fake_train.train_model is not train_model

    model = _StandIn()
    assert fake_train.calibration_records(model, None, [], "meta") == \
        "calibration-result"
    assert calls["calibration"] == 1
    assert all(p.device.type == "meta" for p in model.parameters())
    assert all(b.device.type == "meta" for b in model.buffers())

    model2 = _StandIn()
    assert fake_train.train_model(model2, None, [], None, "meta") == "train-result"
    assert calls["train"] == 1
    assert all(p.device.type == "meta" for p in model2.parameters())


def test_finetune_payload_embeds_patch_and_passes_gates():
    """The staged finetune payload carries both monkeypatches, routes the real
    trainer in-process, and still clears both staging-time AST gates."""
    script = _render_finetune_script()
    laya_lane._kernel_script_gate(script)
    laya_lane._module_scope_gate(script)
    # device patch: both forward entrypoints are wrapped by name
    assert "def force_model_to_device" in script
    assert "def apply_device_patch" in script
    assert "apply_device_patch()" in script
    # perf patch: movement/redundancy wrapper + its opt-out env flag
    assert "def apply_perf_patch" in script
    assert "apply_perf_patch()" in script
    assert 'PERF_PATCH_ENV = "ER_LAYA_PERF_PATCH"' in script
    # both template markers were substituted
    assert "@DEVICE_PATCH@" not in script
    assert "@PERF_PATCH@" not in script
    # and no marker of ANY kind survives: a value missing from the render would
    # leave `@SOMETHING@` in the staged kernel unnoticed
    assert not re.search(r"@[A-Z][A-Z0-9_]*@", script)
    # config-driven kernel: the FULL TrainConfig is baked and constructed
    # in-process (the laya-train CLI subprocess is gone)
    assert "def run_laya_finetune" in script
    assert "TrainConfig(**FINETUNE_CONFIG" in script
    assert "laya_train.finetune(" in script
    assert "train_cli" not in script
    assert 'FINETUNE_DEVICE = "auto"' in script
