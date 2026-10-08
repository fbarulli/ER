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
    recipe = laya_lane.FINETUNE_RECIPE
    values = {
        "LAYA_PACKAGE": laya_lane.FINETUNE_LAYA_PACKAGE,
        "BASE_MODEL": "convaiinnovations/laya",
        "RUN_TAG": "gpu_test",
        "TRAIN_JSONL": "train.jsonl",
        "DEV_JSONL": "dev.jsonl",
        "TEST_JSONL": "test.jsonl",
        "EPOCHS": str(recipe["epochs"]),
        "MICRO_BATCH": str(recipe["micro_batch"]),
        "GRAD_ACCUM": str(recipe["grad_accum"]),
        "ENCODER_LR": repr(recipe["encoder_lr"]),
        "HEAD_LR": repr(recipe["head_lr"]),
        "LOSS": recipe["loss"],
        "SEED": str(recipe["seed"]),
        "REPOSITORY": "anomalyco/er",
        "BRANCH": "main",
        "REVISION": "deadbeef",
        "DEVICE_PATCH": laya_lane.FINETUNE_DEVICE_PATCH_SOURCE,
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
    """The staged finetune payload carries the patch, routes the real CLI
    through it in-process, and still clears both staging-time AST gates."""
    script = _render_finetune_script()
    laya_lane._kernel_script_gate(script)
    laya_lane._module_scope_gate(script)
    assert "def force_model_to_device" in script
    assert "def apply_device_patch" in script
    assert "def run_laya_train" in script
    assert "apply_device_patch()" in script
    assert "train_cli.main(arguments)" in script
    assert "@DEVICE_PATCH@" not in script
    # flags/recipe unchanged
    assert "--device" in script and "str(SEED)" in script
