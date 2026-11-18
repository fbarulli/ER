"""Public-API pins for the extended laya finetune knobs (GPU-free).

All default-OFF and SSOT-driven (FinetuneSpec -> FINETUNE_CONTROL); each has a
class with a single responsibility in `core.laya_controls`, injected verbatim
into the staged perf patch. These tests exercise the PUBLIC class methods.
"""
from __future__ import annotations

import math
import os
import random
import types

import pytest
import torch

from core import laya_controls
from core.laya_config import FinetuneSpec, LayaSpec
from cli import laya_lane

NEW_FIELDS = (
    "no_decay_bias_norm", "optim_state_dtype", "lr_scaling", "base_batch",
    "r_drop", "r_drop_alpha", "drop_path", "drop_path_rate",
    "drop_path_schedule", "dynamic_padding", "pad_to_multiple",
    "batch_size_ramp", "batch_ramp_start_frac", "batch_ramp_epochs",
    "loss_schedule", "w_sph_end", "w_rps_end", "contrastive_margin",
    "contrastive_margin_end",
)


def _perf_namespace():
    namespace = {"os": os, "math": math, "random": random}
    exec(laya_lane.FINETUNE_PERF_PATCH_SOURCE, namespace)
    return namespace


# ── schema / bake ──────────────────────────────────────────────────────────
def test_extended_knobs_are_default_off_and_baked():
    ft = FinetuneSpec()
    assert ft.no_decay_bias_norm is False
    assert ft.optim_state_dtype == "fp32"
    assert ft.lr_scaling == "none" and ft.base_batch == 0
    assert ft.r_drop is False and ft.drop_path is False
    assert ft.dynamic_padding is False and ft.batch_size_ramp is False
    assert ft.loss_schedule == "laya"
    assert ft.contrastive_margin == 0.0
    assert set(NEW_FIELDS).issubset(set(laya_lane.FINETUNE_CONTROL_FIELDS))
    control = laya_lane.finetune_control()
    assert set(NEW_FIELDS).issubset(control)
    # still never a TrainConfig kwarg
    assert set(NEW_FIELDS).isdisjoint(laya_lane.FINETUNE_CONFIG_FIELDS)
    # YAML accepts every key
    LayaSpec(finetune={name: FinetuneSpec().model_dump()[name]
                       for name in NEW_FIELDS})


# ── no_decay ───────────────────────────────────────────────────────────────
def test_param_group_builder_splits_bias_and_norm():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
    groups = [{"params": list(model.parameters()), "lr": 0.1}]
    split = laya_controls.ParamGroupBuilder.apply(model, groups, True)
    no_decay = [g for g in split if g.get("weight_decay") == 0.0]
    covered = [p for g in split for p in g["params"]]
    assert len(covered) == len(list(model.parameters()))
    # every no-decay param is bias/norm (ndim<=1)
    assert all(p.ndim <= 1 for g in no_decay for p in g["params"])
    # disabled -> untouched passthrough
    assert laya_controls.ParamGroupBuilder.apply(model, groups, False) is groups


# ── optimizer state precision (public: optimizer builds + steps) ───────────
@pytest.mark.parametrize("param_dtype, dial", [
    (torch.float32, "bf16"),    # AMP master weights: bf16 state broke fused
    (torch.float16, "fp32"),
])
def test_optimizer_step_keeps_state_in_param_dtype(param_dtype, dial):
    """Native AdamW requires state.dtype == param.dtype (single/foreach/fused);
    the requested optim_state_dtype must never leave a mismatched state that
    makes the next optimizer.step() raise."""
    namespace = _perf_namespace()
    model = torch.nn.Linear(4, 4).to(param_dtype)
    config = types.SimpleNamespace(weight_decay=0.0, head_lr=1e-3,
                                   encoder_lr=1e-3)
    control = {"optimizer": "adamw", "optim_state_dtype": dial,
               "no_decay_bias_norm": False}
    optimizer = namespace["TrainingOptimizer"].make(
        torch, model, [{"params": list(model.parameters()), "lr": 1e-3}],
        config, control)
    for _ in range(2):
        for param in model.parameters():
            param.grad = torch.zeros_like(param)
        optimizer.step()
        namespace["OptimizerStateCaster"].apply(torch, optimizer, dial)
        for param in model.parameters():
            assert optimizer.state[param]["exp_avg"].dtype == param.dtype


# ── lr scaling ─────────────────────────────────────────────────────────────
def test_lr_scaler_rules_and_precedence():
    factor = laya_controls.LrScaler.factor
    assert factor(64, 32, "linear") == 2.0
    assert factor(64, 32, "sqrt") == pytest.approx(math.sqrt(2))
    assert factor(64, 32, "none") == 1.0       # explicit LRs win
    assert factor(64, 0, "linear") == 1.0      # unset base -> no scaling
    assert laya_controls.LrScaler.effective_batch(4, 8, 2) == 64
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    laya_controls.LrScaler.apply(optimizer, 2.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.2)


# ── batch-size ramp ────────────────────────────────────────────────────────
def test_batch_ramp_grad_accum():
    ramp = laya_controls.BatchRamp.grad_accum
    assert ramp(0, 8, 0.25, 2) == 5
    assert ramp(1, 8, 0.25, 2) == 8
    assert ramp(5, 8, 0.25, 2) == 8
    assert ramp(0, 8, 0.25, 0) == 8            # disabled -> target


# ── loss-weight schedule (ONE mechanism) ───────────────────────────────────
def _schedule(**overrides):
    control = laya_lane.finetune_control()
    control.update(overrides)
    return laya_controls.LossWeightSchedule.from_control(
        control, FinetuneSpec())


def test_loss_schedule_laya_mode_reproduces_sigma_at():
    schedule = _schedule()
    for epoch in range(8):
        expected = 0.4 + (0.1 - 0.4) * (epoch / 7)
        assert schedule.at(epoch)["sigma"] == pytest.approx(expected)
        assert schedule.at(epoch)["w_sph"] == 0.75
        assert schedule.at(epoch)["w_rps"] == 1.0


def test_loss_schedule_linear_and_cosine_interpolate_every_term():
    linear = _schedule(loss_schedule="linear", w_sph_end=0.5,
                       w_rps_end=0.0, contrastive_margin=0.0,
                       contrastive_margin_end=1.0)
    at_last = linear.at(7)
    assert at_last["w_sph"] == pytest.approx(0.5)
    assert at_last["w_rps"] == pytest.approx(0.0)
    assert at_last["margin"] == pytest.approx(1.0)
    cosine = _schedule(loss_schedule="cosine", w_sph_end=0.0)
    assert cosine.at(3)["w_sph"] != linear.at(3)["w_sph"]


# ── R-Drop ─────────────────────────────────────────────────────────────────
def test_r_drop_requires_dropout_and_combines():
    class _WithDropout(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.drop = torch.nn.Dropout(0.2)

    class _NoDropout(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.drop = torch.nn.Dropout(0.0)

    assert laya_controls.RDrop.available(_WithDropout()) is True
    assert laya_controls.RDrop.available(_NoDropout()) is False
    logits = torch.zeros(2, 3)
    mask = torch.ones(2, 3, dtype=torch.bool)
    assert laya_controls.RDrop.kl(torch, logits, logits, mask) == 0.0
    combined = laya_controls.RDrop.combine(
        torch.tensor(1.0), torch.tensor(3.0), torch.tensor(0.5), 0.4)
    assert combined == pytest.approx(0.5 * (1.0 + 3.0) + 0.4 * 0.5)


# ── drop path ──────────────────────────────────────────────────────────────
def test_drop_path_rate_and_apply():
    rate = laya_controls.DropPath.rate
    assert rate(0, 4, 0.4, "linear") == pytest.approx(0.1)
    assert rate(3, 4, 0.4, "linear") == pytest.approx(0.4)
    assert rate(0, 4, 0.4, "constant") == pytest.approx(0.4)
    assert rate(0, 4, 0.0, "linear") == 0.0

    class _Block(torch.nn.Module):
        def forward(self, x):
            return x + 1.0

    block = _Block().train()
    assert laya_controls.DropPath.apply(block, 0.0, blocks=[block]) == 0
    assert laya_controls.DropPath.apply(block, 1.0, blocks=[block]) == 1
    x = torch.zeros(4)
    out = block(x)          # rate 1.0 -> residual dropped -> x
    assert torch.allclose(out, x)
    block.eval()
    assert torch.allclose(block(x), x + 1.0)   # eval never drops


# ── dynamic padding ────────────────────────────────────────────────────────
def test_dynamic_padder_rounds_to_multiple():
    batch = {
        "input_ids": torch.zeros(2, 5, dtype=torch.long),
        "attention_mask": torch.ones(2, 5, dtype=torch.long),
        "marker_mask": torch.ones(2, 3, dtype=torch.bool),
    }
    padded = laya_controls.DynamicPadder.pad(torch, batch, 4)
    assert padded["input_ids"].shape == (2, 8)
    assert padded["attention_mask"].shape == (2, 8)
    assert padded["marker_mask"].shape == (2, 3)   # option axis untouched
    assert int(padded["attention_mask"][0, 5:].sum()) == 0
    # multiple 0/1 -> passthrough
    assert laya_controls.DynamicPadder.pad(torch, batch, 0) is batch
    assert laya_controls.DynamicPadder.pad(torch, batch, 1) is batch


# ── contrastive margin (via the injected LossBuilder) ──────────────────────
def test_loss_builder_margin_changes_soft_ce_only_when_positive():
    namespace = _perf_namespace()
    config = types.SimpleNamespace(loss="soft-ce", rl_samples=4)
    fake = types.SimpleNamespace(
        soft_ce_loss=lambda logits, target, mask: logits.sum(),
        rlcd_loss=lambda *a, **k: torch.tensor(0.0))
    logits = torch.zeros(1, 3)
    target = torch.tensor([[1.0, 0.0, 0.0]])
    mask = torch.ones(1, 3)
    compute = namespace["LossBuilder"].compute
    assert compute(torch, fake, config, logits, target, mask, None, 0.4,
                   0.75, 1.0, None, 0.0) == 0.0
    assert compute(torch, fake, config, logits, target, mask, None, 0.4,
                   0.75, 1.0, None, 1.0) == pytest.approx(-2.0)
