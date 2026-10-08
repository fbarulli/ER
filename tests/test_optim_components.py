"""Public-API, GPU-free tests for src/training/optim_components.py.

Each class is exercised through its public surface (constructors, apply/scale/
factor/penalty) so the wiring call sites can rely on the contract without a GPU.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from training import optim_components as oc


# ── 1. no-decay param groups ────────────────────────────────────────────────

def test_no_decay_splits_bias_and_norm():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = nn.Linear(4, 4)
            self.norm = nn.LayerNorm(4)

    model = M()
    names = {id(p): n for n, p in model.named_parameters()}
    groups = oc.NoDecayParamGroups(weight_decay=0.1, enabled=True).apply(
        [{"params": list(model.parameters()), "lr": 1e-3}], names
    )
    by_wd = {}
    for group in groups:
        for param in group["params"]:
            by_wd[names[id(param)]] = group["weight_decay"]
    assert by_wd["dense.weight"] == 0.1
    assert by_wd["dense.bias"] == 0.0
    assert by_wd["norm.weight"] == 0.0
    assert by_wd["norm.bias"] == 0.0
    assert all(group["lr"] == 1e-3 for group in groups)


def test_no_decay_disabled_keeps_single_decay():
    param = nn.Parameter(torch.zeros(1))
    names = {id(param): "x.bias"}
    groups = oc.NoDecayParamGroups(weight_decay=0.2, enabled=False).apply(
        [{"params": [param], "lr": 1.0}], names
    )
    assert groups[0]["weight_decay"] == 0.2
    assert groups[0]["params"] == [param]


# ── 2. optimizer state precision ────────────────────────────────────────────

def test_state_precision_casts_moments_only():
    param = nn.Parameter(torch.randn(4))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    loss = param.sum()
    loss.backward()
    optimizer.step()
    prec = oc.OptimizerStatePrecision("bf16")
    assert prec.enabled
    cast = prec.cast_(optimizer)
    assert cast >= 2
    state = optimizer.state[param]
    assert state["exp_avg"].dtype == torch.bfloat16
    assert state["exp_avg_sq"].dtype == torch.bfloat16
    assert not torch.is_tensor(state["step"]) or state["step"].dtype != torch.bfloat16


def test_state_precision_fp32_is_noop():
    param = nn.Parameter(torch.randn(2))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    optimizer.state[param] = {"step": 0, "exp_avg": torch.zeros(2), "exp_avg_sq": torch.zeros(2)}
    prec = oc.OptimizerStatePrecision("fp32")
    assert prec.enabled is False
    assert prec.cast_(optimizer) == 0
    assert optimizer.state[param]["exp_avg"].dtype == torch.float32


def test_state_precision_rejects_unknown_dtype():
    with pytest.raises(oc.ComponentConfigError):
        oc.OptimizerStatePrecision("fp8")


# ── 3. LR scaling rule ──────────────────────────────────────────────────────

def test_lr_scaling_effective_batch_and_linear():
    rule = oc.LrScalingRule("linear", base_batch=32)
    assert rule.effective_batch(8, 2, 2) == 32
    assert rule.factor(32) == pytest.approx(1.0)
    assert rule.peak_lr(1e-3, micro_batch=8, grad_accum=2, world_size=1) == pytest.approx(5e-4)


def test_lr_scaling_sqrt_and_none_precedence():
    sqrt_rule = oc.LrScalingRule("sqrt", base_batch=16)
    assert sqrt_rule.factor(64) == pytest.approx(2.0)
    none_rule = oc.LrScalingRule("none")
    # explicit LR wins verbatim under none
    assert none_rule.peak_lr(7e-4, micro_batch=64, grad_accum=8, world_size=4) == 7e-4


def test_lr_scaling_requires_base_batch():
    with pytest.raises(oc.ComponentConfigError):
        oc.LrScalingRule("linear")
    with pytest.raises(oc.ComponentConfigError):
        oc.LrScalingRule("bogus", base_batch=1)


# ── 4. R-Drop ───────────────────────────────────────────────────────────────

def test_r_drop_disabled_is_zero():
    reg = oc.RDropRegularizer(0.0)
    assert reg.enabled is False
    a = torch.randn(4)
    assert float(reg.penalty(a, a + 5.0)) == 0.0


def test_r_drop_zero_for_identical_and_positive_for_divergent():
    reg = oc.RDropRegularizer(0.5)
    a = torch.randn(8)
    assert float(reg.penalty(a, a)) == pytest.approx(0.0, abs=1e-6)
    assert float(reg.penalty(a, a + 3.0)) > 0.0


def test_r_drop_rejects_negative_alpha():
    with pytest.raises(oc.ComponentConfigError):
        oc.RDropRegularizer(-0.1)


# ── 5. drop path ────────────────────────────────────────────────────────────

def test_drop_path_eval_is_identity():
    module = oc.DropPath(0.5).eval()
    x = torch.randn(4, 3)
    assert torch.equal(module(x), x)


def test_drop_path_train_scales_and_zero_rate_noop():
    torch.manual_seed(0)
    module = oc.DropPath(0.5).train()
    x = torch.ones(1000, 1)
    out = module(x)
    # surviving units are scaled by 1/keep_prob so E[out] ~= x
    assert out.mean() == pytest.approx(1.0, abs=0.1)
    assert set(out.flatten().tolist()) <= {0.0, 2.0}
    assert torch.equal(oc.DropPath(0.0).train()(x), x)


def test_drop_path_rate_schedule():
    assert oc.drop_path_rate(0.2, 1, 3, schedule="linear") == pytest.approx(0.0)
    assert oc.drop_path_rate(0.2, 3, 3, schedule="linear") == pytest.approx(0.2)
    assert oc.drop_path_rate(0.2, 2, 3, schedule="constant") == pytest.approx(0.2)
    with pytest.raises(oc.ComponentConfigError):
        oc.drop_path_rate(0.2, 1, 3, schedule="zigzag")


def test_apply_drop_path_injects_and_schedules():
    class Branch(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = nn.Dropout(0.0)

        def forward(self, x):
            return x

    branches = [Branch(), Branch()]
    set_epoch = oc.apply_drop_path(branches, lambda e: 0.1 * e)
    set_epoch(3)
    assert all(b.dropout.drop_path.rate == pytest.approx(0.3) for b in branches)
    # eval mode: injected wrapper is an identity over the original dropout
    x = torch.randn(3, 2)
    assert torch.equal(branches[0].eval()(x), x)


# ── 6. dynamic padding ──────────────────────────────────────────────────────

def test_dynamic_pad_width_rounds_to_multiple():
    assert oc.dynamic_pad_width([3, 7, 5]) == 7
    assert oc.dynamic_pad_width([3, 7, 5], pad_to_multiple=8) == 8
    assert oc.dynamic_pad_width([9]) == 9
    assert oc.dynamic_pad_width([9], pad_to_multiple=8) == 16
    with pytest.raises(oc.ComponentConfigError):
        oc.dynamic_pad_width([])
    with pytest.raises(oc.ComponentConfigError):
        oc.dynamic_pad_width([1], pad_to_multiple=0)


# ── 7. batch-size ramp ──────────────────────────────────────────────────────

def test_batch_size_ramp_factors_and_grad_accum():
    ramp = oc.BatchSizeRamp(enabled=True, start_frac=0.25, ramp_epochs=3)
    assert ramp.factor(1) == pytest.approx(0.25)
    assert ramp.factor(3) == pytest.approx(1.0)
    assert ramp.grad_accum(1, 8) == 2
    assert ramp.grad_accum(3, 8) == 8
    disabled = oc.BatchSizeRamp(enabled=False, start_frac=0.25, ramp_epochs=3)
    assert disabled.grad_accum(1, 8) == 8
    with pytest.raises(oc.ComponentConfigError):
        oc.BatchSizeRamp(enabled=True, start_frac=0.0, ramp_epochs=3)


# ── 8/9. loss-weight schedules ──────────────────────────────────────────────

def test_loss_weight_schedule_ramps_selected_terms_only():
    schedule = oc.LossWeightSchedule(
        enabled=True, warmup_epochs=3, schedule="linear",
        uniformity=True, margin=False,
    )
    assert schedule.scheduled("uniformity", 0.05, 1) == pytest.approx(0.0)
    assert schedule.scheduled("uniformity", 0.05, 3) == pytest.approx(0.05)
    # margin term not selected -> base value unchanged at every epoch
    assert schedule.scheduled("margin", 0.1, 1) == pytest.approx(0.1)


def test_loss_weight_schedule_cosine_and_disabled():
    cosine = oc.LossWeightSchedule(
        enabled=True, warmup_epochs=3, schedule="cosine", uniformity=True
    )
    assert cosine.scheduled("uniformity", 1.0, 1) == pytest.approx(0.0)
    assert cosine.scheduled("uniformity", 1.0, 3) == pytest.approx(1.0)
    disabled = oc.LossWeightSchedule(
        enabled=False, warmup_epochs=2, schedule="linear", uniformity=True
    )
    assert disabled.scheduled("uniformity", 0.7, 1) == pytest.approx(0.7)
    with pytest.raises(oc.ComponentConfigError):
        disabled.scheduled("bogus", 1.0, 1)
