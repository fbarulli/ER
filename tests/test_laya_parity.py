"""Scoped numerical-parity + wiring pins for the laya speed levers.

Owner contract: the optimizations must PRESERVE numerics — loss/accuracy match
the unoptimized baseline within tolerance. These pins exercise the PUBLIC
injected classes (exec'd from the staged perf patch, the repo's convention) on
small deterministic modules, one focused test per behaviour:

  * gradient_checkpointing on==off is numerically EXACT (deterministic
    recompute + RNG preservation) -> justifies the checkpointing-OFF default;
  * torch.compile (ModelCompiler) stays within fusion tolerance of eager and
    fails SOFT to eager on a compiler error (never a silent recipe downshift);
  * ControlCheckpointer.unwrap strips the compile (_orig_mod) and DDP (.module)
    wrappers so a persisted state_dict has no wrapper prefix;
  * the ``compile_model`` control actually reaches ModelCompiler inside the
    training loop (dial -> compile wiring, spy-patched so no slow Inductor).

The END-TO-END numeric parity on the REAL 322M-param checkpoint (dev
accuracy/loss within tolerance) is confirmed by the T4 finetune-smoke, not
here — a small synthetic module cannot reproduce laya's fusion drift.
"""
from __future__ import annotations

import math
import os
import random
import types

import torch

from cli import laya_lane

SEED = 1729


def _perf_namespace():
    namespace = {"os": os, "math": math, "random": random}
    exec(laya_lane.FINETUNE_PERF_PATCH_SOURCE, namespace)
    return namespace


# ── gradient checkpointing is numerically exact (the backward lever) ────────
def test_gradient_checkpointing_recompute_is_numerically_exact():
    """Checkpointing OFF must not move the numbers: the recompute is
    deterministic and preserves the dropout RNG, so grads match bit-for-bit.
    This is the invariant that makes the checkpointing-OFF default safe."""

    class _CkptNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head_checkpointing = False
            self.layer = torch.nn.TransformerEncoderLayer(
                8, 2, 16, dropout=0.1, batch_first=True)
            self.out = torch.nn.Linear(8, 3)

        def forward(self, x):
            if self.head_checkpointing and self.training and torch.is_grad_enabled():
                h = torch.utils.checkpoint.checkpoint(
                    self.layer, x, use_reentrant=False)
            else:
                h = self.layer(x)
            return self.out(h)

    torch.manual_seed(SEED)
    net = _CkptNet().train()
    x = torch.randn(4, 5, 8)

    def grads(checkpointing):
        net.head_checkpointing = checkpointing
        net.zero_grad(set_to_none=True)
        torch.manual_seed(SEED)  # identical dropout draw in both arms
        net(x).sum().backward()
        return {n: p.grad.detach().clone()
                for n, p in net.named_parameters() if p.grad is not None}

    off, on = grads(False), grads(True)
    assert set(off) == set(on) and off
    for name in off:
        assert torch.equal(off[name], on[name]), name


# ── torch.compile fidelity + fail-soft (the dispatch/cast-storm lever) ──────
def test_model_compiler_fuses_within_tolerance_and_fails_soft():
    namespace = _perf_namespace()
    ModelCompiler = namespace["ModelCompiler"]

    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Linear(32, 64), torch.nn.GELU(), torch.nn.Linear(64, 16))
    x = torch.randn(8, 32)
    eager = net(x)
    compiled = ModelCompiler.compile(torch, net, "default")
    got = compiled(x)
    assert torch.isfinite(got).all()
    # fusion reorders reductions: close, not bit-exact (real-model drift is
    # larger and is confirmed by the T4 smoke, not this synthetic net).
    assert torch.allclose(got, eager, atol=1e-4, rtol=1e-3)

    # fail-soft: a compiler error returns the EAGER module (recorded, never a
    # silent crash NOR a half-compiled model).
    class _Boom:
        @staticmethod
        def compile(*_a, **_k):
            raise RuntimeError("inductor backend unavailable")

    assert ModelCompiler.compile(_Boom, net, "max-autotune") is net


# ── unwrap strips compile + DDP wrappers for a clean state_dict ─────────────
def test_control_checkpointer_unwrap_strips_compile_and_ddp_wrappers():
    namespace = _perf_namespace()
    unwrap = namespace["ControlCheckpointer"].unwrap

    raw = torch.nn.Linear(4, 4)
    assert unwrap(raw) is raw
    # torch.compile OptimizedModule (_orig_mod)
    compiled = torch.compile(raw)
    assert unwrap(compiled) is raw
    assert not any(key.startswith("_orig_mod")
                   for key in unwrap(compiled).state_dict())
    # DDP (.module) is stripped too (the single-process gloo path)
    ddp = types.SimpleNamespace(module=raw)
    assert unwrap(ddp) is raw


# ── the compile_model dial reaches ModelCompiler inside the loop ────────────
def test_compile_model_control_wires_the_loop_through_model_compiler():
    """Run the REAL _perf_train_model with compile_model on and a spy-patched
    ModelCompiler.compile (no slow Inductor): the dial must route the mode into
    ModelCompiler and the loop must still produce a finite history."""
    namespace = _perf_namespace()
    seen = []
    namespace["ModelCompiler"].compile = staticmethod(
        lambda torch_module, model, mode: (seen.append(mode), model)[1])

    laya = _make_fake_laya()
    import sys
    sys.modules["laya"] = laya
    sys.modules["laya.train"] = laya.train
    namespace["FINETUNE_CONTROL"] = {
        **laya_lane.finetune_control(),
        "compile_model": True, "compile_mode": "reduce-overhead",
        "eval_dev": False, "save_each_epoch": False, "resume": False,
    }
    items = [{"i": i, "label": i % 3, "target": [1.0, 0.0, 0.0]}
             for i in range(8)]
    history = namespace["_perf_train_model"](
        _Net(), types.SimpleNamespace(pad_token_id=0), items, _Cfg(),
        torch.device("cpu"), 8, 4)
    assert seen == ["reduce-overhead"]  # dial value routed to ModelCompiler
    assert history and all(math.isfinite(v) for v in history)


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(4, 6)
        self.head = torch.nn.Linear(6, 3)
        self.head_checkpointing = False

    def forward(self, x):
        return self.head(torch.relu(self.encoder(x)))


class _Cfg:
    epochs = 1
    micro_batch = 2
    grad_accum = 1
    encoder_lr = 1e-2
    head_lr = 1e-2
    min_lr = 1e-3
    weight_decay = 0.0
    grad_clip = 1.0
    loss = "soft-ce"
    label_smoothing = 0.0
    rl_samples = 4
    sigma_start = 0.4
    sigma_end = 0.1
    w_sph = 0.75
    w_rps = 1.0
    shuffle_options = ()
    freeze_encoder = False
    seed = SEED
    amp = False
    gradient_checkpointing = False
    log_every = 0

    def validate(self):
        pass


def _make_fake_laya():
    """A minimal ``laya.train`` so the real loop runs with no laya install."""
    train = types.ModuleType("laya.train")

    def draw_option_order(it, rng, shuffle_options):
        return None

    def encode_item(tok, it, max_len, head_max_len, order, parallel):
        x = torch.zeros(4)
        x[it["label"]] = 1.0
        x[3] = it["i"] / 8.0
        return {"x": x, "y": it["label"]}

    def collate_items(chunks, pad_token_id):
        flat = [enc for chunk in chunks for enc in chunk]
        return {"x": torch.stack([enc["x"] for enc in flat]),
                "marker_mask": torch.ones(len(flat)),
                "target": torch.tensor([enc["y"] for enc in flat],
                                       dtype=torch.long),
                "qtype": torch.zeros(len(flat), dtype=torch.long)}

    def _forward(model, batch, device, amp, detach_encoder):
        return model(batch["x"].to(device))

    def soft_ce_loss(logits, target, mask):
        return torch.nn.functional.cross_entropy(logits, target)

    train.draw_option_order = draw_option_order
    train.encode_item = encode_item
    train.collate_items = collate_items
    train._forward = _forward
    train.soft_ce_loss = soft_ce_loss
    laya = types.ModuleType("laya")
    laya.train = train
    return laya
