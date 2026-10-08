"""Offline pins for the laya fine-tune TRAINING CONTROLS (no GPU required).

Phase 1 controls are DEFAULT-ON (per-epoch dev eval, early stop, best tracking,
per-epoch checkpoint/resume, scheduler menu); phases 2-4 are default-OFF. Every
knob is YAML-driven via `FinetuneSpec` and baked into `FINETUNE_CONTROL`, NEVER
into `laya.train.TrainConfig` (which rejects unknown kwargs).

The pure decision logic lives in `core.laya_controls` and is injected VERBATIM
into the staged perf patch (`FINETUNE_PERF_PATCH_SOURCE`) via `inspect.getsource`;
`test_control_logic_mirror_in_the_perf_patch` execs that staged source and pins
the two in lockstep so they cannot drift.
"""
from __future__ import annotations

import math
import os
import random
import re
import sys
import types

import torch

from core import laya_controls
from core.laya_config import FinetuneSpec, LayaSpec
from cli import laya_lane


# ── YAML-driven control block (separate from TrainConfig) ──────────────────
def test_phase_defaults_are_gated():
    ft = LayaSpec().finetune
    # Phase 1 — ON
    assert ft.eval_dev is True
    assert ft.early_stop is True
    assert ft.early_stop_patience == 2
    assert ft.early_stop_min_delta == 0.0
    assert ft.early_stop_metric == "dev_accuracy"
    assert ft.keep_best is True
    assert ft.save_each_epoch is True
    assert ft.resume is True
    assert ft.lr_scheduler == "cosine"
    assert ft.warmup_frac == 0.0 and ft.warmup_steps == 0
    # Phase 2-4 — OFF (log_grad_norm is the documented Phase 4 exception)
    assert ft.unfreeze_after_epoch is None
    assert ft.layer_decay == 1.0
    assert ft.ema is False and ft.swa is False
    assert ft.optimizer == "adamw"
    assert ft.class_weight is False and ft.balanced_sample is False
    assert ft.hard_example_frac == 0.0 and ft.curriculum is False
    assert ft.adv_eps == 0.0 and ft.temperature_scale is False
    assert ft.compile_model is False and ft.tf32 is False
    assert ft.amp_dtype == "fp16" and ft.deterministic is False
    assert ft.log_grad_norm is True


def test_control_fields_never_reach_trainconfig():
    control = laya_lane.finetune_control()
    assert set(control) == set(laya_lane.FINETUNE_CONTROL_FIELDS)
    assert set(laya_lane.FINETUNE_CONTROL_FIELDS).isdisjoint(
        laya_lane.FINETUNE_CONFIG_FIELDS)
    # TrainConfig surface is byte-identical: no control key leaks in.
    config = laya_lane.finetune_config()
    assert set(config).isdisjoint(laya_lane.FINETUNE_CONTROL_FIELDS)
    assert "early_stop" not in config and "lr_scheduler" not in config


def test_control_defaults_match_spec_surface():
    # ONE declaration: the baked tuples partition the FinetuneSpec surface,
    # with `device` the only non-baked field (the runtime resolver input).
    all_fields = set(FinetuneSpec.model_fields)
    control = set(laya_lane.FINETUNE_CONTROL_FIELDS)
    config = set(laya_lane.FINETUNE_CONFIG_FIELDS)
    assert control.isdisjoint(config)
    assert "device" not in control and "device" not in config
    assert control | config | {"device"} == all_fields


def test_control_yaml_flows_through():
    spec = LayaSpec(finetune={"early_stop": False, "lr_scheduler": "plateau",
                              "warmup_steps": 50, "optimizer": "lamb"})
    control = laya_lane.finetune_control(spec)
    assert control["early_stop"] is False
    assert control["lr_scheduler"] == "plateau"
    assert control["warmup_steps"] == 50
    assert control["optimizer"] == "lamb"


# ── pure early-stop logic ──────────────────────────────────────────────────
def test_early_stop_patience_and_min_delta():
    step = laya_controls.early_stop_step
    # first observation is always an improvement
    assert step(0.5, None, 0, 2, 0.0, False) == (0.5, 0, True, False)
    # patience 2: the counter trips on the SECOND non-improving epoch
    assert step(0.4, 0.5, 0, 2, 0.0, False) == (0.5, 1, False, False)
    assert step(0.4, 0.5, 1, 2, 0.0, False) == (0.5, 2, False, True)
    # patience 0 stops immediately
    assert step(0.4, 0.5, 0, 0, 0.0, False)[3] is True
    # min_delta gates a marginal gain
    assert step(0.51, 0.5, 0, 2, 0.05, False)[2] is False
    assert step(0.56, 0.5, 0, 2, 0.05, False)[2] is True
    # lower-is-better (dev_loss)
    assert step(0.3, 0.5, 0, 2, 0.0, True) == (0.3, 0, True, False)
    assert step(0.6, 0.5, 0, 2, 0.0, True)[2] is False


def test_parse_control_keeps_defaults_for_none_and_missing():
    assert laya_controls.parse_control(None, {"a": 1}) == {"a": 1}
    assert laya_controls.parse_control({"a": None, "b": 2},
                                       {"a": 1, "b": 3}) == {"a": 1, "b": 2}


# ── scheduler factory + warmup ─────────────────────────────────────────────
def _optimizer(lr=0.1):
    param = torch.nn.Parameter(torch.zeros(3))
    return torch.optim.SGD([param], lr=lr)


def test_scheduler_defaults_reproduce_todays_cosine_exactly():
    optimizer = _optimizer()
    scheduler = laya_controls.build_lr_scheduler(
        optimizer, "cosine", 12, 1e-6, 0)
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert scheduler.T_max == 12
    assert scheduler.eta_min == 1e-6


def test_scheduler_kinds_and_warmup():
    kinds = {
        "onecycle": torch.optim.lr_scheduler.OneCycleLR,
        "constant": torch.optim.lr_scheduler.LambdaLR,
        "linear": torch.optim.lr_scheduler.LambdaLR,
        "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    }
    for kind, cls in kinds.items():
        scheduler = laya_controls.build_lr_scheduler(
            _optimizer(), kind, 12, 1e-6, 0)
        assert isinstance(scheduler, cls), kind
    warm = laya_controls.build_lr_scheduler(_optimizer(), "cosine", 12, 1e-6, 4)
    assert isinstance(warm, torch.optim.lr_scheduler.SequentialLR)
    assert list(warm._milestones) == [4]


def test_linear_scheduler_decays_and_warmup_ramps():
    optimizer = _optimizer(lr=1.0)
    scheduler = laya_controls.build_lr_scheduler(optimizer, "linear", 5, 0.0, 0)
    assert optimizer.param_groups[0]["lr"] == 1.0
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] < 1.0


def test_effective_warmup_steps_explicit_beats_fraction():
    assert laya_controls.effective_warmup_steps(0, 0.0, 100) == 0
    assert laya_controls.effective_warmup_steps(10, 0.5, 100) == 10
    assert laya_controls.effective_warmup_steps(0, 0.25, 100) == 25
    # never consumes the whole schedule
    assert laya_controls.effective_warmup_steps(0, 1.0, 10) == 9


# ── metric helpers ─────────────────────────────────────────────────────────
def test_flatten_epoch_metrics_drops_none():
    assert laya_controls.flatten_epoch_metrics(
        "dev", {"accuracy": 0.8, "loss": None}) == {"dev/accuracy": 0.8}


def test_derive_abstain_coverage_from_records():
    records = [
        (0, [10.0, 0.0], [1.0, 0.0], 2),   # confident: conf ~ 1.0
        (0, [0.0, 0.0], [1.0, 0.0], 2),    # uniform: conf 0.5
    ]
    rate, coverage = laya_controls.derive_abstain_coverage(records, 0.6)
    assert rate == 0.5 and coverage == 0.5
    assert laya_controls.derive_abstain_coverage([], 0.5) == (None, None)


# ── staged perf patch mirrors the pure logic in lockstep ───────────────────
def _perf_namespace():
    namespace = {"os": os, "math": math, "random": random}
    exec(laya_lane.FINETUNE_PERF_PATCH_SOURCE, namespace)
    return namespace


def test_control_logic_mirror_in_the_perf_patch():
    namespace = _perf_namespace()
    cases = [
        (0.5, None, 0, 2, 0.0, False),
        (0.4, 0.5, 0, 2, 0.0, False),
        (0.4, 0.5, 1, 2, 0.0, False),
        (0.56, 0.5, 0, 2, 0.05, False),
        (0.3, 0.5, 0, 2, 0.0, True),
    ]
    for case in cases:
        assert namespace["early_stop_step"](*case) == \
            laya_controls.early_stop_step(*case)
    assert namespace["effective_warmup_steps"](0, 0.25, 100) == 25
    assert namespace["parse_control"]({"a": None, "b": 2},
                                      {"a": 1, "b": 3}) == {"a": 1, "b": 2}


# ── the STAGED kernel bakes the controls + the wandb metric names ──────────
def _render_kernel():
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
    return script


def test_staged_kernel_bakes_controls_and_wandb_metrics():
    script = _render_kernel()
    assert not re.search(r"@[A-Z][A-Z0-9_]*@", script)
    assert "FINETUNE_CONTROL = {" in script
    assert "FINETUNE_DEV_ROWS = None" in script
    assert "FINETUNE_OUTPUT_DIR = None" in script
    for metric in ("train/mean_loss", "train/lr", "train/grad_norm",
                   "epoch_time_s", "dev/accuracy", "dev/loss",
                   "dev/abstain_rate", "dev/coverage",
                   "select/best_dev_accuracy", "select/bad_epochs",
                   "early_stop/stopped", "early_stop/stopped_epoch"):
        assert metric in script, metric
    # the dev split is read in-kernel and stashed for the perf patch
    assert 'globals()["FINETUNE_DEV_ROWS"] = laya_train.read_jsonl(' in script
    for helper in ("early_stop_step", "build_lr_scheduler",
                   "effective_warmup_steps", "parse_control"):
        assert "def " + helper in script, helper


# ── Phase 1 integration on CPU: dev eval -> early stop -> checkpoints ──────
class _TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(4, 6)
        self.head = torch.nn.Linear(6, 3)
        self.head_checkpointing = False

    def forward(self, x):
        return self.head(torch.relu(self.encoder(x)))


class _Cfg:
    epochs = 4
    micro_batch = 4
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
    seed = 1729
    amp = False
    gradient_checkpointing = False
    log_every = 0

    def validate(self):
        pass


def _install_fake_laya(accuracies):
    calls = {"n": 0}
    train = types.ModuleType("laya.train")
    train.sigma_at = lambda *a: 0.4
    train.draw_option_order = lambda *a: None
    train.encode_item = lambda tok, it, *a: {"x": torch.zeros(4) + it["i"]}
    def _collate(chunks, pad):
        count = sum(len(chunk) for chunk in chunks)
        return {"marker_mask": torch.ones(count, 3),
                "target": torch.zeros(count, 3),
                "qtype": torch.zeros(count, dtype=torch.long)}
    train.collate_items = _collate
    train._forward = lambda model, batch, device, amp, freeze: model(
        torch.zeros(1, 4))
    train.soft_ce_loss = lambda logits, target, mask: logits.sum()
    train.items_from_rows = lambda tok, rows, *a, **k: (list(rows), {})
    train.calibration_records = lambda model, tok, items, *a, **k: [
        (0, [2.0, 0.0], [1.0, 0.0], 2)]

    def _evaluate(records, *a, **k):
        value = accuracies[min(calls["n"], len(accuracies) - 1)]
        calls["n"] += 1
        return {"items": 1, "loss": 0.5, "accuracy": value,
                "mean_confidence": 0.5, "ece": 0.1, "brier": 0.1,
                "brier_top1": 0.1}
    train.evaluate_records = _evaluate
    laya = types.ModuleType("laya")
    laya.train = train
    sys.modules["laya"] = laya
    sys.modules["laya.train"] = train
    return train


def test_perf_patch_phase1_cpu_dev_eval_early_stop_checkpoints(tmp_path):
    namespace = _perf_namespace()
    _install_fake_laya([0.9, 0.8, 0.8, 0.8])
    control = laya_lane.finetune_control()
    control.update({"eval_dev": True, "early_stop": True,
                    "early_stop_patience": 1, "early_stop_min_delta": 0.0,
                    "early_stop_metric": "dev_accuracy", "keep_best": True,
                    "save_each_epoch": True, "resume": False,
                    "log_grad_norm": True})
    namespace["FINETUNE_CONTROL"] = control
    namespace["FINETUNE_DEV_ROWS"] = [{"i": 0, "label": 0}]
    namespace["FINETUNE_OUTPUT_DIR"] = str(tmp_path)
    items = [{"i": i, "label": i % 3, "target": [1.0, 0.0, 0.0]}
             for i in range(8)]

    history = namespace["_perf_train_model"](
        _TinyNet(), types.SimpleNamespace(pad_token_id=0), items, _Cfg(),
        torch.device("cpu"), 8, 4)

    result = namespace["FINETUNE_CONTROL_RESULT"]
    # patience 1: epoch 0 improves, epoch 1 does not -> stop after epoch 1
    assert result["stopped"] is True
    assert result["stopped_epoch"] == 2
    assert result["epochs_run"] == 2
    assert result["best_dev_accuracy"] == 0.9
    assert len(history) == 2
    # per-epoch checkpoints written for every epoch that ran
    checkpoints = tmp_path / "checkpoints"
    assert (checkpoints / "epoch_0.pt").is_file()
    assert (checkpoints / "epoch_1.pt").is_file()
    # resume: a fresh run with resume=True loads epoch_1 and starts at epoch 2
    namespace2 = _perf_namespace()
    _install_fake_laya([0.9, 0.8, 0.8, 0.8])
    control2 = dict(control, resume=True)
    namespace2["FINETUNE_CONTROL"] = control2
    namespace2["FINETUNE_DEV_ROWS"] = [{"i": 0, "label": 0}]
    namespace2["FINETUNE_OUTPUT_DIR"] = str(tmp_path)
    namespace2["_perf_train_model"](
        _TinyNet(), types.SimpleNamespace(pad_token_id=0), items, _Cfg(),
        torch.device("cpu"), 8, 4)
    assert namespace2["FINETUNE_CONTROL_RESULT"]["epochs_run"] <= 3
