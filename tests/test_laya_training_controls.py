"""Offline pins for the laya fine-tune TRAINING CONTROLS (no GPU required).

Phase 1 controls are DEFAULT-ON (per-epoch dev eval, early stop, best tracking,
per-epoch checkpoint/resume, scheduler menu); phases 2-4 are default-OFF. Every
knob is YAML-driven via `FinetuneSpec` and baked into `FINETUNE_CONTROL`, NEVER
into `laya.train.TrainConfig` (which rejects unknown kwargs).

The control logic is cohesive CLASSES in `core.laya_controls`; the staged perf
patch is injected from those same class sources. These tests exercise the
PUBLIC class methods; the lockstep test execs the staged patch and pins the two
copies of each class to identical behavior.
"""
from __future__ import annotations

import math
import os
import random
import re

import pytest
import torch

from core import laya_controls
from core.laya_config import FinetuneSpec, LayaSpec
from cli import laya_lane


def _control(**overrides):
    """The full baked control block from the SSOT spec (+ test overrides)."""
    control = laya_lane.finetune_control()
    control.update(overrides)
    return control


def _optimizer(lr=0.1):
    return torch.optim.SGD([torch.nn.Parameter(torch.zeros(3))], lr=lr)


def _factory(optimizer, kind, total=12, min_lr=1e-6, warmup=0):
    """The SSOT-driven scheduler factory (spec defaults for plateau/onecycle)."""
    spec = FinetuneSpec()
    return laya_controls.LrSchedulerFactory(
        optimizer, kind, total, min_lr, warmup,
        plateau_mode="max", plateau_factor=spec.plateau_factor,
        plateau_patience=spec.plateau_patience,
        onecycle_pct_start=spec.onecycle_pct_start)


# ── YAML-driven control block (separate from TrainConfig) ──────────────────
def test_phase_defaults_are_gated():
    ft = LayaSpec().finetune
    assert ft.eval_dev is True
    assert ft.early_stop is True and ft.early_stop_patience == 2
    assert ft.early_stop_min_delta == 0.0
    assert ft.early_stop_metric == "dev_accuracy"
    assert ft.keep_best is True and ft.save_each_epoch is True
    assert ft.resume is True
    assert ft.lr_scheduler == "cosine"
    assert ft.warmup_frac == 0.0 and ft.warmup_steps == 0
    # Phase 2-4 off (log_grad_norm is the documented Phase 4 exception)
    assert ft.unfreeze_after_epoch is None and ft.layer_decay == 1.0
    assert ft.ema is False and ft.swa is False and ft.optimizer == "adamw"
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
    config = laya_lane.finetune_config()
    assert set(config).isdisjoint(laya_lane.FINETUNE_CONTROL_FIELDS)
    assert "early_stop" not in config and "lr_scheduler" not in config


def test_control_fields_partition_the_spec_surface():
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


# ── ControlBlock ───────────────────────────────────────────────────────────
def test_control_block_merges_defaults_and_ignores_none():
    block = laya_controls.ControlBlock(
        {"early_stop": False, "patience": None}, {"early_stop": True, "p": 2})
    assert block.get("early_stop") is False
    assert block.get("patience") is None
    assert block.get("p") == 2
    assert block.get("missing") is None
    assert "early_stop" in block
    assert block["p"] == 2
    assert block.as_dict()["p"] == 2


# ── EarlyStopPolicy ────────────────────────────────────────────────────────
def test_early_stop_policy_improvement_and_patience():
    policy = laya_controls.EarlyStopPolicy(2, 0.0, False)
    first = policy.update(0.5, None, 0)
    assert (first.best, first.bad_epochs, first.improved, first.stop) == \
        (0.5, 0, True, False)
    assert policy.update(0.4, 0.5, 0).stop is False
    tripped = policy.update(0.4, 0.5, 1)
    assert (tripped.best, tripped.bad_epochs, tripped.stop) == (0.5, 2, True)


def test_early_stop_policy_min_delta_and_direction():
    policy = laya_controls.EarlyStopPolicy(2, 0.05, False)
    assert policy.is_improvement(0.51, 0.5) is False
    assert policy.is_improvement(0.56, 0.5) is True
    lower = laya_controls.EarlyStopPolicy(2, 0.0, True)
    assert lower.is_improvement(0.3, 0.5) is True
    assert lower.is_improvement(0.6, 0.5) is False


def test_early_stop_policy_patience_zero_stops_immediately():
    policy = laya_controls.EarlyStopPolicy(0, 0.0, False)
    assert policy.update(0.4, 0.5, 0).stop is True


def test_early_stop_policy_from_control_reads_the_block():
    policy = laya_controls.EarlyStopPolicy.from_control(
        _control(early_stop_patience=3, early_stop_min_delta=0.01,
                 early_stop_metric="dev_loss"))
    assert policy.patience == 3 and policy.min_delta == 0.01
    assert policy.lower_is_better is True


# ── LrSchedulerFactory ─────────────────────────────────────────────────────
def test_scheduler_factory_reproduces_landed_cosine_exactly():
    scheduler = _factory(_optimizer(), "cosine").build()
    assert isinstance(scheduler,
                      torch.optim.lr_scheduler.CosineAnnealingLR)
    assert scheduler.T_max == 12 and scheduler.eta_min == 1e-6


def test_scheduler_factory_kinds_and_warmup():
    kinds = {
        "onecycle": torch.optim.lr_scheduler.OneCycleLR,
        "constant": torch.optim.lr_scheduler.LambdaLR,
        "linear": torch.optim.lr_scheduler.LambdaLR,
        "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    }
    for kind, expected in kinds.items():
        factory = _factory(_optimizer(), kind)
        assert isinstance(factory.build(), expected), kind
    warm = _factory(_optimizer(), "cosine", warmup=4).build()
    assert isinstance(warm, torch.optim.lr_scheduler.SequentialLR)
    assert list(warm._milestones) == [4]


def test_scheduler_factory_from_control_uses_the_block():
    factory = laya_controls.LrSchedulerFactory.from_control(
        _control(lr_scheduler="linear", warmup_steps=3),
        _optimizer(), 10, 1e-6, lower_is_better=False)
    assert factory.kind == "linear" and factory.warmup == 3
    assert factory.build() is not None


def test_scheduler_factory_effective_warmup_steps():
    effective = laya_controls.LrSchedulerFactory.effective_warmup_steps
    assert effective(0, 0.0, 100) == 0
    assert effective(10, 0.5, 100) == 10          # explicit wins
    assert effective(0, 0.25, 100) == 25
    assert effective(0, 1.0, 10) == 9             # never eats the schedule


def test_linear_scheduler_decays():
    optimizer = _optimizer(lr=1.0)
    scheduler = _factory(optimizer, "linear", total=5, min_lr=0.0).build()
    assert optimizer.param_groups[0]["lr"] == 1.0
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] < 1.0


def test_plateau_with_warmup_fails_loud():
    # plateau steps on the per-epoch dev metric; a per-update warmup is
    # meaningless, so it must fail loud rather than be silently ignored.
    with pytest.raises(ValueError, match="plateau"):
        _factory(_optimizer(), "plateau", warmup=5)


def test_onecycle_consumes_warmup_as_pct_start(monkeypatch):
    captured = {}
    real = torch.optim.lr_scheduler.OneCycleLR

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(torch.optim.lr_scheduler, "OneCycleLR", spy)
    scheduler = _factory(_optimizer(), "onecycle", total=100, warmup=10).build()
    assert isinstance(scheduler, real)
    # pct_start derived from warmup/total, never a restated default
    assert captured["pct_start"] == pytest.approx(0.1)


# ── MetricFlattener / AbstainCoverage / DevReport ──────────────────────────
def test_metric_flattener_drops_none():
    assert laya_controls.MetricFlattener.flatten(
        "dev", {"accuracy": 0.8, "loss": None}) == {"dev/accuracy": 0.8}


def test_abstain_coverage_estimates_rate():
    records = [
        (0, [10.0, 0.0], [1.0, 0.0], 2),   # confident: conf ~ 1.0
        (0, [0.0, 0.0], [1.0, 0.0], 2),    # uniform: conf 0.5
    ]
    rate, coverage = laya_controls.AbstainCoverage.estimate(records, 0.6)
    assert rate == 0.5 and coverage == 0.5
    assert laya_controls.AbstainCoverage.estimate([], 0.5) == (None, None)
    assert laya_controls.AbstainCoverage.estimate(records, None) == \
        (None, None)


def test_dev_report_from_metrics_and_payload():
    records = [(0, [0.0, 0.0], [1.0, 0.0], 2)]
    report = laya_controls.DevReport.from_metrics(
        {"accuracy": 0.75, "loss": 0.4, "ece": 0.1}, records=records,
        confidence_threshold=0.6)
    assert report.accuracy == 0.75 and report.loss == 0.4
    assert report.abstain_rate == 1.0 and report.coverage == 0.0
    assert report.to_wandb("dev")["dev/accuracy"] == 0.75
    # broadcast round-trip
    restored = laya_controls.DevReport.from_payload(*report.payload())
    assert restored.accuracy == 0.75 and restored.loss == 0.4
    assert laya_controls.DevReport.from_payload(0.0, 0.0, 0.0, -1.0, -1.0) \
        is None


# ── TrainingControls facade (reads the SSOT block) ─────────────────────────
def test_training_controls_facade():
    controls = laya_controls.TrainingControls.parse(_control())
    assert controls.flag("early_stop") is True
    assert controls.early_stop_policy().patience == 2
    factory = controls.scheduler(_optimizer(), 12, 1e-6)
    assert factory.kind == "cosine"
    assert isinstance(factory.build(),
                      torch.optim.lr_scheduler.CosineAnnealingLR)


# ── staged perf patch mirrors the pure classes in lockstep ─────────────────
def _perf_namespace():
    namespace = {"os": os, "math": math, "random": random}
    exec(laya_lane.FINETUNE_PERF_PATCH_SOURCE, namespace)
    return namespace


def test_adversarial_perturber_fgm_vs_awp_targets():
    namespace = _perf_namespace()

    class _Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(4, 4)
            self.embed_tokens = torch.nn.Embedding(5, 4)

    model = _Net()
    params = list(model.parameters())
    targets = namespace["AdversarialPerturber"].targets
    fgm_ids = {id(p) for p in targets(model, params, "fgm")}
    expected = {id(p) for name, p in model.named_parameters()
                if "embed" in name}
    assert fgm_ids == expected and fgm_ids  # fgm perturbs embeddings only
    awp_ids = {id(p) for p in targets(model, params, "awp")}
    assert awp_ids == {id(p) for p in params}


def test_control_checkpointer_run_tag_guard_and_best_persist(tmp_path):
    namespace = _perf_namespace()

    class _Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 2)

    model = _Net()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    checkpointer = namespace["ControlCheckpointer"](torch, str(tmp_path),
                                                   "runA")
    checkpointer.save(model, optimizer, scheduler, 3, 0.9, 0)
    checkpointer.save_best(model, 0.9, 0.91, 3)
    start, best, bad, record = checkpointer.resume(
        model, optimizer, scheduler, torch.device("cpu"))
    assert (start, best, bad) == (4, 0.9, 0)
    assert record is not None and record["accuracy"] == 0.91
    assert (tmp_path / "checkpoints" / "best.pt").is_file()
    # a DIFFERENT run tag must not resume from runA's files
    other = namespace["ControlCheckpointer"](torch, str(tmp_path), "runB")
    assert other.resume(model, optimizer, scheduler,
                        torch.device("cpu")) == (0, None, 0, None)


def test_injected_classes_mirror_the_module():
    namespace = _perf_namespace()
    cases = [(0.5, None, 0), (0.4, 0.5, 0), (0.4, 0.5, 1)]
    local = namespace["EarlyStopPolicy"](2, 0.0, False)
    reference = laya_controls.EarlyStopPolicy(2, 0.0, False)
    for value, best, bad in cases:
        a, b = local.update(value, best, bad), reference.update(value, best, bad)
        assert (a.best, a.bad_epochs, a.stop) == \
            (b.best, b.bad_epochs, b.stop)
    assert namespace["LrSchedulerFactory"].effective_warmup_steps(0, 0.25, 100) \
        == 25
    block = namespace["ControlBlock"]({"early_stop": True}, {})
    assert block.get("early_stop") is True
    assert namespace["MetricFlattener"].flatten("x", {"a": 1, "b": None}) == \
        {"x/a": 1}


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
    assert 'print("epoch %d/%d dev_acc=%.4f dev_loss=%.4f"' in script
    for metric in ("train/mean_loss", "train/lr", "train/grad_norm",
                   "epoch_time_s", "select/best_dev_accuracy",
                   "select/bad_epochs", "early_stop/stopped",
                   "early_stop/stopped_epoch"):
        assert metric in script, metric
    # the dev/* names are built by DevReport.to_wandb(prefix); the keys + the
    # "dev" prefix are the literals baked into the injected class.
    for piece in ('"accuracy"', '"loss"', '"abstain_rate"', '"coverage"',
                  'prefix="dev"'):
        assert piece in script, piece
    assert 'globals()["FINETUNE_DEV_ROWS"] = laya_train.read_jsonl(' in script
    for klass in ("ControlBlock", "EarlyStopPolicy", "LrSchedulerFactory",
                  "MetricFlattener", "DevReport", "TrainingControls",
                  "ProfilerSession", "ControlCheckpointer"):
        assert "class " + klass in script, klass
