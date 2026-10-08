"""Surface-coverage matrix for the laya training-controls + profiler change.

One test per staged surface, all offline (no network, no GPU, no kaggle):

  * every kernel template (finetune, finetune-eval, holdout-eval) renders
    through `@TOKEN@` substitution with NO leftover token and passes both AST
    gates;
  * the finetune kernel bakes EVERY control dial (including the profiler and
    the plateau/onecycle scheduler dials) from the SSOT `FinetuneSpec`;
  * the CLI kernel registry resolves every fine-tune kind;
  * the eval/holdout kernels deliberately do NOT bake the training controls
    (they do not train);
  * the Colab lane is decision-only (corpus kinds fail loud);
  * the config schema accepts every new key and rejects a bad profiler spec.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from core.laya_config import FinetuneSpec, LayaSpec
from cli import laya_lane


def test_tests_import_the_worktree_src_not_a_sibling_checkout():
    """The shared venv .pth can point at the primary tree; conftest + this pin
    guarantee the suite exercises THIS checkout's src."""
    import cli
    import core
    import training

    src = Path(__file__).resolve().parents[1] / "src"
    for module in (core, cli, training):
        assert Path(module.__file__).resolve().is_relative_to(src), \
            f"{module.__name__} resolved outside {src}: {module.__file__}"


def _render(template, values):
    script = laya_lane._template(template, values)
    ast.parse(script)
    laya_lane._kernel_script_gate(script)
    laya_lane._module_scope_gate(script)
    assert not re.search(r"@[A-Z][A-Z0-9_]*@", script)
    return script


def _repo_values(spec, run_tag="surface_t"):
    return {"REPOSITORY": "anomalyco/er", "BRANCH": "main",
            "REVISION": "deadbeef", "RUN_TAG": run_tag}


def render_finetune():
    spec = laya_lane._spec()
    values = {
        "LAYA_PACKAGE": spec.finetune_package,
        "BASE_MODEL_ARCHIVE": spec.base_model_archive,
        "BASE_MODEL_DIR": spec.base_model_dir,
        "TRAIN_JSONL": laya_lane.FINETUNE_CORPUS_FILES[0],
        "DEV_JSONL": laya_lane.FINETUNE_CORPUS_FILES[1],
        "TEST_JSONL": laya_lane.FINETUNE_CORPUS_FILES[2],
        "FINETUNE_CONFIG": repr(laya_lane.finetune_config(spec)),
        "FINETUNE_CONTROL": repr(laya_lane.finetune_control(spec)),
        "FINETUNE_DEVICE": spec.finetune.device,
        "HELD_OUT_BATCH": str(spec.laya_decision_batch_size),
        "WANDB_API_KEY": "",
        "WANDB_PROJECT": "e-r",
        "DEVICE_PATCH": laya_lane.FINETUNE_DEVICE_PATCH_SOURCE,
        "PERF_PATCH": laya_lane.FINETUNE_PERF_PATCH_SOURCE,
        **_repo_values(spec),
    }
    preflight = laya_lane._template(laya_lane.FINETUNE_RUNTIME_PREFLIGHT, values)
    return _render(laya_lane.FINETUNE_KERNEL_SCRIPT,
                   {**values, "RUNTIME_PREFLIGHT": preflight})


def render_finetune_eval():
    spec = laya_lane._spec()
    values = {
        "LAYA_PACKAGE": spec.finetune_package,
        "EVAL_JSONL": laya_lane.FINETUNE_EVAL_SPLIT_FILES["test"],
        "EVAL_SPLIT": "test",
        "CKPT_DIR": spec.finetune_ckpt_dir,
        "CHECKPOINT_PATH": "",
        "BATCH_SIZE": str(spec.finetune_eval_batch_size),
        "EVAL_CALIBRATION": repr(laya_lane.eval_calibration_config(spec)),
        **_repo_values(spec),
    }
    preflight = laya_lane._template(laya_lane.FINETUNE_EVAL_RUNTIME_PREFLIGHT,
                                    values)
    return _render(laya_lane.FINETUNE_EVAL_KERNEL_SCRIPT,
                   {**values, "RUNTIME_PREFLIGHT": preflight})


def render_holdout_eval():
    spec = laya_lane._spec()
    values = {
        "LAYA_PACKAGE": spec.finetune_package,
        "HOLDOUT_JSONL": laya_lane.HOLDOUT_JSONL,
        "CKPT_DIR": spec.finetune_ckpt_dir,
        "BATCH_SIZE": str(spec.holdout_eval_batch_size),
        "THRESHOLD": repr(spec.holdout_eval_threshold),
        "N_BOOT": str(spec.holdout_eval_bootstrap),
        "SEED": "1729",
        **_repo_values(spec),
    }
    preflight = laya_lane._template(laya_lane.LAYA_RUNTIME_PREFLIGHT, {
        **values, "DECISION_CSV": laya_lane.HOLDOUT_JSONL,
        "QUESTION_SCHEMA_FILE": repr(laya_lane.QUESTION_SCHEMA_FILE)})
    return _render(laya_lane.HOLDOUT_EVAL_KERNEL_SCRIPT,
                   {**values, "RUNTIME_PREFLIGHT": preflight})


def test_every_kernel_template_renders_without_leftover_tokens():
    for script in (render_finetune(), render_finetune_eval(),
                   render_holdout_eval()):
        assert script and not re.search(r"@[A-Z][A-Z0-9_]*@", script)


def _baked(script, name):
    for node in ast.walk(ast.parse(script)):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not baked")


def test_finetune_kernel_bakes_every_control_knob_from_the_ssot():
    baked = _baked(render_finetune(), "FINETUNE_CONTROL")
    assert set(baked) == set(laya_lane.FINETUNE_CONTROL_FIELDS)
    assert baked == laya_lane.finetune_control()
    # the added profiler + scheduler dials specifically
    for key in ("profile", "profile_dir", "profile_schedule",
                "plateau_factor", "onecycle_pct_start"):
        assert key in baked, key
    assert baked["profile"] is True
    assert baked["profile_schedule"] == {"wait": 1, "warmup": 1, "active": 1,
                                         "repeat": 1}


def test_eval_and_holdout_kernels_do_not_bake_training_controls():
    # They are eval-only: no trainer, no per-epoch controls, no profiler.
    for script in (render_finetune_eval(), render_holdout_eval()):
        assert "FINETUNE_CONTROL" not in script
        assert "FINETUNE_CONFIG" not in script
        assert "train_model(" not in script
        assert "ProfilerSession" not in script


def test_finetune_kernel_wires_wandb_and_profiler_surfaces():
    script = render_finetune()
    for surface in ("WANDB_PROJECT = \"e-r\"", "WANDB_API_KEY",
                    "def wandb_init(", "def wandb_log_epoch(",
                    "def wandb_log_control_summary(", "def wandb_log_metrics(",
                    "class ProfilerSession", "class ControlCheckpointer",
                    "FINETUNE_DEV_ROWS = None", "FINETUNE_OUTPUT_DIR = None",
                    "torch.profiler.record_function(\"dev_eval\")"):
        assert surface in script, surface
    assert "checkpoint/checkpoints" not in script  # dir built from the global
    assert "os.path.join(str(self._output_dir), \"checkpoints\")" in script


def test_kernel_registry_resolves_every_finetune_kind(monkeypatch):
    spec = LayaSpec(finetune_kernel_slug="o/finetune",
                    finetune_eval_kernel_slug="o/finetune-eval",
                    holdout_eval_kernel_slug="o/holdout-eval")
    monkeypatch.setattr(laya_lane, "_spec", lambda: spec)
    assert set(("finetune", "finetune-eval", "holdout-eval")).issubset(
        set(laya_lane.GPU_KINDS))
    assert laya_lane.kernel_slug("finetune") == "o/finetune"
    assert laya_lane.kernel_slug("finetune-eval") == "o/finetune-eval"
    assert laya_lane.kernel_slug("holdout-eval") == "o/holdout-eval"
    # the corpus kinds ride the decision registry; holdout has its own stage
    assert "finetune" in laya_lane.DECISION_BINDINGS
    assert "finetune-eval" in laya_lane.DECISION_BINDINGS


def test_colab_lane_is_decision_only_and_rejects_corpus_kinds():
    # The Colab lane stages per-row decisions; corpus/training kinds have no
    # F-binding, so they fail loud instead of silently staging an empty run.
    with pytest.raises(RuntimeError):
        laya_lane.decision_binding("finetune")
    assert laya_lane.LayaLane("colab").kind == "colab"


def test_config_schema_accepts_every_new_control_key():
    payload = {name: LayaSpec().finetune.model_dump()[name]
               for name in laya_lane.FINETUNE_CONTROL_FIELDS}
    spec = LayaSpec(finetune=payload)
    assert set(laya_lane.finetune_control(spec)) == \
        set(laya_lane.FINETUNE_CONTROL_FIELDS)
    # unknown keys stay rejected (extra=forbid)
    with pytest.raises(Exception):
        LayaSpec(finetune={"not_a_knob": 1})
    with pytest.raises(Exception):
        LayaSpec(finetune={"profile_schedule": {"wait": 1, "bad": 2}})
