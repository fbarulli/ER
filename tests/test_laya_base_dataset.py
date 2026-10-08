"""Laya fine-tune base-model transport — offline pins.

The finetune base checkpoint is the attached `er-laya-base` dataset
archive (convaiinnovations-laya.tar.zst), extracted in-kernel. The
rendered kernel must extract the archive and pass the extracted LOCAL
directory as `--base`, never the Hugging Face Hub id.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from core.schemas import LayaSpec
from cli import laya_lane

REVISION_PIN = "abc123def"


def _spec(tmp_path, monkeypatch, **updates):
    cfg_spec = LayaSpec(**{
        "finetune_kernel_slug": "fbarulli/er-laya-finetune",
        "finetune_dataset_slug": "fbarulli/er-laya-train",
        "base_model_dataset": "fbarulli/er-laya-base",
        **updates,
    })
    monkeypatch.setattr(laya_lane, "_spec", lambda: cfg_spec)
    monkeypatch.setattr(laya_lane, "TRAIN_ROOT", tmp_path)
    return cfg_spec


def _corpus(tmp_path):
    corpus = tmp_path / "data/laya"
    corpus.mkdir(parents=True, exist_ok=True)
    for name in ("train.jsonl", "dev.jsonl", "test.jsonl"):
        (corpus / name).write_text('{"state": "s"}\n', encoding="utf-8")
    (corpus / "receipt.json").write_text('{"seed": 1729}\n',
                                         encoding="utf-8")


def _hermetic_staging(monkeypatch, *, origin_tip: str = REVISION_PIN):
    monkeypatch.setattr(laya_lane, "_git_revision", lambda: REVISION_PIN)

    def fake_run(args, **kwargs):
        if "rev-parse" in args:
            return subprocess.CompletedProcess(args, 0, stdout=origin_tip,
                                               stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)


def test_staged_finetune_kernel_attaches_the_base_dataset(
        tmp_path, monkeypatch):
    _spec(tmp_path, monkeypatch)
    _corpus(tmp_path)
    _hermetic_staging(monkeypatch)
    receipt = laya_lane.stage_finetune_kernel(run_tag="laya_test")
    stage = Path(receipt["staged"])
    metadata = json.loads((stage / "kernel-metadata.json").read_text())
    # the base-model dataset rides alongside the corpus dataset
    assert metadata["dataset_sources"] == [
        "fbarulli/er-laya-train", "fbarulli/er-laya-base"]
    assert receipt["base_model"] == {
        "dataset": "fbarulli/er-laya-base",
        "archive": "convaiinnovations-laya.tar.zst",
        "dir": "convaiinnovations-laya",
    }


def test_rendered_kernel_extracts_and_points_base_at_local_dir(
        tmp_path, monkeypatch):
    _spec(tmp_path, monkeypatch)
    _corpus(tmp_path)
    _hermetic_staging(monkeypatch)
    receipt = laya_lane.stage_finetune_kernel(run_tag="laya_test")
    script = (Path(receipt["staged"]) / "laya_finetune.py").read_text()
    # the attached archive name + its single top-level member are baked in
    assert 'BASE_MODEL_ARCHIVE = "convaiinnovations-laya.tar.zst"' in script
    assert 'BASE_MODEL_DIR = "convaiinnovations-laya"' in script
    # the kernel locates the archive under /kaggle/input, extracts it, and
    # passes the extracted local dir as the finetune base (no Hub id, no
    # download)
    assert "extract_base_model" in script
    assert "model_dir=str(base_model)" in script
    assert "tarfile.open(fileobj=stream, mode=\"r|\")" in script
    assert '"convaiinnovations/laya"' not in script
    # no Hub fetch: there is no snapshot_download CALL and no hub import
    assert "snapshot_download(" not in script
    assert "huggingface_hub" not in script


def test_finetune_kernel_fails_loud_without_base_dataset(
        tmp_path, monkeypatch):
    _spec(tmp_path, monkeypatch, base_model_dataset=None)
    _corpus(tmp_path)
    _hermetic_staging(monkeypatch)
    with pytest.raises(RuntimeError, match="base_model_dataset"):
        laya_lane.stage_finetune_kernel(run_tag="laya_test")


def _rendered_script(tmp_path, monkeypatch):
    import ast

    _spec(tmp_path, monkeypatch)
    _corpus(tmp_path)
    _hermetic_staging(monkeypatch)
    receipt = laya_lane.stage_finetune_kernel(run_tag="laya_test")
    script = (Path(receipt["staged"]) / "laya_finetune.py").read_text()
    # the staging AST gates already ran; re-prove the rendered payload is
    # parseable and clears both gates (undefined top-level name class)
    ast.parse(script)
    laya_lane._kernel_script_gate(script)
    laya_lane._module_scope_gate(script)
    return script


def test_rendered_kernel_embeds_gpu_sampler_and_perf_patch(
        tmp_path, monkeypatch):
    script = _rendered_script(tmp_path, monkeypatch)
    # GPU sampler: 1 Hz nvidia-smi query into /kaggle/working/gpu_usage.log,
    # with a min/max/mean + peak-mem summary in the receipt.
    assert "--query-gpu=utilization.gpu,memory.used,memory.total" in script
    assert '"--format=csv,noheader"' in script
    assert 'WORKING / "gpu_usage.log"' in script
    assert "start_gpu_sampler" in script
    assert "stop_gpu_sampler" in script
    assert "summarize_gpu_usage" in script
    assert '"gpu_usage": summarize_gpu_usage(WORKING / "gpu_usage.log")' in script
    # absent nvidia-smi is a graceful skip, never a crash
    assert 'shutil.which("nvidia-smi") is None' in script
    # PERF patch: monkeypatched laya.train.train_model + opt-out env flag
    assert 'PERF_PATCH_ENV = "ER_LAYA_PERF_PATCH"' in script
    assert "laya_train.train_model = _perf_train_model" in script
    assert "apply_perf_patch()" in script
    assert "perf_patch_enabled()" in script
    # (1) no per-micro-step host sync: loss.item() is gone, one sync per epoch
    assert "loss.item()" not in script
    assert "total.item()" in script
    # (2) the loop consumes pre-moved tensors; _forward is unchanged
    assert 'mask = batch["marker_mask"].to(device)' in script
    assert 'qtype = batch["qtype"].to(device)' in script
    # (3) encode memoization keyed by (item id, option order)
    assert "encoded = cache.get(key)" in script
    assert "draw_option_order(" in script


def _rendered_finetune_config(tmp_path, monkeypatch, **updates):
    """Render the finetune payload and recover the baked FINETUNE_CONFIG."""
    import ast

    _spec(tmp_path, monkeypatch, **updates)
    _corpus(tmp_path)
    _hermetic_staging(monkeypatch)
    receipt = laya_lane.stage_finetune_kernel(run_tag="laya_test")
    script = (Path(receipt["staged"]) / "laya_finetune.py").read_text()
    for node in ast.walk(ast.parse(script)):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Name)
                        and target.id == "FINETUNE_CONFIG"):
                    return ast.literal_eval(node.value), receipt, script
    raise AssertionError("FINETUNE_CONFIG not baked into the kernel")


def test_recipe_defaults_reproduced_from_config(tmp_path, monkeypatch):
    """The YAML/default FinetuneSpec reproduces the landed recipe exactly and
    every other `laya.train.TrainConfig` knob keeps its upstream default."""
    config, receipt, script = _rendered_finetune_config(tmp_path, monkeypatch)
    # the landed research recipe
    assert config["epochs"] == 8
    assert config["micro_batch"] == 8
    assert config["grad_accum"] == 8
    assert config["encoder_lr"] == 2.5e-5
    assert config["head_lr"] == 1e-4
    assert config["loss"] == "soft-ce"
    assert config["seed"] == 1729
    # the upstream TrainConfig defaults for every other knob
    assert config["min_lr"] == 1e-6
    assert config["weight_decay"] == 0.01
    assert config["grad_clip"] == 1.0
    assert config["rl_samples"] == 4
    assert config["sigma_start"] == 0.4
    assert config["sigma_end"] == 0.1
    assert config["w_sph"] == 0.75
    assert config["w_rps"] == 1.0
    assert config["shuffle_options"] == ()
    assert config["option_layout"] is None
    assert config["max_len"] is None
    assert config["head_max_len"] is None
    assert config["text_column"] == "text"
    assert config["label_column"] == "label"
    assert config["question_id"] == "label"
    assert config["instructions"] is None
    assert config["freeze_encoder"] is False
    assert config["calib_max"] == 400
    assert config["calib_frac"] == 0.1
    assert config["calib_seed"] == 20260922
    assert config["target_error"] == 0.10
    assert config["min_abstain_n"] == 10
    assert config["amp"] is None
    assert config["gradient_checkpointing"] is None
    assert config["log_every"] == 100
    # the kernel applies the FULL config to the trainer surface (the CLI
    # never ran: the non-CLI knobs are set by constructing TrainConfig)
    assert "TrainConfig(**FINETUNE_CONFIG" in script
    assert "laya_train.finetune(" in script
    assert "train_cli" not in script
    # the SEPARATE control block (never a TrainConfig kwarg) rides the
    # receipt and the staged kernel: every declared dial is baked.
    assert set(receipt["control"]) == set(laya_lane.FINETUNE_CONTROL_FIELDS)
    assert receipt["control"]["profile"] is True
    assert "FINETUNE_CONTROL = {" in script
    assert not __import__("re").search(r"@[A-Z][A-Z0-9_]*@", script)
    # the patch is a faithful copy: the recipe-critical schedule lines stay
    assert "window_start = (n_steps // epoch_grad_accum) * epoch_grad_accum" in script
    assert "torch.nn.utils.clip_grad_norm_(params, config.grad_clip)" in script
    assert "scheduler.step()" in script
    assert "torch.manual_seed(config.seed)" in script
    assert "random.Random(config.seed + epoch).shuffle(epoch_items)" in script
    # the receipt carries the full config + the resolved device
    assert receipt["recipe"]["epochs"] == 8
    assert receipt["recipe"]["weight_decay"] == 0.01
    assert receipt["device"] == "auto"


def test_non_cli_knobs_flow_from_yaml_into_the_kernel(tmp_path, monkeypatch):
    """A non-default value for knobs the CLI never exposes (weight_decay,
    label_smoothing, min_lr, rl_samples, log_every) flows YAML ->
    FinetuneSpec -> FINETUNE_CONFIG -> TrainConfig, proving the whole
    trainer surface is config-driven."""
    config, receipt, script = _rendered_finetune_config(
        tmp_path, monkeypatch, finetune={
            "weight_decay": 0.123, "label_smoothing": 0.05,
            "min_lr": 3e-7, "rl_samples": 7, "log_every": 5,
            "shuffle_options": ["choice"], "option_layout": "parallel",
            "amp": False, "gradient_checkpointing": True,
            "device": "cuda", "epochs": 3})
    assert config["weight_decay"] == 0.123
    assert config["label_smoothing"] == 0.05
    assert config["min_lr"] == 3e-7
    assert config["rl_samples"] == 7
    assert config["log_every"] == 5
    assert config["shuffle_options"] == ("choice",)
    assert config["option_layout"] == "parallel"
    assert config["amp"] is False
    assert config["gradient_checkpointing"] is True
    assert config["epochs"] == 3
    # the device knob flows too (baked as FINETUNE_DEVICE)
    assert 'FINETUNE_DEVICE = "cuda"' in script
    assert receipt["recipe"]["weight_decay"] == 0.123
    assert receipt["device"] == "cuda"


# ── the fine-tune EVAL-ONLY path (held-out score, no retrain) ──────────────
# The gap: the finetune kind always trains and laya-cli-eval scores the
# DECISION payload, so a held-out score of a fine-tuned checkpoint needed a
# manual load. `--decision finetune-eval` stages a dedicated eval-only kernel
# that attaches the corpus + the fine-tuned checkpoint and runs
# load_checkpoint -> calibration_records -> evaluate_records on the split.
def test_staged_finetune_eval_kernel_attaches_corpus_and_checkpoint(
        tmp_path, monkeypatch):
    _spec(tmp_path, monkeypatch)
    _corpus(tmp_path)
    _hermetic_staging(monkeypatch)
    receipt = laya_lane.stage_finetune_eval_kernel(run_tag="laya_test")
    stage = Path(receipt["staged"])
    metadata = json.loads((stage / "kernel-metadata.json").read_text())
    assert metadata["code_file"] == "laya_finetune_eval.py"
    # the corpus dataset rides beside the fine-tuned checkpoint dataset
    assert metadata["dataset_sources"] == [
        "fbarulli/er-laya-train", "fbarulli/er-laya-finetune-ckpt"]
    assert (stage / "laya_finetune_eval.py").is_file()
    # the corpus payload lands beside THIS kernel so the push gate finds it
    assert (stage / "dataset_payload" / "test.jsonl").is_file()
    assert receipt["kind"] == "finetune-eval"
    assert receipt["checkpoint_dataset"] == "fbarulli/er-laya-finetune-ckpt"
    assert receipt["checkpoint_dir_hint"] == "checkpoint"
    assert receipt["eval_split"] == "test"
    assert receipt["eval_jsonl"] == "test.jsonl"


def test_rendered_finetune_eval_kernel_is_eval_only(tmp_path, monkeypatch):
    _spec(tmp_path, monkeypatch)
    _corpus(tmp_path)
    _hermetic_staging(monkeypatch)
    receipt = laya_lane.stage_finetune_eval_kernel(run_tag="laya_test")
    script = (Path(receipt["staged"]) / "laya_finetune_eval.py").read_text()
    import ast

    ast.parse(script)
    laya_lane._kernel_script_gate(script)
    laya_lane._module_scope_gate(script)
    # the checkpoint load + held-out evaluation are embedded
    assert "load_checkpoint(" in script
    assert "calibration_records(" in script
    assert "evaluate_records(" in script
    assert 'EVAL_JSONL = "test.jsonl"' in script
    assert 'EVAL_SPLIT = "test"' in script
    assert '"eval_mode": "held_out"' in script
    assert '"is_held_out": True' in script
    assert 'WORKING / "eval_report.json"' in script
    assert '"rl_agent_config.json"' in script
    # the held-out split is the ONLY required attached input in the preflight
    assert '_runtime_files = ("test.jsonl",)' in script
    # NO training and NO Hub fetch (no snapshot_download call, no trainer)
    assert "snapshot_download" not in script
    assert "huggingface_hub" not in script
    assert "train_model(" not in script
    assert "train_cli" not in script
    assert "laya-train" not in script
    # every @TOKEN@ was substituted (a missing value would leak a marker)
    import re

    assert not re.search(r"@[A-Z][A-Z0-9_]*@", script)


def test_finetune_eval_kernel_fails_loud_without_checkpoint_source(
        tmp_path, monkeypatch):
    _spec(tmp_path, monkeypatch, finetune_ckpt_dataset=None)
    _corpus(tmp_path)
    _hermetic_staging(monkeypatch)
    with pytest.raises(RuntimeError, match="checkpoint"):
        laya_lane.stage_finetune_eval_kernel(run_tag="laya_test")


def _fake_laya_train():
    """A stub `laya.train` so the local CPU helper is pinned offline."""
    import sys
    import types

    class _Model:
        def to(self, device):
            return self

        def eval(self):
            return self

    train = types.ModuleType("laya.train")

    def load_checkpoint(path):
        return _Model(), object(), {"max_len": 256, "head_max_len": 128}

    def uses_parallel_layout(cfg):
        return False

    def read_jsonl(path):
        return [{"state": "s0"}, {"state": "s1"}]

    def items_from_rows(tok, rows, max_len, head_max_len, label_smoothing=0.0):
        return [{"qtype": 0, "k": 2} for _ in rows], {}

    def calibration_records(model, tok, items, device, max_len, head_max_len,
                            batch_size=16, parallel=False):
        return [(0, [0.1, 0.9], [0, 1], 2) for _ in items]

    def evaluate_records(records, temperature=None,
                         temperature_by_options=None):
        return {"items": len(records), "accuracy": 1.0, "ece": 0.05,
                "brier": 0.1, "mean_confidence": 0.95}

    def fit_temperature_map(records):
        return {"temperature": [1.0, 1.0], "temperature_by_options": {},
                "n_by_bucket": {"noul:2": len(records)}}

    train.fit_abstention_calls = []

    def fit_abstention_thresholds(records, temperature, temperature_by_options,
                                  *, target_error=0.10, min_bucket_n=100,
                                  binning_map=None, conservative=True):
        train.fit_abstention_calls.append({
            "temperature": temperature,
            "temperature_by_options": temperature_by_options,
            "target_error": target_error, "min_bucket_n": min_bucket_n})
        return {"noul:2": 0.85}

    train.load_checkpoint = load_checkpoint
    train.uses_parallel_layout = uses_parallel_layout
    train.read_jsonl = read_jsonl
    train.items_from_rows = items_from_rows
    train.calibration_records = calibration_records
    train.evaluate_records = evaluate_records
    train.fit_temperature_map = fit_temperature_map
    train.fit_abstention_thresholds = fit_abstention_thresholds
    laya = types.ModuleType("laya")
    laya.train = train
    return laya, train


def test_local_eval_helper_runs_on_cpu_and_writes_held_out_report(
        tmp_path, monkeypatch):
    import sys

    _spec(tmp_path, monkeypatch)
    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir()
    (ckpt / "rl_agent_config.json").write_text("{}", encoding="utf-8")
    data = tmp_path / "test.jsonl"
    data.write_text('{"state": "s"}\n', encoding="utf-8")
    laya, train = _fake_laya_train()
    monkeypatch.setitem(sys.modules, "laya", laya)
    monkeypatch.setitem(sys.modules, "laya.train", train)
    out = tmp_path / "out"
    report = laya_lane.local_eval_checkpoint(
        ckpt, eval_data=data, out_dir=out, split="test")
    assert report["eval_mode"] == "held_out"
    assert report["is_held_out"] is True
    assert report["device"] == "cpu"
    assert report["items"] == 2
    assert report["after"]["accuracy"] == 1.0
    assert (out / "eval_report.json").is_file()
    receipt = json.loads(
        (out / "laya_finetune-eval.receipt.json").read_text())
    assert receipt["is_held_out"] is True
    assert receipt["device"] == "cpu"
    assert receipt["eval_split"] == "test"


def test_local_eval_helper_fails_loud_on_missing_checkpoint(
        tmp_path, monkeypatch):
    _spec(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError, match="rl_agent_config.json"):
        laya_lane.local_eval_checkpoint(tmp_path / "nope",
                                        eval_data=tmp_path / "x.jsonl")


# ── the eval path's calibration/abstention selection is reported ───────────
def test_rendered_eval_kernel_bakes_the_calibration_selection(
        tmp_path, monkeypatch):
    """`laya.eval_calibration` flows YAML -> EVAL_CALIBRATION -> the kernel."""
    import ast

    _spec(tmp_path, monkeypatch)
    _corpus(tmp_path)
    _hermetic_staging(monkeypatch)
    receipt = laya_lane.stage_finetune_eval_kernel(run_tag="laya_test")
    script = (Path(receipt["staged"]) / "laya_finetune_eval.py").read_text()
    baked = None
    for node in ast.walk(ast.parse(script)):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "EVAL_CALIBRATION":
                    baked = ast.literal_eval(node.value)
    # the default selection reproduces the landed eval exactly
    assert baked == {"temperature": True, "abstention": False,
                     "target_error": 0.10, "min_abstain_n": 10,
                     "min_confidence": None}
    assert receipt["eval_calibration"] == baked
    # the kernel CONSUMES laya's own fits and never reimplements them
    assert "fit_temperature_map(" in script
    assert "fit_abstention_thresholds(" in script
    assert "def fit_eval_calibration" in script
    # the fitted values are reported (temperature always; abstention opt-in)
    assert '"temperature": calibration["temperature"]' in script
    assert '"abstention_thresholds"' in script
    assert '"min_confidence"' in script


def test_rendered_eval_kernel_honours_non_default_calibration(
        tmp_path, monkeypatch):
    import ast

    _spec(tmp_path, monkeypatch, eval_calibration={
        "abstention": True, "target_error": 0.3, "min_abstain_n": 4,
        "min_confidence": 0.75})
    _corpus(tmp_path)
    _hermetic_staging(monkeypatch)
    receipt = laya_lane.stage_finetune_eval_kernel(run_tag="laya_test")
    script = (Path(receipt["staged"]) / "laya_finetune_eval.py").read_text()
    baked = None
    for node in ast.walk(ast.parse(script)):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "EVAL_CALIBRATION":
                    baked = ast.literal_eval(node.value)
    assert baked == {"temperature": True, "abstention": True,
                     "target_error": 0.3, "min_abstain_n": 4,
                     "min_confidence": 0.75}
    assert receipt["eval_calibration"]["abstention"] is True


def _local_eval(tmp_path, monkeypatch, **updates):
    import sys

    spec = _spec(tmp_path, monkeypatch, **updates)
    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir(exist_ok=True)
    (ckpt / "rl_agent_config.json").write_text("{}", encoding="utf-8")
    data = tmp_path / "test.jsonl"
    data.write_text('{"state": "s"}\n', encoding="utf-8")
    laya, train = _fake_laya_train()
    monkeypatch.setitem(sys.modules, "laya", laya)
    monkeypatch.setitem(sys.modules, "laya.train", train)
    out = tmp_path / "out"
    report = laya_lane.local_eval_checkpoint(
        ckpt, eval_data=data, out_dir=out, split="test")
    receipt = json.loads(
        (out / "laya_finetune-eval.receipt.json").read_text())
    return spec, report, receipt, train


def test_local_eval_default_report_shape_is_unchanged(tmp_path, monkeypatch):
    """Default config: no new report keys, no abstention fit call."""
    _spec_, report, receipt, train = _local_eval(tmp_path, monkeypatch)
    assert "abstention_thresholds" not in report
    assert "min_confidence" not in report
    assert "n_by_bucket" not in report
    assert report["temperature"] == [1.0, 1.0]
    assert train.fit_abstention_calls == []
    assert receipt["eval_calibration"] == {
        "temperature": True, "abstention": False, "target_error": 0.10,
        "min_abstain_n": 10, "min_confidence": None}


def test_local_eval_reports_fitted_abstention_and_min_confidence(
        tmp_path, monkeypatch):
    _spec_, report, receipt, train = _local_eval(
        tmp_path, monkeypatch,
        eval_calibration={"abstention": True, "target_error": 0.2,
                          "min_abstain_n": 3, "min_confidence": 0.75})
    # the fitted per-bucket thresholds ride the report (laya's own fit)
    assert report["abstention_thresholds"] == {"noul:2": 0.85, "default": 0.75}
    assert report["min_confidence"] == 0.75
    assert report["n_by_bucket"] == {"noul:2": 2}
    # the fit consumed laya's calibrated temperature + the YAML knobs
    assert train.fit_abstention_calls == [{
        "temperature": [1.0, 1.0], "temperature_by_options": {},
        "target_error": 0.2, "min_bucket_n": 3}]
    assert receipt["eval_calibration"]["min_confidence"] == 0.75


def _metric_block(items=2):
    return {"items": items, "loss": 0.5, "accuracy": 1.0,
            "mean_confidence": 0.95, "ece": 0.05, "brier": 0.1,
            "brier_top1": 0.1}


def test_corpus_traceability_reports_the_min_confidence():
    """The lane adapter surfaces the reported abstention scalar (additive)."""
    digest = "a" * 64
    report = {"after": _metric_block(), "rows": 2, "items": 2,
              "eval_split": "test", "batch_size": 16,
              "abstention_thresholds": {"noul:2": 0.7}}
    document = laya_lane.corpus_traceability(
        report, model_id="laya", digests={"corpus_sha256": digest})
    # the fitted map's "default" sentinel is the gate when nothing is pinned
    assert document.provenance.min_confidence is None
    report["abstention_thresholds"] = {"noul:2": 0.7, "default": 0.7}
    document = laya_lane.corpus_traceability(
        report, model_id="laya", digests={"corpus_sha256": digest})
    assert document.provenance.min_confidence == 0.7
    # an explicit pin wins over the fitted default
    report["min_confidence"] = 0.9
    document = laya_lane.corpus_traceability(
        report, model_id="laya", digests={"corpus_sha256": digest})
    assert document.provenance.min_confidence == 0.9


def test_kernel_and_lane_calibration_helpers_stay_in_lockstep(
        tmp_path, monkeypatch):
    """The staged kernel helper and the local lane helper agree exactly.

    The kernel is a staged string (it cannot import `cli.laya_lane`), so its
    `fit_eval_calibration` is a deliberate twin of the lane's. This pins that
    the twin is never allowed to drift: same inputs -> same result.
    """
    import ast

    _spec(tmp_path, monkeypatch, eval_calibration={
        "abstention": True, "target_error": 0.2, "min_abstain_n": 3,
        "min_confidence": 0.75})
    _corpus(tmp_path)
    _hermetic_staging(monkeypatch)
    receipt = laya_lane.stage_finetune_eval_kernel(run_tag="laya_test")
    script = (Path(receipt["staged"]) / "laya_finetune_eval.py").read_text()
    tree = ast.parse(script)
    baked = None
    node = None
    for item in tree.body:
        if isinstance(item, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "EVAL_CALIBRATION"
                for t in item.targets):
            baked = ast.literal_eval(item.value)
        if isinstance(item, ast.FunctionDef) and \
                item.name == "fit_eval_calibration":
            node = item
    assert baked is not None and node is not None
    namespace = {"EVAL_CALIBRATION": baked}
    exec(ast.get_source_segment(script, node), namespace)
    kernel_helper = namespace["fit_eval_calibration"]

    laya, train = _fake_laya_train()
    records = [(0, [0.1, 0.9], [0, 1], 2)]
    assert kernel_helper(train, records) == laya_lane.fit_eval_calibration(
        train, records, baked) == {
            "temperature": [1.0, 1.0], "temperature_by_options": {},
            "n_by_bucket": {"noul:2": 1},
            "abstention_thresholds": {"noul:2": 0.85, "default": 0.75},
            "min_confidence": 0.75}


def test_staged_finetune_kernel_bakes_wandb_key_and_never_leaks_it(
        tmp_path, monkeypatch):
    """Env/secret injection: the WANDB key is read at STAGE time and baked into
    the kernel only; the receipt MUST NOT carry it."""
    _spec(tmp_path, monkeypatch)
    _corpus(tmp_path)
    _hermetic_staging(monkeypatch)
    monkeypatch.setattr(laya_lane, "_env_value",
                        lambda name: "sekret" if name == "WANDB_API_KEY" else None)
    monkeypatch.setattr(laya_lane, "_wandb_project", lambda: "e-r")
    receipt = laya_lane.stage_finetune_kernel(run_tag="laya_test")
    script = (Path(receipt["staged"]) / "laya_finetune.py").read_text()
    assert 'WANDB_API_KEY = "sekret"' in script
    assert 'WANDB_PROJECT = "e-r"' in script
    # the secret is baked into the staged kernel, never into the receipt
    assert "sekret" not in json.dumps(receipt)
