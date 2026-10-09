"""Offline staging + teardown pins for the holdout-eval surface.

Covers the surfaces the audit flagged as untested:
  * `stage_holdout_dataset_payload` (the composed holdout JSONL + receipt);
  * `stage_holdout_eval_kernel` (render, gates, no leftover tokens, receipt);
  * `LayaLane.stage()` routing `holdout-eval` (the DECISION_BINDINGS gap);
  * the session-id / stop dry-run paths.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from core.laya_config import LayaSpec
from cli import laya_lane

REAL_ROOT = Path(__file__).resolve().parents[1]
REVISION = "abc123def"


def _spec(tmp_path, monkeypatch, **updates):
    cfg = LayaSpec(**updates)
    monkeypatch.setattr(laya_lane, "_spec", lambda: cfg)
    monkeypatch.setattr(laya_lane, "TRAIN_ROOT", tmp_path)
    monkeypatch.setattr(laya_lane, "_git_revision", lambda: REVISION)
    return cfg


def _fixtures(tmp_path, spec):
    # the pair composer is loaded from TRAIN_ROOT/scripts (reused, not copied)
    (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy(REAL_ROOT / "scripts/laya_metrics_pairs.py",
                tmp_path / "scripts/laya_metrics_pairs.py")
    question = tmp_path / spec.question_schema
    question.parent.mkdir(parents=True, exist_ok=True)
    question.write_text(json.dumps({"questions": {
        "identity_claim": {"type": "noul", "instructions": "same?"}}}),
        encoding="utf-8")
    catalog = tmp_path / "data/track_setup/eligible_catalog.csv"
    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_text(
        "gtin,attribute\n"
        '1,"Volume: 500; Pack Type: Bottle"\n'
        '2,"Volume: 750; Pack Type: Bottle"\n', encoding="utf-8")
    holdout = tmp_path / spec.holdout_csv
    holdout.parent.mkdir(parents=True, exist_ok=True)
    holdout.write_text(
        "gtin1,gtin2,label,stratum,component\n"
        "1,2,1,real,c1\n2,1,0,p0,c2\n", encoding="utf-8")
    return question, catalog, holdout


def _hermetic(monkeypatch):
    def fake_run(args, **kwargs):
        if "rev-parse" in args:
            return subprocess.CompletedProcess(args, 0, stdout=REVISION,
                                               stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)


def test_stage_holdout_dataset_payload_writes_jsonl_and_receipt(
        tmp_path, monkeypatch):
    spec = _spec(tmp_path, monkeypatch)
    _hermetic(monkeypatch)
    question, catalog, holdout = _fixtures(tmp_path, spec)
    receipt = laya_lane.stage_holdout_dataset_payload(
        dataset_slug=spec.holdout_dataset_slug, holdout_csv=holdout,
        catalog_path=catalog, question_path=question)
    payload = Path(receipt["payload"])
    jsonl = payload / laya_lane.HOLDOUT_JSONL
    assert jsonl.is_file()
    assert receipt["rows"] == 2 and receipt["skipped"] == 0
    assert len(jsonl.read_text(encoding="utf-8").strip().splitlines()) == 2
    assert (payload / laya_lane.DATASET_METADATA_FILE).is_file()
    assert receipt["files"][laya_lane.HOLDOUT_JSONL]


def test_stage_holdout_eval_kernel_renders_and_writes_receipt(
        tmp_path, monkeypatch):
    spec = _spec(tmp_path, monkeypatch)
    _hermetic(monkeypatch)
    _fixtures(tmp_path, spec)
    receipt = laya_lane.stage_holdout_eval_kernel(run_tag="laya_test")
    stage = Path(receipt["staged"])
    assert receipt["kind"] == "holdout-eval"
    assert receipt["code_file"] == laya_lane.HOLDOUT_EVAL_CODE_FILE
    assert receipt["threshold"] == spec.holdout_eval_threshold
    assert receipt["n_boot"] == spec.holdout_eval_bootstrap
    script = (stage / laya_lane.HOLDOUT_EVAL_CODE_FILE).read_text()
    import ast
    import re

    ast.parse(script)
    laya_lane._kernel_script_gate(script)
    laya_lane._module_scope_gate(script)
    assert not re.search(r"@[A-Z][A-Z0-9_]*@", script)
    # The push gate's attached-inputs inventory is the holdout JSONL ONLY: the
    # questions are embedded per row, so a separate schema file is never
    # attached (the decision kernel's inventory would fail the staged payload).
    inventory = {}
    for node in ast.walk(ast.parse(script)):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and \
                        target.id == "_runtime_files":
                    inventory["_runtime_files"] = ast.literal_eval(node.value)
    assert inventory["_runtime_files"] == (laya_lane.HOLDOUT_JSONL,)
    payload = stage / laya_lane.DATASET_PAYLOAD_DIR
    assert (payload / laya_lane.HOLDOUT_JSONL).is_file()
    # the checkpoint dataset rides alongside the holdout dataset
    metadata = json.loads((stage / "kernel-metadata.json").read_text())
    assert spec.holdout_dataset_slug in metadata["dataset_sources"]


def test_layalane_stage_routes_holdout_eval(tmp_path, monkeypatch):
    """The DECISION_BINDINGS gap: the kind `LayaLane.stage()` handles is
    registered, so the class surface routes it to the holdout kernel."""
    spec = _spec(tmp_path, monkeypatch)
    _hermetic(monkeypatch)
    _fixtures(tmp_path, spec)
    assert "holdout-eval" in laya_lane.DECISION_BINDINGS
    assert "holdout-eval" in laya_lane.GPU_KINDS
    receipt = laya_lane.LayaLane("kaggle").stage("holdout-eval")
    assert receipt["kind"] == "holdout-eval"


def test_session_id_and_stop_paths_are_offline():
    plan = laya_lane.stop_kaggle_kernel("owner/kernel", execute=False)
    assert plan["mode"] == "dry-run"
    assert plan["kernel"] == "owner/kernel"
    # no launch-recorded id on this box -> None, never a crash
    assert laya_lane.recorded_session_id("owner/kernel") is None


def test_kernel_slug_covers_holdout_eval(monkeypatch):
    spec = LayaSpec(holdout_eval_kernel_slug="owner/holdout-eval")
    monkeypatch.setattr(laya_lane, "_spec", lambda: spec)
    assert laya_lane.kernel_slug("holdout-eval") == "owner/holdout-eval"


def test_stage_finetune_ckpt_dataset_payload_copies_the_checkpoint(
        tmp_path, monkeypatch):
    """The recovered checkpoint stages as dataset_payload/checkpoint/."""
    spec = _spec(tmp_path, monkeypatch)
    source = tmp_path / "recovered"
    (source / "tokenizer").mkdir(parents=True)
    (source / "encoder").mkdir(parents=True)
    (source / "rl_agent_config.json").write_text("{}", encoding="utf-8")
    (source / "model.safetensors").write_bytes(b"the-real-weights")
    (source / "encoder/config.json").write_text("{}", encoding="utf-8")
    (source / "tokenizer/tokenizer.json").write_text("{}", encoding="utf-8")

    receipt = laya_lane.stage_finetune_ckpt_dataset_payload(
        dataset_slug=spec.finetune_ckpt_dataset, checkpoint_dir=source)

    payload = Path(receipt["payload"])
    assert (payload / "checkpoint/model.safetensors").read_bytes() == \
        b"the-real-weights"
    assert (payload / "checkpoint/rl_agent_config.json").is_file()
    assert (payload / laya_lane.DATASET_METADATA_FILE).is_file()
    assert receipt["member"] == "checkpoint"
    assert receipt["files"]["checkpoint/model.safetensors"]


def test_stage_finetune_ckpt_dataset_payload_fails_loud_without_config(
        tmp_path, monkeypatch):
    spec = _spec(tmp_path, monkeypatch)
    source = tmp_path / "recovered"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"stub")
    with pytest.raises(FileNotFoundError, match="rl_agent_config.json"):
        laya_lane.stage_finetune_ckpt_dataset_payload(
            dataset_slug=spec.finetune_ckpt_dataset, checkpoint_dir=source)
