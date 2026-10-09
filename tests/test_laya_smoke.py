"""End-to-end smoke of the finetune kernel — offline pins.

The smoke is the smallest honest validation of the NEW finetune kernel (dials +
profiler + early-stop/dev-eval + the fail-loud fetchers): it reuses the SAME
kernel template, staging and push surface, but pins a tiny subset corpus and
DEDICATED slugs (``laya.finetune_smoke``). ``laya.finetune_smoke.device`` is
the ONE runtime source (``"cpu"`` default, ``"cuda"`` for the GPU path). These
pins prove the staged payload matches the device selection, carries the smoke
dials, and that the subset builder is deterministic and receipted.
"""
from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from cli import laya_lane
from cli.laya_smoke import FinetuneSmokeCorpus
from core.laya_config import FinetuneSmokeSpec, LayaSpec

REVISION_PIN = "abc123def"
SMOKE_KERNEL = "fbarulli/er-laya-finetune-smoke"
SMOKE_DATASET = "fbarulli/er-laya-train-smoke"


def _spec(tmp_path, monkeypatch, **updates):
    smoke = FinetuneSmokeSpec(kernel_slug=SMOKE_KERNEL,
                              dataset_slug=SMOKE_DATASET)
    cfg_spec = LayaSpec(**{
        "finetune_kernel_slug": "fbarulli/er-laya-finetune",
        "finetune_dataset_slug": "fbarulli/er-laya-train",
        "base_model_dataset": "fbarulli/er-laya-base",
        "finetune_smoke": smoke,
        **updates,
    })
    monkeypatch.setattr(laya_lane, "_spec", lambda: cfg_spec)
    monkeypatch.setattr(laya_lane, "TRAIN_ROOT", tmp_path)
    return cfg_spec


def _corpus(tmp_path, rows=3):
    corpus = tmp_path / "data/laya"
    corpus.mkdir(parents=True, exist_ok=True)
    for name in ("train.jsonl", "dev.jsonl", "test.jsonl"):
        (corpus / name).write_text(
            "".join(json.dumps({"state": f"{name}-{i}"}) + "\n"
                    for i in range(rows)), encoding="utf-8")
    (corpus / "receipt.json").write_text('{"seed": 1729}\n',
                                         encoding="utf-8")


def _hermetic_staging(monkeypatch):
    monkeypatch.setattr(laya_lane, "_git_revision", lambda: REVISION_PIN)

    def fake_run(args, **kwargs):
        if "rev-parse" in args:
            return subprocess.CompletedProcess(args, 0, stdout=REVISION_PIN,
                                               stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)


def _baked(script: str, name: str):
    for node in ast.walk(ast.parse(script)):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not baked into the kernel")


def test_smoke_corpus_is_deterministic_and_receipted(tmp_path):
    source = tmp_path / "full"
    source.mkdir()
    for name, total in (("train.jsonl", 10), ("dev.jsonl", 8),
                        ("test.jsonl", 9)):
        (source / name).write_text(
            "".join(json.dumps({"state": f"{name.split('.')[0]}-{i}"}) + "\n"
                    for i in range(total)), encoding="utf-8")
    smoke = FinetuneSmokeSpec(train_rows=4, dev_rows=3, test_rows=2)
    dest = tmp_path / "smoke"
    receipt = FinetuneSmokeCorpus(smoke, source_dir=source,
                                  dest_dir=dest).build(seed=1729)
    assert receipt["counts"] == {"train": 4, "dev": 3, "test": 2}
    assert receipt["seed"] == 1729
    assert [line for line in (dest / "train.jsonl").read_text().splitlines()] == [
        json.dumps({"state": f"train-{i}"}) for i in range(4)]
    # the receipt records the source digests + its own files inventory
    assert set(receipt["files"]) == {
        "train.jsonl", "dev.jsonl", "test.jsonl", "receipt.json"}
    assert set(receipt["source_sha256"]) == {
        "train.jsonl", "dev.jsonl", "test.jsonl"}


def test_smoke_kernel_stages_cpu_with_dedicated_slugs_and_dials(
        tmp_path, monkeypatch):
    _spec(tmp_path, monkeypatch)
    _corpus(tmp_path)
    _hermetic_staging(monkeypatch)
    receipt = laya_lane.stage_finetune_kernel(run_tag="laya_smoke",
                                              smoke=True)
    stage = Path(receipt["staged"])
    assert stage.name == "finetune-smoke"
    metadata = json.loads((stage / "kernel-metadata.json").read_text())
    # a smoke NEVER requests a GPU, and never attaches the production corpus
    assert metadata["enable_gpu"] is False
    assert metadata["dataset_sources"] == [SMOKE_DATASET,
                                           "fbarulli/er-laya-base"]
    assert receipt["kind"] == "finetune-smoke"
    assert receipt["device"] == "cpu"
    assert receipt["smoke"] is True
    script = (stage / "laya_finetune.py").read_text()
    # the smoke dials are baked from the SSOT, and the receipt member name
    # matches what the fetcher requires for this kind
    config = _baked(script, "FINETUNE_CONFIG")
    assert (config["epochs"], config["micro_batch"],
            config["grad_accum"]) == (1, 1, 1)
    assert _baked(script, "FINETUNE_DEVICE") == "cpu"
    assert _baked(script, "RECEIPT_NAME") == "laya_finetune-smoke.receipt.json"
    # the rendered payload still clears both staging-time AST gates
    laya_lane._kernel_script_gate(script)
    laya_lane._module_scope_gate(script)


def test_smoke_kernel_stages_gpu_when_device_is_cuda(tmp_path, monkeypatch):
    _spec(tmp_path, monkeypatch, finetune_smoke=FinetuneSmokeSpec(
        kernel_slug=SMOKE_KERNEL, dataset_slug=SMOKE_DATASET, device="cuda"))
    _corpus(tmp_path)
    _hermetic_staging(monkeypatch)
    receipt = laya_lane.stage_finetune_kernel(run_tag="laya_smoke",
                                              smoke=True)
    stage = Path(receipt["staged"])
    metadata = json.loads((stage / "kernel-metadata.json").read_text())
    # device="cuda" is the ONE source: metadata + baked env + receipt agree
    assert metadata["enable_gpu"] is True
    assert receipt["device"] == "cuda"
    assert receipt["gpu"] == "T4 (single)"
    script = (stage / "laya_finetune.py").read_text()
    assert _baked(script, "FINETUNE_DEVICE") == "cuda"


def test_smoke_device_rejects_unknown_values():
    with pytest.raises(ValidationError):
        FinetuneSmokeSpec(device="tpu")


def test_smoke_kind_routes_through_the_decision_dispatch(tmp_path, monkeypatch):
    _spec(tmp_path, monkeypatch)
    _corpus(tmp_path)
    _hermetic_staging(monkeypatch)
    receipt = laya_lane.stage_decision_kernel(
        decision_kind="finetune-smoke", run_tag="laya_smoke")
    assert receipt["kind"] == "finetune-smoke"
    assert receipt["smoke"] is True


def test_kernel_slug_resolves_the_smoke_kernel(monkeypatch):
    monkeypatch.setattr(laya_lane, "_spec", lambda: LayaSpec(
        finetune_smoke=FinetuneSmokeSpec(kernel_slug=SMOKE_KERNEL)))
    assert laya_lane.kernel_slug("finetune-smoke") == SMOKE_KERNEL


def test_smoke_kind_needs_its_own_slugs(tmp_path, monkeypatch):
    _spec(tmp_path, monkeypatch,
          finetune_smoke=FinetuneSmokeSpec())
    _corpus(tmp_path)
    _hermetic_staging(monkeypatch)
    try:
        laya_lane.stage_finetune_kernel(run_tag="laya_smoke", smoke=True)
    except RuntimeError as error:
        assert "finetune_smoke.kernel_slug" in str(error)
    else:  # pragma: no cover - a missing slug must never stage
        raise AssertionError("unset smoke kernel slug staged a payload")
