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


def test_smoke_corpus_samples_every_stratum_not_a_prefix(tmp_path):
    """The subset is a stratified sample of the whole split, not its prefix.

    The builder emits the state/mask rows LAST, so a prefix cut drops
    ``single_state`` entirely (the original bias); the subset must keep every
    ``difficulty_slice`` present, in proportion, deterministically.
    """
    source = tmp_path / "full"
    source.mkdir()

    def rows(split: str, single: int, pair: int) -> str:
        return "".join(
            json.dumps({"state": f"{split}-state-{i}",
                        "difficulty_slice": "single_state"}) + "\n"
            for i in range(single)) + "".join(
            json.dumps({"state": f"{split}-pair-{i}",
                        "difficulty_slice": "one_diff"}) + "\n"
            for i in range(pair))

    for name, single, pair in (("train.jsonl", 20, 180),
                               ("dev.jsonl", 10, 90), ("test.jsonl", 10, 90)):
        (source / name).write_text(rows(name.split(".")[0], single, pair),
                                   encoding="utf-8")
    smoke = FinetuneSmokeSpec(train_rows=100, dev_rows=50, test_rows=50)
    dest = tmp_path / "smoke"
    receipt = FinetuneSmokeCorpus(smoke, source_dir=source,
                                  dest_dir=dest).build(seed=1729)
    assert receipt["counts"] == {"train": 100, "dev": 50, "test": 50}
    # both strata survive the cut in the source proportion (10% single_state)
    strata = receipt["strata"]["train"]
    assert 0 < strata["single_state"] < strata["one_diff"]
    # deterministic: a rerun reproduces every byte + digest
    again = FinetuneSmokeCorpus(smoke, source_dir=source,
                                dest_dir=tmp_path / "smoke2").build(seed=1729)
    assert (tmp_path / "smoke2/train.jsonl").read_bytes() == \
        (dest / "train.jsonl").read_bytes()
    assert again["files"]["train.jsonl"] == receipt["files"]["train.jsonl"]
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
