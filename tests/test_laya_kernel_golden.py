"""Golden byte-hash tripwire for every staged laya kernel script.

`src/cli/laya_lane.py` carries the kernel payloads as in-source triple-quoted
constants that are token-substituted and written under
``results/laya_lane/<kind>/<decision>/``. The bit-identical mandate means the
generated script BYTES may never move unless a test deliberately repins them.

This is the single tripwire for the constant-extraction refactor: it stages
every kind offline (hermetic git + config + env) with a fixed run tag and
asserts the sha256 of each generated script against a pinned digest. A moved,
re-wrapped, re-ordered or byte-shifted literal fails here first.

The eight staged surfaces map 1:1 to the embedded text blocks:
``attribute`` / ``identity`` (``DECISION_KERNEL_SCRIPT``), ``laya-cli-eval``
(``EVAL_KERNEL_SCRIPT``), ``colab`` (``NOTEBOOK_SCRIPT``), ``finetune`` /
``finetune-smoke`` (``FINETUNE_KERNEL_SCRIPT``), ``finetune-eval``
(``FINETUNE_EVAL_KERNEL_SCRIPT``) and ``holdout-eval``
(``HOLDOUT_EVAL_KERNEL_SCRIPT``).
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from core.common import training_cfg as _tcfg
from core.laya_config import LayaSpec
from cli import laya_lane

REAL_ROOT = Path(__file__).resolve().parents[1]
REVISION = "0" * 40
RUN_TAG = "laya_golden"

# Pinned sha256 of each generated script's bytes. Fill in from a deliberate
# re-render ONLY when a kernel change is intended.
GOLDEN_SHA256: dict[str, str] = {
    "attribute": "de5dbcdc02eafd4d391eb6e0f93980b38bbdb673710d8d5f938dc13f2d6c38ea",
    "identity": "e1ffd3e9b6d5ce3b241e77496b5d3939d899dd01fb34c4bf8a0d23c1099ebad8",
    "laya-cli-eval": "4193b9c3110394d1ce8dd962125812ed98153681bb70a978fca75b593bc5b0f7",
    "colab": "8de6069ff99110ba9ad165a5046bcb11ac3357f75694393461c15b1a8e6fc686",
    # re-pinned ONCE for the combined candidate kernel text: the deterministic-DDP
    # fix (find_unused_parameters gone, DeterministicDdp.wrap + the zero-weight
    # loss guard, unconditional epoch stop broadcast), the laya-lanes
    # holdout/HPO surfaces, and the laya-delete unbuffered-child fix
    # (os.environ.setdefault("PYTHONUNBUFFERED", "1") so the live W&B/output.log
    # readers see lines immediately) are all merged. Only finetune/finetune-smoke
    # carry the unbuffered child block; holdout-eval keeps its lanes preflight.
    #
    # finetune / finetune-smoke re-pinned again for the fused-safe optimizer-state
    # fix: OptimizerStateCaster now reconciles state to the parameter dtype
    # (native AdamW requires state.dtype == param.dtype on single/foreach/fused)
    # instead of casting it to bf16, which left fused AdamW with mismatched
    # tensors. Only those two surfaces carry the perf-patch control logic. Re-
    # pinned once more for the live `log/line` text channel (WandbLogSink): the
    # kernel boot/trial/epoch lines stream through `wandb.log` while the run is
    # live, since W&B exposes `output.log` only on flush/end, and for the
    # real-time wandb emissions (run events, GPU sampler, timing sink).
    "finetune": "33ac6d39f8edb6758f5383d054356838209cd3bc9bb825ad78ef0da2cd748ee4",
    "finetune-smoke": "0c9f661218b27d8536a5ec4adbfd632740d92a1e7c597141ac2f37a0039d36d2",
    "finetune-eval": "9d91380b42d2c5b098e6c9ab9ee7d7e78bd2ec11509e85d2f4fb9c098150fa1d",
    "holdout-eval": "ff5dedb26f27694b148879405ef010e8c28c90634a8c11eacc34d0c167fa89f5",
}


def _golden_spec() -> LayaSpec:
    """A fully explicit spec (independent of the ambient config file).

    Only ``finetune_smoke`` lacks production defaults; everything else is a
    schema default, so this tripwire pins rendering, not the YAML.
    """
    return LayaSpec(finetune_smoke={
        "kernel_slug": "fbarulli/er-laya-finetune-smoke",
        "dataset_slug": "fbarulli/er-laya-train-smoke",
        "corpus_dir": "results/laya_lane/smoke_corpus",
        "epochs": 1, "micro_batch": 1, "grad_accum": 1,
    })


def _hermetic(tmp_path: Path, monkeypatch) -> None:
    """Freeze git + config + env so the render is reproducible."""
    spec = _golden_spec()
    monkeypatch.setattr(laya_lane, "_spec", lambda: spec)
    monkeypatch.setattr(laya_lane, "TRAIN_ROOT", tmp_path)
    monkeypatch.setattr(laya_lane, "_git_revision", lambda: REVISION)
    monkeypatch.setattr(laya_lane, "_env_value", lambda name: None)
    monkeypatch.setattr(laya_lane, "_wandb_project", lambda: "golden-project")
    base = _tcfg()
    forced = base.model_copy(update={"kaggle": base.kaggle.model_copy(update={
        "branch": "main", "repository": "https://example/repo.git"})})
    monkeypatch.setattr(laya_lane, "training_cfg", lambda: forced)

    def fake_run(args, **kwargs):
        if "rev-parse" in args:
            return subprocess.CompletedProcess(args, 0, stdout=REVISION,
                                               stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)


def _question_schema(tmp_path: Path) -> None:
    path = tmp_path / "config/laya.question.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"questions": {
        "identity_claim": {"type": "noul", "instructions": "same item?"},
        "attribute_alignment": {"type": "choice", "instructions": "verdict?",
                                "criteria": {"aligned": None}},
        "package_state": {"type": "noul", "instructions": "has pack?"}}}),
        encoding="utf-8")


def _decision_fixture(tmp_path: Path, monkeypatch) -> None:
    import core.common as core_common

    dataset = tmp_path / "dataset.csv"
    dataset.write_text(
        "sku_id,sku_name_eng,attribute\n"
        "SKU0,Name 0 500 ml,Volume: 500; Brand: A\n"
        "SKU1,Name 1 750 ml,Volume: 750; Brand: B\n", encoding="utf-8")
    final_validation = tmp_path / "final_validation.csv"
    final_validation.write_text(
        "gtin1,gtin2,true_label,attribute_pairs\n"
        "1,2,1,Vol 500; Pack 6\n"
        "3,4,0,Vol 750; Pack 6\n", encoding="utf-8")
    monkeypatch.setitem(core_common.F, "dataset", dataset)
    monkeypatch.setitem(core_common.F, "final_validation", final_validation)


def _corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "data/laya"
    corpus.mkdir(parents=True, exist_ok=True)
    for name in ("train.jsonl", "dev.jsonl", "test.jsonl"):
        (corpus / name).write_text(
            '{"state": "a", "questions": {}, "expected": {}}\n'
            '{"state": "b", "questions": {}, "expected": {}}\n',
            encoding="utf-8")
    (corpus / "receipt.json").write_text('{"seed": 1729}\n', encoding="utf-8")


def _holdout_fixture(tmp_path: Path, spec: LayaSpec) -> None:
    (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy(REAL_ROOT / "scripts/laya_metrics_pairs.py",
                tmp_path / "scripts/laya_metrics_pairs.py")
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


def _staged_scripts(tmp_path: Path, monkeypatch) -> dict[str, bytes]:
    """Stage all eight surfaces and return ``{label: script bytes}``."""
    _hermetic(tmp_path, monkeypatch)
    _question_schema(tmp_path)
    _decision_fixture(tmp_path, monkeypatch)
    _corpus(tmp_path)
    spec = _golden_spec()
    _holdout_fixture(tmp_path, spec)

    def script(receipt: dict) -> bytes:
        stage = Path(receipt["staged"])
        return (stage / receipt["code_file"]).read_bytes()

    staged: dict[str, bytes] = {}
    for kind in ("attribute", "identity", "laya-cli-eval"):
        receipt = laya_lane.stage_decision_kernel(
            decision_kind=kind, run_tag=RUN_TAG)
        staged[kind] = script(receipt)
    colab = laya_lane.stage_colab_notebook(
        decision_kind="identity", run_tag=RUN_TAG)
    staged["colab"] = Path(colab["notebook"]).read_bytes()
    finetune = laya_lane.stage_finetune_kernel(run_tag=RUN_TAG)
    staged["finetune"] = script(finetune)
    smoke = laya_lane.stage_finetune_kernel(run_tag=RUN_TAG, smoke=True)
    staged["finetune-smoke"] = script(smoke)
    eval_receipt = laya_lane.stage_finetune_eval_kernel(run_tag=RUN_TAG)
    staged["finetune-eval"] = script(eval_receipt)
    holdout = laya_lane.stage_holdout_eval_kernel(run_tag=RUN_TAG)
    staged["holdout-eval"] = script(holdout)
    return staged


def test_staged_kernel_bytes_match_the_golden_hashes(tmp_path, monkeypatch):
    staged = _staged_scripts(tmp_path, monkeypatch)
    assert set(staged) == set(GOLDEN_SHA256)
    actual = {label: hashlib.sha256(body).hexdigest()
              for label, body in staged.items()}
    assert actual == GOLDEN_SHA256
