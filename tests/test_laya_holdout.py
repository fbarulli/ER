"""scripts/laya_holdout.py — component-disjoint, difficulty-tagged holdout."""
from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.laya_holdout import build_holdout
from cli import laya_lane
from core import holdout_eval as core_holdout


def _write(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _fixtures(root: Path) -> dict[str, Path]:
    catalog = _write(root / "catalog.csv", ["sku_id", "gtin", "attribute"], [
        ["s1", "11111111111111", "Volume: 100"],
        ["s2", "22222222222222", "Volume: 100"],
        ["s3", "33333333333333", "Volume: 200"],
    ])
    listing = _write(root / "listing.csv",
                     ["sku_id1", "sku_id2", "label", "split"], [
                         ["s1", "s2", "1", "test"],   # links s1+s2 -> one component
                         ["s1", "s3", "0", "test"],
                     ])
    p0 = _write(root / "p0.csv",
                ["gtin1", "gtin2", "true_label", "endpoint_in_train",
                 "component_id"], [
                    ["11111111111111", "99999999999999", "0", "False", "c-p0"],
                    ["11111111111111", "88888888888888", "1", "True", "c-p0b"],
                ])
    gate = _write(root / "gate.csv",
                  ["gtin1", "gtin2", "gate_decision", "gate_reason", "similarity"], [
                      ["11111111111111", "22222222222222", "hard_no", "Pack blocker", "0.1"],
                      ["11111111111111", "33333333333333", "proceed", "Known critical attributes compatible", "0.8"],
                      ["11111111111111", "44444444444444", "fallback", "Missing flavor evidence", "0.5"],
                  ])
    labeled = _write(root / "labeled.csv", ["gtin1", "gtin2", "true_label"], [])
    return {"listing_path": listing, "catalog_path": catalog, "p0_path": p0,
            "gate_path": gate, "labeled_path": labeled}


def test_holdout_is_component_disjoint_and_difficulty_tagged(tmp_path):
    rows, receipt = build_holdout(**_fixtures(tmp_path))
    by_source = receipt["by_source"]
    assert by_source["listing_pairs"] == 2
    assert by_source["final_validation"] == 2
    # hard_no is the easy pipeline-verified mass -> never in the truth holdout
    assert by_source["gate_results"] == 2

    strata = receipt["by_stratum"]
    assert strata["gate_proceed"] == 1 and strata["gate_fallback"] == 1
    assert strata["p0_disjoint"] == 1 and strata["p0_overlap"] == 1

    # the positive listing pair links its two endpoints into ONE component
    listing_rows = [r for r in rows if r["source"] == "listing_pairs"]
    positive = next(r for r in listing_rows if r["label"] == "1")
    a = next(r for r in listing_rows if r["gtin1"] == positive["gtin1"])
    assert positive["component"]
    assert a["component"] == positive["component"]

    # gate rows are label-less (agreement/difficulty only), never scored as truth
    gate_rows = [r for r in rows if r["source"] == "gate_results"]
    assert all(r["label"] == "" for r in gate_rows)
    assert all(r["label_source"] == "gate_verdict" for r in gate_rows)


def test_holdout_label_census_counts_only_real_labels(tmp_path):
    rows, receipt = build_holdout(**_fixtures(tmp_path))
    labelled = [r for r in rows if r["label"] in ("0", "1")]
    assert receipt["labelled_rows"] == len(labelled)
    assert receipt["positives"] + receipt["negatives"] == receipt["labelled_rows"]


# ── holdout-eval kernel parity: ONE metric implementation ──────────────────

def _kernel_holdout_eval_namespace() -> dict:
    """Exec the exact module source the holdout kernel embeds at staging."""
    source = laya_lane._holdout_eval_module_source()
    namespace: dict = {}
    exec(compile(source, "<core.holdout_eval>", "exec"), namespace)
    return namespace


def test_holdout_kernel_embeds_the_single_metric_module():
    source = laya_lane._holdout_eval_module_source()
    assert "def binary_metrics" in source
    assert "def cluster_bootstrap_ci" in source
    # no drift: the embedded source IS core.holdout_eval's own file
    from core import holdout_eval

    assert source == Path(holdout_eval.__file__).read_text(encoding="utf-8")


@pytest.mark.parametrize("y_true,y_score", [
    ([1, 1, 1], [0.9, 0.8, 0.7]),          # all-positive
    ([0, 0, 0], [0.1, 0.2, 0.3]),          # all-negative
    ([], []),                              # empty
    ([1, 0, 1, 0], [0.9, 0.4, 0.8, 0.3]),  # mixed
])
def test_kernel_binary_metrics_match_core_holdout_eval(y_true, y_score):
    he = _kernel_holdout_eval_namespace()
    want = core_holdout.binary_metrics(np.array(y_true), np.array(y_score),
                                       threshold=0.5)
    got = he["binary_metrics"](np.array(y_true), np.array(y_score),
                               threshold=0.5)
    assert got == want


def test_kernel_single_class_pr_auc_is_missing_not_fabricated():
    """A single observed class must report PR-AUC as missing (never 1.0)."""
    he = _kernel_holdout_eval_namespace()
    all_pos = he["binary_metrics"](np.array([1, 1, 1]),
                                   np.array([0.9, 0.8, 0.7]), threshold=0.5)
    assert all_pos["pr_auc"] is None, "single-class PR-AUC must not be 1.0"
    all_neg = he["binary_metrics"](np.array([0, 0, 0]),
                                   np.array([0.1, 0.2, 0.3]), threshold=0.5)
    assert all_neg["pr_auc"] is None


def test_kernel_empty_slice_is_missing_not_zero():
    he = _kernel_holdout_eval_namespace()
    m = he["binary_metrics"](np.array([]), np.array([]), threshold=0.5)
    assert m["precision"] is None and m["recall"] is None and m["f1"] is None
    assert m["accuracy"] is None and m["pr_auc"] is None


def test_kernel_bootstrap_ci_matches_core_and_drops_nonfinite():
    """The kernel's component bootstrap keeps the np.isfinite filter."""
    he = _kernel_holdout_eval_namespace()
    kernel_ci = he["cluster_bootstrap_ci"]
    comps = np.array(["a", "a", "b", "b", "c", "c"], dtype=object)
    y_true = np.array([1, 1, 0, 0, 1, 0], dtype=int)
    y_score = np.array([0.9, 0.8, 0.4, 0.3, 0.7, 0.2], dtype=float)

    def stat(t, s):
        if len(np.unique(t)) < 2:      # a degenerate resample: PR-AUC is undefined
            return float("nan")
        return float(np.mean(t))

    got = kernel_ci(comps, y_true, y_score, statistic=stat,
                    n_boot=400, seed=1729)
    want = core_holdout.cluster_bootstrap_ci(
        comps, y_true, y_score, statistic=stat, n_boot=400, seed=1729)
    assert got == want
    assert got["n_boot"] < 400, "non-finite draws must be dropped"
    assert np.isfinite(got["lo"]) and np.isfinite(got["hi"])
    assert got["alpha"] == 0.05


# ── DEFECT 2: size_of is ONE injected definition ───────────────────────────

def test_size_of_is_a_single_injected_definition():
    tree = ast.parse(Path(laya_lane.__file__).read_text())
    assert not any(isinstance(node, ast.FunctionDef) and node.name == "size_of"
                   for node in tree.body)
    for template in (laya_lane.FINETUNE_KERNEL_SCRIPT,
                     laya_lane.FINETUNE_EVAL_KERNEL_SCRIPT,
                     laya_lane.HOLDOUT_EVAL_KERNEL_SCRIPT):
        assert "def size_of(" not in template
        assert "@SIZE_OF@" in template


def test_size_of_source_matches_core_manifest_size(tmp_path):
    from core.manifest import file_size

    namespace = {"Path": Path}
    exec(compile(laya_lane._file_bytes_of_source(), "<size_of>", "exec"),
         namespace)
    target = tmp_path / "payload.bin"
    target.write_bytes(b"hello holdout\n" * 4096)
    assert namespace["size_of"](target) == file_size(target)


# ── DEFECT 3: hermetic staging + exec of the staged kernel helpers ──────────

class _FakeComposer:
    def compose_side(self, attribute: str) -> str:
        return attribute

    def compose_state(self, one: str, two: str) -> dict:
        return {"one": one, "two": two}


def _holdout_payload_fixtures(tmp_path: Path) -> dict[str, Path]:
    schema = tmp_path / "config/laya.question.json"
    schema.parent.mkdir(parents=True, exist_ok=True)
    schema.write_text(json.dumps({"questions": {
        "identity_claim": {"type": "noul", "instructions": "same item?"},
    }}), encoding="utf-8")
    catalog = tmp_path / "data/track_setup/eligible_catalog.csv"
    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_text(
        "gtin,attribute\n"
        "11111111111111,Volume: 100\n"
        "22222222222222,Volume: 100\n", encoding="utf-8")
    holdout = tmp_path / "data/laya/holdout.csv"
    holdout.parent.mkdir(parents=True, exist_ok=True)
    holdout.write_text(
        "gtin1,gtin2,label,stratum,component\n"
        "11111111111111,22222222222222,1,gate_proceed,comp-1\n"
        "22222222222222,11111111111111,0,p0_disjoint,comp-2\n",
        encoding="utf-8")
    return {"schema": schema, "catalog": catalog, "holdout": holdout}


def _hermetic_holdout_staging(tmp_path, monkeypatch):
    from core.common import training_cfg as _tcfg
    from core.schemas import LayaSpec

    # no hosted slug here: the lane reads them from the registry
    # (config/hosted_datasets.yaml)
    spec = LayaSpec()
    monkeypatch.setattr(laya_lane, "_spec", lambda: spec)
    monkeypatch.setattr(laya_lane, "TRAIN_ROOT", tmp_path)
    base = _tcfg()
    forced = base.model_copy(update={
        "kaggle": base.kaggle.model_copy(update={"branch": "main"})})
    monkeypatch.setattr(laya_lane, "training_cfg", lambda: forced)
    monkeypatch.setattr(laya_lane, "_git_revision", lambda: "abc123def")
    monkeypatch.setattr(laya_lane, "_pairs_composer", lambda: _FakeComposer())

    import subprocess

    def fake_run(args, **kwargs):
        if "rev-parse" in args:
            return subprocess.CompletedProcess(
                args, 0, stdout="abc123def", stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return spec


def test_stage_holdout_dataset_payload_hermetic(tmp_path, monkeypatch):
    fixtures = _holdout_payload_fixtures(tmp_path)
    monkeypatch.setattr(laya_lane, "TRAIN_ROOT", tmp_path)
    monkeypatch.setattr(laya_lane, "_pairs_composer", lambda: _FakeComposer())
    receipt = laya_lane.stage_holdout_dataset_payload(
        dataset_slug="fbarulli/er-laya-holdout",
        holdout_csv=fixtures["holdout"],
        catalog_path=fixtures["catalog"],
        question_path=fixtures["schema"])
    payload = Path(receipt["payload"])
    assert (payload / "dataset-metadata.json").is_file()
    lines = [json.loads(line) for line in
             (payload / laya_lane.HOLDOUT_JSONL).read_text().splitlines()
             if line.strip()]
    assert len(lines) == 2
    assert {line["expected"]["identity_claim"] for line in lines} == \
        {"true", "false"}
    assert receipt["rows"] == 2 and receipt["skipped"] == 0
    assert receipt["files"][laya_lane.HOLDOUT_JSONL] == \
        laya_lane.file_size(payload / laya_lane.HOLDOUT_JSONL)


def _exec_staged_kernel_helpers(script: str) -> dict:
    """Compile the staged kernel's metric helpers (block + its deps), not main()."""
    tree = ast.parse(script)
    wanted = {"_HOLDOUT_EVAL_SOURCE", "N_BOOT", "SEED"}
    nodes = [node for node in tree.body
             if (isinstance(node, ast.FunctionDef)
                 and node.name in ("_load_holdout_eval", "block"))
             or (isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id in wanted
                         for t in node.targets))]
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    namespace: dict = {}
    exec(compile(module, "<holdout-kernel-helpers>", "exec"), namespace)
    return namespace


def test_stage_holdout_eval_kernel_hermetic_and_execs_helpers(
        tmp_path, monkeypatch):
    fixtures = _holdout_payload_fixtures(tmp_path)
    _hermetic_holdout_staging(tmp_path, monkeypatch)
    receipt = laya_lane.stage_holdout_eval_kernel(run_tag="laya_test")
    staged = Path(receipt["staged"])
    script = (staged / laya_lane.HOLDOUT_EVAL_CODE_FILE).read_text(
        encoding="utf-8")
    assert "@SIZE_OF@" not in script
    assert "@HOLDOUT_EVAL_MODULE@" not in script
    assert "@RUNTIME_PREFLIGHT@" not in script
    assert script.count("def size_of(") == 1
    assert receipt["threshold"] == 0.5

    namespace = _exec_staged_kernel_helpers(script)
    he = namespace["_load_holdout_eval"]()
    labels = np.array([1, 0, 1, 0], dtype=int)
    scores = np.array([0.9, 0.4, 0.8, 0.3], dtype=float)
    comps = np.array(["a", "b", "a", "c"], dtype=object)
    block = namespace["block"]
    blk = block(he, labels, scores, comps, 0.5)
    want = core_holdout.binary_metrics(labels, scores, threshold=0.5)
    assert blk["metrics"] == want
    assert blk["pr_auc"] == want["pr_auc"]
    assert set(blk["cis"]) == {"precision", "recall", "f1", "pr_auc"}
    assert blk["cis"]["f1"]["point"] == want["f1"]
