"""core.results.Results owns the results BETWEEN training and post-training analysis.

Each test pins ONE public behavior: the declared result roles resolve through the
SAME SSOT accessors the tree already uses, the post-training request is produced
from a training output and landed where the GPU lane reads it, the landed
report/baseline lookups read the sealed documents, the threshold ladder delegates
to its one SSOT, and ``identity`` is a structural census.
"""
import json

import pytest

from core import common
from core.results import (
    CONFIG_NAME,
    Results,
    ResultRoleSpec,
    ResultsSpec,
    ThresholdSpec,
    results_spec,
)

TAG = "20261008T000000000000Z"


def _run(root):
    """The declared Results with its root pinned at a test fixture path."""
    return Results.from_config(TAG).model_copy(update={"root": root})


def _write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


# ── declaration: the spec names the single address each role resolves through ──

def test_the_declaration_covers_the_result_surface():
    spec = results_spec()
    assert CONFIG_NAME == "results.yaml"
    assert set(spec.roles) == {
        "request", "prepared_inputs", "vectors", "report",
        "baseline_threshold", "receipt", "holdout_report",
    }
    assert set(spec.sets["saved_ablation"]) <= set(spec.roles)
    assert spec.tracks == ("text", "gnn_only")


def test_roles_resolve_through_the_same_ssot_accessors():
    run = _run(common.RESULTS / "model_tracks" / TAG)
    assert run.vectors("text") == common.artifact(
        "track_ablation_vectors", {"run_tag": TAG, "track": "text"})
    assert run.report("gnn_only") == common.artifact(
        "track_ablation_report", {"run_tag": TAG, "track": "gnn_only"})
    assert run.request("text") == (
        run.root / "text" / "ablation"
        / common.training_cfg().bundle.ablation_request_file)
    assert run.prepared_inputs("text").name == "prepared_inputs.npz"
    assert run.baseline_threshold("text").name == "baseline_threshold.json"
    assert run.receipt() == run.root / "post_training_ablation.json"
    assert run.holdout_report() == (
        common.RESULTS / "laya_lane" / "kaggle" / "holdout-eval" / "holdout_report.json")


def test_paths_enumerates_every_addressed_role():
    run = _run(common.RESULTS / "model_tracks" / TAG)
    keys = set(run.paths())
    assert "text.request" in keys and "gnn_only.report" in keys
    assert {"receipt", "holdout_report"} <= keys
    assert set(run.saved("text")) == set(results_spec().sets["saved_ablation"])


# ── producing the post-training request from training outputs ─────────────────

def test_produce_request_binds_the_selected_checkpoint_onto_the_template(tmp_path):
    run = _run(tmp_path / "run")
    checkpoint = run.root / "text" / "_checkpoints" / "checkpoint-7"
    template = {"track": "text", "checkpoint": "baseline-ckpt",
                "sources": {"baseline-ckpt": "old"}}
    document = run.produce_request(
        "text", template=template, checkpoint=checkpoint,
        checkpoint_identity="structural-id")
    assert document["checkpoint"] == "@suite/text/_checkpoints/checkpoint-7"
    assert document["checkpoint_role"] == "selected"
    assert document["sources"] == {"@suite/text/_checkpoints/checkpoint-7": "structural-id"}
    # the frozen template is never mutated in place.
    assert template["checkpoint"] == "baseline-ckpt"


def test_write_request_lands_where_the_gpu_lane_reads_it(tmp_path):
    run = _run(tmp_path / "run")
    path = run.write_request("text", {"track": "text", "checkpoint": "@suite/x"})
    assert path == run.request("text")
    assert run.load_request("text") == {"track": "text", "checkpoint": "@suite/x"}


# ── consuming landed results: report / baseline / threshold ───────────────────

def test_landed_report_baseline_and_threshold_are_read_back(tmp_path):
    run = _run(tmp_path / "run")
    _write(run.report("gnn_only"), {"threshold": 0.71, "rows": [{"a": 1}]})
    _write(run.baseline_threshold("gnn_only"),
           {"track": "gnn_only", "threshold": 0.71})
    assert run.load_report("gnn_only")["rows"] == [{"a": 1}]
    assert run.baseline("gnn_only")["track"] == "gnn_only"
    assert run.frozen_threshold("gnn_only") == 0.71


def test_a_baseline_without_a_finite_threshold_fails_loud(tmp_path):
    run = _run(tmp_path / "run")
    _write(run.baseline_threshold("text"), {"track": "text"})
    with pytest.raises(ValueError, match="carries no threshold"):
        run.frozen_threshold("text")


def test_landed_is_a_presence_census_never_a_gate(tmp_path):
    run = _run(tmp_path / "run")
    assert run.landed("text") == set()
    _write(run.vectors("text"), {"x": 1})
    assert run.landed("text") == {"vectors"}


# ── thresholds and identity ───────────────────────────────────────────────────

def test_thresholds_delegates_to_the_one_ssot_accessor():
    assert _run(common.RESULTS / "model_tracks" / TAG).thresholds() == common.report_thresholds()


def test_identity_is_a_structural_census(tmp_path):
    run = _run(tmp_path / "run")
    member = run.vectors("text")
    _write(member, {"x": 1})
    before = run.identity()
    assert set(before) == {"name", "run_tag", "root", "members", "counts"}
    assert before["run_tag"] == TAG
    entry = next(m for m in before["members"] if m["name"] == "text.vectors")
    assert entry["present"] is True and entry["size"] == member.stat().st_size
    assert before["counts"]["results"] == len(before["members"])
    # no content-derived id anywhere: no 64-hex-character value is emitted.
    assert not any(
        isinstance(value, str) and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
        for member_row in before["members"] for value in member_row.values()
    )
    # removing a byte changes the census; an absent member is absence, not a raise.
    member.unlink()
    after = run.identity()
    entry = next(m for m in after["members"] if m["name"] == "text.vectors")
    assert entry == {"name": "text.vectors", "present": False, "size": 0}


# ── fail-loud boundaries ──────────────────────────────────────────────────────

def test_unknown_role_and_track_and_scope_misuse_fail_loud(tmp_path):
    run = _run(tmp_path / "run")
    with pytest.raises(KeyError, match="unknown results role"):
        run.path("ghost", "text")
    with pytest.raises(ValueError, match="unknown results track"):
        run.request("cascade")
    with pytest.raises(ValueError, match="track-scoped"):
        run.path("report")
    with pytest.raises(ValueError, match="takes no track"):
        run.path("receipt", "text")


def test_a_declaration_with_a_dangling_reference_is_refused():
    with pytest.raises(ValueError, match="undeclared key"):
        ResultsSpec(
            name="bad", run_layout="suite_outputs", ablation_dir="ablation",
            tracks=("text",),
            threshold_ladder=ThresholdSpec(key="report_thresholds"),
            holdout_dir="laya_lane",
            roles={"request": ResultRoleSpec(via="names", key="absent")},
            names={"present": "x.json"}, sets={"s": ("request",)},
        )
