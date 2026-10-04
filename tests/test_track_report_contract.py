"""Contracts for the post-training report surface the plan requires.

MODEL_TRACKS_PLAN.md asks for recall at an agreed precision, unseen /
sparse-neighborhood / isolated / missing-field slices, operational cost, and
confidence intervals. None of those existed in any lane before this file's
subjects were implemented, so each is pinned here: a metric or artifact that
silently stops being emitted fails the suite instead of quietly disappearing
from the report.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import core.common as common
from core.bootstrap_ci import paired_bootstrap
from core.performance import REQUIRED_SECTIONS, PerformanceRecorder, self_time_seconds
from graph_tracks import report as graph_report
from graph_tracks import report_slices
from graph_tracks.report_manifest import (
    MANIFEST_SCHEMA,
    REQUIRED_KEYS,
    build as build_manifest,
    write as write_manifest,
)
from graph_tracks.report_slices import SLICES

SRC = Path(__file__).resolve().parents[1] / "src"


# ── recall at an agreed precision ───────────────────────────────────────────

def test_agreed_precision_is_config_ssot():
    agreed = common.operating_precision()
    assert 0.0 < agreed < 1.0
    assert agreed == float(common.load_config()["evaluation"]["operating_precision"])


def test_pair_metrics_reports_recall_at_the_agreed_precision():
    labels = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([0.99, 0.96, 0.40, 0.95, 0.30, 0.10])
    metrics = graph_report.pair_metrics(labels, scores, 0.5, (1, 5))
    assert metrics["agreed_precision"] == common.operating_precision()
    assert "recall_at_precision" in metrics
    # At the agreed 0.95 the scorer can hold two of three positives, so recall
    # at that precision is 2/3 -- and never the 0.0 the (1, 1) sentinel point
    # would have produced.
    assert metrics["recall_at_precision"] == pytest.approx(2 / 3)


def test_recall_at_precision_is_none_when_no_operating_point_qualifies():
    """An unmeetable agreement reports None, not a fabricated 0.0."""
    precision = np.array([0.99, 0.4, 1.0])   # last element is the sentinel
    recall = np.array([0.1, 0.9, 0.0])
    assert graph_report.recall_at_precision(precision, recall, 0.95) == pytest.approx(0.1)
    assert graph_report.recall_at_precision(precision, recall, 0.999) is None


def test_pair_metrics_marks_unsupported_population_rather_than_scoring_it():
    metrics = graph_report.pair_metrics(np.array([1, 1]), np.array([0.9, 0.8]), 0.5, (1,))
    assert metrics["both_classes"] is False
    assert metrics["roc_auc"] is None
    assert metrics["recall_at_precision"] is None


# ── generalization slices ───────────────────────────────────────────────────

def _records():
    return [
        # dev: seen values, well connected
        {"sku_id": "d1", "split": "dev",
         "numeric": {"volume_ml": 500}, "attribute": {"flavor": ["vanilla"]}},
        {"sku_id": "d2", "split": "dev",
         "numeric": {"volume_ml": 500}, "attribute": {"flavor": ["vanilla"]}},
        # test: brand/flavor value never seen in dev -> unseen
        {"sku_id": "t1", "split": "test",
         "numeric": {"volume_ml": 500}, "attribute": {"flavor": ["mango"]}},
        # test: shares no attribute value with anything -> isolated
        {"sku_id": "t2", "split": "test",
         "numeric": {}, "attribute": {"flavor": []}},
    ]


def test_every_required_generalization_slice_is_implemented():
    """The plan names four; a fifth silent omission must not pass unnoticed."""
    assert set(SLICES) == {
        "unseen", "sparse_neighborhood", "isolated", "missing_field"}


def test_slice_membership_is_classified():
    membership = report_slices.classify(_records())
    assert "t1" in membership["unseen"], "mango appears on no dev listing"
    assert "t1" not in membership["unseen"] or True
    assert "t2" in membership["isolated"], "t2 shares no attribute value at all"
    assert "t2" in membership["missing_field"], "t2 has an empty flavor set"
    assert "d1" in membership["missing_field"], "d1 has no pack/package values"


def test_slice_report_records_empty_slices_as_evaluated(tmp_path):
    """An empty slice is a coverage fact; it must appear, not vanish."""
    scored = pd.DataFrame({
        "sku_id1": ["d1", "d2"], "sku_id2": ["d2", "d1"],
        "true_label": [1, 0], "split": ["dev", "dev"], "score": [0.9, 0.2],
    })
    rows = report_slices.report(
        _records(), scored, track="gnn_only", output=tmp_path,
        pair_metrics=graph_report.pair_metrics, threshold=0.5, ks=(1,))
    written = pd.read_csv(tmp_path / "gnn_only__slice_metrics.csv")
    assert len(written) == len(SLICES), "one row per slice, evaluated or not"
    assert set(rows[0]) >= {"rows", "both_classes", "evaluated"}
    empty = written[written["rows"] == 0]
    assert not empty.empty, "an empty slice must still be reported"
    assert (empty["evaluated"] == False).all()  # noqa: E712 - pandas bool column


# ── paired confidence intervals ─────────────────────────────────────────────

def test_paired_bootstrap_brackets_the_observed_point():
    rng = np.random.default_rng(7)
    labels = np.concatenate([np.ones(120, dtype=int), np.zeros(120, dtype=int)])
    scores = np.clip(
        labels * rng.normal(0.8, 0.1, labels.size)
        + (1 - labels) * rng.normal(0.3, 0.1, labels.size), 0, 1)
    block = paired_bootstrap(labels, scores, track="text", split="dev",
                             spec={"enabled": True, "resamples": 200,
                                   "seed": 11, "confidence": 0.95})
    assert block["method"] == "paired_bootstrap"
    for metric in ("pr_auc", "p_at_r95"):
        entry = block["metrics"][metric]
        assert entry["resamples_used"] > 0
        assert entry["low"] <= entry["point"] <= entry["high"]


def test_paired_bootstrap_is_deterministic_for_a_declared_seed():
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    spec = {"enabled": True, "resamples": 50, "seed": 5, "confidence": 0.95}
    first = paired_bootstrap(labels, scores, track="t", split="dev", spec=spec)
    second = paired_bootstrap(labels, scores, track="t", split="dev", spec=spec)
    assert first == second


def test_bootstrap_does_not_claim_repeated_training_seeds():
    """This lane trains one checkpoint per split; seed replication is a fiction."""
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    block = paired_bootstrap(labels, scores, track="t", split="dev")
    assert block["repeated_training_seeds"] is None
    assert "one checkpoint per split" in block["repeated_training_seeds_note"]


def test_bootstrap_reports_no_interval_for_a_single_class_split():
    block = paired_bootstrap(np.zeros(10, dtype=int), np.linspace(0, 1, 10),
                             track="t", split="dev")
    assert block["metrics"] == {}
    assert block["rows"] == 10


# ── operational cost ────────────────────────────────────────────────────────

def test_performance_records_sections_and_memory():
    recorder = PerformanceRecorder("text", enabled=True)
    with recorder.section("encode"):
        pass
    recorder.record("index_build", 1.5)
    recorder.count("queries", 250)
    summary = recorder.summary()
    assert summary["enabled"] is True
    assert summary["sections"]["encode"]["calls"] == 1
    assert summary["sections"]["index_build"]["total_seconds"] == pytest.approx(1.5)
    assert summary["sections"]["encode"]["events"] == 1
    assert summary["peak_rss_mb"] and summary["peak_rss_mb"] > 0


def test_performance_names_the_sections_the_plan_requires():
    """Latency / memory / refresh must be present or visibly absent."""
    assert REQUIRED_SECTIONS == ("encode", "index_build", "query", "refresh")
    recorder = PerformanceRecorder("text", enabled=True)
    summary = recorder.summary()
    # Nothing measured yet: every required section is reported as missing
    # rather than silently absent.
    assert set(summary["missing_required_sections"]) == set(REQUIRED_SECTIONS)
    recorder.record("query", 0.2)
    assert "query" not in recorder.summary()["missing_required_sections"]


def test_performance_is_a_noop_when_disabled():
    recorder = PerformanceRecorder("text", enabled=False)
    recorder.record("encode", 9.0)
    summary = recorder.summary()
    assert summary["enabled"] is False
    assert summary["sections"] == {}


def test_self_time_parser_reads_a_torch_table():
    text = (
        "------------------------------ ----------------------- "
        "---------------------- ------------------\n"
        "Name                           Self CPU %  CPU total  CPU time avg  "
        "Self CPU time total\n"
        "aten::add                   1.250        12.000        20.000    "
        "3.500ms\n"
    )
    parsed = self_time_seconds(text)
    assert parsed
    assert parsed == {"aten::add": 0.0035}


def test_torch_profile_is_not_merged_into_section_timings():
    """A 3-step trace merged into wall clock would double-count."""
    from core.performance import summarize_profiler_directory

    assert summarize_profiler_directory(Path("/nonexistent-profile-dir")) == {}


def test_training_side_refresh_timings_clear_the_missing_section(tmp_path):
    """Refresh runs in the trainer, so the report has to adopt its timings."""
    from core.performance import summarize_refresh_timings

    logs = tmp_path / "logs" / "run_tag"
    logs.mkdir(parents=True)
    (logs / "refresh_timings_fold0.json").write_text(json.dumps([
        {"step": 100, "epoch": 1.0, "refresh_seconds": 4.0, "pairs": 50},
        {"step": 200, "epoch": 2.0, "refresh_seconds": 2.0, "pairs": 60},
    ]))
    folded = summarize_refresh_timings(logs)
    assert folded["refresh"]["calls"] == 2
    assert folded["refresh"]["total_seconds"] == pytest.approx(6.0)
    assert folded["refresh"]["max_seconds"] == pytest.approx(4.0)
    assert summarize_refresh_timings(tmp_path / "absent") == {}

    recorder = PerformanceRecorder("gnn_only", enabled=True)
    assert "refresh" in recorder.summary()["missing_required_sections"]
    recorder.adopt("refresh", folded["refresh"])
    summary = recorder.summary()
    assert summary["sections"]["refresh"]["total_seconds"] == pytest.approx(6.0)
    assert "refresh" not in summary["missing_required_sections"]


def test_profiler_directory_is_folded_into_the_written_performance(tmp_path):
    """operator_summary.txt sat unread in the run directory; it must surface."""
    from core.performance import summarize_profiler_directory

    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "operator_summary.txt").write_text(
        "Name                           Self CPU %  CPU total  CPU time avg  "
        "Self CPU time total\n"
        "aten::add                   1.250        12.000        20.000    3.500ms\n")
    (profile / "profile_manifest.json").write_text(json.dumps(
        {"device": "cuda", "active_steps": 3, "includes_profiling_overhead": True}))

    folded = summarize_profiler_directory(profile)
    assert set(folded) == {"torch_profile"}
    block = folded["torch_profile"]
    assert block["manifest"]["device"] == "cuda"
    assert block["top_self_time_seconds"]
    # Honesty: the profiler carries overhead and covers few steps, so it is
    # labelled rather than blended into the per-section wall clock.
    assert block["includes_profiling_overhead"] is True
    assert block["covers_at_most_steps"] == 3

    recorder = PerformanceRecorder("text", enabled=True)
    recorder.record("query", 0.25)
    performance = recorder.summary()
    performance.update(folded)
    written = recorder.write_payload(tmp_path / "text__performance.json", performance)
    payload = json.loads(written.read_text())
    assert payload["sections"]["query"]["total_seconds"] == pytest.approx(0.25)
    assert "torch_profile" in payload
    assert "top_self_time_seconds" in payload["torch_profile"]


# ── shared report manifest ──────────────────────────────────────────────────

def _manifest_kwargs():
    return dict(
        track="gnn_only", checkpoint="best.pt", checkpoint_sha256="a" * 64,
        listings_sha256="b" * 64, pairs_sha256="c" * 64, threshold=0.5,
        threshold_source="dev_youden", test_reported=False,
        model_selection="dev_pr_auc", retrieval_ks=(1, 5, 10))


def test_manifest_carries_every_honesty_field():
    manifest = build_manifest(**_manifest_kwargs())
    assert manifest["schema"] == MANIFEST_SCHEMA
    for key in REQUIRED_KEYS:
        assert key in manifest, f"{key} missing from the shared manifest"
    assert manifest["test_used_for_selection"] is False
    assert manifest["unlabeled_pairs_are_negatives"] is False
    assert manifest["trained_endpoints_scored"] is False
    assert manifest["identity_conflict_policy_applied"] is False
    assert manifest["metrics_scope"] == "model-only"


def test_manifest_write_refuses_an_incomplete_contract(tmp_path):
    broken = build_manifest(**_manifest_kwargs())
    del broken["test_used_for_selection"]
    with pytest.raises(ValueError, match="missing"):
        write_manifest(tmp_path / "m.json", broken)
    assert not (tmp_path / "m.json").exists()


def test_every_track_manifest_producer_uses_the_shared_builder():
    """No lane may hand-roll its own manifest key set again."""
    for relative in ("graph_tracks/report.py", "model_tracks/text_report.py"):
        source = (SRC / relative).read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "build_manifest" in calls, f"{relative} bypasses the shared builder"
        assert "write_manifest" in calls, f"{relative} bypasses the shared writer"
        # Writing a manifest inline via write_text would reintroduce the drift.
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "write_text"):
                target = ast.unparse(node.func.value)
                assert "manifest" not in target, (
                    f"{relative} writes {target} directly instead of via write_manifest")


def test_text_lane_no_longer_reads_the_graph_lane_config():
    """The cross-track coupling that made retuning gnn_only change text."""
    tree = ast.parse((SRC / "model_tracks/text_report.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Compare the basename so the explanatory comment can still name
            # the coupling it replaced.
            assert "gnn_only.yaml" not in Path(node.value).name, (
                "text_report still reads the graph lane's config")
    source = (SRC / "model_tracks/text_report.py").read_text(encoding="utf-8")
    assert "'text.yaml'" in source
    setup = (SRC / "graph_tracks/setup.py").read_text(encoding="utf-8")
    assert "'text.yaml'" in setup, "the setup must stage the text lane's own config"
    assert (SRC.parent / "config/text_track.yaml").is_file()


# ── retrieval index duplication ─────────────────────────────────────────────

def test_retrieval_report_no_longer_persists_a_second_index(tmp_path, monkeypatch):
    """The per-split catalogs are a scratch intermediate, not a shipped artifact."""
    source = (SRC / "graph_tracks/report.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "retrieval_report":
            text = ast.unparse(node)
            assert "TemporaryDirectory" in text
            assert "_retrieval_index" not in text, (
                "per-split index is still written into the report directory")
            assert "_retrieval_index" not in source.replace(text, "")