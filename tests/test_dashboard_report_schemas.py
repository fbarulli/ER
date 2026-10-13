"""The dashboard must render what the producers actually write.

Three real defects are pinned here:

* ``report.json`` is written by two unrelated producers whose schemas share an
  EMPTY key set, so the renderer matched neither once the ablation lane shipped
  and printed the "Post-training error analysis" heading with nothing under it.
* the DVC payload snapshot is a byte-identical copy of a sibling report tree
  that lives inside the run directory, so every graph-track table and plot
  rendered twice.
* runs are ordered by mtime alone, so the default landing run was an in-flight
  ``__logs`` directory with no metrics at all.
"""

from __future__ import annotations

from core.portable_archive import ByteCount
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest


def _load():
    spec = importlib.util.spec_from_file_location(
        'er_training_reports_schemas',
        Path(__file__).parents[1] / 'dashboard/training_reports.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tr = _load()


def _ablation_report() -> dict:
    """A report using the real er-attribute-ablation-report-v1 key set."""
    return {
        "schema": "er-attribute-ablation-report-v1",
        "track": "text",
        "attribute": "volume",
        "intervention": "declaration only",
        "retrieval_intervention": "text ablation",
        "retrieval_scope": "sampled",
        "retrieval_catalog_count": 1200,
        "split": "dev",
        "composition": {"graph": 0.4, "text": 0.6},
        "changed_listings": 3,
        "decision_flip": True,
        "endpoint_input_changed": True,
        "embedding_dtype": "float32",
        "baseline_score": 0.81,
        "ablated_score": 0.11,
        "score_delta": -0.7,
        "embedding_cosine_delta": -0.02,
        "missing_axes": ["masking_profile"],
        "threshold": 0.5,
        "threshold_binding": "frozen",
        "threshold_source": "saved dev calibration; no refit",
        "threshold_provenance": {"size": "d" * 64, "calibration": "dev"},
        "ann_baseline_hits": 40,
        "ann_ablated_hits": 12,
        "checkpoint_role": "selected",
        "known_positive_recall_change": {"10": {"baseline": 0.8, "ablated": 0.3, "change": -0.5}},
        "rows": [
            {"sku_id1": "a", "sku_id2": "b", "baseline_ranks": [1, 2],
             "ablated_ranks": [3, 4], "decision_flip": True, "score_delta": -0.7},
        ],
    }


def test_ablation_schema_renders_no_longer_empty():
    """The regression: legacy-only renderer produced zero output for this."""
    legacy_keys = {"confusion", "attribute_errors", "random_easy", "score_overlap"}
    report = _ablation_report()
    assert not legacy_keys & set(report), "schemas genuinely do not overlap"

    html = tr.render_report(report)
    assert html.strip(), "ablation report rendered nothing"
    assert "<table" in html
    for field in ("Score delta", "Decision flip", "Threshold provenance",
                  "Per-row deltas", "Known-positive recall change"):
        assert field in html, f"{field} was dropped from the rendered report"


def test_legacy_schema_still_renders():
    html = tr.render_report({"confusion": {"balanced_review": {
        "tp": 1, "fp": 2, "fn": 3, "tn": 4, "accuracy": 0.5,
        "precision": 0.3, "recall": 0.25, "f1": 0.27}}})
    assert "Confusion matrices" in html
    assert "balanced_review" in html


def test_unknown_schema_is_visibly_unhandled_not_silently_empty():
    html = tr.render_report({"schema": "er-brand-new-report-v9", "mystery": 7})
    assert "not rendered yet" in html
    assert "mystery" in html


def test_manifest_schema_renders_the_honesty_contract():
    html = tr.render_report({
        "schema": "er-track-report-manifest-v1", "track": "gnn_only",
        "test_used_for_selection": False, "unlabeled_pairs_are_negatives": False,
        "metrics_scope": "model-only", "retrieval_ks": [1, 5, 10],
        "summary": [{"split": "dev", "pr_auc": 0.8}],
        "slices": [{"split": "dev", "slice": "unseen", "pr_auc": 0.4}],
        "performance": {"sections": {"encode": {"calls": 3, "total_seconds": 1.0}},
                        "peak_rss_mb": 512.0,
                        "missing_required_sections": ["refresh"]},
        "confidence_intervals": {"dev": {
            "method": "paired_bootstrap", "repeated_training_seeds": None,
            "repeated_training_seeds_note": "one checkpoint per split",
            "metrics": {"pr_auc": {"point": 0.8, "low": 0.7, "high": 0.9,
                                   "resamples_used": 1000}}}},
    })
    for field in ("Report manifest", "test_used_for_selection", "Generalization slices",
                  "Paired bootstrap", "Operational cost", "one checkpoint per split",
                  "refresh"):
        assert field in html, f"{field} missing from the rendered manifest"


def test_dvc_payload_copies_are_detected_but_the_pointer_file_is_not():
    assert tr.is_duplicate_copy("dvc/gnn_only__payload/gnn_only__reports/x.csv")
    assert tr.is_duplicate_copy("gnn_only__payload/x.csv")
    # The sibling .dvc pointer is metadata about the copy, not a copy of it.
    assert not tr.is_duplicate_copy("gnn_only__dvc/gnn_only__payload.dvc")
    assert not tr.is_duplicate_copy("gnn_only__reports/x.csv")


def test_default_run_prefers_one_that_has_metrics(tmp_path, monkeypatch):
    """The landing page used to be an in-flight logs-only run."""
    logs = tmp_path / "1004__logs"
    logs.mkdir()
    (logs / "suite_events.jsonl").write_text("{}\n")
    complete = tmp_path / "1002.zip"
    with zipfile.ZipFile(complete, "w") as archive:
        archive.writestr("suite_manifest.json", "{}")
        archive.writestr("suite_result.json", "{}")
        archive.writestr("text__reports/text__model_evaluation_summary.csv", "a\n1\n")
    import os
    os.utime(logs, (9e9, 9e9))       # newest
    os.utime(complete, (1e9, 1e9))   # oldest, but the only one with metrics

    available = {"logs": logs, "complete": complete}
    monkeypatch.setattr(tr, "runs", lambda: available)
    assert tr.default_run(available) == "complete"


def test_default_run_falls_back_to_newest_when_nothing_has_metrics(tmp_path):
    lone = tmp_path / "lonely"
    lone.mkdir()
    available = {"lonely": lone}
    assert tr.default_run(available) == "lonely"
    assert tr.default_run({}) is None


def test_profiler_artifacts_are_downloadable_and_visible():
    """operator_summary.txt used to be invisible: not listed, not served."""
    assert tr.is_profiler_artifact("text/text__profile/operator_summary.txt")
    assert tr.is_profiler_artifact("text/text__profile/profile_manifest.json")
    assert not tr.is_profiler_artifact("text/text__reports/scored_pairs.csv")
    # It is audit evidence, not a metric table, so it must not be tabulated.
    assert not tr.is_downloadable_audit("text/text__profile/operator_summary.txt")


def test_duplicate_exclusion_actually_halves_a_dvc_run(tmp_path, monkeypatch):
    """End-to-end: a run with payload copies renders each table exactly once."""
    report_csv = "model_evaluation_summary.csv,pr_auc\ngnn_only,0.8\n"
    original = "gnn_only__completion/gnn_only__reports/gnn_only__model_evaluation_summary.csv"
    copy = "gnn_only__dvc/gnn_only__payload/gnn_only__completion/gnn_only__reports/gnn_only__model_evaluation_summary.csv"
    run = tmp_path / "run"
    for relative, payload in ((original, report_csv), (copy, report_csv)):
        target = run / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload)
    archive = tmp_path / "run.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for relative, payload in ((original, report_csv), (copy, report_csv)):
            zf.writestr(relative, payload)

    for path in (run, archive):
        members = tr.entries(path)
        visible = [m for m in members if not tr.is_duplicate_copy(m)]
        assert len(members) == 2
        assert visible == [original], f"{path} kept a duplicate copy"
        # And the two files really are byte-identical, so nothing is lost.
        digests = {ByteCount(tr.read(path, m)).total for m in members}
        assert len(digests) == 1