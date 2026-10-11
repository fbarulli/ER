"""Pin the GNN cluster-quality plot artifact and its expected structure.

The public deliverable is ``training.cluster_quality_plot.ClusterQualityPlot``:
it turns a graph-lane ``pair_metrics`` row into the four clustering-quality bars
(over/under-merge reusing ``rand_matching``'s ``fp/(tp+fp)`` / ``fn/(tp+fn)``)
and renders a real PNG under the lane's artifact name. The gnn_only report lane
(``graph_tracks.report.complete``) delegates to it as a completed-run deliverable;
that wiring is a one-line call beside the lane's other plots, so the pin is on
the public plot it produces. The legacy ``report.json`` path (the CLI) is kept
and pinned so the refactor did not regress it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.cluster_quality_plot import PLOT_NAME, ClusterQualityPlot

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_cluster_quality_plot_from_pair_metrics_renders_png(tmp_path: Path) -> None:
    """The four bars carry the rand_matching rates and render to a real PNG."""
    # tp=8, fp=2 -> over-merge 2/10=0.2; fn=2 -> under-merge 2/10=0.2
    plot = ClusterQualityPlot.from_pair_metrics(
        {"split": "dev", "tp": 8, "fp": 2, "fn": 2, "precision": 0.8, "recall": 0.8,
         "both_classes": True},
        split="dev", track="gnn_only")
    assert plot.rates == {"pairwise_precision": 0.8, "pairwise_recall": 0.8,
                          "over_merge_rate": pytest.approx(0.2),   # fp/(tp+fp)
                          "under_merge_rate": pytest.approx(0.2)}  # fn/(tp+fn)
    from graph_tracks.artifacts import name
    out = plot.render(tmp_path / name("gnn_only", PLOT_NAME))
    assert out.name == "gnn_only__cluster_quality.png"  # the lane's deliverable name
    assert out.read_bytes()[:8] == _PNG_MAGIC
    assert out.stat().st_size > 1000  # a real figure, not an empty stub


def test_cluster_quality_plot_legacy_report_json_path_preserved(tmp_path: Path) -> None:
    """The pre-existing report.json (calibration_*) source still maps verbatim."""
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"metrics": {"fold_0": {
        "calibration_pairwise_precision": 1.0, "calibration_pairwise_recall": 0.98,
        "calibration_over_merge_rate": 0.0, "calibration_under_merge_rate": 0.02}}}))
    plot = ClusterQualityPlot.from_report_json(report)
    assert plot.rates == {"pairwise_precision": 1.0, "pairwise_recall": 0.98,
                          "over_merge_rate": 0.0, "under_merge_rate": 0.02}
