"""Public interpretation of graph-quality comparisons."""
import pytest

from graph_tracks.interpretation import (
    Change,
    GraphComparisonPolicy,
    GraphQualityInterpreter,
    GraphQualityMetrics,
)


class TestGraphQualityInterpreter:
    @pytest.mark.parametrize(
        ("metric", "baseline", "candidate", "expected", "note"),
        [
            ("recall_at_precision", 0.5, 0.7, Change.IMPROVED, "better retrieval"),
            ("known_positive_recall", 0.8, 0.7, Change.REGRESSED, "Fewer known"),
            ("bcubed_precision", 0.9, 0.8, Change.REGRESSED, "Cluster mixing"),
            ("bcubed_recall", 0.8, 0.7, Change.REGRESSED, "Cluster completeness"),
            ("overmerge_rate", 0.1, 0.2, Change.REGRESSED, "Cluster mixing"),
            ("brier", 0.2, 0.1, Change.IMPROVED, "Descriptive deltas"),
            ("brier", 0.1, 0.2, Change.REGRESSED, "Probability error"),
            ("pr_auc", 0.5, 0.505, Change.STABLE, "Descriptive deltas"),
            ("pr_auc", 0.5, 0.8, Change.IMPROVED, "operating point worsened"),
            ("pr_auc", None, 0.8, Change.UNAVAILABLE, "Unreported metrics"),
            ("pr_auc", 0.8, None, Change.UNAVAILABLE, "Unreported metrics"),
            ("pr_auc", 0.0, 0.0, Change.STABLE, "Descriptive deltas"),
        ],
    )
    def test_compare(
        self, metric: str, baseline: float | None, candidate: float | None,
        expected: Change, note: str,
    ) -> None:
        before = GraphQualityMetrics.model_validate({metric: baseline})
        after = GraphQualityMetrics.model_validate({metric: candidate})
        if metric == "pr_auc":
            before.recall_at_precision = 0.8
            after.recall_at_precision = 0.6
        report = GraphQualityInterpreter.compare(
            before, after, GraphComparisonPolicy(minimum_change=0.01))
        assert report.metrics[metric].change == expected
        assert any(note in item for item in report.notes)
        assert report.metrics[metric].delta == (
            None if baseline is None or candidate is None else candidate - baseline)
