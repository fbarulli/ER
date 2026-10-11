"""Component-clustered holdout evaluation: honest error bars + gate strata."""
from __future__ import annotations

import numpy as np

from core.holdout_eval import (
    CI_STATISTICS,
    binary_metrics,
    cluster_bootstrap_ci,
    stratified_report,
)


def test_binary_metrics_single_class_reports_missing_pr_auc():
    y_true = np.array([1, 1, 1])
    y_score = np.array([0.9, 0.8, 0.7])
    m = binary_metrics(y_true, y_score, threshold=0.5)
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0
    assert m["pr_auc"] is None, "a single observed class has no estimable PR-AUC"
    assert m["positives"] == 3 and m["negatives"] == 0


def test_binary_metrics_empty_is_missing_not_perfect():
    m = binary_metrics(np.array([]), np.array([]), threshold=0.5)
    assert m["n"] == 0
    assert m["accuracy"] is None and m["precision"] is None
    assert m["tp"] == m["fp"] == m["tn"] == m["fn"] == 0


def test_binary_metrics_scores_the_two_class_confusion():
    # two positives below threshold (fn) and two negatives above (fp)
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.4, 0.3, 0.9, 0.8])
    m = binary_metrics(y_true, y_score, threshold=0.5)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (0, 2, 0, 2)
    assert m["precision"] == 0.0 and m["recall"] == 0.0


def test_binary_metrics_derives_sklearn_zero_division_values_bit_for_bit():
    """P/R/F1 come from the counts this function already reports.

    The derivation replaced sklearn's ``precision_recall_fscore_support`` in the
    bootstrap's hot path, so it must equal that call exactly — including its
    confusion-count f-score spelling, which differs from ``2PR/(P+R)`` in the
    last bit — on the degenerate slices a resample actually produces.
    """
    from sklearn.metrics import precision_recall_fscore_support

    cases = [
        ([1, 1, 0, 0], [0.4, 0.3, 0.9, 0.8], 0.5),   # every prediction wrong
        ([1, 1, 1, 0], [0.9, 0.6, 0.55, 0.1], 0.5),  # mixed
        ([1, 1, 1], [0.2, 0.3, 0.4], 0.5),           # nothing predicted positive
        ([0, 0, 0], [0.9, 0.8, 0.7], 0.5),           # everything predicted positive
        ([1, 0, 1, 0, 1], [0.2, 0.1, 0.9, 0.4, 0.6], 0.55),
        ([1], [0.9], 0.0),                           # one row, all-positive threshold
    ]
    for labels, scores, threshold in cases:
        y_true, y_score = np.array(labels), np.array(scores)
        got = binary_metrics(y_true, y_score, threshold=threshold)
        pred = (y_score >= threshold).astype(int)
        want = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0)
        assert (got["precision"], got["recall"], got["f1"]) == tuple(
            float(value) for value in want[:3]), (labels, scores, threshold)
        assert got["tp"] + got["fn"] == got["positives"]
        assert got["tn"] + got["fp"] == got["negatives"]


def test_cluster_bootstrap_widens_when_rows_are_clustered():
    # Outcome constant WITHIN a component: 6 all-1 components, 6 all-0, 4 rows
    # each. The component bootstrap must show a wider interval than an iid row
    # bootstrap, because only 12 independent products exist, not 48 rows.
    comps, labels = [], []
    for i in range(12):
        comps += [f"c{i}"] * 4
        labels += [1 if i < 6 else 0] * 4
    comps = np.array(comps, dtype=object)
    labels = np.array(labels, dtype=int)
    scores = labels.astype(float)
    mean = lambda t, s: float(t.mean())

    clustered = cluster_bootstrap_ci(comps, labels, scores, statistic=mean,
                                     n_boot=400, seed=1729)
    iid = cluster_bootstrap_ci(np.arange(len(labels)), labels, scores,
                               statistic=mean, n_boot=400, seed=1729)
    clustered_width = clustered["hi"] - clustered["lo"]
    iid_width = iid["hi"] - iid["lo"]
    assert clustered_width > iid_width, (
        "clustered CI must be wider than the row-iid CI when products repeat")
    assert clustered["lo"] <= 0.5 <= clustered["hi"]


def test_stratified_report_has_overall_and_per_stratum_blocks():
    rows = [
        {"component": "a", "gate": "proceed", "label": 1, "score": 0.9},
        {"component": "b", "gate": "proceed", "label": 0, "score": 0.4},
        {"component": "c", "gate": "hard_no", "label": 0, "score": 0.1},
        {"component": "d", "gate": "hard_no", "label": 0, "score": 0.2},
    ]
    report = stratified_report(
        rows,
        component_of=lambda r: r["component"],
        stratum_of=lambda r: r["gate"],
        label_of=lambda r: r["label"],
        score_of=lambda r: r["score"],
        threshold=0.5, n_boot=50, seed=1729)
    assert set(report["by_stratum"]) == {"proceed", "hard_no"}
    assert report["overall"]["metrics"]["n"] == 4
    for block in report["by_stratum"].values():
        assert {"precision", "recall", "f1", "pr_auc"} <= set(block["cis"])
        assert "lo" in block["cis"]["f1"]


def test_report_cis_are_the_per_statistic_clustered_bootstrap():
    """ONE resample pass feeds all four CIs, so its numbers are theirs.

    The report draws every CI from a single component resample (per-statistic
    loops redraw the same sequence for the same seed). Pinning the equality
    keeps the fused pass from ever becoming a second, drifting bootstrap.
    """
    rows = [{"component": f"c{i // 2}", "gate": "proceed" if i % 3 else "hard_no",
             "label": i % 2, "score": (i % 5) / 5} for i in range(24)]
    report = stratified_report(
        rows, component_of=lambda r: r["component"], stratum_of=lambda r: r["gate"],
        label_of=lambda r: r["label"], score_of=lambda r: r["score"],
        threshold=0.5, n_boot=60, seed=1729)
    comps = [r["component"] for r in rows]
    y_true = np.array([r["label"] for r in rows], dtype=int)
    y_score = np.array([r["score"] for r in rows], dtype=float)
    assert list(report["overall"]["cis"]) == list(CI_STATISTICS)
    for name in CI_STATISTICS:
        want = cluster_bootstrap_ci(
            comps, y_true, y_score, n_boot=60, seed=1729,
            statistic=lambda y, s, name=name: binary_metrics(
                y, s, threshold=0.5)[name])
        assert report["overall"]["cis"][name] == want, name
