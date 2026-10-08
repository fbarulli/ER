"""Component-clustered holdout evaluation: honest error bars + gate strata."""
from __future__ import annotations

import numpy as np

from core.holdout_eval import (
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
