"""Component-clustered holdout evaluation for the laya-vs-tracks comparison.

Three problems this module fixes, in one place:

* **Row-level random splits leak products.** A pair's identity is a *component*
  (a connected cluster of GTINs, ``training.folds``), so resampling/scoring must
  be clustered by component, never by row.
* **Accuracy lies on imbalanced questions** (identity is ~94% "different"), so
  the reported metric is precision/recall/F1/PR-AUC at a stated threshold.
* **A point estimate hides its uncertainty.** ``cluster_bootstrap_ci`` draws
  whole components with replacement, so the interval widens when the effective
  number of independent products is small — which is exactly the truth here.

Pure numpy + sklearn; no config/torch import, safe to unit test hermetically.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
)

#: Rows are any mapping carrying a component key, a stratum tag, a binary
#: label and a model score; the caller supplies the accessors.
Row = Mapping[str, object]

DEFAULT_THRESHOLD = 0.5
DEFAULT_BOOTSTRAP = 2000
# Deliberate per-experiment bootstrap seed (NOT core.common.SED): this module
# is hermetic (no config import) and the CI draws are a reporting concern,
# not the training determinism seed.
DEFAULT_SEED = 1729
DEFAULT_ALPHA = 0.05


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, *,
                   threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Thresholded binary metrics + PR-AUC for one slice.

    A slice with a single observed class scores ``precision``/``recall``/``f1``
    per the sklearn ``zero_division=0`` convention (1.0 only when every
    prediction is a true positive) and ``pr_auc`` as ``None`` — a metric that
    cannot be estimated is reported as missing, never as perfect.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n = int(y_true.size)
    if n == 0:
        return {"n": 0, "positives": 0, "negatives": 0, "threshold": threshold,
                "accuracy": None, "precision": None, "recall": None, "f1": None,
                "pr_auc": None, "tp": 0, "fp": 0, "tn": 0, "fn": 0}
    pred = (y_score >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0)
    both_classes = len(set(y_true.tolist())) > 1
    return {
        "n": n,
        "positives": int(np.sum(y_true == 1)),
        "negatives": int(np.sum(y_true == 0)),
        "threshold": float(threshold),
        "accuracy": float(np.mean(pred == y_true)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": (float(average_precision_score(y_true, y_score))
                   if both_classes else None),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def cluster_bootstrap_ci(
    components: Sequence[object],
    y_true: Sequence[int],
    y_score: Sequence[float],
    *,
    statistic: Callable[[np.ndarray, np.ndarray], float | None],
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> dict:
    """Percentile CI for ``statistic`` under a *component* (cluster) bootstrap.

    Whole components are drawn with replacement; every row of a drawn
    component enters the resample. This is the honest interval when rows within
    a product are correlated — an i.i.d. row bootstrap would understate it.
    ``statistic`` returning ``None`` (e.g. an undefined PR-AUC on a degenerate
    resample) drops that draw.
    """
    components = np.asarray(components, dtype=object)
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    unique = np.unique(components)
    rows_of: dict[object, np.ndarray] = {
        comp: np.where(components == comp)[0] for comp in unique}
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(int(n_boot)):
        picks = rng.choice(unique, size=unique.size, replace=True)
        idx = np.concatenate([rows_of[comp] for comp in picks])
        value = statistic(y_true[idx], y_score[idx])
        if value is not None and np.isfinite(value):
            samples.append(float(value))
    point = statistic(y_true, y_score)
    if not samples:
        return {"point": (None if point is None else float(point)),
                "lo": None, "hi": None, "n_boot": 0, "alpha": alpha}
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": (None if point is None else float(point)),
            "lo": float(lo), "hi": float(hi),
            "n_boot": len(samples), "alpha": alpha}


def stratified_report(
    rows: Iterable[Row],
    *,
    component_of: Callable[[Row], object],
    stratum_of: Callable[[Row], str],
    label_of: Callable[[Row], int],
    score_of: Callable[[Row], float],
    threshold: float = DEFAULT_THRESHOLD,
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> dict:
    """Point metrics + component-bootstrap CIs, overall and per stratum.

    ``component_of`` must group rows that share a product; ``stratum_of`` tags
    the gate-difficulty bucket (e.g. gate verdict/reason). Each stratum is
    reported with its own clustered CI so a model that only wins on the easy
    bucket is visible immediately.
    """
    rows = list(rows)
    by_stratum: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        by_stratum[stratum_of(row)].append(row)

    def _block(subset: list[Row]) -> dict:
        if not subset:
            return {"metrics": binary_metrics(np.array([]), np.array([]),
                                              threshold=threshold),
                    "cis": {}}
        comps = [component_of(r) for r in subset]
        y_true = np.array([label_of(r) for r in subset], dtype=int)
        y_score = np.array([score_of(r) for r in subset], dtype=float)

        def _f1(t, s):
            return binary_metrics(t, s, threshold=threshold)["f1"]

        def _pr_auc(t, s):
            if len(set(t.tolist())) <= 1:
                return None
            return float(average_precision_score(t, s))

        def _precision(t, s):
            return binary_metrics(t, s, threshold=threshold)["precision"]

        def _recall(t, s):
            return binary_metrics(t, s, threshold=threshold)["recall"]

        return {
            "metrics": binary_metrics(y_true, y_score, threshold=threshold),
            "cis": {
                "precision": cluster_bootstrap_ci(
                    comps, y_true, y_score, statistic=_precision,
                    n_boot=n_boot, seed=seed, alpha=alpha),
                "recall": cluster_bootstrap_ci(
                    comps, y_true, y_score, statistic=_recall,
                    n_boot=n_boot, seed=seed, alpha=alpha),
                "f1": cluster_bootstrap_ci(
                    comps, y_true, y_score, statistic=_f1,
                    n_boot=n_boot, seed=seed, alpha=alpha),
                "pr_auc": cluster_bootstrap_ci(
                    comps, y_true, y_score, statistic=_pr_auc,
                    n_boot=n_boot, seed=seed, alpha=alpha),
            },
        }

    return {
        "threshold": threshold,
        "alpha": alpha,
        "n_boot": n_boot,
        "overall": _block(rows),
        "by_stratum": {name: _block(subset)
                       for name, subset in sorted(by_stratum.items())},
    }
