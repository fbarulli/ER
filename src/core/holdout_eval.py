"""Component-clustered holdout evaluation for the laya-vs-tracks comparison.

Three problems this module fixes, in one place:

* **Row-level random splits leak products.** A pair's identity is a *component*
  (a connected cluster of GTINs, ``training.folds``), so resampling/scoring must
  be clustered by component, never by row.
* **Accuracy lies on imbalanced questions** (identity is ~94% "different"), so
  the reported metric is precision/recall/F1/PR-AUC at a stated threshold.
* **A point estimate hides its uncertainty.** ``cluster_bootstrap_cis`` draws
  whole components with replacement — once, feeding every statistic — so the
  interval widens when the effective number of independent products is small,
  which is exactly the truth here.

Pure numpy + sklearn; no config/torch import, safe to unit test hermetically.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score

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
#: The CIs a stratified holdout report draws, in the order it reports them.
#: One spelling: this module is embedded verbatim in the remote holdout kernel,
#: so the operator-box report and the session report cannot name different CIs.
CI_STATISTICS: tuple[str, ...] = ("precision", "recall", "f1", "pr_auc")


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
    # ONE pass over the labels; bin index is ``pred * 2 + y_true``, so a
    # non-binary label overflows the four bins and fails loud instead of being
    # silently counted as a negative.
    tn, fn, fp, tp = (int(count) for count in
                      np.bincount(pred * 2 + y_true, minlength=4))
    positives, negatives = tp + fn, tn + fp
    # P/R/F1 are the counts above, not a second opinion: sklearn's
    # `zero_division=0` IS an empty denominator scoring 0.0, and asking
    # `precision_recall_fscore_support` for them recomputed the same confusion
    # matrix — 60% of this function's cost, paid once per bootstrap draw.
    predicted_positive = tp + fp
    precision = tp / predicted_positive if predicted_positive else 0.0
    recall = tp / positives if positives else 0.0
    # sklearn's f-score IS the confusion-count form 2*tp/(true+predicted), not
    # 2*P*R/(P+R) — the two spellings differ in the last bit, and this one is
    # bit-identical to the `precision_recall_fscore_support` call it replaces.
    f1_denominator = positives + predicted_positive
    f1 = 2.0 * tp / f1_denominator if f1_denominator else 0.0
    return {
        "n": n,
        "positives": positives,
        "negatives": negatives,
        "threshold": float(threshold),
        "accuracy": (tp + tn) / n,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": (float(average_precision_score(y_true, y_score))
                   if positives and negatives else None),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def _percentile_ci(samples: Sequence[float], point: float | None, *,
                   alpha: float) -> dict:
    """The one CI dict spelling: a statistic that never drew reports no interval."""
    value = None if point is None else float(point)
    if not samples:
        return {"point": value, "lo": None, "hi": None, "n_boot": 0, "alpha": alpha}
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": value, "lo": float(lo), "hi": float(hi),
            "n_boot": len(samples), "alpha": alpha}


def cluster_bootstrap_cis(
    components: Sequence[object],
    y_true: Sequence[int],
    y_score: Sequence[float],
    *,
    batch_statistic: Callable[[np.ndarray, np.ndarray], Mapping[str, float | None]],
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, dict]:
    """Percentile CIs for EVERY statistic named, under ONE *component* resample.

    Whole components are drawn with replacement; every row of a drawn component
    enters the resample. This is the honest interval when rows within a product
    are correlated — an i.i.d. row bootstrap would understate it.

    ``batch_statistic`` returns every statistic from one call, so one resample
    feeds every CI: per-statistic loops redraw the SAME sequence for the same
    seed, so fusing them changes no number — it stops paying for the resample
    and the metric pass once per statistic. A statistic returning ``None``
    (e.g. an undefined PR-AUC on a degenerate resample) drops that draw, so each
    CI reports its own ``n_boot``.
    """
    components = np.asarray(components, dtype=object)
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    unique = np.unique(components)
    rows_of = [np.where(components == component)[0] for component in unique]
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {}
    for _ in range(int(n_boot)):
        picks = rng.choice(len(unique), size=len(unique), replace=True)
        idx = np.concatenate([rows_of[pick] for pick in picks])
        for name, value in batch_statistic(y_true[idx], y_score[idx]).items():
            if value is not None and np.isfinite(value):
                draws.setdefault(name, []).append(float(value))
    observed = batch_statistic(y_true, y_score)
    return {name: _percentile_ci(draws.get(name, ()), observed[name], alpha=alpha)
            for name in observed}


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
    """Percentile CI for ONE ``statistic`` — the single-name ``cluster_bootstrap_cis``."""
    single = "statistic"
    return cluster_bootstrap_cis(
        components, y_true, y_score,
        batch_statistic=lambda y, s: {single: statistic(y, s)},
        n_boot=n_boot, seed=seed, alpha=alpha)[single]


def _thresholded_metrics(
        threshold: float) -> Callable[[np.ndarray, np.ndarray], dict[str, float | None]]:
    """ONE ``binary_metrics`` pass per resample -> the four reported CIs."""
    def batch(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float | None]:
        metrics = binary_metrics(y_true, y_score, threshold=threshold)
        return {name: metrics[name] for name in CI_STATISTICS}
    return batch


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
        return {
            "metrics": binary_metrics(y_true, y_score, threshold=threshold),
            "cis": cluster_bootstrap_cis(
                comps, y_true, y_score,
                batch_statistic=_thresholded_metrics(threshold),
                n_boot=n_boot, seed=seed, alpha=alpha),
        }

    return {
        "threshold": threshold,
        "alpha": alpha,
        "n_boot": n_boot,
        "overall": _block(rows),
        "by_stratum": {name: _block(subset)
                       for name, subset in sorted(by_stratum.items())},
    }
