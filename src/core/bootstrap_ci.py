"""Seeded paired bootstrap confidence intervals over the scored pair population.

MODEL_TRACKS_PLAN.md asks for "paired confidence intervals and repeated seeds".
The model-track lanes train exactly one checkpoint per split, so there are no
repeated training seeds to report: claiming seed replication would be a
fabricated result.  What *is* available honestly is a paired bootstrap over the
same scored pairs, which quantifies sampling uncertainty in the metric itself.

The distinction is recorded in every output (``method`` and
``repeated_training_seeds``) so no downstream reader can mistake a resampling
interval for seed replication.  If a lane ever does train repeated seeds, the
seed interval belongs beside this, not instead of it.

Resampling is over pairs, keeping each pair's ``(label, score)`` together --
that is what makes it *paired*: the same pair can never contribute a label to
one side of a replicate and a score to the other.
"""

from __future__ import annotations

import numpy as np

from core.common import operating_precision, operating_recall, precision_at_recall_key, paired_bootstrap_cfg
from core.schemas import PairedBootstrapSpec

#: Metrics worth an interval. Kept small on purpose: an interval on every
#: threshold-rate column would be noise, and each resample costs a sort.
DEFAULT_METRICS = ("roc_auc", "pr_auc", precision_at_recall_key(), "recall_at_precision")


def _percentile_ci(samples: np.ndarray, confidence: float) -> tuple[float, float]:
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.percentile(samples, [alpha * 100.0, (1.0 - alpha) * 100.0])
    return float(lo), float(hi)


def _statistic(labels: np.ndarray, scores: np.ndarray, metric: str) -> float | None:
    """Compute one metric, or None when the replicate cannot support it."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    if labels.size == 0:
        return None
    both = labels.min() == 0 and labels.max() == 1
    if metric == "roc_auc":
        return float(roc_auc_score(labels, scores)) if both else None
    if metric == "pr_auc":
        return float(average_precision_score(labels, scores)) if both else None
    if metric in (precision_at_recall_key(), "precision_at_recall", "recall_at_precision"):
        if not both:
            return None
        from sklearn.metrics import precision_recall_curve

        precision, recall, _ = precision_recall_curve(labels, scores)
        if metric in (precision_at_recall_key(), "precision_at_recall"):
            body = slice(0, len(precision) - 1)
            ok = recall[body] >= operating_recall()
            return float(precision[body][ok].max()) if ok.any() else None
        target = operating_precision()
        body = slice(0, len(precision) - 1)
        ok = precision[body] >= target
        return float(recall[body][ok].max()) if ok.any() else None
    raise ValueError(f"unsupported bootstrap metric: {metric}")


def paired_bootstrap(
    labels,
    scores,
    *,
    metrics=None,
    track: str,
    split: str,
    spec: dict | None = None,
) -> dict:
    """Return a ``{metric: {point, low, high, resamples_used}}`` mapping.

    ``point`` is the value on the observed population, so a reader can see the
    interval sits around the number already in the summary CSV rather than
    having to reconcile two sources.

    A metric whose replicates are degenerate (single class after resampling,
    which happens on small slices) reports ``point`` with ``low``/``high`` of
    ``None`` and ``resamples_used: 0`` instead of a misleading zero-width
    interval.
    """
    if metrics is None:
        metrics = ("roc_auc", "pr_auc", precision_at_recall_key(), "recall_at_precision")
    spec = PairedBootstrapSpec.model_validate(paired_bootstrap_cfg() if spec is None else spec).model_dump()
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    result: dict[str, dict] = {}
    if not spec["enabled"]:
        return {
            "method": "paired_bootstrap",
            "enabled": False,
            "reason": "evaluation.paired_bootstrap.enabled is false",
            "repeated_training_seeds": None,
            "metrics": result,
        }
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return {
            "method": "paired_bootstrap",
            "enabled": True,
            "seed": int(spec["seed"]),
            "resamples_requested": int(spec["resamples"]),
            "confidence": float(spec["confidence"]),
            "repeated_training_seeds": None,
            "repeated_training_seeds_note": (
                "this lane trains one checkpoint per split; no seed replication "
                "was performed, so these intervals reflect pair-resampling "
                "uncertainty only"
            ),
            "track": track,
            "split": split,
            "rows": int(labels.size),
            "metrics": result,
        }
    rng = np.random.default_rng(int(spec["seed"]))
    n = labels.size
    resamples = int(spec["resamples"])
    confidence = float(spec["confidence"])
    draws = rng.integers(0, n, size=(resamples, n))
    for metric in metrics:
        point = _statistic(labels, scores, metric)
        samples = []
        for row in draws:
            value = _statistic(labels[row], scores[row], metric)
            if value is not None:
                samples.append(value)
        if not samples:
            result[metric] = {
                "point": point,
                "low": None,
                "high": None,
                "resamples_used": 0,
            }
            continue
        low, high = _percentile_ci(np.asarray(samples, dtype=float), confidence)
        result[metric] = {
            "point": point,
            "low": low,
            "high": high,
            "resamples_used": len(samples),
        }
    return {
        "method": "paired_bootstrap",
        "enabled": True,
        "seed": int(spec["seed"]),
        "resamples_requested": resamples,
        "confidence": confidence,
        "repeated_training_seeds": None,
        "repeated_training_seeds_note": (
            "this lane trains one checkpoint per split; no seed replication was "
            "performed, so these intervals reflect pair-resampling uncertainty only"
        ),
        "track": track,
        "split": split,
        "rows": int(n),
        "metrics": result,
    }


__all__ = ["DEFAULT_METRICS", "paired_bootstrap"]