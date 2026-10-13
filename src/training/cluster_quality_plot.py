"""Cluster-quality plot: pairwise precision/recall and clustering error rates.

One figure, four bars — pairwise precision, pairwise recall, the over-merge
rate and the under-merge rate — the clustering-quality view the owner reads at
Gate-2. The rates come from a held-out pair scoring:

* ``from_pair_metrics`` — the graph lanes' ``graph_tracks.report.pair_metrics``
  row (tp/fp/fn/precision/recall). Over-merge is ``fp/(tp+fp)`` and under-merge
  is ``fn/(tp+fn)``, the SAME definitions ``training.rand_matching`` uses, so
  the GNN plot and the text calibration plot never disagree on a rate.
* ``from_calibration_metrics`` — a generated ``report.json`` metric row carrying
  the ``calibration_*`` keys directly (the legacy ANN/text report).

:class:`ClusterQualityPlot` owns the figure; the gnn_only lane renders it as a
completed-run deliverable (``graph_tracks.report.complete``). The module keeps a
thin CLI over the legacy ``report.json`` path.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.common import plot_dpi
from core.run_log import RunLogger

_LOG = RunLogger(__name__)

#: The figure's file name (the Gate-2 cluster-quality deliverable).
PLOT_NAME = "cluster_quality.png"

#: The four bars, in order, with their figure labels.
BARS = (("pairwise_precision", "Pairwise\nprecision"),
        ("pairwise_recall", "Pairwise\nrecall"),
        ("over_merge_rate", "Over-merge\nrate"),
        ("under_merge_rate", "Under-merge\nrate"))
_BAR_COLORS = ("#4c72b0", "#55a868", "#c44e52", "#8172b2")


class ClusterQualityPlot:
    """The clustering-quality figure over one held-out pair scoring."""

    def __init__(self, pairwise_precision: float, pairwise_recall: float,
                 over_merge_rate: float, under_merge_rate: float, *,
                 title: str | None = None) -> None:
        for value in (pairwise_precision, pairwise_recall, over_merge_rate, under_merge_rate):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"a cluster-quality rate must sit in [0, 1], got {value}")
        self.rates = {"pairwise_precision": float(pairwise_precision),
                      "pairwise_recall": float(pairwise_recall),
                      "over_merge_rate": float(over_merge_rate),
                      "under_merge_rate": float(under_merge_rate)}
        self.title = title or "Pairwise precision/recall and clustering error rates"

    @classmethod
    def from_pair_metrics(cls, metrics: Mapping, *, split: str | None = None,
                          track: str | None = None) -> ClusterQualityPlot:
        """Build from a ``graph_tracks.report.pair_metrics`` row.

        Over/under-merge reuse ``rand_matching``'s definitions —
        ``fp/(tp+fp)`` and ``fn/(tp+fn)`` — computed from the row's own counts so
        the plot cannot drift from the calibration lane. A row without both
        classes (``precision_defined`` false) has no clustering quality to draw.
        """
        tp, fp, fn = int(metrics["tp"]), int(metrics["fp"]), int(metrics["fn"])
        if not metrics.get("both_classes", True) or (tp + fp) == 0 or (tp + fn) == 0:
            raise ValueError(
                f"pair metrics for split {split!r} carry no usable positive/negative "
                "support; cannot render a cluster-quality plot")
        title = " ".join(part for part in (track, split) if part) or None
        return cls(metrics["precision"], metrics["recall"],
                   fp / (tp + fp), fn / (tp + fn),
                   title=(f"{title} cluster quality" if title else None))

    @classmethod
    def from_calibration_metrics(cls, metric: Mapping) -> ClusterQualityPlot:
        """Build from a ``report.json`` metric row carrying ``calibration_*`` keys."""
        return cls(metric["calibration_pairwise_precision"], metric["calibration_pairwise_recall"],
                   metric["calibration_over_merge_rate"], metric["calibration_under_merge_rate"])

    @classmethod
    def from_report_json(cls, report_json: Path) -> ClusterQualityPlot:
        """Build from a generated ``report.json`` (its first metric fold)."""
        payload = json.loads(Path(report_json).read_text())
        metric = payload["metrics"][min(payload["metrics"])]
        return cls.from_calibration_metrics(metric)

    def render(self, out: Path) -> Path:
        """Write the figure; return its path."""
        out = Path(out)
        labels = [label for _, label in BARS]
        values = [self.rates[key] for key, _ in BARS]
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        bars = ax.bar(labels, values, color=list(_BAR_COLORS))
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("rate")
        ax.set_title(self.title)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, min(value + 0.025, 1.02),
                    f"{value:.3f}", ha="center")
        fig.savefig(out, dpi=plot_dpi())
        plt.close(fig)
        _LOG.info(f"[cluster-quality] wrote {out}")
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("report_json", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.report_json.with_name(f"ann_{PLOT_NAME}")
    print(ClusterQualityPlot.from_report_json(args.report_json).render(out))


if __name__ == "__main__":
    main()
