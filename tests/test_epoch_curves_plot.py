"""Pin the per-epoch results artifact: per-epoch series -> LINE plot.

Regression (named): the suite-era reports (``graph_tracks.report.complete`` /
``model_tracks.text_report.complete``, 13a1d2f) dropped the per-epoch view the
old reporting lane rendered as lines across epochs
(``training/generate_training_report.py`` ``training_vs_dev_loss_by_epoch`` /
``ranking_quality_by_epoch``); what survived was final-snapshot plots only — a
single bar figure (``cluster_quality.png``, 1a08344). Both completed-run
reports now carry ``<track>__epoch_curves.png`` again. The pin: each epoch's
OWN logged values reach the figure as ``plot`` lines (never ``bar``, never an
aggregate), under the lane artifact name.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from graph_tracks.artifacts import name
from graph_tracks.report import (
    EPOCH_CURVES_PLOT,
    epoch_curves_plot,
    graph_epoch_curves,
    text_epoch_curves,
)

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture()
def plot_calls(monkeypatch: pytest.MonkeyPatch):
    """Spy on every Axes.plot/Axes.bar call the renderer makes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.axes

    lines: list[tuple[list, list]] = []
    bars: list = []
    real_plot, real_bar = matplotlib.axes.Axes.plot, matplotlib.axes.Axes.bar

    def spy_plot(self, *args, **kwargs):
        lines.append((list(args[0]), list(args[1])))
        return real_plot(self, *args, **kwargs)

    def spy_bar(self, *args, **kwargs):
        bars.append(args)
        return real_bar(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", spy_plot)
    monkeypatch.setattr(matplotlib.axes.Axes, "bar", spy_bar)
    return lines, bars


def test_graph_epoch_metrics_render_as_a_line_plot(tmp_path: Path, plot_calls) -> None:
    """jsonl rows (one per epoch) -> per-epoch LINE series under the lane name."""
    lines, bars = plot_calls
    rows = [
        {"epoch": 1, "train_loss": 1.10, "train_classification_loss": 0.70,
         "train_metric_loss": 0.40, "dev_pr_auc": 0.81, "dev_precision_at_recall": 0.72},
        {"epoch": 2, "train_loss": 0.90, "train_classification_loss": 0.55,
         "train_metric_loss": 0.35, "dev_pr_auc": 0.87, "dev_precision_at_recall": 0.78},
        {"epoch": 3, "train_loss": 0.75, "train_classification_loss": 0.45,
         "train_metric_loss": 0.30, "dev_pr_auc": 0.89, "dev_precision_at_recall": 0.80},
    ]
    loss, quality = graph_epoch_curves(rows)
    out = tmp_path / name("gnn_only", EPOCH_CURVES_PLOT)
    epoch_curves_plot(loss, quality, out, title="gnn_only per-epoch results")

    assert out.name == "gnn_only__epoch_curves.png"  # the lane artifact name
    assert out.read_bytes()[:8] == _PNG_MAGIC
    assert out.stat().st_size > 1000  # a real figure, not an empty stub
    # EVERY series is a line, never a bar...
    assert not bars
    assert len(lines) == len(loss) + len(quality)
    # ...carrying each epoch's OWN value at its OWN epoch (not cumulative,
    # not aggregated): x is the epoch sequence, y the logged values verbatim.
    assert all(x == [1.0, 2.0, 3.0] for x, _ in lines)
    drawn = [y for _, y in lines]
    assert [1.10, 0.90, 0.75] in drawn  # train loss, per epoch
    assert [0.81, 0.87, 0.89] in drawn  # dev PR-AUC, per epoch


def test_text_log_history_maps_to_per_epoch_curves() -> None:
    """The HF trainer log_history yields one point per logged epoch event."""
    log_history = [
        {"epoch": 0.5, "loss": 1.71, "step": 10},
        {"epoch": 0.5, "eval_loss": 0.143, "eval_dev_cosine_ap": 0.986,
         "eval_dev_cosine_precision": 0.96, "eval_dev_cosine_recall": 0.99, "step": 10},
        {"epoch": 1.0, "loss": 1.51, "step": 20},
        {"epoch": 1.0, "eval_loss": 0.104, "eval_dev_cosine_ap": 0.988,
         "eval_dev_cosine_precision": 0.97, "eval_dev_cosine_recall": 0.99, "step": 20},
    ]
    loss, quality = text_epoch_curves(log_history)
    assert loss["train loss"] == ([0.5, 1.0], [1.71, 1.51])
    assert loss["dev loss"] == ([0.5, 1.0], [0.143, 0.104])
    assert quality["dev AP"] == ([0.5, 1.0], [0.986, 0.988])
    # an empty history is a loud no-op signal, never a drawn figure
    assert text_epoch_curves([]) == ({}, {})
    # and a history without eval events still carries the train loss alone
    assert text_epoch_curves([log_history[0]])[0].keys() == {"train loss"}


def test_epoch_curves_plot_rejects_an_empty_history(tmp_path: Path) -> None:
    """No series -> no artifact: the renderer fails loud instead of drawing nothing."""
    with pytest.raises(ValueError, match="at least one per-epoch series"):
        epoch_curves_plot({}, {}, tmp_path / "never.png", title="empty")
