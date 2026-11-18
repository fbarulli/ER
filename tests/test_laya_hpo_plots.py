"""The ONE consolidated HPO plot artifact (cli.laya_hpo_plots).

Owner directive: all losses + all other metrics in a SINGLE plot (like the GPU
SM Clock / Network Traffic multi-run panels), top-3 runs only. Pinned here
offline: the fetch over a fake run family and the rendered artifact (PNG +
the series JSON that is the plotted truth).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cli.laya_hpo_plots import RANK_METRIC, ConsolidatedPlot, RunHistoryFetcher


class FakeRun:
    def __init__(self, run_id, name, state, summary, rows):
        self.id, self.name, self.state = run_id, name, state
        self.summary = summary
        self._rows = rows

    def scan_history(self, keys: list[str] | None = None) -> list[dict[str, Any]]:
        return [row for row in self._rows
                if keys is None or all(key in row for key in keys)]


class FakeReader:
    def __init__(self, runs):
        self._runs = runs

    def family(self):
        return list(self._runs)


def _trial(index: int) -> FakeRun:
    rows = [{"epoch": epoch, "_step": 4 + epoch,
             "train/mean_loss": 0.5 - 0.05 * index - 0.1 * epoch,
             "train/lr": 1e-5, "dev/accuracy": 0.8 + 0.01 * index,
             "dev/loss": 0.2, "epoch_time_s": 60.0 + index,
             "train/grad_norm": "NaN" if epoch == 2 else 0.25}
            for epoch in (1, 2, 3)]
    rows.extend({"_step": 100 + step, "epoch": 3,
                 "gpu/util_pct": 90.0 + index, "network.sent": 1000.0 * step}
                for step in (1, 2))
    rows.append({"_step": 103, "dev/loss": 0.15})
    return FakeRun(f"id{index}", f"tag-trial-{index}", "finished",
                   {RANK_METRIC: 0.80 + 0.01 * index}, rows)


def test_consolidated_plot_overlays_every_metric_for_the_top_runs(tmp_path: Path) -> None:
    """5 trial runs + one history-less controller: the fetch keeps every run
    (the controller ranks last), and the artifact overlays ALL per-epoch
    metrics + the system series for the TOP-3 ranked runs only."""
    controller = FakeRun("c0", "tag", "finished", {}, [])
    runs = [controller] + [_trial(index) for index in range(5)]
    histories = RunHistoryFetcher(FakeReader(runs)).fetch()
    assert len(histories) == 6

    result = ConsolidatedPlot(histories, top_runs=3,
                              out_dir=tmp_path).render()
    assert result["top"] == ["tag-trial-4", "tag-trial-3", "tag-trial-2"]

    png = Path(result["plot"])
    assert png.is_file() and png.stat().st_size > 0
    document = json.loads(Path(result["series"]).read_text(encoding="utf-8"))
    assert set(document["plotted"]) == set(result["top"])
    plotted = document["plotted"]["tag-trial-4"]
    # ONE plot carries every loss + metric series...
    for key in ("train/mean_loss", "train/lr", "dev/accuracy", "dev/loss",
                "epoch_time_s"):
        assert key in plotted, key
    # ...and the system panel series ride the same artifact, same top runs.
    assert "gpu/util_pct" in plotted and "network.sent" in plotted
    loss = plotted["train/mean_loss"]
    assert [point[0] for point in loss] == [1, 2, 3]
    assert loss[0][1] == pytest.approx(0.2)
    assert loss[2][1] == pytest.approx(0.0)
    assert [point[0] for point in plotted["gpu/util_pct"]] == [101, 102]
    assert plotted["train/grad_norm"] == [[1, 0.25], [3, 0.25]]
    report = next(run for run in document["runs"] if run["name"] == "tag-trial-4")
    assert len(report["history_gaps"]) == 2
    gap = report["history_gaps"][0]
    assert gap["metric"] == "train/grad_norm" and gap["step"] == 6
    assert "Traceback (most recent call last)" in gap["traceback"]
    assert "finite number" in gap["traceback"]
    missing_coordinate = report["history_gaps"][1]
    assert missing_coordinate["metric"] == "dev/loss"
    assert missing_coordinate["step"] == 103
    assert result["gaps"]


class TestEmptyHistory:
    def test_plot_reports_missing_metrics_instead_of_succeeding(
        self, tmp_path: Path,
    ) -> None:
        histories = RunHistoryFetcher(FakeReader([
            FakeRun("c0", "tag", "finished", {}, []),
        ])).fetch()
        with pytest.raises(RuntimeError, match="No plottable W&B metrics"):
            ConsolidatedPlot(histories, top_runs=3, out_dir=tmp_path).render()
        report = json.loads((tmp_path / "series.json").read_text())
        assert report["plotted"] == {}
