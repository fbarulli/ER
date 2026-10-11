"""The ONE consolidated laya HPO plot artifact (owner directive 2026-10-10).

Owner: "all the loss and all other metrics in a single plot, just like GPU SM
Clock Speed (MHz); Network Traffic (Bytes) — only top 3 runs in the same
plot." W&B's own dashboards cannot be reconfigured from the staged kernels
(no workspace API without an extra dependency), so the mechanism is a
``uv run``-rendered plot artifact over the SAME cloud history the kernels log:

    uv run python -m cli.laya_hpo_plots --run-tag <RUN_TAG> [--top 3]

* ONE consolidated panel overlays EVERY per-epoch metric the shared kernel
  wandb block logs (train/mean_loss, train/lr, train/grad_norm, dev/accuracy,
  dev/loss, epoch_time_s — the ``EPOCH_STEP_METRICS`` namespaces, x = epoch),
  one series per metric per run, exactly like wandb's multi-run system panels.
* A second panel carries the system/GPU series (gpu/util_pct, gpu/mem_used_mb,
  network.sent/recv — x = the global step) for the SAME runs.
* BOTH panels are limited to the TOP-N runs ranked by the promoted dev
  accuracy (``select/best_dev_accuracy``); N is the space-YAML SSOT
  (``plot.top_runs``), the CLI ``--top`` overrides.

Run resolution reuses the canonical reader (``core.wandb_ctx.WandbRunReader``
— the ONE name->run-family resolver); this module never re-implements it. The
artifact lands under the lane's staging roof (``results/laya_lane/plots/<run
tag>/``): ``consolidated.png`` + ``series.json`` (the plotted truth, so a
report never depends on re-fetching the cloud).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, RootModel

#: The per-epoch metric keys the shared kernel wandb block logs (the
#: ``EPOCH_STEP_METRICS`` namespaces' concrete series; x = epoch).
PLOT_METRICS: tuple[str, ...] = (
    "train/mean_loss", "train/lr", "train/grad_norm",
    "dev/accuracy", "dev/loss", "epoch_time_s",
)
#: The system/GPU series for the second panel (x = the run's global step).
SYSTEM_METRICS: tuple[str, ...] = (
    "gpu/util_pct", "gpu/mem_used_mb", "network.sent", "network.recv",
)
#: The ranking key (the promoted dev accuracy rides every trial run summary).
RANK_METRIC = "select/best_dev_accuracy"
#: Every column fetched in ONE scan_history pass per run.
HISTORY_KEYS: tuple[str, ...] = (
    *PLOT_METRICS, *SYSTEM_METRICS,
)


class HistoryPoint(RootModel[tuple[FiniteFloat, FiniteFloat]]):
    """A finite coordinate/value pair shared by the reader and plot."""


class WandbHistoryRow(BaseModel):
    """Parse W&B's sparse network row before building metric series."""

    model_config = ConfigDict(extra="allow")
    epoch: float | None = None
    step: float | None = Field(default=None, alias="_step")

    def point(self, metric: str) -> HistoryPoint:
        coordinate = self.step if metric in SYSTEM_METRICS else self.epoch
        return HistoryPoint.model_validate((coordinate, self.model_extra[metric]))


class HistoryGap(BaseModel):
    metric: str
    step: float | None
    traceback: str


class RunHistory(BaseModel):
    id: str
    name: str
    state: str
    rank_value: float | None
    history: dict[str, list[HistoryPoint]] = Field(default_factory=dict)
    history_gaps: list[HistoryGap] = Field(default_factory=list)
    history_error: str | None = None


class RunHistoryFetcher:
    """Per-epoch + system history for one run family (over the canonical reader).

    Dependency-injected with a ``WandbRunReader``; ONE ``scan_history`` pass
    per run keeps every requested key. A run whose history fetch fails is
    recorded with its full reason (never silently dropped): the JSON artifact
    shows what was readable. Invalid points carry a metric, source step and
    full traceback; a missing coordinate never becomes a misleading chart.
    """

    def __init__(self, reader: Any) -> None:
        self._reader = reader

    def fetch(self) -> list[RunHistory]:
        import traceback

        fetched: list[RunHistory] = []
        for run in self._reader.family():
            entry = RunHistory(
                id=str(run.id), name=str(run.name), state=str(run.state),
                rank_value=self._rank_value(run))
            try:
                # W&B's keys filter intersects rows; these sinks log separately.
                for row in run.scan_history():
                    record = WandbHistoryRow.model_validate(row)
                    for key in HISTORY_KEYS:
                        if key not in record.model_extra:
                            continue
                        try:
                            point = record.point(key)
                        except ValueError:
                            entry.history_gaps.append(HistoryGap(
                                metric=key, step=record.step,
                                traceback=traceback.format_exc()))
                        else:
                            entry.history.setdefault(key, []).append(point)
            except Exception as error:  # noqa: BLE001 - recorded, never fatal
                entry.history_error = (
                    f"{type(error).__name__}: {error}\n"
                    f"{traceback.format_exc()}")
            fetched.append(entry)
        return fetched

    @staticmethod
    def _rank_value(run: Any) -> float | None:
        summary = dict(getattr(run, "summary", {}) or {})
        value = summary.get(RANK_METRIC)
        return float(value) if isinstance(value, (int, float)) else None


class ConsolidatedPlot:
    """ONE figure: all per-epoch losses+metrics overlaid in a single plot and
    the system panel beside it — both limited to the top-N ranked runs."""

    def __init__(self, histories: list[RunHistory], *, top_runs: int,
                 out_dir: Path) -> None:
        self._histories = list(histories)
        self._top_runs = max(1, int(top_runs))
        self._out_dir = Path(out_dir)

    def top(self) -> list[RunHistory]:
        """The top-N runs by the promoted dev accuracy (unranked last)."""
        ranked = sorted(
            self._histories,
            key=lambda entry: (entry.rank_value is not None,
                               entry.rank_value or 0.0),
            reverse=True)
        return ranked[:self._top_runs]

    def render(self) -> dict[str, Any]:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top = self.top()
        self._out_dir.mkdir(parents=True, exist_ok=True)
        figure, (metrics_axis, system_axis) = plt.subplots(
            2, 1, figsize=(12.0, 9.0))
        plotted: dict[str, Any] = {}
        for entry in top:
            for key in PLOT_METRICS:
                series = entry.history.get(key, [])
                if series:
                    metrics_axis.plot([p.root[0] for p in series],
                                      [p.root[1] for p in series], marker=".",
                                      markersize=3, linewidth=1.0,
                                      label=f"{entry.name} {key}")
                    plotted.setdefault(entry.name, {})[key] = [
                        point.model_dump(mode="json") for point in series]
            for key in SYSTEM_METRICS:
                series = entry.history.get(key, [])
                if series:
                    system_axis.plot([p.root[0] for p in series],
                                     [p.root[1] for p in series], linewidth=0.8,
                                     label=f"{entry.name} {key}")
                    plotted.setdefault(entry.name, {})[key] = [
                        point.model_dump(mode="json") for point in series]
        metrics_axis.set_title(
            f"laya HPO — ALL per-epoch metrics, top {len(top)} runs")
        metrics_axis.set_xlabel("epoch")
        metrics_axis.grid(True, alpha=0.3)
        if metrics_axis.lines:
            metrics_axis.legend(fontsize=6, ncol=2)
        system_axis.set_title("GPU / network (same top runs)")
        system_axis.set_xlabel("step")
        system_axis.grid(True, alpha=0.3)
        if system_axis.lines:
            system_axis.legend(fontsize=6, ncol=2)
        figure.tight_layout()
        png = self._out_dir / "consolidated.png"
        figure.savefig(png, dpi=140)
        plt.close(figure)
        reports = [entry.model_dump(mode="json", exclude={"history"})
                   for entry in self._histories]
        document = {
            "top_runs": self._top_runs,
            "rank_metric": RANK_METRIC,
            "plot_metrics": list(PLOT_METRICS),
            "system_metrics": list(SYSTEM_METRICS),
            "runs": reports,
            "plotted": plotted,
        }
        series = self._out_dir / "series.json"
        series.write_text(json.dumps(document, indent=2) + "\n",
                          encoding="utf-8")
        if not plotted:
            raise RuntimeError(f"No plottable W&B metrics; details in {series}")
        return {"plot": str(png), "series": str(series),
                "top": [entry.name for entry in top],
                "runs": len(self._histories),
                "gaps": [report for report in reports
                         if report["history_gaps"] or report["history_error"]]}


class HpoPlotFactory:
    """Construction owner: reader -> fetcher -> plot (no assembly at call sites)."""

    @staticmethod
    def top_runs_default() -> int:
        """The space-YAML SSOT for the panel limit (``plot.top_runs``)."""
        from cli import laya_hpo

        plot = laya_hpo.load_space().get("plot") or {}
        return int(plot.get("top_runs", 3))

    @staticmethod
    def out_dir(run_tag: str) -> Path:
        from cli import laya_lane

        return laya_lane.staging_dir() / "plots" / run_tag

    @classmethod
    def build(cls, run_tag: str, *, project: str | None = None,
              top_runs: int | None = None, out_dir: Path | None = None
              ) -> dict[str, Any]:
        from core.wandb_ctx import WandbRunReader

        if not WandbRunReader.available():
            raise RuntimeError(
                "[laya-hpo-plots] no WANDB_API_KEY resolvable: the plot "
                "artifact renders from the cloud history (export "
                "WANDB_API_KEY / drop it in ../.env first)")
        reader = WandbRunReader(run_tag=run_tag, project=project)
        histories = RunHistoryFetcher(reader).fetch()
        if not histories:
            raise RuntimeError(
                f"[laya-hpo-plots] no wandb run family for tag {run_tag!r}")
        plot = ConsolidatedPlot(
            histories,
            top_runs=(top_runs if top_runs is not None
                      else cls.top_runs_default()),
            out_dir=(out_dir if out_dir is not None else cls.out_dir(run_tag)))
        return plot.render()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", required=True,
                        help="the run tag (display name) of the HPO session")
    parser.add_argument("--project", default=None,
                        help="wandb project (default: tracking.wandb.project)")
    parser.add_argument("--top", type=int, default=None,
                        help="top-N runs in the plot (default: the space "
                             "YAML plot.top_runs)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output dir (default: results/laya_lane/plots/"
                             "<run-tag>/)")
    args = parser.parse_args(argv)
    result = HpoPlotFactory.build(args.run_tag, project=args.project,
                                  top_runs=args.top, out_dir=args.out)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
