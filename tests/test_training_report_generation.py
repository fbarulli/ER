"""The consolidated training report: config-owned path, metrics-only default.

The report location is a config binding (``files.training_report``), and plot
generation is optional and OFF by default, so the basic metric results are
produced without any matplotlib artifacts.
"""
import json

import pandas as pd
import pytest

from core import common
from training import generate_training_report as report


@pytest.fixture(autouse=True)
def _restore_plot_flag():
    yield
    report._PLOTS_ENABLED = False


def _metrics_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fold": 0,
                "status": "ok",
                "auc": 0.81,
                "pr_auc": 0.6,
                "youden_thr": 0.55,
                "f1_at_0.55": 0.7,
                "precision_at_0.55": 0.72,
                "recall_at_0.55": 0.68,
                "final_train_loss": 0.2,
                "train_loss_hist": "[0.5,0.3]",
                "dev_loss_hist": "[0.6,0.4]",
                "dev_ap_hist": "[0.4,0.6]",
            }
        ]
    )


def _write_metrics(tmp_path):
    metrics = tmp_path / "fold_metrics.csv"
    _metrics_frame().to_csv(metrics, index=False)
    return metrics


def test_report_name_and_default_directory_come_from_the_binding():
    binding = common.F["training_report"]
    assert report.REPORT_PATH == binding
    assert report.REPORT_NAME == binding.name
    # the producer no longer hardcodes the leaf or the results subpath
    assert report.REPORT_NAME == "report.json"
    assert report.REPORT_PATH.parent == common.RESULTS


def test_metrics_only_report_is_written_without_matplotlib_artifacts(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(report, "_run_robust_validation", lambda *a, **k: None)
    out = tmp_path / "out"

    payload = report.generate_report(_write_metrics(tmp_path), [], out)

    assert payload["report_version"] == 3
    assert payload["folds"] == 1
    written = out / report.REPORT_NAME
    assert written.is_file()
    assert json.loads(written.read_text())["report_version"] == 3
    assert not sorted(out.glob("*.png"))


def test_default_out_dir_is_the_bound_report_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "_run_robust_validation", lambda *a, **k: None)
    monkeypatch.setattr(report, "REPORT_PATH", tmp_path / "results" / report.REPORT_NAME)

    report.generate_report(_write_metrics(tmp_path), [], None)

    assert (tmp_path / "results" / report.REPORT_NAME).is_file()


def test_plots_are_written_only_when_requested(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "_run_robust_validation", lambda *a, **k: None)
    metrics = _write_metrics(tmp_path)
    out = tmp_path / "out"

    report.generate_report(metrics, [], out, plots=True)

    assert list(out.glob("*.png"))
