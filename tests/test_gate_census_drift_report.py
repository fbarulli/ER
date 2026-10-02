from types import SimpleNamespace

import pandas as pd

from core import common


def test_replayed_outcomes_drive_sample_report(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "RESULTS", tmp_path)
    labels = tmp_path / "labels.csv"
    pd.DataFrame([
        ("001", "002", 1), ("003", "004", 0), ("005", "006", 1),
    ], columns=["gtin1", "gtin2", "true_label"]).to_csv(labels, index=False)
    current = pd.DataFrame([
        ("001", "002", "hard_no", "new blocker"),
        ("003", "004", "proceed", "new compatible"),
        ("005", "006", "fallback", "new review"),
    ], columns=["gtin1", "gtin2", "gate_decision", "gate_reason"])
    report = common.gate_census_drift_report(
        measured={"total_pairs": 3, "hard_no": 1, "proceed": 1, "fallback": 1},
        previous_labeled_csv=labels, current_gate=current,
    )
    assert report["degraded"] == {
        "lost_true_block": 1, "lost_true_review": 1,
        "new_merge_risk": 1, "lost_neg_review": 0,
    }
    assert report["samples"]["lost_true_block"][0]["gtin1"] == "001"
    assert report["samples"]["lost_true_block"][0]["new_reason"] == "new blocker"


def test_missing_labels_read_repo_relative_head_from_any_cwd(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.setattr(common, "TRAIN_ROOT", tmp_path)
    monkeypatch.setattr(common, "RESULTS", tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout=b"gtin1,gtin2,true_label\n001,002,1\n")

    monkeypatch.setattr(subprocess, "run", run)
    current = pd.DataFrame([
        ("001", "002", "proceed", "compatible"),
    ], columns=["gtin1", "gtin2", "gate_decision", "gate_reason"])
    report = common.gate_census_drift_report(
        measured={"total_pairs": 1, "proceed": 1},
        previous_labeled_csv=tmp_path / "data" / "labeled_pairs.csv",
        current_gate=current,
    )
    assert calls == [(
        ["git", "show", "HEAD:data/labeled_pairs.csv"],
        {"cwd": tmp_path, "capture_output": True},
    )]
    assert report["survived"] == 1
