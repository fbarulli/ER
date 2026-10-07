"""One canonical logs root + CR-frame progress capture (owner order A + B).

Pins the two invariants the log-roof order depends on:
1. lane_log()/logs_root() always return paths under TRAIN_ROOT/logs/<lane>/.
2. Captured streams whose payloads carry "\r" tqdm frames are written as
   grep-able lines with the last frame tagged, on both lanes' surfaces.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from cli import log_capture


def test_logs_root_is_the_canonical_train_root_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(log_capture, "TRAIN_ROOT", tmp_path)
    root = log_capture.logs_root()
    assert root == (tmp_path / "logs").resolve(), "one roof, not per-lane sprawl"
    assert root.is_dir()


def test_lane_log_returns_logs_lane_name_path(tmp_path, monkeypatch):
    monkeypatch.setattr(log_capture, "TRAIN_ROOT", tmp_path)
    path = log_capture.lane_log("kaggle", "lane.log")
    assert path.parent == tmp_path / "logs" / "kaggle"
    assert path.name == "lane.log"
    path = log_capture.lane_log("colab", "system.log")
    assert path.parent == tmp_path / "logs" / "colab"


@pytest.mark.parametrize("text", ["", "plain line\n", "no frames here \n"])
def test_progress_frames_without_cr_pass_through_unchanged(text):
    assert log_capture.progress_frames_to_lines(text) == text


def test_progress_cr_frames_become_grepable_lines_with_last_bar():
    text = "10%\r20%\r40%\r100%\r"
    lines = log_capture.progress_frames_to_lines(text).splitlines()
    assert lines == ["10%", "20%", "40%", "100%", "[tqdm] 100%"]
    # no carriage returns survive write time
    assert "\r" not in log_capture.progress_frames_to_lines(text)


def test_progress_blank_cr_frames_never_emit_empty_tag():
    text = "x\n\r\r"
    out = log_capture.progress_frames_to_lines(text)
    assert "\r" not in out
    assert "[tqdm]" not in out or any(
        line.startswith("[tqdm] ") for line in out.splitlines())
