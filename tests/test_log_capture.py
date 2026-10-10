"""One canonical logs root + CR-frame progress capture (owner order A + B).

Pins the invariants the log-roof order depends on:
1. lane_log()/logs_root() always return paths under TRAIN_ROOT/logs/<lane>/.
2. A run launched from a linked .worktrees/<name> checkout logs to the
   CANONICAL (main) tree — never the scratch checkout.
3. Captured streams whose payloads carry "\r" tqdm frames are written as
   grep-able lines with the last frame tagged, on both lanes' surfaces.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from cli import log_capture
from core.project_root import ProjectRoot


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


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


def test_a_linked_worktree_launch_logs_to_the_canonical_tree(tmp_path, monkeypatch):
    """A run launched from ``<canonical>/.worktrees/<name>`` logs to the MAIN tree.

    The roof the colab lane actually writes (``lane_log_at``) resolves through
    ``ProjectRoot.canonical``, so its transcript lands under the canonical
    checkout's ``logs/`` — never the scratch worktree that happened to launch
    the run. Pins the worktree→canonical roof end-to-end against a real linked
    worktree (the ``.worktrees/<name>`` layout that leaked logs before).
    """
    canonical = tmp_path / "canonical"
    (canonical / "config").mkdir(parents=True)
    (canonical / "config" / "paths.yaml").write_text("{}\n", encoding="utf-8")
    (canonical / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    _git(canonical, "init", "-q")
    _git(canonical, "config", "user.email", "t@example.com")
    _git(canonical, "config", "user.name", "t")
    _git(canonical, "add", "-A")
    _git(canonical, "commit", "-qm", "init")
    worktree = canonical / ".worktrees" / "run"
    _git(canonical, "worktree", "add", "-q", str(worktree))

    monkeypatch.delenv(ProjectRoot.ENV_VAR, raising=False)
    monkeypatch.setattr(log_capture, "TRAIN_ROOT", worktree)

    path = log_capture.lane_log_at("logs/colab", "lane.log")
    assert path == (canonical / "logs" / "colab" / "lane.log").resolve()
    assert not (worktree / "logs").exists(), "the scratch worktree keeps no roof"
