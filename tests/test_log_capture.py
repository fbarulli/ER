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
from pydantic import ValidationError

from cli import log_capture
from core.common import training_cfg
from core.project_root import ProjectRoot
from core.schemas import KaggleSpec


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


def test_lane_log_is_checkout_independent(tmp_path, monkeypatch):
    """A linked worktree writes the shared log to the CANONICAL logs root.

    The root name is the SSOT (``paths.logs_dir``); a run launched from a
    per-checkout worktree must resolve to the main worktree's roof, not a
    second ``logs/`` beside that worktree.
    """
    main_root = tmp_path / "main"
    (main_root / "logs").mkdir(parents=True)
    monkeypatch.setattr(log_capture, "TRAIN_ROOT", tmp_path / "worktrees" / "wt")
    monkeypatch.setattr(log_capture, "canonical_root", lambda base: main_root.resolve())
    assert log_capture.lane_log("colab", "lane.log") == (
        main_root / log_capture.data_cfg().paths.logs_dir / "colab" / "lane.log").resolve()


@pytest.mark.parametrize("text", ["", "plain line\n", "no frames here \n"])
def test_progress_frames_without_cr_pass_through_unchanged(text):
    assert log_capture.progress_frames_to_lines(text) == text


@pytest.mark.parametrize("bad_name", ["cpu/lane.log", "logs/kaggle", "lane.log/", "x\\y.log"])
def test_lane_log_file_names_must_be_bare(tmp_path, monkeypatch, bad_name):
    """A separator in the file name would fan out <logs_root>/<lane>/ — refused.

    Both accessors (``lane_log`` and ``lane_log_at``) enforce the same rule, so
    a declared name can never smuggle a per-device cpu/gpu subdir.
    """
    monkeypatch.setattr(log_capture, "TRAIN_ROOT", tmp_path)
    with pytest.raises(ValueError, match="bare file name"):
        log_capture.lane_log("colab", bad_name)
    with pytest.raises(ValueError, match="bare file name"):
        log_capture.lane_log_at("logs/colab", bad_name)


def test_kaggle_logs_dir_must_be_a_bare_lane_name():
    payload = training_cfg().kaggle.model_dump()
    payload["logs_dir"] = "logs/kaggle"
    with pytest.raises(ValidationError, match="bare lane directory"):
        KaggleSpec.model_validate(payload)


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
