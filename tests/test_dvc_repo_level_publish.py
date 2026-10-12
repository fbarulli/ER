"""Checkpoint publication never hosts a second DVC project under an output root.

The 2026-10-12 lane transport died because the checkpoint-publish path passed
the GLOBAL results root as the DVC source: ``Repo.init`` created
``results/.dvc``, DVC's sub-repo ignore then shadowed the whole results root,
and every repo-level ``dvc`` command on a ``results/**`` target failed
"is dvc-ignored" (rc=1). These tests pin the repo-level routing: a results
root is refused fail-loud with nothing created, publication goes through the
ONE repository-level project, and the outer repo can still ``dvc add`` and
collect a ``results/**`` pointer after the publish step.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from core import common
from training import dvc_store

# Reuse the canonical scripted dvc (and its config-patch helper) from the
# streaming tests: the publish step's ``dvc add``/``dvc push`` must be driven
# without touching the real remote, while the outer-repo assertions below run
# the REAL dvc module (``python -m dvc``), which ignores PATH.
from test_dvc_streaming_publish import FAKE_DVC, _debounce


def _make_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tmp repository-level DVC project with a results output root."""
    from dvc.repo import Repo

    root = tmp_path / "repo"
    (root / "results").mkdir(parents=True)
    Repo.init(str(root), no_scm=True).close()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "dvc"
    fake.write_text(FAKE_DVC)
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_DVC_LOG", str(tmp_path / "dvc-calls.jsonl"))
    monkeypatch.setenv("DVC_API_KEY", "test-token")
    # Durable resume pointers stay inside the test repo, never the real root.
    monkeypatch.setattr(
        dvc_store.common,
        "artifact",
        lambda key, fields=None: root / ".resume" / f"{fields['name']}.dvc",
    )
    return root


def _snapshot(root: Path, step: int = 10) -> Path:
    snapshot = root / "results" / "_checkpoint_upload_staging" / "trial" / f"checkpoint-{step}"
    snapshot.mkdir(parents=True)
    (snapshot / "trainer_state.json").write_text("{}")
    return snapshot


def test_publishing_checkpoints_never_creates_a_dvc_project_under_an_output_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _make_repo(tmp_path, monkeypatch)
    results = root / "results"
    snapshot = _snapshot(root)
    # The shape that killed the transport: the results ROOT as the DVC source.
    # It must be refused fail-loud, and nothing may be created under it.
    with pytest.raises(FileNotFoundError, match="repository-level DVC workspace"):
        dvc_store.publish_checkpoints(results, [(snapshot, "checkpoint-10", snapshot)])
    assert not (results / ".dvc").is_dir()
    # The repo-level project publishes the same checkpoint...
    with _debounce(0.3):
        dvc_store.publish_checkpoints(root, [(snapshot, "checkpoint-10", snapshot)])
    assert (root / ".dvc").is_dir()
    # ...without ever nesting a second project under an output root (here,
    # and — proved against the real tree — under any declared root).
    assert not list(results.rglob(".dvc"))
    for output_root in common.dvc_output_roots():
        assert not (output_root / ".dvc").is_dir()


def test_repo_level_add_and_collect_on_a_results_path_succeeds_after_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _make_repo(tmp_path, monkeypatch)
    snapshot = _snapshot(root)
    with _debounce(0.3):
        dvc_store.publish_checkpoints(root, [(snapshot, "checkpoint-10", snapshot)])
    payload = root / "results" / "run.tar.zst"
    payload.write_bytes(b"result bytes")
    added = subprocess.run(
        [sys.executable, "-m", "dvc", "add", "results/run.tar.zst"],
        cwd=root, capture_output=True, text=True,
    )
    assert added.returncode == 0, added.stdout + added.stderr
    assert (root / "results" / "run.tar.zst.dvc").is_file()
    # ``dvc status`` collects the pointer through the same target-resolution
    # path that ``dvc push``/``dvc status --cloud`` use: a shadowed results
    # root makes it fail with "is dvc-ignored" (rc=1), exactly the lane crash.
    status = subprocess.run(
        [sys.executable, "-m", "dvc", "status", "results/run.tar.zst.dvc"],
        cwd=root, capture_output=True, text=True,
    )
    combined = status.stdout + status.stderr
    assert status.returncode == 0, combined
    assert "dvc-ignored" not in combined


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
