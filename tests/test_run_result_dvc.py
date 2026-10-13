"""Run-result DVC tracking: the git/DVC split and the fail-loud boundary."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _git_ignored(relative: str) -> bool:
    return subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", relative],
        cwd=REPO_ROOT,
    ).returncode == 0


def test_committed_results_are_pointers_never_payloads():
    """A run payload is git-ignored; the *.dvc pointer beside it is committed.

    This is the git-vs-DVC split the owner asked for: payload bytes live in the
    dagshub remote, and only the pointer DVC writes may enter git.
    """
    assert _git_ignored("results/model_tracks/inputs/example.tar.zst")
    assert _git_ignored("results/model_tracks/example.training.tar.zst")
    assert _git_ignored("results/embedding_job/inputs/embeddings-example.tar.zst")
    assert not _git_ignored("results/model_tracks/inputs/example.tar.zst.dvc")


def test_publishing_a_result_requires_dvc_credentials(tmp_path, monkeypatch):
    """A publish without ``.env``'s DVC_API_KEY fails loud before any I/O."""
    from core import common
    from model_tracks import run_retention

    monkeypatch.setattr(common, "TRAIN_ROOT", tmp_path)
    monkeypatch.delenv("DVC_API_KEY", raising=False)
    payload = tmp_path / "results" / "run.tar.zst"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"result bytes")

    with pytest.raises(RuntimeError, match="DVC_API_KEY"):
        run_retention.publish_result_paths([payload])
