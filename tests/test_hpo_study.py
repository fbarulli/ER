"""Public-behavior pins for the HPO study-storage owner.

One focused test per selection rule: remote URL wins (redacted), local SQLite
fallback, fail-loud when neither is configured, and a cwd-independent local
study shared by two worker processes. No internals are tested.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from core.credentials import CredentialsSpec, CredentialStore, StudySpec
from training.hpo_study import (
    StudyBackend,
    StudyOwner,
    StudyResolutionError,
)

SECRET = "postgresql://hpo_user:s3cret@db.example.com:5432/optuna"
LOCAL_FRAGMENT = "results/laya_lane/hpo_study.db"


def _owner(tmp_path: Path, *, env_value: str | None,
           local_file: str | None = LOCAL_FRAGMENT) -> StudyOwner:
    spec = CredentialsSpec(study=StudySpec(local_file=local_file))
    values = ({spec.keys.optuna_storage_url: env_value}
              if env_value is not None else {})
    store = CredentialStore.from_mapping(spec, values)
    return StudyOwner.from_config(credentials=spec, credential_store=store,
                                  root=tmp_path)


def test_remote_url_wins_and_is_returned_redacted(tmp_path):
    owner = _owner(tmp_path, env_value=SECRET)
    resolved = owner.resolve(generation_id="gen", model_key="laya")
    assert resolved.backend is StudyBackend.REMOTE
    assert resolved.storage_url() == SECRET
    assert resolved.redacted_url != SECRET
    assert "s3cret" not in resolved.redacted_url
    assert "db.example.com" in resolved.redacted_url
    assert "s3cret" not in repr(resolved)


def test_local_sqlite_is_used_without_a_remote_url(tmp_path):
    owner = _owner(tmp_path, env_value=None)
    resolved = owner.resolve(generation_id="gen", model_key="laya")
    expected = "sqlite:///" + str(tmp_path / LOCAL_FRAGMENT)
    assert resolved.backend is StudyBackend.LOCAL
    assert resolved.storage_url() == expected
    assert Path(resolved.storage_url()[len("sqlite:///"):]).is_absolute()


def test_neither_source_fails_loud_with_a_traceback(tmp_path, caplog):
    owner = _owner(tmp_path, env_value=None, local_file=None)
    caplog.set_level(logging.ERROR, logger="training.hpo_study")
    with pytest.raises(StudyResolutionError):
        owner.resolve(generation_id="gen", model_key="laya")
    assert "Traceback" in caplog.text


def test_two_worker_processes_resolve_the_same_study(tmp_path):
    """The local study path is TRAIN_ROOT-relative, never cwd-relative."""
    driver = (
        "import sys\n"
        "from core.credentials import CredentialsSpec, CredentialStore, StudySpec\n"
        "from training.hpo_study import StudyOwner\n"
        "spec = CredentialsSpec("
        "study=StudySpec(local_file='results/laya_lane/hpo_study.db'))\n"
        "store = CredentialStore.from_mapping(spec, {})\n"
        "owner = StudyOwner.from_config(credentials=spec, "
        "credential_store=store, root=sys.argv[1])\n"
        "print(owner.local_url())\n"
    )
    worktree_src = str(Path(__file__).resolve().parents[1] / "src")
    env = {**os.environ, "PYTHONPATH": worktree_src}
    outputs = []
    for cwd in (tmp_path, "/tmp"):
        result = subprocess.run(
            [sys.executable, "-c", driver, str(tmp_path)],
            cwd=cwd, env=env, capture_output=True, text=True, check=True)
        outputs.append(result.stdout.strip())
    assert outputs[0] == outputs[1]
    assert outputs[0].endswith(LOCAL_FRAGMENT)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
