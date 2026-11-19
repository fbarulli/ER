"""Public-behavior pin for the local HPO study open.

The original Postgres flow opened the study in the controller before spawning
workers (the stale-trial reap), so by launch time the RDB schema existed. The
study-storage refactor added a local SQLite backend but dropped that controller
pre-open, so every parallel worker ran ``RDBStorage`` -> ``CREATE TABLE studies``
concurrently and one died with ``sqlite3.OperationalError: table studies already
exists``. This pins that the controller creates the local study ONCE before any
worker process is spawned.
"""
from __future__ import annotations

from pathlib import Path

from cli import laya_hpo

from test_laya_hpo import _exec_kernel, _stage


def test_controller_creates_the_local_study_before_spawning_workers(
        monkeypatch, tmp_path):
    """No OPTUNA_STORAGE_URL: the shared SQLite study exists on disk before the
    controller spawns any worker, so parallel workers opening it with
    load_if_exists=True never race CREATE TABLE studies."""
    receipt = _stage(monkeypatch, tmp_path, None)
    namespace = _exec_kernel(
        monkeypatch, Path(receipt["staged"]) / laya_hpo.HPO_CODE_FILE, tmp_path)
    namespace["pip_install_runtime"] = lambda: None
    namespace["pip_install_laya"] = lambda: None
    import subprocess as _subprocess
    observed = {"spawned": 0, "study_existed": []}

    class _FakeProcess:
        def wait(self):
            return 0

    class _FakeSubprocess:
        def __getattr__(self, name):
            return getattr(_subprocess, name)

        def Popen(self, _argv, **_kwargs):
            observed["spawned"] += 1
            observed["study_existed"].append(
                Path(namespace["local_study_path"]()).is_file())
            return _FakeProcess()

    namespace["subprocess"] = _FakeSubprocess()
    # The post-worker receipt/archive surfaces are not under test here.
    namespace["write_session_receipt"] = lambda: {}

    class _FakeArchive:
        def __init__(self, options):
            pass

        def run(self, receipt):
            return None

    namespace["SessionArchive"] = _FakeArchive
    namespace["SessionOrchestrator"]().run()
    assert observed["spawned"] >= 1
    assert all(observed["study_existed"])
