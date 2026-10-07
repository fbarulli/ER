"""Two pins for the baked-in colab self-watch (release + delivery default).

No Colab transport is contacted: the surfaces are faked with the same
module-object patch pattern the other colab-lane tests use.
"""
from __future__ import annotations

import sys
from unittest import mock

import pytest

import cli.colab as colab
import cli.log_capture as log_capture


def test_main_remote_execute_spawns_the_detached_self_watch(monkeypatch, tmp_path):
    """A pass-the-gates remote --execute run carries its own setsid watcher:
    `--what self-watch` argv, start_new_session, log under the colab log roof
    (the canonical logs root), spawned off the baked-in main() path itself."""
    spawns: list[dict] = []
    tmp_path.joinpath("state").mkdir()

    def fake_popen(command, **kwargs):
        spawns.append({"command": list(command), **kwargs})
        return mock.Mock()

    monkeypatch.setattr(colab.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(log_capture, "TRAIN_ROOT", tmp_path)
    monkeypatch.setattr(colab, "TRAIN_ROOT", tmp_path)
    monkeypatch.setattr(colab, "_COLAB_CLI_STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(colab, "TRAINING_RESULTS", tmp_path)
    monkeypatch.setattr(sys, "argv", ["colab.py", "--what", "sims", "--gpu", "CPU"])
    monkeypatch.setattr(colab, "start_live_log", mock.Mock())
    monkeypatch.setattr(colab, "close_live_log", mock.Mock())
    monkeypatch.setattr(colab, "check_colab_cli", mock.Mock())
    monkeypatch.setattr(colab, "acquire_colab_launch_lock",
                        mock.Mock(return_value=None))
    monkeypatch.setattr(colab, "release_colab_launch_lock", mock.Mock())
    monkeypatch.setattr(colab, "ensure_session", mock.Mock())
    monkeypatch.setattr(colab, "prepare_remote_layout", mock.Mock())
    monkeypatch.setattr(colab, "install_deps", mock.Mock())
    monkeypatch.setattr(colab, "verify_remote_models", mock.Mock())
    monkeypatch.setattr(colab, "log_gpu_profile", mock.Mock())
    monkeypatch.setattr(colab, "verify_training_inputs", mock.Mock())
    monkeypatch.setattr(colab, "run_sims", mock.Mock())
    monkeypatch.setattr(colab, "drain_local_bundle_prewarm", mock.Mock())
    monkeypatch.setattr(colab, "drain_validation_upload_prewarm", mock.Mock())
    monkeypatch.setattr(colab, "stop", mock.Mock(return_value=True))
    colab.main()
    assert len(spawns) == 1
    spawn = spawns[0]
    assert "--what" in spawn["command"] and "self-watch" in spawn["command"]
    assert "--self-watch-run" in spawn["command"]
    assert spawn["start_new_session"] is True
    assert spawn["stderr"] == colab.subprocess.STDOUT
    log_roof = tmp_path / "logs" / "colab"
    spawned_logs = list(log_roof.glob("self_watch_*.log"))
    assert len(spawned_logs) == 1
    assert "launching watcher for what=sims" in spawned_logs[0].read_text()


def test_plan_paths_and_watchers_never_spin_the_session_surface(monkeypatch, tmp_path):
    """--preflight-only (a plan run) never reaches the spawn point, and a
    self_watch(execute=False) polls the session surface zero times, releases
    nothing, and writes no receipt; only executed remote runs spawn."""
    spawns: list = []
    monkeypatch.setattr(colab, "spawn_self_watch", mock.Mock(
        side_effect=lambda **kwargs: spawns.append(kwargs)))
    monkeypatch.setattr(
        sys, "argv",
        ["colab.py", "--what", "hpo", "--preflight-only", "--gpu", "CPU"],
    )
    with pytest.raises(ValueError, match="--preflight-only supports train/smoke lanes"):
        colab.main()
    assert not spawns
    cli_calls: list = []
    stops: list = []
    monkeypatch.setattr(colab, "colab", mock.Mock(side_effect=lambda *a: cli_calls.append(a)))
    monkeypatch.setattr(colab, "stop", mock.Mock(
        side_effect=lambda **kwargs: stops.append(kwargs)))
    monkeypatch.setattr(colab, "TRAINING_RESULTS", tmp_path)
    plan = colab.self_watch(what="sims", run_id="plan_only", execute=False)
    assert plan["mode"] == "dry-run"
    assert not cli_calls and not stops
    assert not (tmp_path / "self_watch_plan_only").exists()
