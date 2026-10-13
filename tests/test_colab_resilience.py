"""Public-behaviour pins for Colab control-channel recovery and VM release.

No Colab transport is contacted: the transport surfaces are faked with the same
module-object patch pattern the other colab-lane tests use.
"""
from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import pytest

import cli.colab as colab
import cli.colab_reconnect as colab_reconnect
import cli.colab_release as colab_release


def _completed(stdout: str = "", returncode: int = 0, stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_detached_stage_re_attaches_and_resumes_after_a_transient_loss(monkeypatch):
    """A `connection was lost` control failure re-attaches (bounded backoff) and
    the detached stage keeps polling instead of aborting the whole run."""
    calls: list[str] = []

    def fake_capture(session, script, timeout, **kwargs):
        calls.append(script)
        if len(calls) == 1:
            raise RuntimeError("Connection was lost.")
        if len(calls) == 2:
            return json.dumps({"pid": 4321, "log": "/remote/train.log",
                               "status": "/remote/train.log.status"})
        return json.dumps({"offset": 0, "chunk": "", "done": True, "returncode": "0"})

    monkeypatch.setattr(colab, "run_colab_exec_capture", fake_capture)
    monkeypatch.setattr(colab_reconnect.time, "sleep", lambda _delay: None)

    colab.run_detached_stage("train", ["python3", "-c", "train()"], timeout=60)

    assert len(calls) == 3, "the transient loss must be re-attached and the poll resumed"


def test_terminal_session_loss_fails_without_re_attaching(monkeypatch):
    """A vanished session is terminal at once: no re-attach, one control call."""
    calls: list[str] = []

    def fake_capture(session, script, timeout, **kwargs):
        calls.append(script)
        raise RuntimeError(f"[colab] Session '{colab.SESSION}' not found.")

    monkeypatch.setattr(colab, "run_colab_exec_capture", fake_capture)
    monkeypatch.setattr(colab_reconnect.time, "sleep", lambda _delay: None)

    with pytest.raises(RuntimeError, match="lost its control connection"):
        colab.run_detached_stage("train", ["python3", "-c", "train()"], timeout=60)

    assert len(calls) == 1, "a lost session must not be retried"


def test_stop_uses_the_own_release_path_when_the_cli_stop_leaves_the_vm_listed(monkeypatch, tmp_path):
    """`colab stop` failing (or leaving the session listed) cannot keep the VM:
    the launcher's own stop-by-name path releases it and teardown is confirmed."""
    listings = iter([
        f"[{colab.SESSION}] endpoint | Hardware: CPU\n",  # before release: still live
        "",                                              # after own release: gone
    ])
    cli_stop_calls: list[tuple] = []
    own_release_commands: list[list[str]] = []

    def fake_colab(*args, **kwargs):
        cli_stop_calls.append(args)
        if args[0] == "stop":
            return _completed(returncode=1)
        return _completed(stdout=next(listings))

    def fake_run(command, **kwargs):
        own_release_commands.append(list(command))
        return _completed(stdout=json.dumps({"released": True, "action": "unassigned"}))

    monkeypatch.setattr(colab, "colab", fake_colab)
    monkeypatch.setattr(colab, "_COLAB_CLI_STATE_DIR", tmp_path)
    monkeypatch.setattr(colab_release.subprocess, "run", fake_run)

    assert colab.stop() is True
    assert ("stop", "-s", colab.SESSION) in cli_stop_calls
    assert own_release_commands, "the own stop-by-name path must run"
    assert "release-session" in own_release_commands[0]
    assert "-s" in own_release_commands[0] and colab.SESSION in own_release_commands[0]


def test_hardened_entrypoint_stop_closes_what_exists_when_the_manager_is_gone():
    """The reported `_kernel_client._manager is None` AttributeError must not skip
    the close: the entrypoint's stop never raises and still runs a close path."""
    entry = importlib.import_module("cli.colab_cli_entry")
    closed: list[str] = []

    class FakeRuntime:
        def __init__(self) -> None:
            self._kernel_client = None

    class FakeModule:
        ColabRuntime = FakeRuntime

    entry._harden_runtime_stop(FakeModule)

    class OrphanedClient:
        _manager = None

        def stop(self, **kwargs) -> None:
            closed.append("client-stop")

    runtime = FakeRuntime()
    runtime._kernel_client = OrphanedClient()
    FakeRuntime.stop(runtime)  # must not raise

    assert closed == ["client-stop"]


def test_all_tracks_remote_lines_are_emitted_exactly_once(monkeypatch, capsys):
    """One remote [track/...] line crosses the launcher once: the tee is the
    only transcript writer, so no second prefix and no second write."""
    lines = ("[track/gnn_only] listing_contract: 1/2\n"
             "[track/text] listing_contract: 2/2\n")
    calls: list[str] = []

    def fake_capture(session, script, timeout, **kwargs):
        calls.append(script)
        if len(calls) == 1:
            return json.dumps({"pid": 1, "log": "/r/stage.log", "status": "/r/stage.log.status"})
        return json.dumps({"offset": len(lines), "chunk": lines,
                           "done": True, "returncode": "0"})

    transcript_writes: list[str] = []
    monkeypatch.setattr(colab, "run_colab_exec_capture", fake_capture)
    monkeypatch.setattr(colab, "_write_training_log", transcript_writes.append)

    colab.run_detached_stage("all_tracks", ["python3", "-c", "suite()"], timeout=60)

    stdout = capsys.readouterr().out
    for line in lines.splitlines():
        emissions = stdout.count(line) + sum(write.count(line) for write in transcript_writes)
        assert emissions == 1, f"{line!r} crossed the launcher {emissions} times"
    assert "[all_tracks] [track/" not in stdout


def test_tolerated_probe_failure_lands_in_the_transcript_once(monkeypatch, capsys):
    """A tolerated all_tracks probe failure is one transcript line: the tee
    owns the write, so no second direct _write_training_log copy."""

    def fake_capture(session, script, timeout, **kwargs):
        fake_capture.calls += 1
        if fake_capture.calls == 1:
            return json.dumps({"pid": 1, "log": "/r/stage.log", "status": "/r/stage.log.status"})
        if fake_capture.calls == 2:
            raise RuntimeError("probe read failed")
        return json.dumps({"offset": 0, "chunk": "", "done": True, "returncode": "0"})

    fake_capture.calls = 0
    transcript_writes: list[str] = []
    monkeypatch.setattr(colab, "run_colab_exec_capture", fake_capture)
    monkeypatch.setattr(colab, "_write_training_log", transcript_writes.append)
    monkeypatch.setattr(colab_reconnect.time, "sleep", lambda _delay: None)

    colab.run_detached_stage("all_tracks", ["python3", "-c", "suite()"], timeout=60)

    stdout = capsys.readouterr().out
    assert stdout.count("log/status unavailable") == 1
    assert "log/status unavailable" not in "".join(transcript_writes)
