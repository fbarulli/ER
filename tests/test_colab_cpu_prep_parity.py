"""CPU-prep parity contracts (--what bundle vs the standalone bundle lane).

Three capabilities the bundle lane (src/cli/colab_bundle.py, commits
d264af1/9e7f0e4/902689e/99c7ce8) already delivers must hold for the CPU prep
lane launched through colab_backend.py -> cli.colab main --what bundle:

* high-RAM provisioning — a fresh CPU allocation requests the machine shape
  only when config/training.yaml colab.high_mem says so; an owner-launched
  named session is re-verified and never reallocated;
* logging via streaming — the prep cell streams through
  run_colab_exec_stream (root system transcript + local training.log
  mirror), and [done] prints only on clean completion ([failed] otherwise);
* tqdm passthrough — the emitted prepare script never captures the child's
  stderr (fd inheritance is how the bars stream live; 902689e).

No Colab transport is contacted: every test stages an intentional stream or
command fake, the same shape the bundle-lane tests use.
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

from cli import colab


def _fake_colab(received: list[list]):
    """One shared fake for the `colab` CLI surface; records every argv."""

    def fake(*args, **kwargs):
        received.append([*args, "timeout=" + repr(kwargs.get("timeout"))])
        response = mock.Mock()
        if args[0] == "sessions":
            response.returncode = 1
            response.stdout = ""
            response.stderr = ""
        return response

    return fake


def _provision(monkeypatch, *, received: list, high_mem: bool, gpu: str = "CPU"):
    monkeypatch.setattr(colab, "SESSION", "test-prep-vm")
    monkeypatch.setattr(
        colab, "_COLAB", colab._COLAB.model_copy(update={"high_mem": high_mem})
    )
    monkeypatch.setattr(colab, "colab", _fake_colab(received))
    # The control-channel handshake is simulated deliberately: what is under
    # test is the allocation argv, not the kernel probe.
    monkeypatch.setattr(colab, "_verify_session_handshake", lambda *a, **k: None)
    monkeypatch.setattr(colab, "GPU", gpu)


def test_cpu_provisioning_defaults_to_the_standard_shape(monkeypatch):
    """No colab.high_mem in config -> the emitted `colab new` argv is the
    byte-identical pre-parity command (CPU, no shape flag)."""
    received: list = []
    _provision(monkeypatch, received=received, high_mem=False)
    colab.ensure_session()
    new_calls = [call for call in received if call[0] == "new"]
    assert new_calls == [["new", "-s", "test-prep-vm", "timeout=300"]]


def test_config_gated_cpu_provision_requests_the_high_ram_shape(monkeypatch):
    received: list = []
    _provision(monkeypatch, received=received, high_mem=True)
    colab.ensure_session()
    new_calls = [call for call in received if call[0] == "new"]
    assert new_calls == [["new", "-s", "test-prep-vm", "--high-mem", "timeout=300"]]


def test_high_ram_remains_cpu_only(monkeypatch):
    """A GPU accelerator is never reshaped by colab.high_mem."""
    received: list = []
    _provision(monkeypatch, received=received, high_mem=True, gpu="T4")
    colab.ensure_session()
    new_calls = [call for call in received if call[0] == "new"]
    assert new_calls == [["new", "-s", "test-prep-vm", "--gpu", "T4", "timeout=300"]]


def test_owner_allocated_session_is_never_reallocated(monkeypatch):
    """`colab sessions` listing the named VM pins the reuse contract: the
    launcher verifies the handshake and allocates nothing."""
    received: list = []

    def fake(*args, **kwargs):
        received.append(list(args))
        response = mock.Mock()
        if args[0] == "sessions":
            response.returncode = 0
            response.stdout = "my-session   running (High-RAM)"
        return response

    monkeypatch.setattr(colab, "SESSION", "my-session")
    monkeypatch.setattr(colab, "colab", fake)
    handshakes = mock.Mock()
    monkeypatch.setattr(colab, "_verify_session_handshake", handshakes)
    colab.ensure_session()
    assert [call[0] for call in received] == ["sessions"]
    handshakes.assert_called_once()


class _StreamProcess:
    """Intentional stream fake: pre-recorded fd content, rc per lane."""

    def __init__(self, stdout: str, stderr: str, returncode: int = 0):
        self.stdin = _NullStream()
        self.stdout = io.StringIO(stdout)
        self.stderr = io.StringIO(stderr)
        self.returncode = returncode
        self.wait = lambda timeout=None: returncode


class _NullStream:
    def write(self, text): return len(text)

    def flush(self): return None

    def close(self): return None


def test_tqdm_bars_stream_with_carriage_returns(monkeypatch, capsys):
    """tqdm emits \r-separated progress; stream_output must flush each unit
    instead of swallowing it behind a newline (902689e passthrough)."""
    bars = "10%|█| 1/10\r40%|██| 4/10\r100%|████| 10/10\r\n"
    process = _StreamProcess("stage start\n", bars)
    monkeypatch.setattr(colab.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(colab, "_PROBE_RETRIES", 1)
    colab.run_colab_exec_stream("test-prep-vm", "print('prep')", timeout=5)
    terminal = capsys.readouterr().out
    assert "10%|█| 1/10" in terminal
    assert "40%|██| 4/10" in terminal
    assert "100%|████| 10/10" in terminal


def test_run_bundle_streams_into_both_transcripts_without_capturing_stderr(
    monkeypatch, tmp_path
):
    """The CPU prep lane runs its prepare subprocess with inherited streams
    (fd passthrough) and labels the stage so both transcripts carry it."""
    dataset = tmp_path / "dataset.csv"
    dataset.write_text("sku_id\na\n", encoding="utf-8")
    uploads: list = []
    downloads: list = []
    monkeypatch.setattr(
        colab, "_upload_with_retries",
        lambda source, remote, *, timeout: uploads.append((source, remote)),
    )
    seen: dict = {}

    def fake_stream(session, script, *, timeout, log_name, retry_safe=False,
                    exclude_from_live_log=False, training_output=False):
        seen["script"] = script
        seen["kwargs"] = {
            "timeout": timeout, "log_name": log_name,
            "training_output": training_output,
        }

    monkeypatch.setattr(colab, "run_colab_exec_stream", fake_stream)
    monkeypatch.setattr(
        colab, "_download_file_with_visibility",
        lambda **kwargs: downloads.append(kwargs),
    )
    monkeypatch.setattr(colab, "_result_event", lambda *a, **k: None)
    colab.run_bundle(dataset)
    assert uploads and uploads[0][1] == f"{colab.REMOTE_ROOT}/dataset.csv"
    assert seen["kwargs"]["log_name"] == "bundle"
    # BOTH transcripts: the root system log rides stream printing, and
    # training_output forwards the same chunks into training.log.
    assert seen["kwargs"]["training_output"] is True
    script = seen["script"]
    assert "training.prepare_all" in script
    # tqdm passthrough: the child's streams are inherited, never captured.
    assert "stderr=" not in script
    assert downloads and downloads[0]["remote"] == (
        f"{colab.REMOTE_ROOT}/bundle_delivery.tar.gz"
    )


def _run_bundle_main(argv: list[str], bundle) -> list[str]:
    """Drive cli.colab main() for the bundle lane with every surface faked."""
    order: list[str] = []

    def record(label, result=None):
        def side_effect(*_a, **_k):
            order.append(label)
            return result
        return side_effect

    stack = [
        mock.patch.object(sys, "argv", ["colab.py", *argv]),
        mock.patch.object(colab, "start_live_log"),
        mock.patch.object(colab, "close_live_log"),
        mock.patch.object(colab, "_legacy_validation_sources", return_value={}),
        mock.patch.object(colab, "_validate_legacy_bundle_partitions"),
        mock.patch.object(colab, "check_colab_cli"),
        mock.patch.object(colab, "acquire_colab_launch_lock", return_value=None),
        mock.patch.object(colab, "release_colab_launch_lock"),
        mock.patch.object(colab, "ensure_session", record("session")),
        mock.patch.object(colab, "prepare_remote_layout"),
        mock.patch.object(colab, "install_deps"),
        mock.patch.object(colab, "verify_training_inputs"),
        mock.patch.object(colab, "verify_remote_models"),
        mock.patch.object(colab, "log_gpu_profile"),
        mock.patch.object(colab, "start_local_bundle_prewarm"),
        mock.patch.object(colab, "start_validation_upload_prewarm", return_value="stamp"),
        mock.patch.object(colab, "drain_local_bundle_prewarm"),
        mock.patch.object(colab, "drain_validation_upload_prewarm"),
        mock.patch.object(colab, "run_bundle", bundle),
        mock.patch.object(colab, "stop", record("stop")),
        mock.patch.object(colab, "stop_keep_alive_daemon", record("stop_daemon", 1)),
    ]
    saved_gpu = colab.GPU
    saved_env = os.environ.get("EUROMONITOR_KEEP_ALIVE_ALLOWED")
    for patcher in stack:
        patcher.start()
    try:
        colab.main()
    finally:
        colab.GPU = saved_gpu
        if saved_env is None:
            os.environ.pop("EUROMONITOR_KEEP_ALIVE_ALLOWED", None)
        else:
            os.environ["EUROMONITOR_KEEP_ALIVE_ALLOWED"] = saved_env
        for patcher in reversed(stack):
            patcher.stop()
    return order


def test_cpu_prep_lane_announces_done_only_on_clean_completion(
    monkeypatch, tmp_path, capsys
):
    dataset = tmp_path / "cohort.csv"
    dataset.write_text("sku_id\na\n", encoding="utf-8")
    _run_bundle_main(
        ["--what", "bundle", "--gpu", "CPU", "--dataset-csv", str(dataset)],
        mock.Mock(name="run_bundle"),
    )
    captured = capsys.readouterr()
    assert "[done] artifacts downloaded locally" in captured.out
    assert "[failed]" not in captured.out


def test_cpu_prep_lane_reports_failed_when_the_run_raises(
    monkeypatch, tmp_path, capsys
):
    dataset = tmp_path / "cohort.csv"
    dataset.write_text("sku_id\na\n", encoding="utf-8")
    failing = mock.Mock(name="run_bundle", side_effect=RuntimeError("VM died"))
    with pytest.raises(RuntimeError, match="VM died"):
        _run_bundle_main(
            ["--what", "bundle", "--gpu", "CPU", "--dataset-csv", str(dataset)],
            failing,
        )
    captured = capsys.readouterr()
    assert "[failed] launch did not complete successfully" in captured.out
    assert "[done]" not in captured.out
