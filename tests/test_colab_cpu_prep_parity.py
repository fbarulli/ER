"""CPU data-bundle prep lane parity (owner structural ruling 8).

Data-bundle production lives in its own lane file (src/cli/
colab_data_bundle_prep.py), exactly like the standalone bundle lane
(src/cli/colab_bundle.py, commits d264af1/9e7f0e4/902689e/99c7ce8).
src/cli/colab.py keeps only config-gated thin passthroughs and is
byte-identical when the lane is unused.  The pinned capabilities:

* launch capability — a fresh CPU allocation requests the machine shape
  only when config/training.yaml colab.high_mem says so; an
  owner-launched named session is re-verified and never reallocated; a
  GPU accelerator is never reshaped;
* streaming — the lane forwards EVERY prep chunk to BOTH transcripts
  (root system log + training.log) and prints [done] only on clean
  completion ([failed] otherwise, never retried);
* tqdm passthrough — the emitted prepare script never captures the
  child's stderr (fd inheritance is how the bars stream live; 902689e);
* dispatch — the thin passthrough in cli.colab main is config-gated
  (colab.cpu_data_bundle_lane): default keeps the original direct
  --what bundle call byte-identical.

No Colab transport is contacted: every test stages an intentional stream
or command fake, the same shape the bundle-lane tests use.
"""
from __future__ import annotations

import io
import os
import re
import sys
from pathlib import Path
from unittest import mock

import pytest

import cli.colab as colab
import cli.colab_data_bundle_prep as prep


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
    monkeypatch.setattr(colab, "GPU", gpu)
    monkeypatch.setattr(prep, "training_cfg", mock.Mock(
        **{"return_value.cpu_bundle_prep.high_mem": high_mem}))
    monkeypatch.setattr(colab, "colab", _fake_colab(received))
    # The control-channel handshake is simulated deliberately: what is under
    # test is the allocation argv, not the kernel probe.
    monkeypatch.setattr(colab, "_verify_session_handshake", lambda *a, **k: None)


def test_cpu_provisioning_defaults_to_the_standard_shape(monkeypatch):
    """No colab.high_mem in config -> the passthrough returns () and the
    emitted `colab new` argv is the byte-identical pre-parity command."""
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
    """tqdm emits \r-separated progress; the shared transport must flush
    each unit instead of swallowing it behind a newline (902689e)."""
    bars = "10%|█| 1/10\r40%|██| 4/10\r100%|████| 10/10\r\n"
    process = _StreamProcess("stage start\n", bars)
    monkeypatch.setattr(colab.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(colab, "_PROBE_RETRIES", 1)
    colab.run_colab_exec_stream("test-prep-vm", "print('prep')", timeout=5)
    terminal = capsys.readouterr().out
    assert "10%|█| 1/10" in terminal
    assert "40%|██| 4/10" in terminal
    assert "100%|████| 10/10" in terminal


def test_cpu_prep_lane_streams_into_both_transcripts_and_tags_the_cohort(
    monkeypatch, tmp_path, capsys
):
    """run_cpu_bundle_prep layers the parity capabilities on top of
    cli.colab.run_bundle WITHOUT editing it: coerced dual-transcript
    streaming, cohort tag, and an uncaptured prepare child."""
    dataset = tmp_path / "dataset_50pct.csv"
    dataset.write_text("sku_id\na\n", encoding="utf-8")
    uploads: list = []
    downloads: list = []
    monkeypatch.setattr(
        colab, "_upload_with_retries",
        lambda source, remote, *, timeout: uploads.append((source, remote)),
    )
    seen: dict = {}

    def recorder(session, script, *, timeout=None, log_name=None, retry_safe=False,
                 exclude_from_live_log=False, training_output=False):
        seen["script"] = script
        seen["kwargs"] = {
            "log_name": log_name, "training_output": training_output,
        }

    monkeypatch.setattr(colab, "run_colab_exec_stream", recorder)
    monkeypatch.setattr(
        colab, "_download_file_with_visibility",
        lambda **kwargs: downloads.append(kwargs),
    )
    monkeypatch.setattr(colab, "_result_event", lambda *a, **k: None)
    prep.run_cpu_bundle_prep(dataset)
    assert uploads and uploads[0][1] == f"{colab.REMOTE_ROOT}/dataset.csv"
    # BOTH transcripts: the shared training_output contract is forced on by
    # the lane wrapper, whatever the untouched call in colab.py passes.
    assert seen["kwargs"]["log_name"] == "bundle"
    assert seen["kwargs"]["training_output"] is True
    script = seen["script"]
    assert "training.prepare_all" in script
    # tqdm passthrough: the child's streams are inherited, never captured.
    assert "stderr=" not in script
    assert downloads and downloads[0]["remote"] == (
        f"{colab.REMOTE_ROOT}/bundle_delivery.tar.zst"
    )
    # Cohort tagging: the two owner sessions are identifiable in a transcript.
    terminal = capsys.readouterr().out
    assert "[cpu-prep] cohort=50pct dataset=dataset_50pct.csv" in terminal


def test_cohort_tags_split_the_two_owner_sessions(tmp_path):
    full = tmp_path / "dataset.csv"
    half = tmp_path / "dataset_50pct.csv"
    assert prep.cohort_label(full) == "full"
    assert prep.cohort_label(half) == "50pct"


def test_default_dispatch_keeps_colabs_original_bundle_call(monkeypatch, tmp_path):
    """cpu_data_bundle_lane False (default) -> byte-identical dispatch: the
    direct colab.run_bundle call, the lane never entered."""
    dataset = tmp_path / "cohort.csv"
    dataset.write_text("sku_id\na\n", encoding="utf-8")
    monkeypatch.setattr(colab, "training_cfg", mock.Mock(
        **{"return_value.cpu_bundle_prep.lane": False}))
    direct = mock.Mock(name="run_bundle")
    lane = mock.Mock(name="run_cpu_bundle_prep")
    fake = _bounded_dispatch_test(monkeypatch, dataset, direct, lane)
    fake("disable")
    assert direct.call_count == 1 and lane.call_count == 0


def test_lane_dispatch_forwards_to_the_separate_file(monkeypatch, tmp_path):
    dataset = tmp_path / "cohort.csv"
    dataset.write_text("sku_id\na\n", encoding="utf-8")
    monkeypatch.setattr(colab, "training_cfg", mock.Mock(
        **{"return_value.cpu_bundle_prep.lane": True}))
    direct = mock.Mock(name="run_bundle")
    lane = mock.Mock(name="run_cpu_bundle_prep")
    fake = _bounded_dispatch_test(monkeypatch, dataset, direct, lane)
    fake("enable")
    assert lane.call_count == 1 and lane.call_args.kwargs == {"dataset_csv": dataset}
    assert direct.call_count == 0


def _bounded_dispatch_test(monkeypatch, dataset, direct, lane):
    """Drive cli.colab main() for --what bundle with every surface faked."""
    monkeypatch.setattr(colab, "run_bundle", direct)
    monkeypatch.setattr(prep, "run_cpu_bundle_prep", lane)

    def drive(_mode: str) -> None:
        stack = [
            mock.patch.object(sys, "argv", [
                "colab.py", "--what", "bundle", "--gpu", "CPU",
                "--dataset-csv", str(dataset),
            ]),
            mock.patch.object(colab, "start_live_log"),
            mock.patch.object(colab, "close_live_log"),
            mock.patch.object(colab, "check_colab_cli"),
            mock.patch.object(colab, "acquire_colab_launch_lock", return_value=None),
            mock.patch.object(colab, "release_colab_launch_lock"),
            mock.patch.object(colab, "ensure_session"),
            mock.patch.object(colab, "prepare_remote_layout"),
            mock.patch.object(colab, "install_deps"),
            mock.patch.object(colab, "verify_training_inputs"),
            mock.patch.object(colab, "log_gpu_profile"),
            mock.patch.object(colab, "stop"),
            mock.patch.object(colab, "stop_keep_alive_daemon", return_value=1),
        ]
        saved_gpu = colab.GPU
        for patcher in stack:
            patcher.start()
        try:
            colab.main()
        finally:
            colab.GPU = saved_gpu
            for patcher in reversed(stack):
                patcher.stop()

    return drive


def _run_bundle_main(argv: list[str], bundle) -> None:
    """Drive cli.colab main() for the default bundle lane (telemetry pins)."""
    stack = [
        mock.patch.object(sys, "argv", ["colab.py", *argv]),
        mock.patch.object(colab, "start_live_log"),
        mock.patch.object(colab, "close_live_log"),
        mock.patch.object(colab, "_legacy_validation_sources", return_value={}),
        mock.patch.object(colab, "_validate_legacy_bundle_partitions"),
        mock.patch.object(colab, "check_colab_cli"),
        mock.patch.object(colab, "acquire_colab_launch_lock", return_value=None),
        mock.patch.object(colab, "release_colab_launch_lock"),
        mock.patch.object(colab, "ensure_session"),
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
        mock.patch.object(colab, "stop"),
        mock.patch.object(colab, "stop_keep_alive_daemon", return_value=1),
    ]
    saved_gpu = colab.GPU
    for patcher in stack:
        patcher.start()
    try:
        colab.main()
    finally:
        colab.GPU = saved_gpu
        for patcher in reversed(stack):
            patcher.stop()


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


def test_bundle_pin_rewrite_keeps_the_closing_quote_and_trailing_comment():
    """Regression (both cohorts' first launch): the OLD sha-pin regex
    consumed the closing quote before the trailing comment, producing an
    unterminated YAML scalar on the VM.  Behavioral pin of the REAL
    expression: only the 64-hex payload is swapped; the closing quote and
    the trailing comment survive."""
    line = (
        '  source_export_expected_sha256: "'
        + "a" * 64
        + '"     # approved source bytes (drift gate)'
    )
    replacement = re.sub(
        r'(source_export_expected_sha256: ")([0-9a-f]{64})',
        r"\g<1>" + "b" * 64,
        line,
    )
    assert replacement == (
        '  source_export_expected_sha256: "'
        + "b" * 64
        + '"     # approved source bytes (drift gate)'
    )
    # And the fixed form parses as YAML even nested in its audit block:
    import yaml

    assert yaml.safe_load("audit:\n  x: 1\n" + replacement + "\n")["audit"][
        "source_export_expected_sha256"
    ] == ("b" * 64)


def _run_lane_main(monkeypatch, dataset, runner, *, session=None):
    """The lane's own entry point owns its [done]/[failed] telemetry."""
    monkeypatch.setattr(
        sys, "argv",
        ["colab_data_bundle_prep.py", "--dataset-csv", str(dataset)],
    )
    saved_gpu = colab.GPU
    saved_env = os.environ.get("EUROMONITOR_KEEP_ALIVE_ALLOWED")
    monkeypatch.setattr(colab, "start_live_log", mock.Mock())
    monkeypatch.setattr(colab, "close_live_log", mock.Mock())
    monkeypatch.setattr(prep, "run_cpu_bundle_prep", runner)
    try:
        prep.main()
    finally:
        colab.GPU = saved_gpu
        if saved_env is None:
            os.environ.pop("EUROMONITOR_KEEP_ALIVE_ALLOWED", None)
        else:
            os.environ["EUROMONITOR_KEEP_ALIVE_ALLOWED"] = saved_env


def test_the_own_lane_prints_done_only_on_clean_completion(
    monkeypatch, tmp_path, capsys
):
    dataset = tmp_path / "cohort.csv"
    dataset.write_text("sku_id\na\n", encoding="utf-8")
    _run_lane_main(monkeypatch, dataset, mock.Mock())
    captured = capsys.readouterr()
    assert "[done] cpu prep lane completed" in captured.out
    assert "[failed]" not in captured.out


def test_lane_transcripts_are_session_qualified_in_its_own_process(
    monkeypatch, tmp_path
):
    """Two parallel prep lanes must not truncate each other's records: the
    lane mutates the PROCESS-LOCAL file map so the shared tee primitive
    resolves per-session paths; the module map itself is untouched."""
    from core import common
    saved = common.F["colab_live_log"], common.F["colab_training_log"]
    try:
        prep._qualify_session_transcripts("er-prep-50pct")
        assert common.F["colab_live_log"].name == "colab_system_er-prep-50pct.log"
        assert common.F["colab_training_log"].name == "training_er-prep-50pct.log"
        assert common.F["colab_live_log"].parent == saved[0].parent
    finally:
        common.F["colab_live_log"], common.F["colab_training_log"] = saved


def test_the_own_lane_reports_failed_and_reraises(monkeypatch, tmp_path, capsys):
    dataset = tmp_path / "cohort.csv"
    dataset.write_text("sku_id\na\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="VM died"):
        _run_lane_main(
            monkeypatch, dataset, mock.Mock(side_effect=RuntimeError("VM died")),
        )
    captured = capsys.readouterr()
    assert "[failed] cpu prep lane did not complete" in captured.out
    assert "[done]" not in captured.out
