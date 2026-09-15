"""Keep-alive boundaries: provisioning always works; retention is CPU-only.

Two decisions used to be one, and conflating them stopped every GPU lane from
provisioning at all.  These tests pin them apart so it cannot silently revert:

* the DAEMON is spawned by `colab new` itself and is what provisioning needs,
  so it is never denied -- on any lane;
* RETENTION is CPU-only.  `--keep-alive` on a GPU lane is refused outright,
  and a GPU lane's daemon is stopped once the launcher owns the run so a crash
  cannot leave the VM held open.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cli import colab, colab_cli_entry


class TrainingLifecyclePreflightTests(unittest.TestCase):
    def test_held_out_inference_sample_reconstructs_the_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {
                "source": root / "source.csv",
                "training": root / "training.csv",
                "inference": root / "inference.csv",
            }
            paths["source"].write_text("product_id\na\nb\nc\nd\ne\n", encoding="utf-8")
            paths["training"].write_text("product_id\na\nb\nc\n", encoding="utf-8")
            paths["inference"].write_text("product_id\nd\ne\n", encoding="utf-8")
            by_config_path = {
                colab._COLAB.training_dataset_csv: paths["training"],
                colab._FINAL_INFERENCE.input_csv: paths["inference"],
                colab._FINAL_INFERENCE.source_csv: paths["source"],
            }
            with mock.patch.object(
                colab, "_validation_input_path", side_effect=by_config_path.__getitem__
            ), mock.patch.object(colab, "_expand_worker_profiles", return_value=["baseline"]), \
                 mock.patch.object(colab, "_EXPECTED_TRAINING_ROWS", 3), \
                 mock.patch.object(colab, "_EXPECTED_INFERENCE_ROWS", 2), \
                 mock.patch.object(colab, "_EXPECTED_SOURCE_ROWS", 5):
                result = colab.training_lifecycle_preflight(
                    workers=1, model="minilm_l6", masking_profile="baseline"
                )
        self.assertEqual(result["product_id_overlap"], 0)
        self.assertTrue(result["reconstructs_source"])
        self.assertEqual(result["source_rows"], 5)
        self.assertIn(
            "dataset_deduped_sample_3000.csv",
            " ".join(result["remote_completion_argv"]),
        )


class DaemonAtProvisioningTests(unittest.TestCase):
    def test_the_wrapper_never_denies_the_daemon(self):
        """`colab new` spawns the daemon itself; refusing it breaks provisioning."""
        entry = Path(colab_cli_entry.__file__).read_text(encoding="utf-8")
        self.assertNotIn(
            "EUROMONITOR_KEEP_ALIVE_ALLOWED",
            entry,
            "the wrapper must not gate the daemon it is asked to spawn",
        )
        with mock.patch.object(subprocess, "Popen", return_value=mock.Mock(pid=4242)) as spawn:
            pid = colab_cli_entry._spawn_keep_alive("endpoint", "session")
        self.assertEqual(pid, 4242)
        spawn.assert_called_once()
        argv = spawn.call_args.args[0]
        self.assertIn("keep-alive", argv)

    def test_the_daemon_gate_is_open_for_every_lane(self):
        """The launcher publishes an allow, not a per-lane decision."""
        source = Path(colab.__file__).read_text(encoding="utf-8")
        self.assertIn('os.environ["EUROMONITOR_KEEP_ALIVE_ALLOWED"] = "1"', source)


class RetentionIsCpuOnlyTests(unittest.TestCase):
    def _run_main(self, argv: list[str], **patches):
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
            mock.patch.object(colab, "check_colab_cli"),
            mock.patch.object(colab, "acquire_colab_launch_lock", return_value=None),
            mock.patch.object(colab, "release_colab_launch_lock"),
            mock.patch.object(colab, "ensure_session", record("session")),
            mock.patch.object(colab, "prepare_remote_layout"),
            mock.patch.object(colab, "install_deps"),
            mock.patch.object(colab, "verify_remote_models"),
            mock.patch.object(colab, "log_gpu_profile"),
            mock.patch.object(colab, "start_local_bundle_prewarm"),
            mock.patch.object(colab, "start_validation_upload_prewarm", return_value="stamp"),
            mock.patch.object(colab, "drain_local_bundle_prewarm"),
            mock.patch.object(colab, "drain_validation_upload_prewarm"),
            mock.patch.object(colab, "run_train", record("run_train", ("run", 1))),
            mock.patch.object(colab, "stop", record("stop")),
            mock.patch.object(
                colab, "stop_keep_alive_daemon", record("stop_daemon", 1)
            ),
        ]
        for target, value in patches.items():
            stack.append(mock.patch.object(colab, target, value))
        # main() assigns the accelerator to the module global, so a test that
        # runs a GPU launch must put it back or it poisons every later test.
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

    def test_a_gpu_launch_is_refused_with_keep_alive(self):
        with self.assertRaises(ValueError) as caught:
            self._run_main(["--what", "train", "--gpu", "T4", "--allow-gpu", "--keep-alive"])
        message = str(caught.exception)
        self.assertIn("CPU-only", message)
        self.assertIn("T4", message)

    def test_a_gpu_launch_provisions_and_then_stops_the_daemon(self):
        order = self._run_main(["--what", "train", "--gpu", "T4", "--allow-gpu"])
        self.assertIn("session", order)
        self.assertIn(
            "stop_daemon", order,
            "a GPU lane must stop the daemon once the launcher owns the run",
        )
        self.assertLess(
            order.index("session"), order.index("stop_daemon"),
            "the daemon is stopped AFTER provisioning, not before it",
        )
        # And the VM is still released by the launcher's own teardown.
        self.assertIn("stop", order)

    def test_a_cpu_launch_keeps_the_daemon_and_may_retain(self):
        order = self._run_main(["--what", "train", "--gpu", "CPU", "--keep-alive"])
        self.assertIn("session", order)
        self.assertNotIn(
            "stop_daemon", order,
            "a CPU lane asked to retain must keep its keep-alive daemon",
        )
        self.assertNotIn("stop", order, "a retained CPU VM is not torn down")

    def test_a_cpu_launch_without_keep_alive_is_torn_down_but_keeps_the_daemon_running(self):
        order = self._run_main(["--what", "train", "--gpu", "CPU"])
        self.assertNotIn("stop_daemon", order)
        self.assertIn("stop", order)


class DaemonStopTests(unittest.TestCase):
    def test_stopping_reports_when_there_is_nothing_to_stop(self):
        with mock.patch.object(colab, "keep_alive_daemon_pids", return_value=[]):
            captured = _Capture()
            with mock.patch("sys.stdout", captured):
                stopped = colab.stop_keep_alive_daemon(reason="test")
        self.assertEqual(stopped, 0)
        self.assertIn("no keep-alive daemon found", captured.text())

    def test_stopping_signals_every_matching_daemon(self):
        with mock.patch.object(colab, "keep_alive_daemon_pids", return_value=[11, 12]), \
             mock.patch.object(colab.os, "kill") as kill:
            stopped = colab.stop_keep_alive_daemon(reason="test")
        self.assertEqual(stopped, 2)
        self.assertEqual([c.args[0] for c in kill.call_args_list], [11, 12])
        for call in kill.call_args_list:
            self.assertEqual(call.args[1], colab.signal.SIGTERM)

    def test_a_daemon_that_dies_between_listing_and_signalling_is_not_an_error(self):
        with mock.patch.object(colab, "keep_alive_daemon_pids", return_value=[11]), \
             mock.patch.object(
                 colab.os, "kill", side_effect=ProcessLookupError
             ):
            stopped = colab.stop_keep_alive_daemon(reason="test")
        self.assertEqual(stopped, 1)

    def test_the_daemon_match_rule_has_one_definition(self):
        """Both the stale-record path and the backstop use the same predicate."""
        source = Path(colab.__file__).read_text(encoding="utf-8")
        self.assertEqual(
            source.count("colab_cli_entry.py\" in command and \"keep-alive\""), 0,
            "the match rule must not be duplicated inline",
        )
        self.assertGreaterEqual(source.count("_is_keep_alive_daemon(command)"), 2)


class _Capture:
    def __init__(self) -> None:
        self._parts: list[str] = []

    def write(self, text: str) -> int:
        self._parts.append(text)
        return len(text)

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        return False

    def text(self) -> str:
        return "".join(self._parts)


if __name__ == "__main__":
    unittest.main()
