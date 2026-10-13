"""Colab lane boundaries: data preparation, accelerator ack, unconditional release.

Three contracts the launcher enforces before it allocates anything, and one it
enforces after everything:

* RAW CPU PREPARATION never runs in an accelerator launch
  (``enforce_data_preparation_boundary``);
* a GPU launch must be acknowledged with ``--allow-gpu``
  (``ColabGPULane.enforce_boundary``) — and that is the whole accelerator gate;
* ``--what stop`` launches nothing, so it must never demand the ack;
* the VM is RELEASED on every outcome, on every lane: the retention flag and
  the keep-alive daemon machinery that existed only to hold a session open are
  deleted (owner ruling 2026-10-13), so no code path can outlive its run.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cli import colab


class GPUDataPreparationBoundaryTests(unittest.TestCase):
    def setUp(self):
        from cli.colab_lane import ColabGPULane

        self.lane = ColabGPULane()

    def test_gpu_launch_rejects_raw_preparation_and_unprepared_training(self):
        for name in ("bundle", "hpo", "mixed"):
            with self.subTest(lane=name), self.assertRaisesRegex(ValueError, "CPU data preparation"):
                self.lane.enforce_data_preparation_boundary(
                    "A100", name, prepared_inputs=False,
                )
        for name in ("train", "dual-train", "tracks"):
            with self.subTest(lane=name), self.assertRaisesRegex(ValueError, "no CPU-lane prepared bundle"):
                self.lane.enforce_data_preparation_boundary(
                    "A100", name, prepared_inputs=False,
                )

    def test_gpu_training_accepts_prepared_inputs_and_cpu_lane_can_prepare(self):
        for name in ("train", "dual-train", "tracks"):
            self.lane.enforce_data_preparation_boundary(
                "A100", name, prepared_inputs=True,
            )
        for name in ("bundle", "hpo", "mixed", "train", "dual-train", "tracks"):
            self.lane.enforce_data_preparation_boundary(
                "CPU", name, prepared_inputs=False,
            )


class TrainingLifecyclePreflightTests(unittest.TestCase):
    def test_preflight_reports_the_scored_pair_accounting(self):
        """SCORED-PAIR contract (2026-10-01): no reconstruct-the-source claim.

        The preflight derives its rows at exec time from the artifacts
        (census-simulated here) and serves the scored population to the
        remote completion, never the retired component holdout.
        """
        census = {
            "source_export_rows": 71_623,
            "deduped_rows": 63_079,
            "dropped_rows": 8_544,
            "train_side_rows": 49_152,
            "validation_entity_rows": 13_927,
            "scored_pair_rows": 6_351,
            "scored_population_path": "/repo/data/final_validation.csv",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training = root / "training.csv"
            training.write_text("sku_id\na\nb\n", encoding="utf-8")
            with mock.patch.object(colab, "_validation_input_path", return_value=training), \
                 mock.patch.object(colab, "_expand_worker_profiles", return_value=["baseline"]), \
                 mock.patch.object(colab, "_scored_validation_census", return_value=census):
                result = colab.training_lifecycle_preflight(
                    workers=1, model="minilm_l6", masking_profile="baseline"
                )
        # The identity the real census asserts before the dict is built:
        self.assertEqual(
            census["train_side_rows"] + census["validation_entity_rows"],
            census["deduped_rows"],
        )
        self.assertEqual(
            census["deduped_rows"] + census["dropped_rows"],
            census["source_export_rows"],
        )
        self.assertEqual(result["training_rows"], 49_152)
        self.assertEqual(result["inference_rows"], 6_351)
        self.assertTrue(str(result["inference_dataset"]).endswith("final_validation.csv"))
        self.assertNotIn(
            "component_holdout.csv",
            str(result),
            "the retired component holdout is no longer the scored population",
        )
        self.assertNotIn("reconstructs_source", result)


class VmIsAlwaysReleasedTests(unittest.TestCase):
    """The retention flag and the keep-alive daemon are gone; release is total."""

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
            mock.patch.object(colab, "stop", record("stop")),
            mock.patch.object(colab, "spawn_self_watch"),
        ]
        for target, value in patches.items():
            stack.append(mock.patch.object(colab, target, value))
        saved_gpu = colab.GPU
        for patcher in stack:
            patcher.start()
        try:
            colab.main()
        except BaseException:  # a failing lane re-raises; the teardown order is the subject
            pass
        finally:
            colab.GPU = saved_gpu
            for patcher in reversed(stack):
                patcher.stop()
        return order

    def test_every_lane_releases_the_vm_on_success_and_on_failure(self):
        """A CPU launch, a GPU launch and a failed launch all end in stop()."""
        for argv in (["--what", "train", "--gpu", "CPU"],
                     ["--what", "train", "--gpu", "T4", "--allow-gpu"]):
            with self.subTest(argv=argv):
                order = self._run_main(argv, run_train=mock.Mock(return_value=("run", 1)))
                self.assertEqual(order[-1], "stop")

        with self.subTest(argv="failing launch"):
            order = self._run_main(
                ["--what", "train", "--gpu", "CPU"],
                run_train=mock.Mock(side_effect=RuntimeError("lane failed")))
            self.assertEqual(order[-1], "stop")

    def test_no_flag_and_no_daemon_hook_can_hold_a_vm_open(self):
        """The retention surface is gone from the parser and the runtime."""
        self.assertNotIn("--keep-alive", Path(colab.__file__).read_text(encoding="utf-8"))
        for name in ("stop_keep_alive_daemon", "keep_alive_daemon_pids",
                     "_is_keep_alive_daemon"):
            self.assertFalse(hasattr(colab, name), name)


class TeardownOpIsNotGpuGatedTests(unittest.TestCase):
    """`--what stop` launches nothing, so the GPU-launch ack must not gate it."""

    def test_bare_stop_is_not_refused_by_the_gpu_gate(self):
        """Bare `er-colab --what stop` must release, not demand --allow-gpu.

        Regression: `--gpu` defaults to T4 (non-CPU) and the GPU boundary gate
        ran BEFORE the stop dispatch, so a teardown op failed with "GPU launch
        requires --allow-gpu" despite provisioning nothing (Gate-1 colab GPU
        smoke 1010T220607339825Z). Stop now dispatches beside the other
        no-launch op, ahead of the gate.
        """
        with mock.patch.object(sys, "argv", ["colab.py", "--what", "stop"]), \
             mock.patch.object(colab, "stop", return_value=True) as stop_mock:
            colab.main()
        stop_mock.assert_called_once_with(stop_local_owner=True)
