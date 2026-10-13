"""Offline proofs for the Colab setup-path optimisations in src/cli/colab.py.

Every test here runs without a Colab VM: the generated remote installer
program is executed locally against a stubbed `uv`, and the launcher's
orchestration is driven with mocked remote calls.  What still needs a real VM
is stated in COLAB_SETUP_OPTIMISATION_REPORT.md.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cli import colab


def _installer_argv(command: str) -> list[str]:
    """Recover the remote argv from the generated command expression.

    Only the leading ``sys.executable`` is substituted — the program text
    legitimately mentions it too, and must stay intact.
    """
    prefix = "[sys.executable, "
    assert command.startswith(prefix), command
    argv = ast.literal_eval("[" + repr(sys.executable) + ", " + command[len(prefix):])
    assert argv[0] == sys.executable, argv
    assert argv[1] == "-c", argv
    return argv


def _installer_program(command: str) -> str:
    return _installer_argv(command)[2]


def _installer_packages(command: str) -> list[str]:
    program = _installer_program(command)
    return ast.literal_eval(program.split("packages = ", 1)[1].split("\n", 1)[0])


class PrebuiltWheelTests(unittest.TestCase):
    """A shipped wheel must replace its distribution, but only when it fits."""

    def _wheel_name(self) -> str:
        import sysconfig

        version = f"cp{sys.version_info.major}{sys.version_info.minor}"
        platform = sysconfig.get_platform().replace("-", "_")
        return f"hnswlib-0.8.0-{version}-{version}-{platform}.whl"

    def _run_with_stub_uv(self, *, wheel: str | None):
        """Generate the installer against a temp repo root and run it."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wheels = root / "artifacts" / "wheels"
            wheels.mkdir(parents=True)
            paths: list[str] = []
            if wheel is not None:
                (wheels / wheel).write_bytes(b"wheel")
                paths = [f"artifacts/wheels/{wheel}"]
            record = root / "uv-argv"
            stub = root / "uv"
            stub.write_text(
                "#!/bin/sh\n"
                f'printf "%s\\n" "$@" > {str(record)!r}\n'
                "exit 0\n"
            )
            stub.chmod(0o755)
            with mock.patch.object(colab, "REMOTE_ROOT", str(root)):
                program = _installer_program(
                    colab._runtime_install_command(
                        ["hnswlib", "wandb"], prefer_uv=True, wheel_paths=paths
                    )
                )
            completed = subprocess.run(
                [sys.executable, "-c", program],
                capture_output=True,
                text=True,
                env={**os.environ, "PATH": str(root)},
            )
            return completed, record.read_text().splitlines()

    def test_matching_wheel_replaces_the_distribution(self):
        completed, argv = self._run_with_stub_uv(wheel=self._wheel_name())
        self.assertIn("replaces hnswlib", completed.stdout)
        self.assertTrue(
            any(arg.endswith(".whl") for arg in argv),
            f"the wheel was not passed to the installer: {argv}",
        )
        self.assertNotIn("hnswlib", argv, "the sdist requirement must be dropped")
        self.assertIn("wandb", argv)

    def test_foreign_wheel_is_rejected_and_the_index_is_used(self):
        completed, argv = self._run_with_stub_uv(
            wheel="hnswlib-0.8.0-cp39-cp39-win_amd64.whl"
        )
        self.assertIn("does not match", completed.stdout)
        self.assertFalse(any(arg.endswith(".whl") for arg in argv))
        self.assertIn("hnswlib", argv)

    def test_missing_wheel_is_reported_and_the_index_is_used(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "artifacts" / "wheels").mkdir(parents=True)
            record = root / "uv-argv"
            stub = root / "uv"
            stub.write_text(
                "#!/bin/sh\n" f'printf "%s\\n" "$@" > {str(record)!r}\n' "exit 0\n"
            )
            stub.chmod(0o755)
            with mock.patch.object(colab, "REMOTE_ROOT", str(root)):
                program = _installer_program(
                    colab._runtime_install_command(
                        ["hnswlib", "wandb"], prefer_uv=True,
                        wheel_paths=["artifacts/wheels/absent.whl"],
                    )
                )
            completed = subprocess.run(
                [sys.executable, "-c", program], capture_output=True, text=True,
                env={**os.environ, "PATH": str(root)},
            )
            argv = record.read_text().splitlines()
        self.assertIn("prebuilt wheel missing", completed.stdout)
        self.assertIn("hnswlib", argv)


class LauncherOrderTests(unittest.TestCase):
    """Standard full training uploads its configured immutable bundle."""

    def test_main_uploads_the_configured_full_bundle_without_local_rebuild(self):
        order: list[str] = []

        def record(label: str, result=None):
            def side_effect(*_args, **_kwargs):
                order.append(label)
                return result

            return side_effect

        run_train = mock.Mock(return_value=("run", 1))
        with mock.patch.object(sys, "argv", ["colab.py", "--what", "train", "--gpu", "CPU"]), \
             mock.patch.object(colab, "start_live_log"), \
             mock.patch.object(colab, "close_live_log"), \
             mock.patch.object(colab, "check_colab_cli"), \
             mock.patch.object(colab, "acquire_colab_launch_lock", return_value=None), \
             mock.patch.object(colab, "release_colab_launch_lock"), \
             mock.patch.object(colab, "ensure_session", record("session")), \
             mock.patch.object(colab, "prepare_remote_layout", record("layout")), \
             mock.patch.object(colab, "install_deps", record("install_deps")), \
             mock.patch.object(colab, "verify_remote_models", record("models")), \
             mock.patch.object(colab, "log_gpu_profile", record("profile")), \
             mock.patch.object(colab, "run_train", run_train), \
             mock.patch.object(colab, "stop", record("stop")), \
             mock.patch.object(colab, "spawn_self_watch"):
            colab.main()

        self.assertEqual(order[:2], ["session", "layout"], order)
        self.assertEqual(
            run_train.call_args.kwargs["local_prepared_bundles"],
            [colab.TRAIN_ROOT / colab._COLAB.prepared_bundles.baseline()],
        )
        self.assertEqual(
            run_train.call_args.kwargs["remote_dataset_csv"],
            colab._COLAB.training_dataset_csv,
        )

    def test_a_resumed_run_launches_without_the_fresh_bundle_audit(self):
        """A resumed lane keeps its existing identity and reuses its run dir."""
        run_train = mock.Mock(return_value=("run", 1))
        with mock.patch.object(
            sys, "argv", ["colab.py", "--what", "train", "--resume-run", "abc", "--gpu", "CPU"]
        ), \
             mock.patch.object(colab, "start_live_log"), \
             mock.patch.object(colab, "close_live_log"), \
             mock.patch.object(colab, "check_colab_cli"), \
             mock.patch.object(colab, "acquire_colab_launch_lock", return_value=None), \
             mock.patch.object(colab, "release_colab_launch_lock"), \
             mock.patch.object(colab, "ensure_session"), \
             mock.patch.object(colab, "prepare_remote_layout"), \
             mock.patch.object(colab, "install_deps"), \
             mock.patch.object(colab, "verify_remote_models"), \
             mock.patch.object(colab, "log_gpu_profile"), \
             mock.patch.object(colab, "run_train", return_value=("run", 1)), \
             mock.patch.object(colab, "stop"), \
             mock.patch.object(colab, "spawn_self_watch"), \
             mock.patch.object(colab, "run_train", run_train):
            colab.main()
        # A resumed run keeps its own identity: it launches no fresh bundle audit.
        run_train.assert_called_once()
        self.assertIsNone(run_train.call_args.kwargs["local_prepared_bundles"])


class _Capture:
    """Minimal stdout stand-in that keeps what the launcher printed."""

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


class RuntimeInstallCommandTests(unittest.TestCase):
    """`install_deps` must use uv when it can and pip when it cannot."""

    def _run_program(self, program: str, *, stub: str | None, stub_rc: int = 0):
        """Run the generated installer locally with a stubbed `uv` on PATH.

        No packages are requested: the uv path is intercepted by the stub, and
        the pip fallback then exits on an empty requirement list instead of
        installing anything.  The assertion is on which installer ran.
        """
        with tempfile.TemporaryDirectory() as temporary:
            bin_dir = Path(temporary)
            invoked = bin_dir / "uv-was-invoked"
            if stub is not None:
                stub_path = bin_dir / "uv"
                stub_path.write_text(
                    "#!/bin/sh\n"
                    f"printf '%s\\n' {stub!r} > {str(invoked)!r}\n"
                    f"exit {stub_rc}\n"
                )
                stub_path.chmod(0o755)
            completed = subprocess.run(
                [sys.executable, "-c", program],
                capture_output=True,
                text=True,
                # Only the stub directory is on PATH, so "uv absent" is real.
                env={**os.environ, "PATH": str(bin_dir)},
            )
            return completed, invoked.is_file()

    def test_prefers_uv_and_does_not_touch_pip_when_uv_succeeds(self):
        program = _installer_program(colab._runtime_install_command([], prefer_uv=True, wheel_paths=[]))
        completed, invoked = self._run_program(program, stub="uv-ok", stub_rc=0)
        self.assertTrue(invoked)
        self.assertIn("[deps] installer=uv", completed.stdout)
        self.assertNotIn("installer=pip", completed.stdout)
        self.assertEqual(completed.returncode, 0)

    def test_falls_back_to_pip_and_says_so_when_uv_fails(self):
        program = _installer_program(colab._runtime_install_command([], prefer_uv=True, wheel_paths=[]))
        completed, invoked = self._run_program(program, stub="uv-broken", stub_rc=3)
        self.assertTrue(invoked)
        self.assertIn("[deps] installer=uv", completed.stdout)
        self.assertIn("uv install failed; falling back to pip", completed.stdout)
        self.assertIn("[deps] installer=pip", completed.stdout)

    def test_falls_back_to_pip_when_uv_is_absent(self):
        program = _installer_program(colab._runtime_install_command([], prefer_uv=True, wheel_paths=[]))
        completed, invoked = self._run_program(program, stub=None)
        self.assertFalse(invoked)
        self.assertIn("uv is absent on the VM; falling back to pip", completed.stdout)
        self.assertIn("[deps] installer=pip", completed.stdout)

    def test_configuration_can_force_pip_without_invoking_uv(self):
        program = _installer_program(colab._runtime_install_command([], prefer_uv=False, wheel_paths=[]))
        completed, invoked = self._run_program(program, stub="uv-should-not-run")
        self.assertFalse(invoked)
        self.assertIn("uv disabled by configuration; using pip", completed.stdout)
        self.assertIn("[deps] installer=pip", completed.stdout)

    def test_uv_installs_into_the_interpreter_the_trainer_will_use(self):
        """The fast path must not be able to land packages somewhere else."""
        program = _installer_program(
            colab._runtime_install_command(["hnswlib"], prefer_uv=True, wheel_paths=[])
        )
        self.assertIn('"--python", sys.executable', program)
        self.assertIn("['hnswlib']", program)


class InstallDepsLaneTests(unittest.TestCase):
    """The package set is config-owned and selected per lane."""

    def _packages_installed(self, *, minimal_runtime: bool) -> list[str]:
        with mock.patch.object(colab, "run_detached_stage") as stage:
            colab.install_deps(minimal_runtime=minimal_runtime)
        stage.assert_called_once()
        name, command = stage.call_args.args[0], stage.call_args.args[1]
        self.assertEqual(name, "00_deps")
        return _installer_packages(command)

    def test_prepared_lane_installs_the_configured_prepared_list(self):
        installed = self._packages_installed(minimal_runtime=True)
        self.assertEqual(installed, list(colab._RUNTIME_PACKAGES.prepared))
        self.assertNotIn("optuna", installed)
        self.assertNotIn("evaluate", installed)

    def test_full_lane_installs_the_configured_full_list(self):
        installed = self._packages_installed(minimal_runtime=False)
        self.assertEqual(installed, list(colab._RUNTIME_PACKAGES.full))
        self.assertNotIn("optuna", installed)
        self.assertGreater(
            len(colab._RUNTIME_PACKAGES.full), len(colab._RUNTIME_PACKAGES.prepared)
        )


class ConfiguredBundleRunTests(unittest.TestCase):
    def test_run_train_uploads_the_configured_bundle_without_rebuilding(self):
        bundle = colab.TRAIN_ROOT / colab._COLAB.prepared_bundles.baseline()
        with mock.patch.object(
            colab, "run_single_train_and_stream", return_value=("run", 0)
        ) as stream:
            result = colab.run_train(
                colab._TRAIN_FRAC_DEFAULT,
                colab._EPOCHS_DEFAULT,
                sample=None,
                workers=1,
                masking_profile=colab._MASKING_PROFILE,
                collapse_guardrail_profile=colab._COLLAPSE_GUARDRAIL_PROFILE,
                loss=colab._TRAIN_LOSS,
                local_prepared_bundles=[bundle],
                remote_dataset_csv=colab._COLAB.training_dataset_csv,
            )

        self.assertEqual(result, ("run", 0))
        self.assertEqual(stream.call_args.kwargs["prepared_bundle"], bundle)
        self.assertIsNone(stream.call_args.kwargs["remote_checkout_bundle"])
        self.assertNotIn("--dataset", stream.call_args.args[0])

    def test_run_train_refuses_a_request_no_cpu_lane_prepared(self):
        """No local fallback: an unprepared request fails loud, it never builds."""
        with self.assertRaisesRegex(ValueError, "no prepared bundle"):
            colab.run_train(
                colab._TRAIN_FRAC_DEFAULT,
                colab._EPOCHS_DEFAULT,
                sample=None,
                workers=1,
                masking_profile=colab._MASKING_PROFILE,
                collapse_guardrail_profile=colab._COLLAPSE_GUARDRAIL_PROFILE,
                loss=colab._TRAIN_LOSS,
            )


class PreparedBundleUploadTests(unittest.TestCase):
    """Worker directories must cost one exec, not one per uploaded file."""

    def test_all_worker_dirs_are_created_in_a_single_exec(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundles = []
            for number in (1, 2, 3):
                bundle = root / f"worker_{number}_baseline.pkl.gz"
                bundle.write_bytes(b"bundle")
                bundle.with_suffix(bundle.suffix + ".json").write_text("{}\n")
                bundles.append(bundle)

            with mock.patch.object(colab, "run_colab_exec_stream") as stream, \
                 mock.patch.object(colab, "_upload_with_retries") as upload:
                remotes = colab._upload_prepared_bundles(
                    run_id="0915T000000000000Z", bundles=bundles
                )

            self.assertEqual(stream.call_count, 1)
            script = stream.call_args.args[1]
            for number in (1, 2, 3):
                self.assertIn(f"worker_{number}", script)
            self.assertEqual(upload.call_count, 6)
            self.assertEqual(len(remotes), 3)
            self.assertTrue(remotes[0].endswith("/worker_1/worker_1_baseline.pkl.gz"))


class SampleSuiteGuardTests(unittest.TestCase):
    """--sample must never silently no-op on an all-track suite (GAP3)."""

    def test_sample_is_refused_on_a_suite(self):
        """The suite trains the frozen inputs it is handed; --sample is dead there.

        A sampled suite is PREPARED (model_tracks.smoke_inputs.prepare_smoke),
        not capped at launch. The guard fires before any provisioning, so this
        needs no VM mocks; start_live_log is faked because live-from-t0 opens
        the transcript before the arg guards, and a guard test must not write
        the real log roof.
        """
        with mock.patch.object(
            sys, "argv", ["colab.py", "--what", "tracks", "--sample", "5", "--gpu", "CPU"]
        ), mock.patch.object(colab, "start_live_log"), self.assertRaises(ValueError) as raised:
            colab.main()
        self.assertIn("does not apply to an all-track suite", str(raised.exception))
        self.assertIn("prepare_smoke", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
