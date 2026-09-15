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
import threading
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
                        ["hnswlib", "mlflow"], prefer_uv=True, wheel_paths=paths
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
        self.assertIn("mlflow", argv)

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
                        ["hnswlib", "mlflow"], prefer_uv=True,
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


class BundleCacheTests(unittest.TestCase):
    """A byte-identical bundle request must not be rebuilt."""

    def _fixture(self, temporary: str):
        root = Path(temporary)
        dataset = root / "dataset.csv"
        dataset.write_text("product_id\n1\n")
        manifest = mock.Mock(
            n_df=58_529, n_payload=94_124, n_pos=44_690, n_neg=20_860,
            sha256="a" * 64,
        )

        def fake_build(command, **_kwargs):
            target = Path(command[command.index("--prepare-bundle") + 1])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"bundle-bytes")
            target.with_suffix(target.suffix + ".json").write_text("{}\n")

        return root, dataset, manifest, fake_build

    def test_second_identical_request_is_served_from_the_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, dataset, manifest, fake_build = self._fixture(temporary)
            with mock.patch.object(colab, "RESULTS", root / "results"), \
                 mock.patch.object(colab, "_validation_input_path", return_value=dataset), \
                 mock.patch.object(colab.subprocess, "run", side_effect=fake_build), \
                 mock.patch(
                     "training.prepared_bundle.load_prepared_bundle",
                     return_value=(manifest, None),
                 ):
                first = colab._build_local_training_bundles(
                    profiles=["baseline"], model=None, sample=None
                )
            self.assertEqual(len(first), 1)
            with mock.patch.object(colab, "RESULTS", root / "results"), \
                 mock.patch.object(colab, "_validation_input_path", return_value=dataset), \
                 mock.patch.object(colab.subprocess, "run", side_effect=fake_build) as build, \
                 mock.patch(
                     "training.prepared_bundle.load_prepared_bundle",
                     return_value=(manifest, None),
                 ):
                second = colab._build_local_training_bundles(
                    profiles=["baseline"], model=None, sample=None
                )
            self.assertEqual(len(second), 1)
            self.assertEqual(
                build.call_count, 0, "a cache hit must not rebuild the bundle"
            )

    def test_changed_source_invalidates_the_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, dataset, manifest, fake_build = self._fixture(temporary)
            with mock.patch.object(colab, "RESULTS", root / "results"), \
                 mock.patch.object(colab, "_validation_input_path", return_value=dataset), \
                 mock.patch.object(colab.subprocess, "run", side_effect=fake_build), \
                 mock.patch(
                     "training.prepared_bundle.load_prepared_bundle",
                     return_value=(manifest, None),
                 ), \
                 mock.patch.object(colab, "_tree_digest", return_value="before"):
                first = colab._build_local_training_bundles(
                    profiles=["baseline"], model=None, sample=None
                )
            with mock.patch.object(colab, "RESULTS", root / "results"), \
                 mock.patch.object(colab, "_validation_input_path", return_value=dataset), \
                 mock.patch.object(colab.subprocess, "run", side_effect=fake_build) as build, \
                 mock.patch(
                     "training.prepared_bundle.load_prepared_bundle",
                     return_value=(manifest, None),
                 ), \
                 mock.patch.object(colab, "_tree_digest", return_value="after"):
                second = colab._build_local_training_bundles(
                    profiles=["baseline"], model=None, sample=None
                )
            self.assertEqual(
                build.call_count, 1, "a source change must force a rebuild"
            )
            self.assertEqual(second[0].name, first[0].name)
            self.assertNotEqual(second[0].parent.name, first[0].parent.name)

    def test_changed_dataset_invalidates_the_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, dataset, manifest, fake_build = self._fixture(temporary)
            with mock.patch.object(colab, "RESULTS", root / "results"), \
                 mock.patch.object(colab, "_validation_input_path", return_value=dataset), \
                 mock.patch.object(colab.subprocess, "run", side_effect=fake_build), \
                 mock.patch(
                     "training.prepared_bundle.load_prepared_bundle",
                     return_value=(manifest, None),
                 ):
                first = colab._build_local_training_bundles(
                    profiles=["baseline"], model=None, sample=None
                )
            dataset.write_text("product_id\n2\n")
            with mock.patch.object(colab, "RESULTS", root / "results"), \
                 mock.patch.object(colab, "_validation_input_path", return_value=dataset), \
                 mock.patch.object(colab.subprocess, "run", side_effect=fake_build) as build, \
                 mock.patch(
                     "training.prepared_bundle.load_prepared_bundle",
                     return_value=(manifest, None),
                 ):
                second = colab._build_local_training_bundles(
                    profiles=["baseline"], model=None, sample=None
                )
            self.assertEqual(build.call_count, 1, "new dataset bytes must rebuild")
            self.assertNotEqual(second[0].parent.name, first[0].parent.name)

    def test_disabled_cache_always_rebuilds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, dataset, manifest, fake_build = self._fixture(temporary)
            with mock.patch.object(colab, "RESULTS", root / "results"), \
                 mock.patch.object(colab, "_CACHE_PREPARED_BUNDLES", False), \
                 mock.patch.object(colab, "_validation_input_path", return_value=dataset), \
                 mock.patch.object(colab.subprocess, "run", side_effect=fake_build), \
                 mock.patch(
                     "training.prepared_bundle.load_prepared_bundle",
                     return_value=(manifest, None),
                 ):
                colab._build_local_training_bundles(
                    profiles=["baseline"], model=None, sample=None
                )
            with mock.patch.object(colab, "RESULTS", root / "results"), \
                 mock.patch.object(colab, "_CACHE_PREPARED_BUNDLES", False), \
                 mock.patch.object(colab, "_validation_input_path", return_value=dataset), \
                 mock.patch.object(colab.subprocess, "run", side_effect=fake_build) as build, \
                 mock.patch(
                     "training.prepared_bundle.load_prepared_bundle",
                     return_value=(manifest, None),
                 ):
                colab._build_local_training_bundles(
                    profiles=["baseline"], model=None, sample=None
                )
            self.assertEqual(build.call_count, 1, "the disabled cache must not serve")

    def test_an_unusable_cache_entry_is_reported_and_rebuilt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, dataset, manifest, fake_build = self._fixture(temporary)
            with mock.patch.object(colab, "RESULTS", root / "results"), \
                 mock.patch.object(colab, "_validation_input_path", return_value=dataset), \
                 mock.patch.object(colab.subprocess, "run", side_effect=fake_build), \
                 mock.patch(
                     "training.prepared_bundle.load_prepared_bundle",
                     return_value=(manifest, None),
                 ):
                colab._build_local_training_bundles(
                    profiles=["baseline"], model=None, sample=None
                )
            captured = _Capture()
            with mock.patch.object(colab, "RESULTS", root / "results"), \
                 mock.patch.object(colab, "_validation_input_path", return_value=dataset), \
                 mock.patch.object(colab.subprocess, "run", side_effect=fake_build) as build, \
                 mock.patch(
                     "training.prepared_bundle.load_prepared_bundle",
                     side_effect=lambda path, *a, **k: (
                         (_ for _ in ()).throw(
                             ValueError("model input contract moved")
                         )
                         if "_cache" in str(path) else (manifest, None)
                     ),
                 ), \
                 mock.patch("sys.stdout", captured):
                colab._build_local_training_bundles(
                    profiles=["baseline"], model=None, sample=None
                )
            self.assertEqual(build.call_count, 1)
            self.assertIn("not reusable", captured.text())


class ValidationUploadPrewarmTests(unittest.TestCase):
    """Validation uploads must travel while the VM installs its runtime."""

    def tearDown(self) -> None:
        colab.drain_validation_upload_prewarm()

    def test_worker_uploads_once_without_joining_itself(self):
        """Regression guard for the self-join that broke the bundle prewarm.

        The worker runs the real transfer step, so if that step consulted the
        prewarm lookup the thread would join itself and the main thread would
        re-upload serially — the overlap would silently never happen.
        """
        threads: list[str] = []
        original = colab._perform_validation_upload

        def spy(run_id):
            threads.append(threading.current_thread().name)
            return original(run_id)

        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "dataset.csv"
            dataset.write_text("product_id\n1\n")
            with mock.patch.object(colab, "_perform_validation_upload", spy), \
                 mock.patch.object(
                     colab, "_validation_input_path", return_value=dataset
                 ), \
                 mock.patch.object(colab, "_remote_checkout_copy", return_value=None), \
                 mock.patch.object(colab, "run_colab_exec_stream"), \
                 mock.patch.object(colab, "_upload_with_retries"):
                colab.start_validation_upload_prewarm()
                self.assertEqual(
                    colab._lane_run_stamp(), colab._VALIDATION_UPLOAD_PREWARM.stamp
                )
                paths = colab._upload_validation_inputs(colab._lane_run_stamp())

        self.assertEqual(sorted(paths), ["sample", "source", "training"])
        self.assertEqual(len(threads), 1, "the transfer must run exactly once")
        self.assertNotEqual(
            threads[0], threading.current_thread().name,
            "the transfer ran on the caller's thread: the prewarm thread never "
            "reached it (it joined itself)",
        )

    def test_failure_falls_back_to_a_serial_upload(self):
        attempts: list[str] = []

        def flaky(run_id):
            attempts.append(run_id)
            if len(attempts) == 1:
                raise RuntimeError("control channel busy")
            return {"source": "s", "training": "t", "sample": "s"}

        with mock.patch.object(colab, "_perform_validation_upload", flaky):
            colab.start_validation_upload_prewarm()
            paths = colab._upload_validation_inputs(colab._lane_run_stamp())

        self.assertEqual(paths["source"], "s")
        self.assertEqual(len(attempts), 2, "the serial retry must have run")

    def test_mismatched_run_id_uploads_serially(self):
        with mock.patch.object(
            colab, "_perform_validation_upload",
            return_value={"source": "s", "training": "t", "sample": "s"},
        ) as worker:
            colab.start_validation_upload_prewarm()
            colab._upload_validation_inputs("some-other-run")
        self.assertEqual(worker.call_count, 2)

    def test_drain_waits_for_an_abandoned_upload(self):
        release = threading.Event()
        finished = threading.Event()

        def slow(run_id):
            release.wait(timeout=10)
            finished.set()
            return {"source": "s", "training": "t", "sample": "s"}

        with mock.patch.object(colab, "_perform_validation_upload", slow):
            colab.start_validation_upload_prewarm()
            release.set()
            colab.drain_validation_upload_prewarm()
            self.assertTrue(finished.is_set())


class LauncherOrderTests(unittest.TestCase):
    """main() must start the build before it starts paying for the VM."""

    def test_main_starts_the_local_build_before_the_dependency_install(self):
        order: list[str] = []

        def record(label: str, result=None):
            def side_effect(*_args, **_kwargs):
                order.append(label)
                return result

            return side_effect

        prewarm = mock.Mock(side_effect=record("prewarm"))
        upload_prewarm = mock.Mock(side_effect=record("upload_prewarm"))
        with mock.patch.object(sys, "argv", ["colab.py", "--what", "train"]), \
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
             mock.patch.object(colab, "run_train", record("run_train", ("run", 1))), \
             mock.patch.object(colab, "stop", record("stop")), \
             mock.patch.object(colab, "drain_local_bundle_prewarm"), \
             mock.patch.object(colab, "drain_validation_upload_prewarm"), \
             mock.patch.object(colab, "start_local_bundle_prewarm", prewarm), \
             mock.patch.object(
                 colab, "start_validation_upload_prewarm", upload_prewarm
             ):
            colab.main()

        self.assertEqual(order[:2], ["prewarm", "upload_prewarm"], order)
        # Both overlaps must begin before the launcher starts paying for the VM.
        self.assertLess(order.index("prewarm"), order.index("install_deps"), order)
        self.assertLess(order.index("upload_prewarm"), order.index("session"), order)
        prewarm.assert_called_once()
        upload_prewarm.assert_called_once()
        self.assertEqual(
            sorted(prewarm.call_args.kwargs), ["model", "profiles", "sample"]
        )

    def test_a_resumed_run_does_not_prewarm_its_uploads(self):
        """A resumed lane keeps its existing run identity and uploads serially."""
        with mock.patch.object(
            sys, "argv", ["colab.py", "--what", "train", "--resume-run", "abc"]
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
             mock.patch.object(colab, "drain_local_bundle_prewarm"), \
             mock.patch.object(colab, "drain_validation_upload_prewarm"), \
             mock.patch.object(colab, "start_local_bundle_prewarm"), \
             mock.patch.object(colab, "start_validation_upload_prewarm") as upload:
            colab.main()
        upload.assert_not_called()


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
        self.assertIn("optuna", installed)
        self.assertGreater(
            len(colab._RUNTIME_PACKAGES.full), len(colab._RUNTIME_PACKAGES.prepared)
        )


class BundlePrewarmTests(unittest.TestCase):
    """The local bundle build must overlap the remote deps stage."""

    def tearDown(self) -> None:
        colab.drain_local_bundle_prewarm()

    def test_build_runs_while_the_remote_deps_stage_is_still_going(self):
        entered = threading.Event()
        release = threading.Event()
        build_calls: list[dict] = []
        overlap: list[bool] = []

        def fake_build(**request):
            build_calls.append(request)
            entered.set()
            release.wait(timeout=10)
            return [Path("/tmp/worker_1_baseline.pkl.gz")]

        def fake_deps(**_kwargs):
            """Stands in for the remote deps stage; records whether it overlapped."""
            prewarm = colab._BUNDLE_PREWARM
            overlap.append(prewarm is not None and prewarm.thread.is_alive())

        with mock.patch.object(colab, "_build_local_training_bundles", fake_build), \
             mock.patch.object(colab, "install_deps", fake_deps):
            colab.start_local_bundle_prewarm(
                profiles=["baseline"], model=None, sample=None
            )
            self.assertTrue(entered.wait(timeout=10), "the local build never started")
            colab.install_deps(minimal_runtime=True)
            release.set()
            built = colab._take_prewarmed_bundles(
                profiles=["baseline"], model=None, sample=None
            )

        self.assertEqual(
            overlap, [True], "the local build was not running during the install"
        )
        self.assertEqual(
            build_calls, [{"profiles": ["baseline"], "model": None, "sample": None}]
        )
        self.assertEqual(built, [Path("/tmp/worker_1_baseline.pkl.gz")])

    def test_matching_request_reuses_the_concurrent_build(self):
        release = threading.Event()
        calls: list[dict] = []

        def fake(**request):
            calls.append(request)
            release.wait(timeout=10)
            return [Path("/tmp/worker_1.pkl.gz")]

        with mock.patch.object(colab, "_build_local_training_bundles", fake):
            colab.start_local_bundle_prewarm(profiles=["p"], model="m", sample=7)
            release.set()
            result = colab._take_prewarmed_bundles(profiles=["p"], model="m", sample=7)
        self.assertEqual(result, [Path("/tmp/worker_1.pkl.gz")])
        self.assertEqual(len(calls), 1, "the build must not be repeated")

    def test_prepare_helper_consumes_a_matching_prewarm(self):
        """The real builder must short-circuit on the in-flight build."""
        sentinel = [Path("/tmp/worker_1.pkl.gz")]
        with mock.patch.object(
            colab, "_take_prewarmed_bundles", return_value=sentinel
        ) as take, mock.patch.object(colab, "_validation_input_path") as inputs:
            result = colab._prepare_local_training_bundles(
                profiles=["p"], model=None, sample=None
            )
        self.assertEqual(result, sentinel)
        take.assert_called_once_with(
            profiles=["p"], model=None, sample=None, payload="full"
        )
        # No dataset resolution and no bundle subprocess: the build was reused.
        inputs.assert_not_called()

    def test_prewarm_thread_builds_once_without_joining_itself(self):
        """Regression guard for a real defect the mocked tests hid.

        The prewarm thread runs the real builder, so if the builder consulted
        the prewarm lookup the thread would join itself, die with
        ``cannot join current thread``, and the main thread would then rebuild
        serially — same output, no overlap, and no test that only counts builds
        would notice.  This drives the REAL builder (stubbing its subprocess and
        manifest reader) and asserts the build ran on the prewarm thread, not
        on the caller's.
        """
        manifest = mock.Mock(
            n_df=58_529, n_payload=94_124, n_pos=44_690, n_neg=22_393,
            sha256="0" * 64,
        )
        built_on: list[str] = []
        original_build = colab._build_local_training_bundles

        def spy(**request):
            built_on.append(threading.current_thread().name)
            return original_build(**request)

        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "dataset.csv"
            dataset.write_text("product_id\n1\n")
            with mock.patch.object(colab, "RESULTS", Path(temporary)), \
                 mock.patch.object(
                     colab, "_validation_input_path", return_value=dataset
                 ), \
                 mock.patch.object(colab.subprocess, "run") as build, \
                 mock.patch(
                     "training.prepared_bundle.load_prepared_bundle",
                     return_value=(manifest, None),
                 ), \
                 mock.patch.object(colab, "_build_local_training_bundles", spy):
                colab.start_local_bundle_prewarm(
                    profiles=["baseline"], model=None, sample=None
                )
                bundles = colab._prepare_local_training_bundles(
                    profiles=["baseline"], model=None, sample=None
                )
                self.assertIsNone(
                    colab._BUNDLE_PREWARM, "the matching prewarm must be consumed"
                )

        self.assertEqual(build.call_count, 1, "the bundle must be built exactly once")
        self.assertEqual(len(bundles), 1)
        self.assertTrue(bundles[0].name.startswith("worker_1_"))
        self.assertEqual(len(built_on), 1, "the build must run exactly once")
        self.assertNotEqual(
            built_on[0], threading.current_thread().name,
            "the build ran on the caller's thread: the prewarm thread never "
            "reached the builder (it joined itself)",
        )
        command = build.call_args.args[0]
        self.assertIn("--prepare-bundle", command)
        # No training: the bundle-writer path must be what ran.
        self.assertIn("--no-plot", command)

    def test_different_request_is_reported_and_not_reused(self):
        release = threading.Event()

        def fake(**request):
            release.wait(timeout=10)
            return [Path("/tmp/worker_1.pkl.gz")]

        captured = _Capture()
        with mock.patch.object(colab, "_build_local_training_bundles", fake), \
             mock.patch("sys.stdout", captured):
            colab.start_local_bundle_prewarm(profiles=["p"], model=None, sample=None)
            release.set()
            result = colab._take_prewarmed_bundles(
                profiles=["other"], model=None, sample=None
            )
        self.assertIsNone(result)
        self.assertIn("different request", captured.text())

    def test_failure_surfaces_in_the_owning_caller(self):
        def fake(**request):
            raise RuntimeError("bundle inputs are missing")

        with mock.patch.object(colab, "_build_local_training_bundles", fake):
            colab.start_local_bundle_prewarm(profiles=["p"], model=None, sample=None)
            with self.assertRaises(RuntimeError) as caught:
                colab._take_prewarmed_bundles(profiles=["p"], model=None, sample=None)
        self.assertIn("bundle inputs are missing", str(caught.exception))

    def test_drain_waits_for_an_abandoned_build(self):
        """A lane failing before run_train must not leave the log open."""
        release = threading.Event()
        finished = threading.Event()

        def fake(**request):
            release.wait(timeout=10)
            finished.set()
            return []

        with mock.patch.object(colab, "_build_local_training_bundles", fake):
            colab.start_local_bundle_prewarm(profiles=["p"], model=None, sample=None)
            release.set()
            colab.drain_local_bundle_prewarm()
            self.assertTrue(finished.is_set())


class LaneBundleRequestTests(unittest.TestCase):
    """The prewarm request must be the one run_train will actually make."""

    def _args(self, what: str, **overrides):
        base = {
            "what": what,
            "workers": 3,
            "sample": 111,
            "model": "minilm_l6",
            "masking_profile": colab._MASKING_PROFILE,
        }
        return mock.Mock(**{**base, **overrides})

    def test_train_lane_uses_cli_workers_and_sample(self):
        request = colab._lane_bundle_request(self._args("train"))
        self.assertEqual(len(request["profiles"]), 3)
        self.assertEqual(request["sample"], 111)
        self.assertEqual(request["model"], "minilm_l6")

    def test_dual_train_lane_uses_two_workers(self):
        request = colab._lane_bundle_request(self._args("dual-train"))
        self.assertEqual(len(request["profiles"]), 2)
        self.assertEqual(request["sample"], 111)

    def test_smoke_lane_uses_the_smoke_sample_and_workers(self):
        request = colab._lane_bundle_request(self._args("smoke"))
        self.assertEqual(len(request["profiles"]), colab._SMOKE_WORKERS)
        self.assertEqual(request["sample"], colab._SMOKE_SAMPLE)

    def test_lanes_without_local_bundles_are_not_prewarmed(self):
        for what in ("sims", "mixed", "hpo", "stop"):
            self.assertIsNone(colab._lane_bundle_request(self._args(what)))


class UploadReuseTests(unittest.TestCase):
    """Unchanged inputs the VM already has must not be uploaded again."""

    def _fixture(self, temporary: str):
        root = Path(temporary)
        paths = {}
        for key, name in (("source", "source.csv"), ("training", "train.csv")):
            path = root / name
            path.write_text(f"{key} rows\n")
            paths[key] = path
        paths["sample"] = paths["source"]
        validation = mock.Mock(
            enabled=True,
            source_csv="s",
            input_csv="i",
            output_dir="final_inference",
        )

        def resolve(value: str) -> Path:
            return paths[{"s": "source", "i": "sample"}.get(value, "training")]

        return root, paths, validation, resolve

    def test_disabled_validation_still_maps_every_key_without_contacting_the_vm(self):
        """The disabled lane must keep its staging paths and upload nothing."""
        with tempfile.TemporaryDirectory() as temporary:
            root, _paths, validation, resolve = self._fixture(temporary)
            validation.enabled = False
            with mock.patch.object(colab, "TRAIN_ROOT", root), \
                 mock.patch.object(colab, "_FINAL_INFERENCE", validation), \
                 mock.patch.object(colab, "_validation_input_path", side_effect=resolve), \
                 mock.patch.object(colab, "run_colab_exec_stream") as stream, \
                 mock.patch.object(colab, "run_colab_exec_capture") as capture, \
                 mock.patch.object(colab, "_upload_with_retries") as upload:
                remotes = colab._upload_validation_inputs("0915T000000000000Z")

            self.assertEqual(sorted(remotes), ["sample", "source", "training"])
            # source and sample are the same file, so they share one staging path
            self.assertEqual(remotes["sample"], remotes["source"])
            self.assertIn("/validation/source_", remotes["source"])
            self.assertIn("/validation/training_", remotes["training"])
            stream.assert_not_called()
            capture.assert_not_called()
            upload.assert_not_called()

    def test_verified_checkout_copy_is_reused_instead_of_uploaded(self):
        from core.manifest import sha256_file

        with tempfile.TemporaryDirectory() as temporary:
            root, paths, validation, resolve = self._fixture(temporary)
            source_digest = sha256_file(paths["source"])

            def capture(_session, script, timeout):  # noqa: ARG001
                return source_digest + "\n" if paths["source"].name in script else "\n"

            with mock.patch.object(colab, "TRAIN_ROOT", root), \
                 mock.patch.object(colab, "_FINAL_INFERENCE", validation), \
                 mock.patch.object(colab, "_validation_input_path", side_effect=resolve), \
                 mock.patch.object(colab, "run_colab_exec_stream"), \
                 mock.patch.object(colab, "run_colab_exec_capture", side_effect=capture), \
                 mock.patch.object(colab, "_upload_with_retries") as upload:
                remotes = colab._upload_validation_inputs("0915T000000000000Z")

            self.assertEqual(
                [call.args[0] for call in upload.call_args_list], [paths["training"]]
            )
            # source and sample are the same file and share the reused remote.
            self.assertEqual(remotes["source"], remotes["sample"])
            self.assertTrue(remotes["source"].endswith("/source.csv"))
            self.assertNotIn("/prepared_training/", remotes["source"])

    def test_mismatched_checkout_copy_is_uploaded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, paths, validation, resolve = self._fixture(temporary)
            with mock.patch.object(colab, "TRAIN_ROOT", root), \
                 mock.patch.object(colab, "_FINAL_INFERENCE", validation), \
                 mock.patch.object(colab, "_validation_input_path", side_effect=resolve), \
                 mock.patch.object(colab, "run_colab_exec_stream"), \
                 mock.patch.object(
                     colab, "run_colab_exec_capture", return_value="deadbeef\n"
                 ), \
                 mock.patch.object(colab, "_upload_with_retries") as upload:
                colab._upload_validation_inputs("0915T000000000000Z")

            self.assertCountEqual(
                [call.args[0] for call in upload.call_args_list],
                [paths["source"], paths["training"]],
            )

    def test_probe_failure_falls_back_to_uploading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, paths, validation, resolve = self._fixture(temporary)
            with mock.patch.object(colab, "TRAIN_ROOT", root), \
                 mock.patch.object(colab, "_FINAL_INFERENCE", validation), \
                 mock.patch.object(colab, "_validation_input_path", side_effect=resolve), \
                 mock.patch.object(colab, "run_colab_exec_stream"), \
                 mock.patch.object(
                     colab, "run_colab_exec_capture",
                     side_effect=RuntimeError("control channel gone"),
                 ), \
                 mock.patch.object(colab, "_upload_with_retries") as upload:
                colab._upload_validation_inputs("0915T000000000000Z")

            self.assertCountEqual(
                [call.args[0] for call in upload.call_args_list],
                [paths["source"], paths["training"]],
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


if __name__ == "__main__":
    unittest.main()
