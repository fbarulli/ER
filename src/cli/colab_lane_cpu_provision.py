"""CPU lane provisioning + prepare-launch script emission (phase owner).

Split phase of cli/colab_lane.py (the kaggle_lane.py owner-class pattern):
this owner owns the committed-export lane's provisioning order
(check_colab_cli -> ensure_session -> sparse remote layout -> minimal deps)
and the byte-exact remote launcher text — the cohort remap onto dataset.csv
plus the detached, status-file-backed training.prepare_all process.  The
emitted scripts are joined from single-responsibility segments; collaborators
resolve at call time through the running colab identity
(``sys.modules["__colab_runtime_self__"]``) via the lane's dial-ins.
"""
from __future__ import annotations

import shlex
from pathlib import Path

from core.common import TRAIN_ROOT, training_cfg
from core.run_log import RunLogger
from training.prepare_all_trace import timed
from cli.colab_lane_contracts import PREPARE_BUDGET_SECONDS, PREPARE_LOG_NAME, PREPARE_STATUS_NAME

_LOG = RunLogger(__name__)


class ColabCPULaneProvision:
    """Provisioning order + prepare-launch script emission for the CPU lane."""

    @timed
    def provision(self, dataset_csv: Path | None = None) -> None:
        """cli.colab main()'s provisioning order for a full-runtime CPU lane."""
        surface = self.surface
        surface.check_colab_cli()
        surface.ensure_session()
        export = self._committed_export_name(dataset_csv)
        with _LOG.section("colab_lane.cpu_provision.runtime"):
            surface.prepare_remote_layout(
                minimal_runtime=True, sparse_paths=self._sparse_checkout_paths(export))
            surface.install_deps(minimal_runtime=True)

    @timed
    def _committed_export_name(self, dataset_csv: Path | None) -> str:
        """The committed config-listed export this lane checks out; fail loud otherwise."""
        chosen = dataset_csv if dataset_csv is not None else TRAIN_ROOT / "dataset.csv"
        export = chosen.name
        if export not in training_cfg().bundle_prep.export_csvs:
            raise ValueError(
                "bundle exports must be committed config bundle_prep.export_csvs "
                f"entries (no upload exists on this lane); got {export!r}")
        return export

    @timed
    def _sparse_checkout_paths(self, export: str) -> tuple[str, ...]:
        """The sparse checkout paths: base model bundle, smoke inputs, the export."""
        from core.common import resolve_model

        root = TRAIN_ROOT.resolve()
        checkout_paths = (
            Path(resolve_model(training_cfg().training.base_model)),
            TRAIN_ROOT / training_cfg().colab.data_bundle.setup_dir,
            TRAIN_ROOT / export,
        )
        return tuple(
            path.resolve().relative_to(root).as_posix() for path in checkout_paths)

    @timed
    def launch_prepare_script(self, *, launch_args: str, cohort_export: str) -> str:
        """The VM-side prepare launcher, fully rendered.

        Cohort remap + the detached prepare process (prepare_all chained to
        this lane's training-bundle stage). Every placeholder the emitted
        segments carry is filled HERE, beside the segments that own them, so a
        caller can never hand the VM a half-rendered script and this method's
        result is always compilable Python.
        """
        stage = " ".join(shlex.quote(part) for part in self.bundle_stage_command())
        return (self.bundle_head() + self._cohort_remap_segment()
                + self._prepare_process_segment()).replace(
                    "LAUNCH_ARGS_LIST", launch_args).replace(
                    "@COHORT_EXPORT@", cohort_export).replace(
                    "BUNDLE_STAGE_COMMAND", stage)

    @timed
    def _cohort_remap_segment(self) -> str:
        """The remote head that remaps the cloned cohort export onto dataset.csv."""
        return f"""
import json, pathlib, shlex, shutil

# Cohort remap copies the committed export the sparse checkout carries onto
# dataset.csv. The clone is the only source of bytes this lane consumes.
chosen = pathlib.Path(root) / "@COHORT_EXPORT@"
if not chosen.is_file():
    raise SystemExit("committed cohort export absent from the checkout: @COHORT_EXPORT@")
if chosen.name != "dataset.csv":
    shutil.copy2(chosen, pathlib.Path(root) / "dataset.csv")
    print("[bundle-cpu] cohort remap: @COHORT_EXPORT@ -> dataset.csv", flush=True)
"""

    @timed
    def _prepare_process_segment(self) -> str:
        """The remote tail that runs training.prepare_all, then this lane's
        training-bundle stage, detached with a status file.

        The two stages are ONE detached process under ONE budget, chained with
        ``&&``: the bundle stage runs exactly when prepare_all sealed the
        inputs it builds from, and a non-zero rc from either reaches the status
        file the lane's poll refuses on. The bundle stage command comes from the
        stage's own owner (colab_lane_cpu_bundles), so there is one spawn path
        and one stage command in the tree.
        """
        return f"""
base = pathlib.Path(root)
log_path, status_path = base / "{PREPARE_LOG_NAME}", base / "{PREPARE_STATUS_NAME}"
command_args = [sys.executable, "-u", "-m", "training.prepare_all"LAUNCH_ARGS_LIST]
command = " ".join(shlex.quote(part) for part in command_args)
stage = "BUNDLE_STAGE_COMMAND"
wrapped = (
    "echo '[prepare-process] starting pid=$$'; "
    "echo '[prepare-process] resource snapshot before prepare'; free -h || true; "
    "timeout --signal=TERM --kill-after=60 {PREPARE_BUDGET_SECONDS} bash -c "
    + shlex.quote(command + " && " + stage) + "; rc=$?; "
    "echo '[prepare-process] exited rc='$rc; "
    "echo '[prepare-process] resource snapshot after prepare'; free -h || true; "
    "printf '%s\\\\n' \\"$rc\\" > " + shlex.quote(str(status_path)) + "; exit $rc"
)
env = {{**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(pathlib.Path(root) / "src")}}
with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
    child = subprocess.Popen(["/bin/bash", "-lc", wrapped], cwd=root, env=env,
        stdin=subprocess.DEVNULL, stdout=log_file, stderr=subprocess.STDOUT,
        start_new_session=True)
print("[prepare] pid=%d" % child.pid, flush=True)
"""
