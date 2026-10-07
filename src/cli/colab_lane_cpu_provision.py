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
        if export not in training_cfg().kaggle.export_csvs:
            raise ValueError(
                "bundle exports must be committed config kaggle.export_csvs "
                f"entries (no upload exists on this lane); got {export!r}")
        return export

    @timed
    def _sparse_checkout_paths(self, export: str) -> tuple[str, ...]:
        """The sparse checkout paths: base model bundle, smoke inputs, the export."""
        from core.common import resolve_model

        root = TRAIN_ROOT.resolve()
        checkout_paths = (
            Path(resolve_model(training_cfg().training.base_model)),
            TRAIN_ROOT / "data/prepared/smoke_200",
            TRAIN_ROOT / export,
        )
        return tuple(
            path.resolve().relative_to(root).as_posix() for path in checkout_paths)

    @timed
    def launch_prepare_script(self) -> str:
        """The VM-side prepare launcher: cohort remap + detached prepare process."""
        return self.bundle_head() + self._cohort_remap_segment() + self._prepare_process_segment()

    @timed
    def _cohort_remap_segment(self) -> str:
        """The remote head that remaps the cloned cohort export onto dataset.csv."""
        return f"""
import json, pathlib, shlex, shutil

# Cohort remap copies the committed export the sparse checkout carries onto
# dataset.csv — the kaggle lane's proven contract (cli.kaggle_lane). The
# clone is the only source of bytes this lane consumes.
chosen = pathlib.Path(root) / "@COHORT_EXPORT@"
if not chosen.is_file():
    raise SystemExit("committed cohort export absent from the checkout: @COHORT_EXPORT@")
if chosen.name != "dataset.csv":
    shutil.copy2(chosen, pathlib.Path(root) / "dataset.csv")
    print("[bundle-cpu] cohort remap: @COHORT_EXPORT@ -> dataset.csv", flush=True)
"""

    @timed
    def _prepare_process_segment(self) -> str:
        """The remote tail that runs training.prepare_all detached with a status file."""
        return f"""
base = pathlib.Path(root)
log_path, status_path = base / "{PREPARE_LOG_NAME}", base / "{PREPARE_STATUS_NAME}"
command_args = [sys.executable, "-u", "-m", "training.prepare_all"LAUNCH_ARGS_LIST]
command = " ".join(shlex.quote(part) for part in command_args)
wrapped = (
    "echo '[prepare-process] starting pid=$$'; "
    "echo '[prepare-process] resource snapshot before prepare'; free -h || true; "
    "timeout --signal=TERM --kill-after=60 {PREPARE_BUDGET_SECONDS} " + command + "; rc=$?; "
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
