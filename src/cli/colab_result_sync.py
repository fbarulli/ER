"""Result transfer + verification (split phase D of cli.colab).

The result half of the GPU lane: the manifest-backed remote archive build, its
safe extraction and hash verification, the incremental best-checkpoint syncer
that keeps the laptop warm while training runs, and the authoritative
post-run download.  Split from cli/colab.py (the kaggle_lane.py owner-module
pattern) exactly like colab_runtime/colab_retention/colab_self_watch.

Collaborators still owned by cli.colab (transport dial-ins, config constants,
receipts, the bootstrap preamble) resolve at call time through the running
colab module (``sys.modules["__colab_runtime_self__"]``), never a direct
second import, so the legacy ``from cli import colab`` monkeypatch surface —
``TRAINING_RESULTS``, ``_read_remote_text``, ``_list_remote``,
``_download_one_remote_file``, ``_verify_result_bundle`` — keeps driving every
phase and the ``python -m cli.colab`` runtime identity never sees a stale
copy.
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from core.archive_reader import tar_archive
from core.manifest import sha256_file
from core.schemas import ResultBundleManifest


def _hub():
    """The RUNNING cli.colab module (never a second import copy)."""
    return sys.modules.get("__colab_runtime_self__") or sys.modules["cli.colab"]


def _prepare_remote_result_archive(remote_base: str, workers: int) -> str:
    """Build one manifest-backed archive on the VM before transfer."""
    surface = _hub()
    archive_name = surface._RESULT_ARCHIVE_NAME
    manifest_name = surface._RESULT_MANIFEST_NAME
    excluded_dirs = surface._RESULT_DOWNLOAD_EXCLUDED_DIRS
    archive_path = f"{remote_base}/{archive_name}"
    script = surface._BOOTSTRAP + f"""
import hashlib, json, pathlib
from core.archive_reader import tar_archive

base = pathlib.Path({remote_base!r})
archive_path = base / {archive_name!r}
manifest_path = base / {manifest_name!r}
excluded_dirs = set({excluded_dirs!r})

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

included = []
excluded = []

# ``_checkpoints`` is excluded because a run writes one large checkpoint per
# evaluation, but the checkpoint the trainer selected must still reach the
# laptop.  trainer_state.json records best_model_checkpoint and best_metric --
# the same choice training restores at the end -- so keep that one directory
# per worker and drop the rest.
def _best_checkpoint(worker_root):
    best = None
    pattern = "_checkpoints/**/checkpoint-*/trainer_state.json"
    for state_path in sorted(worker_root.glob(pattern)):
        try:
            state = json.loads(state_path.read_text())
        except (OSError, ValueError):
            continue
        recorded = state.get("best_model_checkpoint")
        if not recorded:
            continue
        selected = state_path.parent.parent / pathlib.Path(str(recorded)).name
        if not selected.is_dir():
            continue
        metric = state.get("best_metric")
        rank = (
            float(metric) if metric is not None else float("-inf"),
            int(state.get("global_step") or 0),
        )
        if best is None or rank > best[0]:
            best = (rank, selected)
    return best[1] if best is not None else None

for worker in range(1, {workers + 1}):
    worker_root = base / f"worker_{{worker}}"
    if not worker_root.is_dir():
        raise FileNotFoundError(f"missing remote worker directory: {{worker_root}}")
    best_checkpoint = _best_checkpoint(worker_root)
    for path in sorted(worker_root.rglob("*")):
        if path.is_symlink():
            excluded.append({{
                "worker": worker,
                "path": path.relative_to(worker_root).as_posix(),
                "reason": "symlink_not_transferred",
            }})
            continue
        if not path.is_file():
            continue
        relative = path.relative_to(worker_root)
        if best_checkpoint is not None:
            try:
                path.relative_to(best_checkpoint)
            except ValueError:
                pass
            else:
                included.append({{
                    "worker": worker,
                    "path": relative.as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256(path),
                }})
                continue
        blocked = next((part for part in relative.parts if part in excluded_dirs), None)
        if blocked is not None:
            excluded.append({{
                "worker": worker,
                "path": relative.as_posix(),
                "reason": "configured_directory:" + blocked,
            }})
            continue
        included.append({{
            "worker": worker,
            "path": relative.as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256(path),
        }})

manifest = {{
    "schema_version": "1",
    "run_id": base.name.removeprefix("concurrent_train_"),
    "workers": {workers},
    "included": included,
    "excluded": excluded,
}}
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
archive_path.unlink(missing_ok=True)
with tar_archive(archive_path, "w") as archive:
    for entry in included:
        worker_root = base / f"worker_{{entry['worker']}}"
        source = worker_root / entry["path"]
        archive.add(source, arcname=f"worker_{{entry['worker']}}/{{entry['path']}}")
    archive.add(manifest_path, arcname={manifest_name!r})
print("[result-archive] included={{}} excluded={{}} archive_bytes={{}}".format(
    len(included), len(excluded), archive_path.stat().st_size
), flush=True)
"""
    print(
        surface._stamp(),
        f"[download] preparing one remote result archive for {workers} worker(s): "
        f"{archive_path}",
        flush=True,
    )
    surface.run_colab_exec_stream(
        surface.SESSION,
        script,
        timeout=surface._RESULT_DOWNLOAD_TIMEOUT_SECONDS,
        log_name="result_archive",
    )
    return archive_path


def _verify_result_bundle(root: Path, run_id: str, workers: int) -> ResultBundleManifest:
    """Validate manifest coverage, paths, sizes, and hashes after extraction."""
    manifest_path = root / _hub()._RESULT_MANIFEST_NAME
    if not manifest_path.is_file():
        raise RuntimeError(f"result archive is missing its manifest: {manifest_path}")
    manifest = ResultBundleManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if manifest.run_id != run_id or manifest.workers != workers:
        raise RuntimeError(
            f"result manifest identity mismatch: run_id={manifest.run_id!r}, "
            f"workers={manifest.workers}; expected {run_id!r}, {workers}"
        )
    expected: set[str] = set()
    for entry in manifest.included:
        relative = Path(entry.path)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe result manifest path: {entry.path!r}")
        key = Path(f"worker_{entry.worker}") / relative
        key_text = key.as_posix()
        if key_text in expected:
            raise RuntimeError(f"duplicate result manifest path: {key_text}")
        expected.add(key_text)
        actual = root / key
        if not actual.is_file():
            raise RuntimeError(f"result archive missing manifest file: {key_text}")
        size = actual.stat().st_size
        if size != entry.size:
            raise RuntimeError(
                f"result size mismatch for {key_text}: {size} != {entry.size}"
            )
        digest = sha256_file(actual)
        if digest != entry.sha256:
            raise RuntimeError(f"result SHA-256 mismatch for {key_text}")
    actual_paths = {
        path.relative_to(root).as_posix()
        for worker_root in sorted(root.glob("worker_*"))
        if worker_root.is_dir()
        for path in worker_root.rglob("*")
        if path.is_file()
    }
    if actual_paths != expected:
        missing = sorted(expected - actual_paths)
        unexpected = sorted(actual_paths - expected)
        raise RuntimeError(
            f"result archive coverage mismatch: missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}"
        )
    return manifest


def _extract_result_archive(
    archive_path: Path, local_base: Path, run_id: str, workers: int
) -> ResultBundleManifest:
    """Safely extract and atomically replace the worker result directories."""
    surface = _hub()
    manifest_name = surface._RESULT_MANIFEST_NAME
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}-result-", dir=local_base.parent))
    try:
        with tar_archive(archive_path) as archive:
            members = archive.getmembers()
            for member in members:
                relative = Path(member.name)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or not (member.isfile() or member.isdir())
                ):
                    raise RuntimeError(f"unsafe result archive member: {member.name!r}")
            archive.extractall(temporary)
        manifest = surface._verify_result_bundle(temporary, run_id, workers)
        for worker in range(1, workers + 1):
            source = temporary / f"worker_{worker}"
            target = local_base / source.name
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
            shutil.move(str(source), str(target))
        local_manifest = local_base / manifest_name
        local_manifest.unlink(missing_ok=True)
        shutil.move(str(temporary / manifest_name), str(local_manifest))
        return manifest
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


class _IncrementalResultSync:
    """Keep the newest *best* checkpoint on the laptop while training runs.

    The end-of-run download is one archive transferred after the whole
    lifecycle (train -> inference -> archive download) has collapsed to a single
    moment, so the multi-GB checkpoint set is dead time at the end.  Training
    writes a new ``checkpoint-<step>`` directory steadily, and each directory
    is immutable once written, so the best one can be held locally as it
    appears.

    Retention is *latest-if-better, overwrite*: each worker keeps exactly one
    local directory, ``latest_best``, and a remote checkpoint replaces it only
    when its score strictly beats the score already held.  "Best" is dev
    average precision when the run reports it, else lowest train loss, both
    read from the ``live_status.json`` the trainer already publishes.

    This is a pure latency optimisation and never a correctness dependency:

    * a checkpoint is copied only once its ``checkpoint_manifest.json`` exists,
      which the trainer writes last, so a directory still being written is
      never captured half-finished;
    * everything it fetches is re-fetched and hash-verified by the
      authoritative :func:`download_verified_training_results` pass, which
      remains the only source of truth for a complete run;
    * every failure is swallowed and logged; a flaky sync must not fail a run
      whose data still arrives by the normal path.
    """

    def __init__(self, remote_base: str, run_id: str, *, workers: int) -> None:
        self.remote_base = remote_base
        self.run_id = run_id
        self.workers = workers
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # worker -> (score, remote checkpoint dir) currently held locally.
        self._held: dict[int, tuple[float, str]] = {}
        self._synced_bytes = 0

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop, name=f"result-sync-{self.run_id}", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            # Long enough for an in-flight checkpoint to land, short enough that
            # a stuck sync cannot delay teardown: whatever it misses is still
            # covered by the authoritative download.
            self._thread.join(timeout=120)

    def synced_bytes(self) -> int:
        return self._synced_bytes

    def _loop(self) -> None:
        surface = _hub()
        while not self._stop.wait(surface._INCREMENTAL_SYNC_SECONDS):
            try:
                self._pass()
            except BaseException as exc:  # never fail the run from the syncer
                print(
                    surface._stamp(),
                    f"[result-sync] pass failed ({type(exc).__name__}: {exc}); "
                    "final download still covers every file",
                    flush=True,
                )

    def _pass(self) -> None:
        surface = _hub()
        for worker in range(1, self.workers + 1):
            try:
                self._sync_worker(worker)
            except BaseException as exc:
                # A listing failure, a timeout, or a download error all land
                # here: one pass must never escape into the run.
                print(
                    surface._stamp(),
                    f"[result-sync] worker {worker} pass skipped "
                    f"({type(exc).__name__}: {exc}); final download covers it",
                    flush=True,
                )

    def _sync_worker(self, worker: int) -> None:
        """Copy this worker's best-so-far checkpoint, if it beats the local one.

        The trainer's ``live_status.json`` heartbeat names the checkpoint it
        just finished, so one small read replaces walking the run's tree.  That
        walk is what timed out on the T4 lane: listing the whole checkpoint,
        log, and wandb tree on a 30 s cadence cannot finish inside the exec
        budget while training is competing for the same control channel.
        """
        surface = _hub()
        heartbeat = self._heartbeat(worker)
        if heartbeat is None:
            return
        step, score = heartbeat
        directory = self._checkpoint_directory(worker, step)
        if directory is None:
            return
        if worker in self._held:
            cached_score, cached_dir = self._held[worker]
        else:
            cached_score, cached_dir = self._best_local(worker)
            self._held[worker] = (cached_score, cached_dir)
        if directory == cached_dir or score <= cached_score:
            return
        files = surface._list_remote(directory, max_depth=surface._CHECKPOINT_LISTING_DEPTH)
        if not files:
            return
        if not any(
            Path(name).name == surface._CHECKPOINT_MANIFEST_NAME for name in files
        ):
            # The manifest is written last, so its absence means this
            # checkpoint is still being written and must not be copied yet.
            return
        destination = (
            surface.TRAINING_RESULTS / self.run_id / f"worker_{worker}" / "latest_best"
        )
        staging = destination.with_name("latest_best.staging")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)
        transferred = 0
        checkpoint_name = Path(directory).name
        # Path of the checkpoint relative to the worker root, so the marker
        # records where it actually lives rather than just its leaf name.
        checkpoint_rel = (
            Path(directory)
            .relative_to(Path(f"{self.remote_base}/worker_{worker}"))
            .as_posix()
        )
        for name in sorted(files):
            target = staging / checkpoint_name / Path(name).relative_to(directory)
            target.parent.mkdir(parents=True, exist_ok=True)
            surface._download_one_remote_file(name, target)
            transferred += target.stat().st_size
        (staging / surface._LATEST_BEST_MARKER).write_text(
            json.dumps(
                {
                    "checkpoint": directory,
                    "relative": checkpoint_rel,
                    "score": score,
                    "step": step,
                    "worker": worker,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        # Swap whole directories so the laptop never holds a half checkpoint.
        superseded = destination.with_name(f"latest_best.superseded.{os.getpid()}")
        if destination.exists():
            os.replace(destination, superseded)
        os.replace(staging, destination)
        shutil.rmtree(superseded, ignore_errors=True)
        self._held[worker] = (score, directory)
        self._synced_bytes += transferred
        print(
            surface._stamp(),
            f"[result-sync] worker {worker} kept {checkpoint_name} "
            f"(step {step}, score={score:.4f}, "
            f"{surface._format_bytes(transferred)}) as latest_best",
            flush=True,
        )

    def _heartbeat(self, worker: int) -> tuple[int, float] | None:
        """The step and score the trainer last reported, if it reported one."""
        remote = f"{self.remote_base}/worker_{worker}/live_status.json"
        try:
            payload = json.loads(_hub()._read_remote_text(remote))
        except BaseException:
            return None
        step = payload.get("step")
        if step is None:
            return None
        average_precision = payload.get("dev_average_precision")
        if average_precision is not None:
            return (int(step), float(average_precision))
        loss = payload.get("train_loss")
        if loss is not None:
            return (int(step), -float(loss))
        return None

    def _checkpoint_directory(self, worker: int, step: int) -> str | None:
        """Locate ``checkpoint-<step>`` without walking the whole run tree.

        The trainer nests it under model and run folders whose names are not
        known here, so the two levels above it are matched by glob.  This is a
        bounded probe of one small subtree, not a walk of the whole run: only
        ``_checkpoints/<model>/<run>_f0`` is examined.
        """
        surface = _hub()
        root = f"{self.remote_base}/worker_{worker}/{surface._CHECKPOINT_ROOT_NAME}"
        try:
            matches = surface._list_remote(root, max_depth=surface._CHECKPOINT_LISTING_DEPTH)
        except BaseException:
            return None
        wanted = f"checkpoint-{step}"
        # The listing reports files, so the checkpoint is the directory
        # *containing* them: matching on the file's own name would only ever
        # compare against names like checkpoint_manifest.json and find nothing.
        for name in matches:
            candidate = Path(name)
            if candidate.parent.name == wanted:
                return str(candidate.parent)
        return None

    def _best_local(self, worker: int) -> tuple[float, str | None]:
        """Score and checkpoint name already held locally, if any."""
        surface = _hub()
        pointer = surface.TRAINING_RESULTS / self.run_id / f"worker_{worker}" / "latest_best"
        marker = pointer / surface._LATEST_BEST_MARKER
        if not marker.is_file():
            return (float("-inf"), None)
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            return (float(payload["score"]), str(payload["checkpoint"]))
        except (OSError, ValueError, KeyError, TypeError):
            return (float("-inf"), None)


def _download_one_remote_file(remote: str, local: Path) -> None:
    """Fetch one remote file through the module's Colab CLI wrapper.

    Indirection that keeps the incremental syncer patchable: inside cli.colab
    the name ``colab`` is both the module and the CLI wrapper function, so a
    test cannot patch the wrapper without shadowing the module.
    """
    surface = _hub()
    surface.colab(
        "download", "-s", surface.SESSION, remote, str(local),
        timeout=surface._RESULT_DOWNLOAD_TIMEOUT_SECONDS,
    )


def _read_remote_text(remote: str) -> str:
    """Read one small remote file through the shared stdin-exec channel."""
    surface = _hub()
    script = (
        "import base64, pathlib\n"
        f"p = pathlib.Path({remote!r})\n"
        "print('@@TEXT@@' + base64.b64encode("
        "p.read_bytes() if p.is_file() else b'').decode('ascii'))\n"
    )
    proc = subprocess.Popen(
        surface._colab_command("exec", "-s", surface.SESSION, "--timeout", "60"),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = proc.communicate(script, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"remote read failed: {err[-200:]}")
    for line in out.splitlines():
        if line.startswith("@@TEXT@@"):
            payload = line[len("@@TEXT@@"):]
            if not payload:
                raise RuntimeError(f"remote file is missing: {remote}")
            return base64.b64decode(payload).decode("utf-8")
    raise RuntimeError("remote read returned no marker")


def download_verified_training_results(
    remote_base: str, workers: int, *, smoke: bool = False
) -> None:
    """Transfer one manifest-backed result archive before VM teardown."""
    surface = _hub()
    run_id = Path(remote_base).name.removeprefix("concurrent_train_")
    # Smoke runs join a separate retention lane (smoke_ prefix): the local
    # root must already carry the prefix BEFORE extraction so the overwrite
    # sweep below governs only this lane.
    local_base = surface.TRAINING_RESULTS / (f"smoke_{run_id}" if smoke else run_id)
    local_base.mkdir(parents=True, exist_ok=True)
    surface._result_event(run_id, "download", "started", workers=workers)
    remote_archive = surface._prepare_remote_result_archive(remote_base, workers)
    local_archive = local_base / surface._RESULT_ARCHIVE_NAME
    surface._download_file_with_visibility(
        remote=remote_archive,
        local=local_archive,
        worker=None,
        index=1,
        total=1,
        run_id=run_id,
    )
    manifest = surface._extract_result_archive(local_archive, local_base, run_id, workers)
    surface._result_event(
        run_id,
        "download",
        "completed",
        workers=workers,
        files=len(manifest.included),
        excluded_files=len(manifest.excluded),
        archive=str(local_archive),
        destination=str(local_base),
    )
    print(
        surface._stamp(),
        f"[download] verified result archive -> {local_base} "
        f"({len(manifest.included)} included, {len(manifest.excluded)} excluded)",
        flush=True,
    )
    from model_tracks.run_retention import publish_training_run, replace_smoke
    if smoke:
        replace_smoke(local_base)
        print(surface._stamp(), "[retention] smoke: newest result is the only local smoke run", flush=True)
    else:
        receipt = publish_training_run(local_base)
        print(surface._stamp(), f"[retention] dvc published run={Path(local_base).name} "
              f"pruned={len(receipt.pruned)} in {receipt.seconds:.1f}s", flush=True)
