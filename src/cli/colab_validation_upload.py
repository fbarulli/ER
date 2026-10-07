"""Validation-input upload + prewarm (split phase of cli.colab).

The GPU train lane's validation-input half: resolve one configured lane input
inside the repository, verify/reuse the VM's own checkout copy by content, and
push the validated component source/training/holdout CSVs — concurrently with
the VM dependency install.  Split from cli/colab.py (the kaggle_lane.py
owner-module pattern) exactly like colab_runtime / colab_result_sync /
colab_launch / colab_bundle_prewarm.

Collaborators still owned by cli.colab (config, ``REMOTE_ROOT``/``SESSION``/
``_BOOTSTRAP``, the legacy validation sources, transport, receipts constants)
are re-read through ``_hub()`` at call time, so the legacy
``from cli import colab`` monkeypatch surface keeps driving every phase and the
running colab identity never sees a stale second copy.  The in-flight upload
slot stays on the hub (``_VALIDATION_UPLOAD_PREWARM``) for the same reason.
"""
from __future__ import annotations

import functools
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from core.manifest import sha256_file


def _hub():
    """The RUNNING cli.colab module (never a second import copy)."""
    hub = sys.modules.get("__colab_runtime_self__")
    if hub is not None:
        return hub
    import cli.colab as surface

    return surface


def _timed_colab(kind: str):
    """Lazy step-timing shim: cli.colab owns ``_timed_colab`` at call time."""
    def decorate(function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            return _hub()._timed_colab(kind)(function)(*args, **kwargs)
        return wrapped
    return decorate


def _validation_input_path(configured_value: str) -> Path:
    """Resolve one configured lane input artifact inside the repository.

    Both callers pass the TRAINING dataset today; the scored-pair
    final-inference population is no longer a config literal — it is the
    SSOT final_validation binding and is never resolved here.
    """
    configured = Path(configured_value)
    root = _hub().TRAIN_ROOT
    source = configured if configured.is_absolute() else root / configured
    source = source.resolve()
    if not source.is_relative_to(root.resolve()):
        raise ValueError(
            "the configured lane input CSV must stay inside the repository"
        )
    if _hub()._FINAL_INFERENCE.enabled and not source.is_file():
        raise FileNotFoundError(f"configured final-inference CSV is missing: {source}")
    return source


def _remote_checkout_copy(source: Path) -> str | None:
    """Return the VM's own checkout copy of `source` when its bytes match.

    `prepare_remote_layout` has already put the configured branch at
    REMOTE_ROOT, so any validation input that is committed to the branch is
    on the VM before the first upload.  Re-uploading it costs the launcher
    seconds per run (measured: 14.7 s for the 46 MB deduped source CSV on the
    2026-09-15 T4 launch) and transfers bytes the VM already has.

    The reuse is content-verified, not assumed: the remote file is hashed and
    must equal the local digest, so a locally modified or absent input falls
    back to a normal upload instead of silently training on the wrong rows.
    """
    surface = _hub()
    relative = source.resolve().relative_to(surface.TRAIN_ROOT.resolve()).as_posix()
    remote = f"{surface.REMOTE_ROOT}/{relative}"
    probe = surface._BOOTSTRAP + f"""
import hashlib, pathlib
path = pathlib.Path({remote!r})
if path.is_file():
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    print(digest.hexdigest(), flush=True)
else:
    print("", flush=True)
"""
    try:
        reported = surface.run_colab_exec_capture(surface.SESSION, probe, timeout=120).strip()
    except Exception as exc:
        print(
            surface._stamp(),
            f"[upload] could not verify the VM checkout copy of {source.name} "
            f"({exc}); uploading instead",
            flush=True,
        )
        return None
    if reported and reported == sha256_file(source):
        return remote
    return None


_RUN_STAMP_FORMAT = "%m%dT%H%M%S%fZ"
# How long the prewarmed upload waits for the session gate before giving up and
# letting the lane upload serially.  Generous enough for a slow GPU allocation,
# finite so a lane that never releases the gate cannot hang teardown.
_PREWARM_GATE_TIMEOUT_SECONDS = 900
# How long starting a new prewarm waits for a previous one to retire.  Bounded
# so that starting a lane never depends on an earlier lane's transfer.
_PREWARM_RETIRE_SECONDS = 30


class _ValidationUploadPrewarm:
    """One lane's validation uploads, travelling during the VM dependency install.

    The validation CSVs are read from this repository and pushed to the VM;
    nothing the VM does produces them, so the transfer can overlap the install
    instead of following it.  Measured before the overlap: 36.79 s of uploads
    strictly after a 42.96 s dependency install.

    The thread waits for :func:`ensure_session` before its first transfer.
    Started earlier, it raced session provisioning, failed on a VM that did
    not exist yet, and left the whole payload to be re-uploaded serially at
    the join -- which is the cost the overlap exists to remove.
    """

    def __init__(self, stamp: str) -> None:
        self.stamp = stamp
        self.remote_paths: dict[str, str] | None = None
        self.error: BaseException | None = None
        self.session_ready = threading.Event()
        self.thread = threading.Thread(target=self._upload, daemon=True)

    def _upload(self) -> None:
        # A GPU allocation can take minutes; the upload is worthless until the
        # control channel answers, so wait rather than transfer into a void.
        # Bounded: if the gate is never released -- a lane that failed before
        # releasing it -- this thread must still end, because the launcher's
        # drain joins it and an unbounded wait would hang teardown forever.
        if not self.session_ready.wait(_PREWARM_GATE_TIMEOUT_SECONDS):
            print(
                _hub()._stamp(),
                f"[upload] session was not ready within "
                f"{_PREWARM_GATE_TIMEOUT_SECONDS}s; uploading serially instead",
                flush=True,
            )
            return
        try:
            self.remote_paths = _hub()._perform_validation_upload(self.stamp)
        except BaseException as exc:  # handed back to the owning run, or fallen back from
            self.error = exc
            print(
                _hub()._stamp(),
                f"[upload] concurrent validation upload failed: {exc!r}", flush=True
            )

    def join(self) -> dict[str, str]:
        self.thread.join()
        if self.error is not None:
            raise self.error
        if self.remote_paths is None:
            raise RuntimeError("the concurrent validation upload returned no paths")
        return self.remote_paths


@_timed_colab("step")
def start_validation_upload_prewarm() -> str:
    """Begin this lane's validation uploads; returns the run id they belong to."""
    surface = _hub()
    if surface._VALIDATION_UPLOAD_PREWARM is not None:
        # Running a second prewarm over an unfinished one would race two
        # threads onto the same remote paths, and the first thread would never
        # be joined.  Release and retire it before starting its replacement --
        surface._VALIDATION_UPLOAD_PREWARM.session_ready.set()
        # but never block indefinitely: starting a lane must stay instant, so
        # a previous upload that will not finish is abandoned to its own
        # bounded wait rather than stalling this one.
        surface._VALIDATION_UPLOAD_PREWARM.thread.join(timeout=_PREWARM_RETIRE_SECONDS)
    stamp = datetime.now(timezone.utc).strftime(_RUN_STAMP_FORMAT)
    prewarm = _ValidationUploadPrewarm(stamp)
    surface._VALIDATION_UPLOAD_PREWARM = prewarm
    prewarm.thread.start()
    print(
        surface._stamp(),
        f"[upload] sending validation inputs for run {stamp} concurrently with "
        "the VM dependency install",
        flush=True,
    )
    return stamp


def drain_validation_upload_prewarm() -> None:
    """Never leave an upload thread writing while the live log closes."""
    surface = _hub()
    prewarm, surface._VALIDATION_UPLOAD_PREWARM = surface._VALIDATION_UPLOAD_PREWARM, None
    if prewarm is not None and prewarm.thread.is_alive():
        print(surface._stamp(), "[upload] waiting for the concurrent upload to finish ...", flush=True)
        # The thread may still be blocked on the session gate; release it so a
        # drain at exit cannot hang forever on a VM that never came up.
        prewarm.session_ready.set()
        # Bounded: an upload to a dead VM can stall, and teardown must not wait
        # on it indefinitely.  Whatever lands is verified by the download.
        prewarm.thread.join(timeout=_PREWARM_RETIRE_SECONDS)
        if prewarm.thread.is_alive():
            print(
                surface._stamp(),
                "[upload] concurrent upload did not finish within "
                f"{_PREWARM_RETIRE_SECONDS}s; continuing with teardown",
                flush=True,
            )


@_timed_colab("step")
def release_validation_upload_prewarm() -> None:
    """Let the prewarmed upload start, now that the session can accept it."""
    prewarm = _hub()._VALIDATION_UPLOAD_PREWARM
    if prewarm is not None:
        prewarm.session_ready.set()


def _lane_run_stamp() -> str:
    """Adopt the prewarmed run identity so the uploads belong to this run."""
    prewarm = _hub()._VALIDATION_UPLOAD_PREWARM
    if prewarm is not None:
        return prewarm.stamp
    return datetime.now(timezone.utc).strftime(_RUN_STAMP_FORMAT)


def _upload_validation_inputs(run_id: str) -> dict[str, str]:
    """Validation inputs for one run: the in-flight transfer when it matches.

    The only entry point that consumes an upload prewarm, so the transfer it
    hands over runs once and the overlap in `start_validation_upload_prewarm`
    is real.
    """
    surface = _hub()
    if surface._VALIDATION_UPLOAD_PREWARM is not None:
        prewarm, surface._VALIDATION_UPLOAD_PREWARM = surface._VALIDATION_UPLOAD_PREWARM, None
        if prewarm.stamp == run_id:
            print(
                surface._stamp(),
                "[upload] joining the validation upload started before the VM "
                "setup",
                flush=True,
            )
            # By the time a lane joins, the session is up: release the gate so
            # the overlap actually happens instead of the thread waiting on a
            # signal that this join is itself blocking.
            prewarm.session_ready.set()
            try:
                return prewarm.join()
            except Exception as exc:
                print(
                    surface._stamp(),
                    f"[upload] concurrent validation upload failed ({exc!r}); "
                    "uploading serially instead",
                    flush=True,
                )
        else:
            print(
                surface._stamp(),
                f"[upload] concurrent upload belongs to run {prewarm.stamp}, not "
                f"{run_id}; uploading serially",
                flush=True,
            )
            # This prewarm no longer belongs to the caller, but it still owns
            # a thread waiting on the session gate.  Release and retire it
            # before proceeding so it cannot linger for the full gate timeout
            # (and so its one intended transfer is not silently abandoned).
            prewarm.session_ready.set()
            prewarm.thread.join(timeout=_PREWARM_RETIRE_SECONDS)
            if prewarm.thread.is_alive():
                print(
                    surface._stamp(),
                    "[upload] mismatched concurrent upload is still running; "
                    "continuing with the serial upload",
                    flush=True,
                )
    return surface._perform_validation_upload(run_id)


def _perform_validation_upload(run_id: str) -> dict[str, str]:
    """Transfer the validated component source, training listings, and holdout.

    The worker itself.  It deliberately does NOT consult the prewarm: it is
    what the prewarm thread runs, so looking the prewarm up here would make
    that thread join itself.
    """
    surface = _hub()
    sources = surface._legacy_validation_sources()
    remote_dir = f"{surface.REMOTE_ROOT}/prepared_training/{run_id}/validation"
    remotes: dict[str, str] = {}
    remote_by_source: dict[str, str] = {}
    if not surface._FINAL_INFERENCE.enabled:
        for key, source in sources.items():
            remotes[key] = remote_by_source.setdefault(
                str(source.resolve()), f"{remote_dir}/{key}_{source.name}"
            )
        return remotes
    surface.run_colab_exec_stream(
        surface.SESSION,
        surface._BOOTSTRAP
        + f"""
import pathlib
pathlib.Path({remote_dir!r}).mkdir(parents=True, exist_ok=True)
""",
        timeout=120,
        log_name="validation_input_dir",
        retry_safe=False,
    )
    for key, source in sources.items():
        source_key = str(source.resolve())
        if source_key in remote_by_source:
            remotes[key] = remote_by_source[source_key]
            print(
                surface._stamp(),
                f"[upload] validation {key}={source} reusing "
                f"{remote_by_source[source_key]}",
                flush=True,
            )
            continue
        checkout_copy = surface._remote_checkout_copy(source)
        if checkout_copy is not None:
            remote_by_source[source_key] = checkout_copy
            remotes[key] = checkout_copy
            print(
                surface._stamp(),
                f"[upload] validation {key}={source} reused the verified VM "
                f"checkout copy {checkout_copy} (sha256 matches; not uploaded)",
                flush=True,
            )
            continue
        remote = f"{remote_dir}/{key}_{source.name}"
        print(surface._stamp(), f"[upload] validation {key}={source} -> {remote}", flush=True)
        surface._upload_with_retries(
            source, remote, timeout=surface._RESULT_DOWNLOAD_TIMEOUT_SECONDS
        )
        remote_by_source[source_key] = remote
        remotes[key] = remote
    return remotes
