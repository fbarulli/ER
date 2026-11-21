"""CPU lane prepare-log polling (phase owner).

Split phase of cli/colab_lane.py (the kaggle_lane.py owner-class pattern):
this owner owns streaming the VM-side prepare log into the ONE local
transcript — the offset probe script, one probe read, transient-failure
tolerance, the chunk fan-out, and the status/deadline exits.  The poll runs
against the prepare budget under one time bar; collaborators resolve at call
time through the running colab identity
(``sys.modules["__colab_runtime_self__"]``) via the lane's dial-ins.

Transcript ownership: the prepare process writes its whole output to the
remote ``PREPARE_LOG_NAME`` under the lane's own remote root; this poll reads
that one file and PRINTS each chunk, so the parent tee (the ONE transcript
writer, the file the lane context resolved) captures it beside every other
streamed stage.  No second file handle, no re-derived path.
"""
from __future__ import annotations

import time
from typing import Any

from core.run_log import RunLogger
from training.prepare_all_trace import timed
from cli.colab_lane_contracts import PREPARE_LOG_NAME, PREPARE_STATUS_NAME, _stamp
from cli.colab_reconnect import ControlChannelLost, ControlChannelRecovery

_LOG = RunLogger(__name__)


class ColabCPULanePoll:
    """Prepare-log polling: probes, transit tolerance, transcripts, exits."""

    @timed
    def poll_prepare_log(self, deadline_seconds: int) -> None:
        """Stream the VM-side prepare log into the ONE lane transcript.

        A transient control-channel loss is re-attached by the bounded
        recovery (every re-probe opens a fresh control channel and reads the
        durable log again); only a vanished session or an exhausted budget
        ends the poll -- never the first dropped socket, which killed the
        bundle-1013d launcher while its VM was still alive.
        """
        surface = self.surface
        session = self.session
        recovery = ControlChannelRecovery(session, surface._RECONNECT)
        started = time.monotonic()
        offset = 0
        transit_budget = surface._PROBE_RETRIES
        bar = _LOG.bar(total=deadline_seconds, desc="prepare_poll", unit="s")
        self._announce_prepare_stage(deadline_seconds)
        try:
            while True:
                try:
                    payload = recovery.run(
                        lambda: self._read_prepare_payload(surface, session, offset),
                        context="prepare log/status probe",
                    )
                except ControlChannelLost:
                    raise
                except RuntimeError as exc:
                    transit_budget = self._tolerate_prepare_probe_failure(
                        surface, exc, transit_budget)
                    continue
                transit_budget = surface._PROBE_RETRIES
                chunk = payload["chunk"]
                if chunk:
                    self._forward_prepare_chunk(chunk)
                    offset = int(payload["offset"])
                if self._prepare_status_complete(payload):
                    self._announce_prepare_finished(offset)
                    return
                self._assert_poll_deadline(started, deadline_seconds)
                time.sleep(surface._LOG_POLL_SECONDS)
                bar.update(surface._LOG_POLL_SECONDS)
        finally:
            bar.close()

    def prepare_log_remote_path(self) -> str:
        """The ONE remote prepare log this poll reads (the launcher writes it).

        Derived from the lane's own remote root — the same base the prepare
        process is launched against — never re-spelled beside it, so the poll
        can never read a different tree than the one it launched.
        """
        return f"{self.remote_root}/{PREPARE_LOG_NAME}"

    def _announce_prepare_stage(self, deadline_seconds: int) -> None:
        """Name the ONE remote log and local transcript before the first probe."""
        print(
            _stamp(),
            f"[prepare] stage started; remote log={self.prepare_log_remote_path()} "
            f"status={self.remote_root}/{PREPARE_STATUS_NAME} "
            f"budget={deadline_seconds}s -> transcript={self.transcript_path}",
            flush=True,
        )

    def _announce_prepare_finished(self, offset: int) -> None:
        """Close the stage in the transcript once the remote rc=0 is observed."""
        print(
            _stamp(),
            f"[prepare] stage finished rc=0; remote log={self.prepare_log_remote_path()} "
            f"bytes_read={offset}",
            flush=True,
        )

    def _prepare_probe_script(self, offset: int) -> str:
        """The remote script reading the prepare log tail and status file at one offset."""
        log_path = self.prepare_log_remote_path()
        status_path = f"{self.remote_root}/{PREPARE_STATUS_NAME}"
        return self.bundle_head() + f"""
import json, os

log_path = {log_path!r}
status_path = {status_path!r}
offset = {offset}
payload = {{"offset": offset, "chunk": "", "status": None}}
try:
    with open(log_path, "rb") as handle:
        handle.seek(offset)
        data = handle.read()
    payload["offset"] = offset + len(data)
    payload["chunk"] = data.decode("utf-8", errors="replace")
    if os.path.exists(status_path):
        payload["status"] = int(open(status_path).read().strip() or "-1")
except FileNotFoundError:
    pass
print(json.dumps(payload), flush=True)
"""

    @timed
    def _read_prepare_payload(self, surface: Any, session: str, offset: int) -> dict:
        """One offset probe, executed and parsed."""
        return self.parse_remote_json(
            self.exec_capture(
                session, self._prepare_probe_script(offset),
                timeout=surface._PROBE_TIMEOUT_SECONDS,
            )
        )

    @timed
    def _tolerate_prepare_probe_failure(self, surface: Any, exc: RuntimeError,
                                        transit_budget: int) -> int:
        """Tolerate one transient probe failure; raise on fatal or exhausted budget."""
        if self.transit_fatal(exc):
            raise
        transit_budget -= 1
        if transit_budget <= 0:
            raise
        message = f"[probe] prepare log unavailable; continuing: {exc}"
        print(_stamp(), message, flush=True)
        time.sleep(surface._LOG_POLL_SECONDS)
        return transit_budget

    def _forward_prepare_chunk(self, chunk: str) -> None:
        """Fan one prepare chunk into the ONE lane transcript.

        The chunk is PRINTED, never written through a second handle: the parent
        tee is the transcript's one writer (it expands tqdm's CR frames into
        grep-able lines and keeps the raw frames on the terminal), exactly like
        every other streamed stage. A direct handle here was the split that left
        the transcript with only probe heartbeats while the prepare log filled.
        Untimed on purpose: one chunk is instant, and a start/finish pair per
        chunk would bloat the transcript the prepare fills.
        """
        print(chunk, end="", flush=True)

    def _prepare_status_complete(self, payload: dict) -> bool:
        """True once the remote prepare reported rc=0; fail loud on any failure rc.

        The rc covers the WHOLE detached prepare process — ``prepare_all`` and
        the training-bundle stage chained after it — so a stage that fails loud
        refuses the delivery here exactly as a failed prepare does. The
        failing stage's ``[bundle-stage]``/``[prepare]`` marker and full
        traceback are in the streamed log above the raise.
        """
        status = payload["status"]
        if status is None:
            return False
        if int(status) != 0:
            raise RuntimeError(
                f"the Colab CPU prepare run failed on the VM (rc={status}; "
                "prepare_all and the training-bundle stage); see the [prepare] "
                "log above")
        return True

    def _assert_poll_deadline(self, started: float, deadline_seconds: int) -> None:
        """Fail loud once the poll outlives its deadline."""
        if time.monotonic() - started > deadline_seconds:
            raise RuntimeError(f"prepare poll deadline exceeded ({deadline_seconds}s)")
