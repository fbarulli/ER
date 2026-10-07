"""CPU lane prepare-log polling (phase owner).

Split phase of cli/colab_lane.py (the kaggle_lane.py owner-class pattern):
this owner owns streaming the VM-side prepare log into both local
transcripts — the offset probe script, one probe read, transient-failure
tolerance, the chunk fan-out, and the status/deadline exits.  The poll runs
against the prepare budget under one time bar; collaborators resolve at call
time through the running colab identity
(``sys.modules["__colab_runtime_self__"]``) via the lane's dial-ins.
"""
from __future__ import annotations

import time
from typing import Any

from core.run_log import RunLogger
from training.prepare_all_trace import timed
from cli.colab_lane_contracts import PREPARE_LOG_NAME, PREPARE_STATUS_NAME, _stamp

_LOG = RunLogger(__name__)


class ColabCPULanePoll:
    """Prepare-log polling: probes, transit tolerance, transcripts, exits."""

    @timed
    def poll_prepare_log(self, deadline_seconds: int) -> None:
        """Stream the VM-side prepare log into both local transcripts."""
        surface = self.surface
        session = self.session
        started = time.monotonic()
        offset = 0
        transit_budget = surface._PROBE_RETRIES
        bar = _LOG.bar(total=deadline_seconds, desc="prepare_poll", unit="s")
        try:
            while True:
                try:
                    payload = self._read_prepare_payload(surface, session, offset)
                except RuntimeError as exc:
                    transit_budget = self._tolerate_prepare_probe_failure(
                        surface, exc, transit_budget)
                    continue
                transit_budget = surface._PROBE_RETRIES
                chunk = payload["chunk"]
                if chunk:
                    self._forward_prepare_chunk(surface, chunk)
                    offset = int(payload["offset"])
                if self._prepare_status_complete(payload):
                    return
                self._assert_poll_deadline(started, deadline_seconds)
                time.sleep(surface._LOG_POLL_SECONDS)
                bar.update(surface._LOG_POLL_SECONDS)
        finally:
            bar.close()

    def _prepare_probe_script(self, offset: int) -> str:
        """The remote script reading the prepare log tail and status file at one offset."""
        remote_root = self.remote_root
        return self.bundle_head() + f"""
import json, os

root = "{remote_root}"
log_path = root + "/{PREPARE_LOG_NAME}"
status_path = root + "/{PREPARE_STATUS_NAME}"
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
                training_output=True,
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
        surface._write_training_log(message + "\n")
        print(_stamp(), message, flush=True)
        time.sleep(surface._LOG_POLL_SECONDS)
        return transit_budget

    @timed
    def _forward_prepare_chunk(self, surface: Any, chunk: str) -> None:
        """Fan one prepare chunk into both local transcripts."""
        for line in chunk.splitlines():
            surface._write_training_log(f"[prepare] {line}\n")
            print(f"[prepare] {line}", flush=True)

    def _prepare_status_complete(self, payload: dict) -> bool:
        """True once the remote prepare reported rc=0; fail loud on any failure rc."""
        status = payload["status"]
        if status is None:
            return False
        if int(status) != 0:
            raise RuntimeError(
                f"prepare_all failed on the VM (rc={status}); see the [prepare] log above")
        return True

    def _assert_poll_deadline(self, started: float, deadline_seconds: int) -> None:
        """Fail loud once the poll outlives its deadline."""
        if time.monotonic() - started > deadline_seconds:
            raise RuntimeError(f"prepare poll deadline exceeded ({deadline_seconds}s)")
