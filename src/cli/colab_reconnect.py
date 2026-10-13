"""Colab control-channel recovery: classify a loss, re-attach, resume.

Split from cli/colab.py (the kaggle_lane.py owner-module pattern) exactly like
cli.colab_transport / cli.colab_self_watch.  This is the ONE home for the
upstream CLI's loss markers: the streaming/capturing transport and the two
poll loops that must survive a transient drop ask this class instead of each
re-spelling ``"connection was lost"`` / ``"session '<name>' not found"``.

A lost control channel is not a failed run.  The detached trainer keeps
writing its log/status pair on the VM while the launcher's channel is down, so
the repair is to re-establish the channel (every ``colab exec`` opens a fresh
one) and read the durable log again.  The policy is bounded and config-owned
(``colab.reconnect``): a session that is actually gone is terminal at once,
and exhausting the attempts is terminal too -- never an unbounded retry loop.
"""
from __future__ import annotations

import sys
import time
import traceback
from enum import Enum
from typing import Callable, TypeVar

from cli.colab_hub import hub

T = TypeVar("T")


class ControlChannelLoss(str, Enum):
    """How one Colab control operation failed."""

    #: The channel dropped; the session may still be alive and is worth re-attaching.
    TRANSIENT = "transient"
    #: The session/kernel is gone: there is nothing left to re-attach to.
    SESSION_LOST = "session_lost"
    #: A real remote error (failed command, bad reply), not a channel loss.
    OTHER = "other"


class ControlChannelLost(RuntimeError):
    """A control-channel loss the bounded recovery policy cannot repair."""


class ControlChannelRecovery:
    """Bounded re-attach for one Colab session's control channel.

    ``session`` names the session whose loss markers are recognised, and
    ``policy`` is the config-owned attempt/backoff bound.  The factory
    ``for_running_launcher`` resolves both from the running launcher, so a call
    site never re-spells the session or the policy.
    """

    def __init__(self, session: str, policy) -> None:
        self._session = session
        self._policy = policy

    @classmethod
    def for_running_launcher(cls) -> ControlChannelRecovery:
        """The running launcher's session and its config-owned reconnect policy."""
        surface = hub()
        return cls(surface.SESSION, surface._RECONNECT)

    @property
    def attempts(self) -> int:
        """How many re-attach attempts the policy allows before giving up."""
        return int(self._policy.attempts)

    @staticmethod
    def classify(session: str, detail: str) -> ControlChannelLoss:
        """Read one upstream failure message against ``session``'s loss markers.

        The ONE home for the upstream CLI's wording: the transport retry, the
        detached-stage pollers, the parallel-training poll loop and the lane's
        ``transit_fatal`` contract all ask here instead of each re-spelling
        ``"connection was lost"`` / ``"session '<name>' not found"``.
        """
        text = detail.lower()
        if (
            f"session '{session}' not found".lower() in text
            or "appears to be lost" in text
        ):
            return ControlChannelLoss.SESSION_LOST
        if "connection was lost" in text:
            return ControlChannelLoss.TRANSIENT
        return ControlChannelLoss.OTHER

    def backoff_seconds(self, attempt: int) -> float:
        """Capped exponential backoff for the 1-based re-attach ``attempt``."""
        return min(
            float(self._policy.max_backoff_seconds),
            float(self._policy.initial_backoff_seconds) * (2 ** (attempt - 1)),
        )

    def run(self, operation: Callable[[], T], *, context: str) -> T:
        """Run a control operation, re-attaching a lost channel within the policy.

        Re-running ``operation`` IS the re-attach: every Colab exec opens a
        fresh control channel.  A transient loss waits the capped backoff and
        retries; a vanished session, a non-channel error, or an exhausted
        attempt budget fails the call (loudly, with the full traceback).
        """
        attempt = 0
        while True:
            try:
                return operation()
            except Exception as exc:
                loss = self.classify(self._session, str(exc))
                if loss is not ControlChannelLoss.TRANSIENT:
                    if loss is ControlChannelLoss.SESSION_LOST:
                        self._report_terminal(loss, context, attempt, exc)
                    raise
                attempt += 1
                if attempt > self.attempts:
                    self._report_terminal(loss, context, attempt - 1, exc)
                    raise ControlChannelLost(
                        f"{context}: control channel not re-attached after "
                        f"{self.attempts} attempt(s); last error: {exc}"
                    ) from exc
                self._wait(attempt, context, exc)

    def _wait(self, attempt: int, context: str, cause: Exception) -> None:
        """Log the loss, then wait the policy's capped backoff before retrying."""
        delay = self.backoff_seconds(attempt)
        print(
            hub()._stamp(),
            f"[reconnect] {context}: control channel lost "
            f"(attempt {attempt}/{self.attempts}); re-attaching in {delay:g}s: {cause}",
            flush=True,
        )
        time.sleep(delay)

    def _report_terminal(
        self, loss: ControlChannelLoss, context: str, attempt: int, exc: Exception,
    ) -> None:
        """Record a terminal recovery failure with the full traceback."""
        print(
            hub()._stamp(),
            f"[reconnect] {context}: control channel loss is terminal "
            f"({loss.value}; attempts used={attempt}); no re-attach: {exc}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exception(type(exc), exc, exc.__traceback__)
