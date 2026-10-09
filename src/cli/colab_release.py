"""Colab VM release by name, independent of the upstream CLI's stop() succeeding.

``stop()`` runs in main()'s ``finally`` and is the primary release of a Colab
VM.  The upstream ``colab stop`` command is best-effort: its kernel-client
teardown can fail (the reported ``self._kernel_client._manager is None``
AttributeError), its server-side unassign can fail on auth or network, and its
local record can be missing or stale -- any of which leaves the VM live and
burning accelerator quota.

This module owns the launcher's OWN stop-by-name path: the CLI stop is tried
first (unchanged), and when the session is still listed the launcher releases it
server-side through its own CLI entrypoint, then re-verifies the listing.  Split
from cli/colab.py (the kaggle_lane.py owner-module pattern), like
cli.colab_self_watch.
"""
from __future__ import annotations

import subprocess
import sys
import traceback

from cli.colab_hub import hub, timed_colab


class ColabSessionRelease:
    """Release one named Colab session and report whether release was confirmed.

    ``release()`` returns True only when the server-side listing no longer
    carries the session -- the same contract ``stop()`` always had -- so a
    caller that must not begin local postprocessing on an unreleased VM can
    still branch on it.  Failures are loud, never silent.
    """

    def __init__(self, session: str) -> None:
        self._session = session

    def _listed(self) -> bool:
        """Whether the session still appears in the server-side listing."""
        return hub()._session_listed(self._session)

    def _cli_stop(self) -> None:
        """Run the upstream stop; report a nonzero exit without raising."""
        result = hub().colab("stop", "-s", self._session, check=False, timeout=30)
        if result.returncode:
            print(
                hub()._stamp(),
                f"[warn] VM release command returned rc={result.returncode}; "
                f"stdout={result.stdout[-2000:]!r} stderr={result.stderr[-2000:]!r}",
                file=sys.stderr,
            )

    def _own_stop(self) -> None:
        """Unassign the session through our entrypoint when the CLI stop failed.

        The entrypoint resolves the session's endpoint from its local record and
        unassigns it server-side, so a CLI stop that failed on kernel teardown or
        on its own stale bookkeeping cannot leave the VM held open.
        """
        command = hub()._colab_command("release-session", "-s", self._session)
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=60, check=False)
        except (subprocess.SubprocessError, OSError) as exc:
            print(
                hub()._stamp(),
                f"[warn] own stop-by-name release could not run for "
                f"'{self._session}': {exc}",
                file=sys.stderr,
            )
            traceback.print_exc()
            return
        print(
            hub()._stamp(),
            f"[stop] own stop-by-name release rc={result.returncode}: "
            f"{(result.stdout or '').strip()[-2000:]}",
            flush=True,
        )
        if result.returncode:
            print(
                hub()._stamp(),
                f"[warn] own stop-by-name release failed: "
                f"{(result.stderr or '')[-2000:]!r}",
                file=sys.stderr,
            )

    @timed_colab("step")
    def release(self) -> bool:
        """Stop the session, verify it, and fall back to our own release path."""
        print(hub()._stamp(), f"[stop] tearing down '{self._session}'")
        try:
            self._cli_stop()
        except (subprocess.SubprocessError, OSError) as exc:
            print(
                hub()._stamp(),
                f"[warn] VM release request failed - the VM '{self._session}' may "
                f"STILL BE LIVE and burning Colab GPU quota until it times out or "
                f"is reaped. After handling the failure above, reclaim it with: "
                f"colab stop -s {self._session}   (or 'colab sessions' to check). "
                f"Original error: {exc}",
                file=sys.stderr,
            )
            traceback.print_exc()
        if self._listed():
            print(
                hub()._stamp(),
                f"[warn] teardown left '{self._session}' listed; running the "
                "launcher's own stop-by-name release.",
                file=sys.stderr,
            )
            self._own_stop()
        confirmed = not self._listed()
        if confirmed:
            print(hub()._stamp(), "[stop] teardown verified: session is no longer listed")
        else:
            print(
                hub()._stamp(),
                f"[warn] teardown could not be verified; '{self._session}' may still "
                "be live and consuming quota. Reclaim it with: "
                f"colab stop -s {self._session}",
                file=sys.stderr,
            )
        print(hub()._stamp(), "[stop] VM release requested")
        return confirmed
