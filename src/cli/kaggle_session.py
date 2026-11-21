"""The owned lifecycle of a pushed Kaggle kernel session (transport layer).

``KernelSession`` is the ONE owner of: the account's batch-GPU slot census
(``preflight``), the push-time session capture (``capture``) and the verified
release (``stop`` / ``delete``). It exists because ``kaggle kernels push``
soft-fails with rc=0 (``Kernel push error: Maximum batch GPU session count of
2 reached``) while earlier QUEUED pushes keep their slots, and the old
version-replace stub never released them.

Every method takes a ``owner/slug`` kernel and touches ONLY that account's own
kernels; another owner's live kernel is never cancelled or deleted.
"""
from __future__ import annotations

import re
import time
import traceback
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict


class KernelSlot(BaseModel):
    """One of OUR kernels holding an active Kaggle GPU session slot."""

    model_config = ConfigDict(frozen=True)

    kernel: str
    status: str


class KaggleQuotaExceeded(RuntimeError):
    """``kaggle kernels push`` refused (rc=0) by the batch-GPU session quota."""


class KernelSession:
    """Capture, preflight, and VERIFIED release of a kernel session.

    ``stop`` cancels a recorded session id in place; with no id an ACTIVE
    session is released by deleting the kernel — the only mechanism that frees
    a QUEUED batch-GPU slot. A terminal session is already released and is
    never touched.
    """

    ACTIVE_STATES: ClassVar[tuple[str, ...]] = (
        "queued", "running", "cancelRequested", "cancelAcknowledged")
    TERMINAL_STATES: ClassVar[tuple[str, ...]] = ("complete", "error")
    #: Kaggle's per-account batch-GPU session ceiling. A platform fact the push
    #: soft error names ("Maximum batch GPU session count of 2 reached"), not a
    #: project knob, so it is a constant here and not in the training config.
    PLATFORM_BATCH_GPU_SLOTS: ClassVar[int] = 2

    @staticmethod
    def state(slug: str) -> str:
        """The kernel's normalized session state (``absent`` once deleted)."""
        from cli import kaggle_lane as lane

        try:
            return lane.kernel_status(slug)["status"]
        except RuntimeError as error:
            # The CLI renders a missing/private kernel as "Cannot access kernel
            # ..." (the SDK returns 404); both mean the session is gone.
            text = str(error).lower()
            if ("not found" in text or "404" in text
                    or "cannot access kernel" in text):
                return "absent"
            raise

    @classmethod
    def recorded_id(cls, slug: str) -> int | None:
        """The launch-recorded numeric session id, or None (name fallback)."""
        from cli import kaggle_lane as lane

        handle = lane.recorded_kernel_handle(slug) or ""
        return int(handle) if handle.isdigit() else None

    @classmethod
    def parse_push(cls, slug: str, *, returncode: int, output: str,
                   require_success: bool) -> dict[str, Any]:
        """Parse one push; rc=0 is NOT success (the quota soft error).

        ``require_success`` is the laya lane's contract (the CLI success line
        must be present); the ER lane passes False because its runner already
        fails loud on a non-zero rc, so the explicit "Kernel push error:" line
        is the only silent-soft failure left to catch.
        """
        lowered = output.lower()
        quota = re.search(r"maximum batch gpu session count of (\d+)", lowered)
        soft = "kernel push error:" in lowered
        failed = returncode != 0 or soft or (
            require_success and ("error" in lowered
                                 or "successfully pushed" not in lowered))
        if failed:
            tail = output.strip()[-4000:] or "(kaggle produced no output)"
            if quota:
                raise KaggleQuotaExceeded(
                    f"kaggle kernels push for {slug} was refused by the "
                    f"batch-GPU session quota (limit {quota.group(1)}, "
                    f"rc={returncode}):\n{tail}"
                    + cls._active_session_census(slug)
                    + "\nremedy: `--recreate --decision <kind>` deletes our "
                      "own queued sessions and the kernel, then re-pushes; or "
                      "wait for the running session to finish. A plain "
                      "`--stop` cannot release a queued slot without the "
                      "session id")
            raise RuntimeError(
                f"kaggle kernels push failed (rc={returncode}):\n{tail}")
        version = re.search(r"kernel version (\d+) successfully pushed", lowered)
        url = re.search(r"progress at (\S+)", output)
        return {"kernel": slug, "returncode": returncode,
                "version": int(version.group(1)) if version else None,
                "url": url.group(1) if url else None}

    @classmethod
    def _active_session_census(cls, slug: str) -> str:
        """The exact holders appended to a quota rejection (never guessed)."""
        from cli import kaggle_lane as lane

        try:
            slots = cls.census(slug.partition("/")[0])
        except Exception:  # noqa: BLE001 - best-effort diagnostic, recorded
            lane._log_lane("[session-quota] holder census failed:\n"
                           f"{traceback.format_exc()}")
            return "\nactive GPU sessions: (census unavailable; see lane.log)"
        listed = ", ".join(f"{slot.kernel} ({slot.status})"
                           for slot in slots) or "none"
        return f"\nactive GPU sessions on the account: {listed}"

    @classmethod
    def capture(cls, slug: str, captured: dict[str, Any], *,
                push: dict[str, Any] | None = None) -> dict[str, Any]:
        """Describe a push-time capture: the id, its source, or WHY it is gone.

        A missing id is never silent: the capture owner's own reason is kept,
        and the observed ``kaggle kernels status`` is recorded (the push may
        still be QUEUED — the slot-holding state the stop must see).
        """
        from cli import kaggle_lane as lane

        session_id = captured.get("session_id")
        plan: dict[str, Any] = {
            "kernel": slug, "session_id": session_id,
            "handle": captured.get("handle"),
            "source": ("log_stream_proxy" if session_id is not None
                       else "unavailable"),
            "reason": None if session_id is not None else (
                captured.get("error")
                or "the log-stream proxy served no URL; no numeric session id"),
        }
        if session_id is None:
            try:
                plan["observed_state"] = cls.state(slug)
            except Exception as error:  # noqa: BLE001 - recorded, never masks push
                plan["observed_state"] = "unknown"
                plan["state_error"] = f"{type(error).__name__}: {str(error)[:200]}"
        if push:
            plan["push"] = push
        lane._log_lane(
            f"[{slug}] session capture id={session_id} source={plan['source']} "
            f"reason={plan['reason']} state={plan.get('observed_state')}")
        return plan

    @classmethod
    def stop(cls, slug: str, *, execute: bool,
             wait: bool = True) -> dict[str, Any]:
        """Release ``slug``'s session, verified; never a slot-holding stub.

        A recorded id feeds the SDK ``cancel_kernel_session``. Without one (or
        when the cancel raises) an ACTIVE session is released by deleting the
        kernel; a terminal session is already released, so nothing is touched.
        ``wait=False`` issues the release and returns ``requested``.
        """
        from cli import kaggle_lane as lane

        plan: dict[str, Any] = {
            "mode": "executed" if execute else "dry-run",
            "kernel": slug, "action": "stop_kernel"}
        if not execute:
            return plan
        session_id = cls.recorded_id(slug)
        plan["handle"] = lane.recorded_kernel_handle(slug)
        if session_id is not None:
            cls.cancel(slug, session_id, plan)
        if "cancel_method" not in plan and not wait:
            # No id, no status poll: deletion is the only release it can issue.
            plan["delete"] = cls.delete(slug)
            plan["cancel_method"] = "delete_kernel"
        elif "cancel_method" not in plan:
            observed = cls.state(slug)
            plan["observed_state"] = observed
            if observed in cls.TERMINAL_STATES or observed == "absent":
                plan.update(cancel_method="none_already_released",
                            verdict="stopped", stopped=True,
                            terminal_state=observed)
                return plan
            # No recorded id but an ACTIVE session: one bounded proxy probe can
            # still capture the id, so a RUNNING session is cancelled in place.
            captured = cls._capture_on_demand(slug, plan)
            if captured is None or not cls.cancel(slug, captured, plan):
                # A QUEUED session has no proxy yet: the delete is the release.
                plan["delete"] = cls.delete(slug)
                plan.update(cancel_method="delete_kernel", verdict="stopped",
                            stopped=True, terminal_state="absent")
                return plan
        if not wait:
            plan.update(verdict="requested", stopped=None)
            return plan
        # Verified stop: bounded status polls; only a terminal state is a stop.
        verdict = "still_running"
        deadline = (time.monotonic()
                    + max(lane._spec().limits.stop_verify_polls, 1)
                    * lane._spec().logs_poll_seconds)
        last_state = None
        while time.monotonic() < deadline:
            state = lane.kernel_status(slug)["status"]
            if state != last_state:
                lane._log_lane(f"[stop] {slug} state={state} "
                               f"(method={plan['cancel_method']})")
                last_state = state
            if state in cls.TERMINAL_STATES:
                verdict = "stopped"
                plan["terminal_state"] = state
                break
            time.sleep(lane._spec().logs_poll_seconds)
        plan["verdict"] = verdict
        plan["stopped"] = verdict == "stopped"
        if not plan["stopped"]:
            raise RuntimeError(
                f"stop did not reach a terminal state within the verify window "
                f"({verdict}; session {slug}, method {plan['cancel_method']})")
        return plan

    @staticmethod
    def _capture_on_demand(slug: str, plan: dict[str, Any]) -> int | None:
        """One bounded proxy probe for an ACTIVE session's id (never fabricated)."""
        from cli import kaggle_lane as lane

        try:
            captured = lane.capture_kernel_session_id(slug, attempts=1)
        except Exception as error:  # noqa: BLE001 - recorded; delete still runs
            plan["capture_error"] = f"{type(error).__name__}: {str(error)[:200]}"
            lane._log_lane(f"[stop] on-demand capture failed for {slug}:\n"
                           f"{traceback.format_exc()}")
            return None
        plan["capture"] = captured
        session_id = captured.get("session_id")
        return int(session_id) if session_id is not None else None

    @staticmethod
    def cancel(slug: str, session_id: int, plan: dict[str, Any]) -> bool:
        """Issue the SDK in-place cancel; record a failure with its traceback."""
        from cli import kaggle_lane as lane

        try:
            from kagglesdk.kaggle_client import KaggleClient
            from kagglesdk.kaggle_env import KaggleEnv
            from kagglesdk.kernels.types.kernels_api_service import (
                ApiCancelKernelSessionRequest,
            )

            request = ApiCancelKernelSessionRequest()
            request.kernel_session_id = session_id
            (KaggleClient(env=KaggleEnv.PROD).kernels.kernels_api_client
             .cancel_kernel_session(request))
            plan["cancel_method"] = "sdk_cancel_kernel_session"
            return True
        except Exception as error:  # noqa: BLE001 - fallback is recorded, loud
            plan["cancel_error"] = f"{type(error).__name__}: {str(error)[:200]}"
            lane._log_lane(f"[stop] cancel failed for {slug} "
                           f"session {session_id}:\n{traceback.format_exc()}")
            return False

    @classmethod
    def delete(cls, slug: str) -> dict[str, Any]:
        """SDK-delete the kernel and VERIFY it is gone (the slot is released)."""
        from cli import kaggle_lane as lane

        owner, _, kernel = slug.partition("/")
        if not owner or not kernel:
            raise ValueError(f"kernel slug must be owner/slug, got {slug!r}")
        from kagglesdk.kernels.types.kernels_api_service import ApiDeleteKernelRequest

        request = ApiDeleteKernelRequest()
        request.user_name = owner
        request.kernel_slug = kernel
        try:
            cls._api().delete_kernel(request)
        except Exception as error:
            lane._log_lane(traceback.format_exc())
            # A never-pushed kernel answers 404/403; confirm absence through the
            # status endpoint rather than turning a real permission refusal
            # into a silent success.
            if (cls._status_code(error) in (403, 404)
                    and cls._sdk_state(cls._api(), slug,
                                       absent_on_forbidden=True) == "absent"):
                return {"kernel": slug, "delete_method": "already_absent"}
            raise RuntimeError(
                f"delete failed for {slug}: {type(error).__name__}: {error}"
            ) from error
        cls._await_absent(slug)
        return {"kernel": slug, "delete_method": "sdk_delete_kernel",
                "verified_absent": True}

    @classmethod
    def census(cls, owner: str) -> list[KernelSlot]:
        """OUR OWN kernels holding an ACTIVE GPU session (account census)."""
        from kagglesdk.kernels.types.kernels_api_service import ApiListKernelsRequest
        from kagglesdk.kernels.types.kernels_enums import KernelsListViewType

        api = cls._api()
        slots: list[KernelSlot] = []
        page_token = ""
        while True:
            request = ApiListKernelsRequest()
            request.user = owner
            request.group = KernelsListViewType.PROFILE
            request.page_size = 50
            if page_token:
                request.page_token = page_token
            response = api.list_kernels(request)
            for metadata in response.kernels or ():
                ref = metadata.ref
                if not ref or "/" not in ref:
                    continue
                status = cls._sdk_state(api, ref)
                if status in cls.ACTIVE_STATES and cls._sdk_gpu(api, ref):
                    slots.append(KernelSlot(kernel=ref, status=status))
            page_token = response.next_page_token
            if not page_token:
                return slots

    @classmethod
    def preflight(cls, slug: str, *, execute: bool,
                  release_orphans: bool = False) -> dict[str, Any]:
        """Census our own batch-GPU slots before a push; release or fail loud.

        The ceiling is per ACCOUNT and the API exposes one status per kernel
        (never per queued push), so the census is a lower bound; the push's own
        soft error stays the authoritative gate. A target whose status is
        ``queued`` is an earlier push still holding a slot — re-pushing stacks
        another one, so it blocks unless ``release_orphans`` (``--recreate``,
        whose teardown deletes the target) releases it. Only OUR queued
        kernels are ever auto-released; a RUNNING session is reported, never
        touched.
        """
        from cli import kaggle_lane as lane

        plan: dict[str, Any] = {
            "mode": "executed" if execute else "dry-run", "kernel": slug,
            "limit": cls.PLATFORM_BATCH_GPU_SLOTS,
            "release_orphans": release_orphans}
        if not execute:
            plan.update(available=None,
                        reason="dry-run: account sessions are not probed; "
                               "pass --execute")
            return plan
        owner, _, kernel = slug.partition("/")
        if not owner or not kernel:
            raise ValueError(f"kernel slug must be owner/slug, got {slug!r}")
        try:
            slots = cls.census(owner)
        except Exception as error:  # noqa: BLE001 - recorded; push stays the gate
            lane._log_lane(f"[session-preflight] census failed for {owner}:\n"
                           f"{traceback.format_exc()}")
            plan.update(available=False, would_exceed=None,
                        reason=f"{type(error).__name__}: {str(error)[:400]}")
            return plan
        target = next((slot for slot in slots if slot.kernel == slug), None)
        holders = [slot for slot in slots if slot.kernel != slug]
        released: list[dict[str, Any]] = []
        if release_orphans:
            for slot in [held for held in holders if held.status == "queued"]:
                deletion = cls.delete(slot.kernel)
                released.append({"kernel": slot.kernel, "status": slot.status,
                                 "delete": deletion})
                lane._log_lane(
                    f"[session-preflight] released our QUEUED session "
                    f"{slot.kernel} via {deletion['delete_method']}")
                holders.remove(slot)
        target_queued = bool(target and target.status == "queued")
        occupied = len(holders) + (1 if target_queued else 0)
        blocking = ((occupied + 1) > cls.PLATFORM_BATCH_GPU_SLOTS
                    or (target_queued and not release_orphans))
        plan.update(
            available=True,
            holders=[slot.model_dump() for slot in holders],
            target_state=target.status if target else "absent",
            target_queued=target_queued,
            occupied=occupied,
            free_slots=max(cls.PLATFORM_BATCH_GPU_SLOTS - occupied - 1, 0),
            released=released,
            would_exceed=blocking)
        plan["remedy"] = cls._remedy(slug, plan)
        if blocking:
            raise RuntimeError(plan["remedy"])
        return plan

    @staticmethod
    def _remedy(slug: str, plan: dict[str, Any]) -> str:
        """The exact slots/holders plus the two real remedies."""
        holders = ", ".join(f"{held['kernel']} ({held['status']})"
                            for held in plan.get("holders", ())) or "none"
        running = [held["kernel"] for held in plan.get("holders", ())
                   if held["status"] != "queued"]
        return (
            f"kernel session preflight for {slug}: occupied="
            f"{plan.get('occupied')} of {plan.get('limit')} batch-GPU session "
            f"slots; holders: {holders}"
            + (f"; {slug} has a QUEUED earlier push"
               if plan.get("target_queued") else "")
            + ". Remedy: `--recreate --decision <kind>` (deletes our own queued "
              "sessions and the kernel, then re-pushes) or wait"
            + (f" for {', '.join(running)} to finish" if running else "")
            + ". Another owner's live kernel is never touched.")

    @staticmethod
    def _api():
        """The prod kernels API client (ONE construction site)."""
        from kagglesdk.kaggle_client import KaggleClient
        from kagglesdk.kaggle_env import KaggleEnv

        return KaggleClient(env=KaggleEnv.PROD).kernels.kernels_api_client

    @staticmethod
    def _status_code(error: Exception) -> int | None:
        response = getattr(error, "response", None)
        return getattr(response, "status_code", None)

    @classmethod
    def _status_name(cls, status: Any) -> str:
        """``KernelWorkerStatus.CANCEL_REQUESTED`` -> ``cancelRequested``."""
        name = str(status).replace("KernelWorkerStatus.", "")
        if not name or name == "None":
            return "unknown"
        head, *tail = name.lower().split("_")
        return head + "".join(part.title() for part in tail)

    @classmethod
    def _sdk_state(cls, api, slug: str, *,
                   absent_on_forbidden: bool = False) -> str:
        from kagglesdk.kernels.types.kernels_api_service import (
            ApiGetKernelSessionStatusRequest,
        )

        owner, _, kernel = slug.partition("/")
        request = ApiGetKernelSessionStatusRequest()
        request.user_name = owner
        request.kernel_slug = kernel
        try:
            response = api.get_kernel_session_status(request)
        except Exception as error:
            code = cls._status_code(error)
            # A missing kernel answers 403 or 404; the census keeps 403 loud
            # (it lists only our own kernels), the delete verification does not.
            if code == 404 or (absent_on_forbidden and code == 403):
                return "absent"
            raise
        return cls._status_name(response.status)

    @staticmethod
    def _sdk_gpu(api, slug: str) -> bool:
        """True when the kernel's session runs on an accelerator (GPU/TPU)."""
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

        owner, _, kernel = slug.partition("/")
        request = ApiGetKernelRequest()
        request.user_name = owner
        request.kernel_slug = kernel
        metadata = api.get_kernel(request).metadata
        return bool(metadata and (metadata.enable_gpu
                                  or metadata.machine_shape not in ("", "None")))

    @classmethod
    def _await_absent(cls, slug: str, api=None) -> None:
        """Verify a delete released the kernel, bounded by configured limits."""
        from cli import kaggle_lane as lane

        limits = lane._spec().limits
        api = api or cls._api()
        for attempt in range(limits.stop_attempts):
            if cls._sdk_state(api, slug,
                              absent_on_forbidden=True) == "absent":
                return
            if attempt + 1 < limits.stop_attempts:
                time.sleep(limits.retry_seconds)
        raise RuntimeError(
            f"delete of {slug} was issued but the kernel is still visible "
            f"after {limits.stop_attempts} probes; the session slot may still "
            "be held")
