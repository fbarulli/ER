"""Optional W&B mirror for non-sensitive training and HPO evidence."""

from __future__ import annotations

import os
import time
import traceback
from pathlib import Path
from typing import Any, Iterator

from core.common import training_cfg


class WandbCtx:
    """One run: cloud telemetry is always on. A missing key fails loud —
    no silent local-only degradation (owner order 2026-10-07 'bake wandb in,
    always on'; training/prepared tracking.wandb.mode owns the dashboard lane,
    never this class)."""

    def __init__(self, name: str):
        spec = training_cfg().tracking.wandb
        enabled = bool(os.environ.get("WANDB_API_KEY")) and spec.mode != "disabled"
        self._run = None
        self._name, self._project, self._mode = name, spec.project, spec.mode
        if not enabled:
            raise RuntimeError(
                '[wandb] WANDB_API_KEY absent while tracking.wandb.mode='
                + str(spec.mode)
                + ' — the cloud mirror is always on; export '
                  'WANDB_API_KEY (or drop it in .env) before training')

    def __enter__(self):
        import wandb
        run_name = os.environ.get("WANDB_RUN_NAME", self._name)
        # Disable W&B's broad host sampler. Training emits the deliberately
        # bounded memory series itself, so CPU/GPU/disk/system auto-series do
        # not flood the run or overlap with worker telemetry.
        settings = wandb.Settings(
            x_disable_stats=True,
            x_disable_machine_info=True,
        )
        self._run = wandb.init(
            project=self._project,
            name=run_name,
            mode=self._mode,
            settings=settings,
        )
        print(
            f"[wandb] run started: project={self._project} mode={self._mode} "
            f"id={self._run.id} url={self._run.url}",
            flush=True,
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._run is not None:
            self._run.finish(exit_code=1 if exc_type else 0)
        return False

    def log_config(self, values: dict[str, Any]) -> None:
        if self._run is not None:
            self._run.config.update(values, allow_val_change=True)

    def log_metrics(self, values: dict[str, Any], *, step: int | None = None) -> None:
        if self._run is not None:
            numeric = {k: v for k, v in values.items() if isinstance(v, (int, float))}
            if numeric:
                self._run.log(numeric, step=step)

    @property
    def run_id(self) -> str | None:
        return str(self._run.id) if self._run is not None else None

    @property
    def run_url(self) -> str | None:
        return str(self._run.url) if self._run is not None else None

    def set_summary(self, values: dict[str, Any]) -> None:
        if self._run is not None:
            self._run.summary.update(values)

    def log_artifact(self, path: str | Path, name: str) -> None:
        self.log_artifacts([path], name)

    def log_artifacts(self, paths: list[str | Path], name: str) -> None:
        """Upload files or directories as one downloadable W&B artifact."""
        if self._run is not None:
            import wandb

            artifact = wandb.Artifact(name, type="training-result")
            for raw_path in paths:
                p = Path(raw_path)
                if not p.exists():
                    raise FileNotFoundError(f"W&B artifact missing: {p}")
                if p.is_dir():
                    artifact.add_dir(str(p), name=p.name)
                else:
                    artifact.add_file(str(p), name=p.name)
            self._run.log_artifact(artifact)

    def log_image(self, path: str | Path, name: str) -> None:
        if self._run is not None:
            import wandb

            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(f"W&B image missing: {p}")
            self._run.log({name: wandb.Image(str(p))})


#: W&B run states that end the tracking poll loop.
_WANDB_TERMINAL_STATES = frozenset({"finished", "crashed", "failed", "killed"})

#: The live text channel the staged kernels emit notable lines on. W&B exposes
#: ``output.log`` only on flush/end, so ``log/line`` (a ``wandb.log`` payload)
#: is the only surface that streams while the run is live.
LIVE_LOG_KEY = "log/line"


class WandbRunReader:
    """Read-only live repository over one W&B run (metrics + live text channel).

    The writer counterpart is :class:`WandbCtx`. Given the configured
    ``tracking.wandb.project`` and a run tag, this streams the run's latest
    logged metrics and its live text channel (``log/line``), falling back to the
    ``output.log`` console. The ``WANDB_API_KEY`` is resolved through the
    canonical credential owner (``core.credentials.CredentialStore``) and
    exported to the wandb client's environment silently — never logged or
    printed. Polling honours the config-declared cadence
    (``tracking.wandb.poll_seconds``); a W&B read error is recorded, never
    fatal (the console fallback still runs).
    """

    def __init__(self, *, run_tag: str, project: str | None = None,
                 poll_seconds: float | None = None, api: Any | None = None):
        if project is None or poll_seconds is None:
            cfg = training_cfg().tracking.wandb
            project = project or cfg.project
            poll_seconds = cfg.poll_seconds if poll_seconds is None else poll_seconds
        self.run_tag = run_tag
        self.project = project
        self.poll_seconds = float(poll_seconds)
        self._api = api

    @property
    def path(self) -> str:
        """The ``project/run_tag`` address the wandb API resolves."""
        return f"{self.project}/{self.run_tag}"

    @classmethod
    def available(cls) -> bool:
        """True when a ``WANDB_API_KEY`` is resolvable (remote tracking possible)."""
        from core.credentials import CredentialStore

        return (CredentialStore.from_config()
                .resolve_optional("wandb_api_key") is not None)

    def _client(self):
        if self._api is None:
            from core.credentials import CredentialStore

            # Silent export of the declared secrets; the value never prints.
            CredentialStore.from_config().apply_to_environment()
            import wandb

            self._api = wandb.Api()
        return self._api

    def read_once(self) -> dict[str, Any]:
        """One snapshot: run state, metrics, the live channel, the console log."""
        run = self._client().run(self.path)
        output, console_error = self._console(run)
        live, live_error = self._live(run)
        return {
            "run": self.path,
            "state": str(getattr(run, "state", "unknown")),
            "metrics": dict(getattr(run, "summary", {}) or {}),
            "output": output,
            "console_error": console_error,
            "live": live,
            "live_error": live_error,
        }

    @staticmethod
    def _absent_log(error: BaseException) -> bool:
        """True when the failure means ``output.log`` simply does not exist.

        W&B raises ``FileNotFoundError`` for a missing file and wraps the empty
        file list (``IndexError``) from ``Run.file`` in a ``CommError``; both
        mean "this run captured no console", not a transport failure.
        """
        if isinstance(error, (FileNotFoundError, IndexError)):
            return True
        cause = getattr(error, "exc", None) or error.__cause__
        return isinstance(cause, IndexError)

    @classmethod
    def _console(cls, run) -> tuple[str, str | None]:
        """Read ``output.log``; only an ABSENT log is a recorded soft case.

        A run that captured no console has no ``output.log`` — that is expected
        and recorded as the console reason. Any other failure is a real
        transport/API error and propagates (fail-loud) instead of being
        collapsed into the same recorded reason as an absent file.
        """
        try:
            handle = run.file("output.log").download(replace=True)
        except Exception as error:  # noqa: BLE001 - classify, then re-raise
            if not cls._absent_log(error):
                raise
            return "", f"{type(error).__name__}: {error}\n{traceback.format_exc()}"
        try:
            return handle.read(), None
        finally:
            handle.close()

    @staticmethod
    def _live(run) -> tuple[list[str], str | None]:
        """Read the live text channel's flushed batches, oldest first.

        Each kernel flush is ONE ``log/line`` row, ordered by step, so the
        batches are a stable append-only sequence the caller can index past its
        already-emitted prefix. ``scan_history`` returns the raw (string) values;
        a run/client without it degrades to ``history``. A read error is
        recorded with its full traceback, never raised: ``output.log`` remains
        the fallback while the channel is unavailable.
        """
        reader = getattr(run, "scan_history", None) or getattr(
            run, "history", None)
        if reader is None:
            return [], None
        try:
            rows = reader(keys=[LIVE_LOG_KEY])
            return ([str(row[LIVE_LOG_KEY]) for row in rows
                     if isinstance(row, dict) and row.get(LIVE_LOG_KEY)], None)
        except Exception as error:  # noqa: BLE001 - recorded, not fatal
            return [], f"{type(error).__name__}: {error}\n{traceback.format_exc()}"

    @staticmethod
    def _not_ready(error: BaseException) -> bool:
        """True for the transient "run not visible yet" fetch error.

        The remote run is created by the kernel *after* the push returns, so the
        first poll(s) can beat ``wandb.init``; that is a startup lag, not a
        failure of the run.
        """
        return type(error).__name__ in {"RunNotFoundError", "CommError"}

    def stream(self, *, max_polls: int) -> Iterator[dict[str, Any]]:
        """Yield bounded-cadence updates, stopping at a terminal run state.

        Each update carries only the text produced since the previous poll
        (``new_output``) so a caller appends without duplicating the log. The
        live ``log/line`` channel is primary — the only surface W&B exposes
        while the run is live — and ``output.log`` is the fallback, used only
        until the first live batch arrives. A run the kernel has not registered
        yet yields a ``pending`` update (recorded, never fatal) and is retried
        at the same cadence.
        """
        emitted = 0
        emitted_live = 0
        live_seen = False
        for _ in range(max(1, int(max_polls))):
            try:
                update = self.read_once()
            except Exception as error:  # noqa: BLE001 - re-raised unless transient
                if not self._not_ready(error):
                    raise
                yield {"run": self.path, "state": "pending", "metrics": {},
                       "output": "", "live": [], "new_output": "",
                       "console_error": f"{type(error).__name__}: {error}",
                       "live_error": None}
                time.sleep(self.poll_seconds)
                continue
            batches = update.get("live") or []
            new_live = "".join(batches[emitted_live:])
            emitted_live = len(batches)
            live_seen = live_seen or bool(batches)
            if new_live:
                update["new_output"] = new_live
            elif live_seen:
                update["new_output"] = ""
            else:
                lines = (update["output"] or "").splitlines(keepends=True)
                update["new_output"] = "".join(lines[emitted:])
                emitted = len(lines)
            yield update
            if update["state"] in _WANDB_TERMINAL_STATES:
                return
            time.sleep(self.poll_seconds)

