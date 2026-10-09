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


class WandbRunReader:
    """Read-only live repository over one W&B run (metrics + console output).

    The writer counterpart is :class:`WandbCtx`. Given the configured
    ``tracking.wandb.project`` and a run tag, this streams the run's latest
    logged metrics and the live ``output.log`` console. The ``WANDB_API_KEY``
    is resolved through the canonical credential owner
    (``core.credentials.CredentialStore``) and exported to the wandb client's
    environment silently — never logged or printed. Polling honours the
    config-declared cadence (``tracking.wandb.poll_seconds``); every API error
    fails loud with the full traceback, never swallowed.
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
        """One snapshot: run state, latest metrics, and the full console log."""
        run = self._client().run(self.path)
        output, console_error = self._console(run)
        return {
            "run": self.path,
            "state": str(getattr(run, "state", "unknown")),
            "metrics": dict(getattr(run, "summary", {}) or {}),
            "output": output,
            "console_error": console_error,
        }

    @staticmethod
    def _console(run) -> tuple[str, str | None]:
        """Read ``output.log``; a run without console capture is empty, recorded."""
        try:
            handle = run.file("output.log").download(replace=True)
        except Exception as error:  # noqa: BLE001 - an absent log is not fatal
            return "", f"{type(error).__name__}: {error}\n{traceback.format_exc()}"
        try:
            return handle.read(), None
        finally:
            handle.close()

    def stream(self, *, max_polls: int) -> Iterator[dict[str, Any]]:
        """Yield bounded-cadence updates, stopping at a terminal run state.

        Each update carries only the console lines produced since the previous
        poll (``new_output``) so a caller appends without duplicating the log.
        """
        emitted = 0
        for _ in range(max(1, int(max_polls))):
            update = self.read_once()
            lines = (update["output"] or "").splitlines(keepends=True)
            update["new_output"] = "".join(lines[emitted:])
            emitted = len(lines)
            yield update
            if update["state"] in _WANDB_TERMINAL_STATES:
                return
            time.sleep(self.poll_seconds)

