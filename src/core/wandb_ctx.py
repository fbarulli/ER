"""Optional W&B mirror for non-sensitive training and HPO evidence."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from core.common import training_cfg


class WandbCtx:
    """One run: cloud telemetry is always on for experiment lanes. A missing
    key fails loud — no silent local-only degradation (owner order 2026-10-07
    'bake wandb in, always on'; training/prepared tracking.wandb.mode owns the
    dashboard lane, never this class).

    The ``--prepare-bundle`` lane runs no experiment (it only materializes the
    training inputs), so it builds this context through ``for_input_preparation``
    — a no-op that neither requires nor opens W&B. That path records why
    tracking is off; it is never silent."""

    def __init__(self, name: str, *, track: bool = True) -> None:
        spec = training_cfg().tracking.wandb
        self._run = None
        self._name, self._project, self._mode = name, spec.project, spec.mode
        self._enabled = (
            track
            and spec.mode != "disabled"
            and bool(os.environ.get("WANDB_API_KEY"))
        )
        if track and not self._enabled:
            raise RuntimeError(
                '[wandb] WANDB_API_KEY absent while tracking.wandb.mode='
                + str(spec.mode)
                + ' — the cloud mirror is always on; export '
                  'WANDB_API_KEY (or drop it in .env) before training')

    @classmethod
    def for_input_preparation(cls, name: str) -> WandbCtx:
        """The no-op context for a lane that prepares inputs, not an experiment.

        ``--prepare-bundle`` only materializes the training inputs, so
        WANDB_API_KEY must never be a precondition of preparing a bundle.
        Tracking is intentionally off; the reason is recorded here (fail-soft
        only with a recorded reason — never a silent local-only run).
        """
        context = cls(name, track=False)
        print(
            "[wandb] disabled for input preparation (--prepare-bundle): the "
            "lane runs no experiment, so WANDB_API_KEY is not required",
            flush=True,
        )
        return context

    def __enter__(self):
        if not self._enabled:
            return self
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
