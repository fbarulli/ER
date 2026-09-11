"""Optional W&B mirror for non-sensitive training and HPO evidence."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from core.common import training_cfg


class WandbCtx:
    """One run when WANDB_API_KEY exists; explicit local-only otherwise."""

    def __init__(self, name: str):
        spec = training_cfg().tracking.wandb
        self.enabled = bool(os.environ.get("WANDB_API_KEY")) and spec.mode != "disabled"
        self._run = None
        self._name, self._project, self._mode = name, spec.project, spec.mode

    def __enter__(self):
        if not self.enabled:
            print("[wandb] disabled: WANDB_API_KEY absent; local MLflow remains active", flush=True)
            return self
        import wandb
        run_name = os.environ.get("WANDB_RUN_NAME", self._name)
        self._run = wandb.init(project=self._project, name=run_name, mode=self._mode)
        print(f"[wandb] run started: project={self._project} mode={self._mode}", flush=True)
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

    def set_summary(self, values: dict[str, Any]) -> None:
        if self._run is not None:
            self._run.summary.update(values)

    def log_artifact(self, path: str | Path, name: str) -> None:
        if self._run is not None:
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(f"W&B artifact missing: {p}")
            import wandb
            artifact = wandb.Artifact(name, type="training-result")
            artifact.add_file(str(p))
            self._run.log_artifact(artifact)
