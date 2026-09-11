"""MlflowCtx — mlflow tracking with a LOCAL backend + artifact store.

Ported from second08_training_pipeline.MlflowCtx with one upgrade: the
owner's mandate is "training logs available locally", so when
MLFLOW_TRACKING_URI is unset the context STILL logs — to the local file
store at artifacts/mlruns (SSOT: paths.mlruns_dir in 00_config). Set
MLFLOW_TRACKING_URI to point elsewhere (server, Databricks) and it
follows; set MLFLOW_TRACKING_URI=off to truly disable.

One parent run per invocation; every fold/arm is a nested run; per-fold
metrics CSVs and plots are logged as artifacts.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path

from euromonitor.core.common import _path, load_config

_cfg = load_config()
# NO FALLBACK (owner Q27): paths.mlruns_dir is hard-required — a missing
# key must crash, not silently scatter runs into a default directory.
_MLRUNS = _path(_cfg["paths"]["mlruns_dir"])


class MlflowCtx:
    """No-op when MLFLOW_TRACKING_URI=off; local file store by default."""

    def __init__(self, experiment: str):
        uri = os.environ.get("MLFLOW_TRACKING_URI", "")
        self.enabled = uri.lower() != "off"
        self.experiment = experiment
        self._mlflow = None
        if self.enabled:
            import mlflow

            self._mlflow = mlflow
            if not uri:
                # LOCAL by default (owner mandate: training logs available
                # locally): sqlite backend + file artifact store under
                # artifacts/mlruns — mlflow 3.x deprecates the raw file
                # backend. Browse with:
                #   mlflow ui --backend-store-uri sqlite:///artifacts/mlruns/mlflow.db
                store = Path(_MLRUNS).resolve()
                store.mkdir(parents=True, exist_ok=True)
                uri = f"sqlite:///{store / 'mlflow.db'}"
                mlflow.set_tracking_uri(uri)
            # local artifact store (one place, beside the tracking db):
            # new experiments are created with this artifact location; the
            # env var also covers pre-existing sqlite experiments
            art_root = (Path(_MLRUNS) / "artifacts").resolve()
            os.environ["MLFLOW_ARTIFACT_DESTINATION"] = art_root.as_uri()
            exp = mlflow.get_experiment_by_name(experiment)
            if exp is None:
                mlflow.create_experiment(
                    experiment, artifact_location=art_root.as_uri()
                )
            mlflow.set_experiment(experiment)

    def __enter__(self):
        if self.enabled:
            self.parent = self._mlflow.start_run(run_name=self.experiment)
        return self

    def __exit__(self, *exc):
        if self.enabled:
            self._mlflow.end_run()
        return False

    @property
    def nested(self):
        """Context manager for a child run; no-op when disabled."""
        if self.enabled:
            return self._mlflow.start_run(nested=True)
        return nullcontext()

    def log_params(self, params: dict) -> None:
        if self.enabled:
            self._mlflow.log_params({k: str(v) for k, v in params.items()})

    def log_metrics(self, metrics: dict) -> None:
        if self.enabled:
            import numpy as np

            numeric = {
                k: float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float)) and np.isfinite(v)
            }
            if numeric:
                self._mlflow.log_metrics(numeric)

    def log_artifact(self, path, artifact_path: str | None = None) -> None:
        if self.enabled:
            self._mlflow.log_artifact(str(path), artifact_path=artifact_path)
