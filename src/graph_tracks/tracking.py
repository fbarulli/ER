"""W&B graph tracking with explicit offline support; no MLflow dependency."""
from __future__ import annotations
import os
from pathlib import Path
from graph_tracks.artifacts import name


class GraphWandb:
    def __init__(self, spec, track: str, run_tag: str, output: Path):
        self.spec, self.track, self.run_tag, self.output = spec, track, run_tag, output
        self.run = None

    def __enter__(self):
        if self.spec.mode == 'disabled':
            return self
        if self.spec.mode == 'online' and not os.environ.get('WANDB_API_KEY'):
            raise RuntimeError('online W&B requires WANDB_API_KEY; choose offline explicitly for local runs')
        import wandb
        self.run = wandb.init(project=self.spec.project, mode=self.spec.mode,
            name=name(self.track, self.run_tag), group=self.track, tags=[self.track, 'graph-tracks'],
            dir=str(self.output), reinit='finish_previous',
            settings=wandb.Settings(x_disable_stats=True, x_disable_machine_info=True))
        self.run.define_metric('epoch')
        self.run.define_metric('*', step_metric='epoch')
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.run is not None:
            self.run.finish(exit_code=1 if exc_type else 0)
        return False

    @property
    def run_id(self):
        return self.run.id if self.run is not None else None

    @property
    def run_url(self):
        return self.run.url if self.run is not None and self.spec.mode == 'online' else None

    def log_config(self, values):
        if self.run is not None:
            self.run.config.update(values, allow_val_change=True)

    def log_metrics(self, values, *, step=None):
        if self.run is not None:
            self.run.log({**values, **({'epoch': step} if step is not None else {})})

    def set_summary(self, values):
        if self.run is not None:
            self.run.summary.update(values)

    def log_artifacts(self, paths, artifact_stem='results'):
        if self.run is None:
            return
        import wandb
        artifact = wandb.Artifact(name(self.track, f'{self.run_tag}-{artifact_stem}'), type='training-result',
                                 metadata={'track': self.track, 'run_tag': self.run_tag})
        for path in paths:
            path = Path(path)
            if path.is_dir():
                artifact.add_dir(str(path), name=path.name)
            else:
                artifact.add_file(str(path), name=path.name)
        self.run.log_artifact(artifact)
