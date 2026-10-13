from pathlib import Path
from typing import Literal
from core.execution_policy import AggregationBackend, OptimizerBackend
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: The text trainer's switchable objectives (training.train_prepared ``--loss``).
#: The GNN track has ONE fixed composite objective (graph_tracks.train), so it
#: has no loss knob; only the text lane varies. See docs/colab-lane.md.
TextLoss = Literal['contrastive', 'mnrl', 'triplet']

#: The suite taxonomy (trained lanes + the cascade combinator). Declared HERE
#: because the selection rule needs it and ``model_tracks.resume`` imports this
#: module; resume keeps the ordered TRACKS tuples and re-exports this name.
Track = Literal['text', 'gnn_only', 'cascade']


class SuiteConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', validate_default=True, allow_inf_nan=False)
    setup_dir: str = Field(min_length=1)
    text_bundle: str = Field(min_length=1)
    text_model: str = 'minilm_l6'
    #: The text lane's training objective. ``None`` (the default) omits
    #: ``--loss`` so the trainer's OWN declared default (config/training.yaml
    #: ``loss:``) stays the single source of truth; a suite declares one of the
    #: switchable objectives only to run a text-loss variation.
    text_loss: TextLoss | None = None
    epochs: int = Field(default=10, ge=1)
    device: Literal['cpu', 'cuda'] = 'cuda'
    #: Which suite tracks this launch trains, in the taxonomy order
    #: (``model_tracks.resume.TRACKS``). The default is the full cohort; a
    #: launch selects a subset to run one model per session. The cascade is a
    #: combinator over the SAME run's trained artifacts, so it is selectable
    #: only with both trained lanes.
    tracks: tuple[Track, ...] = ('text', 'gnn_only', 'cascade')
    input_archive_format: Literal['zip', 'tar.zst'] = 'tar.zst'
    result_archive_format: Literal['zip', 'tar.zst'] = 'tar.zst'
    schedule: Literal['parallel'] = 'parallel'
    # Only the trained lanes (text, gnn_only) sit behind the MPS barrier; the
    # cascade is a post-hoc combinator over their artifacts and must not be
    # counted as a parallel worker. The default is the trained-lane count.
    # ``3`` stays accepted so older suite manifests that predate the cascade
    # replacing the third trained lane keep validating unchanged.
    max_parallel: Literal[2, 3] = 2
    gpu_parallel_backend: Literal['mps'] = 'mps'
    # Missing fields in old suite manifests preserve their original execution.
    gpu_optimizer_backend: OptimizerBackend = 'auto'
    gpu_graph_aggregation_backend: AggregationBackend = 'index_add'
    memory_reservations_gb: dict[str, float] = Field(default_factory=dict)
    gpu_headroom_gb: float = Field(default=2.0, ge=0)
    report_test: bool = False
    publish_dvc: bool = False
    publish_git: bool = True
    profiling: bool = False
    post_training_ablation: bool = False
    ablation_config: str = Field(default='config/attribute_ablation.yaml', min_length=1)

    @field_validator('ablation_config')
    @classmethod
    def portable_ablation_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or '..' in path.parts or not path.parts:
            raise ValueError('ablation_config must be a project-relative file path')
        return path.as_posix()

    def graph_execution_overrides(self) -> dict[str, str]:
        if self.device != 'cuda':
            return {}
        return {'optimizer_backend': self.gpu_optimizer_backend,
                'aggregation_backend': self.gpu_graph_aggregation_backend}

    @property
    def dvc_enabled(self) -> bool:
        return self.publish_dvc or self.publish_git

    @model_validator(mode='after')
    def check_tracks(self):
        from model_tracks.resume import TRACKS, TRAINING_TRACKS
        unknown = set(self.tracks) - set(TRACKS)
        if unknown:
            raise ValueError(f'unknown suite track(s): {", ".join(sorted(unknown))}')
        if not self.tracks:
            raise ValueError('a suite must select at least one track')
        if len(self.tracks) != len(set(self.tracks)):
            raise ValueError('suite tracks must be unique')
        if 'cascade' in self.tracks:
            missing = set(TRAINING_TRACKS) - set(self.tracks)
            if missing:
                raise ValueError(
                    'the cascade composes the SAME run\'s trained artifacts; select '
                    f'the trained lanes too: {", ".join(sorted(missing))}')
        return self

    @model_validator(mode='after')
    def check_memory(self):
        from model_tracks.resume import TRACKS
        if set(self.memory_reservations_gb) - set(TRACKS):
            raise ValueError('unknown track memory reservation')
        if any(v <= 0 for v in self.memory_reservations_gb.values()):
            raise ValueError('memory reservations must be positive measured peaks')
        if self.memory_reservations_gb and set(self.memory_reservations_gb) != set(TRACKS):
            raise ValueError('provide measured memory peaks for all three tracks together')
        return self


def load_config(path: Path) -> SuiteConfig:
    """The one suite-manifest loader: read + validate live in core.common."""
    from core.common import load_validated_yaml

    return load_validated_yaml(
        path, SuiteConfig, label='Model-track suite configuration')


def setup_root(cfg: SuiteConfig) -> Path:
    """The canonical prepared-setup root: the suite ``setup_dir`` under TRAIN_ROOT.

    The ONE spelling every model_tracks consumer reads. The packaging stage
    stages a disposable copy and passes that copy's root explicitly (DI) to the
    consumers it drives, so this derivation stays the source root — never a
    second, independently re-derived root that drifts from the one written to.
    """
    from core.common import TRAIN_ROOT

    return (Path(TRAIN_ROOT) / cfg.setup_dir).resolve()
