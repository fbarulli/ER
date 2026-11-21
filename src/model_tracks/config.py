from pathlib import Path, PurePosixPath
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


class PackageIdentity(BaseModel):
    """The portable identity a sealed inputs package installs and trains under.

    ONE seam, derived everywhere: the packaging writer renders the suite config
    through ``model_tracks.package``'s ONE render contract, a launch renders the
    SAME contract through :meth:`from_suite`, and the VM reads it back through
    :meth:`from_archive`. ``setup_dir`` IS the member prefix every
    identity-owned member ships under, so a re-seal rebases the tree
    (:meth:`rebase_member`) instead of rebuilding the data; ``run_tag`` is the
    run identity the tree is installed under.
    """

    model_config = ConfigDict(extra='forbid', frozen=True)

    suite: SuiteConfig
    #: The suite run identity the package is installed under. The members carry
    #: no run tag (the archive name does); a bare package readback leaves it unset.
    run_tag: str | None = Field(default=None, pattern=r'^[A-Za-z0-9_-]+$')

    @model_validator(mode='after')
    def repo_relative_tree(self) -> 'PackageIdentity':
        """The identity-owned member prefix must never escape the repo root."""
        setup = PurePosixPath(self.suite.setup_dir)
        bundle = PurePosixPath(self.suite.text_bundle)
        for path in (setup, bundle):
            if path.is_absolute() or '..' in path.parts:
                raise ValueError(f'package identity paths must be repository-relative: {path}')
        if bundle.parent != setup and setup not in bundle.parents:
            raise ValueError(
                f'package text_bundle {bundle} must live inside its setup tree {setup}')
        return self

    @property
    def setup_dir(self) -> PurePosixPath:
        """The identity-owned member prefix (the suite's portable setup tree)."""
        return PurePosixPath(self.suite.setup_dir)

    def suite_identity(self) -> dict:
        """The identity dict a launch compares — the ONE render/readback contract."""
        return self.suite.model_dump(mode='json')

    def rebase_member(self, member: str, *, onto: 'PackageIdentity') -> str:
        """Map one shipped member from this identity's setup tree onto ``onto``'s.

        Only identity-owned members move; checkout-authoritative code/config and
        the frozen inputs keep their declared portable names.
        """
        source, target = self.setup_dir, onto.setup_dir
        path = PurePosixPath(member)
        if source == target or not (path == source or source in path.parents):
            return member
        if path == source:
            return str(target)
        return str(target / path.relative_to(source))

    def rebased_onto(self, onto: 'PackageIdentity') -> 'PackageIdentity':
        """This identity's data-binding identity with ``onto``'s install tree.

        The setup tree is WHERE a package installs, not what its data binds, so
        it is normalized away before the reuse/derivate/foreign verdict and
        rebased by the install itself.
        """
        return self.model_copy(update={'suite': self.suite.model_copy(update={
            'setup_dir': onto.suite.setup_dir, 'text_bundle': onto.suite.text_bundle})})

    @classmethod
    def from_suite(cls, cfg: SuiteConfig, *, run_tag: str | None = None,
                   root: str | None = None) -> 'PackageIdentity':
        """Render ``cfg``'s identity exactly as the packaging writer bakes it."""
        from model_tracks.package import render_suite_identity
        return cls(suite=render_suite_identity(cfg, root=root), run_tag=run_tag)

    @classmethod
    def from_archive(cls, archive: Path, *, run_tag: str | None = None) -> 'PackageIdentity':
        """The identity a sealed package was baked with (its ONE inline suite config)."""
        from model_tracks.package import baked_suite_identity
        return cls(suite=baked_suite_identity(archive), run_tag=run_tag)


def setup_root(cfg: SuiteConfig) -> Path:
    """The canonical prepared-setup root: the suite ``setup_dir`` under TRAIN_ROOT.

    The ONE spelling every model_tracks consumer reads. The packaging stage
    stages a disposable copy and passes that copy's root explicitly (DI) to the
    consumers it drives, so this derivation stays the source root — never a
    second, independently re-derived root that drifts from the one written to.
    """
    from core.common import TRAIN_ROOT

    return (Path(TRAIN_ROOT) / cfg.setup_dir).resolve()
