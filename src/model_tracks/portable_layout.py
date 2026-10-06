"""src/model_tracks/portable_layout.py — the SSOT for every portable suite path.

Four consumers previously composed the same paths independently (package
shipper, ablation cohort consumer, run re-staging, preflight membership) and
a rename desynced them into plausibly-silent FileNotFoundErrors. Every
composition now lives here: the shipper and the consumer literally call the
same method, so ship keys and consume paths cannot drift.
"""
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class PortableLayout(BaseModel):
    """One resolver for the portable suite content layout.

    INTERNED STRINGS — none: the templates are read from config/paths.yaml
    layouts: (suite_package_shared), the suffix is the shared_graph_data
    contract, the local setup dir comes from training.yaml — the class owns
    NO literals beyond the suffix name, which mirrors
    model_tracks.shared_graph_data.CLEAN_BACKUP_SUFFIX via import.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    @staticmethod
    def _shared_template() -> str:
        from core.common import LAYOUTS
        layout = LAYOUTS['suite_package_shared']
        if layout.root != 'repo' or layout.fields:
            raise ValueError('suite_package_shared must be a static repo layout')
        return layout.template

    @classmethod
    def clean_suffix(cls) -> str:
        """The name-suffix shared_graph_data backs the unprojected gates under."""
        from model_tracks.shared_graph_data import CLEAN_BACKUP_SUFFIX
        return CLEAN_BACKUP_SUFFIX

    @classmethod
    def portable_setup(cls) -> Path:
        """The suite config's portable setup dir (the suite_package_shared layout)."""
        return Path(cls._shared_template())

    @classmethod
    def portable_clean_backup(cls) -> Path:
        """Where the clean-gates backup must SHIP for the consumer suite config."""
        shared = cls.portable_setup()
        return shared.parent / (shared.name + cls.clean_suffix())

    @classmethod
    def local_clean_backup(cls, setup: Path) -> Path:
        """Where the CPU lane discovers the backup beside ITS cfg.setup_dir.

        Discovery keeps the local name: packaging may run with a different
        setup_dir than the portable suite declares (local vs portable name
        gap is resolved exactly here, once)."""
        return setup.parent / (setup.name + cls.clean_suffix())

    @classmethod
    def consumer_clean_backup(cls, setup: Path) -> Path:
        """Where the consumer suite composes the backup beside ITS setup_dir."""
        return setup.parent / (setup.name + cls.clean_suffix())

    @classmethod
    def ship_key(cls, path: Path, *, from_local: Path) -> str:
        """Portable archive member key for one backup file.

        `path` is a file under `from_local` (the local clean backup); the
        portable base replaces the local root so the consumer's compose sees
        the file no matter what the local setup dir was named.
        """
        return str(cls.portable_clean_backup() / path.relative_to(from_local))


def _verify_invariants() -> None:
    """Self-test on any tree with the layout instantiated (call in tests)."""
    layout = PortableLayout()
    shared = layout.portable_setup()
    assert shared.as_posix() == 'data/model_tracks/shared', shared
    local = layout.local_clean_backup(Path('data/track_setup'))
    assert local == Path('data/track_setup__clean_shared_inputs'), local
    consumer = layout.consumer_clean_backup(shared)
    assert consumer == layout.portable_clean_backup(), (consumer, layout.portable_clean_backup())
    # ship key maps one local backup file onto the consumer's path
    sample = local / 'pairs.csv'
    key = layout.ship_key(sample, from_local=local)
    assert Path(key) == consumer / 'pairs.csv', key
