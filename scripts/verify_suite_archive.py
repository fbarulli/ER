#!/usr/bin/env python3
"""Verify a sealed model-tracks suite archive after it has arrived locally.

Re-checks the .size sidecar and re-runs the sealing-time contract
(model_tracks.resume.validate_completed_suite_archive: byte integrity of
every member, run tag, per-track completion markers, exactly one
calibrated report manifest per track, saved-ablation binding), then writes
the machine-readable outcome to the <run_tag>.verification.json sidecar
that dashboard/training_reports.py renders under /training.

Usage:
  PYTHONPATH=src .venv/bin/python scripts/verify_suite_archive.py 1004T155148611490Z
  PYTHONPATH=src .venv/bin/python scripts/verify_suite_archive.py \
      results/model_tracks/1004T155148611490Z.tar.zst --config results/model_tracks/suite.yaml

Exit 0 when the archive verifies, 1 when a check fails or the archive is
unreadable.  The positional argument is a run tag (resolved under
results/model_tracks and results/graph_tracks) or a direct archive path.

RESPONSIBILITY MAP (single-responsibility decomposition; behaviour pinned)
-------------------------------------------------------------------------
- :class:`ArchiveResolver` — run tag or direct path -> the archive file.
- :class:`SuiteSettings` — the optional --config suite contract.
- :func:`main` — verification dispatch + the machine-readable sidecar print.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_tracks import archive_verification
from model_tracks.config import SuiteConfig

import yaml


class ArchiveResolver:
    """Run tag or direct path -> the suite archive (fail loudly)."""

    BASES = ('results/model_tracks', 'results/graph_tracks')
    ENDINGS = ('.zip', '.tar.zst')

    def __init__(self, archive_arg: str):
        self._arg = archive_arg
        self.archive = self._resolve()

    def _resolve(self) -> Path:
        direct = Path(self._arg)
        if direct.exists():
            return direct
        for base in self.BASES:
            for ending in self.ENDINGS:
                candidate = Path(base) / f'{self._arg}{ending}'
                if candidate.exists():
                    return candidate
        raise FileNotFoundError(f'no suite archive for {self._arg!r} under results/')


def resolve(archive_arg: str) -> Path:
    """Back-compat face for :class:`ArchiveResolver`."""
    return ArchiveResolver(archive_arg).archive


class SuiteSettings:
    """The optional --config suite contract the archive must have run under."""

    @staticmethod
    def load(config_path: Path | None):
        if config_path is None:
            return None
        return SuiteConfig.model_validate(yaml.safe_load(config_path.read_text()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', help='run tag or archive path')
    parser.add_argument('--config', type=Path, default=None,
                        help='suite YAML whose configuration the archive must have run under')
    args = parser.parse_args()
    try:
        archive = resolve(args.archive)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    result = archive_verification.verification_result(
        archive, settings=SuiteSettings.load(args.config))
    path = archive_verification.write_verification(archive, result)
    print(json.dumps(result, indent=2))
    print(f'verification written to {path}', file=sys.stderr)
    return 0 if result['status'] == 'verified' else 1


if __name__ == '__main__':
    sys.exit(main())
