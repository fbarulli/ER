#!/usr/bin/env python3
"""Verify a sealed model-tracks suite archive after it has arrived locally.

Re-checks the .sha256 sidecar and re-runs the sealing-time contract
(model_tracks.resume.validate_completed_suite_archive: byte integrity of
every member, run tag, per-track completion markers, exactly one
calibrated report manifest per track, saved-ablation binding), then writes
the machine-readable outcome to the <run_tag>.verification.json sidecar
that dashboard/training_reports.py renders under /training.

Usage:
  PYTHONPATH=src .venv/bin/python scripts/verify_suite_archive.py 1004T155148611490Z
  PYTHONPATH=src .venv/bin/python scripts/verify_suite_archive.py \\
      results/model_tracks/1004T155148611490Z.zip --config results/model_tracks/suite.yaml

Exit 0 when the archive verifies, 1 when a check fails or the archive is
unreadable.  The positional argument is a run tag (resolved under
results/model_tracks and results/graph_tracks) or a direct archive path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_tracks import archive_verification
from model_tracks.config import SuiteConfig

import yaml


def resolve(archive_arg: str) -> Path:
    direct = Path(archive_arg)
    if direct.exists():
        return direct
    for base in (Path('results/model_tracks'), Path('results/graph_tracks')):
        for ending in ('.zip', '.tar.zst'):
            candidate = base / f'{archive_arg}{ending}'
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f'no suite archive for {archive_arg!r} under results/')


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
    settings = None
    if args.config is not None:
        settings = SuiteConfig.model_validate(yaml.safe_load(args.config.read_text()))
    result = archive_verification.verification_result(archive, settings=settings)
    path = archive_verification.write_verification(archive, result)
    print(json.dumps(result, indent=2))
    print(f'verification written to {path}', file=sys.stderr)
    return 0 if result['status'] == 'verified' else 1


if __name__ == '__main__':
    sys.exit(main())
