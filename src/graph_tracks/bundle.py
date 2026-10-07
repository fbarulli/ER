"""Portable graph result ZIP for standalone Colab/manual workers.

Contains model checkpoints, prepared inputs, reports and DVC pointers. Secrets
and DVC's duplicated payload are excluded. Local DVC cache is opt-in; raw result
files remain usable for inference/resume without the cache.

One responsibility per unit: _require_absent_output, _load_run,
_select_bundle_files, bundle (orchestrator), main.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.run_log import RunLogger
from graph_tracks.artifacts import name
from graph_tracks.data import file_hash

_LOG = RunLogger(__name__)

_SECRET_NAMES = frozenset({'.env', 'config.local'})
_EXCLUDED_TREE_PARTS = frozenset({'wandb', '.git'})


def _raise_if_output_exists(output: Path) -> None:
    """The bundle name is exclusive: refuse to overwrite any existing path."""
    if output.exists():
        raise FileExistsError(output)


def _load_run(source: Path) -> tuple[dict, Path]:
    """Exactly one track run manifest identifies the bundle."""
    manifests = list(source.glob('*__run_manifest.json'))
    if len(manifests) != 1:
        raise ValueError('bundle needs exactly one track run manifest')
    return json.loads(manifests[0].read_text()), manifests[0]


def _is_bundle_member(relative: Path, track: str, *, include_dvc_cache: bool) -> bool:
    """Keep raw result files; prune secrets, wandb/git trees and DVC payload."""
    # DVC payload duplicates the raw results/checkpoints outside its workspace.
    if any(part == name(track, 'payload') for part in relative.parts):
        return False
    if not include_dvc_cache and '.dvc' in relative.parts and 'cache' in relative.parts:
        return False
    return True


def _select_bundle_files(source: Path, track: str, *,
                         include_dvc_cache: bool) -> dict[str, Path]:
    """The flat member map (path-relative) the archive will carry."""
    files: dict[str, Path] = {}
    for path in _LOG.progress(sorted(source.rglob('*')), desc='bundle.scan',
                              unit='file'):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if set(relative.parts) & _EXCLUDED_TREE_PARTS or path.name in _SECRET_NAMES:
            continue
        if path.is_symlink():
            raise ValueError('result bundle must not contain symbolic links')
        if not _is_bundle_member(relative, track, include_dvc_cache=include_dvc_cache):
            continue
        files[str(relative)] = path
    return files


def bundle(source: Path, output: Path, *, include_dvc_cache=False):
    """Ship one graph track's results as a self-describing portable archive."""
    _raise_if_output_exists(output)
    run, _ = _load_run(source)
    track = run['track']
    files = _select_bundle_files(source, track, include_dvc_cache=include_dvc_cache)
    metadata = {'schema': 'er-graph-bundle-v1', 'track': track,
                'run_tag': run['run_tag'],
                'dvc_cache_included': include_dvc_cache,
                'wandb_logs_included': False}
    from core.portable_archive import write_archive
    return write_archive(output, files,
                         manifest_name=name(track, 'bundle_manifest.json'),
                         metadata=metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-dvc-cache', action='store_true')
    args = parser.parse_args()
    bundle(args.source, args.output, include_dvc_cache=args.include_dvc_cache)


if __name__ == '__main__':
    main()
