"""Portable graph result ZIP for standalone Colab/manual workers.

Contains model checkpoints, prepared inputs, reports and DVC pointers. Secrets
and DVC's duplicated payload are excluded. Local DVC cache is opt-in; raw result
files remain usable for inference/resume without the cache.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from graph_tracks.artifacts import name
from graph_tracks.data import file_hash


def bundle(source: Path, output: Path, *, include_dvc_cache=False):
    if output.exists():
        raise FileExistsError(output)
    manifests = list(source.glob('*__run_manifest.json'))
    if len(manifests) != 1:
        raise ValueError('bundle needs exactly one track run manifest')
    run = json.loads(manifests[0].read_text())
    track = run['track']
    files = []
    for path in sorted(source.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if 'wandb' in relative.parts or '.git' in relative.parts or path.name in {'.env', 'config.local'}:
            continue
        # DVC payload duplicates the raw results/checkpoints outside its workspace.
        if any(part == name(track, 'payload') for part in relative.parts):
            continue
        if not include_dvc_cache and '.dvc' in relative.parts and 'cache' in relative.parts:
            continue
        if path.is_symlink():
            raise ValueError('result bundle must not contain symbolic links')
        files.append(path)
    metadata = {'schema': 'er-graph-bundle-v1', 'track': track, 'run_tag': run['run_tag'],
                'dvc_cache_included': include_dvc_cache, 'wandb_logs_included': False}
    from core.portable_archive import write_archive
    return write_archive(output, {str(path.relative_to(source)):path for path in files},
                         manifest_name=name(track,'bundle_manifest.json'), metadata=metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-dvc-cache', action='store_true')
    args = parser.parse_args()
    bundle(args.source, args.output, include_dvc_cache=args.include_dvc_cache)

if __name__ == '__main__':
    main()
