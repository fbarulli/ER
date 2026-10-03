"""Portable suite identity and verified worker completion for recovery."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

TRACKS = ('text', 'gnn_only', 'hybrid')


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def suite_identity(cfg, inputs: dict, run_tag: str) -> dict:
    from core.common import TRAIN_ROOT
    settings = cfg.model_dump()
    for key in ('setup_dir', 'text_bundle'):
        settings.pop(key, None)
    # Paths and runtime reports may change when the archive moves machines.
    # Frozen populations, checkpoint and model code must remain identical.
    text = inputs['text']
    identity = {key: text[key] for key in ('bundle_sha256', 'payload', 'masking_profile', 'rows')}
    for key in ('source_catalog_sha256', 'labeled_pairs_sha256'):
        if key in inputs:
            identity[key] = inputs[key]
    setup = (TRAIN_ROOT / cfg.setup_dir).resolve()
    frozen_names = {'eligible_catalog.csv', 'listing_splits.csv', 'listing_pairs.csv',
                    'shared_minilm__embeddings.npz', 'setup_manifest.json'}
    identity['setup'] = {path.relative_to(setup).as_posix(): digest(path)
                         for path in setup.rglob('*') if path.is_file()
                         and (path.parent == setup and path.name in frozen_names
                              or 'prepared' in path.relative_to(setup).parts and path.suffix in {'.csv', '.json'})}
    implementation = {}
    for directory in ('training', 'graph_tracks', 'core', 'model_tracks'):
        for path in sorted((TRAIN_ROOT / 'src' / directory).glob('*.py')):
            implementation[path.relative_to(TRAIN_ROOT).as_posix()] = digest(path)
    for name in ('training.yaml', 'identity_dimensions.yaml', 'identity_reviews.json', 'vocabulary.json'):
        path = TRAIN_ROOT / 'config' / name
        implementation[f'config/{name}'] = digest(path)
    return {'schema': 'er-suite-resume-v1', 'run_tag': run_tag, 'config': settings,
            'inputs': identity, 'implementation': implementation}


def validate_suite(output: Path, identity: dict) -> None:
    path = output / 'suite_manifest.json'
    if not path.is_file():
        raise ValueError('resume requires an existing suite manifest')
    prior = json.loads(path.read_text())
    if prior.get('resume_identity') != identity:
        raise ValueError('resume provenance mismatch: run, configuration, frozen inputs or implementation changed')


def artifact_files(output: Path):
    excluded = {'wandb', 'mlruns', 'profiles', '_artifact_publications', '.dvc', '.git'}
    return [path for path in output.rglob('*') if path.is_file() and not path.is_symlink()
            and not excluded.intersection(path.relative_to(output).parts)
            and path.name not in {'track_complete.json', 'track_inventory.json', 'worker.yaml', 'worker_events.jsonl'}
            and not path.name.endswith('.log')]


def record_completion(output: Path, track: str, *, postprocess_complete: bool = True) -> None:
    files = {path.relative_to(output).as_posix(): digest(path) for path in artifact_files(output)}
    if not files:
        raise ValueError(f'cannot complete empty track: {track}')
    target = output / 'track_inventory.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps({'track': track, 'files': files}, indent=2) + '\n')
    temporary.replace(target)
    marker = output / 'track_complete.json'
    temporary = marker.with_suffix('.tmp')
    temporary.write_text(json.dumps({'track': track, 'status': 'ok', 'postprocess_complete': postprocess_complete}) + '\n')
    temporary.replace(marker)


def completed_track(output: Path, track: str, *, postprocess_complete: bool = True) -> bool:
    marker = output / 'track_complete.json'
    if not marker.exists():
        return False
    if json.loads(marker.read_text()) != {'track': track, 'status': 'ok', 'postprocess_complete': postprocess_complete}:
        raise ValueError(f'invalid completion marker: {track}')
    inventory_path = output / 'track_inventory.json'
    if not inventory_path.is_file():
        raise ValueError(f'completion inventory missing: {track}')
    inventory = json.loads(inventory_path.read_text())
    if inventory.get('track') != track or not inventory.get('files'):
        raise ValueError(f'invalid completion inventory: {track}')
    for relative, expected in inventory['files'].items():
        path = output / relative
        if Path(relative).is_absolute() or '..' in Path(relative).parts or path.is_symlink():
            raise ValueError(f'unsafe completion artifact: {relative}')
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'completed artifact changed: {track}/{relative}')
    return True


def graph_checkpoint(output: Path, track: str, run_tag: str) -> Path | None:
    # Graph trainers isolate their own run inside the worker output root.
    # Retain flat-root compatibility for previously materialized trees.
    roots = [output, output / f'{track}__{run_tag}']
    paths = [path for folder in roots
             for path in (folder / '_checkpoints' / track / f'{run_tag}_f0').glob(
                 f'checkpoint-*/{track}__graph_model.pt')]
    return max(paths, key=lambda path: int(path.parent.name.split('-')[-1]), default=None)
