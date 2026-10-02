"""Rebuild every full-training input without training or changing smoke inputs.

Run from ER: PYTHONPATH=src .venv/bin/python -m training.prepare_all
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

STAGES = ('dedupe', 'cross_country_pairs', 'number_reference', 'verify_reference',
          'canonical_and_gates', 'gate_census', 'labeled_pairs', 'validation',
          'graph_inputs', 'full_bundle', 'verify_handoff')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_stage_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest['status'] != 'complete':
        raise ValueError(f'Incomplete prerequisite: {path}')
    for entry in manifest['inputs'] + manifest['outputs']:
        if sha256(entry['path']) != entry['sha256']:
            raise ValueError(f'Stale prerequisite: {entry["path"]}')


def refresh_gate_census(gate_csv, config_path, report_path):
    import pandas as pd
    frame = pd.read_csv(gate_csv, dtype={'gtin1': str, 'gtin2': str})
    if frame.duplicated(['gtin1', 'gtin2']).any():
        raise ValueError('Duplicate candidate pairs')
    if set(frame.gate_decision) - {'hard_no', 'proceed', 'fallback'}:
        raise ValueError('Unknown gate decisions')
    counts = {'total_pairs': len(frame), **{
        key: int(frame.gate_decision.eq(key).sum())
        for key in ('hard_no', 'proceed', 'fallback')}}
    pattern = r'(  gate_census_pin:\n)(    total_pairs: \d+\n    hard_no: \d+\n    proceed: \d+\n    fallback: \d+\n)'
    text, n = re.subn(pattern, lambda match: match.group(1) + ''.join(
        f'    {key}: {value}\n' for key, value in counts.items()), Path(config_path).read_text())
    if n != 1:
        raise ValueError('Expected exactly one configured gate census')
    Path(report_path).write_text(json.dumps(counts, indent=2) + '\n')
    Path(config_path).write_text(text)
    return counts


def prepare_all(*, run_dir=None, resume_from='dedupe'):
    from core.common import F, RESULTS, TRAIN_ROOT, TRAINING_CONFIG_PATH, resolve_model
    root = Path(TRAIN_ROOT)
    run_dir = Path(run_dir or RESULTS / 'training_prep' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = RESULTS / 'training_prep.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another training preparation is already running') from None
        if resume_from == 'validation':
            for name in ('data_prep', 'labeled_pairs'):
                verify_stage_manifest(RESULTS / 'manifests' / (name + '.json'))
        checkpoint = resolve_model('minilm_l6')
        setup = root / 'data/track_setup'
        bundle = root / 'data/prepared/full/worker_1_baseline.pkl.gz'
        smoke = root / 'data/prepared/smoke_200'
        smoke_before = {str(path): sha256(path) for path in smoke.rglob('*') if path.is_file()}
        env = os.environ.copy()
        env['PYTHONPATH'] = str(root / 'src') + os.pathsep + str(root)
        env['MLFLOW_TRACKING_URI'] = 'off'
        env['EUROMONITOR_SHARED_BASE_DATA'] = str(run_dir / 'shared_base.pkl')
        env.pop('WANDB_API_KEY', None)
        completed = []
        manifest = {'status': 'running', 'resume_from': resume_from,
                    'training_started': False, 'smoke_updated': False,
                    'stages': completed, 'run_dir': str(run_dir),
                    'shared_base_payload': env['EUROMONITOR_SHARED_BASE_DATA']}
        manifest_path = run_dir / 'manifest.json'
        def publish():
            temporary = manifest_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(manifest, indent=2) + '\n')
            temporary.replace(manifest_path)
        def run(name, arguments):
            print(f'[prepare] {name} -> {run_dir / (name + ".log")}', flush=True)
            with (run_dir / (name + '.log')).open('w') as log:
                subprocess.run([sys.executable, *arguments], cwd=root, env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        def archive(path, name):
            if path.exists():
                destination = run_dir / 'before' / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    raise FileExistsError(destination)
                shutil.move(str(path), str(destination))
        publish()
        try:
            for name in STAGES[STAGES.index(resume_from):]:
                if name == 'dedupe':
                    run(name, ['-m', 'training.dedupe'])
                elif name == 'cross_country_pairs':
                    run(name, ['-m', 'training.build_second04_pairs'])
                elif name == 'number_reference':
                    run(name, ['-m', 'training.build_reference'])
                elif name == 'verify_reference':
                    run(name, ['-m', 'training.build_reference', '--verify'])
                elif name == 'canonical_and_gates':
                    run(name, ['-m', 'training.data_prep'])
                elif name == 'gate_census':
                    manifest['gate_census'] = refresh_gate_census(
                        F['gate_results'], TRAINING_CONFIG_PATH, run_dir / 'gate_census.json')
                elif name == 'labeled_pairs':
                    run(name, ['-m', 'training.labeled_pairs'])
                elif name == 'validation':
                    run(name, ['-m', 'training.build_final_validation'])
                elif name == 'graph_inputs':
                    archive(setup, 'track_setup')
                    run(name, ['-m', 'graph_tracks.setup', '--output', str(setup),
                               '--text-checkpoint', str(checkpoint)])
                elif name == 'full_bundle':
                    archive(bundle, bundle.name)
                    archive(bundle.with_suffix(bundle.suffix + '.json'), bundle.name + '.json')
                    run(name, ['-m', 'training.train', '--dataset', str(F['dataset_deduped']),
                               '--payload', 'full', '--prepare-bundle', str(bundle),
                               '--no-mask-effect', '--no-plot'])
                    target = setup / 'text_prepared.pkl.gz'
                    shutil.copy2(bundle, target)
                    shutil.copy2(bundle.with_suffix(bundle.suffix + '.json'),
                                 target.with_suffix(target.suffix + '.json'))
                else:
                    from training.prepared_bundle import load_prepared_bundle
                    from graph_tracks.text_cache import checkpoint_hash
                    header, prepared = load_prepared_bundle(bundle)
                    for key in ('canonical_records', 'gate_results', 'labeled_pairs'):
                        if prepared[key + '_csv'] != Path(F[key]).read_bytes():
                            raise ValueError(f'Bundle contains stale {key}')
                    graph = json.loads((setup / 'setup_manifest.json').read_text())
                    for key, field in [('dataset_deduped', 'source_catalog_sha256'),
                                       ('labeled_pairs', 'labeled_pairs_sha256')]:
                        if graph[field] != sha256(F[key]):
                            raise ValueError(f'Graph inputs contain stale {key}')
                    if graph['text_checkpoint_sha256'] != checkpoint_hash(Path(checkpoint)):
                        raise ValueError('Graph checkpoint hash does not match')
                    if {str(path): sha256(path) for path in smoke.rglob('*') if path.is_file()} != smoke_before:
                        raise ValueError('Smoke files changed during full preparation')
                    keys = ('dataset_deduped', 'sku_to_rep', 'number_reference',
                            'second04_pairs_positive', 'canonical_records', 'gate_results',
                            'labeled_pairs', 'final_validation', 'validation_fold_map')
                    paths = [Path(F[key]) for key in keys] + list(setup.rglob('*.csv')) + [bundle]
                    manifest['outputs'] = {str(path): {'sha256': sha256(path), 'bytes': path.stat().st_size}
                                           for path in paths}
                    manifest['bundle'] = header.model_dump(mode='json')
                    manifest['smoke_unchanged_verified'] = True
                completed.append(name)
                publish()
            manifest['status'] = 'complete'
            publish()
            print(f'[prepare] complete -> {manifest_path}', flush=True)
            return manifest_path
        except BaseException as error:
            manifest.update(status='failed', failed_stage=name, error=str(error))
            publish()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--resume-from', choices=['dedupe', 'validation'], default='dedupe',
                        help='validation requires verified current CSV stage manifests')
    args = parser.parse_args()
    prepare_all(run_dir=args.run_dir, resume_from=args.resume_from)


if __name__ == '__main__':
    main()
