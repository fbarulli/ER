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
          'graph_inputs', 'full_bundle', 'suite_inputs', 'verify_handoff')


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


def prepare_all(*, run_dir=None, resume_from='dedupe', tracks_config=None,
                negative_supply_run_tag=None):
    from core.common import F, RESULTS, TRAIN_ROOT, TRAINING_CONFIG_PATH, resolve_model, training_cfg
    from model_tracks.config import load_config as load_suite
    root = Path(TRAIN_ROOT)
    config_path = Path(tracks_config or root / 'config/model_tracks.yaml').resolve()
    suite = load_suite(config_path)
    training = training_cfg()
    if suite.text_model != training.training.base_model:
        raise ValueError('suite text_model and training.base_model must agree before preparation')
    if suite.epochs > training.training.epochs:
        raise ValueError('suite epochs exceed locally prepared training epochs; update training.yaml first')
    lane = training.negative_supply
    if lane.mode == 'lane':
        if negative_supply_run_tag and negative_supply_run_tag != lane.pairs_run_tag:
            raise ValueError('negative-supply run tag differs from the configured training lane')
        negative_supply_run_tag = lane.pairs_run_tag
    if negative_supply_run_tag and not re.fullmatch(r'[A-Za-z0-9_-]+', negative_supply_run_tag):
        raise ValueError('negative-supply run tag must contain only letters, digits, underscores, hyphens')
    run_dir = Path(run_dir or RESULTS / 'training_prep' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')).resolve()
    negative_supply_run_tag = negative_supply_run_tag or ('prep_' + run_dir.name)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', negative_supply_run_tag):
        raise ValueError('preparation directory name requires an explicit valid --negative-supply-run-tag')
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
        checkpoint = resolve_model(suite.text_model)
        setup = (root / suite.setup_dir).resolve()
        text_bundle = (root / suite.text_bundle).resolve()
        bundle = root / 'data/prepared/full/worker_1_baseline.pkl.gz'
        suite_archive = run_dir / 'all_tracks_inputs.zip'
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
                    'shared_base_payload': env['EUROMONITOR_SHARED_BASE_DATA'],
                    'tracks_config': str(config_path),
                    'negative_supply_mode': lane.mode,
                    'negative_supply_run_tag': negative_supply_run_tag,
                    'hybrid_embeddings': 'GPU pending: frozen baseline forward before hybrid training'}
        manifest_path = run_dir / 'manifest.json'
        def publish():
            temporary = manifest_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(manifest, indent=2) + '\n')
            temporary.replace(manifest_path)
        def run(name, arguments, *, check=True):
            print(f'[prepare] {name} -> {run_dir / (name + ".log")}', flush=True)
            with (run_dir / (name + '.log')).open('w') as log:
                return subprocess.run([sys.executable, *arguments], cwd=root, env=env,
                                      stdout=log, stderr=subprocess.STDOUT, check=check)
        def archive(path, name):
            if path.exists():
                destination = run_dir / 'before' / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    raise FileExistsError(destination)
                shutil.move(str(path), str(destination))
        publish()
        try:
            stages = list(STAGES)
            if negative_supply_run_tag:
                stages[stages.index('validation'):stages.index('validation')] = ['negative_supply', 'discriminator']
            first = 'negative_supply' if resume_from == 'validation' else resume_from
            for name in stages[stages.index(first):]:
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
                elif name == 'negative_supply':
                    archive(RESULTS / 'negative_supply' / negative_supply_run_tag, 'negative_supply')
                    run(name, ['-m', 'training.negative_supply', '--run-tag', negative_supply_run_tag])
                elif name == 'discriminator':
                    arguments = [str(root / 'scripts/negative_supply_discriminator.py'),
                                 str(RESULTS / 'negative_supply' / negative_supply_run_tag / 'pairs.csv'),
                                 '--out', str(run_dir / 'discriminator.json')]
                    result = run(name, arguments, check=False)
                    verdict = json.loads((run_dir / 'discriminator.json').read_text())
                    manifest['discriminator'] = verdict
                    if result.returncode and (lane.mode == 'lane' or verdict.get('verdict') != 'SEPARABLE'):
                        raise subprocess.CalledProcessError(result.returncode, arguments)
                    if result.returncode:
                        print('[prepare] diagnostic lane is SEPARABLE; active gate mode is unchanged', flush=True)
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
                               '--model', suite.text_model, '--no-mask-effect', '--no-plot'])
                    if text_bundle != bundle:
                        archive(text_bundle, 'text_bundle.pkl.gz')
                        archive(text_bundle.with_suffix(text_bundle.suffix + '.json'), 'text_bundle.pkl.gz.json')
                        text_bundle.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(bundle, text_bundle)
                        shutil.copy2(bundle.with_suffix(bundle.suffix + '.json'),
                                     text_bundle.with_suffix(text_bundle.suffix + '.json'))
                elif name == 'suite_inputs':
                    run(name, ['-m', 'model_tracks.package', '--config', str(config_path),
                               '--output', str(suite_archive)])
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
                    from model_tracks.package import verify
                    package = verify(suite_archive)
                    manifest['suite_package'] = {'path': str(suite_archive),
                                                'sha256': sha256(suite_archive),
                                                'preflight': package['preflight']}
                    if {str(path): sha256(path) for path in smoke.rglob('*') if path.is_file()} != smoke_before:
                        raise ValueError('Smoke files changed during full preparation')
                    keys = ('dataset_deduped', 'sku_to_rep', 'dedupe_summary', 'ambiguous_offer_groups',
                            'removals', 'dedupe_conflicts', 'number_reference',
                            'second04_pairs_positive', 'canonical_records', 'gate_results',
                            'labeled_pairs', 'final_validation', 'validation_fold_map')
                    paths = [Path(F[key]) for key in keys] + [p for p in setup.rglob('*') if p.is_file()]
                    paths += [bundle, bundle.with_suffix(bundle.suffix + '.json'), text_bundle,
                              text_bundle.with_suffix(text_bundle.suffix + '.json'), suite_archive]
                    if negative_supply_run_tag:
                        paths += [RESULTS / 'negative_supply' / negative_supply_run_tag / filename
                                  for filename in ('pairs.csv', 'manifest.json')]
                        paths.append(run_dir / 'discriminator.json')
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
    parser.add_argument('--tracks-config', type=Path,
                        help='three-track configuration (default: config/model_tracks.yaml)')
    parser.add_argument('--negative-supply-run-tag',
                        help='generate and audit the real-first lane; gate mode keeps this diagnostic-only')
    parser.add_argument('--resume-from', choices=['dedupe', 'validation'], default='dedupe',
                        help='validation requires verified current CSV stage manifests')
    args = parser.parse_args()
    prepare_all(run_dir=args.run_dir, resume_from=args.resume_from,
                tracks_config=args.tracks_config, negative_supply_run_tag=args.negative_supply_run_tag)


if __name__ == '__main__':
    main()
