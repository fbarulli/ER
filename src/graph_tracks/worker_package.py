"""Build a portable, preflighted graph worker input ZIP; never provision or train."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from core.bundle import Bundle, BundleRole
from core.portable_archive import write_archive

import yaml

from graph_tracks.data import file_hash
from graph_tracks.preflight import preflight


def _bundle_inventory(files: dict[str, Path], inline: dict[str, str]) -> dict[str, str]:
    """The Bundle-facing inventory (config ``files_key``).

    ``Bundle.load`` verifies against ``training_cfg().bundle.files_key``. The
    portable package keeps its historical ``files_sha256`` key (written by
    ``write_archive``) for existing consumers, so the manifest carries both.
    """
    from core.portable_archive import cached_file_digest
    inventory = {target: cached_file_digest(source) for target, source in files.items()}
    inventory.update({target: hashlib.sha256(value.encode()).hexdigest()
                      for target, value in inline.items()})
    return inventory


def package(config: Path, output: Path, *, device: str = 'cuda',
            run_tag: str | None = None) -> Path:
    from core.common import TRAIN_ROOT, training_cfg
    from graph_tracks.config import load_config
    cfg = load_config(config)
    checks = preflight(config, check_device=False)
    if output.exists():
        raise FileExistsError(output)
    base = Path('data/graph_worker') / cfg.track
    settings = cfg.model_dump()
    files = {}
    for key in ('listings', 'pairs', 'input_manifest', 'text_cache'):
        if not settings.get(key):
            continue
        source = (TRAIN_ROOT / settings[key]).resolve()
        destination = base / f'{key}{source.suffix}'
        if key == 'text_cache':
            destination = base / 'text_provenance' / source.name
            for relative in ('embedding_inputs.json', 'eligible_catalog.csv',
                             'prepared/input_manifest.json', 'prepared/listings.json'):
                files[(destination.parent / relative).as_posix()] = source.parent / relative
        files[destination.as_posix()] = source
        settings[key] = destination.as_posix()
    settings.update(device=device, output_dir='results/graph_tracks')
    from graph_tracks.prepared_inputs import PLAN, ARRAYS
    for filename in (PLAN, ARRAYS, 'pair_lineage.json'):
        source = (TRAIN_ROOT / cfg.listings).resolve().parent / filename
        if source.is_file():
            files[(base / filename).as_posix()] = source
    from graph_tracks.report_attributes import FILENAME
    report_attributes = (TRAIN_ROOT / cfg.listings).resolve().parent / FILENAME
    if report_attributes.is_file():
        files[(base / FILENAME).as_posix()] = report_attributes
    from model_tracks.package import runtime_snapshot_files
    files.update(runtime_snapshot_files())
    dependency = TRAIN_ROOT / 'requirements/graph_tracks.txt'
    files[dependency.relative_to(TRAIN_ROOT).as_posix()] = dependency
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=TRAIN_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    from graph_tracks.config import GraphConfig
    configuration = yaml.safe_dump(GraphConfig.model_validate(settings).model_dump(), sort_keys=False)
    config_target = (base / 'worker.yaml').as_posix()
    manifest = {'schema': 'er-graph-worker-package-v1', 'base_git_revision': revision,
                'track': cfg.track, 'local_preflight': checks,
                'target_device': device, 'target_runtime_verified': False,
                # The package is run-agnostic; its identity is the pinned
                # revision until a worker supplies the real run tag. Bundle's
                # inputs role requires a run_tag key, so bind it to the revision.
                'run_tag': run_tag or revision}
    readme = (
        f'Check out ER revision {revision}, then extract this ZIP into that checkout.\n'
        'The ZIP includes the shared runtime source/config overlay, hashed in package_manifest.json.\n'
        'Install PyTorch for the target runtime and requirements/graph_tracks.txt.\n'
        'Verify files before use:\n'
        f'PYTHONPATH=src python -m graph_tracks.worker_package --verify {base}/package_manifest.json\n'
        'Validate the target runtime before starting a worker:\n'
        f'PYTHONPATH=src python -m graph_tracks.preflight --config {base}/worker.yaml\n'
        'When training is authorized:\n'
        f'PYTHONPATH=src python -m graph_tracks.train --config {base}/worker.yaml --run-tag YOUR_RUN_TAG\n'
        'Collect the complete result ZIP before VM teardown using graph_tracks.bundle.\n'
        'This package does not provision a VM or start training. No credentials are included.\n')
    inline = {config_target: configuration, (base / 'README.txt').as_posix(): readme}
    bundle_spec = training_cfg().bundle
    manifest[bundle_spec.files_key] = _bundle_inventory(files, inline)
    manifest_name = (base / 'package_manifest.json').as_posix()
    write_archive(output, files, manifest_name=manifest_name,
                  metadata=manifest, inventory_key='files_sha256', inline=inline)
    return Bundle.load(output, BundleRole.inputs, manifest_name=manifest_name).path



def verify(manifest_path: Path) -> None:
    from core.common import TRAIN_ROOT
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 'er-graph-worker-package-v1':
        raise ValueError('unsupported package schema')
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=TRAIN_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    if revision != manifest['base_git_revision']:
        raise ValueError('worker checkout revision mismatch')
    for target, expected in manifest['files_sha256'].items():
        path = (TRAIN_ROOT / target).resolve()
        if not path.is_relative_to(TRAIN_ROOT.resolve()) or file_hash(path) != expected:
            raise ValueError(f'worker package file mismatch: {target}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify:
        verify(args.verify)
        print('worker package verified')
    elif args.config and args.output:
        print(package(args.config, args.output, device=args.device))
    else:
        parser.error('provide --verify or both --config and --output')


if __name__ == '__main__':
    main()
