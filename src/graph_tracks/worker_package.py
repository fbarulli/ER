"""Build a portable, preflighted graph worker input ZIP; never provision or train.

Sealing goes through the shared writer (``Bundle.seal_archive``), so the ZIP is
written and verified exactly once and the returned archive IS a loadable
``inputs`` Bundle. The package manifest keeps the historical ``files_sha256``
inventory (already-published packages and the printed ``--verify`` instructions
keep working) alongside the Bundle-facing ``files`` inventory the shared writer
records; both are the same name -> sha256 map.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from core.bundle import Bundle, BundleRole

import yaml

from graph_tracks.data import file_hash
from graph_tracks.preflight import preflight


def _legacy_inventory_key() -> str:
    """The historical member-inventory manifest key (``files_sha256``).

    Derived from the Bundle spec (``bundle.files_key`` + ``_sha256``), never
    re-spelled: the authoritative inventory key is the config one.
    """
    from core.common import training_cfg
    return f"{training_cfg().bundle.files_key}_sha256"


def _setup_layout():
    """The declared prepared-setup layout (training.preparation.graph_setup)."""
    from core.common import prepared_setup_layout
    return prepared_setup_layout()


def _package_base(track: str) -> Path:
    """The packaged worker's portable member root (``paths.yaml`` layout).

    Read as the layout TEMPLATE with only its declared ``track`` field, exactly
    like ``model_tracks.package.package_member``: every member key in this ZIP is
    repository-relative (the ZIP is extracted into a checkout), so the resolved
    absolute address is the wrong shape here. A non-repo root, an undeclared
    placeholder set or an unresolvable template fails loud instead of shipping a
    silently different tree.
    """
    from core.common import LAYOUTS
    layout = LAYOUTS['graph_worker_package']
    if layout.root != 'repo' or set(layout.fields) != {'track'}:
        raise ValueError('graph_worker_package must be a repo-relative {track} layout')
    return Path(layout.template.format(track=str(track)))


def _bundle_inventory(files: dict[str, Path], inline: dict[str, str]) -> dict[str, str]:
    """The historical ``files_sha256`` inventory (same shape as ``files``).

    ``Bundle.seal_archive`` recomputes the authoritative ``bundle.files_key``
    inventory through the ONE builder (``core.portable_archive
    .source_inventory``) while it writes and verifies it against the written
    bytes, so the boundary check owns integrity; this mirror exists only for
    the legacy key. Forwarding to that same builder means both keys carry the
    identical map and the sources are hashed once per process (the digest
    cache), never by a second loop.
    """
    from core.portable_archive import source_inventory
    return source_inventory(files, inline)


def package(config: Path, output: Path, *, device: str = 'cuda',
            run_tag: str | None = None) -> Path:
    from core.common import TRAIN_ROOT, training_cfg
    from graph_tracks.config import load_config
    cfg = load_config(config)
    checks = preflight(config, check_device=False)
    if output.exists():
        raise FileExistsError(output)
    base = _package_base(cfg.track)
    layout = _setup_layout()
    settings = cfg.model_dump()
    files = {}
    for key in ('listings', 'pairs', 'input_manifest', 'text_cache'):
        if not settings.get(key):
            continue
        source = (TRAIN_ROOT / settings[key]).resolve()
        destination = base / f'{key}{source.suffix}'
        if key == 'text_cache':
            destination = base / 'text_provenance' / source.name
            for relative in (layout.embedding_request, layout.catalog,
                             f'{layout.prepared_dir}/{layout.input_manifest}',
                             f'{layout.prepared_dir}/{layout.listings}'):
                files[(destination.parent / relative).as_posix()] = source.parent / relative
        files[destination.as_posix()] = source
        settings[key] = destination.as_posix()
    # The packaged worker's output tree is the lane config's own ``output_dir``
    # (GraphConfig, declared per lane in config/graph_tracks_*.yaml and rewritten
    # from those templates by graph_tracks.setup). Spelling it here would be a
    # second declaration that silently overrides a retuned lane.
    settings.update(device=device)
    from graph_tracks.prepared_inputs import PLAN, ARRAYS
    for filename in (PLAN, ARRAYS, layout.pair_lineage):
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
    worker_config_file = training_cfg().bundle.worker_config_file
    config_target = (base / worker_config_file).as_posix()
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
        f'PYTHONPATH=src python -m graph_tracks.preflight --config {base}/{worker_config_file}\n'
        'When training is authorized:\n'
        f'PYTHONPATH=src python -m graph_tracks.train --config {base}/{worker_config_file} --run-tag YOUR_RUN_TAG\n'
        'Collect the complete result ZIP before VM teardown using graph_tracks.bundle.\n'
        'This package does not provision a VM or start training. No credentials are included.\n')
    inline = {config_target: configuration, (base / 'README.txt').as_posix(): readme}
    manifest[_legacy_inventory_key()] = _bundle_inventory(files, inline)
    manifest_name = (base / 'package_manifest.json').as_posix()
    sealed = Bundle.seal_archive(output, files, role=BundleRole.inputs,
                                 manifest_name=manifest_name, metadata=manifest,
                                 inline=inline)
    return sealed.path


def verify(manifest_path: Path) -> None:
    """Re-verify an extracted worker package against its own manifest.

    Prefers the Bundle-facing inventory (``training_cfg().bundle.files_key``,
    the key ``Bundle.load`` checks) and falls back to the historical
    ``files_sha256`` mirror so already-written packages still verify.
    """
    from core.common import TRAIN_ROOT, training_cfg
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 'er-graph-worker-package-v1':
        raise ValueError('unsupported package schema')
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=TRAIN_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    if revision != manifest['base_git_revision']:
        raise ValueError('worker checkout revision mismatch')
    inventory = (manifest.get(training_cfg().bundle.files_key)
                 or manifest.get(_legacy_inventory_key()))
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError('worker package manifest carries no member inventory')
    # The tree is already unpacked, so member names resolve against the
    # checkout; the ONE inventory comparison owns the digest check.
    from core.portable_archive import compare_inventory
    actual = {}
    for target in inventory:
        path = (TRAIN_ROOT / target).resolve()
        if not path.is_relative_to(TRAIN_ROOT.resolve()):
            raise ValueError(f'worker package file mismatch: {target}')
        actual[target] = file_hash(path)
    compare_inventory(inventory, actual, mismatch='worker package file mismatch')


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
