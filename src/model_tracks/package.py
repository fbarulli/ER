"""One deduplicated prepared-input package for one all-track Colab run.

Every function is a sequence of logged, timed, progress-tracked task blocks:
task boundaries emit `[timing] ... rss_mb=...` into the run's existing time
log, loops carry tqdm bars (or shim lines in non-tty logs), and each block
releases its intermediates as early as possible — a suite_inputs OOM should
be attributable to the task that grew.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Iterator
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import yaml

from core.archive_reader import open_archive
from core.portable_archive import (
    INVENTORY, RuntimeSnapshot, verify_archive, verify_open_archive, write_archive,
)
from core.progress import tracked
from core.step_trace import rss_mb, send, timed, trace_step
from model_tracks.config import load_config
from model_tracks.preflight import preflight

PACKAGE_MANIFEST = 'model_tracks_package.json'
RECOVERY_MANIFEST = 'suite_recovery_manifest.json'
GRAPH_TRACKS = ('gnn_only', 'hybrid')
TRACK_CONFIGS = frozenset({'gnn_only.yaml', 'hybrid.yaml', 'text.yaml'})
INPUT_KEYS = ('dataset_deduped', 'labeled_pairs', 'canonical_records', 'gate_results')
RECOVERY_EXCLUDED = frozenset({'wandb', 'mlruns', 'mps_pipe', 'mps_log', '.git', '.dvc'})


def package_member(key: str) -> str:
    """Resolve the package's portable member layout through paths.yaml."""
    from core.common import LAYOUTS
    layout = LAYOUTS[key]
    if layout.root != 'repo' or layout.fields:
        raise ValueError('package member requires a repository-relative static layout: ' + key)
    return layout.template


def _target() -> Path:
    return Path(package_member('suite_package_shared'))


def _setup_layout():
    from core.common import training_cfg
    return training_cfg().preparation.graph_setup


def _sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + '.json')


def _walk_files(root: Path, *, excluded_dirs=frozenset(), excluded_suffixes=()) -> Iterator[Path]:
    """Prune excluded trees before descent; never follow directory symlinks."""
    def fail(error):
        raise error

    for directory, directories, names in os.walk(root, onerror=fail, followlinks=False):
        parent = Path(directory)
        directories[:] = [name for name in directories
                          if name not in excluded_dirs
                          and not (excluded_suffixes and name.endswith(excluded_suffixes))
                          and not (parent / name).is_symlink()]
        for name in names:
            path = parent / name
            if not path.is_symlink() and path.is_file():
                yield path


def _read_json(path: Path) -> Any:
    with path.open(encoding='utf-8') as handle:
        return json.load(handle)


def _archive_json(archive, name: str) -> Any:
    with archive.open(name) as handle:
        return json.load(handle)


def _release_bundle(path: Path) -> None:
    """Drop the supervisor's cached reference as well as caller-owned objects."""
    from training.preparation_run import active_preparation
    run = active_preparation()
    if run is not None:
        run.release_bundle(path)
    gc.collect()
    send(f'[package] released_bundle={path.name} peak_rss_mb={rss_mb()}')


@timed
def runtime_snapshot_files(*, ablation_config: Path | None = None) -> dict[str, Path]:
    """Shared local source/config overlay for prepared Colab jobs."""
    from core.common import TRAIN_ROOT
    from core.progress import tracked
    files = {}
    with trace_step('snapshot.collect_source_files'):
        for path in tracked((p for p in _walk_files(TRAIN_ROOT / 'src',
                            excluded_dirs=frozenset({'__pycache__'})) if p.suffix == '.py'), total=None,
                            desc='snapshot.source_files'):
            files[path.relative_to(TRAIN_ROOT).as_posix()] = path
    with trace_step('snapshot.pin_scripts_and_configs'):
        for name in ('src/pipeline.py','scripts/diet_manifest.py',
                     'scripts/run_colab_ablation.py','scripts/run_colab_embeddings.py',
                     'src/cli/colab.py','src/cli/__init__.py'):
            files[name] = TRAIN_ROOT/name
        for name in ('paths.yaml','training.yaml','identity_dimensions.yaml','identity_reviews.json','vocabulary.json','text_track.yaml','attribute_ablation.yaml'):
            files['config/'+name] = TRAIN_ROOT/'config'/name
        # The semantic family registry is a REQUIRED frozen input: calibration
        # refuses to start without it (core.attribute_decision raises rather than
        # run on an open vocabulary). It lives under results/, which is gitignored
        # AND excluded from the Colab sparse checkout, so unless it ships in the
        # package the GPU calibration lane can never find it. Fail here, at package
        # time, instead of on the remote after provisioning an accelerator.
        registry = TRAIN_ROOT / package_member('semantic_family_registry')
        if not registry.is_file():
            raise FileNotFoundError(
                'semantic family registry missing: run scripts/build_attribute_semantics.py '
                'before packaging (no silent open-vocabulary gap)')
        files[registry.relative_to(TRAIN_ROOT).as_posix()] = registry
    with trace_step('snapshot.pin_ablation_config'):
        if ablation_config is not None:
            source = Path(ablation_config).resolve()
            files[source.relative_to(TRAIN_ROOT).as_posix()] = source
    return RuntimeSnapshot(files=files).files


def _dump_model_json(model, path: Path) -> None:
    """Stream JSON to a sibling file, replacing the artifact only on success."""
    from model_tracks.training_data import TrainingJSONEncoder
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix=path.name + '.', suffix='.partial',
                                     delete=False) as handle:
        candidate = Path(handle.name)
        try:
            json.dump(model, handle, cls=TrainingJSONEncoder, ensure_ascii=False)
            handle.write('\n')
            handle.close()
            candidate.replace(path)
        finally:
            candidate.unlink(missing_ok=True)


@timed
def _prepare_shared_population(setup, bundle):
    """Project the bundle, then release the shared population before tokenization."""
    from model_tracks.training_data import from_bundle, TrackTrainingBinding
    from model_tracks.shared_graph_data import prepare_shared_graph
    with trace_step('package.build_shared_population'):
        shared = from_bundle(bundle)
    with trace_step('package.write_shared_files'):
        _dump_model_json(shared, setup / _setup_layout().shared_training_data)
        text_binding = TrackTrainingBinding(track='text', shared_data_sha256=shared.fingerprint,
            example_ids=[row.example_id for row in shared.examples],
            endpoint_indices=[row.payload_index for row in shared.endpoints])
        _dump_model_json(text_binding, setup / _setup_layout().text_training_binding)
        del text_binding
        prepare_shared_graph(setup, bundle, shared)


@timed
def _prepare_graph_inputs(setup):
    """Materialize CPU graph inputs without retaining preparation intermediates."""
    with trace_step('package.graph_prepare'):
        # All tensors and native tokenizer features are fixed on local CPU before
        # provisioning; selected weights are bound only by the GPU exporter.
        from graph_tracks.prepared_inputs import prepare_training
        from graph_tracks.config import load_config as load_graph_config
        track_settings = [load_graph_config(setup/(track+'.yaml'), expected_track=track).model_dump() for track in GRAPH_TRACKS]
        sizes = {settings['inference_batch_size'] for settings in track_settings}
        if len(sizes) != 1:
            raise ValueError('shared prepared graph inference batch sizes must agree')
        prepare_training(setup/'prepared/listings.json',setup/'prepared/pairs.csv',batch_size=sizes.pop())


def _make_composer():
    """Build deterministic endpoint text with a bounded LRU cache."""
    from model_tracks.training_data import frozen_endpoint_text
    from core.model_input import model_input_composition,build_sku_text,model_input_info
    from core.sku_identity import row_identity
    from graph_tracks.text_cache import composition_fingerprint
    from model_tracks.ablation import digest
    import pandas as pd
    from core.common import training_cfg
    limits = training_cfg().packaging
    with trace_step('package.build_composition_contract'):
        composition_contract = {'spec':model_input_composition().model_dump(mode='json'),
                                'implementation':composition_fingerprint()}
        # This cache lasts for one verified local preparation only. Exact raw
        # rows and the frozen composition contract key both baseline and
        # interventions. Bound both row count and retained string bytes.
        composed = OrderedDict()
        cache_bytes = 0
        # Frozen endpoint text is a CONTRACT, not a hint: a virtual endpoint whose
        # bundle text is empty is still authoritative, so virtualness decides and an
        # empty cell is a value rather than a reason to recompose. The digest itself
        # is verified once per endpoint by the shared graph projection.
        def compose(row):
            nonlocal cache_bytes
            frozen = frozen_endpoint_text(row.get('sku_id'), row.get('frozen_payload'),
                                          column_present='frozen_payload' in row)
            if frozen is not None:
                return frozen
            key = digest({'row':row,'composition':composition_contract})
            if key in composed:
                composed.move_to_end(key)
                return composed[key]
            series = pd.Series(row)
            text = build_sku_text(series, model_input_info(row_identity(series).as_mapping()))
            cost = sys.getsizeof(key) + sys.getsizeof(text)
            if limits.composition_cache_entries and cost <= limits.composition_cache_bytes:
                while composed and (len(composed) >= limits.composition_cache_entries or cache_bytes + cost > limits.composition_cache_bytes):
                    old_key, old_text = composed.popitem(last=False)
                    cache_bytes -= sys.getsizeof(old_key) + sys.getsizeof(old_text)
                composed[key] = text
                cache_bytes += cost
            return text
    return compose


@timed
def _prepare_exports(cfg, setup, bundle):
    """Prepare exports with bounded reuse; return only the model for validation."""
    from core.common import TRAIN_ROOT
    from model_tracks.text_export import prepare as prepare_text_export
    from core.common import resolve_model, runtime
    compose = _make_composer()
    token_cache = {}
    with trace_step('package.text_export'):
        prepare_text_export(setup,Path(resolve_model(cfg.text_model)),batch_size=runtime('batch_size_embed'),composer=compose,token_cache=token_cache)
    with trace_step('package.baseline_export'):
        from model_tracks.baseline_export import prepare as prepare_baseline
        prepare_baseline(setup,Path(resolve_model(cfg.text_model)),composer=compose)
    # The bundle was last needed by the exhaustive cohort; composed text and
    # token caches expire with the preparation once the native model is out.
    native_model = next(value for key,value in token_cache.items() if key[0] == 'model')
    token_cache.clear()
    return native_model


def _portable_graph_config(cfg, setup: Path, track: str) -> dict:
    from core.common import TRAIN_ROOT
    from graph_tracks.config import load_config as load_graph_config
    settings = load_graph_config(setup / f'{track}.yaml', expected_track=track).model_dump()
    for key in ('listings', 'pairs', 'input_manifest', 'text_cache'):
        if settings.get(key):
            source = (TRAIN_ROOT / settings[key]).resolve()
            settings[key] = str(_target() / source.relative_to(setup))
    settings.update(device=cfg.device, report_test=cfg.report_test)
    settings.update(cfg.graph_execution_overrides())
    return settings


def _portable_config_models(cfg, setup: Path) -> dict[str, dict]:
    """One rendering contract shared by package creation and reuse validation."""
    from graph_tracks.config import load_text_config
    text = load_text_config(setup / _setup_layout().text_config).model_dump()
    text.update(report_test=cfg.report_test)
    suite = cfg.model_dump()
    suite.update(setup_dir=str(_target()), text_bundle=str(_target() / 'text_prepared.pkl.gz'))
    return {
        package_member('suite_package_config'): suite,
        str(_target() / 'text.yaml'): text,
        **{str(_target() / f'{track}.yaml'): _portable_graph_config(cfg, setup, track)
           for track in GRAPH_TRACKS},
    }


def _runtime_sources(cfg) -> dict[str, Path]:
    from core.common import F, TRAIN_ROOT
    files = runtime_snapshot_files(ablation_config=TRAIN_ROOT / cfg.ablation_config)
    for key in INPUT_KEYS:
        source = Path(F[key]).resolve()
        files[source.relative_to(TRAIN_ROOT).as_posix()] = source
    return files


@timed
def _collect_package_sources(cfg, setup: Path, bundle_path: Path) -> dict[str, Path]:
    """Inventory each prepared artifact once; omit prior archives and sidecars."""
    bundle_sources = {bundle_path, _sidecar(bundle_path)}
    omitted_names = TRACK_CONFIGS | {'text_prepared.pkl.gz', 'text_prepared.pkl.gz.json'}
    files = {}
    for path in tracked(_walk_files(setup), desc='package.collect_files'):
        if (path.name in omitted_names or path.name.endswith(('.zip', '.tar.zst', '.profile.json'))
                or '.partial-' in path.name or path.name.endswith('.partial')):
            continue
        if path.resolve() not in bundle_sources:
            files[str(_target() / path.relative_to(setup))] = path
    files.update(_runtime_sources(cfg))
    files[str(_target() / 'text_prepared.pkl.gz')] = bundle_path
    files[str(_target() / 'text_prepared.pkl.gz.json')] = _sidecar(bundle_path)
    # Ablation cohort staging (GPU suite) rebuilds from the UNPROJECTED clean
    # gates; the projected setup overrides them, so the immutable clean backup
    # ships whenever it exists. The PortableLayout class owns both the local
    # discovery name and the portable member key — ship/consume can't desync.
    from model_tracks.portable_layout import PortableLayout
    clean_backup = PortableLayout.local_clean_backup(setup)
    if clean_backup.is_dir():
        for path in sorted(clean_backup.rglob('*')):
            if path.is_file():
                files[PortableLayout.ship_key(path, from_local=clean_backup)] = path
    return files


@timed
def package(config: Path, output: Path) -> Path:
    from core.common import TRAIN_ROOT, git_revision

    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)

    with trace_step('package.resolve_paths'):
        cfg = load_config(config)
        setup = (TRAIN_ROOT / cfg.setup_dir).resolve()
        bundle_path = (TRAIN_ROOT / cfg.text_bundle).resolve()
    from training.prepared_bundle import load_prepared_bundle
    with trace_step('package.load_bundle', bundle=bundle_path.name):
        _, bundle = load_prepared_bundle(bundle_path)
    try:
        _prepare_shared_population(setup, bundle)
        gc.collect()
        _prepare_graph_inputs(setup)
        native_model = _prepare_exports(cfg, setup, bundle)
    finally:
        del bundle
        _release_bundle(bundle_path)
    try:
        with trace_step('package.preflight'):
            checks = preflight(config, allow_gpu_pending=True, native_token_model=native_model)
    finally:
        del native_model
        _release_bundle(bundle_path)
    files = _collect_package_sources(cfg, setup, bundle_path)
    with trace_step('package.inline_configs'):
        inline = {name: yaml.safe_dump(value, sort_keys=False)
                  for name, value in _portable_config_models(cfg, setup).items()}
    with trace_step('package.write_archive'):
        revision = git_revision()
        archive = write_archive(output,files,inline=inline,manifest_name=PACKAGE_MANIFEST,
                                metadata={'schema':'er-model-tracks-package-v1','revision':revision,'preflight':checks},
                                profile=True)
    send(f'[package] archive={archive} rss_mb={rss_mb()}')
    return archive


@timed
def verify(path: Path) -> dict:
    return verify_archive(path, PACKAGE_MANIFEST)


def _verify_configs(archive, cfg, setup: Path) -> None:
    for name, expected in _portable_config_models(cfg, setup).items():
        with archive.open(name) as handle:
            if yaml.safe_load(handle) != expected:
                raise ValueError('prepared package configuration changed; regenerate locally: ' + name)


def _expected_sources(cfg, setup: Path, inventory: dict) -> Iterator[tuple[str, Path]]:
    from core.common import TRAIN_ROOT
    yield from _runtime_sources(cfg).items()
    bundle = (TRAIN_ROOT / cfg.text_bundle).resolve()
    for name in inventory:
        member = Path(name)
        if member.is_relative_to(_target()) and member.name not in TRACK_CONFIGS:
            if member == _target() / 'text_prepared.pkl.gz':
                source = bundle
            elif member == _target() / 'text_prepared.pkl.gz.json':
                source = _sidecar(bundle)
            else:
                source = setup / member.relative_to(_target())
            yield name, source


def _verify_sources(cfg, setup: Path, inventory: dict) -> None:
    from graph_tracks.data import file_hash
    for name, source in tracked(_expected_sources(cfg, setup, inventory), desc='verify_current.hashes'):
        if not source.is_file() or inventory.get(name) != file_hash(source):
            raise ValueError('prepared package source changed: ' + name)


@timed
def verify_current(path: Path, config: Path) -> dict:
    """Fail on stale local inputs before inflating every large archive member."""
    from core.common import TRAIN_ROOT, resolve_model
    from graph_tracks.text_cache import checkpoint_hash
    cfg = load_config(config)
    setup = (TRAIN_ROOT / cfg.setup_dir).resolve()
    with open_archive(path) as archive:
        _validate_archive_paths(archive)
        metadata = _archive_json(archive, PACKAGE_MANIFEST)
        inventory = INVENTORY.validate_python(metadata['files'])
        with trace_step('verify_current.inline_configs'):
            _verify_configs(archive, cfg, setup)
        with trace_step('verify_current.hash_sources'):
            _verify_sources(cfg, setup, inventory)
        with trace_step('verify_current.checkpoint'):
            request = _archive_json(archive, str(_target() / _setup_layout().embedding_request))
            expected = request['metadata']['checkpoint_sha256']
            del request
            if expected != checkpoint_hash(Path(resolve_model(cfg.text_model))):
                raise ValueError('prepared package baseline checkpoint changed')
        with trace_step('verify_current.archive_integrity'):
            return verify_open_archive(archive, PACKAGE_MANIFEST)


def _recovery_sources(output: Path, destination: Path) -> dict[str, Path]:
    files = {}
    for path in tracked(_walk_files(output, excluded_dirs=RECOVERY_EXCLUDED,
                                   excluded_suffixes=('.publication', '__payload')),
                        desc='recovery.scan_files'):
        if path.name in {'.env', 'config.local'} or path.resolve() == destination:
            continue
        files[path.relative_to(output).as_posix()] = path
    return files


@timed
def recovery_package(output: Path, destination: Path, run_tag: str, *, input_package: dict | None = None) -> Path:
    """Capture stopped workers' portable state, pruning caches before traversal."""
    output, destination = Path(output).resolve(), Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    with trace_step('recovery_package.check_manifest'):
        if _read_json(output / 'suite_manifest.json').get('run_tag') != run_tag:
            raise ValueError('recovery suite run mismatch')
    files = _recovery_sources(output, destination.resolve())
    return write_archive(destination, files, manifest_name=RECOVERY_MANIFEST,
                         metadata={'schema': 'er-suite-recovery-v1', 'run_tag': run_tag,
                                   'input_package': input_package})


def _validate_archive_paths(archive) -> None:
    """Validate before opening any member, even when its manifest is missing."""
    seen = set()
    for info in archive.infolist():
        member = Path(info.filename)
        if not info.filename or member.is_absolute() or '..' in member.parts:
            raise ValueError('unsafe archive path')
        if info.is_dir() or (info.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError('archive member must be a regular file (no symbolic links)')
        if info.filename in seen:
            raise ValueError('duplicate archive members')
        seen.add(info.filename)


def _recovery_inventory(source, run_tag: str) -> dict:
    _validate_archive_paths(source)
    metadata = _archive_json(source, RECOVERY_MANIFEST)
    if metadata.get('schema') != 'er-suite-recovery-v1' or metadata.get('run_tag') != run_tag:
        raise ValueError('recovery suite run mismatch')
    inventory = INVENTORY.validate_python(metadata['files'])
    if set(source.namelist()) != set(inventory) | {RECOVERY_MANIFEST}:
        raise ValueError('archive has undeclared or missing members')
    if _archive_json(source, 'suite_manifest.json').get('run_tag') != run_tag:
        raise ValueError('recovery suite manifest mismatch')
    return inventory


def _extract_verified_members(source, staging: Path, inventory: dict) -> None:
    """Copy and SHA256-check in one pass, keeping only a 1 MiB buffer in RAM."""
    from core.common import training_cfg
    buffer_bytes = training_cfg().archives.copy_buffer_bytes
    for name, expected in tracked(inventory.items(), desc='restore.members'):
        target = staging / name
        if not target.resolve().is_relative_to(staging):
            raise ValueError('unsafe recovery archive member')
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with source.open(name) as member, target.open('xb') as handle:
            while chunk := member.read(buffer_bytes):
                digest.update(chunk)
                handle.write(chunk)
        if digest.hexdigest() != expected:
            raise ValueError('archive integrity mismatch: ' + name)


def _publish_recovery(staging: Path, output: Path) -> None:
    # Reserve the name exclusively, then replace our own empty directory. A
    # concurrent restore cannot overwrite a previously published recovery.
    output.mkdir()
    try:
        staging.replace(output)
    except BaseException:
        output.rmdir()
        raise


@timed
def restore_recovery(archive: Path, output: Path, run_tag: str) -> Path:
    """Restore ZIP or tar.zst once, publishing only after every digest passes."""
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    with open_archive(archive) as source:
        inventory = _recovery_inventory(source, run_tag)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.' + output.name + '-', dir=output.parent) as temp:
            staging = Path(temp) / 'payload'
            staging.mkdir()
            with trace_step('restore_recovery.unpack', members=len(inventory)):
                _extract_verified_members(source, staging.resolve(), inventory)
            _publish_recovery(staging, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(package(args.config, args.output))


if __name__ == '__main__':
    main()
