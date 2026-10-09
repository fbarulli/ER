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
from core.portable_archive import ByteCount
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import yaml

from core.archive_reader import archive_sidecar
from core.bundle import bundle_spec
from core.portable_archive import RuntimeSnapshot
from core.progress import tracked
from core.run_log import RunLogger
from core.step_trace import rss_mb, send, timed, trace_step
from core.tracing import (
    ENTITY_ROW_CAP,
    ENTITY_SAMPLE_PER_REASON,
    TraceRun,
)
from model_tracks.config import load_config
from model_tracks.preflight import preflight

_LOG = RunLogger(__name__)

#: The stage name prepare_all runs this module as, and the name this stage's
#: consolidated-trace rows carry (core.tracing ``stage`` column).
STAGE = 'suite_inputs'


def _source_layout_key() -> str:
    """The declared source-code neighborhood (paths.yaml layouts block)."""
    from core.common import LAYOUTS
    return str(LAYOUTS['source_code_dir'].template)


def package_manifest() -> str:
    """The inputs bundle's manifest member name (``bundle.manifest_inputs``)."""
    from core.bundle import BundleRole, manifest_name
    return manifest_name(BundleRole.inputs)


RECOVERY_SCHEMA = 'er-suite-recovery-v1'
from model_tracks.resume import GNN_ONLY_TRACKS as GRAPH_TRACKS
#: Lane configs inlined into the package: the trained graph lane plus the
#: cascade combinator (which consumes those trained artifacts).
PACKAGED_LANES = (GRAPH_TRACKS + ('cascade',))
TRACK_CONFIGS = frozenset({'gnn_only.yaml', 'cascade.yaml', 'text.yaml'})
INPUT_KEYS = ('dataset_deduped', 'labeled_pairs', 'canonical_records', 'gate_results')
RECOVERY_EXCLUDED = frozenset({'wandb', 'mps_pipe', 'mps_log', '.git', '.dvc'})


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
    from core.common import prepared_setup_layout
    return prepared_setup_layout()


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


def _release_bundle(path: Path) -> None:
    """Drop the supervisor's cached reference as well as caller-owned objects."""
    from training.preparation_run import active_preparation
    run = active_preparation()
    if run is not None:
        run.release_bundle(path)
    gc.collect()
    send(f'[package] released_bundle={path.name} peak_rss_mb={rss_mb()}')


def _snapshot_pinned_items() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The declared pinned files/configs (training.yaml packaging block)."""
    from core.common import training_cfg
    pinned = training_cfg().packaging.snapshot_pinned_files
    configs = training_cfg().packaging.snapshot_pinned_configs
    return pinned, configs


@timed
def runtime_snapshot_files(*, ablation_config: Path | None = None) -> dict[str, Path]:
    """Shared local source/config overlay for prepared Colab jobs."""
    from core.common import TRAIN_ROOT
    files = {}
    with trace_step('snapshot.collect_source_files'):
        for path in tracked((p for p in _walk_files(TRAIN_ROOT / _source_layout_key(),
                            excluded_dirs=frozenset({'__pycache__'})) if p.suffix == '.py'), total=None,
                            desc='snapshot.source_files'):
            files[path.relative_to(TRAIN_ROOT).as_posix()] = path
    with trace_step('snapshot.pin_scripts_and_configs'):
        pinned_files, pinned_configs = _snapshot_pinned_items()
        for name in pinned_files:
            files[name] = TRAIN_ROOT/name
        for name in pinned_configs:
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
    """Stream JSON to a sibling file, publishing only on success.

    ``core.manifest.atomic_write_stream`` owns the sibling + fsync + replace
    mechanism (and its ``.tmp-<pid>`` residue contract), so a large payload is
    never buffered just to reuse :func:`core.manifest.atomic_write_json`.
    """
    from model_tracks.training_data import TrainingJSONEncoder
    from core.manifest import atomic_write_stream
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_write_stream(path) as handle:
        json.dump(model, handle, cls=TrainingJSONEncoder, ensure_ascii=False)
        handle.write('\n')


@timed
def _prepare_shared_population(setup, bundle):
    """Project the bundle, then release the shared population before tokenization."""
    from model_tracks.training_data import from_bundle, TrackTrainingBinding
    from model_tracks.shared_graph_data import prepare_shared_graph
    with trace_step('package.build_shared_population'):
        shared = from_bundle(bundle)
    with trace_step('package.write_shared_files'):
        _dump_model_json(shared, setup / _setup_layout().shared_training_data)
        text_binding = TrackTrainingBinding(track='text', shared_data_size=shared.fingerprint,
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
        layout = _setup_layout()
        track_settings = [load_graph_config(setup/(track+'.yaml'), expected_track=track).model_dump() for track in GRAPH_TRACKS]
        sizes = {settings['inference_batch_size'] for settings in track_settings}
        if len(sizes) != 1:
            raise ValueError('shared prepared graph inference batch sizes must agree')
        prepare_training(setup/layout.prepared_dir/layout.listings,setup/layout.prepared_dir/'pairs.csv',batch_size=sizes.pop())


def _streamed_json_size(value) -> str:
    """STREAMED, never materialized byte size for composition keys.

    `json.dumps` builds the whole document as one contiguous string before
    measuring; the packaging pass composes thousands of rows while the token
    cache is still resident, so `JSONEncoder.iterencode` is used with the
    SAME kwargs the emitted bytes always had — the size recorded for a row is
    identical to the shared encoder's output; only peak memory drops.
    """
    size = ByteCount()
    for chunk in json.JSONEncoder(sort_keys=True, ensure_ascii=False).iterencode(value):
        size.update(chunk.encode())
    return size.total


def _make_composer():
    """Build deterministic endpoint text with a bounded LRU cache."""
    from model_tracks.training_data import frozen_endpoint_text
    from core.model_input import model_input_composition,build_sku_text,model_input_info
    from core.sku_identity import row_identity
    from graph_tracks.text_cache import composition_fingerprint
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
        # empty cell is a value rather than a reason to recompose. The endpoint
        # row itself is verified once per endpoint by the shared graph projection.
        def compose(row):
            nonlocal cache_bytes
            frozen = frozen_endpoint_text(row.get('sku_id'), row.get('frozen_payload'),
                                          column_present='frozen_payload' in row)
            if frozen is not None:
                return frozen
            key = _streamed_json_size({'row':row,'composition':composition_contract})
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
    with trace_step('package.ablation_suite'):
        # Ablation staging lives in the CPU data bundle: prepare_suite mints the
        # per-track templates (tokens/tensors/frozen requests) the training
        # suite forwards from on the GPU, so no accelerator session stages them.
        #
        # UNCONDITIONAL when post_training_ablation is on: a bundle without the
        # templates makes the GPU worker emit attribute_ablation_export/skipped
        # (worker.py) and push the ablation to a local/CPU run. That is exactly
        # the regression we must not reintroduce, so the perf opt-out
        # (ER_PERF_BUNDLE_ABLATION_STAGING / ER_PERF_LEGACY) may NOT drop this
        # staging, and the bundle is verified to actually carry the templates.
        if cfg.post_training_ablation:
            from model_tracks.staged_ablation import prepare_suite
            from model_tracks.post_training_ablation import ABLATION_TRACKS
            spec = bundle_spec()
            prepare_suite(setup,Path(resolve_model(cfg.text_model)),TRAIN_ROOT/cfg.ablation_config,composer=compose,token_cache=token_cache,bundle=bundle)
            missing = [track for track in ABLATION_TRACKS
                       if not (setup/spec.ablation_templates_dir/track/spec.ablation_request_file).is_file()]
            if missing:
                raise RuntimeError(
                    'ablation templates missing from the prepared bundle for '
                    f'{missing}; the GPU session would skip ablation and push it '
                    'to a local CPU run (owner order: ablation is GPU inference)')
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
           for track in PACKAGED_LANES},
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
    """Publish the immutable suite package: stage, preflight, collect, write."""
    from training.prepared_bundle import load_prepared_bundle
    bundle_path, setup, cfg = _resolve_package_paths(config)
    _require_absent_output(output)
    with trace_step('package.load_bundle', bundle=bundle_path.name):
        _, bundle = load_prepared_bundle(bundle_path)
    try:
        native_model, checks = _stage_and_preflight(config, cfg, setup, bundle,
                                                    bundle_path)
    finally:
        del bundle
        _release_bundle(bundle_path)
    files = _collect_package_sources(cfg, setup, bundle_path)
    sealed, inlined = _write_package_archive(output, cfg, setup, files, checks)
    # The stage's rows into the ONE consolidated trace (core.tracing): the
    # member census with sizes, and the SEAL the transport actually carries.
    # Committed after the archive exists so the rows describe published bytes.
    trace = TraceRun(STAGE)
    PackageTrace.record(trace, files=files, inlined=inlined, sealed=sealed,
                        checks=checks)
    trace.write()
    send(f'[package] archive={sealed.path} rss_mb={rss_mb()}')
    return sealed.path


def _require_absent_output(output: Path) -> None:
    """The package name is exclusive: refuse to overwrite any existing path."""
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)


def _resolve_package_paths(config: Path) -> tuple[Path, Path, Any]:
    """Config-owned paths: the suite yaml, its setup dir and text bundle."""
    from core.common import TRAIN_ROOT
    cfg = load_config(config)
    return ((Path(TRAIN_ROOT) / cfg.text_bundle).resolve(),
            (Path(TRAIN_ROOT) / cfg.setup_dir).resolve(), cfg)


def _stage_and_preflight(config: Path, cfg, setup: Path, bundle,
                         bundle_path: Path) -> tuple[Any, dict]:
    """Shared population + graph inputs + exports, then the package preflight.

    Release order is deliberate: the bundle ref leaves after exports (the
    downstream GPU workers are the last consumers), and the token cache
    materializes the native model the preflight validates.
    """
    _prepare_shared_population(setup, bundle)
    gc.collect()
    _prepare_graph_inputs(setup)
    try:
        native_model = _prepare_exports(cfg, setup, bundle)
    finally:
        _release_bundle(bundle_path)
    try:
        with trace_step('package.preflight'):
            # The package preflight runs over shipped bytes; nothing here
            # tolerates a not-yet-produced input (owner directive 2026-10-08).
            return native_model, preflight(config)
    finally:
        del native_model
        _release_bundle(bundle_path)


def _write_package_archive(output: Path, cfg, setup: Path,
                           files: dict, checks: dict) -> tuple[Any, dict[str, str]]:
    """Inline the portable configs, then seal the inputs bundle for one run.

    Returns the sealed Bundle (its manifest carries the member inventory the
    boundary re-checks) and the inlined config members, so the caller can trace
    what was sealed without reopening the archive.
    """
    from core.bundle import Bundle, BundleRole
    from core.common import git_revision
    _require_absent_output(output)
    with trace_step('package.inline_configs'):
        inline = {name: yaml.safe_dump(value, sort_keys=False)
                  for name, value in _portable_config_models(cfg, setup).items()}
    with trace_step('package.write_archive'):
        # The Bundle owns the one writer (size-while-writing + one verify), so
        # the inputs archive is sealed here exactly like every other crossing.
        sealed = Bundle.seal_archive(
            output, files, role=BundleRole.inputs, inline=inline, profile=True,
            metadata={'schema': 'er-model-tracks-package-v1',
                      'revision': git_revision(), 'preflight': checks})
    return sealed, inline


def _member_bytes(path: Path) -> int:
    """A member source's size, or 0 when it is not a readable regular file."""
    try:
        return int(path.stat().st_size) if path.is_file() else 0
    except OSError:
        return 0


class PackageTrace:
    """Stage ``suite_inputs`` in the ONE consolidated trace (``core.tracing``).

    The packaging step sealed one archive and published its member inventory,
    but reported nothing to the trace: "what shipped, how big, and its member
    inventory" meant reopening the archive, and a shrunk or bloated member set was
    invisible until a GPU worker failed. Emitted here:

      run   members.collected    collected candidate members -> the members the
                                 writer actually seals (the manifest member is
                                 container metadata and is dropped by the
                                 writer), with the per-group census in detail
      group member.reason_census one row per member GROUP, EXACT counts
      ent   member               each sampled member, NAMED, with its byte size
                                 and its local source path
      run   member.sample_budget the entity-sampling budget actually spent
      run   archive.sealed       the sealed archive: whole-file size, bytes,
                                 member count, the inlined portable configs and
                                 the preflight verdict

    A member's GROUP is a READBACK classification derived from its portable
    name (:func:`member_group`); it never affects what is collected or sealed.
    Caps are core.tracing's own (ENTITY_SAMPLE_PER_REASON / ENTITY_ROW_CAP).
    """

    @staticmethod
    def member_group(member: str) -> str:
        """The declared member class of one portable member name (readback only)."""
        name = str(member)
        if name.startswith('config/'):
            return 'config'
        if name.startswith('scripts/') or name.endswith('.py'):
            return 'source_code'
        if name.endswith('.csv'):
            return 'inputs_csv'
        if 'text_prepared' in name:
            return 'text_bundle'
        if name.endswith('.npz'):
            return 'graph_tensors'
        if name.endswith(('.yaml', '.yml', '.json')):
            return 'declared_config'
        return 'other'

    @classmethod
    def records(cls, files: dict, inlined: dict) -> list[dict]:
        """One record per archive member: its portable name, size and source."""
        records = [
            {'member': str(name), 'bytes': _member_bytes(path),
             'source': str(path), 'group': cls.member_group(name)}
            for name, path in files.items()
        ]
        records.extend(
            {'member': str(name), 'bytes': len(str(text).encode('utf-8')),
             'source': 'rendered at seal (package.inline_configs)',
             'group': cls.member_group(name)}
            for name, text in inlined.items()
        )
        return records

    @classmethod
    def record(cls, trace: TraceRun, *, files: dict, inlined: dict,
               sealed, checks) -> None:
        """The member census, the per-member sizes and the seal of one run."""
        manifest_member = str(sealed.manifest_name)
        kept = {name: path for name, path in files.items() if name != manifest_member}
        records = cls.records(kept, inlined)
        by_group: dict[str, dict[str, int]] = {}
        for record in records:
            bucket = by_group.setdefault(record['group'], {'members': 0, 'bytes': 0})
            bucket['members'] += 1
            bucket['bytes'] += record['bytes']
        collected_bytes = sum(record['bytes'] for record in records)
        trace.add(
            'members', 'collected',
            in_count=len(files), out_count=len(kept),
            reason=(
                'every file collected for the package becomes a portable '
                'member; the bundle manifest is container metadata and is '
                'refused as a member by the writer, and the inlined portable '
                'configs are rendered at seal time'
            ),
            detail={
                'target': str(_target()),
                'manifest_member': manifest_member,
                'archive_members': len(records),
                'member_bytes': collected_bytes,
                'inlined_configs': sorted(str(name) for name in inlined),
                'by_group': by_group,
            },
            source='model_tracks.package._collect_package_sources',
        )
        trace.add_entities(
            'member', records,
            key_of=lambda record: record['member'],
            reason_of=lambda record: record['group'],
            detail_of=lambda record: {'bytes': record['bytes'],
                                      'source': record['source']},
            source='the sealed inputs bundle (portable member names)',
            per_reason=ENTITY_SAMPLE_PER_REASON,
            total_cap=ENTITY_ROW_CAP,
        )
        archive = Path(sealed.path)
        trace.add(
            'archive', 'sealed',
            reason=(
                'the inputs bundle is the one transport artifact: sealed and '
                'verified once by its writer, so its whole-file size is the '
                'token every boundary re-checks'
            ),
            detail={
                'path': str(archive),
                'size': sealed.path.stat().st_size,
                'bytes': _member_bytes(archive),
                'members': len(records),
                'source_bytes': collected_bytes,
                'manifest_member': manifest_member,
                'role': str(sealed.role.value),
                'inline_configs': sorted(str(name) for name in inlined),
                'schema': str(sealed.manifest.get('schema', '')),
                'revision': str(sealed.manifest.get('revision', '')),
                'preflight_checks': (sorted(str(name) for name in checks)
                                     if isinstance(checks, dict) else []),
            },
            source='model_tracks.package._write_package_archive (Bundle.seal_archive)',
        )


@timed
def verify(path: Path) -> dict:
    """Verify one sealed inputs bundle at its boundary and return its manifest."""
    from core.bundle import Bundle, BundleRole
    return Bundle.load(Path(path), BundleRole.inputs).manifest


def _recovery_sources(output: Path, destination: Path) -> dict[str, Path]:
    from core.bundle import Bundle
    spec = bundle_spec()
    files = {}
    for path in tracked(_walk_files(output, excluded_dirs=RECOVERY_EXCLUDED,
                                   excluded_suffixes=(spec.publication_sidecar_suffix,
                                                      spec.payload_suffix)),
                        desc='recovery.scan_files'):
        if path.name in set(spec.local_only_filenames) or path.resolve() == destination:
            continue
        files[path.relative_to(output).as_posix()] = path
    Bundle.assert_recovery_retains(output, files)
    return files


@timed
def recovery_package(output: Path, destination: Path, run_tag: str, *, input_package: dict | None = None) -> Path:
    """Seal stopped workers' portable state as one recovery bundle.

    The recovery role is the ``all epochs + optimizer`` contract: nothing is
    selected away, and the sealed archive is written and verified once by its
    writer (:meth:`core.bundle.Bundle.seal_archive` sizes every member as it
    writes), so no caller re-reads the sealed bytes for a transport token.
    """
    from core.bundle import Bundle, BundleRole
    spec = bundle_spec()
    output, destination = Path(output).resolve(), Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    with trace_step('recovery_package.check_manifest'):
        if _read_json(output / spec.suite_manifest_file).get(spec.run_tag_key) != run_tag:
            raise ValueError('recovery suite run mismatch')
    files = _recovery_sources(output, destination.resolve())
    sealed = Bundle.seal_archive(
        destination, files, role=BundleRole.recovery,
        metadata={'schema': RECOVERY_SCHEMA, spec.run_tag_key: run_tag,
                  'input_package': input_package})
    return sealed.path


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
    """Restore ZIP or tar.zst once through the verified recovery Bundle.

    The archive is verified exactly once at the :meth:`core.bundle.Bundle.load`
    boundary (member sizes, traversal and symlink safety), and the verified
    tree is then materialized and published only after every size check passed.
    """
    from core.bundle import Bundle, BundleRole
    spec = bundle_spec()
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    handle = Bundle.load(Path(archive), BundleRole.recovery)
    if handle.run_tag() != run_tag:
        raise ValueError('recovery suite run mismatch')
    if _read_json_member(handle, spec.suite_manifest_file).get(spec.run_tag_key) != run_tag:
        raise ValueError('recovery suite manifest mismatch')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + output.name + '-', dir=output.parent) as temp:
        staging = Path(temp) / 'payload'
        staging.mkdir()
        with trace_step('restore_recovery.unpack', members=len(handle.members())):
            handle.materialize(staging.resolve())
        _publish_recovery(staging, output)
    return output


def _read_json_member(handle, member: str) -> Any:
    return json.loads(handle.read(member))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(package(args.config, args.output))


if __name__ == '__main__':
    main()
