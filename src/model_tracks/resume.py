"""Portable suite identity and verified worker completion for recovery."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal
from core.portable_archive import (
    Digest, RuntimeSnapshot, cached_file_digest,
)
from core.step_trace import timed
from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator
from model_tracks.config import SuiteConfig


def _spec():
    """The bundle contract from config (lazy import keeps the cycle open)."""
    from core.bundle import bundle_spec
    return bundle_spec()


def _setup_layout():
    """The declared prepared-setup layout (training.preparation.graph_setup)."""
    from core.common import prepared_setup_layout
    return prepared_setup_layout()

TRACKS = ('text', 'gnn_only', 'cascade')
#: Tracks that train behind the shared start barrier. The cascade trains
#: nothing: it composes the trained text ranker and gnn_only scorer after them.
TRAINING_TRACKS = ('text', 'gnn_only')
#: Tracks that run after training by composing trained artifacts (no barrier).
POSTPROCESS_TRACKS = ('cascade',)


Track = Literal['text', 'gnn_only', 'cascade']


class TrackCompletion(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    track: Track
    status: Literal['ok']
    postprocess_complete: StrictBool


class TrackInventory(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    track: Track
    files: dict[str, Digest] = Field(min_length=1)

    @model_validator(mode='after')
    def check_members(self):
        for relative in self.files:
            path = Path(relative)
            if not path.parts or path.is_absolute() or '..' in path.parts or path.as_posix() != relative:
                raise ValueError(f'unsafe completion artifact: {relative}')
        return self


class RuntimeBinding(BaseModel):
    model_config = ConfigDict(extra='allow')
    implementation: dict[str, Digest]


class TrainingInputBinding(BaseModel):
    model_config = ConfigDict(extra='allow')
    run_tag: str = Field(pattern=r'^[A-Za-z0-9_-]+$')
    inputs: dict[str, Any]
    resume_identity: RuntimeBinding
    settings: SuiteConfig = Field(alias='config')


def _training_identity(identity: dict[str, Any] | None) -> dict[str, Any]:
    return {key: value for key, value in (identity or {}).items() if key != 'implementation'}


def _events_skip_ablation(text: str) -> bool:
    """True when an event stream records a deliberate ablation-export skip.

    The GPU worker emits the skip as ``attribute_ablation_export``/``skipped``
    (bundle shipped no ablation templates); older/suite streams may use
    ``ablation``. The phases and status come from the bundle contract.
    """
    spec = _spec()
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if (event.get('phase') in spec.ablation_skip_phases
                and event.get('status') == spec.ablation_skip_status):
            return True
    return False


def recorded_ablation_skip(output: Path) -> bool:
    """True when a suite's event logs record that GPU ablation export was skipped.

    The skip lives in the per-track ``worker_events.jsonl`` (the suite event
    stream carries no ablation phase), so scan those as well as the suite log.
    """
    spec = _spec()
    candidates = [output / spec.suite_events_file]
    candidates += sorted(output.glob(f"*/{spec.worker_events_file}"))
    return any(events.is_file() and _events_skip_ablation(events.read_text())
               for events in candidates)


def runtime_source_inventory(files: dict[str, Digest], *,
                             ablation_config: str | None = None,
                             include_registry: bool = True) -> dict[str, Digest]:
    """The package members that form a suite's recorded runtime snapshot.

    Mirrors :func:`model_tracks.package.runtime_snapshot_files`: the checkout
    source/config/script neighborhoods, the pinned semantic family registry
    (which lives under ``artifacts/`` and would otherwise be dropped by a
    prefix-only filter), and the ablation config. ``include_registry=False``
    keeps only the checkout neighborhoods (the legacy source pin's stricter
    surface, which never hashes the gitignored registry).
    """
    from model_tracks.package import package_member
    inventory = {relative: expected for relative, expected in files.items()
                 if relative.startswith(('src/', 'config/', 'scripts/'))}
    if include_registry:
        registry = package_member('semantic_family_registry')
        if registry in files:
            inventory[registry] = files[registry]
    if ablation_config in files:
        inventory[ablation_config] = files[ablation_config]
    return inventory


def validate_training_binding(document: dict[str, Any], inputs: dict[str, Any],
                              settings: SuiteConfig, run_tag: str) -> TrainingInputBinding:
    """Check a downloaded result against the input package it was trained from.

    The input DATA identity is enforced: run tag, suite settings, and the
    package preflight block (``inputs``). The recorded runtime implementation is
    deliberately NOT compared against the input package's source inventory: the
    revision pin is removed by owner policy, so the trained checkout can be newer
    than the package that supplied the inputs. The binding still records the
    implementation actually used (``resume_identity.implementation``), which is
    what the result archive carries.
    """
    binding = TrainingInputBinding.model_validate(document)
    if (binding.run_tag != run_tag or binding.settings != settings
            or binding.inputs != inputs['preflight']):
        raise ValueError('Training suite differs from verified input/config/runtime snapshot')
    return binding


def validate_archived_track(source, manifest: dict[str, Any], track: Track,
                            *, postprocess_complete: bool) -> TrackInventory:
    """One completion contract for downloaded and recovered worker generations.

    ``source`` is any verified reader over the archive (an open ``open_archive``
    reader or :meth:`core.bundle.Bundle.reader`); it is never re-verified here.
    """
    spec = _spec()
    inventory = TrackInventory.model_validate_json(
        source.read(f'{track}/' + spec.inventory_file))
    marker = TrackCompletion.model_validate_json(
        source.read(f'{track}/' + spec.complete_file))
    if inventory.track != track or marker != TrackCompletion(
            track=track, status='ok', postprocess_complete=postprocess_complete):
        raise ValueError(f'archive contains an incomplete track: {track}')
    for relative, expected in inventory.files.items():
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError(f'unsafe completion artifact: {relative}')
        if manifest[spec.files_key].get(track + '/' + relative) != expected:
            raise ValueError(f'archive lacks current artifact: {track}/{relative}')
    return inventory


def validate_completed_suite_archive(archive: Path, run_tag: str,
                                     *, settings: SuiteConfig | None = None,
                                     bundle=None) -> dict[str, Any]:
    """Validate CPU completion semantics as well as archive byte integrity.

    ``bundle`` is the already-verified boundary handle for this archive (from
    :meth:`core.bundle.Bundle.load`, or the handle the sealing writer returned);
    when omitted the archive is verified here, exactly once.
    """
    from core.bundle import Bundle, BundleRole
    from graph_tracks.report_manifest import TrackReportManifest, report_member
    spec = _spec()
    if bundle is None:
        bundle = Bundle.load(Path(archive), BundleRole.result)
    metadata = bundle.manifest
    if metadata.get(spec.run_tag_key) != run_tag:
        raise ValueError('completed archive belongs to a different run')
    with bundle.reader() as source:
        binding = TrainingInputBinding.model_validate_json(
            source.read(spec.suite_manifest_file))
        if binding.run_tag != run_tag or settings is not None and binding.settings != settings:
            raise ValueError('completed archive suite configuration differs')
        # A GPU run that shipped no ablation templates records the deliberate
        # skip; the completed suite then legitimately has no saved ablation.
        names = source.namelist()
        ablation_skipped = (spec.suite_events_file in names
                            and _events_skip_ablation(
                                source.read(spec.suite_events_file).decode()))
        for track in TRACKS:
            inventory = validate_archived_track(source, metadata, track,
                                                postprocess_complete=True)
            suffix = report_member(track)
            reports = [relative for relative in inventory.files
                       if Path(relative).name == suffix
                       and not any(part.startswith('interrupted-') or '.interrupted-' in part
                                   for part in Path(relative).parts)]
            if len(reports) != 1:
                raise ValueError('completed archive lacks one calibrated track report: ' + track)
            report = TrackReportManifest.model_validate_json(
                source.read(track + '/' + reports[0]))
            if report.track != track or report.test_reported and not binding.settings.report_test:
                raise ValueError('completed archive report configuration differs: ' + track)
            if binding.settings.post_training_ablation and not ablation_skipped and track != 'cascade':
                from model_tracks.post_training_ablation import SavedAblationReport
                path = 'ablation/report.json'
                if path not in inventory.files:
                    raise ValueError('completed archive lacks saved ablation: ' + track)
                ablation = SavedAblationReport.model_validate_json(
                    source.read(track + '/' + path))
                if (ablation.track != track or ablation.threshold != report.threshold
                        or ablation.threshold_binding.track != track
                        or ablation.threshold_binding.checkpoint_sha256 != report.checkpoint_sha256):
                    raise ValueError('completed archive ablation calibration differs: ' + track)
    return metadata


def verify_suite_archive(archive: Path, output: Path, run_tag: str, identity: dict[str, Any],
                         *, gpu_only: bool = False):
    """Reuse only an archive containing the verified current worker generation.

    Returns the verified :class:`core.bundle.Bundle` handle, so the caller keeps
    the boundary digest instead of re-reading the archive for its transport
    token.
    """
    from core.bundle import Bundle, BundleRole
    spec = _spec()
    handle = Bundle.load(Path(archive), BundleRole.result)
    if handle.run_tag() != run_tag:
        raise ValueError('existing archive belongs to a different suite')
    with handle.reader() as source:
        archived_suite = json.loads(source.read(spec.suite_manifest_file))
        if _training_identity(archived_suite.get('resume_identity')) != _training_identity(identity):
            raise ValueError('existing archive has different suite provenance')
        for track in TRACKS:
            complete = expected_postprocess(track, gpu_only=gpu_only)
            if not completed_track(output / track, track, postprocess_complete=complete):
                raise ValueError(f'incomplete track: {track}')
            inventory = TrackInventory.model_validate_json(
                (output / track / spec.inventory_file).read_text())
            archived_inventory = validate_archived_track(
                source, handle.manifest, track, postprocess_complete=complete)
            if archived_inventory != inventory:
                raise ValueError(f'existing archive contains stale worker artifacts: {track}')
    return handle


def digest(path: Path) -> str:
    """SHA256 of one artifact, memoized per process on (path, mtime_ns, size).

    Completion is checked repeatedly over the same unchanged artifacts within a
    run; the cache removes the repeated reads while a rewrite (different
    mtime/size) still forces a fresh hash and is detected.
    """
    return cached_file_digest(path)


def suite_identity(cfg: SuiteConfig, inputs: dict[str, Any], run_tag: str) -> dict[str, Any]:
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
    layout = _setup_layout()
    frozen_names = {layout.catalog, layout.splits, layout.pairs,
                    layout.shared_embeddings, layout.manifest}
    identity['setup'] = {path.relative_to(setup).as_posix(): digest(path)
                         for path in setup.rglob('*') if path.is_file()
                         and (path.parent == setup and path.name in frozen_names
                              or 'prepared' in path.relative_to(setup).parts and path.suffix in {'.csv', '.json'})}
    from graph_tracks.config import load_config as load_graph_config, load_text_config
    lanes = {}
    for track in ('gnn_only', 'cascade', 'text'):
        path = setup / (track + '.yaml')
        lane = (load_text_config(path) if track == 'text' else load_graph_config(path, expected_track=track)).model_dump()
        # Input hashes bind content; locations differ in the portable archive.
        for key in ('listings', 'pairs', 'input_manifest', 'text_cache',
                    'text_index', 'gnn_checkpoint', 'output_dir'):
            lane.pop(key, None)
        lanes[track] = lane
    identity['lanes'] = lanes
    from model_tracks.package import runtime_snapshot_files
    implementation = RuntimeSnapshot(files=runtime_snapshot_files(
        ablation_config=TRAIN_ROOT / cfg.ablation_config)).inventory()
    return {'schema': 'er-suite-resume-v1', 'run_tag': run_tag, 'config': settings,
            'inputs': identity, 'implementation': implementation}


def validate_suite(output: Path, identity: dict[str, Any]) -> None:
    spec = _spec()
    path = output / spec.suite_manifest_file
    if not path.is_file():
        raise ValueError('resume requires an existing suite manifest')
    prior = json.loads(path.read_text())
    if _training_identity(prior.get('resume_identity')) != _training_identity(identity):
        raise ValueError('resume provenance mismatch: run, configuration or frozen inputs changed')


def selected_checkpoint_dirs(root: Path) -> frozenset[str]:
    """Posix dirs (relative to ``root``) of the checkpoints a track consumes.

    Retired into :meth:`core.bundle.Bundle.selected_checkpoint_dirs` (the
    selection contract lives with the bundle role it enforces); this name is
    kept only for external callers that still address it here.
    """
    from core.bundle import Bundle, BundleRole
    return Bundle.from_directory(root, BundleRole.result).selected_checkpoint_dirs()


def artifact_files(output: Path) -> list[Path]:
    """Result-archive artifacts: the bundle member predicate plus marker/log skips.

    The predicate and the selection both come from the ``result`` role; this
    only drops the per-track bookkeeping that must never inventory itself.
    """
    from core.bundle import Bundle, BundleRole
    spec = _spec()
    tree = Bundle.from_directory(output, BundleRole.result)
    selected = tree.selected_checkpoint_dirs()
    markers = {spec.complete_file, spec.inventory_file, spec.worker_config_file,
               spec.worker_events_file}
    return [path for path in output.rglob('*') if path.is_file() and not path.is_symlink()
            and tree.is_result_member(path.relative_to(output).as_posix(),
                                      selected_checkpoints=selected)
            and path.name not in markers and not path.name.endswith('.log')]



@timed
def record_completion(output: Path, track: Track, *, postprocess_complete: bool = True) -> None:
    spec = _spec()
    files = {path.relative_to(output).as_posix(): digest(path) for path in artifact_files(output)}
    if not files:
        raise ValueError(f'cannot complete empty track: {track}')
    from core.manifest import atomic_write_text
    inventory = TrackInventory(track=track, files=files)
    completion = TrackCompletion(track=track, status='ok', postprocess_complete=postprocess_complete)
    atomic_write_text(output / spec.inventory_file, inventory.model_dump_json(indent=2) + '\n')
    atomic_write_text(output / spec.complete_file, completion.model_dump_json() + '\n')


@timed
def completed_track(output: Path, track: Track, *, postprocess_complete: bool = True) -> bool:
    spec = _spec()
    marker = output / spec.complete_file
    if not marker.exists():
        return False
    completion = TrackCompletion.model_validate_json(marker.read_text())
    if completion != TrackCompletion(track=track, status='ok', postprocess_complete=postprocess_complete):
        raise ValueError(f'invalid completion marker: {track}')
    inventory_path = output / spec.inventory_file
    if not inventory_path.is_file():
        raise ValueError(f'completion inventory missing: {track}')
    inventory = TrackInventory.model_validate_json(inventory_path.read_text())
    if inventory.track != track:
        raise ValueError(f'invalid completion inventory: {track}')
    for relative, expected in inventory.files.items():
        path = output / relative
        if Path(relative).is_absolute() or '..' in Path(relative).parts or path.is_symlink():
            raise ValueError(f'unsafe completion artifact: {relative}')
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'completed artifact changed: {track}/{relative}')
    return True


def expected_postprocess(track: Track, *, gpu_only: bool) -> bool:
    """The completion flag a worker records for ``track``.

    Under ``gpu_only`` the trained lanes defer their CPU reports to local
    completion, so their markers land ``postprocess_complete=False``. The
    cascade is a pure postprocess combinator with no checkpoint to transport:
    its worker always finishes the report it composes from the trained
    artifacts, so its marker is complete regardless of ``gpu_only``.
    """
    return True if track in POSTPROCESS_TRACKS else not gpu_only


def graph_checkpoint(output: Path, track: Track, run_tag: str) -> Path | None:
    # Graph trainers isolate their own run inside the worker output root.
    # Retain flat-root compatibility for previously materialized trees.
    from graph_tracks.artifacts import name as artifact_name
    spec = _spec()
    roots = [output, output / f'{track}__{run_tag}']
    paths = [path for folder in roots
             for path in (folder / spec.checkpoint_dir / track / f'{run_tag}_f0').glob(
                 f'checkpoint-*/{artifact_name(track, "graph_model.pt")}')]
    return max(paths, key=lambda path: int(path.parent.name.split('-')[-1]), default=None)
