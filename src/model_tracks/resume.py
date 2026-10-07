"""Portable suite identity and verified worker completion for recovery."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal
from core.archive_reader import open_archive
from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator
from core.portable_archive import Digest, RuntimeSnapshot
from model_tracks.config import SuiteConfig

TRACKS = ('text', 'gnn_only', 'hybrid')


Track = Literal['text', 'gnn_only', 'hybrid']


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


def validate_training_binding(document: dict[str, Any], inputs: dict[str, Any],
                              settings: SuiteConfig, run_tag: str) -> TrainingInputBinding:
    binding = TrainingInputBinding.model_validate(document)
    source_inventory = {relative: expected for relative, expected in inputs['files'].items()
                        if relative.startswith(('src/', 'config/', 'scripts/'))}
    if settings.ablation_config in inputs['files']:
        source_inventory[settings.ablation_config] = inputs['files'][settings.ablation_config]
    if (binding.run_tag != run_tag or binding.settings != settings
            or binding.inputs != inputs['preflight']
            or binding.resume_identity.implementation != source_inventory):
        raise ValueError('Training suite differs from verified input/config/runtime snapshot')
    return binding


def validate_archived_track(bundle, manifest: dict[str, Any], track: Track,
                            *, postprocess_complete: bool) -> TrackInventory:
    """One completion contract for downloaded and recovered worker generations."""
    inventory = TrackInventory.model_validate_json(bundle.read(track + '/track_inventory.json'))
    marker = TrackCompletion.model_validate_json(bundle.read(track + '/track_complete.json'))
    if inventory.track != track or marker != TrackCompletion(
            track=track, status='ok', postprocess_complete=postprocess_complete):
        raise ValueError(f'archive contains an incomplete track: {track}')
    for relative, expected in inventory.files.items():
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError(f'unsafe completion artifact: {relative}')
        if manifest['files'].get(track + '/' + relative) != expected:
            raise ValueError(f'archive lacks current artifact: {track}/{relative}')
    return inventory


def validate_completed_suite_archive(archive: Path, run_tag: str,
                                     *, settings: SuiteConfig | None = None) -> dict[str, Any]:
    """Validate CPU completion semantics as well as ZIP byte integrity."""
    from core.portable_archive import verify_archive
    from graph_tracks.report_manifest import TrackReportManifest
    metadata = verify_archive(archive, 'suite_bundle_manifest.json')
    if metadata.get('run_tag') != run_tag:
        raise ValueError('completed archive belongs to a different run')
    with open_archive(archive) as bundle:
        binding = TrainingInputBinding.model_validate_json(bundle.read('suite_manifest.json'))
        if binding.run_tag != run_tag or settings is not None and binding.settings != settings:
            raise ValueError('completed archive suite configuration differs')
        for track in TRACKS:
            inventory = validate_archived_track(bundle, metadata, track, postprocess_complete=True)
            suffix = ('text__completion_manifest.json' if track == 'text'
                      else track + '__report_manifest.json')
            reports = [relative for relative in inventory.files
                       if Path(relative).name == suffix
                       and not any(part.startswith('interrupted-') or '.interrupted-' in part
                                   for part in Path(relative).parts)]
            if len(reports) != 1:
                raise ValueError('completed archive lacks one calibrated track report: ' + track)
            report = TrackReportManifest.model_validate_json(bundle.read(track + '/' + reports[0]))
            if report.track != track or report.test_reported and not binding.settings.report_test:
                raise ValueError('completed archive report configuration differs: ' + track)
            if binding.settings.post_training_ablation:
                from model_tracks.post_training_ablation import SavedAblationReport
                path = 'ablation/report.json'
                if path not in inventory.files:
                    raise ValueError('completed archive lacks saved ablation: ' + track)
                ablation = SavedAblationReport.model_validate_json(bundle.read(track + '/' + path))
                if (ablation.track != track or ablation.threshold != report.threshold
                        or ablation.threshold_binding.track != track
                        or ablation.threshold_binding.checkpoint_sha256 != report.checkpoint_sha256):
                    raise ValueError('completed archive ablation calibration differs: ' + track)
    return metadata


def verify_suite_archive(archive: Path, output: Path, run_tag: str, identity: dict[str, Any],
                         *, postprocess_complete: bool) -> dict[str, Any]:
    """Reuse only an archive containing the verified current worker generation."""
    from core.portable_archive import verify_archive
    import zipfile
    manifest = verify_archive(archive, 'suite_bundle_manifest.json')
    if manifest.get('run_tag') != run_tag:
        raise ValueError('existing archive belongs to a different suite')
    with open_archive(archive) as bundle:
        archived_suite = json.loads(bundle.read('suite_manifest.json'))
        if _training_identity(archived_suite.get('resume_identity')) != _training_identity(identity):
            raise ValueError('existing archive has different suite provenance')
        for track in TRACKS:
            if not completed_track(output / track, track, postprocess_complete=postprocess_complete):
                raise ValueError(f'incomplete track: {track}')
            inventory = TrackInventory.model_validate_json((output / track / 'track_inventory.json').read_text())
            archived_inventory = validate_archived_track(
                bundle, manifest, track, postprocess_complete=postprocess_complete)
            if archived_inventory != inventory:
                raise ValueError(f'existing archive contains stale worker artifacts: {track}')
    return manifest


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


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
    frozen_names = {'eligible_catalog.csv', 'listing_splits.csv', 'listing_pairs.csv',
                    'shared_minilm__embeddings.npz', 'setup_manifest.json'}
    identity['setup'] = {path.relative_to(setup).as_posix(): digest(path)
                         for path in setup.rglob('*') if path.is_file()
                         and (path.parent == setup and path.name in frozen_names
                              or 'prepared' in path.relative_to(setup).parts and path.suffix in {'.csv', '.json'})}
    from graph_tracks.config import load_config as load_graph_config, load_text_config
    lanes = {}
    for track in ('gnn_only', 'hybrid', 'text'):
        path = setup / (track + '.yaml')
        lane = (load_text_config(path) if track == 'text' else load_graph_config(path, expected_track=track)).model_dump()
        # Input hashes bind content; locations differ in the portable archive.
        for key in ('listings', 'pairs', 'input_manifest', 'text_cache', 'output_dir'):
            lane.pop(key, None)
        lanes[track] = lane
    identity['lanes'] = lanes
    from model_tracks.package import runtime_snapshot_files
    implementation = RuntimeSnapshot(files=runtime_snapshot_files(
        ablation_config=TRAIN_ROOT / cfg.ablation_config)).inventory()
    return {'schema': 'er-suite-resume-v1', 'run_tag': run_tag, 'config': settings,
            'inputs': identity, 'implementation': implementation}


def validate_suite(output: Path, identity: dict[str, Any]) -> None:
    path = output / 'suite_manifest.json'
    if not path.is_file():
        raise ValueError('resume requires an existing suite manifest')
    prior = json.loads(path.read_text())
    if _training_identity(prior.get('resume_identity')) != _training_identity(identity):
        raise ValueError('resume provenance mismatch: run, configuration or frozen inputs changed')


def artifact_files(output: Path) -> list[Path]:
    excluded = {'wandb', 'mlruns', 'profiles', '_artifact_publications', '.dvc', '.git'}
    return [path for path in output.rglob('*') if path.is_file() and not path.is_symlink()
            and not excluded.intersection(path.relative_to(output).parts)
            and path.name not in {'track_complete.json', 'track_inventory.json', 'worker.yaml', 'worker_events.jsonl'}
            and not path.name.endswith('.log')]


def record_completion(output: Path, track: Track, *, postprocess_complete: bool = True) -> None:
    files = {path.relative_to(output).as_posix(): digest(path) for path in artifact_files(output)}
    if not files:
        raise ValueError(f'cannot complete empty track: {track}')
    from core.manifest import atomic_write_text
    inventory = TrackInventory(track=track, files=files)
    completion = TrackCompletion(track=track, status='ok', postprocess_complete=postprocess_complete)
    atomic_write_text(output / 'track_inventory.json', inventory.model_dump_json(indent=2) + '\n')
    atomic_write_text(output / 'track_complete.json', completion.model_dump_json() + '\n')


def completed_track(output: Path, track: Track, *, postprocess_complete: bool = True) -> bool:
    marker = output / 'track_complete.json'
    if not marker.exists():
        return False
    completion = TrackCompletion.model_validate_json(marker.read_text())
    if completion != TrackCompletion(track=track, status='ok', postprocess_complete=postprocess_complete):
        raise ValueError(f'invalid completion marker: {track}')
    inventory_path = output / 'track_inventory.json'
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


def graph_checkpoint(output: Path, track: Track, run_tag: str) -> Path | None:
    # Graph trainers isolate their own run inside the worker output root.
    # Retain flat-root compatibility for previously materialized trees.
    roots = [output, output / f'{track}__{run_tag}']
    paths = [path for folder in roots
             for path in (folder / '_checkpoints' / track / f'{run_tag}_f0').glob(
                 f'checkpoint-*/{track}__graph_model.pt')]
    return max(paths, key=lambda path: int(path.parent.name.split('-')[-1]), default=None)
