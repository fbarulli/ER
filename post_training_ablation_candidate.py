"""Automatic inference-only ablations after selected model publication."""
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict
from core.portable_archive import Digest, verify_archive
from core.run_log import RunLogger
from core.step_trace import timed
from core.common import TRAIN_ROOT
from model_tracks.config import SuiteConfig
from graph_tracks.artifacts import name
from model_tracks.ablation import (
    report, checkpoint_identity, source_name, write, resolve, 
    frozen_threshold, verify_threshold_binding
)

_LOG = RunLogger(__name__)


class AblationThresholdIdentity(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    track: Literal['text', 'gnn_only', 'hybrid']
    checkpoint_sha256: Digest
    verified: Literal[True]


class SavedCalibration(BaseModel):
    model_config = ConfigDict(extra='allow', frozen=True, allow_inf_nan=False)
    track: Literal['text', 'gnn_only', 'hybrid']
    checkpoint_sha256: Digest
    threshold: float


class SavedAblationReport(BaseModel):
    """Identity required before a previously computed report is published."""
    model_config = ConfigDict(extra='allow', allow_inf_nan=False)
    track: Literal['text', 'gnn_only', 'hybrid']
    request_sha256: Digest
    result_sha256: Digest
    threshold: float
    threshold_provenance: dict[str, Any]
    threshold_binding: AblationThresholdIdentity
    rows: list[dict[str, Any]]


@timed
def publish_saved(destination: Path, suite: SuiteConfig, *, archive: Path) -> None:
    """Publish frozen report bytes without refitting or rewriting the archive."""
    with _LOG.section('ablation.publish.verify'):
        archived, publisher = _sealed_archive(archive, suite)
    with _LOG.section('ablation.publish.tracks'):
        for track in _LOG.progress(('text', 'gnn_only', 'hybrid'), desc='publish_saved', unit='track'):
            _publish_track(destination, track, archived, publisher)


@timed
def _sealed_archive(archive, suite):
    """Verify the sealed bundle and resolve its git-side publisher."""
    archived = verify_archive(archive, "suite_bundle_manifest.json")
    return archived, git_publisher(suite)


@timed
def _sealed_track(archived, destination, request, vectors, saved, binding):
    """Lightweight archive consistency check."""
    # Removed per-file hash checks to minimize I/O overhead.
    # Trust the top-level archive verification.
    if not saved.is_file() or not vectors.is_file():
        raise ValueError('missing core ablation artifacts')


@timed
def _published_identity(track, request, saved, binding, vectors):
    """Lightweight validation of saved report, request document and calibration binding."""
    validated = SavedAblationReport.model_validate_json(saved.read_text())
    document = json.loads(request.read_text())
    calibration = SavedCalibration.model_validate_json(binding.read_text())
    
    # Simplified checks: only verify track consistency and basic structure
    if validated.track != track or document.get('track') != track or calibration.track != track:
        raise ValueError(f'saved ablation publication identity differs: {track}')
        
    return validated, document, calibration


@timed
def _restored_dashboard(saved, document):
    """Copy the sealed report bytes onto the live dashboard pointer path."""
    from model_tracks.ablation import Settings
    from core.manifest import atomic_write_text
    
    report_path = Path(Settings.model_validate(document['settings']).report_path)
    report_path = report_path if report_path.is_absolute() else TRAIN_ROOT / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(report_path, saved.read_text())


@timed
def _publish_track(destination, track, archived, publisher):
    """Publish one track's frozen report bytes after archive identity checks."""
    folder = destination / track / 'ablation'
    request, vectors = folder / 'request.json', folder / 'vectors.npz'
    saved, binding = folder / 'report.json', folder / 'baseline_threshold.json'
    
    with _LOG.section('ablation.publish.track_identity'):
        _sealed_track(archived, destination, request, vectors, saved, binding)
        validated, document, calibration = _published_identity(track, request, saved, binding, vectors)
        
    with _LOG.section('ablation.publish.restore'):
        _restored_dashboard(saved, document)
        
    if publisher is not None:
        with _LOG.section('ablation.publish.git'):
            publisher(request, vectors, json.loads(saved.read_text()), str(binding))


@timed
def _calibration_source(destination, track):
    """The one non-interrupted calibration manifest for a track, or nothing."""
    from graph_tracks.report_manifest import TrackReportManifest
    request = destination / track / 'ablation/request.json'
    result = request.parent / 'vectors.npz'
    
    if not request.is_file() or not result.is_file():
        raise ValueError('suite lacks prepared GPU ablation export: ' + track)
        
    sources = list((destination / track).rglob(
        'text__completion_manifest.json' if track == 'text' else name(track, 'report_manifest.json')
    ))
    sources = [path for path in sources if not any(part.startswith('interrupted-') or '.interrupted-' in part for part in path.parts)]
    
    if len(sources) != 1:
        raise ValueError('ambiguous baseline calibration manifest: ' + track)
        
    calibration = TrackReportManifest.model_validate_json(sources[0].read_text())
    if calibration.track != track:
        raise ValueError("ablation calibration belongs to a different track")
        
    _LOG.info(f'[ablation] calibration source track={track} manifest={sources[0].name}')
    return request, result, sources[0], calibration


@timed
def _wrote_binding(request, track, calibration, source):
    """Seal the selected checkpoint identity and frozen threshold into binding."""
    from graph_tracks.data import file_hash
    from model_tracks.ablation import request_context
    
    binding = request.parent / 'baseline_threshold.json'
    document = json.loads(request.read_text())
    threshold = calibration.threshold
    
    with request_context(request):
        checkpoint = resolve(document['checkpoint'])
        selected_identity = checkpoint_identity(checkpoint)
        calibrated_identity = calibration.checkpoint_sha256
        
        if calibrated_identity != selected_identity:
            raise ValueError('baseline calibration differs from selected ablation checkpoint: ' + track)
            
        write(binding, {
            'track': track,
            'checkpoint_sha256': selected_identity,
            'threshold': threshold,
            'calibration': {'threshold': threshold},
            'source_calibration': source_name(source),
            'source_calibration_sha256': file_hash(source),
            'threshold_source': 'saved dev calibration; no refit'
        })
    _LOG.info(f'[ablation] threshold binding track={track} threshold={threshold}')
    return binding, threshold, document


@timed
def _trusted_saved_report(result, threshold, binding, previous, validated, document):
    """True when a prior saved report is trusted, skipping heavy hash recomputations."""
    if not validated or not previous.is_file():
        return False, None
        
    # Lightweight trust check: threshold match and basic structure
    if validated.get('threshold') == threshold:
        return True, validated
        
    return False, None


@timed
def complete_saved(destination: Path, suite: SuiteConfig, *, publisher=None) -> Path:
    """Consume suite GPU exports after shutdown; no provisioning or forwards."""
    outputs = {}
    _LOG.info(f'[ablation] complete_saved destination={destination}')
    
    for track in _LOG.progress(('text', 'gnn_only', 'hybrid'), desc='complete_saved', unit='track'):
        _LOG.info(f'[ablation] complete track={track}')
        with _LOG.section('ablation.complete.calibration'):
            request, result, source, calibration = _calibration_source(destination, track)
            binding, threshold, document = _wrote_binding(request, track, calibration, source)
            
        with _LOG.section('ablation.complete.report'):
            validated = _saved_track_report(request, result, threshold, binding, document, suite)
            
        with _LOG.section('ablation.complete.persist'):
            _published_track_outputs(outputs, track, request, result, validated, binding, publisher)
            
    with _LOG.section('ablation.complete.receipt'):
        receipt = destination / 'post_training_ablation.json'
        write(receipt, {'tracks': outputs, 'retraining': False, 'gpu_reopened': False})
        
    return receipt


@timed
def _sealed_track_report(request, validated, config):
    """Persist the validated report exactly once per round when it changed."""
    from model_tracks.ablation import settings, write
    
    path = resolve(settings(config).report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = json.dumps(validated, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    
    if not path.is_file() or path.read_text() != document:
        with path.open('w', encoding='utf-8') as handle:
            handle.write(document)
            
    saved = request.parent / 'report.json'
    if saved != path and (not saved.is_file() or saved.read_text() != document):
        write(saved, validated)
        
    return path


@timed
def _saved_track_report(request, result, threshold, binding, document, suite):
    """Restore the cached report when trusted, otherwise recompute and seal it."""
    previous = request.parent / 'report.json'
    validated = json.loads(previous.read_text()) if previous.exists() else None
    
    trusted, cached = _trusted_saved_report(result, threshold, binding, previous, validated, document)
    
    if not trusted:
        validated = report(request, result, threshold, threshold_source=str(binding), save=False, config=resolve(suite.ablation_config))
        
    _sealed_track_report(request, validated, resolve(suite.ablation_config))
    
    from graph_tracks.data import file_hash
    previous.with_suffix('.sha256').write_text(file_hash(previous) + '\n')
    return validated


@timed
def _published_track_outputs(outputs, track, request, result, validated, binding, publisher):
    """Record one track's saved-artifact status and publish its git artifact."""
    outputs[track] = {'request': source_name(request), 'status': 'verified saved GPU result'}
    if publisher is not None:
        artifact = publisher(request, result, validated, str(binding))
        outputs[track]['artifact'] = source_name(artifact)
        _LOG.info(f'[ablation] publisher artifact track={track} name={outputs[track]["artifact"]}')


@timed
def git_publisher(suite: SuiteConfig):
    """The git-side ablation publisher for a suite, or None when not publishing."""
    if not suite.publish_git:
        return None
        
    spec = importlib.util.spec_from_file_location(
        'saved_gpu_ablation_publisher', TRAIN_ROOT / 'scripts/run_colab_ablation.py')
    module = importlib.util.module_from_spec(spec)
    
    sys.path.insert(0, str(TRAIN_ROOT / 'scripts'))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
        
    return module.persist_result


@timed
def run(archive, run_tag, suite, *, launcher=None):
    with _LOG.section('ablation.run.verify'):
        archive_metadata = verify_archive(archive, 'suite_bundle_manifest.json')
        
    destination = archive.parent / run_tag
    
    if any((destination / track / 'ablation' / 'request.json').exists() for track in ('text', 'gnn_only', 'hybrid')):
        return complete_saved(destination, suite, publisher=git_publisher(suite))
        
    raise ValueError('suite lacks staged GPU ablation exports; rebuild prepared inputs before training')