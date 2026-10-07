"""Automatic inference-only ablations after selected model publication."""
import importlib.util
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from core.portable_archive import Digest
from core.run_log import RunLogger
from core.step_trace import timed
from model_tracks.config import SuiteConfig
from pathlib import Path
from core.common import TRAIN_ROOT
from core.portable_archive import verify_archive
from graph_tracks.artifacts import name
from model_tracks.ablation import report, checkpoint_identity, source_name, write, resolve, frozen_threshold, verify_threshold_binding

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


def _sealed_archive(archive, suite):
    """Verify the sealed bundle and resolve its git-side publisher."""
    archived = verify_archive(archive, "suite_bundle_manifest.json")
    return archived, git_publisher(suite)


def _sealed_track(archived, destination, request, vectors, saved, binding):
    """Raise unless every exported artifact matches the sealed archive bytes."""
    from graph_tracks.data import file_hash
    folder = request.parent
    for path in (request, vectors, saved, binding, folder / 'prepared_inputs.npz', saved.with_suffix('.sha256')):
        relative = path.relative_to(destination).as_posix()
        if archived['files'].get(relative) != file_hash(path):
            raise ValueError('saved ablation differs from sealed archive: ' + relative)


def _published_identity(track, request, saved, binding, vectors):
    """The validated saved report, request document and calibration binding."""
    from graph_tracks.data import file_hash
    validated = SavedAblationReport.model_validate_json(saved.read_text())
    document = json.loads(request.read_text())
    calibration = SavedCalibration.model_validate_json(binding.read_text())
    if (validated.track != track or document['track'] != track
            or validated.request_sha256 != file_hash(request)
            or validated.result_sha256 != file_hash(vectors)
            or saved.with_suffix('.sha256').read_text().strip() != file_hash(saved)
            or validated.threshold_provenance.get('sha256') != file_hash(binding)
            or calibration.track != track or validated.threshold != calibration.threshold
            or validated.threshold_binding.track != track
            or validated.threshold_binding.checkpoint_sha256 != calibration.checkpoint_sha256):
        raise ValueError('saved ablation publication identity differs: ' + track)
    return validated, document, calibration


def _restored_dashboard(saved, document):
    """Copy the sealed report bytes onto the live dashboard pointer path."""
    from model_tracks.ablation import Settings
    # Settings in the request are the frozen producer's settings. Live config
    # may have changed while the immutable training run was in flight.
    report_path = Path(Settings.model_validate(document['settings']).report_path)
    report_path = report_path if report_path.is_absolute() else TRAIN_ROOT / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    from core.manifest import atomic_write_text
    atomic_write_text(report_path, saved.read_text())


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
        publisher(request, vectors, json.loads(saved.read_text()), str(binding))


def _calibration_source(destination, track):
    """The one non-interrupted calibration manifest for a track, or nothing."""
    import json
    from graph_tracks.report_manifest import TrackReportManifest
    request = destination/track/'ablation/request.json'
    result = request.parent/'vectors.npz'
    if not request.is_file() or not result.is_file():
        raise ValueError('suite lacks prepared GPU ablation export: '+track)
    sources = list((destination/track).rglob('text__completion_manifest.json' if track == 'text' else name(track,'report_manifest.json')))
    sources = [path for path in sources if not any(part.startswith('interrupted-') or '.interrupted-' in part for part in path.parts)]
    if len(sources) != 1:
        raise ValueError('ambiguous baseline calibration manifest: '+track)
    calibration = TrackReportManifest.model_validate_json(sources[0].read_text())
    if calibration.track != track:
        raise ValueError("ablation calibration belongs to a different track")
    return request, result, sources[0], calibration


def _wrote_binding(request, track, calibration, source):
    """Seal the selected checkpoint identity and frozen threshold into binding."""
    from graph_tracks.data import file_hash
    from model_tracks.ablation import request_context
    binding = request.parent/'baseline_threshold.json'
    document = json.loads(request.read_text())
    threshold = calibration.threshold
    with request_context(request):
        checkpoint = resolve(document['checkpoint'])
        selected_identity = checkpoint_identity(checkpoint)
        calibrated_identity = calibration.checkpoint_sha256
        if calibrated_identity != selected_identity:
            raise ValueError('baseline calibration differs from selected ablation checkpoint: '+track)
        write(binding,{'track':track,'checkpoint_sha256':selected_identity,
            'threshold':threshold,'calibration':{'threshold':threshold},
            'source_calibration':source_name(source),
            'source_calibration_sha256':__import__('graph_tracks.data',fromlist=['file_hash']).file_hash(source),
            'threshold_source':'saved dev calibration; no refit'})
    return binding, threshold, document


def _trusted_saved_report(result, threshold, binding, previous, validated, document):
    """True when a prior saved report bytes-identically covers this result.

    Callers then only re-verify the vectors behind the cached report instead
    of recomputing the threshold-frozen comparison.
    """
    from graph_tracks.data import file_hash
    from model_tracks.ablation import request_context, validate_vectors
    request = previous.parent/'request.json'
    trusted = (validated and previous.with_suffix('.sha256').is_file()
               and previous.with_suffix('.sha256').read_text().strip() == file_hash(previous)
               and validated.get('request_sha256') == file_hash(request)
               and validated.get('result_sha256') == file_hash(result)
               and validated.get('threshold') == threshold
               and validated.get('threshold_provenance',{}).get('sha256') == file_hash(binding))
    if not trusted:
        return False, None
    with request_context(request):
        validate_vectors(request,result)
        attestation = frozen_threshold(str(binding),threshold)
        if validated.get('threshold_binding') != verify_threshold_binding(document,attestation) or validated.get('threshold_provenance') != attestation:
            raise ValueError('cached ablation calibration binding differs')
    return True, validated


@timed
def complete_saved(destination: Path, suite: SuiteConfig, *, publisher=None) -> Path:
    """Consume suite GPU exports after shutdown; no provisioning or forwards."""
    from graph_tracks.data import file_hash
    outputs = {}
    for track in _LOG.progress(('text','gnn_only','hybrid'), desc='complete_saved', unit='track'):
        with _LOG.section('ablation.complete.calibration'):
            request, result, source, calibration = _calibration_source(destination, track)
            binding, threshold, document = _wrote_binding(request, track, calibration, source)
        with _LOG.section('ablation.complete.report'):
            previous = request.parent/'report.json'
            validated = json.loads(previous.read_text()) if previous.exists() else None
            trusted, cached = _trusted_saved_report(result, threshold, binding, previous, validated, document)
            from model_tracks.ablation import save_report
            if trusted:
                validated = cached
            else:
                validated = report(request,result,threshold,threshold_source=str(binding),save=False,config=resolve(suite.ablation_config))
            save_report(request,validated,config=resolve(suite.ablation_config))
            previous.with_suffix('.sha256').write_text(file_hash(previous)+'\n')
        with _LOG.section('ablation.complete.persist'):
            outputs[track] = {'request':source_name(request),'status':'verified saved GPU result'}
            if publisher is not None:
                artifact = publisher(request,result,validated,str(binding))
                outputs[track]['artifact'] = source_name(artifact)
    receipt = destination/'post_training_ablation.json'
    write(receipt,{'tracks':outputs,'retraining':False,'gpu_reopened':False})
    return receipt


def git_publisher(suite: SuiteConfig):
    """The git-side ablation publisher for a suite, or None when not publishing.

    Split out of :func:`run` so a caller that must produce the reports BEFORE
    archiving them can still publish the very same artifacts in one pass,
    instead of running :func:`complete_saved` twice.
    """
    if not suite.publish_git:
        return None
    spec = importlib.util.spec_from_file_location(
        'saved_gpu_ablation_publisher', TRAIN_ROOT / 'scripts/run_colab_ablation.py')
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.path.insert(0, str(TRAIN_ROOT / 'scripts'))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module.persist_result


@timed
def run(archive, run_tag, suite, *, launcher=None):
    archive_metadata = verify_archive(archive,'suite_bundle_manifest.json')
    destination = archive.parent/run_tag
    if any((destination/track/'ablation/request.json').exists() for track in ('text','gnn_only','hybrid')):
        return complete_saved(destination,suite,publisher=git_publisher(suite))
    raise ValueError('suite lacks staged GPU ablation exports; rebuild prepared inputs before training')
