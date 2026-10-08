"""Automatic inference-only ablations after selected model publication."""
import importlib.util
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from core.bundle import bundle_spec
from core.portable_archive import Digest
from core.run_log import RunLogger
from core.step_trace import timed
from core.tracing import SCOPE_ENTITY, flush_stage_trace, stage_trace
from model_tracks.config import SuiteConfig
#: The track vocabulary is single-sourced in ``resume`` (text / gnn_only /
#: cascade). These ablation models used to re-spell it with the retired
#: ``hybrid`` member, so they could neither accept the landed cascade lane nor
#: stay in step with the rest of the codebase; importing the one Literal keeps
#: the saved-artifact identity contract on the same set as every other surface.
from model_tracks.resume import Track, TRAINING_TRACKS
from pathlib import Path
from core.common import TRAIN_ROOT
from graph_tracks.artifacts import name
from model_tracks.ablation import report, checkpoint_identity, source_name, write, resolve, frozen_threshold, verify_threshold_binding

_LOG = RunLogger(__name__)

#: The stage name this module owns in the ONE consolidated pipeline trace.
STAGE = "post_training_ablation"

#: The module's trace writer: the shared shim's slot (``None`` until first use;
#: see :func:`core.tracing.stage_trace`), so importing this module never touches
#: the trace layout. Never reset: the consumer runs once per track inside one
#: suite and ``core.tracing`` commits a stage run-scoped, so a reset would drop
#: the earlier track's rows.
_TRACE = None


def trace():
    """The ONE writer for the ``post_training_ablation`` stage of the current run."""
    global _TRACE
    _TRACE = stage_trace(STAGE, _TRACE)
    return _TRACE


def flush_trace():
    """Commit this process's ablation-consumer rows once; a no-op while empty."""
    return flush_stage_trace(_TRACE)


#: Tracks that produce a post-training ablation export. The trained lanes do;
#: the cascade trains nothing and ships no ablation, so it is excluded.
ABLATION_TRACKS = TRAINING_TRACKS


class AblationThresholdIdentity(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    track: Track
    checkpoint_sha256: Digest
    verified: Literal[True]


class SavedCalibration(BaseModel):
    model_config = ConfigDict(extra='allow', frozen=True, allow_inf_nan=False)
    track: Track
    checkpoint_sha256: Digest
    threshold: float


class SavedAblationReport(BaseModel):
    """Identity required before a previously computed report is published."""
    model_config = ConfigDict(extra='allow', allow_inf_nan=False)
    track: Track
    result_sha256: Digest
    threshold: float
    threshold_provenance: dict[str, Any]
    threshold_binding: AblationThresholdIdentity
    rows: list[dict[str, Any]]


@timed
def publish_saved(destination: Path, suite: SuiteConfig, *, archive: Path,
                 bundle=None) -> None:
    """Publish frozen report bytes without refitting or rewriting the archive.

    ``bundle`` is the already-verified boundary handle for ``archive`` (from
    :meth:`core.bundle.Bundle.load` or the sealing writer). Passing it keeps the
    publication path at one integrity check per VM crossing instead of
    re-verifying the same bytes here.
    """
    with _LOG.section('ablation.publish.verify'):
        archived, publisher = _sealed_archive(archive, suite, bundle=bundle)
        trace().add(
            "publish_saved", "verify",
            reason='the sealed archive is verified once and the trusted handle is reused',
            detail={'archive': source_name(archive), 'boundary_handle_supplied': bundle is not None,
                    'archive_sha256': getattr(archived, 'digest', None),
                    'publisher': publisher is not None},
            source=source_name(archive),
        )
    with _LOG.section('ablation.publish.tracks'):
        for track in _LOG.progress(ABLATION_TRACKS, desc='publish_saved', unit='track'):
            _publish_track(destination, track, archived, publisher)
    flush_trace()


def _sealed_archive(archive, suite, *, bundle=None):
    """The sealed bundle (verified only when no boundary handle was supplied)."""
    from core.bundle import Bundle, BundleRole
    archived = (Bundle.load(archive, BundleRole.result)
                if bundle is None else bundle)
    return archived, git_publisher(suite)


def _sealed_track(archived, destination, request, vectors, saved, binding):
    """Raise unless every exported artifact matches the sealed archive bytes."""
    from graph_tracks.data import file_hash
    folder = request.parent
    # ``archived`` is a verified Bundle handle or the plain verified manifest.
    manifest = getattr(archived, 'manifest', archived)
    for path in (request, vectors, saved, binding, folder / 'prepared_inputs.npz', saved.with_suffix('.sha256')):
        relative = path.relative_to(destination).as_posix()
        if manifest['files'].get(relative) != file_hash(path):
            trace().add(
                "publish_saved", "track_identity_rejected",
                scope=SCOPE_ENTITY, key=request.parent.parent.name,
                reason='the on-disk ablation artifact differs from the sealed archive byte-for-byte; '
                       'the track is quarantined, never republished',
                detail={'track': request.parent.parent.name, 'relative': relative,
                        'sealed_sha256': manifest['files'].get(relative),
                        'on_disk_sha256': file_hash(path)},
                source=source_name(path),
            )
            flush_trace()
            raise ValueError('saved ablation differs from sealed archive: ' + relative)


def _published_identity(track, request, saved, binding, vectors):
    """The validated saved report, request document and calibration binding."""
    from graph_tracks.data import file_hash
    validated = SavedAblationReport.model_validate_json(saved.read_text())
    document = json.loads(request.read_text())
    calibration = SavedCalibration.model_validate_json(binding.read_text())
    if (validated.track != track or document['track'] != track
            or validated.result_sha256 != file_hash(vectors)
            or saved.with_suffix('.sha256').read_text().strip() != file_hash(saved)
            or validated.threshold_provenance.get('sha256') != file_hash(binding)
            or calibration.track != track or validated.threshold != calibration.threshold
            or validated.threshold_binding.track != track
            or validated.threshold_binding.checkpoint_sha256 != calibration.checkpoint_sha256):
        trace().add(
            "publish_saved", "report_identity_rejected",
            scope=SCOPE_ENTITY, key=track,
            reason='the saved report, its request and its calibration binding do not agree',
            detail={'track': track, 'report': source_name(saved),
                    'validated_track': validated.track, 'document_track': document['track'],
                    'calibration_track': calibration.track},
            source=source_name(saved),
        )
        flush_trace()
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
    trace().add(
        "publish_saved", "dashboard_restored",
        in_count=1, out_count=1,
        reason='the sealed report bytes are copied onto the live dashboard pointer path',
        detail={'report_path': source_name(report_path), 'saved': source_name(saved)},
        source=source_name(saved),
    )


def _publish_track(destination, track, archived, publisher):
    """Publish one track's frozen report bytes after archive identity checks."""
    folder = destination / track / 'ablation'
    request, vectors = folder / bundle_spec().ablation_request_file, folder / 'vectors.npz'
    saved, binding = folder / 'report.json', folder / 'baseline_threshold.json'
    with _LOG.section('ablation.publish.track_identity'):
        _sealed_track(archived, destination, request, vectors, saved, binding)
        validated, document, calibration = _published_identity(track, request, saved, binding, vectors)
        trace().add(
            "publish_saved", "track_identity",
            scope=SCOPE_ENTITY, key=track,
            reason='every artifact matches the sealed archive bytes and the report/threshold '
                   'identities agree, so the frozen report is safe to republish',
            detail={'track': track, 'rows': len(validated['rows']),
                    'threshold': validated['threshold'],
                    'request': source_name(request), 'vectors': source_name(vectors),
                    'report': source_name(saved), 'binding': source_name(binding),
                    'calibration_checkpoint_sha256': calibration.checkpoint_sha256,
                    'publisher': publisher is not None},
            source=source_name(saved),
        )
    with _LOG.section('ablation.publish.restore'):
        _restored_dashboard(saved, document)
    if publisher is not None:
        publisher(request, vectors, json.loads(saved.read_text()), str(binding))


@timed
def _calibration_source(destination, track):
    """The one non-interrupted calibration manifest for a track, or nothing."""
    import json
    from graph_tracks.report_manifest import TrackReportManifest
    request = destination/track/'ablation'/bundle_spec().ablation_request_file
    result = request.parent/'vectors.npz'
    if not request.is_file() or not result.is_file():
        raise ValueError('suite lacks prepared GPU ablation export: '+track)
    sources = list((destination/track).rglob('text__completion_manifest.json' if track == 'text' else name(track,'report_manifest.json')))
    sources = [path for path in sources if not any(part.startswith('interrupted-') or '.interrupted-' in part for part in path.parts)]
    if len(sources) != 1:
        trace().add(
            "complete_saved", "calibration_source_rejected",
            scope=SCOPE_ENTITY, key=track,
            reason=('no calibration manifest survived for this track'
                    if not sources else
                    'more than one non-interrupted calibration manifest exists for this track; '
                    'the track is quarantined rather than guessed'),
            detail={'track': track, 'candidates': [source_name(path) for path in sources],
                    'request': source_name(request), 'result': source_name(result)},
            source=source_name(destination/track),
        )
        flush_trace()
        raise ValueError('ambiguous baseline calibration manifest: '+track)
    calibration = TrackReportManifest.model_validate_json(sources[0].read_text())
    if calibration.track != track:
        trace().add(
            "complete_saved", "calibration_track_rejected",
            scope=SCOPE_ENTITY, key=track,
            reason='the calibration manifest belongs to a different track; the track is '
                   'quarantined rather than cross-bound',
            detail={'track': track, 'manifest': source_name(sources[0]),
                    'manifest_track': calibration.track},
            source=source_name(sources[0]),
        )
        flush_trace()
        raise ValueError("ablation calibration belongs to a different track")
    trace().add(
        "complete_saved", "calibration_source",
        scope=SCOPE_ENTITY, key=track,
        reason='exactly one non-interrupted calibrated report manifest supplies the frozen threshold',
        detail={'track': track, 'manifest': source_name(sources[0]),
                'threshold': calibration.threshold,
                'threshold_source': calibration.threshold_source,
                'checkpoint_sha256': calibration.checkpoint_sha256,
                'request': source_name(request), 'result': source_name(result),
                'candidates': len(sources)},
        source=source_name(sources[0]),
    )
    _LOG.info(f'[ablation] calibration source track={track} manifest={sources[0].name}')
    return request, result, sources[0], calibration


@timed
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
            trace().add(
                "complete_saved", "checkpoint_mismatch",
                scope=SCOPE_ENTITY, key=track,
                reason='the baseline calibration was fit on a different checkpoint than the one '
                       'this ablation selected; the frozen threshold is not applied',
                detail={'track': track, 'selected_checkpoint': source_name(checkpoint),
                        'selected_sha256': selected_identity,
                        'calibrated_sha256': calibrated_identity,
                        'binding': source_name(binding)},
                source=source_name(checkpoint),
            )
            flush_trace()
            raise ValueError('baseline calibration differs from selected ablation checkpoint: '+track)
        trace().add(
            "complete_saved", "checkpoint_select",
            scope=SCOPE_ENTITY, key=track,
            reason='the selected checkpoint identity must equal the calibrated one before the '
                   'frozen threshold is bound to it',
            detail={'track': track, 'selected_checkpoint': source_name(checkpoint),
                    'selected_sha256': selected_identity,
                    'calibrated_sha256': calibrated_identity,
                    'threshold': threshold,
                    'source_calibration': source_name(source),
                    'binding': source_name(binding)},
            source=source_name(checkpoint),
        )
        write(binding,{'track':track,'checkpoint_sha256':selected_identity,
            'threshold':threshold,'calibration':{'threshold':threshold},
            'source_calibration':source_name(source),
            'source_calibration_sha256':__import__('graph_tracks.data',fromlist=['file_hash']).file_hash(source),
            'threshold_source':'saved dev calibration; no refit'})
    _LOG.info(f'[ablation] threshold binding track={track} threshold={threshold}')
    return binding, threshold, document


@timed
def _trusted_saved_report(result, threshold, binding, previous, validated, document):
    """True when a prior saved report bytes-identically covers this result.

    Callers then only re-verify the vectors behind the cached report instead
    of recomputing the threshold-frozen comparison.
    """
    from graph_tracks.data import file_hash
    from model_tracks.ablation import request_context, validate_vectors
    request = previous.parent/bundle_spec().ablation_request_file
    trusted = (validated and previous.with_suffix('.sha256').is_file()
               and previous.with_suffix('.sha256').read_text().strip() == file_hash(previous)
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
    _LOG.info(f'[ablation] complete_saved destination={destination}')
    for track in _LOG.progress(ABLATION_TRACKS, desc='complete_saved', unit='track'):
        _LOG.info(f'[ablation] complete track={track}')
        with _LOG.section('ablation.complete.calibration'):
            request, result, source, calibration = _calibration_source(destination, track)
            binding, threshold, document = _wrote_binding(request, track, calibration, source)
        with _LOG.section('ablation.complete.report'):
            validated = _saved_track_report(request, result, threshold, binding, document, suite)
        with _LOG.section('ablation.complete.persist'):
            _published_track_outputs(outputs, track, request, result, validated, binding, publisher)
    with _LOG.section('ablation.complete.receipt'):
        receipt = destination/'post_training_ablation.json'
        write(receipt,{'tracks':outputs,'retraining':False,'gpu_reopened':False})
        trace().add(
            "complete_saved", "receipt",
            in_count=len(ABLATION_TRACKS), out_count=len(outputs),
            reason='the consumer re-verifies the saved GPU exports; it never retrains and never '
                   'reopens the GPU',
            detail={'receipt': source_name(receipt), 'tracks': sorted(outputs),
                    'retraining': False, 'gpu_reopened': False},
            source=source_name(receipt),
        )
    flush_trace()
    return receipt


@timed
def _sealed_track_report(request, validated, config):
    """Persist the validated report exactly once per round when it changed.

    Byte-identity short-circuit: the round's documents must be serialized
    exactly once (not once per write site), and when the existing dashboard
    pointer already holds those exact bytes the rewrite is a legal no-op.
    Digests are compared against `json.dumps(sort_keys=True)` — the same
    canonical form `write()` persists.
    """
    from model_tracks.ablation import settings, write
    path = resolve(settings(config).report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = json.dumps(validated, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if not path.is_file() or path.read_text() != document:
        # Same encode/bytes as ablation.write(), materialized once through the
        # C-accelerated encoder instead of 56k generator write() calls. The
        # 1GB streaming safety belongs to ablation.write() and prepare()'s
        # request persistence; a sealed ablation report is ~0.5-1% of that.
        with path.open('w', encoding='utf-8') as handle:
            handle.write(document)
    saved = request.parent/'report.json'
    if saved != path and (not saved.is_file() or saved.read_text() != document):
        write(saved, validated)
    return path


def _saved_track_report(request, result, threshold, binding, document, suite):
    """Restore the cached report when trusted, otherwise recompute and seal it."""
    from graph_tracks.data import file_hash
    previous = request.parent/'report.json'
    validated = json.loads(previous.read_text()) if previous.exists() else None
    trusted, cached = _trusted_saved_report(result, threshold, binding, previous, validated, document)
    from model_tracks.ablation import save_report
    if trusted:
        validated = cached
    else:
        validated = report(request,result,threshold,threshold_source=str(binding),save=False,config=resolve(suite.ablation_config))
    _sealed_track_report(request,validated,resolve(suite.ablation_config))
    previous.with_suffix('.sha256').write_text(file_hash(previous)+'\n')
    trace().add(
        "complete_saved", "report",
        scope=SCOPE_ENTITY, key=request.parent.parent.name,
        reason=('the cached report already covers this exact request/result/threshold, so it was '
                're-verified instead of recomputed' if trusted else
                'no trusted cached report existed, so the comparison was recomputed'),
        detail={'track': request.parent.parent.name, 'trusted_cache': bool(trusted),
                'rows': len(validated.get('rows', [])),
                'threshold': validated.get('threshold'),
                'report': source_name(previous),
                'report_sha256': previous.with_suffix('.sha256').read_text().strip()},
        source=source_name(previous),
    )
    return validated


def _published_track_outputs(outputs, track, request, result, validated, binding, publisher):
    """Record one track's saved-artifact status and publish its git artifact."""
    outputs[track] = {'request':source_name(request),'status':'verified saved GPU result'}
    if publisher is not None:
        artifact = publisher(request,result,validated,str(binding))
        outputs[track]['artifact'] = source_name(artifact)
        _LOG.info(f'[ablation] publisher artifact track={track} name={outputs[track]["artifact"]}')
    trace().add(
        "complete_saved", "published",
        scope=SCOPE_ENTITY, key=track,
        reason=('the frozen report artifact was published through the git publisher'
                if publisher is not None else
                'no git publisher is configured; the report stays local'),
        detail={'track': track, 'status': outputs[track]['status'],
                'request': source_name(request),
                'artifact': outputs[track].get('artifact'),
                'publisher': publisher is not None},
        source=source_name(request),
    )


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
def run(archive, run_tag, suite, *, launcher=None, bundle=None):
    from core.bundle import Bundle, BundleRole
    archive_metadata = (Bundle.load(archive, BundleRole.result)
                        if bundle is None else bundle)
    destination = archive.parent/run_tag
    if any((destination/track/'ablation'/bundle_spec().ablation_request_file).exists() for track in ABLATION_TRACKS):
        return complete_saved(destination,suite,publisher=git_publisher(suite))
    raise ValueError('suite lacks staged GPU ablation exports; rebuild prepared inputs before training')
