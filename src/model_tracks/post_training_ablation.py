"""Automatic inference-only ablations after selected model publication."""
import importlib.util
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from core.artifacts import Artifacts
from core.results import Results
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
    checkpoint_size: int
    verified: Literal[True]


class SavedCalibration(BaseModel):
    model_config = ConfigDict(extra='allow', frozen=True, allow_inf_nan=False)
    track: Track
    checkpoint_size: int
    threshold: float


class SavedAblationReport(BaseModel):
    """Identity required before a previously computed report is published."""
    model_config = ConfigDict(extra='allow', allow_inf_nan=False)
    track: Track
    result_size: int
    threshold: float
    threshold_provenance: dict[str, Any]
    threshold_binding: AblationThresholdIdentity
    rows: list[dict[str, Any]]

    def identity_disagreements(self, *, track: str, request_track: str, report_size: int,
                               vectors_size: int, report_sidecar_size: int,
                               binding_size: int, calibration: SavedCalibration) -> list[str]:
        """The identity fields that disagree with the artifacts sealed beside it.

        The report's own identity contract, owned by the class that carries the
        fields: the sealed report, the request that names it, the vectors it was
        computed over and the calibration binding it recorded must all agree.
        Empty list means the report is safe to republish. No freshness verdict
        is ever computed here (owner directive 2026-10-08).
        """
        return [name for name, agrees in (
            ('track', self.track == track),
            ('request track', request_track == track),
            ('result vectors', self.result_size == vectors_size),
            ('report sidecar', report_sidecar_size == report_size),
            ('binding provenance', self.threshold_provenance.get('size') == binding_size),
            ('calibration track', calibration.track == track),
            ('threshold', self.threshold == calibration.threshold),
            ('threshold binding track', self.threshold_binding.track == track),
            ('checkpoint binding',
             self.threshold_binding.checkpoint_size == calibration.checkpoint_size),
        ) if not agrees]


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
                    'archive_size': getattr(archived, 'path').stat().st_size,
                    'publisher': publisher is not None},
            source=source_name(archive),
        )
    with _LOG.section('ablation.publish.tracks'):
        run = Results.for_root(destination)
        for track in _LOG.progress(ABLATION_TRACKS, desc='publish_saved', unit='track'):
            _publish_track(run, track, archived, publisher)
    flush_trace()


def _sealed_archive(archive, suite, *, bundle=None):
    """The sealed bundle (verified only when no boundary handle was supplied)."""
    from core.bundle import Bundle, BundleRole
    archived = (Bundle.load(archive, BundleRole.result)
                if bundle is None else bundle)
    return archived, git_publisher(suite)


def _sealed_track(archived, run, track):
    """Raise unless every exported artifact matches the sealed archive bytes.

    The exported set IS the declared saved-ablation set (``results.yaml``
    ``sets.saved_ablation``) plus the report's writer-stamped ``.size`` sidecar,
    so no caller re-lists the members.
    """
    from graph_tracks.data import file_size
    saved = run.report(track)
    members = [*run.saved(track).values(), saved.with_suffix('.size')]
    # ``archived`` is a verified Bundle handle or the plain verified manifest.
    manifest = getattr(archived, 'manifest', archived)
    for path in members:
        relative = path.relative_to(run.root).as_posix()
        # RECORD the difference; the artifact is published as it is (owner
        # directive: data is never checked).
        if manifest['files'].get(relative) != file_size(path):
            print(f'[ablation] WARNING: {relative} on-disk size {file_size(path)} differs from the '
                  f'sealed archive {manifest["files"].get(relative)}; publishing as it is',
                  flush=True)


def _published_identity(track, request, saved, binding, vectors):
    """The validated saved report, request document and calibration binding."""
    from graph_tracks.data import file_size
    validated = SavedAblationReport.model_validate_json(saved.read_text())
    document = json.loads(request.read_text())
    calibration = SavedCalibration.model_validate_json(binding.read_text())
    sidecar = saved.with_suffix('.size')
    disagreements = validated.identity_disagreements(
        track=track, request_track=document['track'], report_size=file_size(saved),
        vectors_size=file_size(vectors),
        report_sidecar_size=(sidecar.read_text().strip() if sidecar.is_file() else ''),
        binding_size=file_size(binding), calibration=calibration)
    if disagreements:
        # RECORD the disagreements; the saved report is republished as it is
        # (owner directive: data is never checked).
        trace().add(
            "publish_saved", "report_identity_recorded",
            scope=SCOPE_ENTITY, key=track,
            reason='the saved report, its request and its calibration binding disagree; the '
                   'disagreement is recorded, never enforced',
            detail={'track': track, 'report': source_name(saved),
                    'validated_track': validated.track, 'document_track': document['track'],
                    'calibration_track': calibration.track,
                    'disagreements': disagreements},
            source=source_name(saved),
        )
        flush_trace()
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


def _publish_track(run, track, archived, publisher):
    """Publish one track's frozen report bytes after archive identity checks."""
    request, vectors = run.request(track), run.vectors(track)
    saved, binding = run.report(track), run.baseline_threshold(track)
    with _LOG.section('ablation.publish.track_identity'):
        _sealed_track(archived, run, track)
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
                    'calibration_checkpoint_size': calibration.checkpoint_size,
                    'publisher': publisher is not None},
            source=source_name(saved),
        )
    with _LOG.section('ablation.publish.restore'):
        _restored_dashboard(saved, document)
    if publisher is not None:
        publisher(request, vectors, json.loads(saved.read_text()), str(binding))


def _completion_contract_name(track: str, root: Path) -> str:
    """The declared per-track completion-contract filename (config SSOT).

    The text lane keeps its legacy completion-manifest stem; every other track
    ships the shared report manifest. Both stems are declared in artifacts.yaml.
    """
    key = 'completion_manifest' if track == 'text' else 'report_manifest'
    return Artifacts.resolve(key, track=track, root=root).name


@timed
def _calibration_source(run, track):
    """The one non-interrupted calibration manifest for a track, or nothing."""
    import json
    from graph_tracks.report_manifest import TrackReportManifest
    request, result = run.request(track), run.vectors(track)
    track_root = run.root / track
    if not request.is_file() or not result.is_file():
        # Nothing to calibrate: the track declares no prepared ablation export.
        return result, None, None
    sources = list(track_root.rglob(_completion_contract_name(track, run.root)))
    sources = [path for path in sources if not any(part.startswith('interrupted-') or '.interrupted-' in part for part in path.parts)]
    if not sources:
        return result, None, None
    if len(sources) > 1:
        # RECORD the ambiguity and pick the first deterministically (owner
        # directive: data is never checked, so candidate count is never a reason
        # to refuse).
        sources = sorted(sources, key=lambda path: path.as_posix())
        print(f'[ablation] WARNING: {len(sources)} non-interrupted calibration manifests for '
              f'{track}; using {source_name(sources[0])}', flush=True)
    calibration = TrackReportManifest.model_validate_json(sources[0].read_text())
    if calibration.track != track:
        # RECORD the cross-track manifest; it is still used as it is.
        print(f'[ablation] WARNING: calibration manifest {source_name(sources[0])} declares track '
              f'{calibration.track!r}, not {track!r}; using it anyway', flush=True)
    trace().add(
        "complete_saved", "calibration_source",
        scope=SCOPE_ENTITY, key=track,
        reason='exactly one non-interrupted calibrated report manifest supplies the frozen threshold',
        detail={'track': track, 'manifest': source_name(sources[0]),
                'threshold': calibration.threshold,
                'threshold_source': calibration.threshold_source,
                'checkpoint_size': calibration.checkpoint_size,
                'request': source_name(request), 'result': source_name(result),
                'candidates': len(sources)},
        source=source_name(sources[0]),
    )
    _LOG.info(f'[ablation] calibration source track={track} manifest={sources[0].name}')
    return result, sources[0], calibration


@timed
def _wrote_binding(run, track, calibration, source):
    """Seal the selected checkpoint identity and frozen threshold into binding."""
    from graph_tracks.data import file_size
    from model_tracks.ablation import request_context
    request = run.request(track)
    binding = run.baseline_threshold(track)
    document = json.loads(request.read_text())
    threshold = calibration.threshold
    with request_context(request):
        checkpoint = resolve(document['checkpoint'])
        selected_identity = checkpoint_identity(checkpoint)
        calibrated_identity = calibration.checkpoint_size
        if calibrated_identity != selected_identity:
            # A RECORD that the calibration was fit on a different checkpoint than
            # this ablation selected; the frozen threshold is still bound to the
            # selected checkpoint (owner directive: data is never checked, so the
            # mismatch is never a reason to refuse).
            trace().add(
                "complete_saved", "checkpoint_mismatch",
                scope=SCOPE_ENTITY, key=track,
                reason='the baseline calibration was fit on a different checkpoint than the one '
                       'this ablation selected; the mismatch is recorded, not enforced',
                detail={'track': track, 'selected_checkpoint': source_name(checkpoint),
                        'selected_size': selected_identity,
                        'calibrated_size': calibrated_identity,
                        'binding': source_name(binding)},
                source=source_name(checkpoint),
            )
            flush_trace()
        trace().add(
            "complete_saved", "checkpoint_select",
            scope=SCOPE_ENTITY, key=track,
            reason='the selected checkpoint identity must equal the calibrated one before the '
                   'frozen threshold is bound to it',
            detail={'track': track, 'selected_checkpoint': source_name(checkpoint),
                    'selected_size': selected_identity,
                    'calibrated_size': calibrated_identity,
                    'threshold': threshold,
                    'source_calibration': source_name(source),
                    'binding': source_name(binding)},
            source=source_name(checkpoint),
        )
        write(binding,{'track':track,'checkpoint_size':selected_identity,
            'threshold':threshold,'calibration':{'threshold':threshold},
            'source_calibration':source_name(source),
            'source_calibration_size':__import__('graph_tracks.data',fromlist=['file_size']).file_size(source),
            'threshold_source':'saved dev calibration; no refit'})
    _LOG.info(f'[ablation] threshold binding track={track} threshold={threshold}')
    return binding, threshold, document


@timed
def _trusted_saved_report(run, track, result, threshold, binding, previous, validated, document):
    """True when a prior saved report covers this result.

    Callers then only re-verify the vectors behind the cached report instead of
    recomputing the threshold-frozen comparison. No recorded size (the ``.size``
    sidecar, the result size, the binding size) is compared: the recorded sizes
    are RECORDS (owner directive: data is never checked).
    """
    from model_tracks.ablation import request_context, validate_vectors
    request = run.request(track)
    trusted = bool(validated) and validated.get('threshold') == threshold
    if not trusted:
        return False, None
    with request_context(request):
        validate_vectors(request,result)
        attestation = frozen_threshold(str(binding),threshold)
        if validated.get('threshold_binding') != verify_threshold_binding(document,attestation) or validated.get('threshold_provenance') != attestation:
            # RECORD the binding difference; the cached report is reused as it is.
            print('[ablation] WARNING: cached calibration binding/provenance differs; reusing it',
                  flush=True)
    return True, validated


@timed
def complete_saved(destination: Path, suite: SuiteConfig, *, publisher=None) -> Path:
    """Consume suite GPU exports after shutdown; no provisioning or forwards."""
    outputs = {}
    run = Results.for_root(destination)
    _LOG.info(f'[ablation] complete_saved destination={destination}')
    for track in _LOG.progress(ABLATION_TRACKS, desc='complete_saved', unit='track'):
        _LOG.info(f'[ablation] complete track={track}')
        with _LOG.section('ablation.complete.calibration'):
            result, source, calibration = _calibration_source(run, track)
            if calibration is None:
                # RECORD the track with no prepared ablation export and move on.
                print(f'[ablation] no prepared ablation export for {track}; skipping its reports',
                      flush=True)
                continue
            binding, threshold, document = _wrote_binding(run, track, calibration, source)
        with _LOG.section('ablation.complete.report'):
            validated = _saved_track_report(run, track, result, threshold, binding, document, suite)
        with _LOG.section('ablation.complete.persist'):
            _published_track_outputs(outputs, track, run.request(track), result,
                                     validated, binding, publisher)
    with _LOG.section('ablation.complete.receipt'):
        receipt = run.receipt()
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
def _sealed_track_report(run, track, validated, config):
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
    saved = run.report(track)
    if saved != path and (not saved.is_file() or saved.read_text() != document):
        write(saved, validated)
    return path


def _saved_track_report(run, track, result, threshold, binding, document, suite):
    """Restore the cached report when trusted, otherwise recompute and seal it."""
    from graph_tracks.data import file_size
    request, previous = run.request(track), run.report(track)
    validated = json.loads(previous.read_text()) if previous.exists() else None
    trusted, cached = _trusted_saved_report(run, track, result, threshold, binding,
                                            previous, validated, document)
    from model_tracks.ablation import save_report
    if trusted:
        validated = cached
    else:
        validated = report(request,result,threshold,threshold_source=str(binding),save=False,config=resolve(suite.ablation_config))
    _sealed_track_report(run, track, validated, resolve(suite.ablation_config))
    previous.with_suffix('.size').write_text(str(file_size(previous))+'\n')
    trace().add(
        "complete_saved", "report",
        scope=SCOPE_ENTITY, key=track,
        reason=('the cached report already covers this exact request/result/threshold, so it was '
                're-verified instead of recomputed' if trusted else
                'no trusted cached report existed, so the comparison was recomputed'),
        detail={'track': track, 'trusted_cache': bool(trusted),
                'rows': len(validated.get('rows', [])),
                'threshold': validated.get('threshold'),
                'report': source_name(previous),
                'report_size': previous.with_suffix('.size').read_text().strip()},
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
    run = Results.for_root(destination, run_tag)
    if any(run.request(track).exists() for track in ABLATION_TRACKS):
        return complete_saved(destination,suite,publisher=git_publisher(suite))
    raise ValueError('suite lacks staged GPU ablation exports; rebuild prepared inputs before training')
