"""Complete a result bundle on CPU and publish it — the finalize entry.

The operator-box finalize surface is retired: this module is a thin, lane-neutral
caller of :func:`model_tracks.bundle_steps.finalize` (the one place CPU
post-processing, ablation and sealing run). It exists so a lane or a local
operator can hand it the two verified archives plus a run tag and get back the
sealed result archive, and so publication has a single owner.

Two archives cross the wire and are each verified exactly once, at
:meth:`core.bundle.Bundle.load`:

* the result archive (GPU output: the selected checkpoint plus the reports
  the finalize step post-processes),
* the prepared inputs archive (the data bundle the suite trained from).

No later stage re-verifies them: the whole-archive digests come from that same
boundary pass, so the local completion identity costs no extra read. The
finalize step returns the WRITER's own handle for the archive it just sealed
(its digest was captured while writing and its manifest carries the member
inventory), so the just-sealed bytes are never re-opened to re-verify them.
"""
import json
from pathlib import Path
from model_tracks.package import package_member
from core.archive_reader import archive_sidecar

from core.run_log import RunLogger
from core.tracing import flush_stage_trace, stage_trace
from graph_tracks.data import file_hash
from model_tracks.config import SuiteConfig

log = RunLogger(__name__)

LOCAL_SOURCE_MARKER = 'local_source.json'

#: The stage name this module owns in the ONE consolidated pipeline trace.
STAGE = "local_complete"

#: The module's trace writer: the shared shim's slot (``None`` until first use;
#: see :func:`core.tracing.stage_trace`), so importing this module never touches
#: the trace layout. One completion job emits boundary, transport, finalize and
#: publication rows into one commit.
_TRACE = None


def trace():
    """The ONE writer for the ``local_complete`` stage of the current run."""
    global _TRACE
    _TRACE = stage_trace(STAGE, _TRACE)
    return _TRACE


def flush_trace():
    """Commit this process's local-completion rows once; a no-op while empty."""
    return flush_stage_trace(_TRACE)


def _spec():
    from core.bundle import _bundle_spec
    return _bundle_spec()


def _publish(final: Path, settings: SuiteConfig, run_tag: str, *, ablation_done: bool = False, destination: Path | None = None, bundle=None) -> Path:
    """Publish one sealed result archive; ``bundle`` is its verified boundary handle.

    Every downstream check receives the handle, so a single archive is verified
    once at its VM crossing instead of at each step of the publication chain.
    """
    from model_tracks.resume import validate_completed_suite_archive
    validate_completed_suite_archive(final, run_tag, settings=settings, bundle=bundle)
    trace().add(
        'publish', 'validated',
        in_count=1, out_count=1,
        reason='the sealed archive and its completion contract are validated once at this boundary',
        detail={'archive': str(final), 'run_tag': run_tag,
                'boundary_handle_supplied': bundle is not None,
                'ablation_done': bool(ablation_done),
                'dvc_enabled': bool(settings.dvc_enabled),
                'publish_git': bool(settings.publish_git),
                'post_training_ablation': bool(settings.post_training_ablation)},
        source=str(final),
    )
    if settings.post_training_ablation and not ablation_done:
        trace().add(
            'publish', 'ablation_missing',
            reason='post-training ablation is enabled but the caller reports it incomplete; '
                   'sealing is refused rather than publishing an archive without its reports',
            detail={'archive': str(final), 'run_tag': run_tag},
            source=str(final),
        )
        flush_trace()
        raise ValueError('complete saved ablation before sealing the publication archive')
    if settings.dvc_enabled:
        from model_tracks.publish import persist_results
        persist_results(final, run_tag, bundle=bundle)
        trace().add(
            'publish', 'results_persisted',
            reason='the sealed archive is persisted to its configured result sink',
            detail={'archive': str(final), 'run_tag': run_tag},
            source=str(final),
        )
    if settings.post_training_ablation:
        from model_tracks.resume import recorded_ablation_skip
        run_state = destination or final.parent / run_tag
        # A GPU suite that shipped no ablation templates recorded the deliberate
        # skip; there are then no exports to publish, and the sealed suite
        # legitimately carries no saved ablation.
        if recorded_ablation_skip(run_state):
            log.info('[publication] suite recorded no ablation export; nothing to publish')
            trace().add(
                'publish', 'ablation',
                reason='the suite recorded a deliberate ablation skip, so there are no exports to publish',
                detail={'run_state': str(run_state), 'recorded_skip': True},
                source=str(run_state / 'suite_events.jsonl'),
            )
        else:
            from model_tracks.post_training_ablation import publish_saved
            publish_saved(run_state, settings, archive=final, bundle=bundle)
            trace().add(
                'publish', 'ablation',
                reason='the frozen saved ablation reports are published from the sealed archive bytes',
                detail={'run_state': str(run_state), 'recorded_skip': False},
                source=str(run_state / 'post_training_ablation.json'),
            )
    if settings.publish_git:
        from model_tracks.publish import materialize
        materialize(final, run_tag, push=True, bundle=bundle)
        trace().add(
            'publish', 'git',
            reason='the sealed archive is materialized into the git publication tree and pushed',
            detail={'archive': str(final), 'run_tag': run_tag, 'pushed': True},
            source=str(final),
        )
    return final


def _require_legacy_source_pin(inputs, settings: SuiteConfig) -> None:
    """Legacy mode re-checks the live checkout against the packaged inventory.

    The suite's recorded runtime is already checked against the verified input
    package by ``validate_training_binding`` (self-consistent). Requiring the
    live checkout to still be byte-identical is a freshness gate that misfires
    once unrelated commits land after packaging; the frozen snapshot path runs
    the packaged runtime instead. Legacy mode keeps the strict checkout pin.
    """
    from core.common import TRAIN_ROOT
    from core.perf_switches import legacy_mode
    if not legacy_mode():
        return
    for relative, expected in inputs.manifest[_spec().files_key].items():
        if relative.startswith(('src/', 'config/', 'scripts/')) or relative == settings.ablation_config:
            if file_hash(TRAIN_ROOT / relative) != expected:
                trace().add(
                    'legacy_pin', 'code_changed',
                    scope=SCOPE_ENTITY, key=relative,
                    reason='legacy mode pins the live checkout to the packaged inventory; the '
                           'completion runtime differs from the one that trained',
                    detail={'relative': relative, 'train_root': str(TRAIN_ROOT)},
                    source=str(TRAIN_ROOT / relative),
                )
                flush_trace()
                raise ValueError(f'Local completion code/config differs from training: {relative}')


def _reuse(final: Path, destination: Path, input_archive: Path, inputs, identity: dict,
           run_tag: str, settings: SuiteConfig, publish: bool) -> Path:
    """A durable final archive is sufficient: restore what a caller may need."""
    from core.bundle import Bundle, BundleRole
    from model_tracks.bundle_steps import extract_prepared_inputs
    from model_tracks.resume import validate_completed_suite_archive

    existing = Bundle.load(final, BundleRole.result)
    validate_completed_suite_archive(final, run_tag, settings=settings, bundle=existing)
    for key, value in identity.items():
        if existing.manifest.get(key) != value:
            raise ValueError('existing completion archive has different inputs/checkpoints')
    if not destination.exists():
        destination.mkdir(parents=True)
        existing.materialize(destination)
        (destination / LOCAL_SOURCE_MARKER).write_text(json.dumps(identity))
    # The prepared inputs are inputs to the process, never deliverables: the
    # result member predicate drops them from any seal.
    extract_prepared_inputs(inputs, destination / _spec().prepared_inputs_dir,
                            package_member('suite_package_config'))
    trace().add(
        'complete', 'reused',
        in_count=1, out_count=1,
        reason='a durable final archive already exists and its identity matches this completion, '
               'so it is validated and republished instead of recomputed',
        detail={'final': str(final), 'destination': str(destination),
                'identity': {key: str(value) for key, value in identity.items()},
                'publish': bool(publish), 'digest': existing.digest},
        source=str(final),
    )
    published = _publish(final, settings, run_tag, ablation_done=True, bundle=existing) if publish else final
    flush_trace()
    return published


def complete(training_archive: Path, input_archive: Path, run_tag: str, *, publish: bool = True) -> Path:
    """Finalize one downloaded suite: verify once, finalize, publish."""
    from core.bundle import Bundle, BundlePipeline, BundleRole
    from model_tracks import bundle_steps
    from model_tracks.resume import validate_completed_suite_archive, validate_training_binding
    import yaml

    spec = _spec()
    training_archive, input_archive = Path(training_archive), Path(input_archive)
    # One integrity check per VM crossing: the two archives are verified here
    # and every later step reads the trusted handle.
    result = Bundle.load(training_archive, BundleRole.result)
    inputs = Bundle.load(input_archive, BundleRole.inputs)
    settings = SuiteConfig.model_validate(
        yaml.safe_load(inputs.read(package_member('suite_package_config'))))
    if result.run_tag() != run_tag:
        raise ValueError('local completion run mismatch')
    validate_training_binding(result.read_json(spec.suite_manifest_file),
                              inputs.manifest, settings, run_tag)
    _require_legacy_source_pin(inputs, settings)
    trace().add(
        'complete', 'boundaries',
        in_count=2, out_count=2,
        reason='each archive is integrity-checked exactly once at its Bundle boundary; every '
               'later step reads the trusted handle',
        detail={'run_tag': run_tag,
                'training_archive': str(training_archive), 'training_digest': result.digest,
                'training_members': len(result.members()),
                'input_archive': str(input_archive), 'input_digest': inputs.digest,
                'input_members': len(inputs.members()),
                'post_training_ablation': bool(settings.post_training_ablation),
                'report_test': bool(settings.report_test),
                'publish': bool(publish)},
        source=str(input_archive),
    )
    destination = training_archive.parent / run_tag
    final = training_archive.parent / f'{run_tag}.{settings.result_archive_format}'
    identity = {'training_archive_sha256': result.digest,
                'input_archive_sha256': inputs.digest}
    if final.exists():
        return _reuse(final, destination, input_archive, inputs, identity,
                      run_tag, settings, publish)
    marker = destination / LOCAL_SOURCE_MARKER
    if destination.exists():
        if not marker.is_file() or json.loads(marker.read_text()) != identity:
            raise ValueError('existing local completion belongs to different inputs')
    else:
        destination.mkdir(parents=True)
        marker.write_text(json.dumps(identity))
    # Transport timings were created after the immutable training archive was
    # sealed; preserve the receipt-carried sidecars beside the run state.
    import shutil
    copied, absent = [], []
    for suffix in ('.profile.json', '.dvc_profile.jsonl'):
        sidecar = archive_sidecar(training_archive, suffix)
        if sidecar.is_file():
            metrics = destination / 'resource_profile'
            metrics.mkdir(exist_ok=True)
            shutil.copy2(sidecar, metrics / ('remote_training' + suffix))
            copied.append('remote_training' + suffix)
        else:
            absent.append(suffix)
    trace().add(
        'complete', 'transport_timings',
        in_count=len(copied) + len(absent), out_count=len(copied),
        reason='the transport receipts were produced after the immutable training archive was '
               'sealed, so they ride beside the run state instead of inside the archive',
        detail={'copied': copied, 'absent': absent,
                'destination': str(destination / 'resource_profile')},
        source=str(training_archive),
    )
    trace().add_entities(
        'complete.transport_timing', copied,
        key_of=lambda name: name,
        reason_of=lambda name: 'receipt_copied_beside_run_state',
        detail_of=lambda name: {'artifact': name, 'destination': str(destination / 'resource_profile')},
        source=str(training_archive),
    )
    pipeline = BundlePipeline(
        role=BundleRole.result, device='cpu', lane='local', inputs=input_archive,
        output=final, work_dir=destination,
        prepared_dir=destination / spec.prepared_inputs_dir,
        postprocess_location=spec.postprocess_location_local, metadata=identity)
    handle = bundle_steps.finalize(pipeline, result, inputs=inputs)
    trace().add(
        'complete', 'finalized',
        in_count=1, out_count=1,
        reason='the ONE finalize step ran: materialize, extracted inputs, per-track postprocess, '
               'ablation and seal (its own stage rows carry the detail)',
        detail={'final': str(final), 'exists': final.is_file(),
                'bytes': final.stat().st_size if final.is_file() else 0,
                'digest': handle.digest,
                'work_dir': str(destination),
                'prepared_dir': str(destination / spec.prepared_inputs_dir),
                'postprocess_location': spec.postprocess_location_local},
        source=str(final),
    )
    # The finalize step returned the WRITER's handle for the archive it just
    # sealed: its digest was captured while writing and its manifest already
    # mirrors the archive's (member inventory included), so the completion
    # contract runs against those trusted bytes instead of a second
    # Bundle.load of the archive in this same process.
    archive_sidecar(final, spec.sha256_sidecar_suffix).write_text(handle.digest + '\n')
    validate_completed_suite_archive(final, run_tag, settings=settings, bundle=handle)
    trace().add(
        'complete', 'validated',
        in_count=len(handle.members()), out_count=len(handle.members()),
        reason='the sealed archive is the writer\'s own verified handle: its digest and manifest '
               'inventory came from the sealing pass, so every later step shares it instead of '
               're-opening the bytes',
        detail={'final': str(final), 'digest': handle.digest,
                'members': len(handle.members()), 'run_tag': run_tag,
                'sha256_sidecar': str(archive_sidecar(final, spec.sha256_sidecar_suffix))},
        source=str(final),
    )
    published = _publish(final, settings, run_tag, ablation_done=True, bundle=handle) if publish else final
    trace().add(
        'complete', 'published',
        in_count=1, out_count=1 if publish else 0,
        reason=('the finalized archive was published to its configured sinks'
                if publish else
                'publish=False was requested, so the finalized archive is returned unpublished'),
        detail={'final': str(final), 'run_tag': run_tag, 'publish': bool(publish),
                'digest': handle.digest},
        source=str(final),
    )
    flush_trace()
    return published
