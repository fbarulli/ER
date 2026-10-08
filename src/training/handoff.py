"""One offline consumer boundary: verify every training input, then stop.

The preparation run owns its validated objects; this module is the consumer
boundary inside the same process.  It touches exactly what the text trainer
and the graph workers consume before an optimizer step -- the bundle is
unpickled once per run (this boundary reuses the run's cached object) -- and
persists ``handoff.json``: the per-input load meter, the loss/batch
correctness attestation for the frozen objective, and the final artifact
inventory.  Training after this boundary is just training: the trainer
verifies an attestation instead of re-running this check stack.

No CUDA work is claimed here: GPU-only embeddings stay an explicit
prerequisite recorded in the report.  A CPU validation run proves the data
contract, not device execution.  Every declared file name comes from the
SSOT layout in ``training.preparation`` (config/training.yaml) via
``training_cfg()``; nothing in this module hardcodes a producer path.
Every check is a named, timed surface (core.timing): the sections land in
the stage timing JSON and in the run's consolidated offender report.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from core.run_log import RunLogger
from core.timing import Timing
from core.tracing import SCOPE_ENTITY, TraceRun
from training.prepare_all_trace import timed

_LOG = RunLogger(__name__)

#: The stage name prepare_all runs this module as, and the name this stage's
#: consolidated-trace rows carry (core.tracing ``stage`` column).
STAGE = 'verify_handoff'

#: The boundary's checks, in execution order. A check missing from a
#: ``HandoffTrace``'s passed list when the boundary raises is the one that
#: raised, so a failure names itself instead of guessing.
CHECK_ORDER: tuple[str, ...] = (
    'provenance', 'bundle_load', 'graph_manifest',
    'worker_settings', 'loss_batch', 'package_verify', 'inventory',
    'manifest_verify',
)

#: ``reason`` cells are NOT capped by core.tracing (only ``detail`` is), so the
#: boundary bounds its own: a raise message can be arbitrarily long.
REASON_CHARS = 240


def _bounded_reason(text: object, limit: int = REASON_CHARS) -> str:
    """One bounded single-line reason string (the trace caps ``detail`` only)."""
    flat = ' '.join(str(text).split())
    if len(flat) <= limit:
        return flat
    return f'{flat[:limit]}…<elided {len(flat) - limit} chars>'


class HandoffLoad(BaseModel):
    """One input the handoff boundary read or re-validated, with its cost."""

    model_config = ConfigDict(extra="forbid")

    input: str
    path: str
    loads: int = Field(default=1, ge=1)
    bytes: int = Field(default=0, ge=0)
    seconds: float = Field(default=0.0, ge=0.0)
    size: int | None = None


class LossBatchAttestation(BaseModel):
    """The frozen objective's batch-correctness contract for the loss.

    The loss functions require exact batch composition: every epoch's frozen
    sampler must cover every objective row exactly once per device, under the
    batch sizes the worker will run and the loss the worker will train with.
    """

    model_config = ConfigDict(extra="forbid")

    loss: str
    epochs: int = Field(ge=1)
    batch_sizes: dict[str, int]
    folds: int = Field(ge=1)
    plan_identity: dict[str, Any]
    coverage: Literal["every objective row exactly once per epoch"]
    validated_by: list[str]


class HandoffReport(BaseModel):
    """Persisted evidence (run_dir/handoff.json) that inputs are ready."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["pass"]
    gpu_embeddings: str
    checks: dict[str, Any]
    inputs: list[HandoffLoad]
    loss_batch_correctness: LossBatchAttestation | None = None
    final_inventory: dict[str, dict[str, str | int]]
    bundle_header: dict[str, Any]
    suite_package: dict[str, Any]
    total_seconds: float = Field(ge=0.0)


class HandoffTrace:
    """Stage ``verify_handoff`` in the ONE consolidated trace (core.tracing).

    The boundary verified every training input and persisted ``handoff.json``,
    but the ONE trace held nothing about it: "which input was loaded through
    which consumer path, and which check attested what" lived only in the
    report, and a FAILING boundary left no trace row at all. Emitted here:

      run   check.<name>      one row per boundary check, with its readback
                              (reason ``passed``; a check that raises is
                              recorded by :meth:`failure` instead)
      run   loads.metered     every input the boundary read: loads, bytes,
                              seconds, and how many entries are content-pinned
      ent   loads.input       one row per metered input, NAMED with its path,
                              size, load count and size (the entries ARE the
                              declared input set, so they are enumerated in
                              full — no sampling applies)
      run   report.handoff    the persisted report's own census (status, the
                              checks that ran, the inventory, the attestation)
      run   boundary.failed   the check that raised, the checks that had
                              already passed and the error, when the boundary
                              cannot pass (the error still propagates)

    Rows are committed once, at the end: on success a write failure fails the
    stage loudly, while :meth:`failure` must never mask the boundary's own
    error and therefore only warns when the trace cannot be written.
    """

    def __init__(self, trace: TraceRun | None = None) -> None:
        self.trace = trace if trace is not None else TraceRun(STAGE)
        self.checks: list[str] = []

    def check(self, name: str, *, detail: object = None, source: str = '') -> None:
        """Record one boundary check that PASSED, with its readback."""
        self.checks.append(str(name))
        self.trace.add(
            'check', name, reason='passed',
            detail={'check': str(name), 'verdict': 'passed', 'readback': detail},
            source=source,
        )

    def _loads(self, meter: "_LoadMeter") -> None:
        """The load meter's totals, then one NAMED row per metered input."""
        entries = list(meter.entries)
        self.trace.add(
            'loads', 'metered',
            reason=(
                'every declared input the boundary read or re-validated '
                'through its consumer path; each entry is enumerated below'
            ),
            detail={
                'inputs': len(entries),
                'loads': sum(int(entry.loads) for entry in entries),
                'bytes': sum(int(entry.bytes) for entry in entries),
                'seconds': round(sum(float(entry.seconds) for entry in entries), 6),
                'size_pinned': sum(1 for entry in entries if entry.size),
            },
            source="handoff.json inputs[] (the boundary's own load meter)",
        )
        for entry in entries:
            self.trace.add(
                'loads', 'input', scope=SCOPE_ENTITY, key=entry.input,
                reason=f'loaded {int(entry.loads)}x through its consumer path',
                detail={
                    'path': entry.path, 'loads': int(entry.loads),
                    'bytes': int(entry.bytes), 'seconds': float(entry.seconds),
                    'size': entry.size or '',
                },
                source="handoff.json inputs[] (the boundary's own load meter)",
            )

    def report(self, report: HandoffReport, meter: "_LoadMeter") -> None:
        """Record the persisted report's census and its loads, then commit."""
        self._loads(meter)
        self.trace.add(
            'report', 'handoff',
            reason=(
                'the boundary passed: every check above attested, every '
                'declared input loaded, and handoff.json is the persisted '
                'evidence the trainer verifies instead of re-running this stack'
            ),
            detail={
                'status': report.status,
                'checks': sorted(report.checks),
                'loads': len(report.inputs),
                'load_seconds': round(
                    sum(float(entry.seconds) for entry in report.inputs), 6),
                'inventory_artifacts': len(report.final_inventory),
                'bundle_size': report.bundle_header.get('size'),
                'suite_package_size': report.suite_package.get('size'),
                'loss_batch_attested': report.loss_batch_correctness is not None,
                'gpu_embeddings': report.gpu_embeddings,
                'total_seconds': report.total_seconds,
            },
            source="handoff.json (the boundary's own report)",
        )
        self.trace.write()

    def failure(self, failure: BaseException, meter: "_LoadMeter") -> None:
        """Record the failing check, commit, and let the error propagate.

        The commit is guarded on purpose: a secondary trace failure must never
        replace the boundary's own error (the thing the operator acts on).
        """
        pending = [name for name in CHECK_ORDER if name not in self.checks]
        self._loads(meter)
        self.trace.add(
            'boundary', 'failed',
            reason=_bounded_reason(f'{type(failure).__name__}: {failure}'),
            detail={
                'failed_check': pending[0] if pending else '',
                'error_type': type(failure).__name__,
                'checks_passed': list(self.checks),
                'inputs_metered': len(meter.entries),
            },
            source='training.handoff.verify_training_loads',
        )
        try:
            self.trace.write()
        except Exception as error:  # pragma: no cover - never mask the boundary
            _LOG.warning(
                f'[trace] verify_handoff rows could not be committed: {error}')


class _LoadMeter:
    """Per-input load accounting for one handoff verification."""

    def __init__(self) -> None:
        self.entries: list[HandoffLoad] = []

    def record(self, name: str, path: Path, *, seconds: float = 0.0,
               loads: int = 1, size: int | None = None) -> None:
        self.entries.append(HandoffLoad(
            input=name, path=str(path), loads=loads,
            bytes=path.stat().st_size if path.is_file() else 0,
            seconds=round(seconds, 6), size=size,
        ))

    def timed(self, name: str, path: Path, fn, *, size: int | None = None):
        started = time.monotonic()
        try:
            return fn()
        finally:
            self.record(name, path, seconds=time.monotonic() - started, size=size)


def _check_provenance(root, suite_config_path, checkpoint, provenance) -> dict:
    """Record the run's preparation identity for the boundary report.

    No comparison and no freshness verdict (owner directive 2026-10-08): the
    identity is read back so the boundary enumerates what it was built from.
    """
    from training.prepare_all import preparation_provenance
    return preparation_provenance(Path(root), Path(suite_config_path), Path(checkpoint))


def _load_bundle_verified(full_bundle, meter: _LoadMeter):
    """The one verified bundle load of the run (cache-hit after full_bundle)."""
    from training.prepared_bundle import load_prepared_bundle
    bundle_path = Path(full_bundle)
    sidecar = bundle_path.with_suffix(bundle_path.suffix + '.json')
    started = time.monotonic()
    header, prepared = load_prepared_bundle(bundle_path, verify_inputs=True)
    meter.record('text_bundle', bundle_path, seconds=time.monotonic() - started,
                 size=getattr(header, 'size', None))
    meter.record('bundle_header', sidecar)
    return header, prepared, sidecar


def _read_graph_manifest(setup_dir, layout, meter: _LoadMeter) -> dict:
    """Load the graph setup manifest the worker settings are read from.

    No size is re-derived against the run tree (owner directive 2026-10-08):
    the manifest is the graph setup's own record, read as declared.
    """
    manifest_path = Path(setup_dir) / layout.manifest
    return meter.timed('graph_setup_manifest', manifest_path,
                       lambda: json.loads(manifest_path.read_text()))


def _check_worker_settings(setup_dir, layout, suite, meter: _LoadMeter) -> dict:
    """The exact worker settings the suite will execute (config-level)."""
    from model_tracks.worker import graph_worker_settings
    from tqdm import tqdm
    summary: dict[str, Any] = {}
    for track in tqdm(('gnn_only', 'cascade'), total=2, desc='worker_settings',
                      unit='track', leave=False, disable=False,
                      dynamic_ncols=True):
        track_config = Path(setup_dir) / layout.track_config(track)
        if track_config.is_file():
            settings = graph_worker_settings(Path(setup_dir), suite, track)
            meter.record(layout.track_config(track), track_config)
            summary[track] = {'device': settings['device'],
                              'epochs': settings['epochs'],
                              'listings': settings.get('listings'),
                              'pairs': settings.get('pairs'),
                              'input_manifest': settings.get('input_manifest')}
    return {'validated_tracks': sorted(summary), **summary}


def _validate_frozen_plan_identity(plan, loss) -> dict:
    """The frozen plan is version 1, keyed to this loss, with healthy folds."""
    identity = dict(plan.get('identity') or {})
    if plan.get('version') != 1 or not identity:
        raise ValueError('prepared objective lacks a valid frozen plan identity')
    if identity.get('loss') != loss:
        raise ValueError('frozen objective loss differs from the configured loss; rebuild locally')
    if plan['inputs'].get('skipped') or not plan['inputs'].get('folds'):
        raise ValueError('prepared training row plan has failed/skipped folds')
    return identity


def _resolve_batch_sizes(plan, smoke: bool) -> dict[str, int]:
    """The worker's batch sizes (runtime config), or the smoke-saved ones."""
    from core.common import runtime
    batch_sizes = {device: int(runtime('batch_size_' + device))
                   for device in ('cpu', 'cuda')}
    if smoke:
        # Lifecycle smokes retain their saved batch settings, like the worker.
        saved = plan['inputs']['folds'][0]['objective']['sampler']
        batch_sizes = {device: saved[device]['batch_size'] for device in batch_sizes}
    return batch_sizes


def _validated_by_note(plan_identity_revalidate: bool) -> list[str]:
    """Who attested the frozen plan: preflight always, boundary when standalone."""
    validated_by = ['model_tracks.preflight (suite_inputs, same run and process)',
                    'training.train_prepared (trainer start, per training run)']
    if plan_identity_revalidate:
        validated_by.append('training.handoff (data size recomputed at this boundary)')
    return validated_by


def _maybe_recompute_plan_size(prepared, plan, loss, smoke: bool,
                                 plan_identity_revalidate: bool,
                                 validated_by: list[str]) -> None:
    """The plan size stays preflight-attested unless this boundary runs standalone."""
    if not plan_identity_revalidate:
        return
    from core.common import SEED
    from training.run_plan import validate_run_plan
    validate_run_plan(prepared, plan, loss=loss, train_frac=1., sample=smoke, seed=SEED)
    validated_by.append('training.handoff (data size recomputed at this boundary)')


def _maybe_check_tokens(timing: Timing, prepared) -> dict[str, Any]:
    """Validate the native token table when present; surface its policy."""
    tokens = prepared.get('training_tokens')
    if tokens is None:
        return {}
    from training.token_inputs import validate_training_tokens
    validate_training_tokens(tokens)
    timing.mark('tokens')
    return {'policy': tokens.get('policy'), 'payload_size': tokens.get('payload_size')}


def _attest_loss_batch(prepared, graph: dict, suite, *,
                       plan_identity_revalidate: bool = False):
    """Loss/batch correctness of the frozen objective (the trainer's contract).

    The data-size identity was enforced by the suite preflight during
    suite_inputs of this same run and process; here the structural contract
    is attested over the cached plan, and the size is only recomputed when
    this boundary runs standalone.
    """
    from core.common import training_cfg
    plan = prepared.get('training_plan')
    smoke = bool(graph.get('smoke', False))
    if plan is None:
        return None, {'plan': 'absent; the suite preflight requires the frozen '
                              'plan during packaging, so a real build cannot reach '
                              'this boundary without it'}
    from training.run_plan import validate_epoch_batches
    loss = training_cfg().training.loss
    identity = _validate_frozen_plan_identity(plan, loss)
    validated_by = _validated_by_note(plan_identity_revalidate)
    _maybe_recompute_plan_size(prepared, plan, loss, smoke,
                                 plan_identity_revalidate, validated_by)
    batch_sizes = _resolve_batch_sizes(plan, smoke)
    timing = Timing('training.handoff.loss_batch')
    timing.mark('plan_identity')
    validate_epoch_batches(plan, epochs=suite.epochs, batch_sizes=batch_sizes)
    timing.mark('epoch_batches')
    token_checks = _maybe_check_tokens(timing, prepared)
    attestation = LossBatchAttestation(
        loss=loss, epochs=suite.epochs, batch_sizes=batch_sizes,
        folds=len(plan['inputs']['folds']), plan_identity=identity,
        coverage='every objective row exactly once per epoch',
        validated_by=validated_by)
    return attestation, token_checks


def _verify_package(suite_archive, meter: _LoadMeter) -> tuple[dict, dict]:
    """The transport artifact the consumer extracts, verified member by member."""
    from model_tracks.package import verify
    from training.prepare_all import size
    archive_path = Path(suite_archive)
    package = meter.timed('suite_package', archive_path, lambda: verify(archive_path))
    suite_package = {'path': str(suite_archive), 'size': size(archive_path),
                     'preflight': package.get('preflight', {})}
    return suite_package, {'preflight_report': bool(package.get('preflight'))}


def _final_inventory(reusable_paths, full_bundle, text_bundle,
                     suite_archive) -> dict:
    """Final inventory of every declared artifact (the resume contract)."""
    from training.prepare_all import file_inventory
    bundle_path = Path(full_bundle)
    text_bundle_path = Path(text_bundle)
    return file_inventory([
        *reusable_paths,
        bundle_path, bundle_path.with_suffix(bundle_path.suffix + '.json'),
        text_bundle_path, text_bundle_path.with_suffix(text_bundle_path.suffix + '.json'),
        Path(suite_archive),
    ])


def _manifest_directory(root: str | Path) -> Path:
    """The audit manifest directory, resolved under the run root.

    Stages publish their per-stage manifests into ``audit.manifest_dir``
    (relative to the repo root). This boundary runs inside the same
    preparation, so it re-checks the SAME directory; resolving it under
    ``root`` rather than ``core.common._path`` keeps the check hermetic
    under test roots that override the repo layout.
    """
    from core.common import training_cfg
    manifest_dir = Path(training_cfg().audit.manifest_dir)
    return manifest_dir if manifest_dir.is_absolute() else Path(root) / manifest_dir


def _verify_published_manifests(manifest_dir: str | Path) -> dict[str, list[str]]:
    """Re-verify every published stage manifest against the files on disk.

    Walks the audit registry (``audit.manifest_stages``) and re-checks each
    stage whose manifest this run actually published through
    ``core.manifest.verify_manifest`` (outputs present and hash-matching,
    expected outputs produced, row accounting closed, no ``.tmp-*`` residue).
    A stage whose manifest is absent is left alone: at this boundary a missing
    marker means the stage never published one (later lanes such as
    ``evaluate_models``/``zero_shot_sims`` run after the handoff), not that a
    PUBLISHED marker drifted — only that drift fails loud here.
    """
    from core.common import training_cfg
    from core.manifest import verify_manifest
    verified: list[str] = []
    for stage in training_cfg().audit.manifest_stages:
        if (Path(manifest_dir) / f'{stage}.json').exists():
            verify_manifest(stage, manifest_dir=manifest_dir)
            verified.append(stage)
    return {'verified_stages': verified}


def verify_training_loads(*, root, suite, suite_config_path, checkpoint,
                          setup_dir, full_bundle, text_bundle, suite_archive,
                          provenance, smoke_dir, smoke_original, reusable_paths,
                          plan_identity_revalidate: bool = False) -> HandoffReport:
    """Load every declared training input through its consumer path.

    Producer-side checks (provenance, frozen CSV agreement, graph source
    hashes) run first; consumer-side checks (worker settings, the frozen
    objective's batch contract, the package archive) follow.  The bundle is
    loaded with ``verify_inputs=True`` exactly once -- after the first full
    load the run's cache serves it, so the payload is unpickled once per run.

    Every check, every metered load and the report itself also land in the ONE
    consolidated trace (core.tracing, stage ``verify_handoff``): a passing
    boundary enumerates exactly what it checked, and a FAILING one commits the
    check that raised plus the checks that had already passed, then re-raises.
    """
    trace = HandoffTrace()
    meter = _LoadMeter()
    try:
        report = _verify_loads(
            trace, meter, root=root, suite=suite,
            suite_config_path=suite_config_path, checkpoint=checkpoint,
            setup_dir=setup_dir, full_bundle=full_bundle, text_bundle=text_bundle,
            suite_archive=suite_archive, provenance=provenance,
            smoke_dir=smoke_dir, smoke_original=smoke_original,
            reusable_paths=reusable_paths,
            plan_identity_revalidate=plan_identity_revalidate,
        )
    except BaseException as failure:
        trace.failure(failure, meter)
        raise
    trace.report(report, meter)
    return report


def _verify_loads(trace: HandoffTrace, meter: "_LoadMeter", *, root, suite,
                  suite_config_path, checkpoint, setup_dir, full_bundle,
                  text_bundle, suite_archive, provenance, smoke_dir,
                  smoke_original, reusable_paths,
                  plan_identity_revalidate: bool = False) -> HandoffReport:
    """The boundary's checks, each recorded as it passes (see the caller)."""
    started = time.monotonic()
    timing = Timing('training.handoff')
    from core.common import training_cfg
    layout = training_cfg().preparation.graph_setup

    with timing.section('provenance'):
        current = _check_provenance(root, suite_config_path, checkpoint, provenance)
    trace.check(
        'provenance',
        detail={'readback': 'preparation identity read back for the record',
                'text_checkpoint': current.get('text_checkpoint'),
                'provenance_keys': sorted(current)},
        source='training.prepare_all.preparation_provenance')
    with timing.section('bundle_load'):
        header, prepared, sidecar = _load_bundle_verified(full_bundle, meter)
    trace.check(
        'bundle_load',
        detail={'payload_variant': getattr(header, 'payload_variant', None),
                'masking_profile': getattr(header, 'masking_profile', None),
                'size': getattr(header, 'size', None),
                'verify_inputs': True, 'bundle_members': len(prepared)},
        source='training.prepared_bundle.load_prepared_bundle(verify_inputs=True)')
    with timing.section('graph_manifest'):
        graph = _read_graph_manifest(setup_dir, layout, meter)
    trace.check(
        'graph_manifest',
        detail={key: graph.get(key) for key in (
            'source_catalog_size', 'labeled_pairs_size',
            'text_checkpoint_size', 'smoke')},
        source=f'{layout.manifest} read as the graph setup\'s declared record')
    with timing.section('worker_settings'):
        settings_summary = _check_worker_settings(setup_dir, layout, suite, meter)
    trace.check(
        'worker_settings', detail=settings_summary,
        source='model_tracks.worker.graph_worker_settings (per track config)')
    with timing.section('loss_batch'):
        attestation, token_checks = _attest_loss_batch(
            prepared, graph, suite, plan_identity_revalidate=plan_identity_revalidate)
    trace.check(
        'loss_batch',
        detail={
            'plan': 'absent' if attestation is None else 'attested',
            'loss_batch_correctness': (
                None if attestation is None else attestation.model_dump(mode='json')),
            'training_tokens': token_checks or None,
        },
        source='training.run_plan.validate_epoch_batches over the frozen plan')
    with timing.section('package_verify'):
        suite_package, package_checks = _verify_package(suite_archive, meter)
    trace.check(
        'package_verify',
        detail={'size': suite_package.get('size'), **package_checks},
        source='model_tracks.package.verify (the sealed inputs bundle)')
    with timing.section('inventory'):
        final_inventory = _final_inventory(reusable_paths, full_bundle,
                                           text_bundle, suite_archive)
    trace.check(
        'inventory',
        detail={'artifacts': len(final_inventory)},
        source='training.prepare_all.file_inventory (the resume contract)')
    with timing.section('manifest_verify'):
        manifest_summary = _verify_published_manifests(_manifest_directory(root))
    trace.check(
        'manifest_verify',
        detail={'verified_stages': manifest_summary['verified_stages']},
        source=('core.manifest.verify_manifest re-checked over the published '
                'audit.manifest_stages markers'))

    checks: dict[str, Any] = {
        'provenance': 'recorded (src/scripts/config/raw inputs/checkpoint read back)',
        'text_bundle': {'payload_variant': getattr(header, 'payload_variant', None),
                        'masking_profile': getattr(header, 'masking_profile', None),
                        'verify_inputs': True},
        'graph_setup': {'source_catalog_size': graph.get('source_catalog_size'),
                        'labeled_pairs_size': graph.get('labeled_pairs_size'),
                        'smoke': bool(graph.get('smoke', False))},
        'graph_worker_settings': settings_summary,
        'suite_package': package_checks,
        'manifest_verify': manifest_summary['verified_stages'],
    }
    if attestation is None:
        checks['loss_batch_correctness'] = token_checks
    elif token_checks:
        checks['training_tokens'] = token_checks
    return HandoffReport(
        status='pass',
        gpu_embeddings='GPU pending: frozen baseline forward before graph training',
        checks=checks, inputs=meter.entries, loss_batch_correctness=attestation,
        final_inventory=final_inventory, bundle_header=header.model_dump(mode='json'),
        suite_package=suite_package,
        total_seconds=round(time.monotonic() - started, 3),
    )


def write_handoff_report(report: HandoffReport, path: Path) -> None:
    """Persist the report atomically; handoff.json is the boundary's record."""
    from core.manifest import atomic_write_json
    payload = report.model_dump(mode='json')
    payload['finished_at'] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(payload, Path(path))
