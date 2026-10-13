"""One track's post-training attribute-ablation export, run as its own process.

The export is memory-bound (the frozen template stays resident while the
selected checkpoint's encoder streams vectors through it), so the suite
supervisor spawns this lane once per trained track ONLY after the parallel
training pool has drained — one lane at a time, each in a fresh process so its
RSS returns to the OS before the next track starts (the 1013 OOM class: the gnn
worker encoded while the text worker still trained and the kernel killed it).

The lane consumes only on-disk artifacts: the trainer-selected checkpoint, the
frozen template shipped in the prepared setup, and the inference export the
training worker saved. It produces the same bound request + encoded vectors the
in-pool worker produced before, minus the in-process model handoff —
:func:`model_tracks.ablation.encode` loads the encoder from the checkpoint when
one is not supplied, so nothing downstream changes.
"""
from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path

from core.bundle import bundle_spec
from core.results import Results
from core.run_log import RunLogger
from core.tracing import SCOPE_ENTITY, flush_stage_trace, stage_trace
from model_tracks.config import load_config
from model_tracks.resume import TRAINING_TRACKS

_LOG = RunLogger(__name__)

#: The stage name this module owns in the ONE consolidated pipeline trace.
STAGE = "ablation_lane"

#: The module's trace writer: the shared shim's slot (``None`` until first use;
#: see :func:`core.tracing.stage_trace`), so importing this module never touches
#: the trace layout.
_TRACE = None


def trace():
    """The ONE writer for the ``ablation_lane`` stage of the current run."""
    global _TRACE
    _TRACE = stage_trace(STAGE, _TRACE)
    return _TRACE


def flush_trace():
    """Commit this process's ablation-lane rows once; a no-op while empty."""
    return flush_stage_trace(_TRACE)


def selected_checkpoint(output: Path, track: str) -> Path:
    """The track's trainer-selected checkpoint, read from its own results root."""
    if track == 'text':
        from training.validation_inference import resolve_best_checkpoint
        checkpoint, _ = resolve_best_checkpoint(output)
        return checkpoint
    from core.bundle import Bundle, BundleRole
    checkpoint = Bundle.from_directory(output, BundleRole.result).checkpoint(track)
    if checkpoint is None or not checkpoint.is_file():
        raise ValueError('selected graph checkpoint unavailable')
    return checkpoint


def _export(events, output: Path, setup: Path, track: str, cfg, *, gpu_only: bool) -> None:
    """Bind the track's frozen template to its selected checkpoint and encode vectors."""
    spec = bundle_spec()
    template = setup / spec.ablation_templates_dir / track / spec.ablation_request_file
    if gpu_only and not template.is_file():
        # A GPU session relies on the CPU data bundle shipping the templates;
        # a bundle built before that contract would FileNotFoundError here,
        # so skip with a named reason. Local sessions keep the loud read.
        events.emit('attribute_ablation_export', 'skipped', device=cfg.device,
                    reason='bundle shipped no ablation templates')
        trace().add(
            "attribute_ablation_export", "skipped",
            scope=SCOPE_ENTITY, key=track,
            reason='bundle_shipped_no_ablation_templates',
            detail={'track': track, 'device': cfg.device, 'template': str(template),
                    'gpu_only': gpu_only},
            source=str(setup / spec.ablation_templates_dir),
        )
        return
    checkpoint = selected_checkpoint(output, track)
    events.emit('attribute_ablation_export', 'started', device=cfg.device)
    from model_tracks.staged_ablation import forward as forward_ablation
    with _LOG.section('phase.ablation_export', track=track, device=cfg.device):
        forward_ablation(output, setup, track, checkpoint, device=cfg.device)
    events.emit('attribute_ablation_export', 'completed', device=cfg.device)
    request_path = Results.for_root(output.parent).request(track)
    trace().add(
        "attribute_ablation_export", "completed",
        scope=SCOPE_ENTITY, key=track, in_count=1, out_count=1,
        reason='the frozen template is bound to the selected checkpoint and its '
               'vectors are encoded for the saved ablation',
        detail={'track': track, 'device': cfg.device,
                'checkpoint': str(checkpoint),
                'request': str(request_path)},
        source=str(request_path),
    )


def run(config: Path, track: str, run_tag: str, *, resume: bool = False):
    from model_tracks.telemetry import WorkerEvents
    if track not in TRAINING_TRACKS:
        raise ValueError(f'{track} ships no attribute-ablation export')
    output = Path(os.environ['EUROMONITOR_RESULTS_DIR'])
    events = WorkerEvents(output, track, run_tag)
    events.emit('input_validation', 'started', config=str(config), resume_requested=resume)
    try:
        _run(config, track, run_tag, resume=resume, events=events)
    except BaseException as exc:
        events.emit('failure', 'failed', error_type=type(exc).__name__,
                    error=str(exc), failed_phase=events.last_phase, traceback=traceback.format_exc())
        trace().add(
            "failure", "aborted",
            scope=SCOPE_ENTITY, key=track,
            reason=f'the ablation lane aborted with {type(exc).__name__}; the rows above show '
                   'the last step that ran for this track',
            detail={'track': track, 'error_type': type(exc).__name__, 'error': str(exc),
                    'failed_phase': events.last_phase},
            source=str(config),
        )
        flush_trace()
        raise
    flush_trace()


def _run(config: Path, track: str, run_tag: str, *, resume: bool, events) -> None:
    from core.common import TRAIN_ROOT
    cfg = load_config(config)
    output = Path(os.environ['EUROMONITOR_RESULTS_DIR'])
    setup = (TRAIN_ROOT / cfg.setup_dir).resolve()
    gpu_only = os.environ.get('ER_GPU_TRAINING_ONLY') == '1'
    with _LOG.section('phase.ablation_lane', track=track, run_tag=run_tag,
                      gpu_only=gpu_only):
        _export(events, output, setup, track, cfg, gpu_only=gpu_only)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--track', required=True, choices=TRAINING_TRACKS)
    parser.add_argument('--run-tag', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.config, args.track, args.run_tag, resume=args.resume)


if __name__ == '__main__':
    main()
