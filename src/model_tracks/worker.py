"""Adapters for existing prepared text and graph trainers in a shared run."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import shlex
import yaml

from core.run_log import RunLogger
from model_tracks.config import load_config
from model_tracks.parallel import wait_for_start

_LOG = RunLogger(__name__)


def graph_worker_settings(setup: Path, cfg, track: str, *, gpu_only: bool = False) -> dict:
    """The exact graph configuration this worker trains under.

    The pre-training data gate builds its graph inputs through this function so
    the validated configuration cannot drift from the executed one.
    """
    from graph_tracks.config import GraphConfig, load_config as load_graph_config
    settings = load_graph_config(setup / f'{track}.yaml', expected_track=track).model_dump()
    settings.update(device=cfg.device, epochs=cfg.epochs, report_test=cfg.report_test,
                    postprocess=not gpu_only, include_inputs=False)
    if cfg.dvc_enabled or gpu_only:
        # The suite publisher owns persistence; avoid a second mutable
        # local DVC snapshot while background uploads are active.
        settings['dvc'] = {**settings.get('dvc', {}), 'enabled': False}
    settings.update(cfg.graph_execution_overrides())
    return GraphConfig.model_validate(settings).model_dump()


def run(config: Path, track: str, run_tag: str, *, resume: bool = False):
    from model_tracks.telemetry import WorkerEvents
    output = Path(os.environ['EUROMONITOR_RESULTS_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    events = WorkerEvents(output, track, run_tag)
    events.emit('input_validation', 'started', config=str(config), resume_requested=resume)
    try:
        _run(config, track, run_tag, resume=resume, events=events)
    except BaseException as exc:
        events.emit('failure', 'failed', error_type=type(exc).__name__,
                    error=str(exc), failed_phase=events.last_phase, traceback=traceback.format_exc())
        raise


def _run(config: Path, track: str, run_tag: str, *, resume: bool, events):
    from core.common import TRAIN_ROOT
    cfg = load_config(config)
    gpu_only = os.environ.get('ER_GPU_TRAINING_ONLY') == '1'
    setup = (TRAIN_ROOT / cfg.setup_dir).resolve()
    output = Path(os.environ['EUROMONITOR_RESULTS_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    if track == 'text':
        from training.prepared_bundle import PreparedBundleManifest
        bundle_path = (TRAIN_ROOT / cfg.text_bundle).resolve()
        # The trainer owns full payload loading/validation after the barrier.
        # This adapter only needs the typed header to construct its command.
        manifest = PreparedBundleManifest.model_validate_json(
            bundle_path.with_suffix(bundle_path.suffix + '.json').read_text())
        events.emit('input_validation', 'configured', device=cfg.device,
                    report_test=cfg.report_test, bundle=str(cfg.text_bundle),
                    payload=manifest.payload_variant)
        command = [sys.executable, '-m', 'training.train_prepared', '--bundle', cfg.text_bundle,
                   '--shared-training-data', str(setup / 'shared_training_data.json'),
                   '--training-binding', str(setup / 'text_training_binding.json'),
                   '--model', cfg.text_model, '--epochs', str(cfg.epochs),
                   '--payload', manifest.payload_variant, '--run-tag', run_tag,
                   '--device', cfg.device,
                   '--report-test' if cfg.report_test else '--no-report-test']
        if resume and any((output / '_checkpoints').rglob('trainer_state.json')):
            command.append('--resume')
        setup_manifest = json.loads((setup / 'setup_manifest.json').read_text())
        if setup_manifest.get('smoke'):
            command.extend(['--sample', str(setup_manifest['source_listing_count'])])
    else:
        from graph_tracks.config import GraphConfig
        settings = graph_worker_settings(setup, cfg, track, gpu_only=gpu_only)
        worker_config = output / 'worker.yaml'
        worker_config.write_text(yaml.safe_dump(settings, sort_keys=False))
        events.emit('input_validation', 'configured', device=cfg.device,
                    report_test=cfg.report_test, worker_config=str(worker_config),
                    validation_owner='graph trainer load_inputs before training')
        command = [sys.executable, '-m', 'graph_tracks.train', '--config', str(worker_config),
                   '--run-tag', run_tag]
        if resume:
            from model_tracks.resume import graph_checkpoint
            checkpoint = graph_checkpoint(output, track, run_tag)
            if checkpoint:
                command.extend(['--resume', str(checkpoint)])
            else:
                # A failure before the first checkpoint has no optimizer state.
                # Preserve its manifest as evidence and restart that track.
                for folder in (output, output / f'{track}__{run_tag}'):
                    previous = folder / f'{track}__run_manifest.json'
                    if previous.exists():
                        previous.replace(previous.with_name(f'{track}__run_manifest.interrupted.json'))
    events.emit('command', 'prepared', command=shlex.join(command),
                resume_requested=resume, checkpoint_resume='--resume' in command,
                checkpoint=(command[command.index('--resume') + 1]
                            if track != 'text' and '--resume' in command else None),
                sample=(int(command[command.index('--sample') + 1]) if '--sample' in command else None))
    events.emit('barrier', 'waiting', barrier=os.environ['ER_TRACK_BARRIER'])
    wait_for_start(Path(os.environ['ER_TRACK_BARRIER']), track)
    events.emit('barrier', 'released')
    events.emit('training', 'started', includes_graph_postprocess=track != 'text')
    with _LOG.section('phase.training', track=track):
        subprocess.run(command, cwd=TRAIN_ROOT, env=os.environ.copy(), check=True)
    events.emit('training', 'completed', includes_graph_postprocess=track != 'text')
    if gpu_only or track == 'text' or cfg.post_training_ablation:
        with _LOG.section('phase.inference_export', track=track):
            events.emit('inference_export','started',device=cfg.device)
            if track == 'text':
                from model_tracks.text_export import forward
                _,selected_text_model = forward(output,setup,return_model=True,device=cfg.device)
            else:
                from graph_tracks.config import GraphConfig
                from graph_tracks.infer import forward_outputs
                from graph_tracks.artifacts import name
                selected = list(output.rglob(name(track,'best_checkpoint.json')))
                if len(selected) != 1:
                    raise ValueError('ambiguous selected graph checkpoint')
                checkpoint = Path(json.loads(selected[0].read_text())['path'])
                settings['device'] = cfg.device
                settings.update(cfg.graph_execution_overrides())
                _,selected_graph_encoder = forward_outputs(checkpoint,TRAIN_ROOT/settings['listings'],TRAIN_ROOT/settings['pairs'],
                    output/(track+'__inference'),GraphConfig.model_validate(settings),
                    text_cache=TRAIN_ROOT/settings['text_cache'] if settings.get('text_cache') else None,
                    return_encoder=True)
            events.emit('inference_export','completed',device=cfg.device)
            if cfg.post_training_ablation:
                from model_tracks.staged_ablation import forward as forward_ablation
                if track == 'text':
                    from training.validation_inference import resolve_best_checkpoint
                    checkpoint,_ = resolve_best_checkpoint(output)
                if gpu_only and not (setup/'ablation_templates'/track/'request.json').is_file():
                    # A GPU session relies on the CPU data bundle shipping the
                    # templates; a bundle built before that contract would
                    # FileNotFoundError here, so skip with a named reason.
                    # Local sessions keep the loud read.
                    events.emit('attribute_ablation_export','skipped',device=cfg.device,
                                reason='bundle shipped no ablation templates')
                else:
                    events.emit('attribute_ablation_export','started',device=cfg.device)
                    forward_ablation(output,setup,track,checkpoint,text_model=selected_text_model if track == 'text' else None,device=cfg.device,
                        graph_encoder=selected_graph_encoder if track != 'text' else None)
                    events.emit('attribute_ablation_export','completed',device=cfg.device)
            if track == 'text':
                del selected_text_model
            else:
                del selected_graph_encoder
    if track == 'text' and not gpu_only:
        with _LOG.section('phase.postprocess', track=track, report_test=cfg.report_test):
            from model_tracks.text_report import complete
            events.emit('postprocess', 'started', report_test=cfg.report_test)
            complete(output, setup, device=cfg.device, report_test=cfg.report_test)
            events.emit('postprocess', 'completed')
        from model_tracks.incremental import ArtifactPublisher
        with _LOG.section('phase.incremental_publish', track=track):
            with ArtifactPublisher(output) as publisher:
                if publisher.enabled:
                    events.emit('publication', 'started', publication_owner='ArtifactPublisher')
                    publisher.submit('postprocess', [p for p in output.iterdir()
                        if p.name.startswith('text__') or p.name == 'profiles'])
                    events.emit('publication', 'submitted')
                else:
                    events.emit('publication', 'skipped', reason='incremental publication disabled')
            if publisher.enabled:
                events.emit('publication', 'context_closed',
                            detail='publisher context finished; remote receipts are in publication metadata')
    from model_tracks.resume import record_completion
    with _LOG.section('phase.completion', track=track):
        record_completion(output, track, postprocess_complete=not gpu_only)
        events.emit('completion', 'verified', inventory='track_inventory.json',
                    marker='track_complete.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--track', choices=['text', 'gnn_only', 'hybrid'], required=True)
    parser.add_argument('--run-tag', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.config, args.track, args.run_tag, resume=args.resume)


if __name__ == '__main__':
    main()
