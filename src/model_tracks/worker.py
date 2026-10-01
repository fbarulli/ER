"""Adapters for existing prepared text and graph trainers in a shared run."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml

from model_tracks.config import load_config
from model_tracks.parallel import wait_for_start


def run(config: Path, track: str, run_tag: str, *, resume: bool = False):
    from core.common import TRAIN_ROOT
    cfg = load_config(config)
    setup = (TRAIN_ROOT / cfg.setup_dir).resolve()
    output = Path(os.environ['EUROMONITOR_RESULTS_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    if track == 'text':
        from training.prepared_bundle import load_prepared_bundle
        manifest, _ = load_prepared_bundle((TRAIN_ROOT / cfg.text_bundle).resolve())
        command = [sys.executable, '-m', 'training.train_prepared', '--bundle', cfg.text_bundle,
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
        settings = yaml.safe_load((setup / f'{track}.yaml').read_text())
        settings.update(device=cfg.device, epochs=cfg.epochs, report_test=cfg.report_test,
                        postprocess=True)
        if cfg.dvc_enabled:
            # The suite publisher owns persistence; avoid a second mutable
            # local DVC snapshot while background uploads are active.
            settings['dvc'] = {**settings.get('dvc', {}), 'enabled': False}
        worker_config = output / 'worker.yaml'
        worker_config.write_text(yaml.safe_dump(settings, sort_keys=False))
        command = [sys.executable, '-m', 'graph_tracks.train', '--config', str(worker_config),
                   '--run-tag', run_tag]
        if resume:
            from model_tracks.resume import graph_checkpoint
            checkpoint = graph_checkpoint(output, track, run_tag)
            if checkpoint:
                command.extend(['--resume', str(checkpoint)])
            elif any(output.glob(f'{track}__run_manifest.json')):
                # A failure before the first checkpoint has no optimizer state.
                # Preserve its manifest as evidence and restart that track.
                previous = output / f'{track}__run_manifest.json'
                previous.replace(output / f'{track}__run_manifest.interrupted.json')
    wait_for_start(Path(os.environ['ER_TRACK_BARRIER']), track)
    subprocess.run(command, cwd=TRAIN_ROOT, env=os.environ.copy(), check=True)
    if track == 'text':
        from model_tracks.text_report import complete
        complete(output, setup, device=cfg.device, report_test=cfg.report_test)
        from model_tracks.incremental import ArtifactPublisher
        with ArtifactPublisher(output) as publisher:
            publisher.submit('postprocess', [p for p in output.iterdir()
                if p.name.startswith('text__') or p.name == 'profiles'])
    from model_tracks.resume import record_completion
    record_completion(output, track)


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
