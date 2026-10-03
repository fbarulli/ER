"""Complete downloaded training checkpoints on local CPU, then publish."""
import json
from pathlib import Path
import zipfile

from core.portable_archive import verify_archive, write_archive
from graph_tracks.data import file_hash

def _publish(final, settings, run_tag):
    if settings.dvc_enabled:
        from model_tracks.publish import persist_results
        persist_results(final, run_tag)
    if settings.publish_git:
        from model_tracks.publish import materialize
        materialize(final, run_tag, push=True)
    return final


def complete(training_archive: Path, input_archive: Path, run_tag: str) -> Path:
    from core.common import TRAIN_ROOT
    from model_tracks.resume import TRACKS, completed_track, record_completion
    from model_tracks.config import SuiteConfig
    import yaml

    training = verify_archive(training_archive, 'suite_bundle_manifest.json')
    inputs = verify_archive(input_archive, 'model_tracks_package.json')
    with zipfile.ZipFile(input_archive) as archive:
        settings = SuiteConfig.model_validate(yaml.safe_load(archive.read('data/model_tracks/suite.yaml')))
    if training['run_tag'] != run_tag:
        raise ValueError('local completion run mismatch')
    # The local report implementation must match the code that produced training.
    for relative, expected in inputs['files'].items():
        if relative.startswith(('src/', 'config/', 'scripts/')):
            if file_hash(TRAIN_ROOT / relative) != expected:
                raise ValueError(f'Local completion code/config differs from training: {relative}')
    destination = training_archive.parent / run_tag
    final = training_archive.parent / f'{run_tag}.zip'
    if final.exists():
        existing = verify_archive(final, 'suite_bundle_manifest.json')
        if existing.get('training_archive_sha256') != file_hash(training_archive) or existing.get('input_archive_sha256') != file_hash(input_archive):
            raise ValueError('existing completion archive has different inputs/checkpoints')
        return _publish(final, settings, run_tag)
    marker = destination / 'local_source.json'
    identity = {'training_archive_sha256': file_hash(training_archive),
                'input_archive_sha256': file_hash(input_archive)}
    if destination.exists():
        if not marker.is_file() or json.loads(marker.read_text()) != identity:
            raise ValueError('existing local completion belongs to different inputs')
    else:
        destination.mkdir()
        with zipfile.ZipFile(training_archive) as archive:
            # Extract only SHA256-inventoried members, never unlisted extras.
            for relative in training['files']:
                archive.extract(relative, destination)
        marker.write_text(json.dumps(identity))
    prepared = destination / 'local_inputs'
    prepared.mkdir(exist_ok=True)
    with zipfile.ZipFile(input_archive) as archive:
        for relative in inputs['files']:
            if relative.startswith('data/model_tracks/'):
                target = prepared / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                data = archive.read(relative)
                if target.exists() and target.read_bytes() != data:
                    raise ValueError(f'local prepared input changed: {relative}')
                target.write_bytes(data)
        settings = SuiteConfig.model_validate(yaml.safe_load(archive.read('data/model_tracks/suite.yaml')))
    setup = prepared / settings.setup_dir
    suite = json.loads((destination / 'suite_manifest.json').read_text())
    if suite.get('inputs') != inputs['preflight']:
        raise ValueError('training suite did not use the verified prepared inputs')
    for track in TRACKS:
        output = destination / track
        track_marker = json.loads((output / 'track_complete.json').read_text())
        if track_marker.get('postprocess_complete'):
            completed_track(output, track)
            continue
        completed_track(output, track, postprocess_complete=False)
        print(f'[local-postprocess/{track}] starting on CPU', flush=True)
        if track == 'text':
            from model_tracks.text_report import complete as text_complete
            # Preserve interrupted reports rather than mixing attempts:
            # rename every text__* artifact from a prior attempt (a fixed
            # allowlist would silently miss a future text__* output and
            # mix it into the new attempt). Checkpoints live under
            # _checkpoints/** and are not matched by this glob.
            import time
            for path in sorted(output.glob('text__*')):
                path.rename(path.with_name(f'interrupted-{time.time_ns()}-{path.name}'))
            text_complete(output, setup, device='cpu', report_test=settings.report_test)
        else:
            from graph_tracks.config import GraphConfig
            from graph_tracks.preflight import preflight
            from graph_tracks.report import complete as graph_complete
            config = yaml.safe_load((setup / f'{track}.yaml').read_text())
            for key in ('listings', 'pairs', 'input_manifest', 'text_cache'):
                if config.get(key):
                    config[key] = str(prepared / config[key])
            config.update(device='cpu', report_test=settings.report_test, postprocess=True)
            config_path = output / 'local_report.yaml'
            config_path.write_text(yaml.safe_dump(config))
            preflight(config_path, check_device=False)
            selected = list(output.rglob(f'{track}__best_checkpoint.json'))
            if len(selected) != 1:
                raise ValueError(f'ambiguous selected checkpoint: {track}')
            recorded = Path(json.loads(selected[0].read_text())['path'])
            checkpoints = list(output.rglob(f'{recorded.parent.name}/{recorded.name}'))
            if len(checkpoints) != 1:
                raise ValueError(f'selected checkpoint unavailable: {track}')
            import time
            report = output / f'{track}__local_completion'
            if report.exists():
                report.rename(report.with_name(f'{report.name}.interrupted-{time.time_ns()}'))
            report.mkdir()
            graph_complete(checkpoints[0], Path(config['listings']), Path(config['pairs']),
                           report, GraphConfig.model_validate(config),
                           text_cache=Path(config['text_cache']) if config.get('text_cache') else None)
        record_completion(output, track)
        print(f'[local-postprocess/{track}] complete', flush=True)
    suite['postprocess_location'] = 'local CPU'
    (destination / 'suite_manifest.json').write_text(json.dumps(suite, indent=2))
    files = {p.relative_to(destination).as_posix(): p for p in destination.rglob('*')
             if p.is_file() and not p.is_symlink()
             and not {'local_inputs', 'wandb', 'mlruns', '.git', '.dvc'}.intersection(p.relative_to(destination).parts)}
    write_archive(final, files, manifest_name='suite_bundle_manifest.json',
                  metadata={'run_tag': run_tag, **identity, 'postprocess_location': 'local CPU'})
    final.with_suffix('.sha256').write_text(file_hash(final) + '\n')
    return _publish(final, settings, run_tag)
