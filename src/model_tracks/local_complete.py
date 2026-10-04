"""Complete downloaded training checkpoints on local CPU, then publish."""
import json
from pathlib import Path
import zipfile

from core.portable_archive import verify_archive, write_archive, RESULT_ARCHIVE_EXCLUDED_DIRS
from graph_tracks.data import file_hash
from model_tracks.config import SuiteConfig

def _publish(final: Path, settings: SuiteConfig, run_tag: str, *, ablation_done: bool = False, destination: Path | None = None) -> Path:
    from model_tracks.resume import validate_completed_suite_archive
    validate_completed_suite_archive(final, run_tag, settings=settings)
    if settings.post_training_ablation and not ablation_done:
        raise ValueError('complete saved ablation before sealing the publication archive')
    if settings.dvc_enabled:
        from model_tracks.publish import persist_results
        persist_results(final, run_tag)
    if settings.post_training_ablation:
        from model_tracks.post_training_ablation import publish_saved
        publish_saved(destination or final.parent / run_tag, settings, archive=final)
    if settings.publish_git:
        from model_tracks.publish import materialize
        materialize(final, run_tag, push=True)
    return final


def complete(training_archive: Path, input_archive: Path, run_tag: str, *, publish: bool = True) -> Path:
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
    from model_tracks.resume import validate_training_binding
    with zipfile.ZipFile(training_archive) as archive:
        validate_training_binding(json.loads(archive.read('suite_manifest.json')),
                                  inputs, settings, run_tag)
    # The local report implementation must match the code that produced training.
    for relative, expected in inputs['files'].items():
        if relative.startswith(('src/', 'config/', 'scripts/')) or relative == settings.ablation_config:
            if file_hash(TRAIN_ROOT / relative) != expected:
                raise ValueError(f'Local completion code/config differs from training: {relative}')
    destination = training_archive.parent / run_tag
    final = training_archive.parent / f'{run_tag}.zip'
    if final.exists():
        from model_tracks.resume import validate_completed_suite_archive
        existing = validate_completed_suite_archive(final, run_tag, settings=settings)
        if existing.get('training_archive_sha256') != file_hash(training_archive) or existing.get('input_archive_sha256') != file_hash(input_archive):
            raise ValueError('existing completion archive has different inputs/checkpoints')
        # A durable final archive is sufficient to reconstruct reports; prepared
        # inputs come from the verified original input archive, never a cache.
        if not destination.exists():
            destination.mkdir()
            with zipfile.ZipFile(final) as archive:
                for relative in existing['files']:
                    archive.extract(relative,destination)
            (destination/'local_source.json').write_text(json.dumps({
                'training_archive_sha256':file_hash(training_archive),
                'input_archive_sha256':file_hash(input_archive)}))
        restored_inputs = destination/'local_inputs'
        with zipfile.ZipFile(input_archive) as archive:
            for relative in inputs['files']:
                if relative.startswith('data/model_tracks/'):
                    target = restored_inputs/relative
                    target.parent.mkdir(parents=True,exist_ok=True)
                    data = archive.read(relative)
                    if target.exists() and target.read_bytes() != data:
                        raise ValueError('restored prepared input changed: '+relative)
                    target.write_bytes(data)
        return _publish(final, settings, run_tag, ablation_done=True) if publish else final
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
    baseline = destination/'baseline/shared_minilm__embeddings.npz'
    if baseline.is_file():
        from training.prepare_embeddings import validate_result
        request_path = setup/'embedding_inputs.json'
        request = json.loads(request_path.read_text())
        validate_result(baseline,request,request_sha256=file_hash(request_path))
        import shutil
        cache = setup/'shared_minilm__embeddings.npz'
        if cache.exists() and file_hash(cache) != file_hash(baseline):
            raise ValueError('restored frozen baseline differs from suite GPU export')
        if not cache.exists():
            shutil.copy2(baseline,cache)
    if settings.post_training_ablation and (destination/'baseline/ablation/request.json').is_file():
        from model_tracks.baseline_ablation import complete as complete_baseline_ablation
        complete_baseline_ablation(destination/'baseline',setup,config=TRAIN_ROOT/settings.ablation_config)
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
                if path.name == 'text__vectors.npz':
                    continue
                path.rename(path.with_name(f'interrupted-{time.time_ns()}-{path.name}'))
            text_complete(output, setup, device='cpu', report_test=settings.report_test)
        else:
            from graph_tracks.config import GraphConfig, load_config as load_graph_config
            from graph_tracks.preflight import preflight
            from graph_tracks.report import complete as graph_complete
            config = load_graph_config(setup / f'{track}.yaml', expected_track=track).model_dump()
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
                           text_cache=Path(config['text_cache']) if config.get('text_cache') else None,
                           saved_inference=output/(track+'__inference'))
        record_completion(output, track)
        print(f'[local-postprocess/{track}] complete', flush=True)
    ablation_done = False
    if settings.post_training_ablation:
        from model_tracks.post_training_ablation import complete_saved
        # Compute once before sealing; publication consumes those exact bytes.
        complete_saved(destination, settings)
        ablation_done = True
        # Ablation adds durable reports after the per-track pair report.
        for track in TRACKS:
            record_completion(destination / track, track)
    suite['postprocess_location'] = 'local CPU'
    (destination / 'suite_manifest.json').write_text(json.dumps(suite, indent=2))
    files = {p.relative_to(destination).as_posix(): p for p in destination.rglob('*')
             if p.is_file() and not p.is_symlink()
             and not ({'local_inputs'} | RESULT_ARCHIVE_EXCLUDED_DIRS).intersection(p.relative_to(destination).parts)}
    write_archive(final, files, manifest_name='suite_bundle_manifest.json',
                  metadata={'run_tag': run_tag, **identity, 'postprocess_location': 'local CPU'})
    from model_tracks.resume import validate_completed_suite_archive
    validate_completed_suite_archive(final, run_tag, settings=settings)
    final.with_suffix('.sha256').write_text(file_hash(final) + '\n')
    return _publish(final, settings, run_tag, ablation_done=ablation_done) if publish else final
