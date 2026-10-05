"""One deduplicated prepared-input package for one all-track Colab run."""
import json
from pathlib import Path
import subprocess
import yaml

from core.portable_archive import write_archive, verify_archive, verified_archive
from model_tracks.config import load_config
from model_tracks.preflight import preflight


def runtime_snapshot_files(*, ablation_config: Path | None = None) -> dict[str, Path]:
    """Shared local source/config overlay for prepared Colab jobs."""
    from core.common import TRAIN_ROOT
    files = {}
    files.update({path.relative_to(TRAIN_ROOT).as_posix(): path
                  for path in (TRAIN_ROOT / 'src').rglob('*.py')})
    for name in ('src/pipeline.py','scripts/diet_manifest.py',
                 'scripts/run_colab_ablation.py','scripts/run_colab_embeddings.py',
                 'src/cli/colab.py','src/cli/__init__.py'):
        files[name] = TRAIN_ROOT/name
    for name in ('paths.yaml','training.yaml','identity_dimensions.yaml','identity_reviews.json','vocabulary.json','text_track.yaml','attribute_ablation.yaml'):
        files['config/'+name] = TRAIN_ROOT/'config'/name
    # The semantic family registry is a REQUIRED frozen input: calibration
    # refuses to start without it (core.attribute_decision raises rather than
    # run on an open vocabulary). It lives under results/, which is gitignored
    # AND excluded from the Colab sparse checkout, so unless it ships in the
    # package the GPU calibration lane can never find it. Fail here, at package
    # time, instead of on the remote after provisioning an accelerator.
    registry = TRAIN_ROOT / 'results' / 'semantics' / 'family_registry.json'
    if not registry.is_file():
        raise FileNotFoundError(
            'semantic family registry missing: run scripts/build_attribute_semantics.py '
            'before packaging (no silent open-vocabulary gap)')
    files['results/semantics/family_registry.json'] = registry
    if ablation_config is not None:
        source = Path(ablation_config).resolve()
        files[source.relative_to(TRAIN_ROOT).as_posix()] = source
    from core.portable_archive import RuntimeSnapshot
    return RuntimeSnapshot(files=files).files


def package(config: Path, output: Path) -> Path:
    from core.common import F, TRAIN_ROOT
    from core.timing import Timing

    timing = Timing("model_tracks.package")
    cfg = load_config(config)
    setup = (TRAIN_ROOT / cfg.setup_dir).resolve()
    from training.prepared_bundle import load_prepared_bundle
    from model_tracks.training_data import (
        from_bundle,
        frozen_endpoint_text,
        TrackTrainingBinding,
    )
    from model_tracks.shared_graph_data import prepare_shared_graph
    _, bundle = load_prepared_bundle((TRAIN_ROOT / cfg.text_bundle).resolve())
    shared = from_bundle(bundle)
    (setup / 'shared_training_data.json').write_text(shared.model_dump_json(indent=2) + '\n')
    text_binding = TrackTrainingBinding(track='text', shared_data_sha256=shared.fingerprint,
        example_ids=[row.example_id for row in shared.examples],
        endpoint_indices=[row.payload_index for row in shared.endpoints])
    (setup / 'text_training_binding.json').write_text(text_binding.model_dump_json(indent=2) + '\n')
    prepare_shared_graph(setup, bundle, shared)
    timing.mark('shared_training_population')
    from graph_tracks.config import load_text_config
    text_settings = load_text_config(setup / 'text.yaml').model_dump()
    # All tensors and native tokenizer features are fixed on local CPU before
    # provisioning; selected weights are bound only by the GPU exporter.
    from graph_tracks.prepared_inputs import prepare_training
    from graph_tracks.config import load_config as load_graph_config
    track_settings = [load_graph_config(setup/(track+'.yaml'), expected_track=track).model_dump() for track in ('gnn_only','hybrid')]
    sizes = {settings['inference_batch_size'] for settings in track_settings}
    if len(sizes) != 1:
        raise ValueError('shared prepared graph inference batch sizes must agree')
    cache = setup/'shared_minilm__embeddings.npz'
    prepare_training(setup/'prepared/listings.json',setup/'prepared/pairs.csv',batch_size=sizes.pop())
    timing.mark('graph_prepare')
    from model_tracks.text_export import prepare as prepare_text_export
    from core.common import resolve_model
    from core.common import runtime
    from core.model_input import model_input_composition,build_sku_text,model_input_info
    from core.sku_identity import row_identity
    from graph_tracks.text_cache import composition_fingerprint
    from model_tracks.ablation import digest
    import pandas as pd
    # This cache lasts for one verified local preparation only. Exact raw rows
    # and the frozen composition contract key both baseline and interventions.
    composition_contract = {'spec':model_input_composition().model_dump(mode='json'),
                            'implementation':composition_fingerprint()}
    composed,token_cache = {},{}
    # Frozen endpoint text is a CONTRACT, not a hint: a virtual endpoint whose
    # bundle text is empty is still authoritative, so virtualness decides and an
    # empty cell is a value rather than a reason to recompose. The digest itself
    # is verified once per endpoint by the shared graph projection.
    def compose(row):
        frozen = frozen_endpoint_text(row.get('sku_id'), row.get('frozen_payload'),
                                      column_present='frozen_payload' in row)
        if frozen is not None:
            return frozen
        key = digest({'row':row,'composition':composition_contract})
        if key not in composed:
            series = pd.Series(row)
            composed[key] = build_sku_text(series,model_input_info(row_identity(series).as_mapping()))
        return composed[key]
    prepare_text_export(setup,Path(resolve_model(cfg.text_model)),batch_size=runtime('batch_size_embed'),composer=compose,token_cache=token_cache)
    timing.mark('text_export')
    if cfg.post_training_ablation:
        from model_tracks.staged_ablation import prepare_suite
        prepare_suite(setup,Path(resolve_model(cfg.text_model)),TRAIN_ROOT/cfg.ablation_config,composer=compose,token_cache=token_cache,bundle=bundle)
        timing.mark('ablation_suite')
    from model_tracks.baseline_export import prepare as prepare_baseline
    prepare_baseline(setup,Path(resolve_model(cfg.text_model)),composer=compose)
    timing.mark('baseline_export')
    native_model = next(value for key,value in token_cache.items() if key[0] == 'model')
    # Preflight independently reloads and validates the bundle. Release the
    # producer's object graph first rather than retaining two full populations.
    del bundle, shared, text_binding
    composed.clear()
    token_cache.clear()
    checks = preflight(config,allow_gpu_pending=True,native_token_model=native_model)
    timing.mark('preflight')
    target = Path('data/model_tracks/shared')
    bundle_path = (TRAIN_ROOT / cfg.text_bundle).resolve()
    bundle_sources = {bundle_path, bundle_path.with_suffix(bundle_path.suffix + '.json')}
    files = {str(target / p.relative_to(setup)):p for p in setup.rglob('*') if p.is_file()
             and p.name not in {'gnn_only.yaml','hybrid.yaml','text.yaml'} and p.suffix not in {'.zip'}
             and p.resolve() not in bundle_sources
             and p.name not in {'text_prepared.pkl.gz','text_prepared.pkl.gz.json'}}
    text_settings.update(report_test=cfg.report_test)
    inline = {str(target/'text.yaml'): yaml.safe_dump(text_settings, sort_keys=False)}
    for track in ('gnn_only','hybrid'):
        settings = load_graph_config(setup/f'{track}.yaml', expected_track=track).model_dump()
        for key in ('listings','pairs','input_manifest','text_cache'):
            if settings.get(key):
                source = (TRAIN_ROOT / settings[key]).resolve()
                settings[key] = str(target / source.relative_to(setup))
        settings.update(device=cfg.device, report_test=cfg.report_test)
        settings.update(cfg.graph_execution_overrides())
        inline[str(target/f'{track}.yaml')] = yaml.safe_dump(settings,sort_keys=False)
    bundle = (TRAIN_ROOT / cfg.text_bundle).resolve()
    files[str(target/'text_prepared.pkl.gz')] = bundle
    files[str(target/'text_prepared.pkl.gz.json')] = bundle.with_suffix(bundle.suffix+'.json')
    settings = cfg.model_dump()
    settings.update(setup_dir=str(target),text_bundle=str(target/'text_prepared.pkl.gz'))
    inline['data/model_tracks/suite.yaml'] = yaml.safe_dump(settings,sort_keys=False)
    # Freeze the same shared runtime overlay used by inference-only jobs.
    files.update(runtime_snapshot_files(ablation_config=TRAIN_ROOT/cfg.ablation_config))
    for key in ('dataset_deduped', 'labeled_pairs', 'canonical_records', 'gate_results'):
        source = Path(F[key]).resolve()
        files[source.relative_to(TRAIN_ROOT).as_posix()] = source
    revision = subprocess.run(['git','rev-parse','HEAD'],cwd=TRAIN_ROOT,capture_output=True,text=True,check=True).stdout.strip()
    archive = write_archive(output,files,inline=inline,manifest_name='model_tracks_package.json',
                            metadata={'schema':'er-model-tracks-package-v1','revision':revision,'preflight':checks},
                            profile=True)
    timing.mark('archive_write')
    timing.dump_if_requested()
    return archive


def verify(path: Path):
    return verify_archive(path,'model_tracks_package.json')


def verify_current(path: Path, config: Path):
    """Reuse a completed CPU package only while local inputs/config stay bound."""
    from core.archive_reader import open_archive
    from core.common import TRAIN_ROOT, F, resolve_model
    from graph_tracks.data import file_hash
    from graph_tracks.text_cache import checkpoint_hash
    cfg = load_config(config)
    setup = (TRAIN_ROOT/cfg.setup_dir).resolve()
    target = Path('data/model_tracks/shared')
    expected_suite = cfg.model_dump()
    expected_suite.update(setup_dir=str(target), text_bundle=str(target/'text_prepared.pkl.gz'))
    from graph_tracks.config import load_text_config, load_config as load_graph_config
    expected_text = load_text_config(setup / 'text.yaml').model_dump()
    expected_text.update(report_test=cfg.report_test)
    with verified_archive(path, 'model_tracks_package.json') as (archive, metadata):
        if yaml.safe_load(archive.read('data/model_tracks/suite.yaml')) != expected_suite:
            raise ValueError('prepared package suite config changed; regenerate locally')
        if yaml.safe_load(archive.read(str(target/'text.yaml'))) != expected_text:
            raise ValueError('prepared package text configuration changed; regenerate locally')
        request = json.loads(archive.read(str(target/'embedding_inputs.json')))
        if request['metadata']['checkpoint_sha256'] != checkpoint_hash(Path(resolve_model(cfg.text_model))):
            raise ValueError('prepared package baseline checkpoint changed')
        for track in ('gnn_only', 'hybrid'):
            settings = load_graph_config(setup/(track+'.yaml'), expected_track=track).model_dump()
            for key in ('listings','pairs','input_manifest','text_cache'):
                if settings.get(key):
                    settings[key] = str(target/(TRAIN_ROOT/settings[key]).resolve().relative_to(setup))
            settings.update(device=cfg.device, report_test=cfg.report_test)
            settings.update(cfg.graph_execution_overrides())
            if yaml.safe_load(archive.read(str(target/(track+'.yaml')))) != settings:
                raise ValueError('prepared package graph configuration changed: '+track)
    sources = runtime_snapshot_files(ablation_config=TRAIN_ROOT/cfg.ablation_config)
    for key in ('dataset_deduped','labeled_pairs','canonical_records','gate_results'):
        source = Path(F[key]).resolve()
        sources[source.relative_to(TRAIN_ROOT).as_posix()] = source
    bundle = (TRAIN_ROOT/cfg.text_bundle).resolve()
    for name, expected in metadata['files'].items():
        member = Path(name)
        if member.is_relative_to(target) and member.name not in {'gnn_only.yaml','hybrid.yaml','text.yaml'}:
            source = (bundle if member == target/'text_prepared.pkl.gz' else
                      bundle.with_suffix(bundle.suffix+'.json') if member == target/'text_prepared.pkl.gz.json'
                      else setup/member.relative_to(target))
            sources[name] = source
    for name, source in sources.items():
        if not source.is_file() or metadata['files'].get(name) != file_hash(source):
            raise ValueError('prepared package source changed: '+name)
    return metadata


def recovery_package(output: Path, destination: Path, run_tag: str, *, input_package: dict | None = None) -> Path:
    """Capture stopped workers' portable state without credentials or caches."""
    manifest = json.loads((output / 'suite_manifest.json').read_text())
    if manifest.get('run_tag') != run_tag:
        raise ValueError('recovery suite run mismatch')
    excluded = {'wandb', 'mlruns', 'mps_pipe', 'mps_log', '.git', '.dvc'}
    files = {}
    for path in output.rglob('*'):
        relative = path.relative_to(output)
        if (not path.is_file() or path.is_symlink()
                or not path.resolve().is_relative_to(output.resolve())
                or any(part in excluded or part.endswith(('.publication', '__payload'))
                       for part in relative.parts)
                or path.name in {'.env', 'config.local'}):
            continue
        files[relative.as_posix()] = path
    return write_archive(destination, files, manifest_name='suite_recovery_manifest.json',
                         metadata={'schema': 'er-suite-recovery-v1', 'run_tag': run_tag,
                                   'input_package': input_package})


def restore_recovery(archive: Path, output: Path, run_tag: str) -> Path:
    """Verify before restoring an interrupted suite into a fresh output directory."""
    import zipfile
    metadata = verify_archive(archive, 'suite_recovery_manifest.json')
    if metadata.get('schema') != 'er-suite-recovery-v1' or metadata.get('run_tag') != run_tag:
        raise ValueError('recovery suite run mismatch')
    with zipfile.ZipFile(archive) as source:
        declared = set(metadata['files']) | {'suite_recovery_manifest.json'}
        if set(source.namelist()) != declared:
            raise ValueError('unlisted recovery archive members')
        manifest = json.loads(source.read('suite_manifest.json'))
        if manifest.get('run_tag') != run_tag:
            raise ValueError('recovery suite manifest mismatch')
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=True)
        for member in metadata['files']:
            target = output / member
            if not target.resolve().is_relative_to(output.resolve()):
                raise ValueError('unsafe recovery archive member')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read(member))
    return output


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(package(args.config, args.output))


if __name__ == '__main__':
    main()
