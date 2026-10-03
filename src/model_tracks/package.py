"""One deduplicated prepared-input package for one all-track Colab run."""
import json
from pathlib import Path
import subprocess
import yaml

from core.portable_archive import write_archive, verify_archive
from model_tracks.config import load_config
from model_tracks.preflight import preflight


def runtime_snapshot_files():
    """Shared local source/config overlay for prepared Colab jobs."""
    from core.common import TRAIN_ROOT
    files = {}
    for directory in ('src/graph_tracks','src/model_tracks','src/training','src/core'):
        files.update({p.relative_to(TRAIN_ROOT).as_posix():p for p in (TRAIN_ROOT/directory).glob('*.py')})
    for name in ('src/pipeline.py','scripts/diet_manifest.py'):
        files[name] = TRAIN_ROOT/name
    for name in ('paths.yaml','training.yaml','identity_dimensions.yaml','identity_reviews.json','vocabulary.json'):
        files['config/'+name] = TRAIN_ROOT/'config'/name
    return files


def package(config: Path, output: Path):
    from core.common import F, TRAIN_ROOT
    cfg = load_config(config)
    checks = preflight(config)
    setup = (TRAIN_ROOT / cfg.setup_dir).resolve()
    target = Path('data/model_tracks/shared')
    files = {str(target / p.relative_to(setup)):p for p in setup.rglob('*') if p.is_file()
             and p.name not in {'gnn_only.yaml','hybrid.yaml'} and p.suffix not in {'.zip'}
             and p.name not in {'text_prepared.pkl.gz','text_prepared.pkl.gz.json'}}
    inline = {}
    for track in ('gnn_only','hybrid'):
        settings = yaml.safe_load((setup/f'{track}.yaml').read_text())
        for key in ('listings','pairs','input_manifest','text_cache'):
            if settings.get(key):
                source = (TRAIN_ROOT / settings[key]).resolve()
                settings[key] = str(target / source.relative_to(setup))
        settings.update(device=cfg.device, report_test=cfg.report_test)
        inline[str(target/f'{track}.yaml')] = yaml.safe_dump(settings,sort_keys=False)
    bundle = (TRAIN_ROOT / cfg.text_bundle).resolve()
    files[str(target/'text_prepared.pkl.gz')] = bundle
    files[str(target/'text_prepared.pkl.gz.json')] = bundle.with_suffix(bundle.suffix+'.json')
    settings = cfg.model_dump()
    settings.update(setup_dir=str(target),text_bundle=str(target/'text_prepared.pkl.gz'))
    inline['data/model_tracks/suite.yaml'] = yaml.safe_dump(settings,sort_keys=False)
    # Freeze the same shared runtime overlay used by inference-only jobs.
    files.update(runtime_snapshot_files())
    for key in ('dataset_deduped', 'labeled_pairs', 'canonical_records', 'gate_results'):
        source = Path(F[key]).resolve()
        files[source.relative_to(TRAIN_ROOT).as_posix()] = source
    revision = subprocess.run(['git','rev-parse','HEAD'],cwd=TRAIN_ROOT,capture_output=True,text=True,check=True).stdout.strip()
    return write_archive(output,files,inline=inline,manifest_name='model_tracks_package.json',
                         metadata={'schema':'er-model-tracks-package-v1','revision':revision,'preflight':checks})


def verify(path: Path):
    return verify_archive(path,'model_tracks_package.json')


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
