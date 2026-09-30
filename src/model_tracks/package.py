"""One deduplicated prepared-input package for one all-track Colab run."""
import json
from pathlib import Path
import subprocess
import yaml

from core.portable_archive import write_archive, verify_archive
from model_tracks.config import load_config
from model_tracks.preflight import preflight


def package(config: Path, output: Path):
    from core.common import TRAIN_ROOT
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
    # Ship one source overlay including extracted modules and shared parsers,
    # so the prepared-input producer and remote consumer use the same code.
    for directory in ('src/graph_tracks','src/model_tracks','src/training','src/core'):
        files.update({str(p.relative_to(TRAIN_ROOT)):p for p in (TRAIN_ROOT/directory).glob('*.py')})
    revision = subprocess.run(['git','rev-parse','HEAD'],cwd=TRAIN_ROOT,capture_output=True,text=True,check=True).stdout.strip()
    return write_archive(output,files,inline=inline,manifest_name='model_tracks_package.json',
                         metadata={'schema':'er-model-tracks-package-v1','revision':revision,'preflight':checks})


def verify(path: Path):
    return verify_archive(path,'model_tracks_package.json')
