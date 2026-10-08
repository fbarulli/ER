"""Supervisor-only publication of all three selected inference models."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from core.archive_reader import open_archive, archive_sidecar

from core.portable_archive import verify_archive

def push_artifacts(paths: list[Path], message: str) -> None:
    """Publish only explicit artifact paths, using the existing Git save flow."""
    from core.common import TRAIN_ROOT
    pending = subprocess.run(['git', 'diff', '--cached', '--name-only'], cwd=TRAIN_ROOT,
                             text=True, capture_output=True, check=True).stdout.strip()
    if pending:
        raise RuntimeError('artifact publication refuses to commit unrelated staged changes')
    relatives = [str(path.resolve().relative_to(TRAIN_ROOT.resolve())) for path in paths]
    subprocess.run(['git', 'add', '-f', '--', *relatives], cwd=TRAIN_ROOT, check=True)
    changed = subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=TRAIN_ROOT).returncode
    if changed == 1:
        subprocess.run(['git', 'commit', '-m', message], cwd=TRAIN_ROOT, check=True)
    elif changed != 0:
        raise RuntimeError('could not inspect staged publication')
    subprocess.run(['git', 'push', 'origin', 'HEAD'], cwd=TRAIN_ROOT, check=True)


def persist_results(archive: Path, run_tag: str) -> Path:
    """Publish the entire immutable suite, with a verified clean DVC pull."""
    from core import common
    from graph_tracks.data import file_hash
    from training import dvc_store
    metadata = verify_archive(archive, 'suite_bundle_manifest.json')
    if metadata['run_tag'] != run_tag:
        raise ValueError('publication run mismatch')
    workspace = archive_sidecar(archive, '.publication')
    workspace.mkdir(exist_ok=True)
    payload = workspace / archive.name
    if payload.exists() and file_hash(payload) != file_hash(archive):
        raise ValueError('existing publication payload differs from suite')
    if not payload.exists():
        shutil.copy2(archive, payload)
    # Explicitly add the ZIP: the legacy publisher's suffix filter covers
    # reports, while this single target also includes binary checkpoints.
    import os
    token = os.environ.get('DVC_API_KEY')
    if not token:
        raise RuntimeError('DVC_API_KEY is required for suite persistence')
    dvc_store._configure(workspace, token)
    dvc_store._run(['dvc', 'add', payload.name], workspace)
    dvc_store.publish(workspace, run_tag, 1)
    index = common.artifact('dvc_publication_manifest', {'run_id': run_tag, 'worker': 1})
    publication = json.loads(index.read_text())
    refs = [index, *(common.TRAIN_ROOT / entry['pointer'] for entry in publication['pointers'])]
    receipt = archive_sidecar(archive, '.publication.json')
    receipt.write_text(json.dumps({
        'run_tag': run_tag, 'archive_sha256': file_hash(archive),
        'verified_download': True,
        'references': {p.relative_to(common.TRAIN_ROOT).as_posix(): p.read_text() for p in refs},
    }, indent=2) + '\n')
    return receipt


def materialize(archive: Path, run_tag: str, *, push: bool = False) -> Path:
    from core.common import TRAIN_ROOT
    from graph_tracks.artifacts import name
    from graph_tracks.data import file_hash
    from training.validation_inference import resolve_best_checkpoint
    import torch
    from model_tracks.resume import validate_completed_suite_archive
    metadata = validate_completed_suite_archive(archive, run_tag)
    if metadata['run_tag'] != run_tag:
        raise ValueError('publication run mismatch')
    destination = TRAIN_ROOT/'artifacts/models/tracks'/run_tag
    if destination.exists():
        existing = json.loads((destination/'models_manifest.json').read_text())
        if existing.get('source_archive_sha256') != file_hash(archive):
            raise ValueError('existing models came from a different suite archive')
        if any(file_hash(destination/key) != digest for key, digest in existing['files'].items()):
            raise ValueError('existing publication model files differ')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists():
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            workspace = Path(temporary)
            restored = workspace/'restored'
            with open_archive(archive) as source:
                source.extractall(restored)  # path/link validation was done by verify_archive
            staged = workspace/'models'
            staged.mkdir()
            checkpoint, _ = resolve_best_checkpoint(restored/'text')
            ignored = {'optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','training_args.bin','scaler.pt'}
            shutil.copytree(checkpoint,staged/'text',ignore=lambda _, names: [n for n in names if n in ignored])
            for track in ('gnn_only',):
                selected = list((restored/track).rglob(name(track,'best_checkpoint.json')))
                if len(selected)!=1:
                    raise ValueError(f'ambiguous selected checkpoint: {track}')
                recorded = Path(json.loads(selected[0].read_text())['path'])
                candidates = list((restored/track).rglob(f'{recorded.parent.name}/{recorded.name}'))
                if len(candidates)!=1:
                    raise ValueError(f'selected checkpoint unavailable: {track}')
                from graph_tracks.artifacts import checkpoint_track
                checkpoint_track(candidates[0])
                payload = torch.load(candidates[0],map_location='cpu',weights_only=False)
                fields = ('schema','manifest','vocabulary','support_records','support_text','text_dim','model','scorer')
                deployed = {key:payload[key] for key in fields}
                folder = staged/track
                folder.mkdir()
                model = folder/name(track,'graph_model.pt')
                torch.save(deployed,model)
                (folder/name(track,'checkpoint_manifest.json')).write_text(json.dumps({
                    'schema':'er-graph-checkpoint-v1','track':track,'files':{model.name:file_hash(model)},
                    'inference_only':True,'source_checkpoint_sha256':file_hash(candidates[0])},indent=2)+'\n')
            inventory = {str(p.relative_to(staged)):file_hash(p) for p in staged.rglob('*') if p.is_file()}
            for path in staged.rglob('*'):
                if path.is_file() and path.stat().st_size>=100*1024**2:
                    raise ValueError(f'model file exceeds GitHub regular-file limit: {path.name}')
            (staged/'models_manifest.json').write_text(json.dumps({
                'run_tag':run_tag,'source_archive_sha256':file_hash(archive),'files':inventory,
                'tracks':['text','gnn_only'],'graph_models_inference_only':True,
                'cascade_composed_from':['text ranker (ANN candidates)','gnn_only pair scorer (decisions)']
            },indent=2)+'\n')
            staged.rename(destination)
    if push:
        references = TRAIN_ROOT / 'dvc_refs' / run_tag / 'worker_1'
        if not references.is_dir():
            raise RuntimeError('model publication requires durable suite DVC references')
        reference_dirs = [p for p in (TRAIN_ROOT/'dvc_refs').iterdir()
                          if p.is_dir() and (p.name == run_tag or p.name.startswith(run_tag+'-'))]
        push_artifacts([destination, *reference_dirs], f'models: publish all tracks from {run_tag}')
    return destination
