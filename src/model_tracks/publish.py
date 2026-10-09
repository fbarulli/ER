"""Supervisor-only publication of all three selected inference models."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from core.archive_reader import archive_sidecar
from core.artifacts import Artifacts
from core.bundle import bundle_spec


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


def persist_results(archive: Path, run_tag: str, *, bundle=None) -> Path:
    """Publish the entire immutable suite, with a verified clean DVC pull.

    ``bundle`` is the already-verified boundary handle for ``archive``; a caller
    that already verified the bytes at the VM crossing passes it so publication
    does not open the same archive a second time.
    """
    from core.bundle import Bundle, BundleRole
    from core import common
    from core.portable_archive import file_size
    from training import dvc_store
    # One boundary check; the handle also supplies the archive's size.
    handle = Bundle.load(Path(archive), BundleRole.result) if bundle is None else bundle
    if handle.run_tag() != run_tag:
        raise ValueError('publication run mismatch')
    workspace = archive_sidecar(archive, bundle_spec().publication_sidecar_suffix)
    workspace.mkdir(exist_ok=True)
    payload = workspace / archive.name
    if payload.exists() and file_size(payload) != handle.path.stat().st_size:
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
    receipt = archive_sidecar(archive, bundle_spec().publication_receipt_suffix)
    receipt.write_text(json.dumps({
        'run_tag': run_tag, 'archive_size': handle.path.stat().st_size,
        'verified_download': True,
        'references': {p.relative_to(common.TRAIN_ROOT).as_posix(): p.read_text() for p in refs},
    }, indent=2) + '\n')
    return receipt


def _selected_graph_checkpoint(tree, track: str) -> Path:
    """The one selected graph checkpoint; any ambiguity is fatal.

    The selection rule is the Bundle's (``Bundle.checkpoint``, called below): the
    ``*__best_checkpoint.json`` marker names the selected model, the member is
    located by name under the track (parent-qualified first, bare name as the
    transport fallback), and several matches are refused there instead of being
    ranked by name order. Publication adds the ONE thing the handle still
    tolerates: a second recorded marker, which must stop the publication rather
    than be resolved by marker name order. A recorded selection with no member
    stays unavailable.
    """
    root = tree._root()
    track_root = root / track if (root / track).is_dir() else root
    markers = sorted(track_root.rglob(bundle_spec().best_checkpoint_glob))
    if len(markers) != 1:
        recorded = ', '.join(marker.relative_to(root).as_posix() for marker in markers) or 'none'
        raise ValueError(f'ambiguous selected checkpoint: {track} ({recorded})')
    selected = tree.checkpoint(track)
    if selected is None or not selected.is_file():
        raise ValueError(f'selected checkpoint unavailable: {track}')
    return selected


def materialize(archive: Path, run_tag: str, *, push: bool = False, bundle=None) -> Path:
    from core.bundle import Bundle, BundleRole
    from core.common import TRAIN_ROOT, training_cfg
    from core.portable_archive import file_size
    from graph_tracks.artifacts import name
    from graph_tracks.data import file_size
    from model_tracks.resume import validate_completed_suite_archive
    import torch
    spec = bundle_spec()
    # The boundary check happens once; the completion contract and the size
    # both come off that handle (or the caller's already-verified one).
    handle = Bundle.load(Path(archive), BundleRole.result) if bundle is None else bundle
    validate_completed_suite_archive(archive, run_tag, bundle=handle)
    if handle.run_tag() != run_tag:
        raise ValueError('publication run mismatch')
    destination = TRAIN_ROOT/'artifacts/models/tracks'/run_tag
    if destination.exists():
        existing = json.loads(Artifacts.resolve('models_manifest', root=destination).read_text())
        if existing.get('source_archive_size') != handle.path.stat().st_size:
            raise ValueError('existing models came from a different suite archive')
        if any(file_size(destination/key) != size
               for key, size in existing['files'].items()):
            raise ValueError('existing publication model files differ')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists():
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            workspace = Path(temporary)
            # The verified handle materializes the tree; the selected checkpoints
            # come from the bundle's role contract, not a per-surface re-search.
            tree = handle.materialize(workspace/'restored')
            staged = workspace/'models'
            staged.mkdir()
            checkpoint = tree.checkpoint('text')
            if checkpoint is None:
                raise ValueError('selected checkpoint unavailable: text')
            ignored = set(spec.deployment_ignored_filenames)
            shutil.copytree(checkpoint,staged/'text',ignore=lambda _, names: [n for n in names if n in ignored])
            for track in ('gnn_only',):
                selected = _selected_graph_checkpoint(tree, track)
                from graph_tracks.artifacts import checkpoint_track
                checkpoint_track(selected)
                payload = torch.load(selected,map_location='cpu',weights_only=False)
                fields = ('schema','manifest','vocabulary','support_records','support_text','text_dim','model','scorer')
                deployed = {key:payload[key] for key in fields}
                folder = staged/track
                folder.mkdir()
                model = folder/name(track,'graph_model.pt')
                torch.save(deployed,model)
                (folder/name(track, training_cfg().colab.checkpoint_manifest_name)).write_text(json.dumps({
                    'schema':'er-graph-checkpoint-v1','track':track,'files':{model.name:file_size(model)},
                    'inference_only':True,'source_checkpoint_size':file_size(selected)},indent=2)+'\n')
            inventory = {str(p.relative_to(staged)):file_size(p) for p in staged.rglob('*') if p.is_file()}
            for path in staged.rglob('*'):
                if path.is_file() and path.stat().st_size>=100*1024**2:
                    raise ValueError(f'model file exceeds GitHub regular-file limit: {path.name}')
            Artifacts.resolve('models_manifest', root=staged).write_text(json.dumps({
                'run_tag':run_tag,'source_archive_size':handle.path.stat().st_size,'files':inventory,
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
