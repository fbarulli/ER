"""Isolated graph DVC snapshots, including all checkpoints and inference outputs.

No repository Git/DVC state is changed. Local cache operation needs no remote
credentials. Push is opt-in; configured remote credentials follow standard DVC
configuration/environment, never stored in result manifests.
"""
from __future__ import annotations
import argparse
import json
import os
from urllib.parse import urlsplit
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from graph_tracks.artifacts import name
from graph_tracks.data import file_size


def _run(args, cwd):
    result = subprocess.run([sys.executable, '-m', 'dvc', *args], cwd=cwd,
                            capture_output=True, text=True)
    if result.returncode:
        # DVC output may include a credential-bearing remote; retain no secrets.
        raise RuntimeError(f'DVC {args[0]} failed with exit code {result.returncode}')
    return result.stdout



def _authenticate(project: Path, remote: str):
    parsed = urlsplit(remote)
    if parsed.scheme in {'http', 'https'}:
        user, password = os.environ.get('DVC_HTTP_USER'), os.environ.get('DVC_HTTP_PASSWORD')
        if parsed.hostname == 'dagshub.com' and os.environ.get('DVC_API_KEY'):
            user, password = parsed.path.strip('/').split('/')[0], os.environ['DVC_API_KEY']
        if password:
            if not user:
                raise ValueError('HTTP DVC credentials require a username')
            _run(['remote', 'modify', '--local', 'graph-store', 'auth', 'basic'], project)
            _run(['remote', 'modify', '--local', 'graph-store', 'user', user], project)
            _run(['remote', 'modify', '--local', 'graph-store', 'password', password], project)
            (project / '.dvc' / 'config.local').chmod(0o600)

def inventory(path: Path):
    return {str(p.relative_to(path)): file_size(p) for p in sorted(path.rglob('*')) if p.is_file()}


def snapshot(source: Path, track: str, *, remote=None, push=False, generation="final") -> Path:
    from graph_tracks.config import DvcSpec
    DvcSpec(enabled=True, remote=remote, push=push)
    source = source.resolve()
    project = source / name(track, f'dvc-{generation}')
    if project.exists():
        raise FileExistsError(project)
    project.mkdir()
    _run(['init', '--no-scm'], project)
    payload = project / name(track, 'payload')
    payload.mkdir()
    for path in source.iterdir():
        if '__dvc-' in path.name or path.name in {'wandb', '_artifact_publications'} or path.name.endswith('__dvc_events.jsonl'):
            continue
        target = payload / path.name
        if path.is_dir():
            shutil.copytree(path, target)
        elif path.is_file():
            shutil.copy2(path, target)
    sizes = inventory(payload)
    manifest = project / name(track, 'dvc_manifest.json')
    manifest.write_text(json.dumps({'schema': 'er-graph-dvc-v1', 'track': track,
        'payload': payload.name, 'files': sizes, 'remote_configured': bool(remote),
        'pushed': push}, indent=2, sort_keys=True) + '\n')
    _run(['add', payload.name], project)
    if remote:
        _run(['remote', 'add', '-d', 'graph-store', remote], project)
        _authenticate(project, remote)
    if push:
        if not remote:
            raise ValueError('DVC push needs explicit remote')
        _run(['push'], project)
    # DVC is WRITE-ONLY storage (owner mandate 2026-10-09): the push IS the
    # deliverable. No read-back restore verifies it, and no lane loads from DVC
    # (restore() remains an operator-only tool below).
    return project


def restore(project: Path, output: Path) -> Path:
    """OPERATOR-ONLY: pull one archived graph snapshot back (never a lane load).

    DVC is write-only storage for every lane; this manual retrieval tool exists
    for an operator outside the pipelines and must not be called by a lane.
    """
    if output.exists():
        raise FileExistsError(output)
    markers = list(project.glob('*__dvc_manifest.json'))
    if len(markers) != 1:
        raise ValueError('DVC project must contain exactly one graph manifest')
    manifest = json.loads(markers[0].read_text())
    track = manifest['track']
    if manifest['payload'] != name(track, 'payload'):
        raise ValueError('DVC payload track mismatch')
    with tempfile.TemporaryDirectory(prefix='er-graph-dvc-pull-') as tmp:
        clean = Path(tmp)
        _run(['init', '--no-scm'], clean)
        pointer = project / (manifest['payload'] + '.dvc')
        shutil.copy2(pointer, clean / pointer.name)
        if manifest['pushed']:
            shutil.copy2(project / '.dvc' / 'config', clean / '.dvc' / 'config')
            local = project / '.dvc' / 'config.local'
            if local.exists():
                shutil.copy2(local, clean / '.dvc' / 'config.local')
            from dvc.repo import Repo
            with Repo(clean) as dvc_repo:
                remote = dvc_repo.config['remote']['graph-store']['url']
            _authenticate(clean, remote)
            _run(['pull', pointer.name], clean)
        else:
            _run(['cache', 'dir', str((project / '.dvc' / 'cache').resolve())], clean)
            _run(['checkout', pointer.name], clean)
        payload = clean / manifest['payload']
        if inventory(payload) != manifest['files']:
            raise RuntimeError('restored DVC files do not match recorded size inventory')
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(payload, output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    restore(args.project, args.output)

if __name__ == '__main__':
    main()
