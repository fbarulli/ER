"""Exercise the actual remote Git program against an offline filtered server."""
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest
from cli import colab


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def checkout(root, repository, paths=()):
    with mock.patch.multiple(colab, REMOTE_ROOT=str(root), REPOSITORY=repository,
                             BRANCH='main', GIT_REMOTE_NAME='ER'), \
         mock.patch.object(colab, 'run_colab_exec_stream') as remote:
        colab.prepare_remote_layout(minimal_runtime=bool(paths), sparse_paths=paths)
    subprocess.run([sys.executable, '-c', remote.call_args.args[1]], check=True,
                   capture_output=True, text=True)


def test_sparse_checkout_fresh_reuse_and_legacy(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    git(source, 'init', '-b', 'main')
    git(source, 'config', 'user.email', 'test@example.com')
    git(source, 'config', 'user.name', 'Test')
    files = ['src/runtime.py', 'config/training.yaml', 'scripts/helper.py',
             'artifacts/wheels/runtime.whl', 'pyproject.toml', 'colab_backend.py',
             'models/minilm/weights.bin', 'results/model_tracks/inputs/current.tar.gz',
             'results/model_tracks/inputs/old.tar.gz', 'data/raw.csv']
    for name in files:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    git(source, 'add', '.')
    git(source, 'commit', '-m', 'runtime')
    pinned = git(source, 'rev-parse', 'HEAD')
    (source / 'src/runtime.py').write_text('new runtime')
    git(source, 'commit', '-am', 'transport publication')
    server = tmp_path / 'server.git'
    subprocess.run(['git', 'clone', '--bare', str(source), str(server)], check=True,
                   capture_output=True)
    git(server, 'config', 'uploadpack.allowFilter', 'true')
    git(server, 'config', 'uploadpack.allowAnySHA1InWant', 'true')
    root = tmp_path / 'remote'
    paths = ('results/model_tracks/inputs/current.tar.gz', 'models/minilm')
    checkout(root, server.as_uri(), paths)
    assert git(root, 'rev-list', '--count', 'HEAD') == '1'
    assert (root / paths[0]).is_file()
    assert (root / 'models/minilm/weights.bin').is_file()
    assert not (root / 'results/model_tracks/inputs/old.tar.gz').exists()
    assert not (root / 'data/raw.csv').exists()
    # Historical blobs must not merely be hidden in the working tree.
    raw_blob = git(source, 'rev-parse', 'HEAD:data/raw.csv')
    objects = git(root, 'rev-list', '--objects', '--all', '--missing=print')
    assert '?' + raw_blob in objects
    git(root, 'fetch', '--depth=1', '--filter=blob:none', '--no-tags', 'ER', pinned)
    git(root, 'checkout', '--detach', pinned)
    assert (root / 'src/runtime.py').read_text() == 'src/runtime.py'
    checkout(root, server.as_uri(), paths)
    assert (root / 'src/runtime.py').read_text() == 'new runtime'
    assert not (root / 'data/raw.csv').exists()
    # A legacy lane still checks out the full current tree.
    checkout(root, server.as_uri())
    assert (root / 'data/raw.csv').is_file()


def test_sparse_paths_reject_patterns_and_traversal():
    for value in ('../escape', '/absolute', 'results/*', 'results/[old]', '.'):
        with pytest.raises(ValueError):
            colab.prepare_remote_layout(minimal_runtime=True, sparse_paths=(value,))
