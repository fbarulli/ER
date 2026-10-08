"""Exercise the actual remote Git program against an offline filtered server."""
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest
from cli import colab

ROOT = Path(__file__).resolve().parents[1]


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
             'model_tracks_package.json',
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
    assert (root / 'model_tracks_package.json').is_file()
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


def test_runtime_checkout_includes_bundle_manifest():
    from cli import colab_runtime
    assert '/model_tracks_package.json' in colab_runtime.runtime_checkout_paths()


def test_runtime_checkout_guard_passes_on_repo():
    from cli import colab_runtime
    colab_runtime.validate_runtime_checkout()


def test_runtime_checkout_guard_rejects_missing_file(monkeypatch):
    from cli import colab_runtime
    monkeypatch.setattr(colab_runtime, 'RUNTIME_REQUIRED_ROOT_FILES',
                        ('definitely_missing_file.xyz',))
    with pytest.raises(FileNotFoundError):
        colab_runtime.validate_runtime_checkout()


def test_runtime_checkout_guard_rejects_unpushed_file(tmp_path, monkeypatch):
    from cli import colab_runtime
    name = 'model_tracks_package.json'
    (tmp_path / name).write_text('{}')
    monkeypatch.setattr(colab_runtime, 'TRAIN_ROOT', tmp_path)
    monkeypatch.setattr(colab_runtime, 'RUNTIME_REQUIRED_ROOT_FILES', (name,))
    with pytest.raises(RuntimeError):
        colab_runtime.validate_runtime_checkout()


def test_runtime_checkout_guard_checks_launch_paths():
    from cli import colab_runtime
    colab_runtime.validate_runtime_checkout(
        extra_paths=('src', 'artifacts/models/all-MiniLM-L6-v2'))
    with pytest.raises(RuntimeError):
        colab_runtime.validate_runtime_checkout(extra_paths=('no/such/launch/path',))


def test_resolve_prepared_package_accepts_archive(tmp_path):
    archive = tmp_path / 'all_tracks_inputs.tar.zst'
    archive.write_bytes(b'x')
    assert colab.resolve_prepared_input_package(archive) == archive


def test_resolve_prepared_package_accepts_bundle_directory(tmp_path):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'all_tracks_inputs.tar.zst').write_bytes(b'x')
    assert colab.resolve_prepared_input_package(bundle) == bundle / 'all_tracks_inputs.tar.zst'


def test_resolve_prepared_package_honours_receipt(tmp_path):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'my_bundle.tar.zst').write_bytes(b'x')
    (bundle / 'bundle.receipt.json').write_text('{"archive": "my_bundle.tar.zst"}')
    assert colab.resolve_prepared_input_package(bundle) == bundle / 'my_bundle.tar.zst'


def test_resolve_prepared_package_lists_candidates_on_miss(tmp_path, monkeypatch):
    root = tmp_path / 'results'
    bundle = root / 'kaggle_lane' / 'full' / 'bundle'
    bundle.mkdir(parents=True)
    (bundle / 'all_tracks_inputs.tar.zst').write_bytes(b'x')
    (bundle / 'bundle.receipt.json').write_text('{"cohort": "full", "revision": "abc123"}')
    monkeypatch.setattr(colab, 'RESULTS', root)
    with pytest.raises(FileNotFoundError) as excinfo:
        colab.resolve_prepared_input_package(tmp_path / 'nope')
    message = str(excinfo.value)
    assert 'kaggle_lane/full/bundle' in message
    assert 'cohort=full' in message


# ── ONE checkout contract home (consolidation audit 2026-10-08, finding 3) ──
# Three homes used to answer "what does a prepared Colab runtime check out":
# the hardcoded RUNTIME_DIRECTORY_PATHS/RUNTIME_REQUIRED_ROOT_FILES, the
# lane's hand-rolled checkout_relative_guard charset, and (for the Kaggle
# lane) the config SSOT's kaggle.checkout_paths. The Colab side is now one
# home: the declared lists + their emitted patterns + the path-shape rule all
# live in cli.colab_runtime, and the lane guard delegates to it. The
# hand-rolled guard also owned a SECOND charset (it additionally refused
# ``{}``); the shared rule now refuses it everywhere, so a brace path can no
# longer reach the emitted git pathspec on one side only.

def test_checkout_path_shape_is_one_contract():
    from cli import colab_runtime
    shape = colab_runtime.is_checkout_relative_path
    assert shape('src')
    assert shape('artifacts/models/all-MiniLM-L6-v2')
    for refused in ('/absolute', '../escape', '', '.', 'results/*', 'results/[old]',
                    'results/{old}', 'a\\b', 'a\nb'):
        assert not shape(refused), refused
    # single_component is the lane's extra requirement, not a second charset.
    assert shape('run_1', single_component=True)
    assert not shape('artifacts/models', single_component=True)


def test_lane_guard_delegates_to_the_checkout_contract(monkeypatch):
    from cli import colab_runtime
    from cli.colab_lane_contracts import ColabLaneBase
    real = colab_runtime.is_checkout_relative_path
    calls: list = []

    def spy(value, *, single_component=False):
        calls.append((value, single_component))
        return real(value, single_component=single_component)

    monkeypatch.setattr(colab_runtime, 'is_checkout_relative_path', spy)
    ColabLaneBase.checkout_relative_guard('run_7', message='unused')
    assert calls == [('run_7', True)]
    with pytest.raises(ValueError, match='resume run id must be plain'):
        ColabLaneBase.checkout_relative_guard(
            'nested/run', message='resume run id must be plain')


def test_runtime_checkout_patterns_are_the_declared_contract():
    from cli import colab_runtime
    assert colab_runtime.runtime_checkout_paths() == tuple(
        '/' + name for name in (*colab_runtime.RUNTIME_DIRECTORY_PATHS,
                                *colab_runtime.RUNTIME_REQUIRED_ROOT_FILES))


def test_the_runtime_checkout_lists_have_one_home():
    declaring = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / 'src').rglob('*.py')
        if 'RUNTIME_DIRECTORY_PATHS =' in path.read_text(encoding='utf-8'))
    assert declaring == ['src/cli/colab_runtime.py']


def test_sparse_paths_refuse_the_shared_rejected_charset():
    for value in ('results/*', 'results/{old}', 'a\\b', 'a\nb'):
        with pytest.raises(ValueError):
            colab.prepare_remote_layout(minimal_runtime=True, sparse_paths=(value,))


def test_runtime_checkout_members_ask_the_one_shape_contract():
    """The staging selection delegates the shape rule to the shared predicate.

    ``core.runtime_inputs.checkout_members`` used to re-spell the traversal /
    pattern charset and missed ``{}``; it now asks
    ``cli.colab_runtime.is_checkout_relative_path`` for every member (including
    the launch's extras), so a path can no longer pass locally and be refused by
    the VM's sparse checkout.
    """
    from cli import colab_runtime
    from core import runtime_inputs

    for refused in ('/absolute', '../escape', 'results/*', 'results/{old}',
                    'a\\b', 'a\nb'):
        assert not colab_runtime.is_checkout_relative_path(refused), refused
        with pytest.raises(ValueError, match='repository-relative'):
            runtime_inputs.checkout_members((refused,))
    assert 'run_1' in runtime_inputs.checkout_members(('run_1',))
