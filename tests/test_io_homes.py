"""The consolidated I/O homes: one digest, one inventory, one publish, one copy.

Each concept has exactly ONE implementation outside ``core``:

* the file digest -> ``core.portable_archive.raw_file_digest`` /
  ``cached_file_digest`` (``core.manifest.sha256_file`` and
  ``graph_tracks.data.file_hash`` forward there);
* the source inventory -> ``core.portable_archive.source_inventory`` and its
  comparison -> ``core.portable_archive.compare_inventory``;
* the atomic publish -> ``core.manifest.atomic_write* / atomic_write_stream /
  publish_replacing``;
* the only copies outside ``core`` are the pinned standalone ones: the NER
  producer (bare remote runtime) and the Kaggle kernel template helper
  (deployment artifact), both pinned here against the shared implementation.

These pin the CONTRACTS (bytes and error classes), not the classes that carry
them.
"""
from __future__ import annotations

import hashlib
import json
import os
import types
from pathlib import Path

import pytest


def test_the_one_file_digest_home_agrees_across_every_entry_point(tmp_path: Path) -> None:
    from core import manifest, portable_archive
    from graph_tracks.data import file_hash

    sample = tmp_path / 'artifact.bin'
    payload = b'one digest, one implementation\n' * 128
    sample.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()

    assert portable_archive.raw_file_digest(sample) == expected
    assert portable_archive.cached_file_digest(sample) == expected
    assert manifest.sha256_file(sample) == expected
    assert file_hash(sample) == expected
    # str paths take the same route as Paths
    assert manifest.sha256_file(str(sample)) == expected

    with pytest.raises(FileNotFoundError):
        manifest.sha256_file(tmp_path / 'absent.bin')


def test_change_detection_never_serves_a_stale_digest(tmp_path: Path) -> None:
    """``file_hash``/``sha256_file`` are change detectors, so they never memoize.

    The memoized route (``cached_file_digest``) is the one that must be asked
    for by name; the verification paths (frozen ablation sources, worker
    package ``--verify``, git transport checks) must observe a same-size
    rewrite that keeps ``mtime_ns``, or a frozen input could silently be
    reported on with stale bytes.
    """
    from core.manifest import sha256_file
    from core.perf_switches import perf_enabled
    from core.portable_archive import cached_file_digest
    from graph_tracks.data import file_hash

    sample = tmp_path / 'source.csv'
    sample.write_bytes(b'one')
    stat = sample.stat()
    assert cached_file_digest(sample) == hashlib.sha256(b'one').hexdigest()
    assert file_hash(sample) == hashlib.sha256(b'one').hexdigest()

    sample.write_bytes(b'two')  # same length, then force the same mtime_ns
    os.utime(sample, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    assert file_hash(sample) == hashlib.sha256(b'two').hexdigest()
    assert sha256_file(sample) == hashlib.sha256(b'two').hexdigest()
    if perf_enabled('digest.cache'):
        # the memoized policy is the one with the documented staleness window
        assert cached_file_digest(sample) == hashlib.sha256(b'one').hexdigest()


def test_ablation_file_hash_uses_the_shared_memoized_digest(tmp_path: Path) -> None:
    """The ablation lane's digest is the shared home, not a fourth copy.

    ``model_tracks.ablation.file_hash`` rode its own ``_HASH_MEMO`` over a second
    stat/replace body. It now asks ``core.portable_archive.cached_file_digest``
    by name (the memoized policy) and keeps only the lane's directory case.
    """
    from core import portable_archive
    from graph_tracks.data import file_hash as graph_file_hash
    from model_tracks import ablation

    sample = tmp_path / 'frozen.csv'
    payload = b'frozen ablation source\n' * 16
    sample.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert ablation.file_hash(sample) == expected
    assert ablation.file_hash(sample) == portable_archive.cached_file_digest(sample)

    # a same-size rewrite that keeps mtime_ns leaves both on the SAME policy
    stat = sample.stat()
    sample.write_bytes(b'X' * len(payload))
    os.utime(sample, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert ablation.file_hash(sample) == portable_archive.cached_file_digest(sample)

    # a checkpoint DIRECTORY keeps the lane's composition hash, not one file hash
    folder = tmp_path / 'checkpoint'
    folder.mkdir()
    (folder / 'weights').write_bytes(b'w')
    assert ablation.file_hash(folder) == graph_file_hash(folder)

    # the local memo copy is gone: the shared home owns the cache policy
    assert not hasattr(ablation, '_HASH_MEMO')


def test_source_inventory_is_the_one_inventory_builder(tmp_path: Path) -> None:
    from core.portable_archive import source_inventory

    source = tmp_path / 'listing.csv'
    source.write_bytes(b'id\n1\n')
    inline = {'worker.yaml': 'track: gnn_only\n'}
    inventory = source_inventory({'prepared/listing.csv': source}, inline)

    assert inventory == {
        'prepared/listing.csv': hashlib.sha256(b'id\n1\n').hexdigest(),
        'worker.yaml': hashlib.sha256(b'track: gnn_only\n').hexdigest(),
    }

    link = tmp_path / 'link.csv'
    link.symlink_to(source)
    with pytest.raises(ValueError, match='regular files'):
        source_inventory({'link.csv': link}, {})


def test_compare_inventory_is_the_one_comparator() -> None:
    from core.portable_archive import compare_inventory

    inventory = {'a.txt': '0' * 64, 'b.txt': '1' * 64}
    compare_inventory(inventory, dict(inventory))

    with pytest.raises(ValueError, match='undeclared or missing members'):
        compare_inventory(inventory, {'a.txt': '0' * 64})
    with pytest.raises(ValueError, match='b.txt'):
        compare_inventory(inventory, {'a.txt': '0' * 64, 'b.txt': '2' * 64})
    # a caller with its own surface name keeps its own message
    with pytest.raises(ValueError, match='worker package file mismatch: b.txt'):
        compare_inventory(inventory, {'a.txt': '0' * 64, 'b.txt': '2' * 64},
                          mismatch='worker package file mismatch')


def test_atomic_write_stream_matches_atomic_write_and_cleans_residue(tmp_path: Path) -> None:
    from core.manifest import atomic_write_stream, atomic_write_text

    streamed = tmp_path / 'shared.json'
    direct = tmp_path / 'direct.json'
    with atomic_write_stream(streamed) as handle:
        json.dump({'rows': 2}, handle, ensure_ascii=False)
        handle.write('\n')
    atomic_write_text(direct, json.dumps({'rows': 2}, ensure_ascii=False) + '\n')

    assert streamed.read_bytes() == direct.read_bytes()
    assert list(tmp_path.glob('*.tmp-*')) == []

    # an interrupted stream leaves the previous content plus no residue
    streamed.write_bytes(b'previous\n')
    with pytest.raises(RuntimeError):
        with atomic_write_stream(streamed) as handle:
            handle.write('partial')
            raise RuntimeError('interrupted')
    assert streamed.read_bytes() == b'previous\n'
    assert list(tmp_path.glob('*.tmp-*')) == []


def test_publish_replacing_publishes_the_completed_sibling(tmp_path: Path) -> None:
    from core.manifest import publish_replacing

    final = tmp_path / 'result.tar.zst'
    final.write_bytes(b'stale generation')
    temp = tmp_path / 'result.tar.zst.partial'
    temp.write_bytes(b'verified download')

    published = publish_replacing(temp, final)

    assert published == final
    assert final.read_bytes() == b'verified download'
    assert not temp.exists()


def test_kaggle_kernels_inject_the_one_pinned_digest(tmp_path: Path) -> None:
    """The template's helper is the ONE allowed digest copy outside ``core``.

    It cannot import the repo package (a kernel source runs from
    /kaggle/working before/around the checkout), so the contract is: exactly
    one definition per kernel, injected from one place, byte-identical to the
    shared implementation.
    """
    from cli.kaggle_kernel_templates import KernelTemplates
    from core.manifest import sha256_file

    namespace: dict = {'LANE': {'archives': {'copy_buffer_bytes': 65536}},
                       'hashlib': hashlib}
    exec(compile(KernelTemplates.SHA256_HELPER, '<kernel-sha256-helper>', 'exec'),
         namespace)
    kernel_sha256 = namespace['sha256_file']

    sample = tmp_path / 'archive.bin'
    sample.write_bytes(b'kaggle bundle bytes' * 64)
    assert kernel_sha256(sample) == sha256_file(sample)

    for name in ('TRAIN_KERNEL_SHARED', 'BUNDLE_KERNEL_SCRIPT'):
        source = getattr(KernelTemplates, name)
        assert source.count('def sha256_file') == 0, name
        assert source.count('@SHA256_HELPER@') == 1, name


def test_worker_package_verify_uses_the_shared_comparator(tmp_path: Path,
                                                          monkeypatch) -> None:
    """``--verify`` on an extracted worker package keeps its message + guard."""
    from core import common
    from graph_tracks import worker_package

    payload = tmp_path / 'data/graph_worker/gnn_only/worker.yaml'
    payload.parent.mkdir(parents=True)
    payload.write_text('track: gnn_only\n')
    inventory = {'data/graph_worker/gnn_only/worker.yaml':
                 hashlib.sha256(payload.read_bytes()).hexdigest()}
    manifest = tmp_path / 'data/graph_worker/gnn_only/package_manifest.json'
    manifest.write_text(json.dumps({
        'schema': 'er-graph-worker-package-v1', 'base_git_revision': 'rev',
        'files': inventory, 'files_sha256': inventory}))

    monkeypatch.setattr(common, 'TRAIN_ROOT', tmp_path)
    monkeypatch.setattr(
        worker_package.subprocess, 'run',
        lambda *a, **k: types.SimpleNamespace(stdout='rev\n'))
    worker_package.verify(manifest)

    payload.write_text('tampered: cascade\n')
    with pytest.raises(ValueError, match='worker package file mismatch'):
        worker_package.verify(manifest)


def test_worker_package_verify_rejects_a_traversal_member(tmp_path: Path,
                                                          monkeypatch) -> None:
    from core import common
    from graph_tracks import worker_package

    manifest = tmp_path / 'package_manifest.json'
    manifest.write_text(json.dumps({
        'schema': 'er-graph-worker-package-v1', 'base_git_revision': 'rev',
        'files': {'../outside.txt': hashlib.sha256(b'x').hexdigest()}}))
    monkeypatch.setattr(common, 'TRAIN_ROOT', tmp_path)
    monkeypatch.setattr(
        worker_package.subprocess, 'run',
        lambda *a, **k: types.SimpleNamespace(stdout='rev\n'))

    with pytest.raises(ValueError, match='worker package file mismatch'):
        worker_package.verify(manifest)


def test_manifest_atomic_json_publishes_and_keeps_crash_residue_loud(tmp_path: Path) -> None:
    """The manifest home: write-last publish plus the ``.tmp-<pid>`` residue rule.

    A stale sibling from a crashed run that collides with a reused pid must
    fail LOUD (``O_EXCL``), never be silently overwritten — that residue is the
    crash evidence the manifest verifier fails on.
    """
    from core.manifest import atomic_write_json

    target = tmp_path / 'suite_manifest.json'
    atomic_write_json({'status': 'complete'}, target)
    assert json.loads(target.read_text()) == {'status': 'complete'}
    assert list(tmp_path.glob(f'{target.name}.tmp-*')) == []

    residue = tmp_path / f'{target.name}.tmp-{os.getpid()}'
    residue.write_text('crashed mid-write')
    with pytest.raises(FileExistsError):
        atomic_write_json({'status': 'second'}, target)

    assert json.loads(target.read_text()) == {'status': 'complete'}
    assert residue.read_text() == 'crashed mid-write'
