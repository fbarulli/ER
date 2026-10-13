"""The ONE sidecar path rule, and the surfaces that must speak it.

A companion of an archive is ``<name><suffix>`` with the archive's compressed
ending STRIPPED: ``<run_tag>.tar.zst`` pairs with ``<run_tag>.size``, never
``<run_tag>.tar.zst.size``. That is the shape of every real companion on
disk (see
``results/model_tracks/<run_tag>.size`` beside ``<run_tag>.tar.zst``), so a
producer or consumer that re-derives ``name + suffix`` names a file that does
not exist next to any real archive.

``core.archive_reader.archive_sidecar`` is the ONE home; the Colab transport,
the Kaggle kernel templates, the fetched-output reader and the laya lane must
all resolve through it.
"""
from __future__ import annotations

from core.portable_archive import ByteCount
import inspect
from pathlib import Path

import pytest

from core.archive_reader import ARCHIVE_ENDINGS, archive_sidecar, tar_archive


def test_archive_sidecar_strips_the_declared_ending():
    assert ARCHIVE_ENDINGS == ('.tar.zst', '.zip')
    assert archive_sidecar('run.tar.zst', '.size') == Path('run.size')
    assert archive_sidecar(Path('/x/y/items.zip'), '.publication') == \
        Path('/x/y/items.publication')
    # Only the DECLARED endings are stripped; any other suffix is replaced by
    # the companion suffix (``Path.with_suffix``), and a bare name is appended.
    assert archive_sidecar('value.json', '.json') == Path('value.json')
    assert archive_sidecar('value', '.size') == Path('value.size')


def test_real_result_archive_companions_are_the_stripped_shape():
    """Against the real archives in this workspace.

    The rule must name the companion that actually exists, that companion must
    NOT be the ``.tar.zst.size`` append shape, and its content must be the
    archive's own size.
    """
    results = Path(__file__).resolve().parents[1] / 'results' / 'model_tracks'
    pairs = []
    for archive in sorted(results.glob('*.tar.zst')):
        companion = archive_sidecar(archive, '.size')
        if companion.is_file():
            pairs.append((archive, companion))
    if not pairs:
        pytest.skip('no result archive with a companion on disk in this checkout')
    for archive, companion in pairs:
        assert not archive.with_name(archive.name + '.size').exists(), (
            f'the append shape must not exist beside {archive.name}')
        assert companion.read_text().strip() == \
            ByteCount(archive.read_bytes()).total, \
            f'{companion.name} is not the size of {archive.name}'


def test_transport_writer_and_reader_agree_on_the_one_rule(tmp_path):
    """The Colab crossing's token lands on the rule, and is read back there."""
    from cli.colab_bundle_transport import (
        digest_sidecar, record_digest_script, transport_archive_size,
    )

    archive = tmp_path / 'bundle_delivery.tar.zst'
    member = tmp_path / 'member.txt'
    member.write_text('delivered bytes\n', encoding='utf-8')
    with tar_archive(archive, 'w') as tar:
        tar.add(member, arcname='training_prep/member.txt')
    token = str(archive.stat().st_size)

    # digest_sidecar IS the one rule (no second append implementation).
    assert digest_sidecar(archive).name == 'bundle_delivery.size'
    assert digest_sidecar(archive) == archive_sidecar(archive, '.size')

    # The emitted remote source writes the token through the same rule.
    exec(record_digest_script('delivery', label='bundle'),
         {'delivery': str(archive)})
    assert digest_sidecar(archive).read_text().strip() == token
    assert transport_archive_size(archive) == int(token)


def test_kaggle_kernel_scripts_resolve_companions_through_the_shared_rule():
    """No emitted kernel writes a digest companion any more.

    The size token travels in the manifest the fetch reads, so the
    digest-style sidecar write is gone from every kernel source.
    """
    from cli.kaggle_kernel_templates import KernelTemplates
    from cli.kaggle_kernels import TRAIN_RESULT_BUNDLE_SHIP

    for source in (KernelTemplates.TRAIN_KERNEL_SHARED,
                   KernelTemplates.FINALIZE_KERNEL_BODY,
                   TRAIN_RESULT_BUNDLE_SHIP):
        # the retired append shape: ``name + <a suffix>`` (never re-spelled)
        assert ' + LANE["files"]["sidecar_suffix"]' not in source
        # and no digest token is written beside the archive
        assert '.sha' + '256' not in source, source[:80]


def test_fetched_output_reader_resolves_the_companion_through_the_shared_rule():
    """The fetched-output reader resolves its archive through the SSOT identity."""
    from cli.kaggle_outputs import KaggleOutputs

    source = inspect.getsource(KaggleOutputs.fetch_kernel_output)
    assert 'lane.kernel_identity(kind, spec)' in source
    # no digest companion is read back at the fetch boundary
    assert '.sha' + '256' not in source
    assert 'archive_name + spec.files' not in source
