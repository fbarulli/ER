"""The bundle data-only install never clobbers the pinned checkout.

Regression class (owner order 2026-10-07): a prepared bundle built earlier
embedded a stale ``config/paths.yaml`` without the ``source_code_dir`` layout
key; extracting it over the freshly cloned pinned checkout replaced the
checkout's config and crashed with ``KeyError: 'source_code_dir'``.
``install_data_members`` installs bundle DATA only and leaves
``src/``/``config/``/``scripts/`` to the checkout.
"""
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tarfile
import zipfile

import pytest

from core.archive_reader import TarReader, open_archive
from core.portable_archive import (
    CHECKOUT_AUTHORITATIVE_PREFIXES,
    install_data_members,
    is_checkout_authoritative,
)

MEMBER_FILES = {
    'src/model_tracks/run.py': '# stale bundle code\n',
    'config/paths.yaml': 'layouts: {}\n',
    'scripts/diet_manifest.py': '# stale bundle script\n',
    'data/model_tracks/setup/inputs.csv': 'bundle data\n',
    'artifacts/evidence/semantics/family_registry.json': '{}\n',
}
MEMBER_DIRS = ('data/model_tracks/shared/',)
EXPECTED_INSTALLED = [name for name in MEMBER_FILES
                      if not is_checkout_authoritative(name)]


def _zip_archive(path: Path) -> zipfile.ZipFile:
    with zipfile.ZipFile(path, 'w') as archive:
        for name in MEMBER_DIRS:
            archive.writestr(name, '')
        for name, content in MEMBER_FILES.items():
            archive.writestr(name, content)
    return zipfile.ZipFile(path)


def _tar_archive(path: Path) -> TarReader:
    with tarfile.open(path, 'w') as archive:
        for name in MEMBER_DIRS:
            info = tarfile.TarInfo(name)
            info.type = tarfile.DIRTYPE
            archive.addfile(info)
        for name, content in MEMBER_FILES.items():
            payload = content.encode()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return TarReader(tarfile.open(path), SimpleNamespace(copy_buffer_bytes=65536))


def _close(archive) -> None:
    closer = getattr(archive, 'close', None) or getattr(archive.archive, 'close', None)
    closer()


@pytest.mark.parametrize('builder', [_zip_archive, _tar_archive])
def test_install_data_members_never_clobbers_checkout(tmp_path, builder):
    (tmp_path / 'src/model_tracks').mkdir(parents=True)
    (tmp_path / 'config').mkdir()
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'src/model_tracks/run.py').write_text('# pinned checkout code\n')
    (tmp_path / 'config/paths.yaml').write_text('source_code_dir: pinned\n')
    (tmp_path / 'scripts/diet_manifest.py').write_text('# pinned checkout script\n')

    archive = builder(tmp_path / 'bundle')
    try:
        installed = install_data_members(archive, tmp_path)
    finally:
        _close(archive)

    assert (tmp_path / 'src/model_tracks/run.py').read_text() == '# pinned checkout code\n'
    assert (tmp_path / 'config/paths.yaml').read_text() == 'source_code_dir: pinned\n'
    assert (tmp_path / 'scripts/diet_manifest.py').read_text() == '# pinned checkout script\n'
    assert (tmp_path / 'data/model_tracks/setup/inputs.csv').read_text() == 'bundle data\n'
    assert (tmp_path / 'artifacts/evidence/semantics/family_registry.json').read_text() == '{}\n'
    assert (tmp_path / 'data/model_tracks/shared').is_dir()
    assert sorted(installed) == sorted(EXPECTED_INSTALLED)


def test_checkout_authoritative_predicate():
    assert CHECKOUT_AUTHORITATIVE_PREFIXES == ('src/', 'config/', 'scripts/')
    for name in ('src', 'src/', 'src/a.py', 'config', 'config/paths.yaml',
                 'scripts', 'scripts/x.py'):
        assert is_checkout_authoritative(name), name
    for name in ('data/x.csv', 'artifacts/e.json', 'source.py', 'scriptsx.py',
                 'configs/paths.yaml', 'srcx/mod.py'):
        assert not is_checkout_authoritative(name), name


# ── the sealed-archive inventory key is config-owned (audit 2026-10-08) ─────
# portable_archive used to default inventory_key='files' while NO caller passed
# the config value, so a renamed bundle.files_key would have the writer's
# manifest and the boundary verify disagree. The default now resolves through
# inventory_key_home(), so one config value steers seal AND verify.

def test_archive_inventory_key_follows_the_config(monkeypatch, tmp_path):
    from core import common
    from core.bundle import Bundle, BundleRole
    from core.portable_archive import inventory_key_home, verify_archive

    monkeypatch.setattr(common.training_cfg().bundle, 'files_key', 'custom_inventory')
    assert inventory_key_home() == 'custom_inventory'
    source = tmp_path / 'a.txt'
    source.write_text('hello')
    output = tmp_path / 'sealed.tar.zst'
    Bundle.seal_archive(output, {'a.txt': source}, role=BundleRole.inputs,
                        manifest_name='m.json')
    with open_archive(output) as archive:
        manifest = json.loads(archive.read('m.json'))
    assert set(manifest) == {'custom_inventory'}
    assert 'files' not in manifest
    metadata = verify_archive(output, 'm.json')
    assert 'custom_inventory' in metadata
    loaded = Bundle.load(output, 'inputs', manifest_name='m.json')
    assert 'custom_inventory' in loaded.manifest


def test_the_committed_inventory_key_default_is_files():
    from core.common import training_cfg
    from core.portable_archive import inventory_key_home

    assert inventory_key_home() == training_cfg().bundle.files_key
