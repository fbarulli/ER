import pytest


def test_configures_and_rotates_credentials_without_subprocesses(tmp_path, monkeypatch):
    Config = pytest.importorskip('dvc.config').Config
    from training import dvc_store
    monkeypatch.setattr(dvc_store, '_run', lambda *_: pytest.fail('configuration launched subprocess'))
    monkeypatch.setenv('DVC_SITE_CACHE_DIR', str(tmp_path/'site-cache'))
    remote = 'https://dagshub.com/fbarulli/ER.dvc'
    dvc_store._configure_workspace(tmp_path, token='first-placeholder', remote=remote)
    dvc_store._configure_workspace(tmp_path, token='rotated-placeholder', remote=remote)
    config = Config(str(tmp_path/'.dvc'))
    assert config['remote']['dagshub']['password'] == 'rotated-placeholder'
    assert config['remote']['dagshub']['user'] == 'fbarulli'
    assert config['core']['remote'] == 'dagshub'
    assert 'placeholder' not in (tmp_path/'.dvc/config').read_text()
    assert (tmp_path/'.dvc/config.local').stat().st_mode & 0o777 == 0o600
