import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from model_tracks import post_training_ablation as auto


def test_automatic_suite_ablation_refuses_to_reopen_gpu_without_saved_exports(tmp_path,monkeypatch):
    """``run`` opens the result archive once through ``Bundle.load`` (role result)."""
    from core.bundle import Bundle, BundleRole
    opened = []
    monkeypatch.setattr(
        Bundle, 'load',
        classmethod(lambda cls, path, role, **kwargs: opened.append((path, role))))
    suite = SimpleNamespace(publish_git=False)
    def forbidden(*args,**kwargs):
        pytest.fail('posttraining must not reopen GPU')
    with pytest.raises(ValueError,match='staged GPU ablation exports'):
        auto.run(tmp_path/'completed.zip','run',suite,launcher=forbidden)
    assert opened == [(tmp_path/'completed.zip', BundleRole.result)]
    # no stray archive-verify helper lives on this module: the boundary load by
    # role is the only archive read.
    assert not hasattr(auto, 'verify_archive')
    assert not hasattr(auto, 'read_archive_manifest')


def test_run_reuses_a_supplied_boundary_handle_without_reopening(tmp_path,monkeypatch):
    """A supplied ``bundle`` handle skips the boundary load entirely."""
    from core.bundle import Bundle
    def explode(*args,**kwargs):
        raise AssertionError('a supplied boundary handle must not be re-opened')
    monkeypatch.setattr(Bundle, 'load', classmethod(explode))
    with pytest.raises(ValueError,match='staged GPU ablation exports'):
        auto.run(tmp_path/'completed.zip','run',SimpleNamespace(publish_git=False),
                 bundle=object())
