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
    # the stray verify_archive spelling is gone: one boundary load, by role
    assert not hasattr(auto, 'verify_archive')


def test_run_reuses_a_supplied_boundary_handle_without_reopening(tmp_path,monkeypatch):
    """A supplied ``bundle`` handle skips the boundary load entirely."""
    from core.bundle import Bundle
    def explode(*args,**kwargs):
        raise AssertionError('a supplied boundary handle must not be re-opened')
    monkeypatch.setattr(Bundle, 'load', classmethod(explode))
    with pytest.raises(ValueError,match='staged GPU ablation exports'):
        auto.run(tmp_path/'completed.zip','run',SimpleNamespace(publish_git=False),
                 bundle=object())


def test_saved_ablation_rejects_calibration_for_other_checkpoint(tmp_path):
    from model_tracks.ablation import write
    track = tmp_path/'text'
    folder = track/'ablation';folder.mkdir(parents=True)
    checkpoint = track/'checkpoint';checkpoint.mkdir();(checkpoint/'weights').write_bytes(b'selected')
    write(folder/'request.json',{'checkpoint':str(checkpoint)})
    (folder/'vectors.npz').write_bytes(b'exports')
    # The calibration manifest carries the threshold at the top level; that is
    # the shape every report producer writes and the one complete_saved reads.
    # A full calibrated report whose checkpoint identity misses the selected one.
    from graph_tracks.report_manifest import build as build_manifest
    write(track/'text__completion_manifest.json', build_manifest(
        track='text', checkpoint=str(checkpoint), checkpoint_size='f' * 64,
        listings_size='1' * 64, pairs_size='2' * 64,
        threshold=.5, threshold_source='dev_youden', test_reported=False,
        model_selection='dev_pr_auc', retrieval_ks=[10]))
    with pytest.raises(ValueError,match='calibration differs'):
        auto.complete_saved(tmp_path,SimpleNamespace(ablation_config='unused'))
    assert not (folder/'baseline_threshold.json').exists()
