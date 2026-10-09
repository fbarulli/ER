"""Runtime proofs for config-owned Colab and evaluation surfaces."""
import builtins
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml
from pydantic import ValidationError

from core import common
from graph_tracks.config import GraphConfig, TextConfig, load_text_config

ROOT = Path(__file__).resolve().parents[1]


def test_all_lane_defaults_follow_retuned_ann_ssot(monkeypatch):
    monkeypatch.setattr(common, 'ann_retrieval_ks', lambda: (1, 3, 17))
    for filename in ('graph_tracks_gnn.yaml', 'graph_tracks_cascade.yaml'):
        cfg = GraphConfig.model_validate(yaml.safe_load((ROOT/'config'/filename).read_text()))
        assert cfg.retrieval_ks == [1, 3, 17]
    assert load_text_config(ROOT/'config/text_track.yaml').retrieval_ks == [1, 3, 17]
    assert TextConfig(track='text', output_dir='reports', retrieval_ks=[1, 7]).retrieval_ks == [1, 7]


def test_ann_default_does_not_hide_import_failures(monkeypatch):
    real_import = builtins.__import__
    def fail(name, *args, **kwargs):
        if name == 'core.common':
            raise ImportError('broken core dependency')
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', fail)
    with pytest.raises(ImportError, match='broken core dependency'):
        TextConfig(track='text', output_dir='reports')
    # A standalone package may use its explicitly materialized ladder.
    assert TextConfig(track='text', output_dir='reports', retrieval_ks=[1, 7]).retrieval_ks == [1, 7]


@pytest.mark.parametrize('updates', [dict(retrieval_ks=[]), dict(retrieval_ks=[0]),
                                    dict(retrieval_ks=[1, 1]), dict(hnsw_m=1),
                                    dict(track='gnn_only'), dict(unexpected=True)])
def test_text_config_rejects_invalid_contract(tmp_path, updates):
    path = tmp_path/'text.yaml'
    settings = dict(track='text', output_dir='reports')
    settings.update(updates)
    path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValidationError):
        load_text_config(path)


def test_text_missing_config_fails_before_checkpoint_work(tmp_path, monkeypatch):
    from model_tracks.text_report import complete
    from training import validation_inference
    resolver = Mock(side_effect=AssertionError('checkpoint work must not start'))
    monkeypatch.setattr(validation_inference, 'resolve_best_checkpoint', resolver)
    with pytest.raises(FileNotFoundError, match='text lane config'):
        complete(tmp_path/'output', tmp_path, device='cpu', report_test=False)
    resolver.assert_not_called()
    assert not (tmp_path/'output').exists()


def test_colab_upload_uses_configured_retries_and_backoff(monkeypatch, tmp_path):
    import subprocess
    from cli import colab
    cfg = common.training_cfg().colab
    assert colab._INCREMENTAL_SYNC_SECONDS == cfg.incremental_sync_seconds
    assert colab._CHECKPOINT_MANIFEST_NAME == cfg.checkpoint_manifest_name
    assert colab._LATEST_BEST_MARKER == cfg.latest_best_marker
    monkeypatch.setattr(colab, '_REMOTE_UPLOAD_RETRIES', 3)
    monkeypatch.setattr(colab, '_COLAB', cfg.model_copy(update={
        'remote_upload_backoff_seconds': 2, 'remote_upload_max_backoff_seconds': 3}))
    upload = Mock(side_effect=[subprocess.TimeoutExpired('upload', 1),
                              subprocess.CalledProcessError(1, 'upload'), None])
    sleep = Mock()
    monkeypatch.setattr(colab, 'colab', upload)
    monkeypatch.setattr(colab, 'run_colab_exec_stream', Mock())
    monkeypatch.setattr(colab.time, 'sleep', sleep)
    colab._upload_with_retries(tmp_path/'input.zip', '/remote/input.zip', timeout=1)
    assert upload.call_count == 3
    assert [call.args[0] for call in sleep.call_args_list] == [2, 3]


@pytest.mark.parametrize('name', ['../manifest.json', '/tmp/manifest.json', r'a\b', '..'])
def test_colab_artifact_names_are_validated(name):
    from core.schemas import ColabSpec
    settings = common.training_cfg().colab.model_dump()
    settings['checkpoint_manifest_name'] = name
    with pytest.raises(ValidationError, match='basenames'):
        ColabSpec.model_validate(settings)


def test_report_operating_columns_do_not_confuse_ranking_or_recall_targets():
    from training.generate_training_report import _metric_column
    columns = ['precision_at_1', 'precision_at_90pct_recall', 'precision_at_0.62']
    assert _metric_column(columns, 'precision_at_') == 'precision_at_0.62'
    assert _metric_column(columns[:2], 'precision_at_') is None
    with pytest.raises(ValueError, match='ambiguous'):
        _metric_column(columns + ['precision_at_0.55'], 'precision_at_')
