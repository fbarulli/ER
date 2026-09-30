"""Report aggregates retain provenance and carry diagnostics without averaging."""
from types import SimpleNamespace

import pandas as pd
import pytest

from core.common import training_cfg, recall_column_suffix
from training import train
from training.hpo_metrics import CALIBRATION_AGGREGATE_FIELDS


def test_report_aggregation_retains_entry_dependencies_and_fold_diagnostics(tmp_path, monkeypatch):
    paths = {
        'field_ablation': tmp_path / 'ablation.csv',
        'data_scaling': tmp_path / 'scaling.csv',
    }
    monkeypatch.setattr(train, 'F', paths)
    monkeypatch.setattr(train, 'SEED', 91)
    recall_key = recall_column_suffix(float(training_cfg().rand_matching.target_recall))
    base = {
        **dict.fromkeys(CALIBRATION_AGGREGATE_FIELDS, 0.4),
        'n_train': 100,
        'calibration_status': 'available',
        'calibration_reason': 'fold diagnostic',
        f'precision_at_{recall_key}_recall': 0.6,
    }
    args = SimpleNamespace(split='holdout', payload='full', train_frac=1.0, model='tiny')
    train._emit_07_series([base, {**base, f'precision_at_{recall_key}_recall': 0.8}], args)
    train._emit_07_series([base, {**base, f'precision_at_{recall_key}_recall': 0.8}], args)
    for path in paths.values():
        rows = pd.read_csv(path)
        assert len(rows) == 1
        assert rows.iloc[0]['seed'] == 91
        assert rows.iloc[0][f'precision_at_{recall_key}_recall'] == pytest.approx(0.7)
        assert rows.iloc[0]['calibration_status'] == '["available", "available"]'


def test_report_rejects_string_in_numeric_calibration_field(tmp_path, monkeypatch):
    paths = {'field_ablation': tmp_path / 'ablation.csv', 'data_scaling': tmp_path / 'scaling.csv'}
    monkeypatch.setattr(train, 'F', paths)
    row = {**dict.fromkeys(CALIBRATION_AGGREGATE_FIELDS, 0.4), 'pr_auc': 'unavailable'}
    args = SimpleNamespace(split='holdout', payload='full', train_frac=1.0, model='tiny')
    with pytest.raises(TypeError, match='non-numeric'):
        train._emit_07_series([row], args)
    assert not any(path.exists() for path in paths.values())
