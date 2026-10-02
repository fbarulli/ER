import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location('jev_rebuild_sample', Path(__file__).parents[1] / 'jev/rebuild_sample.py')
sampler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampler)


def test_capacity_redistribution_and_insufficient_population():
    import pytest
    assert sampler.allocate({'a': 1, 'b': 8, 'c': 0}, 7) == {'a': 1, 'b': 6, 'c': 0}
    with pytest.raises(ValueError, match='only 9 available'):
        sampler.allocate({'a': 1, 'b': 8}, 10)


def test_sampling_balances_labels_and_is_independent_of_csv_order():
    rows = [{'gtin1': str(i), 'gtin2': str(i + 100), 'training_label': label,
             'stratum': f'label_{label}|{gate}'}
            for label, gate, start, count in [(0, 'hard_no', 0, 20), (1, 'proceed', 20, 10), (1, 'fallback', 30, 1)]
            for i in range(start, start + count)]
    selected, allocation = sampler.select_sample(rows, 10, 47, .5)
    again, _ = sampler.select_sample(list(reversed(rows)), 10, 47, .5)
    assert selected == again
    assert sum(row['training_label'] for row in selected) == 5
    assert allocation['label_1|fallback']['selected'] == 1
    assert allocation['label_1|proceed']['selected'] == 4
    assert allocation['label_0|hard_no']['inclusion_probability'] == .25


def test_exclusions_include_failed_calls_controls_and_reverse_ledger_pairs(tmp_path):
    (tmp_path / 'sample_ledger.json').write_text(json.dumps([{'pairs': [['2', '1']]}]))
    (tmp_path / 'sample_control_5.json').write_text(json.dumps([{'gtin1': '3', 'gtin2': '4'}]))
    (tmp_path / 'audit_results_7.jsonl').write_text(json.dumps({'gtin1': '6', 'gtin2': '5', 'status': 'error'}) + '\n')
    assert sampler.reserved_pairs(tmp_path) == {('1', '2'), ('3', '4'), ('5', '6')}


def test_diagnostic_sampling_keeps_missing_training_labels_and_is_order_independent():
    rows = [{'gtin1': str(i), 'gtin2': str(i + 100), 'training_label': None,
             'stratum': 'review' if i < 2 else 'reject'} for i in range(10)]
    selected, allocations = sampler.select_cells(rows, 6, 48)
    reverse, _ = sampler.select_cells(list(reversed(rows)), 6, 48)
    assert selected == reverse
    assert len(selected) == 6
    assert all(row['training_label'] is None for row in selected)
    assert allocations['review']['selected'] == 2
    assert allocations['reject']['selected'] == 4
