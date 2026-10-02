"""Compare items whose partners change, including items without any pairs."""
import importlib.util
import json
from pathlib import Path

import pandas as pd


def test_item_census_keeps_isolated_items_and_changed_partners(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / 'scripts/compare_item_pair_sets.py'
    spec = importlib.util.spec_from_file_location('item_pair_comparison', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / 'jev/rebuild_8'
    old = output / 'before/data'
    new = tmp_path / 'data'
    old.mkdir(parents=True)
    new.mkdir()
    records = [{'gtin': gtin, 'mode_brand': 'Acme', 'canonical': 'drink',
                'flavor_set': '[]', 'volume_set': '[330]', 'pack_set': '[1]',
                'attribute_consistency_flags': '[]'} for gtin in ['1', '2', '3', '4']]
    for directory in (old, new):
        pd.DataFrame(records).to_csv(directory / 'canonical_records.csv', index=False)
    pd.DataFrame([{'gtin1': '1', 'gtin2': '2', 'true_label': 1}]).to_csv(old / 'labeled_pairs.csv', index=False)
    pd.DataFrame([{'gtin1': '1', 'gtin2': '3', 'true_label': 1}]).to_csv(new / 'labeled_pairs.csv', index=False)
    for directory, decisions in [(old, ['proceed', 'fallback']), (new, ['fallback', 'proceed'])]:
        pd.DataFrame([{'gtin1': '1', 'gtin2': partner, 'gate_decision': decision}
                      for partner, decision in zip(['2', '3'], decisions)]).to_csv(directory / 'gate_results.csv', index=False)
    (tmp_path / 'jev/sample_ledger.json').write_text(json.dumps([
        {'round': 1, 'status': 'tested', 'pairs': [['1', '2']]}
    ]))
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'OUTPUT', output)
    module.main()
    summary = json.loads((output / 'same_items_summary.json').read_text())
    assert summary['items_compared'] == 4
    assert summary['items_with_changed_labeled_partner_sets'] == 3
    assert summary['items_losing_all_eligible_partners'] == 1
    partners = {row['gtin']: row for row in map(json.loads, (output / 'all_item_partner_sets.jsonl').read_text().splitlines())}
    assert partners['1']['old_positive_partners'] == ['2']
    assert partners['1']['new_positive_partners'] == ['3']
    assert partners['4']['old_positive_partners'] == partners['4']['new_positive_partners'] == []
