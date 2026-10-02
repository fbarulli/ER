"""Compare rebuilt extraction and gate artifacts with a preserved snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round', type=int, default=8)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    directory = args.output.resolve() if args.output else ROOT / 'jev' / f'rebuild_{args.round}'
    before = directory / 'before'
    old_gate = read(before / 'data/gate_results.csv')
    new_gate = read(ROOT / 'data/gate_results.csv')
    keys = ['gtin1', 'gtin2']
    assert not old_gate.duplicated(keys).any()
    assert not new_gate.duplicated(keys).any()
    paired = old_gate.merge(new_gate, on=keys, how='outer', suffixes=('_old', '_new'), indicator=True, validate='one_to_one')
    shared = paired[paired['_merge'] == 'both'].copy()
    transitions = Counter(shared.gate_decision_old + '->' + shared.gate_decision_new)
    changed = shared[shared.gate_decision_old != shared.gate_decision_new]
    changed.to_csv(directory / 'gate_transitions.csv', index=False)
    old_records = read(before / 'data/canonical_records.csv')
    new_records = read(ROOT / 'data/canonical_records.csv')
    assert not new_records.gtin.duplicated().any()
    record_delta = old_records.merge(new_records, on='gtin', suffixes=('_old', '_new'), validate='one_to_one')
    fields = ['canonical', 'volume_set', 'pack_set', 'flavor_set', 'attribute_consistency_flags']
    changed_fields = {field: int((record_delta[f'{field}_old'] != record_delta[f'{field}_new']).sum())
                      for field in fields if f'{field}_old' in record_delta and f'{field}_new' in record_delta}
    old_labels = read(before / 'data/labeled_pairs.csv')
    new_labels = read(ROOT / 'data/labeled_pairs.csv')
    assert set(new_labels.true_label) <= {'0', '1'}
    label_delta = old_labels.merge(new_labels, on=keys, how='outer', suffixes=('_old', '_new'), indicator=True, validate='one_to_one')
    label_changes = label_delta[(label_delta['_merge'] != 'both') | (label_delta.true_label_old != label_delta.true_label_new)]
    label_changes.to_csv(directory / 'label_population_changes.csv', index=False)
    # Labels must follow the regenerated gate and configured similarity cuts.
    from core.common import training_cfg
    cfg = training_cfg().pairs
    joined = new_labels.merge(new_gate, on=keys, validate='one_to_one')
    assert len(joined) == len(new_labels)
    positive = joined.true_label == '1'
    assert (joined.loc[positive, 'gate_decision'] == 'proceed').all()
    assert (joined.loc[~positive, 'gate_decision'] == 'hard_no').all()
    assert (joined.loc[positive, 'similarity'].astype(float) >= cfg.proceed_sim_threshold).all()
    assert (joined.loc[~positive, 'similarity'].astype(float) >= cfg.hardneg_sim_threshold).all()
    ids = set(new_records.gtin)
    assert set(new_gate.gtin1) | set(new_gate.gtin2) <= ids
    audited = [json.loads(line) for line in (ROOT / 'jev/audit_results_7.jsonl').read_text().splitlines() if line.strip()]
    audited = [row for row in audited if row['status'] == 'ok' and row['copy'] == 'a_order']
    current = {tuple(sorted((row.gtin1, row.gtin2))): row.gate_decision for row in new_gate.itertuples()}
    saved_jev = Counter()
    for row in audited:
        decision = current.get(tuple(sorted((row['gtin1'], row['gtin2']))), 'outside_rebuilt_census')
        score = row['noul']
        band = 'different' if score <= .2 else 'same' if score >= .8 else 'uncertain'
        saved_jev[f'{decision}|{band}'] += 1
    report = {
        'round': args.round,
        'gate_census': {'total_pairs': len(new_gate), **new_gate.gate_decision.value_counts().to_dict()},
        'gate_pair_population': paired['_merge'].value_counts().to_dict(),
        'gate_transitions': dict(transitions),
        'changed_gate_pairs': len(changed),
        'old_canonical_records': len(old_records), 'new_canonical_records': len(new_records),
        'changed_canonical_fields': changed_fields,
        'old_labels': old_labels.true_label.value_counts().to_dict(),
        'new_labels': new_labels.true_label.value_counts().to_dict(),
        'label_population_changes': label_delta['_merge'].value_counts().to_dict(),
        'retained_pairs_changed_label': int(((label_delta['_merge'] == 'both') & (label_delta.true_label_old != label_delta.true_label_new)).sum()),
        'round_7_saved_jev_bands_by_rebuilt_gate': dict(saved_jev),
        'source_sha256': {name: digest(ROOT / name) for name in ['dataset.csv', 'data/canonical_records.csv', 'data/gate_results.csv', 'data/labeled_pairs.csv']},
        'meaning': 'Rebuilt gate-derived training labels, not new human or JEV truth labels. Previous CSVs remain in before/. Stored retailer descriptions are short captured fields. Existing prepared training bundles require fresh preparation before training.',
    }
    (directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
