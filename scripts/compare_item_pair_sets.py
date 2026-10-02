"""Compare every item's old and rebuilt partner sets, including isolated items."""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'jev/rebuild_8'


def labels(path):
    neighbors = defaultdict(dict)
    pairs = {}
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            a, b, label = row['gtin1'], row['gtin2'], int(row['true_label'])
            pair = tuple(sorted((a, b)))
            assert a != b and pair not in pairs
            pairs[pair] = label
            neighbors[a][b] = label
            neighbors[b][a] = label
    return neighbors, pairs


def gates(path):
    neighbors = defaultdict(Counter)
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            for item in ('gtin1', 'gtin2'):
                neighbors[row[item]][row['gate_decision']] += 1
    return neighbors


def main(argv=None):
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args([] if argv is None else argv)
    OUTPUT = args.output.resolve()
    old_path = OUTPUT / 'before/data'
    old_records = pd.read_csv(old_path / 'canonical_records.csv', dtype=str, keep_default_na=False).set_index('gtin')
    new_records = pd.read_csv(ROOT / 'data/canonical_records.csv', dtype=str, keep_default_na=False).set_index('gtin')
    assert old_records.index.is_unique and new_records.index.is_unique
    old_items, new_items = set(old_records.index), set(new_records.index)
    assert old_items == new_items, 'Item census changed; do not silently substitute items.'
    old_neighbors, old_pairs = labels(old_path / 'labeled_pairs.csv')
    new_neighbors, new_pairs = labels(ROOT / 'data/labeled_pairs.csv')
    old_gates = gates(old_path / 'gate_results.csv')
    new_gates = gates(ROOT / 'data/gate_results.csv')
    rows = []
    partner_records = []
    for item in sorted(old_items):
        old, new = old_neighbors[item], new_neighbors[item]
        old_pos = {partner for partner, label in old.items() if label == 1}
        new_pos = {partner for partner, label in new.items() if label == 1}
        old_neg = set(old) - old_pos
        new_neg = set(new) - new_pos
        retained = set(old) & set(new)
        relabeled = {partner for partner in retained if old[partner] != new[partner]}
        added, removed = set(new) - set(old), set(old) - set(new)
        row = {'gtin': item, 'brand': new_records.loc[item, 'mode_brand'],
               'old_positive_partners': len(old_pos), 'new_positive_partners': len(new_pos),
               'old_negative_partners': len(old_neg), 'new_negative_partners': len(new_neg),
               'retained_pair_ids': len(retained), 'added_pair_ids': len(added),
               'removed_pair_ids': len(removed), 'relabeled_retained_pairs': len(relabeled),
               'lost_positive_partners': len(old_pos - new_pos), 'gained_positive_partners': len(new_pos - old_pos),
               'lost_negative_partners': len(old_neg - new_neg), 'gained_negative_partners': len(new_neg - old_neg),
               'pair_set_changed': bool(added or removed or relabeled),
               'old_no_eligible_partners': not old, 'new_no_eligible_partners': not new}
        for decision in ('proceed', 'fallback', 'hard_no'):
            row[f'old_candidate_{decision}_partners'] = old_gates[item][decision]
            row[f'new_candidate_{decision}_partners'] = new_gates[item][decision]
        for field in ('canonical', 'flavor_set', 'volume_set', 'pack_set', 'attribute_consistency_flags'):
            row[f'{field}_changed'] = old_records.loc[item, field] != new_records.loc[item, field]
        rows.append(row)
        partner_records.append({'gtin': item,
                                'old_positive_partners': sorted(old_pos), 'new_positive_partners': sorted(new_pos),
                                'old_negative_partners': sorted(old_neg), 'new_negative_partners': sorted(new_neg),
                                'added_pair_ids': sorted(added), 'removed_pair_ids': sorted(removed),
                                'relabeled_partners': [{'gtin': partner, 'old_label': old[partner], 'new_label': new[partner]}
                                                       for partner in sorted(relabeled)]})
    frame = pd.DataFrame(rows)
    assert len(frame) == len(new_items)
    assert frame.old_positive_partners.sum() == 2 * sum(label == 1 for label in old_pairs.values())
    assert frame.new_positive_partners.sum() == 2 * sum(label == 1 for label in new_pairs.values())
    assert frame.added_pair_ids.sum() == 2 * len(set(new_pairs) - set(old_pairs))
    assert frame.removed_pair_ids.sum() == 2 * len(set(old_pairs) - set(new_pairs))
    frame.to_csv(OUTPUT / 'all_item_pair_changes.csv', index=False)
    with (OUTPUT / 'all_item_partner_sets.jsonl').open('w') as stream:
        for row in partner_records:
            stream.write(json.dumps(row) + '\n')
    previous = []
    for entry in json.loads((ROOT / 'jev/sample_ledger.json').read_text()):
        if entry['status'] != 'tested':
            continue
        items = {gtin for pair in entry['pairs'] for gtin in pair}
        subset = frame[frame.gtin.isin(items)]
        previous.append({'round': entry['round'], 'historical_items': len(items), 'items_in_fixed_census': len(subset),
                         'items_with_changed_partner_sets': int(subset.pair_set_changed.sum()),
                         'old_positive_partner_links': int(subset.old_positive_partners.sum()),
                         'new_positive_partner_links': int(subset.new_positive_partners.sum()),
                         'old_negative_partner_links': int(subset.old_negative_partners.sum()),
                         'new_negative_partner_links': int(subset.new_negative_partners.sum()),
                         'items_without_new_eligible_partners': int(subset.new_no_eligible_partners.sum())})
    report = {'comparison_unit': 'item with its old and new partner sets',
              'items_compared': len(frame), 'items_added': len(new_items - old_items), 'items_removed': len(old_items - new_items),
              'items_with_changed_labeled_partner_sets': int(frame.pair_set_changed.sum()),
              'items_losing_positive_partners': int((frame.lost_positive_partners > 0).sum()),
              'items_gaining_positive_partners': int((frame.gained_positive_partners > 0).sum()),
              'items_with_no_old_eligible_partners': int(frame.old_no_eligible_partners.sum()),
              'items_with_no_new_eligible_partners': int(frame.new_no_eligible_partners.sum()),
              'items_losing_all_eligible_partners': int((~frame.old_no_eligible_partners & frame.new_no_eligible_partners).sum()),
              'previous_run_item_cohorts': previous,
              'current_artifact_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ['data/canonical_records.csv', 'data/gate_results.csv', 'data/labeled_pairs.csv']},
              'meaning': 'Complete canonical-item census, including items with no candidates or labels. Positive and negative partners follow each snapshot\'s configured gate-derived labeled population. Partner-link counts count both endpoints and are not independent pair counts. Historical run cohorts reuse their item IDs, then obtain partners from each complete old/new catalog; saved pair judgments are not reassigned to new partners.'}
    (OUTPUT / 'same_items_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
