"""Stage a fresh, snapshot-bound JEV audit of rebuilt labeled pairs, offline."""
from __future__ import annotations

import argparse
import csv
from core.portable_archive import ByteCount
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from core.project_root import find_project_root

ROOT = find_project_root(Path(__file__))


def pair_key(row):
    return tuple(sorted((row['gtin1'], row['gtin2'])))


def file_size(path) -> int:
    return ByteCount(path.read_bytes()).total


def reserved_pairs(directory):
    """Include unsuccessful/staged calls, controls, and ledger reservations."""
    excluded = set()
    ledger = directory / 'sample_ledger.json'
    if ledger.exists():
        for entry in json.loads(ledger.read_text()):
            excluded.update(tuple(sorted(pair)) for pair in entry['pairs'])
    for pattern in ('sample_doubled*.json', 'sample_control*.json', 'audit_results*.jsonl'):
        for path in sorted(directory.glob(pattern)):
            rows = (json.loads(line) for line in path.read_text().splitlines() if line.strip()) if path.suffix == '.jsonl' else json.loads(path.read_text())
            excluded.update(pair_key(row) for row in rows)
    return excluded


def allocate(capacities, total):
    """Equal allocation over populated cells; redistribute exhausted capacity."""
    if total < 0 or total > sum(capacities.values()):
        raise ValueError(f'requested {total} pairs, only {sum(capacities.values())} available')
    counts = dict.fromkeys(sorted(capacities), 0)
    while total:
        for key in counts:
            if counts[key] < capacities[key]:
                counts[key] += 1
                total -= 1
                if not total:
                    break
    return counts


def select_sample(rows, count, seed, positive_fraction=.8):
    """Enrich positives, then balance current gate × similarity cells per label."""
    buckets = defaultdict(list)
    for row in rows:
        buckets[row['stratum']].append(row)
    label_sizes = Counter(row['training_label'] for row in rows)
    if count > len(rows):
        raise ValueError(f'requested {count} pairs, only {len(rows)} available')
    positive_target = min(round(count * positive_fraction), label_sizes[1])
    negative_target = min(count - positive_target, label_sizes[0])
    positive_target = min(count - negative_target, label_sizes[1])
    label_targets = {0: negative_target, 1: positive_target}
    allocations = {}
    rng = random.Random(seed)
    selected = []
    for label, target in label_targets.items():
        capacities = {key: len(pool) for key, pool in buckets.items() if pool[0]['training_label'] == label}
        for key, n in allocate(capacities, target).items():
            pool = sorted(buckets[key], key=pair_key)
            selected.extend(rng.sample(pool, n))
            allocations[key] = {'eligible_population': len(pool), 'selected': n,
                                'inclusion_probability': n / len(pool)}
    return selected, allocations


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(clean(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def select_cells(rows, count, seed):
    """Balance diagnostic cells without inventing a binary training label."""
    buckets = defaultdict(list)
    for row in rows:
        buckets[row['stratum']].append(row)
    rng = random.Random(seed)
    selected, allocations = [], {}
    for cell, n in allocate({cell: len(pool) for cell, pool in buckets.items()}, count).items():
        pool = sorted(buckets[cell], key=pair_key)
        selected.extend(rng.sample(pool, n))
        allocations[cell] = {'eligible_population': len(pool), 'selected': n,
                             'inclusion_probability': n / len(pool)}
    return selected, allocations


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pairs-csv', type=Path, default=ROOT / 'data/labeled_pairs.csv')
    ap.add_argument('--gate-csv', type=Path, default=ROOT / 'data/gate_results.csv')
    ap.add_argument('--calls', type=int, default=1000, help='total staged API calls')
    ap.add_argument('--order-checks', type=int, default=100, help='extra swapped original-input calls')
    ap.add_argument('--positive-fraction', type=float, default=.8)
    ap.add_argument('--round', type=int, default=7)
    ap.add_argument('--seed', type=int, default=47)
    ap.add_argument('--previous-pairs-csv', type=Path, help='pre-rebuild labeled pairs for lost-positive checks')
    ap.add_argument('--removed-positive-checks', type=int, default=0)
    args = ap.parse_args()
    pair_count = args.calls - args.order_checks
    if pair_count < 1 or args.round < 1 or not 0 <= args.order_checks <= pair_count:
        ap.error('require positive round and calls, with 0 <= order-checks <= calls / 2')
    if not 0 <= args.positive_fraction <= 1:
        ap.error('positive-fraction must be between zero and one')
    if not 0 <= args.removed_positive_checks < pair_count or (args.removed_positive_checks and not args.previous_pairs_csv):
        ap.error('removed-positive-checks must be below unique pairs and requires previous-pairs-csv')
    directory = ROOT / 'jev'
    paths = {name: directory / filename for name, filename in {
        'sample': f'sample_doubled_{args.round}.json',
        'states': f'input_states_{args.round}.json',
        'summary': f'sample_{args.round}_summary.json',
        'checkpoint': f'audit_results_{args.round}.jsonl',
    }.items()}
    ledger_path = directory / 'sample_ledger.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    if any(path.exists() for path in paths.values()) or any(entry['round'] == args.round for entry in ledger):
        ap.error('round already reserved or started; choose a new round')
    source_paths = {'pairs': args.pairs_csv, 'gate': args.gate_csv,
                    'records': ROOT / 'data/canonical_records.csv'}
    if args.previous_pairs_csv:
        source_paths['previous_pairs'] = args.previous_pairs_csv
    hashes = {name: file_size(path) for name, path in source_paths.items()}
    excluded = reserved_pairs(directory)
    with args.gate_csv.open(newline='', encoding='utf-8') as handle:
        gates = {}
        for row in csv.DictReader(handle):
            key = pair_key(row)
            if key in gates:
                raise ValueError(f'duplicate unordered gate pair: {key}')
            gates[key] = row
    sys.path.insert(0, str(ROOT / 'src'))
    from training.gate_replay import canonical_records_from_csv
    from pipeline import three_way_gate
    records = canonical_records_from_csv()
    rows = []
    seen = {}
    accounting = Counter()
    population = Counter()
    with args.pairs_csv.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if not {'gtin1', 'gtin2', 'true_label'} <= set(reader.fieldnames or []):
            ap.error('pairs CSV must have gtin1, gtin2, true_label columns')
        for row in reader:
            accounting['input_rows'] += 1
            key = pair_key(row)
            label = int(row['true_label'])
            if label not in (0, 1) or key[0] == key[1]:
                raise ValueError(f'invalid labeled pair: {row}')
            if key in seen:
                if seen[key] != label:
                    raise ValueError(f'contradictory labels for {key}')
                accounting['duplicate_unordered_rows'] += 1
                continue
            seen[key] = label
            population[str(label)] += 1
            if key in excluded:
                accounting['previously_reserved_pairs'] += 1
                continue
            if key not in gates or any(gtin not in records for gtin in key):
                raise ValueError(f'pair missing from gate or canonical snapshot: {key}')
            old = gates[key]
            similarity = float(old['similarity'])
            if not math.isfinite(similarity) or not 0 <= similarity <= 1:
                raise ValueError(f'invalid similarity for {key}')
            current = three_way_gate(records[key[0]], records[key[1]])
            band = 'below_0.8' if similarity < .8 else '0.8_to_0.9' if similarity < .9 else '0.9_to_1.0'
            rows.append({'gtin1': key[0], 'gtin2': key[1], 'training_label': label,
                         'audit_cohort': 'current_training_pairs',
                         'gate': current['decision'], 'gate_reason': current['reason'],
                         'frozen_gate': old['gate_decision'], 'similarity': similarity,
                         'similarity_band': band,
                         'stratum': f'label_{label}|{current["decision"]}|{band}'})
            if len(rows) % 1000 == 0:
                print(f'replayed {len(rows)} fresh pairs', flush=True)
    diagnostics, diagnostic_allocations = [], {}
    diagnostic_population = []
    if args.previous_pairs_csv:
        with args.previous_pairs_csv.open(newline='', encoding='utf-8') as handle:
            prior_labels = {pair_key(row): int(row['true_label']) for row in csv.DictReader(handle)}
        for row in rows:
            row['old_training_label'] = prior_labels.get(pair_key(row))
        for key, old_label in sorted(prior_labels.items()):
            if old_label != 1 or seen.get(key) == 1 or key in excluded:
                continue
            if key not in gates or any(gtin not in records for gtin in key):
                accounting['lost_positive_outside_current_records'] += 1
                continue
            current = three_way_gate(records[key[0]], records[key[1]])
            similarity = float(gates[key]['similarity'])
            band = 'below_0.8' if similarity < .8 else '0.8_to_0.9' if similarity < .9 else '0.9_to_1.0'
            diagnostic_population.append({'gtin1': key[0], 'gtin2': key[1],
                'training_label': seen.get(key), 'old_training_label': 1,
                'audit_cohort': 'lost_positive_partners',
                'gate': current['decision'], 'gate_reason': current['reason'],
                'frozen_gate': gates[key]['gate_decision'], 'similarity': similarity,
                'similarity_band': band, 'stratum': f'lost_positive|{current["decision"]}|{band}'})
        diagnostics, diagnostic_allocations = select_cells(diagnostic_population, args.removed_positive_checks, args.seed + 2)
    diagnostic_keys = {pair_key(row) for row in diagnostics}
    training_population = [row for row in rows if pair_key(row) not in diagnostic_keys]
    selected, allocations = select_sample(training_population, pair_count - len(diagnostics), args.seed, args.positive_fraction)
    selected.extend(diagnostics)
    allocations.update(diagnostic_allocations)
    if diagnostics:
        checks, check_allocations = select_cells(selected, args.order_checks, args.seed + 1)
    else:
        checks, check_allocations = select_sample(selected, args.order_checks, args.seed + 1, args.positive_fraction)
    check_keys = {pair_key(row) for row in checks}
    states = {'original_data': {}}
    sample = []
    for row in selected:
        a, b = row['gtin1'], row['gtin2']
        reverse = three_way_gate(records[b], records[a])
        if reverse['decision'] != row['gate']:
            raise ValueError(f'gate is order-sensitive for {(a, b)}')
        for gtin in (a, b):
            record = records[gtin]
            listings = json.loads(record['source_rows'])
            if not listings:
                raise ValueError(f'no original source listings for {gtin}')
            states['original_data'][gtin] = {'gtin': gtin, 'listings': listings}
        sample.append({**row, 'input_scope': 'original_data', 'copy': 'a_order'})
        if pair_key(row) in check_keys:
            sample.append({**row, 'gtin1': b, 'gtin2': a, 'gate_reason': reverse['reason'],
                           'input_scope': 'original_data', 'copy': 'b_swapped'})
    if hashes != {name: file_size(path) for name, path in source_paths.items()}:
        raise ValueError('source changed while sampling; rerun with a stable snapshot')
    listings = [listing for state in states['original_data'].values() for listing in state['listings']]
    description_lengths = [len(str(listing.get('description_short_eng') or '')) for listing in listings]
    summary = {'strategy': 'positive-enriched, current-gate and similarity stratified random sampling',
               'seed': args.seed, 'requested_pairs': pair_count, 'requested_calls': args.calls,
               'positive_fraction': args.positive_fraction, 'order_check_allocations': check_allocations, 'unique_pairs': len(selected),
               'calls': len(sample), 'source_paths': {k: str(v) for k, v in source_paths.items()},
               'source_size': hashes, 'row_accounting': dict(accounting),
               'source_label_population': dict(population), 'eligible_pairs': len(rows),
               'current_training_pairs_staged': pair_count - len(diagnostics),
               'lost_positive_partner_checks': len(diagnostics),
               'lost_positive_eligible_population': len(diagnostic_population),
               'excluded_previously_staged_or_tested_pairs': len(excluded),
               'allocations': allocations,
               'gate_drift_eligible': dict(Counter(f'{r["frozen_gate"]}->{r["gate"]}' for r in rows)),
               'source_capture': {'listings': len(listings),
                                  'listings_with_description': sum(length > 0 for length in description_lengths),
                                  'maximum_description_characters': max(description_lengths, default=0),
                                  'description_field': 'description_short_eng preserved in full from the supplied complete dataset; no additional truncation'},
               'meaning': 'Training labels are gate-derived, not human ground truth. Full original listings for all pairs; stratified swapped-order subset. Primary estimates use a_order only; do not count swaps as independent pairs. Raw means are not prevalence estimates. Inclusion probabilities refer to fresh eligible pairs; cells with zero selections are not estimable.'}
    for name, value in [('states', states), ('sample', sample)]:
        with paths[name].open('x', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=1)
            handle.write('\n')
    summary['input_states_size'] = file_size(paths['states'])
    with paths['summary'].open('x', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
        handle.write('\n')
    ledger.append({'round': args.round, 'kind': 'rebuilt_labeled_pairs_original_audit',
                   'status': 'staged_not_tested', 'sample': str(paths['sample'].relative_to(ROOT)),
                   'sample_size': file_size(paths['sample']),
                   'checkpoint': str(paths['checkpoint'].relative_to(ROOT)),
                   'input_states_size': summary['input_states_size'], 'source_size': hashes,
                   'calls_staged': len(sample), 'calls_completed': 0, 'unique_pairs': len(selected),
                   'pairs': [list(pair_key(row)) for row in sorted(selected, key=pair_key)]})
    temporary = ledger_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(ledger, indent=2) + '\n')
    temporary.replace(ledger_path)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('source_paths', 'source_size')}, indent=2))
    print('Staged only; no live calls.')


if __name__ == '__main__':
    main()
