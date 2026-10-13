"""Verify and report a rebuilt-population JEV audit without making API calls."""
import argparse
from core.portable_archive import ByteCount
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from core.project_root import find_project_root

ROOT = find_project_root(Path(__file__))
sys.path.insert(0, str(ROOT / 'jev'))
from client import ADAPTERS, build_questions, _extract_noul


def bucket(score):
    return 'same' if score >= .8 else 'different' if score <= .2 else 'uncertain'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--round', type=int, default=7)
    args = ap.parse_args()
    directory = ROOT / 'jev'
    sample_path = directory / f'sample_doubled_{args.round}.json'
    states_path = directory / f'input_states_{args.round}.json'
    summary_path = directory / f'sample_{args.round}_summary.json'
    result_path = directory / f'audit_results_{args.round}.jsonl'
    sample = json.loads(sample_path.read_text())
    states = json.loads(states_path.read_text())
    summary = json.loads(summary_path.read_text())
    key = lambda row: (row['input_scope'], row['gtin1'], row['gtin2'])
    staged = {key(row): row for row in sample}
    assert len(staged) == len(sample)
    ledger_path = directory / 'sample_ledger.json'
    ledger = json.loads(ledger_path.read_text())
    entry = next(row for row in ledger if row['round'] == args.round)
    sha = lambda path: ByteCount(path.read_bytes()).total
    assert sha(sample_path) == entry['sample_size']
    assert sha(states_path) == summary['input_states_size']
    successful = {}
    errors = Counter()
    for line in result_path.read_text().splitlines():
        row = json.loads(line)
        assert key(row) in staged
        if row['status'] != 'ok':
            errors[row['status']] += 1
            continue
        assert all(row[field] == value for field, value in staged[key(row)].items())
        request = {'model': ADAPTERS[row['adapter']]['model'],
                   'state': {'record_a': states[row['input_scope']][row['gtin1']],
                             'record_b': states[row['input_scope']][row['gtin2']]},
                   'questions': build_questions()}
        assert row['request_size'] == ByteCount(json.dumps(request).encode()).total
        assert row['noul'] == _extract_noul(row['raw_response'], 'is_same_product')
        assert math.isfinite(row['noul']) and 0 <= row['noul'] <= 1
        successful[key(row)] = row
    if set(successful) != set(staged):
        raise ValueError(f'incomplete run: {len(successful)}/{len(staged)} successes; errors={dict(errors)}')
    primary = [row for row in successful.values() if row['copy'] == 'a_order']
    groups = defaultdict(list)
    for row in primary:
        cohort = row.get('audit_cohort', 'current_training_pairs')
        groups[f'{cohort}|label_{row["training_label"]}|{row["frozen_gate"]}->{row["gate"]}'].append(row)
    group_report = {name: {'pairs': len(rows), 'mean_score': sum(r['noul'] for r in rows) / len(rows),
                           'score_buckets': dict(Counter(bucket(r['noul']) for r in rows))}
                    for name, rows in sorted(groups.items())}
    weighted = defaultdict(lambda: Counter())
    weighted_lost = Counter()
    for row in primary:
        allocation = summary['allocations'][row['stratum']]
        weight = allocation['eligible_population'] / allocation['selected']
        counts = (weighted_lost if row.get('audit_cohort') == 'lost_positive_partners'
                  else weighted[str(row['training_label'])])
        counts['population'] += weight
        counts[bucket(row['noul'])] += weight
        counts['score_sum'] += weight * row['noul']
    weighted_report = {label: {'eligible_population': round(counts['population']),
                               'mean_score': counts['score_sum'] / counts['population'],
                               'bucket_fractions': {name: counts[name] / counts['population'] for name in ('same', 'different', 'uncertain')}}
                       for label, counts in weighted.items()}
    lost_report = ({'eligible_population': round(weighted_lost['population']),
                    'mean_score': weighted_lost['score_sum'] / weighted_lost['population'],
                    'bucket_fractions': {name: weighted_lost[name] / weighted_lost['population'] for name in ('same', 'different', 'uncertain')}}
                   if weighted_lost['population'] else None)
    order_rows = []
    for reverse in successful.values():
        if reverse['copy'] != 'b_swapped':
            continue
        forward = successful[(reverse['input_scope'], reverse['gtin2'], reverse['gtin1'])]
        order_rows.append({'gtin1': forward['gtin1'], 'gtin2': forward['gtin2'],
                           'stratum': forward['stratum'], 'forward': forward['noul'],
                           'swapped': reverse['noul'], 'gap': abs(forward['noul'] - reverse['noul']),
                           'bucket_changed': bucket(forward['noul']) != bucket(reverse['noul'])})
    report = {'round': args.round, 'completed_calls': len(successful), 'unique_pairs': len(primary),
              'score_bands': {'different': '<=0.2', 'same': '>=0.8', 'uncertain': 'between'},
              'training_label_and_gate_transition': group_report,
              'weighted_fresh_training_population': weighted_report,
              'weighted_lost_positive_partners': lost_report,
              'order_checks': {'pairs': len(order_rows),
                               'bucket_changes': sum(r['bucket_changed'] for r in order_rows),
                               'gaps_at_least_0_2': sum(r['gap'] >= .2 for r in order_rows),
                               'largest_gaps': sorted(order_rows, key=lambda r: r['gap'], reverse=True)[:20]},
              'errors_in_checkpoint': dict(errors),
              'meaning': 'JEV judgments of exact retail-product identity; training labels are gate-derived. Current-training estimates cover fresh eligible training pairs after separately sampled lost-positive diagnostics are reserved; lost-positive estimates cover their separate fresh eligible population. Swapped calls excluded from primary estimates. Score bands are reporting conventions, not calibrated truth probabilities. No score is transferred to a different partner.'}
    report_path = directory / f'report_{args.round}.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    provenance = {'round': args.round, 'questions': build_questions(),
                  'models': sorted({row['model'] for row in successful.values()}),
                  'adapter': sorted({row['adapter'] for row in successful.values()}),
                  'completed_utc': max(row['completed_utc'] for row in successful.values()),
                  'calls_completed': len(successful), 'unique_pairs': len(primary),
                  'request_hashes_verified': True, 'raw_responses_saved': True,
                  'size': {str(path.relative_to(ROOT)): sha(path) for path in (sample_path, states_path, result_path, summary_path, report_path)}}
    (directory / f'audit_run_{args.round}.json').write_text(json.dumps(provenance, indent=2) + '\n')
    entry.update(status='tested', calls_completed=len(successful))
    temporary = ledger_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(ledger, indent=2) + '\n')
    temporary.replace(ledger_path)
    print(json.dumps({k: v for k, v in report.items() if k != 'order_checks'}, indent=2))
    print('Order checks:', {k: v for k, v in report['order_checks'].items() if k != 'largest_gaps'})


if __name__ == '__main__':
    main()
