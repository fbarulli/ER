"""Replay saved JEV pairs with fresh source-derived features, without API calls."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from training.gate_replay import canonical_records_from_csv
from pipeline import generate_canonical, NgramIDF, three_way_gate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--round', type=int, default=7)
    args = ap.parse_args()
    directory = ROOT / 'jev'
    sample = [row for row in json.loads((directory / f'sample_doubled_{args.round}.json').read_text()) if row['copy'] == 'a_order']
    audited = {(row['gtin1'], row['gtin2']): row for row in
               (json.loads(line) for line in (directory / f'audit_results_{args.round}.jsonl').read_text().splitlines())
               if row['status'] == 'ok' and row['copy'] == 'a_order'}
    records = canonical_records_from_csv()
    source = {gtin: json.loads(record['source_rows']) for gtin, record in records.items()}
    groups = {gtin: [(row.get('sku_name_eng', ''), row.get('attribute', '')) for row in rows]
              for gtin, rows in source.items()}
    idf = NgramIDF(groups)
    required = {row[field] for row in sample for field in ('gtin1', 'gtin2')}
    fresh = {}
    for i, gtin in enumerate(sorted(required), 1):
        rows = source[gtin]
        record = generate_canonical(gtin, records[gtin]['mode_brand'], groups[gtin], idf, None,
                                    descriptions=[r.get('description_short_eng', '') for r in rows],
                                    urls=[r.get('sku_url', '') for r in rows],
                                    image_urls=[r.get('image_url', '') for r in rows],
                                    breadcrumbs_engs=[r.get('breadcrumbs_eng', '') for r in rows],
                                    categories=[r.get('category', '') for r in rows])
        record['source_rows'] = records[gtin]['source_rows']
        record['description_evidence'] = records[gtin].get('description_evidence', [])
        record['breadcrumb_evidence'] = records[gtin].get('breadcrumb_evidence', [])
        fresh[gtin] = record
        if i % 300 == 0:
            print(f'rebuilt {i}/{len(required)} canonical records', flush=True)
    transitions = Counter()
    score_groups = defaultdict(Counter)
    results = []
    for row in sample:
        key = row['gtin1'], row['gtin2']
        current = three_way_gate(fresh[key[0]], fresh[key[1]])
        reverse = three_way_gate(fresh[key[1]], fresh[key[0]])
        assert current['decision'] == reverse['decision'], key
        transitions[f'{row["gate"]}->{current["decision"]}'] += 1
        score = audited[key]['noul']
        band = 'different' if score <= .2 else 'same' if score >= .8 else 'uncertain'
        score_groups[current['decision']][band] += 1
        results.append({'gtin1': key[0], 'gtin2': key[1], 'training_label': row['training_label'],
                        'prior_gate': row['gate'], 'prior_reason': row['gate_reason'],
                        'current_gate': current['decision'], 'current_reason': current['reason'],
                        'saved_jev_score': score, 'score_band': band})
    inspected = json.loads((directory / f'inspection_{args.round}.json').read_text())['pairs']
    inspected_keys = {(row['gtin1'], row['gtin2']) for row in inspected}
    report = {'round': args.round, 'pairs': len(results), 'rebuilt_records': len(fresh),
              'transitions': dict(transitions),
              'saved_jev_bands_by_new_gate': {k: dict(v) for k, v in score_groups.items()},
              'inspected_cases': [row for row in results if (row['gtin1'], row['gtin2']) in inspected_keys],
              'results': results,
              'limitations': 'Offline replay reuses existing JEV scores and original sources. Fresh feature extraction plus current gate is compared with saved pre-fix decisions. No new JEV judgments, no full candidate census regeneration. IDF uses full saved canonical-source universe; per-brand IDF omitted, affecting residual ngrams only.'}
    output = directory / f'identity_fix_replay_{args.round}.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'results'}, indent=2))


if __name__ == '__main__':
    main()
