"""Stage fresh gate-polarity × similarity samples with attribute coverage, offline."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from training.gate_replay import canonical_records_from_csv
from pipeline import three_way_gate
from core.attribute_conflicts import canonical_attribute_info, full_attribute_evaluation

FIELDS = ('volume', 'pack', 'package_type', 'flavor', 'carbonation', 'sweetener', 'pulp', 'pack_material')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=43)
    ap.add_argument('--pairs', type=int, default=500)
    ap.add_argument('--round', type=int, default=3)
    ap.add_argument('--pool-per-stratum', type=int, default=400)
    args = ap.parse_args()
    if args.pairs < 1 or args.pool_per_stratum < 1:
        ap.error('pair and pool counts must be positive')
    suffix = f'_{args.round}'
    output_path = ROOT/'jev'/f'sample_doubled{suffix}.json'
    checkpoint_path = ROOT/'jev'/f'audit_results{suffix}.jsonl'
    if checkpoint_path.exists():
        ap.error('checkpoint already exists; use a new round to protect tested samples')
    retained = [x for x in json.loads(output_path.read_text()) if x['copy'] == 'a_order'] if output_path.exists() else []
    if len(retained) > args.pairs:
        ap.error('requested total is smaller than the already reserved sample')
    rng = random.Random(args.seed)
    excluded = set()
    for name in ('audit_results.jsonl', 'audit_results_2.jsonl'):
        for line in (ROOT/'jev'/name).read_text().splitlines():
            row = json.loads(line)
            excluded.add(tuple(sorted((row['gtin1'], row['gtin2']))))
    ledger_path = ROOT/'jev/sample_ledger.json'
    if ledger_path.exists():
        for item in json.loads(ledger_path.read_text()):
            if item['round'] != args.round:
                excluded.update(tuple(pair) for pair in item['pairs'])
    # Reserve every previously staged pair as well, including unsuccessful calls.
    for sample_path in sorted((ROOT/'jev').glob('sample_doubled*.json')):
        if sample_path == output_path: continue
        for row in json.loads(sample_path.read_text()):
            excluded.add(tuple(sorted((row['gtin1'], row['gtin2']))))
    pools = defaultdict(list)
    seen = set()
    with (ROOT/'data/gate_results.csv').open() as fh:
        for row in csv.DictReader(fh):
            pair = tuple(sorted((row['gtin1'], row['gtin2'])))
            if pair in excluded or pair in seen or pair[0] == pair[1]: continue
            seen.add(pair)
            band = 'high' if float(row['similarity'] or 0) >= .8 else 'lower'
            pools[(row['gate_decision'], band)].append(row)
    records = canonical_records_from_csv()
    candidates = []
    for pool_key, pool in sorted(pools.items()):
        candidates.extend(rng.sample(pool, min(args.pool_per_stratum, len(pool))))
    candidate_keys = {tuple(sorted((x['gtin1'],x['gtin2']))) for x in candidates}
    for row in retained:
        pair = tuple(sorted((row['gtin1'],row['gtin2'])))
        if pair not in candidate_keys:
            candidates.append({**row, 'gate_decision':row['frozen_gate']})
            candidate_keys.add(pair)
    retained_keys = {tuple(sorted((x['gtin1'],x['gtin2']))) for x in retained}
    buckets = defaultdict(list)
    infos = {}
    def info(gtin):
        if gtin not in infos: infos[gtin] = canonical_attribute_info(records[gtin])
        return infos[gtin]
    for i, row in enumerate(candidates, 1):
        a, b = row['gtin1'], row['gtin2']
        current = three_way_gate(records[a], records[b])
        evaluation = full_attribute_evaluation(info(a), info(b))
        states = evaluation['dimension_states']
        # Critical states are explicit for channels outside the universe registry.
        from core.attribute_conflicts import critical_attribute_evaluation
        critical = critical_attribute_evaluation(info(a), info(b))
        critical_states = {field: ('conflict' if field in critical['conflicts'] else 'agree' if field in critical['agreements'] else 'unknown') for field in FIELDS}
        polarity = {'proceed':'positive', 'hard_no':'negative', 'fallback':'uncertain'}[current['decision']]
        band = 'high' if float(row['similarity'] or 0) >= .8 else 'lower'
        item = {'gtin1':a, 'gtin2':b, 'gate':current['decision'], 'gate_reason':current['reason'],
                'frozen_gate':row['gate_decision'], 'similarity':row['similarity'],
                'polarity':polarity, 'similarity_band':band, 'stratum':f'{polarity}_{band}',
                'attribute_states':critical_states, 'universe_states':states}
        buckets[item['stratum']].append(item)
        if i % 150 == 0: print(f'replayed {i}/{len(candidates)}', flush=True)
    selected = []
    coverage = Counter()
    allocation = {}
    strata = ('positive_high','positive_lower','negative_high','negative_lower','uncertain_high','uncertain_lower')
    for stratum_index, stratum in enumerate(strata):
        target = args.pairs // len(strata) + (stratum_index < args.pairs % len(strata))
        pool = buckets[stratum]
        rng.shuffle(pool)
        pool_size = len(pool)
        picks = [x for x in pool if tuple(sorted((x['gtin1'],x['gtin2']))) in retained_keys]
        for pick in picks:
            pool.remove(pick)
            for k,v in pick['attribute_states'].items(): coverage[(stratum,k,v)] += 1
            for k,v in pick['universe_states'].items(): coverage[(stratum,'universe:'+k,v)] += 1
        if len(picks) > target or pool_size < target:
            raise ValueError(f'{stratum}: {pool_size} candidates, {len(picks)} reserved, target {target}; increase --pool-per-stratum')
        while pool and len(picks) < target:
            def novelty(item):
                tokens = [(stratum,k,v) for k,v in item['attribute_states'].items()]
                tokens += [(stratum,'universe:'+k,v) for k,v in item['universe_states'].items() if v in ('conflict','agree','subset')]
                return sum(1/(1+coverage[token]) for token in tokens)
            pick = max(pool, key=novelty)
            pool.remove(pick)
            picks.append(pick)
            for k,v in pick['attribute_states'].items(): coverage[(stratum,k,v)] += 1
            for k,v in pick['universe_states'].items(): coverage[(stratum,'universe:'+k,v)] += 1
        selected.extend(picks)
        allocation[stratum] = {'candidate_pool':pool_size, 'selected':len(picks), 'requested':target}
    assert len(selected) == args.pairs
    assert retained_keys <= {tuple(sorted((x['gtin1'],x['gtin2']))) for x in selected}
    doubled = []
    for row in selected:
        reverse_states = {k: {'missing_left':'missing_right', 'missing_right':'missing_left'}.get(v,v) for k,v in row['universe_states'].items()}
        reverse = three_way_gate(records[row['gtin2']], records[row['gtin1']])
        assert reverse['decision'] == row['gate']
        doubled.extend([{**row,'copy':'a_order'}, {**row,'gtin1':row['gtin2'],'gtin2':row['gtin1'], 'gate_reason':reverse['reason'], 'universe_states':reverse_states,'copy':'b_swapped'}])
    out = output_path
    out.write_text(json.dumps(doubled, indent=1)+'\n')
    summary = {'seed':args.seed,'retained_pairs':len(retained),'requested_pairs':args.pairs,'unique_pairs':len(selected),'calls':len(doubled),
               'excluded_previously_staged_or_tested_pairs':len(excluded), 'candidate_pairs_replayed':len(candidates),
               'allocations':allocation, 'attribute_coverage':{'|'.join(k):v for k,v in sorted(coverage.items())},
               'source_sha256':{name:hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest() for name in ('gate_results.csv','canonical_records.csv')},
               'meaning':'Positive/negative are CURRENT gate decisions, not human ground truth. Similarity high >= 0.8; lower < 0.8. Attribute-balanced selection within each stratum from a deterministic candidate subsample; not a prevalence estimate.'}
    (ROOT/'jev'/f'sample_{args.round}_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('attribute_coverage','source_sha256')},indent=2))
    print('Staged only; no live calls.')

if __name__ == '__main__': main()
