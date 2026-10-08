"""Replay the frozen residual cohort; never resample away repaired/held cases."""
from __future__ import annotations
import dataclasses
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

from core.project_root import find_project_root

ROOT = find_project_root(Path(__file__))
sys.path.insert(0, str(ROOT/'src'))
from core.common import AUDIT_FINDINGS_DIR
from core.sku_identity import row_identity, identity_conflict, evaluate_sku_identity


def main():
    folder=AUDIT_FINDINGS_DIR
    cases=json.loads((folder/'residual_cases.json').read_text())
    cache={}
    def identity(row):
        if row['sku_id'] not in cache:
            cache[row['sku_id']]=row_identity(row)
        return cache[row['sku_id']]
    results=[]
    for c in cases:
        a,b=identity(c['left']),identity(c['right'])
        text=[dataclasses.replace(x,gtin_trusted=False,gtin_key='') for x in (a,b)]
        conflicts=identity_conflict(*text)
        decision=evaluate_sku_identity(a,b)
        if decision['decision']=='review':
            action='hold_identifier_scope' if decision['identity_review_reasons'] else 'review_source_conflict'
        elif c['kind']=='different_gtin':
            action='keep_distinct_repaired_descriptors' if conflicts else 'keep_distinct_need_variant_evidence'
        else:
            action='retain_match_descriptors_repaired' if not conflicts else 'retain_match_review_remaining_feed_conflicts'
        results.append({'case_id':c['id'],'sku_ids':[c[s]['sku_id'] for s in ('left','right')],
            'kind':c['kind'],'before_conflicts':c['descriptor_conflicts'],'after_conflicts':conflicts,
            'before_decision':c['production']['decision'],'after_decision':decision['decision'],
            'hold_reasons':decision['identity_review_reasons'],'disposition':action})
    summary={'cohort':'frozen original residuals; not a new sampling or full recall estimate',
        'cases':len(results),'actions':dict(Counter(r['disposition'] for r in results)),
        'decisions':dict(Counter(r['after_decision'] for r in results)),
        'hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [ROOT/'dataset.csv',ROOT/'config/identity_reviews.json',ROOT/'src/pipeline.py',ROOT/'src/core/url_evidence.py',folder/'residual_cases.json']}}
    (folder/'residual_replay.json').write_text(json.dumps({'summary':summary,'cases':results},indent=2)+'\n')
    lines=['# Frozen residual replay','','Each original case remains in this replay, including quarantined identifiers. This prevents improved metrics caused by silently removing difficult pairs.','','```json',json.dumps(summary,indent=2),'```','','| Case | Source SKUs | Before conflicts | After conflicts | Action |','|---|---|---|---|---|']
    for r in results:
        lines.append('| '+ ' | '.join([r['case_id'],', '.join(r['sku_ids']),', '.join(r['before_conflicts']) or 'none',', '.join(r['after_conflicts']) or 'none',r['disposition']])+' |')
    (folder/'RESIDUAL_REPLAY.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
