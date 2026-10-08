"""Finalize rebuilt labels and measure the complete all-item handoff."""
from core.portable_archive import ByteCount
import json
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd

from core.project_root import find_project_root

ROOT = find_project_root(Path(__file__))
OUT = ROOT / 'jev/full_evidence'

def main():
    inputs = json.loads((OUT/'run_inputs.json').read_text())
    assert all(ByteCount((ROOT/name).read_bytes()).total == value
               for name,value in inputs.items()), 'Rebuild inputs changed during execution.'
    gates = pd.read_csv(ROOT/'data/gate_results.csv', dtype={'gtin1':str,'gtin2':str}, keep_default_na=False)
    records = pd.read_csv(ROOT/'data/canonical_records.csv', dtype=str, keep_default_na=False)
    before = pd.read_csv(OUT/'before/data/canonical_records.csv', dtype=str, keep_default_na=False)
    old_gates = pd.read_csv(OUT/'before/data/gate_results.csv', dtype=str, keep_default_na=False)
    assert set(records.gtin) == set(before.gtin) and len(records)==13216
    old_sources = before.set_index('gtin').source_rows
    new_sources = records.set_index('gtin').source_rows
    assert all(json.loads(new_sources[item]) == json.loads(old_sources[item])
               for item in old_sources.index), 'Original source captures changed.'
    assert set(zip(gates.gtin1,gates.gtin2)) == set(zip(old_gates.gtin1,old_gates.gtin2))
    counts = {'total_pairs':len(gates), **{key:int((gates.gate_decision==key).sum()) for key in ['hard_no','proceed','fallback']}}
    assert sum(counts[k] for k in ['hard_no','proceed','fallback']) == counts['total_pairs']
    # Keep the independent measured census alongside the config pin.
    (OUT/'measured_census.json').write_text(json.dumps(counts,indent=2)+'\n')
    # measured_census.json IS the record (pins removed 2026-10-06 by owner
    # ruling; no config rewrite happens here anymore).
    env = __import__('os').environ.copy()
    env['PYTHONPATH']='src:.'
    for args in [
        ['src/training/labeled_pairs.py'],
        ['scripts/report_jev_rebuild.py','--output',str(OUT)],
        ['scripts/compare_item_pair_sets.py','--output',str(OUT)],
    ]:
        with (OUT/(Path(args[0]).stem+'.log')).open('w') as log:
            subprocess.run([str(ROOT/'.venv/bin/python'),*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    from training.gate_replay import canonical_records_from_csv
    from core.attribute_conflicts import canonical_attribute_info
    from training.rand_matching import targeted_veto_gate
    canonical = canonical_records_from_csv()
    info = {gtin:canonical_attribute_info(rec) for gtin,rec in canonical.items()}
    routes, missing = Counter(), Counter()
    for row in gates[gates.gate_decision=='proceed'].itertuples():
        a,b = canonical[row.gtin1],canonical[row.gtin2]
        decision = targeted_veto_gate(info[row.gtin1],info[row.gtin2],sku_brand=a['mode_brand'],candidate_brand=b['mode_brand'],exact_gtin=False)
        assert decision['targeted_gate_route'] != 'reject'
        routes[decision['targeted_gate_route']] += 1
        missing.update(x for x in decision['targeted_missing_attributes'].split(',') if x)
    census = {'resolved':sum(routes.values()),'routes':dict(routes),'missing_census':dict(missing)}
    (OUT/'targeted_gate_census.json').write_text(json.dumps(census,indent=2)+'\n')
    indexed = gates.set_index(['gtin1','gtin2'])
    labels = pd.read_csv(ROOT/'data/labeled_pairs.csv', dtype={'gtin1':str,'gtin2':str})
    positive = set(zip(labels[labels.true_label==1].gtin1,labels[labels.true_label==1].gtin2))
    results=[]
    for row in map(json.loads,(ROOT/'jev/audit_results_8.jsonl').read_text().splitlines()):
        if row['status']!='ok' or row['copy']!='a_order':continue
        key=tuple(sorted((row['gtin1'],row['gtin2'])))
        current = indexed.loc[key]
        band='different' if row['noul']<=.2 else 'same' if row['noul']>=.8 else 'uncertain'
        results.append({'gtin1':key[0],'gtin2':key[1],'old_training_label':row['training_label'],
                        'saved_jev_score':row['noul'],'score_band':band,'new_gate':current.gate_decision,
                        'new_reason':current.gate_reason,'new_similarity':float(current.similarity),
                        'new_positive_label':key in positive})
    old_pos=[r for r in results if r['old_training_label']==1]
    report={'saved_round8_pairs':len(results),'previous_640_positive_gate_bands':dict(Counter(r['new_gate']+'|'+r['score_band'] for r in old_pos)),
            'previous_640_positive_still_labeled_bands':dict(Counter(r['score_band'] for r in old_pos if r['new_positive_label'])),
            'new_gate_census':counts,'targeted_routes':dict(routes),'details':results,
            'limitations':'Diagnostic replay of saved judgments; not a new accuracy estimate. The primary comparison is all canonical items and their changing partner sets.'}
    (OUT/'saved_jev_replay.json').write_text(json.dumps(report,indent=2)+'\n')
    provenance={str(path.relative_to(ROOT)):ByteCount(path.read_bytes()).total for path in [ROOT/'dataset.csv',ROOT/'data/canonical_records.csv',ROOT/'data/gate_results.csv',ROOT/'data/labeled_pairs.csv',ROOT/'src/pipeline.py',ROOT/'src/core/pair_policy.py',ROOT/'src/core/declared_identity.py',ROOT/'src/core/attribute_decision.py',ROOT/'config/training.yaml']}
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='details'},indent=2))

if __name__=='__main__':main()
