"""Freeze round-4 inputs and assign a stratified 250/250 evidence split."""
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from core.project_root import find_project_root
ROOT = find_project_root(Path(__file__))
sys.path.insert(0,str(ROOT/'src'))
from training.gate_replay import canonical_records_from_csv

def clean(x):
    if isinstance(x, dict): return {k:clean(v) for k,v in x.items()}
    if isinstance(x, (set,frozenset)): return sorted(clean(v) for v in x)
    if isinstance(x, (list,tuple)): return [clean(v) for v in x]
    return x

def main():
    path=ROOT/'jev/sample_doubled_4.json'
    if (ROOT/'jev/audit_results_4.jsonl').exists(): raise ValueError('round already started')
    sample=json.loads(path.read_text()); originals=[x for x in sample if x['copy']=='a_order']
    buckets=defaultdict(list)
    for row in originals: buckets[row['stratum']].append(row)
    rng=random.Random(44); assignments={};extra_gate=True
    for name,rows in sorted(buckets.items()):
        rng.shuffle(rows); n=len(rows)//2
        if len(rows)%2:
            n+=extra_gate;extra_gate=not extra_gate
        for i,row in enumerate(rows): assignments[tuple(sorted((row['gtin1'],row['gtin2'])))]='gate_data' if i<n else 'original_data'
    assert Counter(assignments.values())=={'gate_data':250,'original_data':250}
    records=canonical_records_from_csv();states={'gate_data':{},'original_data':{}}
    for row in sample:
        scope=assignments[tuple(sorted((row['gtin1'],row['gtin2'])))];row['input_scope']=scope
        for field in ('gtin1','gtin2'):
            gtin=row[field];record=records[gtin]
            if scope=='gate_data':
                states[scope][gtin]=clean({k:v for k,v in record.items() if k!='source_rows'})
            else:
                source_rows=json.loads(record['source_rows'])
                assert source_rows
                states[scope][gtin]={'gtin':gtin,'listings':source_rows}
    path.write_text(json.dumps(sample,indent=1)+'\n')
    frozen=ROOT/'jev/input_states_4.json';frozen.write_text(json.dumps(states,ensure_ascii=False,indent=1)+'\n')
    summary_path=ROOT/'jev/sample_4_summary.json';summary=json.loads(summary_path.read_text());summary.update(split_seed=44,input_cohorts={scope:dict(Counter(r['stratum'] for r in sample if r['copy']=='a_order' and r['input_scope']==scope)) for scope in states},input_states_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),input_definitions={'gate_data':'Merged canonical gate evidence; source_rows omitted; no gate verdict provided to JEV.','original_data':'All original source_rows with full fields and untruncated descriptions.'},comparison='Independent stratified cohorts, not a paired causal comparison. Same question and model for both.')
    summary_path.write_text(json.dumps(summary,indent=2)+'\n')
    ledger_path=ROOT/'jev/sample_ledger.json';ledger=json.loads(ledger_path.read_text());assert not any(x['round']==4 for x in ledger)
    pairs={tuple(sorted((r['gtin1'],r['gtin2']))) for r in sample};prior={tuple(x) for item in ledger for x in item['pairs']};assert not pairs & prior
    ledger.append({'round':4,'status':'staged_not_tested','sample':'jev/sample_doubled_4.json','sample_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'checkpoint':'jev/audit_results_4.jsonl','calls_staged':len(sample),'calls_completed':0,'unique_pairs':len(pairs),'pairs':[list(x) for x in sorted(pairs)]})
    ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
    print(json.dumps(summary['input_cohorts'],indent=2));print('Frozen',frozen.stat().st_size,'bytes; reserved 500 fresh pairs.')
if __name__=='__main__':main()
