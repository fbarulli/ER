"""Freeze original and processed inputs for the same 100 fresh round-6 pairs."""
import hashlib,json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from training.gate_replay import canonical_records_from_csv
from prepare_input_split import clean

def main():
    n=6;path=ROOT/'jev/sample_doubled_6.json'
    if (ROOT/'jev/audit_results_6.jsonl').exists():raise ValueError('round already started')
    base=json.loads(path.read_text());assert len(base)==200
    records=canonical_records_from_csv();states={'gate_data':{},'original_data':{}};sample=[]
    for row in base:
        for scope in states:
            sample.append({**row,'input_scope':scope})
            for field in ('gtin1','gtin2'):
                gtin=row[field];record=records[gtin]
                states[scope][gtin]=clean({k:v for k,v in record.items() if k!='source_rows'}) if scope=='gate_data' else {'gtin':gtin,'listings':json.loads(record['source_rows'])}
    pairs={tuple(sorted((x['gtin1'],x['gtin2']))) for x in sample}
    assert len(pairs)==100 and len(sample)==400
    ledger_path=ROOT/'jev/sample_ledger.json';ledger=json.loads(ledger_path.read_text());assert not any(x['round']==n for x in ledger)
    prior={tuple(p) for item in ledger for p in item['pairs']};assert not pairs & prior
    path.write_text(json.dumps(sample,indent=1)+'\n')
    frozen=ROOT/'jev/input_states_6.json';frozen.write_text(json.dumps(states,ensure_ascii=False,indent=1)+'\n')
    summary_path=ROOT/'jev/sample_6_summary.json';summary=json.loads(summary_path.read_text());summary.update(calls=400,input_cohorts={scope:dict(Counter(x['stratum'] for x in sample if x['copy']=='a_order' and x['input_scope']==scope)) for scope in states},input_definitions={'gate_data':'Merged canonical evidence; source_rows omitted; no gate verdict or pair-level agreement summary sent.','original_data':'All source listings, full fields and untruncated descriptions.'},comparison='Paired comparison: identical 100 pairs in each input format, each tested in both orders.',input_states_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest());summary_path.write_text(json.dumps(summary,indent=2)+'\n')
    ledger.append({'round':n,'kind':'paired_comparison','status':'staged_not_tested','sample':str(path.relative_to(ROOT)),'sample_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'checkpoint':'jev/audit_results_6.jsonl','calls_staged':400,'calls_completed':0,'unique_pairs':100,'pairs':[list(x) for x in sorted(pairs)]});ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
    print('Reserved 100 fresh pairs; froze both inputs; 400 ordered calls.')
if __name__=='__main__':main()
