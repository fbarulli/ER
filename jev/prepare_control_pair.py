"""Freeze the deliberately repeated round-4 variant pair in both input formats."""
import hashlib,json,sys
from pathlib import Path

from core.project_root import find_project_root
ROOT = find_project_root(Path(__file__))
sys.path.insert(0,str(ROOT/'src'))
from training.gate_replay import canonical_records_from_csv
from prepare_input_split import clean

pair=('810036262576','810036266772')
records=canonical_records_from_csv()
source=json.loads((ROOT/'jev/sample_doubled_4.json').read_text())
base=next(x for x in source if x['copy']=='a_order' and set((x['gtin1'],x['gtin2']))==set(pair))
states={'gate_data':{},'original_data':{}};sample=[]
for scope in states:
    for gtin in pair:
        record=records[gtin]
        states[scope][gtin]=clean({k:v for k,v in record.items() if k!='source_rows'}) if scope=='gate_data' else {'gtin':gtin,'listings':json.loads(record['source_rows'])}
    for row in source:
        if set((row['gtin1'],row['gtin2']))==set(pair):
            sample.append({**row,'input_scope':scope,'intentional_repeat':True})
p=ROOT/'jev/sample_control_5.json';assert not (ROOT/'jev/audit_results_5.jsonl').exists()
p.write_text(json.dumps(sample,indent=1)+'\n')
(ROOT/'jev/input_states_5.json').write_text(json.dumps(states,ensure_ascii=False,indent=1)+'\n')
ledger_path=ROOT/'jev/sample_ledger.json';ledger=json.loads(ledger_path.read_text())
ledger.append({'round':5,'kind':'controlled_repeat','status':'staged_not_tested','sample':'jev/sample_control_5.json','sample_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'checkpoint':'jev/audit_results_5.jsonl','calls_staged':4,'calls_completed':0,'unique_pairs':1,'pairs':[list(pair)],'repeats_round':4,'reason':'Original listings distinguish S\u2019morey Time from High Voltage; canonical flavor evidence is latte/coffee on both sides.'})
ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
print('Prepared 4 controlled calls for',pair)
