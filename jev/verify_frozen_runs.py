"""Verify frozen request hashes, response scores, and sample coverage for runs 4–5."""
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'jev'))
from client import ADAPTERS,build_questions,_extract_noul

def main():
    ledger_path=ROOT/'jev/sample_ledger.json';ledger=json.loads(ledger_path.read_text())
    for n in (4,5):
        item=next(x for x in ledger if x['round']==n)
        sample_path=ROOT/item['sample'];result_path=ROOT/item['checkpoint'];states_path=ROOT/'jev'/f'input_states_{n}.json'
        sample=json.loads(sample_path.read_text());states=json.loads(states_path.read_text());rows=[json.loads(l) for l in result_path.read_text().splitlines()]
        key=lambda x:(x['input_scope'],x['gtin1'],x['gtin2'])
        staged={key(x):x for x in sample}; assert len(staged)==len(sample)
        assert len(rows)==len(staged) and {key(x) for x in rows}==set(staged)
        for row in rows:
            assert row['status']=='ok' and all(row[k]==v for k,v in staged[key(row)].items())
            state={'record_a':states[row['input_scope']][row['gtin1']], 'record_b':states[row['input_scope']][row['gtin2']]}
            request={'model':ADAPTERS['openrouter']['model'],'state':state,'questions':build_questions()}
            assert row['request_sha256']==hashlib.sha256(json.dumps(request).encode()).hexdigest()
            assert row['noul']==_extract_noul(row['raw_response'],'is_same_product')
            assert 0 <= row['noul'] <= 1
        item.update(status='tested',calls_completed=len(rows))
        provenance={'round':n,'adapter':'openrouter','model':ADAPTERS['openrouter']['model'],'questions':build_questions(),'completed_utc':max(x['completed_utc'] for x in rows),'calls_completed':len(rows),'unique_pairs':item['unique_pairs'],'request_hashes_verified':True,'raw_responses_saved':True,'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (sample_path,result_path,states_path)}}
        (ROOT/'jev'/f'audit_run_{n}.json').write_text(json.dumps(provenance,indent=2)+'\n')
        print(f'Round {n}: {len(rows)} complete; all request hashes and raw response scores verified.')
        if n==5:
            groups=defaultdict(list)
            for row in rows: groups[row['input_scope']].append(row)
            report={'pair':item['pairs'][0],'gate':'proceed','reason_selected':item['reason'],'input_scores':{scope:{row['copy']:row['noul'] for row in group} for scope,group in groups.items()},'interpretation':'Both representations lean toward different products. Gate inputs are more order-sensitive (0.24 vs 0.06); original listings are 0.04 in both orders. One selected case does not establish overall input-format effects.'}
            (ROOT/'jev/control_comparison_5.json').write_text(json.dumps(report,indent=2)+'\n')
    ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
if __name__=='__main__':main()
