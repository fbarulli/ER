"""Summarize paired round-6 JEV scores without treating judgments as truth."""
import json
from collections import defaultdict
from pathlib import Path

from core.project_root import find_project_root
ROOT = find_project_root(Path(__file__))

def bucket(scores):
    return 'low' if max(scores)<.2 else 'high' if min(scores)>.8 else 'uncertain'

def main():
    rows=[json.loads(l) for l in (ROOT/'jev/audit_results_6.jsonl').read_text().splitlines()]
    groups=defaultdict(lambda:defaultdict(list))
    for row in rows:
        assert row['status']=='ok'
        groups[tuple(sorted((row['gtin1'],row['gtin2'])))][row['input_scope']].append(row)
    pairs=[]
    for pair,scopes in groups.items():
        assert set(scopes)=={'gate_data','original_data'}
        item={'gtin1':pair[0],'gtin2':pair[1],'gate':scopes['gate_data'][0]['gate'],'stratum':scopes['gate_data'][0]['stratum']}
        for scope,records in scopes.items():
            assert len(records)==2 and {x['copy'] for x in records}=={'a_order','b_swapped'}
            scores=[x['noul'] for x in records]
            item[scope]={'scores':{x['copy']:x['noul'] for x in records},'mean':sum(scores)/2,'bucket':bucket(scores),'order_gap':abs(scores[0]-scores[1])}
        item['original_minus_processed']=item['original_data']['mean']-item['gate_data']['mean']
        pairs.append(item)
    cohorts={}
    for scope in ('gate_data','original_data'):
        cohorts[scope]={}
        for gate in ('proceed','hard_no','fallback'):
            sub=[x for x in pairs if x['gate']==gate]
            cohorts[scope][gate]={'pairs':len(sub),'mean_score':sum(x[scope]['mean'] for x in sub)/len(sub) if sub else None,'both_below_0_2':sum(x[scope]['bucket']=='low' for x in sub),'both_above_0_8':sum(x[scope]['bucket']=='high' for x in sub)}
    disagreement=[x for x in pairs if x['gate_data']['bucket']!=x['original_data']['bucket']]
    result={'round':6,'unique_pairs':len(pairs),'completed_calls':len(rows),'input_cohorts':cohorts,'mean_original_minus_processed':sum(x['original_minus_processed'] for x in pairs)/len(pairs),'bucket_disagreements':len(disagreement),'opposite_confident_judgments':sum({x['gate_data']['bucket'],x['original_data']['bucket']}=={'low','high'} for x in pairs),'pairs':sorted(pairs,key=lambda x:-abs(x['original_minus_processed'])),'interpretation':'Same products, same question and model, separate input formats; model order sensitivity and uncertainty remain. Balanced discovery sample does not estimate population accuracy.'}
    (ROOT/'jev/paired_comparison_6.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='pairs'},indent=2))
if __name__=='__main__':main()
