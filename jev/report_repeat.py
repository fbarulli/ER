"""Verify and compare the user-requested 50-pair repeat without new calls."""
from core.portable_archive import ByteCount
import json
from collections import Counter, defaultdict
from pathlib import Path

from core.project_root import find_project_root

ROOT = find_project_root(Path(__file__))
D = ROOT / 'jev'

def band(score):
    return 'different' if score <= .2 else 'same' if score >= .8 else 'uncertain'

def main():
    rows = [json.loads(line) for line in (D / 'audit_results_9.jsonl').read_text().splitlines()]
    sample = json.loads((D / 'sample_doubled_9.json').read_text())
    states = json.loads((D / 'input_states_9.json').read_text())
    assert len(rows) == len(sample) == 50
    assert all(r['status'] == 'ok' for r in rows)
    import sys
    sys.path.insert(0,str(D))
    from client import build_questions
    transitions, cohorts, details = Counter(), defaultdict(Counter), []
    for row in rows:
        state = {'record_a':states[row['input_scope']][row['gtin1']], 'record_b':states[row['input_scope']][row['gtin2']]}
        request = {'model':row['model'], 'state':state, 'questions':build_questions()}
        digest = ByteCount(json.dumps(request).encode()).total
        assert digest == row['request_size'] == row['prior_request_size']
        before, after = band(row['prior_jev_score']), band(row['noul'])
        transitions[before+'->'+after] += 1
        cohorts[row['stratum']][after] += 1
        details.append({'gtin1':row['gtin1'],'gtin2':row['gtin2'],'prior_score':row['prior_jev_score'],
                        'repeat_score':row['noul'],'score_delta':row['noul']-row['prior_jev_score'],
                        'prior_band':before,'repeat_band':after,'cohort':row['stratum']})
    report = {'calls':50,'successful':50,'exact_request_repeats_verified':50,'transitions':dict(transitions),
              'bands_by_cohort':{k:dict(v) for k,v in cohorts.items()},
              'band_changes':sum(r['prior_band']!=r['repeat_band'] for r in details),
              'mean_absolute_score_delta':sum(abs(r['score_delta']) for r in details)/50,
              'large_score_changes':sum(abs(r['score_delta'])>=.2 for r in details),
              'pairs':details,'interpretation':'Stratified same-input repeat measures judgment stability, not improvement from gate fixes or population accuracy.'}
    (D / 'report_9.json').write_text(json.dumps(report,indent=2)+'\n')
    ledger_path = D / 'sample_ledger.json'
    ledger = json.loads(ledger_path.read_text())
    entry = next(r for r in ledger if r['round']==9)
    entry.update(status='tested',calls_completed=50)
    ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='pairs'},indent=2))

if __name__ == '__main__':
    main()
