"""Stage exactly 50 previously judged pairs for a reproducible repeat check."""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'jev'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def stage():
    output = DIRECTORY / 'sample_doubled_9.json'
    if output.exists():
        raise FileExistsError(output)
    previous = [json.loads(line) for line in (DIRECTORY / 'audit_results_8.jsonl').read_text().splitlines()]
    previous = [r for r in previous if r['status'] == 'ok' and r['copy'] == 'a_order']
    def cell(row):
        if row['training_label'] == 0:
            return 'old_negative'
        if row['training_label'] != 1:
            return 'diagnostic'
        score = row['noul']
        return 'positive_same' if score >= .8 else 'positive_different' if score <= .2 else 'positive_uncertain'
    allocations = {'positive_different': 20, 'positive_uncertain': 10, 'positive_same': 10, 'old_negative': 10}
    rng = random.Random(950)
    selected = []
    frozen = json.loads((DIRECTORY / 'input_states_8.json').read_text())
    states = {'original_data': {}}
    for name, count in allocations.items():
        rows = sorted((r for r in previous if cell(r) == name), key=lambda r:(r['gtin1'], r['gtin2']))
        for row in rng.sample(rows, count):
            sample = {key: value for key, value in row.items() if key not in {
                'noul', 'status', 'adapter', 'model', 'completed_utc', 'request_sha256', 'raw_response'}}
            sample.update(audit_cohort='repeat_old_pairs', stratum=name,
                          prior_jev_score=row['noul'], prior_request_sha256=row['request_sha256'], prior_round=8)
            selected.append(sample)
            for field in ('gtin1', 'gtin2'):
                gtin = row[field]
                states['original_data'][gtin] = frozen['original_data'][gtin]
    assert len(selected) == len({tuple(sorted((r['gtin1'], r['gtin2']))) for r in selected}) == 50
    # Confirm exact full source evidence is still the supplied dataset capture.
    import csv
    with (ROOT / 'data/canonical_records.csv').open(newline='') as stream:
        captured = {r['gtin']:json.loads(r['source_rows']) for r in csv.DictReader(stream) if r['gtin'] in states['original_data']}
    assert all(state['listings'] == captured[gtin] for gtin, state in states['original_data'].items())
    output.write_text(json.dumps(selected, ensure_ascii=False, indent=2)+'\n')
    state_path = DIRECTORY / 'input_states_9.json'
    state_path.write_text(json.dumps(states, ensure_ascii=False, indent=2)+'\n')
    summary = {'strategy':'stratified repeat of prior round-8 pairs', 'calls':50, 'unique_pairs':50,
               'allocations':allocations, 'seed':950, 'sample_sha256':digest(output),
               'input_states_sha256':digest(state_path),
               'interpretation':'Diagnostic repeat, not a population accuracy estimate. Same model inputs and order as the prior call; gates are evaluated offline separately.'}
    (DIRECTORY / 'sample_9_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    ledger_path = DIRECTORY / 'sample_ledger.json'
    ledger = json.loads(ledger_path.read_text())
    ledger.append({'round':9,'status':'staged_not_tested','sample':'jev/sample_doubled_9.json',
                   'sample_sha256':digest(output),'checkpoint':'jev/audit_results_9.jsonl',
                   'calls_staged':50,'calls_completed':0,'unique_pairs':50,'purpose':'user-requested repeat of old pairs',
                   'pairs':[list(sorted((r['gtin1'],r['gtin2']))) for r in selected]})
    ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    stage()
