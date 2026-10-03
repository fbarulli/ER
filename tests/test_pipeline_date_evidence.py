"""Date context travels through extraction and listing evidence without vetoing."""
import json
import pytest
from pipeline import extract_all, generate_canonical,NgramIDF,three_way_gate

@pytest.mark.parametrize('column,argument',[('sku_name_eng','sku_name_eng'),('attribute','attribute'),
                                          ('description_short_eng','description_short_eng'),('breadcrumbs_eng','breadcrumbs_eng'),
                                          ('category','category')])
def test_all_text_sources_retain_date_context(column,argument):
    text='Best before 2026-10-31'
    args=dict(sku_name_eng='Vanilla water 330ml',attribute='')
    args[argument]=text if argument!='sku_name_eng' else 'Vanilla water 330ml '+text
    result=extract_all(**args)
    entries=[e for e in result['date_evidence'] if e['column']==column]
    assert entries and entries[0]['role']=='expiry'
    assert entries[0]['gate_use']=='stock_review_context'
    assert any(e['field']=='source_date' and e['value']==entries[0] for e in result['evidence_ledger'])
    source=args[argument]
    assert source[entries[0]['start']:entries[0]['end']]==entries[0]['raw_match']


def test_date_context_persists_without_identity_rejection():
    rows=[('Vanilla water 330ml best before 2026-10-31','')]
    record=generate_canonical('1234567890123','Example',rows,NgramIDF({'1234567890123':rows}),None)
    evidence=json.loads(record['evidence_ledger'])
    assert any(e['field']=='source_date' and e['listing']==0 for e in evidence)
    assert three_way_gate(record,record)['decision']=='proceed'
