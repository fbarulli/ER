import importlib.util
import json
from pathlib import Path
import pytest
from fastapi import HTTPException

@pytest.fixture
def reports():
    spec = importlib.util.spec_from_file_location('jev_reports_test', Path(__file__).parents[1]/'dashboard/jev_reports.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_saved_500_pair_sample_and_coverage_display(reports):
    page = reports.jev()
    assert '500' in page and '1000' in page
    assert 'Tested' in page
    assert 'Attribute coverage' in page and 'positive_high' in page
    assert 'Low-score proceeds' in page
    assert 'Pairs (84)' in reports.jev(stratum='positive_high')

def test_checkpoint_progress_is_live_and_sample_bounded(reports, tmp_path, monkeypatch):
    monkeypatch.setattr(reports, 'PROJECT', tmp_path)
    root = tmp_path/'jev';root.mkdir()
    (root/'sample_ledger.json').write_text(json.dumps([{'round':3,'sample':'jev/sample.json','checkpoint':'jev/results.jsonl','unique_pairs':1}]))
    (root/'sample.json').write_text(json.dumps([{'gtin1':'1','gtin2':'2','copy':'a_order','stratum':'<script>','similarity':'1'}, {'gtin1':'2','gtin2':'1','copy':'b_swapped','stratum':'<script>','similarity':'1'}]))
    (root/'results.jsonl').write_text('\n'.join(json.dumps(x) for x in [{'gtin1':'1','gtin2':'2','status':'ok'}, {'gtin1':'1','gtin2':'2','status':'ok'}, {'gtin1':'other','gtin2':'pair','status':'ok'}]))
    page=reports.jev()
    assert 'Partially tested' in page
    assert '<script>' not in page and '&lt;script&gt;' in page
    with (root/'results.jsonl').open('a') as fh: fh.write('\n'+json.dumps({'gtin1':'2','gtin2':'1','status':'ok'}))
    assert 'Partially tested' not in reports.jev()

def test_artifact_download_rejects_traversal_and_unknown_round(reports):
    with pytest.raises(HTTPException): reports.artifact('../training_prep.md')
    with pytest.raises(HTTPException): reports.jev(round=99)
    assert reports.artifact('sample_doubled_3.json').filename == 'sample_doubled_3.json'

def test_controlled_repeat_displays_all_four_calls_and_both_formats(reports):
    page=reports.jev(round=5)
    assert 'Same-pair input comparison' in page
    assert 'gate_data' in page and 'original_data' in page
    assert '0.24' in page and '0.06' in page
    assert '>4</td>' in page
    latest=reports.jev()
    assert '<h2>Round 4</h2>' in latest
    assert 'Judgments by input cohort' in latest
