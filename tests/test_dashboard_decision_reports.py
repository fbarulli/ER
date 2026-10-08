import csv
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def reports(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'dashboard'))
    module = importlib.import_module('decision_reports')
    paths = {k:tmp_path / (k + '.json') for k in module.F}
    monkeypatch.setattr(module, 'F', paths)
    monkeypatch.setattr(module, 'TRAIN_ROOT', tmp_path)
    # controlled_influence/_bundle_header discover artifacts under
    # core.common.RESULTS (config-owned, EUROMONITOR_RESULTS_DIR-aware); the
    # dashboard binds it at import, so isolate it to this tmp tree or real
    # results/model_tracks artifacts leak into these assertions.
    monkeypatch.setattr(module, 'RESULTS', tmp_path / 'results')
    monkeypatch.setattr(module, 'runs', lambda: {})
    paths['decision_suite_config'].write_text('setup_dir: setup\ntext_bundle: unused\n')
    paths['decision_ledger'].parent.mkdir(exist_ok=True)
    paths['decision_ledger'] = tmp_path / 'jev/sample_ledger.json'
    paths['decision_ledger'].parent.mkdir()
    paths['decision_ledger'].write_text('[]')
    setup = tmp_path / 'setup'
    (setup / 'prepared').mkdir(parents=True)
    records = [{'sku_id':sku,'split':'train','attribute':{},'numeric':{}} for sku in ['a','b','c']]
    (setup / 'prepared/listings.json').write_text(json.dumps({'schema':'er-graph-listings-v1','listings':records}))
    catalog = [{'sku_id':'a','gtin':'1','sku_name_eng':'<script>','brand':'Alpha','category':'Drink'},
               {'sku_id':'b','gtin':'1','sku_name_eng':'second','brand':'Alpha','category':'Drink'},
               {'sku_id':'c','gtin':'2','sku_name_eng':'third','brand':'Beta','category':'Drink'}]
    write_csv(setup / 'eligible_catalog.csv', catalog)
    write_csv(paths['gate_results'], [{'gtin1':'1','gtin2':'2','gate_decision':'fallback','gate_reason':'review','similarity':'.4'}])
    write_csv(paths['canonical_records'], [{'gtin':'1','volume_set':'[100]','evidence_ledger':'[]'},
                                          {'gtin':'2','volume_set':'[200]','evidence_ledger':'[]'}])
    write_csv(paths['labeled_pairs'], [{'gtin1':'1','gtin2':'2','true_label':'0'}])
    write_csv(setup / 'prepared/pairs.csv', [{'sku_id1':'a','sku_id2':'c','label':'0','split':'train'}])
    return module


def stage_audits(module):
    root = module.F['decision_ledger'].parent
    cases = [{'gtin1':'1','gtin2':'2','input_scope':'gate_data','copy':'a_order'},
             {'gtin1':'2','gtin2':'1','input_scope':'original_data','copy':'b_swapped'}]
    (root / 'sample.json').write_text(json.dumps(cases))
    (root / 'results.jsonl').write_text('\n'.join(json.dumps({**x,'status':'ok','noul':score,'gate':'proceed'})
        for x,score in zip(cases,[.8,.2])) + '\n' + json.dumps({'gtin1':'1','gtin2':'2','input_scope':'unrelated','status':'ok','noul':1}))
    module.F['decision_ledger'].write_text(json.dumps([{'round':7,'sample':'jev/sample.json','checkpoint':'jev/results.jsonl',
                                                      'sample_sha256':module.file_hash(root / 'sample.json')}]))


def test_trace_preserves_order_scope_old_gate_and_new_names(reports):
    stage_audits(reports)
    result = reports.inspect(gtin1='00000000000002',gtin2='1',round=7,scope='gate_data')
    trace = result['traces'][0]
    assert trace['pair'] == ('00000000000001','00000000000002')
    assert trace['gate']['gate_decision'] == 'fallback'
    assert len(trace['jev_history']) == 2
    assert {x['input_scope'] for x in trace['jev_history']} == {'gate_data','original_data'}
    assert all(x['gate'] == 'proceed' and x['sample_checksum_matches'] for x in trace['jev_history'])
    assert trace['prepared_pairs'][0]['sku_id1'] == 'a'
    assert len(trace['listings']['00000000000001']) == 2
    assert result['embedding']['status'] == 'pending'
    assert trace['embedding_cosine'] is None


def test_cosine_uses_all_listing_combinations(reports, monkeypatch):
    vectors = np.array([[1.,0.],[0.,1.],[1.,0.]],dtype=np.float32)
    monkeypatch.setattr(reports,'embedding_state',lambda *args:({'status':'valid','reason':'checked'}, {'a':0,'b':1,'c':2},vectors))
    cosine = reports.inspect()['traces'][0]['embedding_cosine']
    assert cosine == {'min':0.,'mean':.5,'max':1.,'listing_combinations':2}


def test_stale_cache_disables_scores(reports, monkeypatch):
    setup = reports.TRAIN_ROOT / 'setup'
    (setup / 'shared_minilm__embeddings.npz').write_bytes(b'placeholder')
    monkeypatch.setattr(reports,'load_text_cache',lambda *args:(np.ones((3,2)),{}))
    def stale(*args):
        raise ValueError('text cache provenance is stale: catalog_sha256')
    monkeypatch.setattr(reports,'validate_prepared_provenance',stale)
    result = reports.inspect()
    assert result['embedding']['status'] == 'unusable'
    assert 'catalog_sha256' in result['embedding']['reason']
    assert result['traces'][0]['embedding_cosine'] is None


def test_extra_embedding_ids_cannot_be_used(reports, monkeypatch):
    setup = reports.TRAIN_ROOT / 'setup'
    (setup / 'shared_minilm__embeddings.npz').write_bytes(b'placeholder')
    (setup / 'embedding_inputs.json').write_text('{"ids":["a","b","c","extra"]}')
    monkeypatch.setattr(reports,'load_text_cache',lambda *args:(np.ones((3,2)),{}))
    monkeypatch.setattr(reports,'validate_prepared_provenance',lambda *args:None)
    result = reports.inspect()
    assert result['embedding']['status'] == 'unusable'
    assert 'population' in result['embedding']['reason']
    assert result['traces'][0]['embedding_cosine'] is None


def test_checksum_mismatch_and_file_updates_are_live(reports):
    stage_audits(reports)
    reports.F['decision_rebuild_report'].write_text(json.dumps({'source_sha256':{
        reports.F['canonical_records'].name:'old'}}))
    first = reports.inspect()
    assert first['rebuild_provenance'][0]['matches'] is False
    path = reports.F['gate_results']
    path.write_text(path.read_text().replace('fallback','hard_no'))
    (reports.F['decision_ledger'].parent / 'sample.json').write_text('[]')
    second = reports.inspect()
    assert second['traces'][0]['gate']['gate_decision'] == 'hard_no'
    assert second['traces'][0]['jev_history'] == []


def test_safe_html_and_json_nan(reports):
    response = reports.decisions(gtin1='1',gtin2='2')
    assert b'<script>' not in response.body
    assert b'&lt;script&gt;' in response.body
    assert b'Extracted attributes at the gate' in response.body
    assert response.headers['cache-control'] == 'no-store'
    reports.F['decision_training_report'].write_text('{"aggregate":{"metric":NaN}}')
    result = json.loads(reports.decisions_json().body)
    assert result['reports'][0]['report']['aggregate']['metric'] is None


def test_bounds_and_unknown_pairs(reports):
    for args in [{'offset':-1},{'limit':101},{'gtin1':'1'}]:
        with pytest.raises(HTTPException) as exc:
            reports.inspect(**args)
        assert exc.value.status_code == 422
    result = reports.inspect(gtin1='3',gtin2='4')
    assert result['traces'][0]['gate']['gate_decision'] == 'not in current gate population'
    assert not result['traces'][0]['listings']['00000000000003']


def test_ledger_traversal_is_rejected(reports):
    with pytest.raises(ValueError):
        reports.ledger_path('../outside.json')


def test_saved_scores_use_frozen_mapping_and_keep_threshold(reports, monkeypatch):
    root = reports.TRAIN_ROOT / 'run'
    root.mkdir()
    write_csv(root / 'text/_prepared_inputs/canonical_records.csv', [
        {'gtin':'1','source_rows':json.dumps([{'product_id':'historical-a','barcode':'1'}])},
        {'gtin':'2','source_rows':json.dumps([{'sku_id':'historical-b','gtin':'2'}])}])
    write_csv(root / 'hybrid/hybrid__scored_pairs.csv', [
        {'product_id1':'historical-a','product_id2':'historical-b','score':'.61','prediction':'1','split':'dev'}])
    (root / 'hybrid/hybrid__report_manifest.json').write_text(json.dumps({'threshold':.6,'checkpoint_sha256':'saved'}))
    monkeypatch.setattr(reports,'runs',lambda:{'run':root})
    trace = reports.inspect()['traces'][0]
    saved = trace['saved_model_evidence'][0]
    assert saved['mapping_verified'] is True
    assert saved['manifest']['threshold'] == .6
    assert saved['row']['score'] == '.61'
    assert saved['line'] == 2


def test_evidence_change_during_request_rejected(reports, monkeypatch):
    original = reports.report_inventory
    def mutate(available):
        result = original(available)
        reports.F['gate_results'].write_text('changed')
        return result
    monkeypatch.setattr(reports,'report_inventory',mutate)
    with pytest.raises(HTTPException) as exc:
        reports.inspect()
    assert exc.value.status_code == 409


def test_attribute_is_primary_and_uses_shared_engine(reports):
    result = reports.inspect(attribute='volume')
    assert [x['attribute'] for x in result['attribute_tracking']] == ['volume']
    case = result['attribute_tracking'][0]['cases'][0]
    assert case['current_comparison']['result'] == 'CONFLICT'
    assert case['comparison_values'] == {'left':['100.0'],'right':['200.0']}
    assert result['traces'][0]['gate']['gate_decision'] == 'fallback'
    assert 'shared-engine' in result['traces'][0]['attribute_evidence']['status']
    names = [x['attribute'] for x in reports.inspect()['attribute_tracking']]
    assert 'flavour' in names and 'flavor' not in names
    with pytest.raises(HTTPException):
        reports.inspect(attribute='invented')


def test_generation_slices_preserve_unknowns_and_entity_scope(reports):
    visibility = reports.F['decision_visibility']
    visibility.mkdir()
    write_csv(visibility / 'run/mask_visibility.csv', [
        {'fields_hit':'["volume"]','barcode':'1','difficulty_slice':'hard','masking_profile':'baseline',
         'target_mode':'counterfactual','fold':'2','epoch':'3'},
        {'fields_hit':'["package_material"]','barcode':'2','difficulty_slice':'','masking_profile':'',
         'target_mode':'swap_values','fold':'2','epoch':'3'}])
    result = reports.inspect(attribute='volume')
    rows = result['generation_tracking']['combinations']
    volume = next(x for x in rows if x['attribute'] == 'volume')
    assert volume['difficulty_slice'] == 'hard'
    assert volume['generated_data_variant'] == 'counterfactual'
    assert volume['masking_profile'] == 'baseline'
    material = next(x for x in rows if x['attribute'] == 'pack material type')
    assert material['difficulty_slice'] == material['masking_profile'] == 'unknown'
    assert result['attribute_tracking'][0]['generation_slices'][0] == volume
    assert result['traces'][0]['generation_evidence'][0]['association'].startswith('entity-only')
    assert 'not measured' == result['attribute_influence']['status']


def test_legacy_report_attribute_companion_is_explicitly_unusable(reports):
    path = reports.TRAIN_ROOT / 'setup/prepared/report_attributes.json'
    path.write_text('{"schema":"er-report-attributes-v1","attributes":[],"listings":[]}')
    result = reports.inspect()
    assert result['prepared_inputs']['report_attributes']['status'] == 'unusable'
    assert all(not x for x in result['traces'][0]['report_attribute_classes'].values())


def test_changed_loaded_config_is_rejected(reports, tmp_path, monkeypatch):
    config = tmp_path / 'changed.yaml'
    config.write_text('new configuration')
    monkeypatch.setattr(reports,'_LOADED_CONFIG_HASHES',{config:'previous configuration checksum'})
    with pytest.raises(HTTPException) as exc:
        reports.inspect()
    assert exc.value.status_code == 409
    assert 'restart' in exc.value.detail


def test_stale_controlled_influence_is_withheld(reports):
    path = reports.F['decision_ablation_report']
    path.write_text(json.dumps({'schema':'er-attribute-ablation-report-v1','request_path':'missing',
                               'rows':[{'attribute':'volume','score_delta':99}]}))
    result = reports.controlled_influence('volume')
    assert result['rows'] == []
    assert result['status'].startswith('invalid or stale')


def test_controlled_ablation_joins_pair_gate_jev_and_attribute(reports,monkeypatch):
    stage_audits(reports)
    row = {'attribute':'volume','channel':'text','gtin1':'1','gtin2':'2','score_delta':-.2,
           'difficulty_slice':'hard','masking_profile':'targeted','generation_variant':'conflict'}
    monkeypatch.setattr(reports,'controlled_influence',lambda attribute='':{
        'status':'verified frozen-checkpoint intervention','meaning':'test','rows':[row]})
    result = reports.inspect(attribute='volume')
    trace = result['traces'][0]
    assert trace['controlled_ablation_results'] == [row]
    assert trace['gate']['gate_reason'] == 'review'
    assert trace['jev_history']
    assert result['attribute_tracking'][0]['cases'][0]['controlled_ablation_results'] == [row]


# ── the decision surface reads declared names, never its own copies ─────────
# ``saved_run_context`` used to spell ``suite_manifest.json``/``run_tag``, and
# the prepared-setup reads spelled ``eligible_catalog.csv`` etc.; both classes
# of name are declarations owned elsewhere (BundleSpec / preparation.graph_setup),
# so re-pointing the declaration must move this reader.

def test_saved_run_context_follows_the_bundle_spec(reports, tmp_path, monkeypatch):
    from core import common

    spec = common.training_cfg().bundle
    monkeypatch.setattr(spec, 'suite_manifest_file', 'renamed_manifest.json')
    monkeypatch.setattr(spec, 'run_tag_key', 'renamed_tag')
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'renamed_manifest.json').write_text(json.dumps({'renamed_tag': 'run-9'}))

    assert reports.saved_run_context(run, []) is None
    context = reports.saved_run_context(run, ['renamed_manifest.json'])
    assert context['run_tag'] == 'run-9'
    assert context['source'] == 'renamed_manifest.json'


def test_prepared_setup_reads_follow_the_declared_layout(reports, tmp_path, monkeypatch):
    from core import common

    layout = common.prepared_setup_layout()
    monkeypatch.setattr(layout, 'catalog', 'renamed_catalog.csv')
    monkeypatch.setattr(layout, 'listings', 'renamed_listings.json')
    monkeypatch.setattr(layout, 'embedding_request', 'renamed_request.json')
    setup = tmp_path / 'setup'
    (setup / layout.prepared_dir).mkdir(parents=True, exist_ok=True)
    write_csv(setup / 'renamed_catalog.csv', [{'sku_id': 'a', 'gtin': '1'}])
    (setup / layout.prepared_dir / 'renamed_listings.json').write_text(
        json.dumps({'schema': 'er-graph-listings-v1', 'listings': []}))
    (setup / 'renamed_request.json').write_text(json.dumps({'ids': ['a']}))

    assert reports.embedding_state(setup, ['a'], 'missing-model')[0]['path'].endswith(
        layout.shared_embeddings)
    state, _ = reports.request_state(setup, 'missing-model', ['a'])
    assert state.get('source', '').endswith('renamed_request.json')
