import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from model_tracks import ablation as a


def save_vectors(path,**arrays):
    for key in ('vectors','scores','candidate_vectors'):
        if key in arrays:
            arrays[key] = np.asarray(arrays[key],dtype=np.float32)
    np.savez(path,**arrays)


def test_stratified_sample_retains_joint_axes():
    cfg = a.Settings(sample_pairs=4, slice_columns=['difficulty_slice','masking_profile'])
    frame = pd.DataFrame([dict(sku_id1=str(i),sku_id2='b',label=str(i%2),split='dev',
                              difficulty_slice=str(i%2),masking_profile=str(i//2)) for i in range(8)])
    first = a.sample_pairs(frame,cfg)
    assert first == a.sample_pairs(frame,cfg)
    assert len({(r['label'],r['difficulty_slice'],r['masking_profile']) for r in first}) == 4
    assert all(r['split']=='dev' for r in first)


def test_graph_removal_preserves_other_channels_and_context():
    record = {'sku_id':'a','split':'dev','attribute':{'flavor':['orange'],'brand':['A']},'numeric':{'volume_ml':[100],'pack':[2]}}
    changed = a.graph_removed(record,['numeric.volume_ml','attribute.flavor'])
    assert changed['numeric'] == {'pack':[2]}
    assert changed['attribute'] == {'brand':['A']}
    assert record['numeric']['volume_ml'] == [100]
    with pytest.raises(ValueError):
        a.graph_removed(record,['numeric.unknown'])


def test_declaration_scope_keeps_title_and_unrelated_attributes():
    row = {'sku_name_eng':'Orange 500ml', 'attribute':' Volume : 500 ml; Flavour: orange;Health claims: vitamin'}
    result = a.declaration_removed(row,'volume')
    assert result['sku_name_eng'] == row['sku_name_eng']
    assert '500' not in result['attribute']
    assert 'Flavour' in result['attribute']
    assert a.declaration_removed(row,'coffee type') == row


def test_sources_are_checked_for_presence_only(tmp_path,monkeypatch):
    """A changed source is NOT stale; only a missing one is refused.

    Owner directive 2026-10-08: zero freshness checks. ``validate_sources``
    never re-derives the digest a request recorded, so both a differing digest
    and changed content pass; a source that is gone still fails loudly.
    """
    import pytest as _pytest
    source = tmp_path/'pairs.csv'; source.write_text('one')
    monkeypatch.setattr(a,'composition_fingerprint',lambda:'composer')
    request = {'sources':{str(source):a.file_size(source)},'composition':'composer',
               'implementation_size':a.file_size(Path(a.__file__))}
    a.validate_sources(request)
    source.write_text('two')
    a.validate_sources(request)
    request['sources'] = {str(source): 'a-digest-from-another-tree'}
    a.validate_sources(request)
    source.unlink()
    with _pytest.raises(ValueError, match='ablation source missing'):
        a.validate_sources(request)


def test_report_frozen_threshold_flips_ranks_and_unknown_axes(tmp_path,monkeypatch):
    monkeypatch.setattr(a,'validate_sources',lambda request:None)
    monkeypatch.setattr(a,'settings',lambda config=None:a.Settings(report_path=str(tmp_path/'report.json')))
    ckpt = tmp_path/'text-checkpoint'; ckpt.write_bytes(b'weights')
    frozen = tmp_path/'baseline.json'; frozen.write_text(json.dumps({'threshold':.5,'track':'text','checkpoint_size':a.file_size(ckpt)}))
    # A prepared pair's real shape: split plus the evidence map (_pair_evidence).
    # The map must cover every attribute a variant ablates, or the narrowed
    # per-variant evidence is all-None and the row contract requires a scope.
    pair = {'sku_id1':'a','sku_id2':'b','label':'1','split':'dev','difficulty_slice':'hard','masking_profile':None,
            'current_attribute_evidence':{'volume':{'exact_match':True},
                                          'coffee type':{'exact_match':False}}}
    request = {'ids':['a','b','c'],'pairs':[pair],'variants':[
        {'attribute':None,'channel':'baseline','changed_listings':0},
        {'attribute':'volume','channel':'text','changed_listings':1},
        {'attribute':'coffee type','channel':'text','changed_listings':0}],
        'settings':a.Settings().model_dump(),'track':'text','checkpoint':str(ckpt),'sources':{str(ckpt):a.file_size(ckpt)},'composition':'x',
        'implementation_size':'x','intervention':'declaration only','retrieval_scope':'sampled',
        'missing_axes':['masking_profile']}
    path = tmp_path/'request.json'; a.write(path,request)
    base = np.array([[1.,0.],[.8,.6],[0.,1.]])
    altered = np.array([[1.,0.],[0.,1.],[.8,.6]])
    output = tmp_path/'vectors.npz'
    save_vectors(output,vectors=np.stack([base,altered,base]),scores=[[.75],[.25],[.75]],request_size=a.file_size(path))
    report = json.loads(a.report(path,output,.5,threshold_source=str(frozen)).read_text())
    row = report['rows'][0]
    assert row['decision_flip']
    # .75/.25 are exactly representable, so the float32 score_delta the emitter
    # stores equals ablated-baseline exactly. Real ablation deltas are small
    # enough that the same identity holds (Sterbenz); a fixture pairing like
    # .8/.1 would instead expose float32 rounding the real data never sees.
    assert row['score_delta'] == pytest.approx(-.5)
    assert row['baseline_ranks'][0] == 1 and row['ablated_ranks'][0] == 1
    assert row['masking_profile'] is None
    assert report['threshold'] == .5
    assert report['rows'][1]['decision_flip'] is False
    save_vectors(output,vectors=np.stack([base,altered,base]),scores=[[.75],[.25],[.75]],request_size='stale')
    # ZERO freshness checks (owner directive 2026-10-08): a recorded request hash
    # that no longer matches the request file is NOT compared, so the report is
    # produced from the export's own structural identity instead of being refused.
    assert a.report(path,output,.5,threshold_source=str(frozen)).is_file()


def test_threshold_binds_to_attested_track_and_checkpoint(tmp_path,monkeypatch):
    monkeypatch.setattr(a,'validate_sources',lambda request:None)
    monkeypatch.setattr(a,'settings',lambda config=None:a.Settings(report_path=str(tmp_path/'report.json')))
    src = tmp_path/'summary.csv'
    src.write_text('model,split,threshold_source,checkpoint,threshold\nhybrid,dev,dev_youden,hybrid__graph_model.pt,0.5\n')
    ckpt = tmp_path/'hybrid__graph_model.pt'; ckpt.write_bytes(b'ckpt-bytes')
    # A prepared pair's real shape: split plus the evidence map (_pair_evidence).
    pair = {'sku_id1':'a','sku_id2':'b','label':'1','split':'dev','difficulty_slice':'hard','masking_profile':None,
            'current_attribute_evidence':{'volume':{'exact_match':True}}}
    request = {'ids':['a','b','c'],'pairs':[pair],'variants':[
        {'attribute':None,'channel':'baseline','changed_listings':0},
        {'attribute':'volume','channel':'text','changed_listings':1}],
        'settings':a.Settings().model_dump(),'track':'hybrid','checkpoint':str(ckpt),
        'sources':{str(ckpt):a.file_size(ckpt)},'composition':'x',
        'implementation_size':'x','intervention':'declaration only','retrieval_scope':'sampled',
        'missing_axes':[]}
    path = tmp_path/'request.json'; a.write(path,request)
    vectors = np.array([[1.,0.],[.8,.6],[0.,1.]])
    output = tmp_path/'vectors.npz'
    save_vectors(output,vectors=np.stack([vectors,vectors]),scores=[[.8],[.8]],request_size=a.file_size(path))
    report = json.loads(a.report(path,output,.5,threshold_source=str(src)).read_text())
    assert report['threshold_provenance']['track'] == 'hybrid'
    assert report['threshold_provenance']['checkpoint'] == 'hybrid__graph_model.pt'
    bad_track = tmp_path/'bad_track.csv'
    bad_track.write_text('model,split,threshold_source,checkpoint,threshold\ngnn_only,dev,dev_youden,gnn_only__graph_model.pt,0.5\n')
    with pytest.raises(ValueError,match='track differs'):
        a.report(path,output,.5,threshold_source=str(bad_track))
    bad_ckpt = tmp_path/'bad_ckpt.csv'
    bad_ckpt.write_text('model,split,threshold_source,checkpoint,threshold\nhybrid,dev,dev_youden,other__graph_model.pt,0.5\n')
    with pytest.raises(ValueError,match='checkpoint differs'):
        a.report(path,output,.5,threshold_source=str(bad_ckpt))
    ckpt.write_bytes(b'imposter-bytes')
    with pytest.raises(ValueError,match='identity differs'):
        a.report(path,output,.5,threshold_source=str(src))


def test_threshold_manifest_binds_absolute_checkpoint(tmp_path,monkeypatch):
    monkeypatch.setattr(a,'validate_sources',lambda request:None)
    monkeypatch.setattr(a,'settings',lambda config=None:a.Settings(report_path=str(tmp_path/'report.json')))
    ckpt = tmp_path/'text__model.pt'; ckpt.write_bytes(b'text-ckpt')
    manifest = tmp_path/'text__completion_manifest.json'
    manifest.write_text(json.dumps({'checkpoint':str(ckpt),
        'summary':[{'model':'text','split':'dev','threshold_source':'dev_youden','threshold':0.5}]}))
    # A prepared pair's real shape: split plus the evidence map (_pair_evidence).
    pair = {'sku_id1':'a','sku_id2':'b','label':'1','split':'dev','difficulty_slice':'hard','masking_profile':None,
            'current_attribute_evidence':{'volume':{'exact_match':True}}}
    request = {'ids':['a','b','c'],'pairs':[pair],'variants':[
        {'attribute':None,'channel':'baseline','changed_listings':0},
        {'attribute':'volume','channel':'text','changed_listings':1}],
        'settings':a.Settings().model_dump(),'track':'text','checkpoint':str(ckpt),
        'sources':{str(ckpt):a.file_size(ckpt)},'composition':'x',
        'implementation_size':'x','intervention':'declaration only','retrieval_scope':'sampled',
        'missing_axes':[]}
    path = tmp_path/'request.json'; a.write(path,request)
    vectors = np.array([[1.,0.],[.8,.6],[0.,1.]])
    output = tmp_path/'vectors.npz'
    save_vectors(output,vectors=np.stack([vectors,vectors]),scores=[[.8],[.8]],request_size=a.file_size(path))
    report = json.loads(a.report(path,output,.5,threshold_source=str(manifest)).read_text())
    assert report['threshold_provenance']['checkpoint'] == str(ckpt)
    ckpt.write_bytes(b'rotated')
    with pytest.raises(ValueError,match='identity differs'):
        a.report(path,output,.5,threshold_source=str(manifest))


def test_report_evidence_omits_null_key_for_attribute_less_variants(tmp_path,monkeypatch):
    monkeypatch.setattr(a,'validate_sources',lambda request:None)
    monkeypatch.setattr(a,'settings',lambda config=None:a.Settings(report_path=str(tmp_path/'report.json')))
    ckpt = tmp_path/'text-checkpoint'; ckpt.write_bytes(b'weights')
    frozen = tmp_path/'baseline.json'; frozen.write_text(json.dumps({'threshold':.5,'track':'text','checkpoint_size':a.file_size(ckpt)}))
    pair = {'sku_id1':'a','sku_id2':'b','label':'1','split':'dev','difficulty_slice':'hard','masking_profile':None,
            'current_attribute_evidence':{'volume':{'exact_match':True}}}
    request = {'ids':['a','b','c'],'pairs':[pair],'variants':[
        {'attribute':None,'channel':'baseline','changed_listings':0},
        {'attribute':None,'channel':'baseline-replica','changed_listings':0},
        {'attribute':'volume','channel':'text','changed_listings':1}],
        'settings':a.Settings().model_dump(),'track':'text','checkpoint':str(ckpt),'sources':{str(ckpt):a.file_size(ckpt)},'composition':'x',
        'implementation_size':'x','intervention':'declaration only','retrieval_scope':'sampled',
        'missing_axes':['masking_profile']}
    path = tmp_path/'request.json'; a.write(path,request)
    base = np.array([[1.,0.],[.8,.6],[0.,1.]])
    output = tmp_path/'vectors.npz'
    save_vectors(output,vectors=np.stack([base,base,base]),scores=[[.8],[.8],[.8]],request_size=a.file_size(path))
    report = json.loads(a.report(path,output,.5,threshold_source=str(frozen)).read_text())
    replica, volume = report['rows']
    # attribute-less variants carry the pair's full evidence map, never {None: None}
    assert replica['current_attribute_evidence'] == {'volume':{'exact_match':True}}
    assert volume['current_attribute_evidence'] == {'volume':{'exact_match':True}}
    assert 'null' not in json.dumps([row['current_attribute_evidence'] for row in report['rows']])


def test_worker_rejects_an_unsupported_device(tmp_path):
    """The device guard, as it actually stands.

    This used to assert that the worker forbids CPU. That invariant is gone:
    SuiteConfig.device is Literal['cpu', 'cuda'] with 'cuda' only as the
    default, worker.py passes the suite device straight into encode(), and
    worker.py:121 runs the text report lane on the local CPU path. So CPU is a
    supported device here and config/model_tracks.yaml picks CUDA only because
    that is what the Colab run uses.

    What must still hold is that the device is validated against the declared
    Literal rather than passed through to torch, and that requesting CUDA on a
    machine without it fails loudly instead of silently timing on CPU. This
    test also failed for a second, unrelated reason: it wrote '{}' as the
    request, so encode() reached validate_sources() and raised KeyError
    'sources' before the device was ever consulted.
    """
    from model_tracks.embedding_forward import validate_embedding_device

    assert validate_embedding_device('cpu') == 'cpu'
    with pytest.raises(Exception, match='cpu|cuda'):
        validate_embedding_device('tpu')
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match='requires CUDA'):
            validate_embedding_device('cuda')



def test_prepare_uses_real_shared_composer_and_marks_registry_noop(tmp_path,monkeypatch):
    import sys, types, torch
    from core import encoding_inputs
    monkeypatch.setitem(sys.modules,"sentence_transformers",types.SimpleNamespace(SentenceTransformer=lambda *args,**kwargs:object()))
    monkeypatch.setattr(encoding_inputs,"tokenization_policy",lambda model:{"truncation":False})
    monkeypatch.setattr(encoding_inputs,"prepare_text_features",lambda model,texts,**kwargs:{"input_ids":torch.zeros(len(texts),1,dtype=torch.long),"attention_mask":torch.ones(len(texts),1,dtype=torch.long)})
    import yaml
    checkpoint = tmp_path/'checkpoint'; checkpoint.mkdir(); (checkpoint/'frozen').write_text('test')
    catalog = tmp_path/'catalog.csv'; pairs = tmp_path/'pairs.csv'; config = tmp_path/'config.yaml'
    pd.DataFrame([{'sku_id':'a','gtin':'0000000000001','brand':'A','sku_name_eng':'Water','attribute':'Volume: 500 ml'},
                  {'sku_id':'b','gtin':'0000000000002','brand':'A','sku_name_eng':'Water','attribute':'Volume: 1000 ml'}]).to_csv(catalog,index=False)
    pd.DataFrame([{'sku_id1':'a','sku_id2':'b','label':'0','split':'dev'}]).to_csv(pairs,index=False)
    cfg = a.settings().model_dump(); cfg.update(attributes=['volume','coffee type'],output_dir=str(tmp_path/'out'))
    config.write_text(yaml.safe_dump(cfg))
    request_path = a.prepare(catalog,pairs,checkpoint,config=config)
    request = json.loads(request_path.read_text())
    assert request['variants'][1]['changed_listings'] == 2
    assert request['variants'][2]['changed_listings'] == 0
    assert request['variants'][2]['text_indices'] == request['variants'][0]['text_indices']
    assert len(set(request['texts'])) == len(request['texts'])
    assert request['missing_axes']
    a.validate_sources(request)
    pairs.write_text('changed')
    # A changed source is not a freshness verdict: the prepared request stays
    # valid (owner directive 2026-10-08).
    a.validate_sources(request)


def test_prepare_reports_empty_slice_column_as_missing(tmp_path,monkeypatch):
    import sys, types, torch
    from core import encoding_inputs
    monkeypatch.setitem(sys.modules,"sentence_transformers",types.SimpleNamespace(SentenceTransformer=lambda *args,**kwargs:object()))
    monkeypatch.setattr(encoding_inputs,"tokenization_policy",lambda model:{"truncation":False})
    monkeypatch.setattr(encoding_inputs,"prepare_text_features",lambda model,texts,**kwargs:{"input_ids":torch.zeros(len(texts),1,dtype=torch.long),"attention_mask":torch.ones(len(texts),1,dtype=torch.long)})
    import yaml
    checkpoint = tmp_path/'checkpoint'; checkpoint.mkdir(); (checkpoint/'frozen').write_text('test')
    catalog = tmp_path/'catalog.csv'; pairs = tmp_path/'pairs.csv'; config = tmp_path/'config.yaml'
    pd.DataFrame([{'sku_id':'a','gtin':'0000000000001','brand':'A','sku_name_eng':'Water','attribute':'Volume: 500 ml'},
                  {'sku_id':'b','gtin':'0000000000002','brand':'A','sku_name_eng':'Water','attribute':'Volume: 1000 ml'}]).to_csv(catalog,index=False)
    # difficulty_slice is present but empty; masking_profile is absent.
    pd.DataFrame([{'sku_id1':'a','sku_id2':'b','label':'0','split':'dev','difficulty_slice':''},
                  {'sku_id1':'b','sku_id2':'a','label':'1','split':'dev','difficulty_slice':''}]).to_csv(pairs,index=False)
    cfg = a.settings().model_dump(); cfg.update(attributes=['volume'],output_dir=str(tmp_path/'out'),
        slice_columns=['difficulty_slice','masking_profile'])
    config.write_text(yaml.safe_dump(cfg))
    request_path = a.prepare(catalog,pairs,checkpoint,config=config)
    request = json.loads(request_path.read_text())
    assert set(request['missing_axes']) == {'difficulty_slice','masking_profile'}
    assert all(p['difficulty_slice'] is None for p in request['pairs'])
    assert request['pairs'][0]['label'] == '0' and request['pairs'][1]['label'] == '1'


def test_colab_launcher_clones_git_and_releases_on_worker_failure(tmp_path,monkeypatch):
    import sys
    import importlib
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    launcher = importlib.import_module('run_colab_ablation')
    monkeypatch.setattr(launcher,'verify_threshold_binding',lambda *args:{'verified':True})
    worker = tmp_path/'src/model_tracks/ablation.py'; worker.parent.mkdir(parents=True); worker.write_text('worker')
    folder = tmp_path/'results/attribute_ablation/run'; folder.mkdir(parents=True)
    request = folder/'request.json'; request.write_text('{"sources": {}}')
    monkeypatch.setattr(launcher,'TRAIN_ROOT',tmp_path)
    monkeypatch.setattr(launcher,'validate_sources',lambda value:None)
    monkeypatch.setattr(launcher,'frozen_threshold',lambda *args:None)
    monkeypatch.setattr(launcher,'load_prepared',lambda *args:type('Loaded',(),{'close':lambda self:None})())
    (folder/'prepared_inputs.npz').write_bytes(b'prepared')
    monkeypatch.setattr(launcher,'runtime_snapshot_files',lambda:{'src/model_tracks/ablation.py':worker})
    events = []
    for method in ('check_colab_cli','start_live_log','ensure_session','stop','close_live_log'):
        monkeypatch.setattr(launcher.backend,method,lambda method=method:events.append(method))
    monkeypatch.setattr(launcher.backend,'acquire_colab_launch_lock',lambda:'lock')
    monkeypatch.setattr(launcher.backend,'release_colab_launch_lock',lambda lock:events.append('release'))
    for method in ('stop_keep_alive_daemon','prepare_remote_layout','install_deps'):
        monkeypatch.setattr(launcher.backend,method,lambda method=method,**kwargs:events.append(method))
    def detached(stage,command,**kwargs):
        events.append(stage)
        compile(command[-1],'<remote ablation bootstrap>','exec')
        assert 'tarfile' in command[-1] and 'ablation.py' in command[-1]
        raise RuntimeError('worker failed')
    monkeypatch.setattr(launcher.backend,'run_detached_stage',detached)
    with pytest.raises(RuntimeError,match='worker failed'):
        launcher.main(request,threshold=.5,threshold_source='report',publisher=lambda paths,message:events.append('publish'))
    assert events.index('publish') < events.index('prepare_remote_layout') < events.index('attribute_ablation')
    assert events[-3:] == ['stop','close_live_log','release']


def _git_repo(tmp_path):
    import subprocess
    for command in (['git','init','-q'],):
        subprocess.run(command,cwd=tmp_path,check=True)
    subprocess.run(['git','config','user.email','t@t'],cwd=tmp_path,check=True)
    subprocess.run(['git','config','user.name','t'],cwd=tmp_path,check=True)
    return subprocess


def test_launcher_preflights_untracked_directory_checkpoint(tmp_path,monkeypatch):
    import sys, importlib
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    launcher = importlib.import_module('run_colab_ablation')
    monkeypatch.setattr(launcher,'verify_threshold_binding',lambda *args:{'verified':True})
    monkeypatch.setattr(launcher,'TRAIN_ROOT',tmp_path)
    monkeypatch.setattr(launcher,'resolve',lambda p: Path(p) if Path(p).is_absolute() else tmp_path/Path(p))
    _git_repo(tmp_path)
    committed = tmp_path/'results'/'text__checkpoint'; committed.mkdir(parents=True)
    (committed/'config.json').write_text('{}')
    import subprocess
    subprocess.run(['git','add','-A'],cwd=tmp_path,check=True)
    subprocess.run(['git','commit','-q','-m','init'],cwd=tmp_path,check=True)
    fresh = tmp_path/'results'/'fresh__checkpoint'; fresh.mkdir()
    (fresh/'config.json').write_text('{}')
    request = {'sources':{'results/text__checkpoint':'x','results/fresh__checkpoint':'x'}}
    request_path = tmp_path/'request.json'; request_path.write_text(json.dumps(request))
    monkeypatch.setattr(launcher,'validate_sources',lambda value:None)
    monkeypatch.setattr(launcher,'load_prepared',lambda *args:type('Loaded',(),{'close':lambda self:None})())
    monkeypatch.setattr(launcher,'frozen_threshold',lambda *args:None)
    with pytest.raises(ValueError,match='not committed to the current branch'):
        launcher.main(request_path,threshold=.5,threshold_source='report')


def test_launcher_accepts_committed_directory_checkpoint(tmp_path,monkeypatch):
    import sys, importlib
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    launcher = importlib.import_module('run_colab_ablation')
    monkeypatch.setattr(launcher,'verify_threshold_binding',lambda *args:{'verified':True})
    monkeypatch.setattr(launcher,'TRAIN_ROOT',tmp_path)
    monkeypatch.setattr(launcher,'resolve',lambda p: Path(p) if Path(p).is_absolute() else tmp_path/Path(p))
    _git_repo(tmp_path)
    committed = tmp_path/'results'/'text__checkpoint'; committed.mkdir(parents=True)
    (committed/'config.json').write_text('{}')
    import subprocess
    subprocess.run(['git','add','-A'],cwd=tmp_path,check=True)
    subprocess.run(['git','commit','-q','-m','init'],cwd=tmp_path,check=True)
    request = {'sources':{'results/text__checkpoint':'x'}}
    request_path = tmp_path/'request.json'; request_path.write_text(json.dumps(request))
    worker = tmp_path/'src/model_tracks/ablation.py'; worker.parent.mkdir(parents=True); worker.write_text('worker')
    (tmp_path/'prepared_inputs.npz').write_bytes(b'prepared')
    monkeypatch.setattr(launcher,'validate_sources',lambda value:None)
    monkeypatch.setattr(launcher,'load_prepared',lambda *args:type('Loaded',(),{'close':lambda self:None})())
    monkeypatch.setattr(launcher,'frozen_threshold',lambda *args:None)
    monkeypatch.setattr(launcher,'runtime_snapshot_files',lambda:{})
    monkeypatch.setattr(launcher,'push_artifacts',lambda paths,message:None)
    # the pre-flight passes; the launch stops at the (absent) Colab CLI
    def absent_cli():
        raise RuntimeError('colab cli absent')
    monkeypatch.setattr(launcher.backend,'check_colab_cli',absent_cli)
    with pytest.raises(RuntimeError,match='colab cli absent'):
        launcher.main(request_path,threshold=.5,threshold_source='report')


def test_launcher_reports_once_and_persists_validated_output(tmp_path,monkeypatch):
    import sys, importlib
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    launcher = importlib.import_module('run_colab_ablation')
    monkeypatch.setattr(launcher,'verify_threshold_binding',lambda *args:{'verified':True})
    folder = tmp_path/'results/attribute_ablation/run'; folder.mkdir(parents=True)
    request = folder/'request.json'; request.write_text('{"sources": {}}')
    (folder/'prepared_inputs.npz').write_bytes(b'prepared')
    worker = tmp_path/'src/model_tracks/ablation.py'; worker.parent.mkdir(parents=True); worker.write_text('worker')
    monkeypatch.setattr(launcher,'TRAIN_ROOT',tmp_path)
    monkeypatch.setattr(launcher,'validate_sources',lambda value:None)
    monkeypatch.setattr(launcher,'load_prepared',lambda *args:type('Loaded',(),{'close':lambda self:None})())
    monkeypatch.setattr(launcher,'frozen_threshold',lambda *args:{'path':'report','size':'x','selection':'saved'})
    monkeypatch.setattr(launcher,'runtime_snapshot_files',lambda:{})
    monkeypatch.setattr(launcher,'push_artifacts',lambda paths,message:None)
    validated = {'schema':'er-attribute-ablation-report-v1','rows':[],'threshold':.5}
    calls = []
    lifecycle = []
    monkeypatch.setattr(launcher,'validate_vectors',lambda *args:lifecycle.append('validate'))
    def fake_report(request_path,result,threshold,threshold_source,save=True):
        assert result == folder/'vectors.npz' and result.is_file()
        assert lifecycle == ['validate','stop']
        calls.append(save)
        return validated if not save else 'recomputed-path'
    monkeypatch.setattr(launcher,'report',fake_report)
    persisted = []
    monkeypatch.setattr(launcher,'save_report',lambda request_path,output,config=None: persisted.append(output) or 'handoff-path')
    def fake_persist(result,handoff,publisher=None,additional_files=None,namespace=None,prefix=None):
        persisted.append(('persist',str(result),handoff))
        return 'published'
    monkeypatch.setattr(launcher,'persist_embeddings',fake_persist)
    for method in ('check_colab_cli','start_live_log','ensure_session','stop','close_live_log'):
        monkeypatch.setattr(launcher.backend,method,lambda method=method:None)
    monkeypatch.setattr(launcher.backend,'stop',lambda:lifecycle.append('stop'))
    monkeypatch.setattr(launcher.backend,'acquire_colab_launch_lock',lambda:'lock')
    monkeypatch.setattr(launcher.backend,'release_colab_launch_lock',lambda lock:None)
    for method in ('stop_keep_alive_daemon','prepare_remote_layout','install_deps'):
        monkeypatch.setattr(launcher.backend,method,lambda method=method,**kwargs:None)
    monkeypatch.setattr(launcher.backend,'run_detached_stage',lambda stage,command,timeout=None:None)
    monkeypatch.setattr(launcher.backend,'run_colab_exec_capture',lambda session,script,timeout=None:'{}')
    monkeypatch.setattr(launcher.backend,'_parse_remote_json',lambda raw:{'size': len(b'vectors')})
    def fake_download(remote,local):
        local.write_bytes(b'vectors')
    monkeypatch.setattr(launcher.backend,'_download_one_remote_file',fake_download)
    monkeypatch.setattr(launcher.backend,'LIVE_LOG_PATH',tmp_path/'nonexistent_live_log')
    original_gpu = launcher.backend.GPU
    try:
        outcome = launcher.main(request,threshold=.5,threshold_source='report')
    finally:
        launcher.backend.GPU = original_gpu
    assert calls == [False]  # report computed exactly once, validation-only
    assert persisted[0] is validated  # the validated output is persisted, never recomputed
    assert persisted[1] == ('persist',str(folder/'vectors.npz'),'handoff-path')
    assert outcome == 'published'
    assert (folder/'vectors.npz').read_bytes() == b'vectors'
    lifecycle.clear()
    monkeypatch.setattr(launcher,'report',lambda *args,**kwargs:validated)
    assert launcher.main(request,threshold=.5,threshold_source='report') == 'published'
    assert lifecycle == []
    assert len(persisted) == 4


def test_threshold_requires_identity_even_if_numeric_value_matches(tmp_path):
    source = tmp_path/'baseline.json'; source.write_text('{"threshold":0.5}')
    checkpoint = tmp_path/'weights'; checkpoint.write_bytes(b'weights')
    request = {'track':'text','checkpoint':str(checkpoint),'sources':{str(checkpoint):a.file_size(checkpoint)}}
    with pytest.raises(ValueError,match='missing'):
        a.verify_threshold_binding(request,a.frozen_threshold(str(source),.5))
    source.write_text('{"track":"text","threshold":0.5}')
    with pytest.raises(ValueError,match='missing'):
        a.verify_threshold_binding(request,a.frozen_threshold(str(source),.5))


def test_full_catalog_has_candidates_outside_sample(tmp_path):
    from model_tracks.ablation_retrieval import RetrievalComparison
    vectors = np.asarray([[1.,0.],[.6,.8],[.99,np.sqrt(1-.99**2)]],dtype=np.float32)
    request = {'track':'text','ids':['a','b'],'pairs':[{'sku_id1':'a','sku_id2':'b'}]}
    comparison = RetrievalComparison(['a','b','extra'],vectors,request,tmp_path/'request.json',a.Settings(retrieval_ks=[1,2]))
    try:
        assert comparison.ranks(vectors[:2])[0][0] == 2
        hits = comparison.ann_hits(vectors[:2])
        assert hits[0]['1'][0] is False
        assert hits[0]['2'][0] is True
    finally:
        comparison.close()


# ── the emitted rows validate against the frozen threshold ─────────────────

def _straddle_row():
    """A legitimate no-flip row whose scores straddle the draft's hardcoded 0.0.

    Under the frozen threshold used below (0.7) both scores sit on the same
    side, so ``decision_flip`` is genuinely False; the rejected draft's
    predicate comparing against 0.0 called it a flip and rejected the row.
    """
    return {'sku_id1':'a','sku_id2':'b','label':'0','split':'train','attribute':None,'channel':None,
        'baseline_score':-0.20,'ablated_score':0.30,'score_delta':0.50,'decision_flip':False,
        'baseline_error':False,'ablated_error':True,'changed_listings':1,
        'endpoint_input_changed':[False,False],'embedding_cosine_delta':[0.0,0.0],
        'baseline_ranks':[1],'ablated_ranks':[1],'ann_baseline_hits':None,'ann_ablated_hits':None,
        'known_positive_recall_change':{},'current_attribute_evidence':{'x':None},
        'evidence_scope':'frozen payload; raw-row evidence unavailable'}


def test_the_flip_is_decided_by_the_frozen_threshold_not_a_hardcoded_zero():
    row = _straddle_row()
    assert a.assert_ablation_rows([row],0.7)[0] is row
    with pytest.raises(ValueError,match='decision_flip'):
        a.assert_ablation_rows([dict(row,decision_flip=True)],0.7)
    # 0.0 sits between the two scores, so a 0.0-shaped predicate inverts the row
    with pytest.raises(ValueError,match='decision_flip'):
        a.assert_ablation_rows([row],0.0)


def test_the_real_ablation_report_matches_the_frozen_threshold():
    path = Path(__file__).resolve().parents[1]/'results/attribute_ablation/report.json'
    if not path.is_file():
        pytest.skip('results/attribute_ablation/report.json not present')
    report = json.loads(path.read_text(encoding='utf-8'))
    rows = report['rows']
    assert a.assert_ablation_rows(rows,report['threshold']) is rows
    assert sum(row['decision_flip'] for row in rows) == 5


def test_report_refuses_a_wrong_flip_before_saving(tmp_path,monkeypatch):
    monkeypatch.setattr(a,'validate_sources',lambda request:None)
    monkeypatch.setattr(a,'settings',lambda config=None:a.Settings(report_path=str(tmp_path/'report.json')))
    ckpt = tmp_path/'text-checkpoint'; ckpt.write_bytes(b'weights')
    frozen = tmp_path/'baseline.json'; frozen.write_text(json.dumps({'threshold':.5,'track':'text','checkpoint_size':a.file_size(ckpt)}))
    pair = {'sku_id1':'a','sku_id2':'b','label':'1','split':'dev',
            'current_attribute_evidence':{'volume':{'exact_match':True}}}
    request = {'ids':['a','b','c'],'pairs':[pair],'variants':[
        {'attribute':None,'channel':'baseline','changed_listings':0},
        {'attribute':'volume','channel':'text','changed_listings':1}],
        'settings':a.Settings().model_dump(),'track':'text','checkpoint':str(ckpt),
        'sources':{str(ckpt):a.file_size(ckpt)},'composition':'x',
        'implementation_size':'x','intervention':'declaration only','retrieval_scope':'sampled',
        'missing_axes':[]}
    path = tmp_path/'request.json'; a.write(path,request)
    vectors = np.array([[1.,0.],[.8,.6],[0.,1.]])
    result = tmp_path/'vectors.npz'
    save_vectors(result,vectors=np.stack([vectors,vectors]),scores=[[.75],[.25]],request_size=a.file_size(path))
    saved = []
    monkeypatch.setattr(a,'_comparison_rows',lambda *args,**kwargs:[dict(_straddle_row(),decision_flip=True)])
    monkeypatch.setattr(a,'save_report',lambda *args,**kwargs: saved.append(args) or 'handoff')
    with pytest.raises(ValueError,match='decision_flip'):
        a.report(path,result,.5,threshold_source=str(frozen))
    assert saved == []
