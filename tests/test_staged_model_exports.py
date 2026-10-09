"""Selected-weight binding keeps local plans and restored provenance intact."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from model_tracks import ablation, staged_ablation, text_export


def test_bound_request_preserves_prepared_tensor_hash_and_relative_checkpoint(tmp_path,monkeypatch):
    import core.common
    monkeypatch.setattr(core.common, 'TRAIN_ROOT', tmp_path)
    setup = tmp_path/'setup';template = setup/'ablation_templates/gnn_only';template.mkdir(parents=True)
    checkpoint = tmp_path/'run/gnn_only/checkpoint.pt';checkpoint.parent.mkdir(parents=True)
    payload = {'vocabulary':{},'support_records':[],'manifest':{'track':'gnn_only'}}
    torch.save(payload,checkpoint)
    tensors = template/'prepared_inputs.npz';tensors.write_bytes(b'frozen local topology')
    request = {'checkpoint':'@setup/template.pt',
               'sources':{'@setup/template.pt':'placeholder'},'settings':{'retrieval_catalog':'full'},'prepared_inputs':{'size':ablation.file_size(tensors)}}
    (template/'request.json').write_text(json.dumps(request))
    calls = []
    monkeypatch.setattr(staged_ablation,'encode',lambda *args,**kwargs:calls.append((args,kwargs)))
    path = staged_ablation.forward(checkpoint.parent,setup,'gnn_only',checkpoint,device='cuda')
    bound = json.loads(path.read_text())
    assert bound['checkpoint'] == '@suite/gnn_only/checkpoint.pt'
    assert bound['sources'] == {bound['checkpoint']:ablation.file_size(checkpoint)}
    assert bound['prepared_inputs'] == request['prepared_inputs']
    assert (path.parent/'prepared_inputs.npz').read_bytes() == tensors.read_bytes()
    assert len(calls) == 1 and calls[0][1]['device'] == 'cuda'


def test_portable_sources_resolve_after_restoring_suite_and_inputs(tmp_path,monkeypatch):
    suite = tmp_path/'restored/run'
    request_path = suite/'text/ablation/request.json';request_path.parent.mkdir(parents=True)
    setup = suite/'local_inputs/data/model_tracks/shared';setup.mkdir(parents=True)
    catalog = setup/'eligible_catalog.csv';catalog.write_bytes(b'original catalog')
    checkpoint = suite/'text/checkpoint/weights';checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'selected')
    request = {'portable_setup':'data/model_tracks/shared',
               'sources':{'@setup/eligible_catalog.csv':ablation.file_size(catalog),
                          '@suite/text/checkpoint/weights':ablation.file_size(checkpoint)},
               'composition':'composer','implementation_size':ablation.file_size(Path(ablation.__file__))}
    request_path.write_text(json.dumps(request))
    monkeypatch.setattr(ablation,'composition_fingerprint',lambda:'composer')
    with ablation.request_context(request_path):
        ablation.validate_sources(request)
        assert ablation.resolve('@suite/text/checkpoint/weights') == checkpoint
        with pytest.raises(ValueError,match='unsafe'):
            ablation.resolve('@setup/../../../../outside')
    catalog.write_bytes(b'changed')
    with ablation.request_context(request_path):
        # A changed source is not a freshness verdict (owner directive
        # 2026-10-08): the request still validates.
        ablation.validate_sources(request)


def test_text_report_has_no_model_forward_and_requires_saved_gpu_export(tmp_path,monkeypatch):
    from model_tracks import text_report
    from graph_tracks import text_cache
    from training import validation_inference
    monkeypatch.setattr(text_cache,'create_cache',lambda *args,**kwargs:pytest.fail('local model forward forbidden'))
    monkeypatch.setattr(validation_inference,'resolve_best_checkpoint',lambda output:(tmp_path/'checkpoint',{}))
    # Missing fixed prepared population fails before any cache generation.
    with pytest.raises(FileNotFoundError):
        text_report.complete(tmp_path,tmp_path,device='cpu',report_test=False)


def test_gpu_worker_exports_vectors_and_ablations_before_completion(tmp_path,monkeypatch):
    from model_tracks import worker,resume
    from core import common
    from training import prepared_bundle,validation_inference
    setup = tmp_path/'setup';setup.mkdir()
    (setup/'setup_manifest.json').write_text('{}')
    # Templates present (the historical GPU-forward path): the worker still
    # reads and forwards them even under ER_GPU_TRAINING_ONLY=1.
    (setup/'ablation_templates'/'text').mkdir(parents=True)
    (setup/'ablation_templates'/'text'/'request.json').write_text('{}')
    output = tmp_path/'run/text'
    monkeypatch.setenv('EUROMONITOR_RESULTS_DIR',str(output))
    monkeypatch.setenv('ER_GPU_TRAINING_ONLY','1')
    monkeypatch.setenv('ER_TRACK_BARRIER',str(tmp_path/'barrier'))
    monkeypatch.setattr(common,'TRAIN_ROOT',tmp_path)
    cfg = SimpleNamespace(setup_dir='setup',text_bundle='bundle',text_model='baseline',epochs=1,
                          device='cuda',report_test=False,post_training_ablation=True)
    monkeypatch.setattr(worker,'load_config',lambda _:cfg)
    monkeypatch.setattr(prepared_bundle,'load_prepared_bundle',lambda _: (SimpleNamespace(payload_variant='full'),{}))
    # worker.py reads the bundle's typed sidecar header (bundle -> bundle.json)
    # to build the trainer command before the barrier; stubbing the full
    # loader does not cover that read, so the sidecar must exist on disk.
    import json as _json
    from training.prepared_bundle import PreparedBundleManifest as _Manifest
    _header = {
        'payload_variant': 'full', 'masking_profile': 'baseline',
        'model_input': {'profile': 'cleaned', 'include_evidence': False},
        'n_df': 1, 'n_payload': 1, 'n_pos': 1, 'n_neg': 1, 'n_train_neg': 1,
        'n_labeled_pairs_bytes': 1, 'n_canonical_records_bytes': 1,
        'n_gate_results_bytes': 1, 'size': '0' * 64,
    }
    _Manifest.model_validate(_header)   # fail here, not three frames deep
    (tmp_path/'bundle.json').write_text(_json.dumps(_header))
    monkeypatch.setattr(worker,'wait_for_start',lambda *args:None)
    order = []
    monkeypatch.setattr(worker.subprocess,'run',lambda *args,**kwargs:order.append('train'))
    monkeypatch.setattr(text_export,'forward',lambda *args,**kwargs:(order.append('vectors'),object()))
    monkeypatch.setattr(staged_ablation,'forward',lambda *args,**kwargs:order.append('ablations'))
    from model_tracks import text_report
    monkeypatch.setattr(text_report,'build_index',lambda *args,**kwargs:order.append('index'))
    monkeypatch.setattr(validation_inference,'resolve_best_checkpoint',lambda _:(output/'checkpoint',{}))
    monkeypatch.setattr(resume,'record_completion',lambda *args,**kwargs:order.append('complete'))
    events = SimpleNamespace(emit=lambda *args,**kwargs:None)
    worker._run(tmp_path/'suite.yaml','text','run-text',resume=False,events=events)
    # The ANN index the same-suite cascade consumes must be built BEFORE the
    # completion marker, and only the index: reports/scoring stay deferred to
    # the local finalize.
    assert order == ['train','vectors','ablations','index','complete']


def test_gpu_worker_skips_ablation_export_without_bundle_templates(tmp_path,monkeypatch):
    """47f0641 ships the bundle without ablation templates; the GPU worker
    records a named skip instead of reading a missing request.json. Local
    lanes (no ER_GPU_TRAINING_ONLY) keep the loud FileNotFoundError."""
    from model_tracks import worker,resume
    from core import common
    from training import prepared_bundle,validation_inference
    setup = tmp_path/'setup';setup.mkdir()
    (setup/'setup_manifest.json').write_text('{}')
    output = tmp_path/'run/text'
    monkeypatch.setenv('EUROMONITOR_RESULTS_DIR',str(output))
    monkeypatch.setenv('ER_GPU_TRAINING_ONLY','1')
    monkeypatch.setenv('ER_TRACK_BARRIER',str(tmp_path/'barrier'))
    monkeypatch.setattr(common,'TRAIN_ROOT',tmp_path)
    cfg = SimpleNamespace(setup_dir='setup',text_bundle='bundle',text_model='baseline',epochs=1,
                          device='cuda',report_test=False,post_training_ablation=True)
    monkeypatch.setattr(worker,'load_config',lambda _:cfg)
    monkeypatch.setattr(prepared_bundle,'load_prepared_bundle',lambda _: (SimpleNamespace(payload_variant='full'),{}))
    import json as _json
    from training.prepared_bundle import PreparedBundleManifest as _Manifest
    _header = {
        'payload_variant': 'full', 'masking_profile': 'baseline',
        'model_input': {'profile': 'cleaned', 'include_evidence': False},
        'n_df': 1, 'n_payload': 1, 'n_pos': 1, 'n_neg': 1, 'n_train_neg': 1,
        'n_labeled_pairs_bytes': 1, 'n_canonical_records_bytes': 1,
        'n_gate_results_bytes': 1, 'size': '0' * 64,
    }
    _Manifest.model_validate(_header)
    (tmp_path/'bundle.json').write_text(_json.dumps(_header))
    monkeypatch.setattr(worker,'wait_for_start',lambda *args:None)
    order = []
    emitted = []
    monkeypatch.setattr(worker.subprocess,'run',lambda *args,**kwargs:order.append('train'))
    monkeypatch.setattr(text_export,'forward',lambda *args,**kwargs:(order.append('vectors'),object()))
    def record(*args,**kwargs):
        order.append('ablations')
        pytest.fail('staged_ablation.forward must not run on templateless GPU sessions')
    monkeypatch.setattr(staged_ablation,'forward',record)
    from model_tracks import text_report
    monkeypatch.setattr(text_report,'build_index',lambda *args,**kwargs:order.append('index'))
    monkeypatch.setattr(validation_inference,'resolve_best_checkpoint',lambda _:(output/'checkpoint',{}))
    monkeypatch.setattr(resume,'record_completion',lambda *args,**kwargs:order.append('complete'))
    events = SimpleNamespace(emit=lambda *args,**kwargs:emitted.append((args,kwargs)))
    worker._run(tmp_path/'suite.yaml','text','run-text',resume=False,events=events)
    # The templateless session skips the ablation export but still builds the
    # index the same-suite cascade consumes.
    assert order == ['train','vectors','index','complete']
    skips = [kwargs for args,kwargs in emitted if args[:2] == ('attribute_ablation_export','skipped')]
    assert skips and skips[0]['reason'] == 'bundle shipped no ablation templates'


def test_local_worker_keepsloud_ablation_template_error(tmp_path,monkeypatch):
    """Local completion keeps the loud read: no ER_GPU_TRAINING_ONLY means a
    missing template is still a hard FileNotFoundError."""
    from model_tracks import worker,resume
    from core import common
    from training import prepared_bundle,validation_inference
    from model_tracks import staged_ablation
    setup = tmp_path/'setup';setup.mkdir()
    (setup/'setup_manifest.json').write_text('{}')
    output = tmp_path/'run/text'
    monkeypatch.setenv('EUROMONITOR_RESULTS_DIR',str(output))
    monkeypatch.delenv('ER_GPU_TRAINING_ONLY',raising=False)
    monkeypatch.setenv('ER_TRACK_BARRIER',str(tmp_path/'barrier'))
    monkeypatch.setattr(common,'TRAIN_ROOT',tmp_path)
    cfg = SimpleNamespace(setup_dir='setup',text_bundle='bundle',text_model='baseline',epochs=1,
                          device='cpu',report_test=False,post_training_ablation=True)
    monkeypatch.setattr(worker,'load_config',lambda _:cfg)
    monkeypatch.setattr(prepared_bundle,'load_prepared_bundle',lambda _: (SimpleNamespace(payload_variant='full'),{}))
    import json as _json
    from training.prepared_bundle import PreparedBundleManifest as _Manifest
    _header = {
        'payload_variant': 'full', 'masking_profile': 'baseline',
        'model_input': {'profile': 'cleaned', 'include_evidence': False},
        'n_df': 1, 'n_payload': 1, 'n_pos': 1, 'n_neg': 1, 'n_train_neg': 1, 'n_labeled_pairs_bytes': 1,
        'n_canonical_records_bytes': 1, 'n_gate_results_bytes': 1, 'size': '0' * 64,
    }
    (tmp_path/'bundle.json').write_text(_json.dumps(_header))
    _Manifest.model_validate(_header)
    monkeypatch.setattr(worker,'wait_for_start',lambda *args:None)
    monkeypatch.setattr(worker.subprocess,'run',lambda *args,**kwargs:None)
    monkeypatch.setattr(text_export,'forward',lambda *args,**kwargs:(None,object()))
    monkeypatch.setattr(validation_inference,'resolve_best_checkpoint',lambda _:(output/'checkpoint',{}))
    with pytest.raises(FileNotFoundError):
        worker._run(tmp_path/'suite.yaml','text','run-text',resume=False,events=SimpleNamespace(emit=lambda *args,**kwargs:None))


def test_pending_baseline_validates_native_tokens_without_cache_or_model(tmp_path,monkeypatch):
    from model_tracks import baseline_export
    setup = tmp_path/'setup';setup.mkdir()
    tokens = setup/'prepared_text.npz'
    np.savez(tokens,**{'text/0/input_ids':np.asarray([[1,2]],dtype=np.int64),
                      'text/0/attention_mask':np.ones((1,2),dtype=np.int64)})
    request = {'schema':'er-embedding-request-v2','ids':['a'],'texts':['fixed'],
        'metadata':{'checkpoint_size':'frozen','text_size':baseline_export.texts_size(['fixed'])},
        'prepared_text':{'size':ablation.file_size(tokens),'truncated_inputs':0,'token_lengths':[2],
            'tokenization':{'input_token_limit':512,'truncation':False,'truncate_dim':None},'token_batches':[
                {'prefix':'text/0','keys':['input_ids','attention_mask'],'constants':{},'start':0,'count':1}]}}
    (setup/'embedding_inputs.json').write_text(json.dumps(request))
    monkeypatch.setattr(baseline_export,'input_identity',lambda *args:{'checkpoint_size':'frozen'})
    monkeypatch.setattr(baseline_export,'load_records',lambda *args:[{'sku_id':'a'}])
    from core import encoding_inputs
    monkeypatch.setattr(encoding_inputs,'tokenization_policy',lambda _:request['prepared_text']['tokenization'])
    assert not (setup/'shared_minilm__embeddings.npz').exists()
    assert baseline_export.validate_pending(setup,tmp_path/'checkpoint',native_model=object())['status'] == 'prepared GPU pending'
    tokens.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='tokens changed'):
        baseline_export.validate_pending(setup,tmp_path/'checkpoint',native_model=object())
