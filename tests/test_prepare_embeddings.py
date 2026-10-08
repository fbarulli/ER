import json
from pathlib import Path

import numpy as np
import pytest

from training import prepare_embeddings as job


def _data_parallel_like_stubs():
    """Fresh stubs: tokenizer + [0].auto_model + config, as _align_model_token_ids reads them."""
    from types import SimpleNamespace

    class _Inner(list):
        tokenizer = SimpleNamespace(pad_token_id=0, bos_token_id=1, eos_token_id=2)

    inner = _Inner([SimpleNamespace(
        auto_model=SimpleNamespace(
            config=SimpleNamespace(pad_token_id=5, bos_token_id=None, eos_token_id=5),
            generation_config=SimpleNamespace(pad_token_id=9, bos_token_id=9, eos_token_id=9),
        )
    )])
    wrapped = SimpleNamespace(module=inner)
    return inner, wrapped


def _alignment_state(model):
    from training.training import _align_model_token_ids
    _align_model_token_ids(model)
    auto_model = model[0].auto_model
    return json.dumps({'config': vars(auto_model.config),
                       'generation_config': vars(auto_model.generation_config)},
                      sort_keys=True, default=repr)


def test_checkpoint_token_alignment_accepts_data_parallel_wrappers():
    """GPU v28: on multi-GPU kernels HF wraps the SentenceTransformer in
    DataParallel, which is not subscriptable and crashed the checkpoint
    publisher (`'DataParallel' object is not subscriptable`). The duck-typed
    `.module` unwrap must make alignment succeed on the wrapper and land in
    exactly the same state as alignment of the bare model."""
    from training.training import _align_model_token_ids

    bare, wrapped = _data_parallel_like_stubs()
    _align_model_token_ids(wrapped.module)
    assert wrapped.module[0].auto_model.config.pad_token_id == 0
    assert wrapped.module[0].auto_model.config.bos_token_id == 1
    assert wrapped.module[0].auto_model.config.eos_token_id == 2
    assert wrapped.module[0].auto_model.generation_config.pad_token_id == 0

    # the unwrapped path must be byte-identical: same end state via the wrapper
    direct, also_wrapped = _data_parallel_like_stubs()
    assert _alignment_state(also_wrapped.module) == _alignment_state(direct)


def inputs(tmp_path, monkeypatch):
    from core.common import TRAIN_ROOT
    from core.identity_policy import POLICY_PATH
    from core.model_input import model_input_composition
    setup = tmp_path / 'setup'
    (setup / 'prepared').mkdir(parents=True)
    catalog = setup / 'eligible_catalog.csv'
    catalog.write_text('sku_id\na\nb\n')
    checkpoint = tmp_path / 'model'
    checkpoint.mkdir()
    (checkpoint / 'weights').write_bytes(b'frozen')
    listings = setup / 'prepared/listings.json'
    listings.write_text(json.dumps([{'sku_id': 'a'}, {'sku_id': 'b'}]))
    expected = {'listings_sha256': job.file_hash(listings), 'catalog_sha256': job.file_hash(catalog),
                'identity_policy_sha256': job.file_hash(POLICY_PATH),
                'identity_dimensions_sha256': job.file_hash(TRAIN_ROOT / 'config/identity_dimensions.yaml'),
                'checkpoint_sha256': job.checkpoint_hash(checkpoint),
                'composition': model_input_composition().model_dump(mode='json')}
    (setup / 'prepared/input_manifest.json').write_text(json.dumps(expected))
    (setup / 'setup_manifest.json').write_text(json.dumps({'text_checkpoint_sha256': expected['checkpoint_sha256']}))
    monkeypatch.setattr(job, 'load_records', lambda _: [{'sku_id': 'a'}, {'sku_id': 'b'}])
    monkeypatch.setattr(job, 'compose_texts', lambda _: (['a', 'b'], ['text a', 'text b']))
    calls = []

    def create(catalog, checkpoint, output, **kwargs):
        calls.append(kwargs)
        with output.open('wb') as handle:
            np.savez(handle, ids=np.asarray(['a', 'b']), embeddings=np.full((2, 3), 1 / np.sqrt(3), dtype='float32'),
                     metadata=json.dumps(kwargs['input_metadata']))
    monkeypatch.setattr(job, 'create_cache', create)
    return setup, checkpoint, calls


def test_generate_reuse_and_reject_stale_cache(tmp_path, monkeypatch):
    setup, checkpoint, calls = inputs(tmp_path, monkeypatch)
    assert job.prepare(setup, checkpoint, device='cpu')['status'] == 'created'
    assert job.prepare(setup, checkpoint, device='cpu')['status'] == 'reused'
    assert len(calls) == 1
    (checkpoint / 'weights').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checkpoint differs'):
        job.prepare(setup, checkpoint, device='cpu')


def test_failed_generation_does_not_publish_cache(tmp_path, monkeypatch):
    setup, checkpoint, _ = inputs(tmp_path, monkeypatch)
    def fail(*args, **kwargs):
        args[2].write_bytes(b'partial')
        raise RuntimeError('interrupted')
    monkeypatch.setattr(job, 'create_cache', fail)
    with pytest.raises(RuntimeError, match='interrupted'):
        job.prepare(setup, checkpoint, device='cpu')
    assert not (setup / 'shared_minilm__embeddings.npz').exists()
    assert not list(setup.glob('embedding-job-*'))


def test_cuda_job_refuses_cpu_fallback(tmp_path, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA unavailable'):
        job.prepare(tmp_path, tmp_path)


@pytest.mark.parametrize('change', ['texts', 'manifest', 'listings'])
def test_reuse_rebuilds_an_incompatible_cache(tmp_path, monkeypatch, change):
    """An incompatible cache is REBUILT, never refused (owner directive 2026-10-08).

    There is no staleness gate left: changed texts, a rewritten prepared
    manifest, or a rewritten listings file make the existing export not match
    this request, so it is rebuilt in place instead of failing the run.
    """
    setup, checkpoint, _ = inputs(tmp_path, monkeypatch)
    job.prepare(setup, checkpoint, device='cpu')
    if change == 'texts':
        monkeypatch.setattr(job, 'compose_texts', lambda _: (['a', 'b'], ['changed', 'text b']))
    elif change == 'manifest':
        manifest = setup / 'prepared/input_manifest.json'
        manifest.write_text(manifest.read_text() + '\n')
    else:
        (setup / 'prepared/listings.json').write_text('[]')
    assert job.prepare(setup, checkpoint, device='cpu')['status'] == 'created'


def test_reuse_accepts_changed_composition_implementation(tmp_path, monkeypatch):
    # owner order 2026-10-07: the composition_implementation_sha256 field stays
    # recorded, but is never compared; fingerprint drift must not block reuse.
    setup, checkpoint, _ = inputs(tmp_path, monkeypatch)
    job.prepare(setup, checkpoint, device='cpu')
    monkeypatch.setattr(job, 'composition_fingerprint', lambda: 'changed-parser-code')
    assert job.prepare(setup, checkpoint, device='cpu')['status'] == 'reused'


def test_consumer_rejects_tampered_prepared_texts(tmp_path, monkeypatch):
    setup, checkpoint, _ = inputs(tmp_path, monkeypatch)
    job.prepare(setup, checkpoint, device='cpu')
    cache = setup / 'shared_minilm__embeddings.npz'
    _, metadata = job.load_text_cache(cache, ['a', 'b'])
    job.validate_prepared_provenance(cache, metadata)
    request_path = setup / 'embedding_inputs.json'
    request = json.loads(request_path.read_text())
    request['texts'][0] = 'tampered text'
    request_path.write_text(json.dumps(request))
    with pytest.raises(ValueError, match='text content hash'):
        job.validate_prepared_provenance(cache, metadata)


@pytest.mark.parametrize('ids,vectors', [(['b', 'a'], [[1, 0], [0, 1]]),
                                      (['a', 'b', 'extra'], [[1, 0]] * 3),
                                      (['a', 'b'], [[0, 0], [1, 0]]),
                                      (['a', 'b'], [[float('nan'), 0], [1, 0]]),
                                      (['a', 'b'], [[2, 0], [1, 0]])])
def test_invalid_gpu_result_cannot_be_accepted(tmp_path, monkeypatch, ids, vectors):
    setup, checkpoint, _ = inputs(tmp_path, monkeypatch)
    request = job.prepare_request(setup, checkpoint)
    candidate = tmp_path / 'candidate.npz'
    np.savez(candidate, ids=ids, embeddings=np.asarray(vectors, dtype='float32'), metadata=json.dumps(request['metadata']))
    with pytest.raises(ValueError):
        job.validate_result(candidate, request)


def test_no_request_digest_freshness_gate(tmp_path, monkeypatch):
    """ZERO freshness checks (owner directive 2026-10-08): the result contract
    has no request-hash staleness parameter at all, and a recorded request hash
    that disagrees with the request file is ignored rather than compared."""
    import inspect

    assert 'request_sha256' not in inspect.signature(job.validate_result).parameters
    setup, checkpoint, _ = inputs(tmp_path, monkeypatch)
    job.prepare(setup, checkpoint, device='cpu')
    request = job.prepare_request(setup, checkpoint)
    cache = setup / 'shared_minilm__embeddings.npz'
    with np.load(cache, allow_pickle=False) as data:
        ids, vectors = data['ids'], data['embeddings']
        metadata = json.loads(str(data['metadata']))
    metadata['request_sha256'] = 'a-request-hash-from-another-run'
    np.savez(cache, ids=ids, embeddings=vectors, metadata=json.dumps(metadata))
    job.validate_result(cache, request)


def test_inputs_changed_during_encoding_are_never_published(tmp_path, monkeypatch):
    setup, checkpoint, _ = inputs(tmp_path, monkeypatch)
    original = job.create_cache
    def changing(*args, **kwargs):
        original(*args, **kwargs)
        (setup / 'eligible_catalog.csv').write_text('sku_id\nchanged\n')
    monkeypatch.setattr(job, 'create_cache', changing)
    with pytest.raises(ValueError, match='changed during generation'):
        job.prepare(setup, checkpoint, device='cpu')
    assert not (setup / 'shared_minilm__embeddings.npz').exists()


def test_data_gate_census_tracks_are_json_serializable():
    from model_tracks.data_gate import DataGateResult, TrackInputCensus, census_tracks
    from model_tracks.telemetry import WorkerEvents

    gate = DataGateResult(suite={'preflight': True},
                          tracks={'hybrid': TrackInputCensus(
                              listings=2, pairs={'train': {'positive': 1, 'negative': 1}},
                              text_dimension=3, device='cuda')},
                          attestation='a' * 64)
    payload = census_tracks(gate)
    events = WorkerEvents(Path('log'), 'suite', 'census-test', filename='census.jsonl')
    events.emit('data_gate', 'passed', tracks=payload, attestation=gate.attestation)
    reloaded = json.loads(events.path.read_text())
    assert reloaded['tracks']['hybrid']['listings'] == 2
    assert reloaded['tracks']['hybrid']['device'] == 'cuda'


def test_data_gate_census_tracks_are_json_serializable(tmp_path):
    from model_tracks.data_gate import DataGateResult, TrackInputCensus, census_tracks
    from model_tracks.telemetry import WorkerEvents

    gate = DataGateResult(suite={'preflight': True},
                          tracks={'hybrid': TrackInputCensus(
                              listings=2, pairs={'train': {'positive': 1, 'negative': 1}},
                              text_dimension=3, device='cuda')},
                          attestation='a' * 64)
    payload = census_tracks(gate)
    events = WorkerEvents(tmp_path, 'suite', 'census-test', filename='census.jsonl')
    events.emit('data_gate', 'passed', tracks=payload, attestation=gate.attestation)
    reloaded = json.loads((tmp_path / 'census.jsonl').read_text())
    assert reloaded['tracks']['hybrid']['listings'] == 2
    assert reloaded['tracks']['hybrid']['device'] == 'cuda'
