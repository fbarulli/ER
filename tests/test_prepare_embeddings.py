import json
from pathlib import Path

import numpy as np
import pytest

from training import prepare_embeddings as job


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


@pytest.mark.parametrize('change', ['texts', 'implementation', 'manifest', 'listings'])
def test_reuse_rejects_changed_provenance(tmp_path, monkeypatch, change):
    setup, checkpoint, _ = inputs(tmp_path, monkeypatch)
    job.prepare(setup, checkpoint, device='cpu')
    if change == 'texts':
        monkeypatch.setattr(job, 'compose_texts', lambda _: (['a', 'b'], ['changed', 'text b']))
    elif change == 'implementation':
        monkeypatch.setattr(job, 'composition_fingerprint', lambda: 'changed-parser-code')
    elif change == 'manifest':
        manifest = setup / 'prepared/input_manifest.json'
        manifest.write_text(manifest.read_text() + '\n')
    else:
        (setup / 'prepared/listings.json').write_text('[]')
    with pytest.raises(ValueError, match='stale'):
        job.prepare(setup, checkpoint, device='cpu')


def test_consumer_rejects_tampered_prepared_texts(tmp_path, monkeypatch):
    setup, checkpoint, _ = inputs(tmp_path, monkeypatch)
    job.prepare(setup, checkpoint, device='cpu')
    cache = setup / 'shared_minilm__embeddings.npz'
    _, metadata = job.load_text_cache(cache, ['a', 'b'])
    manifest = json.loads((setup / 'prepared/input_manifest.json').read_text())
    job.validate_prepared_provenance(cache, metadata, manifest)
    request_path = setup / 'embedding_inputs.json'
    request = json.loads(request_path.read_text())
    request['texts'][0] = 'tampered text'
    request_path.write_text(json.dumps(request))
    with pytest.raises(ValueError, match='text content hash'):
        job.validate_prepared_provenance(cache, metadata, manifest)


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


def test_request_digest_binds_result_to_exact_upload(tmp_path, monkeypatch):
    setup, checkpoint, _ = inputs(tmp_path, monkeypatch)
    job.prepare(setup, checkpoint, device='cpu')
    request = job.prepare_request(setup, checkpoint)
    with pytest.raises(ValueError, match='request_sha256'):
        job.validate_result(setup / 'shared_minilm__embeddings.npz', request, request_sha256='another-job')


def test_inputs_changed_during_encoding_are_never_published(tmp_path, monkeypatch):
    setup, checkpoint, _ = inputs(tmp_path, monkeypatch)
    original = job.create_cache
    def changing(*args, **kwargs):
        original(*args, **kwargs)
        (setup / 'eligible_catalog.csv').write_text('sku_id\nchanged\n')
    monkeypatch.setattr(job, 'create_cache', changing)
    with pytest.raises(ValueError, match='stale'):
        job.prepare(setup, checkpoint, device='cpu')
    assert not (setup / 'shared_minilm__embeddings.npz').exists()
