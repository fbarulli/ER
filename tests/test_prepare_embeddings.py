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
    catalog.write_text('product_id\na\nb\n')
    checkpoint = tmp_path / 'model'
    checkpoint.mkdir()
    (checkpoint / 'weights').write_bytes(b'frozen')
    expected = {'catalog_sha256': job.file_hash(catalog),
                'identity_policy_sha256': job.file_hash(POLICY_PATH),
                'identity_dimensions_sha256': job.file_hash(TRAIN_ROOT / 'config/identity_dimensions.yaml'),
                'checkpoint_sha256': job.checkpoint_hash(checkpoint),
                'composition': model_input_composition().model_dump(mode='json')}
    (setup / 'prepared/input_manifest.json').write_text(json.dumps(expected))
    (setup / 'setup_manifest.json').write_text(json.dumps({'text_checkpoint_sha256': expected['checkpoint_sha256']}))
    monkeypatch.setattr(job, 'load_records', lambda _: [{'product_id': 'a'}, {'product_id': 'b'}])
    calls = []

    def create(catalog, checkpoint, output, **kwargs):
        calls.append(kwargs)
        with output.open('wb') as handle:
            np.savez(handle, ids=np.asarray(['a', 'b']), embeddings=np.ones((2, 3), dtype='float32'),
                     metadata=json.dumps(expected))
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
