"""Offline checks for the remote program and local cache handoff."""
import ast
import importlib.util
import json
from pathlib import Path
import types
from unittest.mock import patch


def launcher():
    path = Path(__file__).resolve().parents[1] / 'scripts/run_colab_embeddings.py'
    spec = importlib.util.spec_from_file_location('embedding_launcher', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def test_cpu_requires_explicit_smoke_scope():
    import pytest
    module, _ = launcher()
    with pytest.raises(ValueError,match='explicit embedding smoke'):
        module.main(device='cpu')


def test_remote_program_executes_without_indentation_error(tmp_path):
    _, path = launcher()
    assignment = next(node for node in ast.walk(ast.parse(path.read_text()))
                      if isinstance(node, ast.Assign) and any(
                          isinstance(target, ast.Name) and target.id == 'script'
                          for target in node.targets))
    package = tmp_path / 'inputs.tar.gz'
    import tarfile
    source = tmp_path / 'encode.py'
    source.write_text('')
    with tarfile.open(package, 'w:gz') as archive:
        archive.add(source, arcname='encode.py')
    script = eval(compile(ast.Expression(assignment.value), '<launcher>', 'eval'),
                  {'remote_package': str(package), 'remote_checkpoint': '/content/EuromonitoR/artifacts/models/all-MiniLM-L6-v2', 'job': str(tmp_path), 'package': package, 'device':'cuda', 'file_hash': lambda _: __import__('hashlib').sha256(package.read_bytes()).hexdigest()})
    with patch('subprocess.run') as run:
        exec(compile(script, '<remote>', 'exec'), {})
    args, kwargs = run.call_args
    assert args[0][1] == str(tmp_path / 'encode.py')
    assert '--request' in args[0]
    assert 'training.prepare_embeddings' not in script
    assert kwargs['check'] is True


def test_local_handoff_validates_cache_before_suite(tmp_path):
    module, _ = launcher()
    cache = tmp_path / 'data/track_setup/shared_minilm__embeddings.npz'
    calls = []
    def validate(*args, **kwargs):
        calls.append('cache')
        assert args[0] == cache
        assert args[1] == {}
        return {'status': 'reused', 'sha256': 'verified'}
    def preflight(config):
        calls.append('suite')
        assert config == tmp_path / 'config/model_tracks.yaml'
        return {'hybrid': {'listings': 27820}}
    with patch.object(module, 'TRAIN_ROOT', tmp_path), \
         patch('core.common.resolve_model', return_value=str(tmp_path / 'checkpoint')), \
         patch.object(module, 'validate_result', side_effect=validate), \
         patch.object(module, 'file_hash', return_value='verified'), \
         patch('model_tracks.preflight.preflight', side_effect=preflight):
        module.complete_local_handoff(cache, {})
    assert calls == ['cache', 'suite']
    report = json.loads((tmp_path / 'results/embedding_job/local_handoff.json').read_text())
    assert report['status'] == 'complete'
    assert report['cache'] == str(cache)


def test_stale_local_cache_blocks_handoff(tmp_path):
    import pytest
    module, _ = launcher()
    with patch.object(module, 'TRAIN_ROOT', tmp_path), \
         patch('core.common.resolve_model', return_value=str(tmp_path / 'checkpoint')), \
         patch.object(module, 'validate_result', side_effect=ValueError('stale cache')), \
         patch('model_tracks.preflight.preflight') as preflight:
        with pytest.raises(ValueError, match='stale cache'):
            module.complete_local_handoff(tmp_path / 'cache.npz', {})
    preflight.assert_not_called()
    assert not (tmp_path / 'results/embedding_job/local_handoff.json').exists()


def test_embedding_save_reuses_existing_git_artifact_flow(tmp_path):
    module, _ = launcher()
    setup = tmp_path / 'setup'
    (setup / 'prepared').mkdir(parents=True)
    for name in ('embedding_inputs.json', 'eligible_catalog.csv', 'setup_manifest.json',
                 'shared_minilm__embeddings.npz', 'prepared/input_manifest.json'):
        (setup / name).write_bytes(name.encode())
    handoff = tmp_path / 'handoff.json'
    handoff.write_text('{}')
    captured = []
    def existing_publisher(paths, message):
        archive = paths[0]
        run_tag = archive.parent.name
        assert message.startswith('embeddings: save verified cache ')
        assert archive.name == module.backend._RESULT_ARCHIVE_NAME
        assert archive.name.endswith('.tar.gz')
        import tarfile
        with tarfile.open(archive, 'r:gz') as result:
            manifest = json.load(result.extractfile(module.backend._RESULT_MANIFEST_NAME))
            assert manifest['run_id'] == run_tag
            paths = {item['path'] for item in manifest['included']}
            assert 'shared_minilm__embeddings.npz' in paths
            assert 'embedding_inputs.json' in paths
            assert 'local_handoff.json' in paths
        captured.append(archive)
        return archive
    with patch.object(module, 'TRAIN_ROOT', tmp_path), \
         patch('model_tracks.publish.push_artifacts', side_effect=existing_publisher):
        module.persist_embeddings(setup / 'shared_minilm__embeddings.npz', handoff)
    assert len(captured) == 1


def test_valid_embeddings_preserve_failed_training_handoff_evidence(tmp_path,monkeypatch):
    module,_ = launcher()
    monkeypatch.setattr(module,'TRAIN_ROOT',tmp_path)
    monkeypatch.setattr(module,'validate_result',lambda *args,**kwargs:None)
    monkeypatch.setattr(module,'file_hash',lambda *args:'verified')
    cache = tmp_path/'vectors.npz'; cache.write_bytes(b'vectors')
    from model_tracks import preflight
    def blocked(*args):
        raise ValueError('diet coverage gate failed')
    monkeypatch.setattr(preflight,'preflight',blocked)
    result = module.complete_local_handoff(cache,{})
    document = json.loads(result.read_text())
    assert document['status'] == 'blocked'
    assert document['sha256'] == 'verified'
    assert document['preflight']['error'] == 'diet coverage gate failed'
