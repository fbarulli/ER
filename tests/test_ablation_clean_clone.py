"""Clean-clone restoration and dashboard verification for ablation artifacts.

Verifies that ablation reports, prepared inputs, and threshold bindings can be
restored and validated from a fresh clone of the saved artifacts, even when
TRAIN_ROOT differs from the original workspace.
"""
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _composition_fingerprint():
    from graph_tracks.text_cache import composition_fingerprint
    return composition_fingerprint()


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True, indent=2) + '\n')


def _make_portable_request(tmp_path, monkeypatch):
    """Create a minimal portable ablation request with sources in a local_inputs dir.

    The portable_setup mechanism resolves @setup paths relative to
    <suite>/local_inputs/<portable_setup>. The request lives at
    <tmp_path>/suite/text/ablation/request.json, so suite = <tmp_path>/suite
    and setup = <tmp_path>/suite/local_inputs/local_inputs.
    """
    setup_dir = tmp_path / 'suite' / 'local_inputs' / 'local_inputs'
    setup_dir.mkdir(parents=True)

    catalog = setup_dir / 'catalog.csv'
    catalog.write_text('sku_id,gtin,sku_name_eng\na,1,Alpha\nb,2,Beta\n')

    pairs = setup_dir / 'pairs.csv'
    pairs.write_text('sku_id1,sku_id2,label,split\na,b,1,dev\n')

    checkpoint = setup_dir / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'model.pt').write_text('fake checkpoint')

    config = setup_dir / 'config.yaml'
    config.write_text('sample_pairs: 10\n')

    from graph_tracks.text_cache import checkpoint_hash
    source_files = [catalog, pairs, checkpoint, config]
    sources = {}
    for f in source_files:
        if f.is_dir():
            sources['@setup/' + f.name] = checkpoint_hash(f)
        else:
            sources['@setup/' + f.name] = _sha256(f)

    request = {
        'schema': 'er-attribute-ablation-v2',
        'track': 'text',
        'checkpoint_role': 'selected',
        'settings': {
            'sample_pairs': 10, 'seed': 1729, 'split': 'dev', 'batch_size': 256,
            'accelerator': 'T4', 'attributes': ['volume'], 'output_dir': 'results/attribute_ablation',
            'report_path': 'results/attribute_ablation/report.json', 'retrieval_catalog': 'full',
            'hnsw_m': 16, 'hnsw_ef_construction': 200, 'hnsw_ef_search': 100,
            'retrieval_ks': [1, 5, 10], 'graph_fields': {}, 'slice_columns': [],
        },
        'portable_setup': 'local_inputs',
        'sources': sources,
        'composition': _composition_fingerprint(),
        'implementation_sha256': _sha256(Path(__file__).parents[1] / 'src/model_tracks/ablation.py'),
        'checkpoint': '@setup/checkpoint',
        'text_checkpoint': None,
        'text_checkpoint': None,
        'candidate_ids': ['a', 'b'],
        'candidate_text_indices': [0, 1],
        'candidate_records': [],
        'ids': ['a', 'b'],
        'texts': ['Alpha 500ml', 'Beta 330ml'],
        'pairs': [{'sku_id1': 'a', 'sku_id2': 'b', 'label': '1', 'split': 'dev',
                    'gtin1': '1', 'gtin2': '2', 'current_attribute_evidence': {}}],
        'variants': [
            {'attribute': None, 'channel': 'baseline', 'text_indices': [0, 1], 'records': [], 'changed_listings': 0},
            {'attribute': 'volume', 'channel': 'text', 'text_indices': [0, 1], 'records': [], 'changed_listings': 0},
        ],
        'intervention': 'declared attribute removed; title/brand and training graph context fixed',
        'retrieval_scope': 'fixed full catalog; query-only interventions; incomplete known-positive truth',
        'missing_axes': [],
    }
    return request, setup_dir


def _make_prepared_inputs(setup_dir):
    """Create a minimal prepared_inputs.npz."""
    prepared = setup_dir / 'prepared_inputs.npz'
    np.savez(prepared, tokens=np.zeros((2, 10), dtype=np.int64))
    return prepared


def _make_vectors(setup_dir):
    """Create a minimal vectors.npz with normalized vectors."""
    vectors = np.array([
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.9, 0.1], [0.1, 0.9]],
    ], dtype=np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)
    scores = np.array([[0.8], [0.7]], dtype=np.float32)
    candidate_vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    candidate_vectors = candidate_vectors / np.linalg.norm(candidate_vectors, axis=-1, keepdims=True)
    result = setup_dir / 'vectors.npz'
    np.savez(result, vectors=vectors, scores=scores, candidate_vectors=candidate_vectors)
    return result


def _make_threshold_source(setup_dir):
    """Create a baseline threshold source file."""
    from graph_tracks.text_cache import checkpoint_hash
    checkpoint_hash_value = checkpoint_hash(setup_dir / 'checkpoint')
    threshold_file = setup_dir / 'baseline_threshold.json'
    _write_json(threshold_file, {
        'track': 'text',
        'checkpoint': 'checkpoint',
        'threshold': 0.5,
        'checkpoint_sha256': checkpoint_hash_value,
    })
    return threshold_file


def _make_report(tmp_path, request_path, result_path, threshold_source, request, result):
    """Create a minimal ablation report."""
    from graph_tracks.text_cache import checkpoint_hash
    setup_dir = request_path.parent.parent.parent / 'local_inputs' / 'local_inputs'
    actual_checkpoint_hash = checkpoint_hash(setup_dir / 'checkpoint')
    report = {
        'schema': 'er-attribute-ablation-report-v1',
        'track': 'text',
        'checkpoint_role': 'selected',
        'request_path': str(request_path),
        'request_sha256': _sha256(request_path),
        'result_path': str(result_path),
        'result_sha256': _sha256(result_path),
        'sources': request['sources'],
        'composition': request['composition'],
        'implementation_sha256': request['implementation_sha256'],
        'embedding_dtype': 'float32',
        'threshold': 0.5,
        'threshold_source': str(threshold_source),
        'threshold_provenance': {
            'path': str(threshold_source),
            'sha256': _sha256(threshold_source),
            'selection': 'saved dev calibration; no refit',
            'track': 'text',
            'checkpoint': 'checkpoint',
            'checkpoint_sha256': actual_checkpoint_hash,
        },
        'threshold_binding': {
            'track': 'text',
            'checkpoint_sha256': actual_checkpoint_hash,
            'verified': True,
        },
        'split': 'dev',
        'sample_pairs': 1,
        'intervention': request['intervention'],
        'retrieval_scope': request['retrieval_scope'],
        'missing_axes': [],
        'retrieval_catalog_count': 2,
        'retrieval_intervention': 'query only; fixed candidates',
        'rows': [{
            'sku_id1': 'a', 'sku_id2': 'b', 'label': '1',
            'attribute': 'volume', 'channel': 'text',
            'baseline_score': 0.8, 'ablated_score': 0.7, 'score_delta': -0.1,
            'decision_flip': False,
            'embedding_cosine_delta': [0.0, 0.0],
            'baseline_ranks': [[1, 2]], 'ablated_ranks': [[1, 2]],
            'ann_baseline_hits': [{'1': [True, True], '5': [True, True], '10': [True, True]}],
            'ann_ablated_hits': [{'1': [True, True], '5': [True, True], '10': [True, True]}],
            'known_positive_recall_change': {'1': [0, 0], '5': [0, 0], '10': [0, 0]},
        }],
    }
    return report


@pytest.fixture
def portable_ablation(tmp_path, monkeypatch):
    """Create a complete portable ablation artifact set in a simulated clean clone."""
    request, setup_dir = _make_portable_request(tmp_path, monkeypatch)
    prepared = _make_prepared_inputs(setup_dir)
    result = _make_vectors(setup_dir)
    threshold_source = _make_threshold_source(setup_dir)

    prepared_data = np.load(prepared)
    request['prepared_inputs'] = {
        'sha256': _sha256(prepared),
        'shape': list(prepared_data['tokens'].shape),
    }

    suite_dir = tmp_path / 'suite' / 'text' / 'ablation'
    suite_dir.mkdir(parents=True, exist_ok=True)
    request_path = suite_dir / 'request.json'
    _write_json(request_path, request)

    import shutil
    shutil.copy2(prepared, suite_dir / 'prepared_inputs.npz')
    shutil.copy2(result, suite_dir / 'vectors.npz')
    shutil.copy2(threshold_source, suite_dir / 'baseline_threshold.json')

    report = _make_report(tmp_path, request_path, result, threshold_source, request, result)
    report_path = suite_dir / 'report.json'
    _write_json(report_path, report)

    return {
        'tmp_path': tmp_path,
        'setup_dir': setup_dir,
        'suite_dir': suite_dir,
        'request_path': request_path,
        'result_path': result,
        'threshold_source': threshold_source,
        'report_path': report_path,
        'report': report,
        'request': request,
    }


def test_portable_request_context_resolves_setup_paths(portable_ablation, monkeypatch):
    """Verify that @setup paths resolve correctly within request_context."""
    from model_tracks import ablation as a

    request_path = portable_ablation['request_path']
    request = portable_ablation['request']

    with a.request_context(request_path):
        resolved = a.resolve('@setup/checkpoint')
        assert resolved == portable_ablation['setup_dir'] / 'checkpoint'
        assert resolved.is_dir()


def test_portable_request_context_resolves_suite_paths(portable_ablation, monkeypatch):
    """Verify that @suite paths resolve correctly within request_context."""
    from model_tracks import ablation as a

    request_path = portable_ablation['request_path']

    with a.request_context(request_path):
        resolved = a.resolve('@suite/text/ablation/request.json')
        assert resolved == request_path


def test_validate_sources_passes_in_portable_context(portable_ablation, monkeypatch):
    """Verify that validate_sources works with portable_setup paths."""
    from model_tracks import ablation as a

    request_path = portable_ablation['request_path']
    request = portable_ablation['request']

    with a.request_context(request_path):
        a.validate_sources(request)


def test_load_prepared_validates_checksum(portable_ablation, monkeypatch):
    """Verify that load_prepared validates the prepared inputs checksum."""
    from model_tracks import ablation as a

    request_path = portable_ablation['request_path']
    request = portable_ablation['request']

    with a.request_context(request_path):
        loaded = a.load_prepared(request_path, request)
        assert loaded is not None
        loaded.close()


def test_controlled_report_validates_portable_ablation(portable_ablation, monkeypatch):
    """Verify that _controlled_report validates a portable ablation report."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'dashboard'))
    module = importlib.import_module('decision_reports')

    report_path = portable_ablation['report_path']
    result = module._controlled_report(report_path)

    assert result['status'] == 'verified frozen-checkpoint intervention'
    assert len(result['rows']) == 1
    assert result['rows'][0]['attribute'] == 'volume'


def test_controlled_influence_discovers_portable_report(portable_ablation, monkeypatch):
    """Verify that controlled_influence discovers and validates portable reports."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'dashboard'))
    module = importlib.import_module('decision_reports')

    pointer = portable_ablation['tmp_path'] / 'decision_ablation_report.json'
    _write_json(pointer, {'path': str(portable_ablation['report_path'])})

    monkeypatch.setattr(module, 'F', {**module.F, 'decision_ablation_report': pointer})
    monkeypatch.setattr(module, 'TRAIN_ROOT', portable_ablation['tmp_path'])

    # controlled_influence discovers reports from results/model_tracks/*/*/ablation/report.json
    # or from the pointer's glob pattern. Place a copy at the expected discovery path.
    discovery_dir = portable_ablation['tmp_path'] / 'results' / 'model_tracks' / 'text' / 'selected' / 'ablation'
    discovery_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(portable_ablation['report_path'], discovery_dir / 'report.json')
    shutil.copy2(portable_ablation['request_path'], discovery_dir / 'request.json')
    shutil.copy2(portable_ablation['request_path'].parent / 'prepared_inputs.npz', discovery_dir / 'prepared_inputs.npz')
    shutil.copy2(portable_ablation['request_path'].parent / 'vectors.npz', discovery_dir / 'vectors.npz')
    shutil.copy2(portable_ablation['request_path'].parent / 'baseline_threshold.json', discovery_dir / 'baseline_threshold.json')

    result = module.controlled_influence()
    assert result['status'] == 'verified frozen-checkpoint intervention'
    assert len(result['rows']) == 1


def test_clean_clone_with_different_train_root(portable_ablation, monkeypatch):
    """Verify that ablation artifacts validate when TRAIN_ROOT is a different directory.

    This simulates a clean clone where the workspace root differs from the
    original training workspace.
    """
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'dashboard'))
    module = importlib.import_module('decision_reports')

    clean_clone_root = portable_ablation['tmp_path'] / 'clean_clone'
    clean_clone_root.mkdir()

    monkeypatch.setattr(module, 'TRAIN_ROOT', clean_clone_root)

    report_path = portable_ablation['report_path']
    result = module._controlled_report(report_path)

    assert result['status'] == 'verified frozen-checkpoint intervention'


def test_stale_source_hash_fails_closed(portable_ablation, monkeypatch):
    """Verify that a modified source file fails validation."""
    from model_tracks import ablation as a

    request_path = portable_ablation['request_path']
    request = portable_ablation['request']

    catalog = portable_ablation['setup_dir'] / 'catalog.csv'
    catalog.write_text('sku_id,gtin,sku_name_eng\na,1,Alpha MODIFIED\nb,2,Beta\n')

    with a.request_context(request_path):
        with pytest.raises(ValueError, match='ablation source changed'):
            a.validate_sources(request)


def test_stale_request_hash_fails_closed(portable_ablation, monkeypatch):
    """Verify that a modified request file fails validation."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'dashboard'))
    module = importlib.import_module('decision_reports')

    request_path = portable_ablation['request_path']
    request = json.loads(request_path.read_text())
    request['intervention'] = 'modified intervention'
    _write_json(request_path, request)

    report_path = portable_ablation['report_path']
    result = module._controlled_report(report_path)

    assert result['status'].startswith('invalid or stale')


def test_stale_result_hash_fails_closed(portable_ablation, monkeypatch):
    """Verify that a modified result file fails validation."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'dashboard'))
    module = importlib.import_module('decision_reports')

    result_path = portable_ablation['result_path']
    vectors = np.load(result_path)
    modified = {k: vectors[k] for k in vectors.files}
    modified['scores'] = np.array([[0.9], [0.8]], dtype=np.float32)
    np.savez(result_path, **modified)

    report_path = portable_ablation['report_path']
    result = module._controlled_report(report_path)

    assert result['status'].startswith('invalid or stale')


def test_threshold_binding_mismatch_fails_closed(portable_ablation, monkeypatch):
    """Verify that a threshold binding mismatch fails validation."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'dashboard'))
    module = importlib.import_module('decision_reports')

    report_path = portable_ablation['report_path']
    report = json.loads(report_path.read_text())
    report['threshold_binding']['checkpoint_sha256'] = 'wrong_hash'
    _write_json(report_path, report)

    result = module._controlled_report(report_path)

    assert result['status'].startswith('invalid or stale')


def test_report_without_portable_setup_fails_in_clean_clone(portable_ablation, monkeypatch):
    """Verify that a report without portable_setup fails when TRAIN_ROOT differs."""
    from model_tracks import ablation as a

    request_path = portable_ablation['request_path']
    request = json.loads(request_path.read_text())
    del request['portable_setup']
    _write_json(request_path, request)

    with pytest.raises(ValueError, match='portable ablation source requires verified request context'):
        with a.request_context(request_path):
            a.resolve('@setup/checkpoint')
