"""Focused pins for the model_tracks no-freshness rewrite (owner directive 2026-10-08).

Every test here states one behavior the rewrite owns:

* a recorded digest is never re-derived and compared to decide an artifact is
  stale (:mod:`model_tracks.ablation`, :mod:`model_tracks.text_export`,
  :mod:`dashboard.decision_reports`);
* an incompatible cached intermediate is rebuilt silently instead of failing
  (:mod:`model_tracks.bundle_steps`, :mod:`model_tracks.ablation`);
* the cascade text-index prereq stays built in the gpu_only path (pinned in
  ``tests/test_staged_model_exports.py``).

Bundle identity (``core.bundle.Bundle.load``) is untouched and tested elsewhere.
"""
import json
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

#: The enforcement messages/keys the rewrite removed from the owned files.
#: Their ABSENCE is the proof that no freshness verdict is computed there.
REMOVED_ENFORCEMENT = (
    "text export tokens changed",
    "text export source changed",
    "ablation source changed",
    "restored prepared input changed",
    "restored frozen baseline differs",
    "Configuration changed since startup",
    "prepared tensors differ",
    "existing request differs",
)

OWNED_FILES = (
    "src/model_tracks/preflight.py",
    "src/model_tracks/run.py",
    "src/model_tracks/bundle_steps.py",
    "src/model_tracks/ablation.py",
    "src/model_tracks/post_training_ablation.py",
    "src/model_tracks/text_export.py",
    "src/model_tracks/text_report.py",
    "src/model_tracks/worker.py",
    "src/model_tracks/colab.py",
    "dashboard/decision_reports.py",
)


def test_no_removed_freshness_enforcement_remains_in_owned_files():
    offenders = {name: [message for message in REMOVED_ENFORCEMENT if message in (ROOT/name).read_text()]
                 for name in OWNED_FILES}
    assert {name: found for name, found in offenders.items() if found} == {}


def test_no_recorded_hash_is_compared_against_a_recomputed_one_in_owned_files():
    comparison = re.compile(r"request_size['\"]?\]?\s*[!=]=|['\"]request_size['\"]\s*\)?\s*[!=]=")
    offenders = [name for name in OWNED_FILES if comparison.search((ROOT/name).read_text())]
    assert offenders == []


def test_validate_sources_is_presence_only(tmp_path):
    from model_tracks import ablation
    source = tmp_path / 'catalog.csv'
    source.write_text('one')
    request = {'sources': {str(source): 'a-digest-from-another-tree'}}
    # A recorded digest that no longer matches, and changed content, are both
    # accepted: neither is a freshness verdict.
    ablation.validate_sources(request)
    source.write_text('two')
    ablation.validate_sources(request)
    source.unlink()
    with pytest.raises(ValueError, match='ablation source missing'):
        ablation.validate_sources(request)


class _FakeForward:
    """Stands in for PreparedEmbeddingForward and records what it was handed."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.export_location = 'Colab CPU'
        self.embedding_dtype = 'float32'
        _FakeForward.instances.append(self)

    def forward(self, *, model=None):
        import types
        model = types.SimpleNamespace(_er_forward_performance={'sections': {}})
        return np.zeros((2, 3), dtype=np.float32), model, 'checkpoint-sha', 'request-sha'

    def write(self, path, ids, vectors, metadata, validate):
        Path(path).write_bytes(b'written')
        return path


def test_text_forward_ignores_recorded_source_digests(tmp_path, monkeypatch):
    from training import validation_inference
    from model_tracks import text_export

    setup = tmp_path / 'setup'; setup.mkdir()
    output = tmp_path / 'run'; output.mkdir()
    layout = text_export._setup_layout()
    request = {
        'schema': 'er-text-export-v1', 'ids': ['a', 'b'],
        'plan': {'tokenization': {'padding': 'longest'}},
        'tokens_size': '0' * 64, 'catalog_size': '1' * 64, 'listings_size': '2' * 64,
        'pairs_size': '3' * 64, 'text_size': '4' * 64, 'composition': {},
        'composition_implementation_size': '5' * 64,
        'export_implementation_size': '6' * 64, 'token_implementation_size': '7' * 64,
    }
    (setup / layout.text_export_request).write_text(json.dumps(request))
    # The tokens bytes match NONE of the recorded digests, and the catalog /
    # listings / pairs files do not even exist: forward must still run.
    (setup / 'prepared_text.npz').write_bytes(b'bytes that match no recorded digest')
    monkeypatch.setattr(validation_inference, 'resolve_best_checkpoint',
                        lambda _: (tmp_path / 'checkpoint', {}))
    monkeypatch.setattr(text_export, 'PreparedEmbeddingForward', _FakeForward)

    path = text_export.forward(output, setup, device='cpu')

    assert path == output / 'text__vectors.npz'
    contract = _FakeForward.instances[-1]
    assert contract.kwargs['tokens_size'] == request['tokens_size']  # passed through, never recomputed
    assert contract.kwargs['plan'] == request['plan']
    assert contract.kwargs['tokens_path'] == setup / 'prepared_text.npz'


def test_extract_prepared_inputs_rebuilds_a_changed_member(tmp_path):
    from core.bundle import Bundle, BundleRole
    from model_tracks.bundle_steps import extract_prepared_inputs

    member = tmp_path / 'catalog.csv'
    member.write_text('original')
    sealed = Bundle.seal_archive(tmp_path / 'inputs.zip',
                                 {'data/mt/setup/catalog.csv': member}, role=BundleRole.inputs)
    handle = Bundle.load(sealed.path, BundleRole.inputs)

    destination = tmp_path / 'tree'
    target = destination / 'data/mt/setup/catalog.csv'
    target.parent.mkdir(parents=True)
    target.write_text('tampered-and-longer')

    extract_prepared_inputs(handle, destination, 'data/mt/setup/suite_package_config.yaml')

    # The incompatible cached intermediate is rebuilt from the verified bundle
    # bytes instead of quarantining the finalize tree.
    assert target.read_text() == 'original'


def test_restore_frozen_baseline_rebuilds_a_mismatching_cache(tmp_path, monkeypatch):
    from training import prepare_embeddings
    from model_tracks import bundle_steps

    monkeypatch.setattr(prepare_embeddings, 'validate_result', lambda *args, **kwargs: None)
    layout = bundle_steps._setup_layout()
    destination = tmp_path / 'run'
    setup = destination / 'setup'; setup.mkdir(parents=True)
    baseline_dir = destination / bundle_steps.BASELINE_DIR; baseline_dir.mkdir()
    shared = bundle_steps._shared_embeddings_name()
    (baseline_dir / shared).write_bytes(b'frozen baseline')
    (setup / shared).write_bytes(b'incompatible cache')
    request = setup / layout.embedding_request
    request.parent.mkdir(parents=True, exist_ok=True)
    request.write_text('{}')

    bundle_steps._restore_frozen_baseline(destination, setup)

    assert (setup / shared).read_bytes() == b'frozen baseline'


def test_saved_ablation_report_identity_disagreements_are_named():
    from model_tracks.post_training_ablation import SavedAblationReport, SavedCalibration

    report = SavedAblationReport.model_validate({
        'track': 'text', 'result_size': 10, 'threshold': 0.5,
        'threshold_provenance': {'size': 20},
        'threshold_binding': {'track': 'text', 'checkpoint_size': 30, 'verified': True},
        'rows': []})
    calibration = SavedCalibration.model_validate(
        {'track': 'text', 'checkpoint_size': 30, 'threshold': 0.5})
    agree = dict(track='text', request_track='text', report_size=40,
                 vectors_size=10, report_sidecar_size=40,
                 binding_size=20, calibration=calibration)

    assert report.identity_disagreements(**agree) == []
    assert report.identity_disagreements(**{**agree, 'vectors_size': 99}) == ['result vectors']
    assert report.identity_disagreements(
        **{**agree, 'calibration': calibration.model_copy(update={'threshold': 0.9})}) == ['threshold']
