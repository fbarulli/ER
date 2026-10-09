"""Structural identity: a byte length may never stand in for a declared value.

Owner directive 2026-10-08: identity in this repository is STRUCTURAL -- declared
parameter values and stat censuses -- never a content digest, and never a byte
length. A byte length aliases every distinct input of the same size, which made
two different ablation requests share one staging folder, two different graph
removals share one inference job / tensor set, every ablated variant reuse the
baseline's retrieval ranks, and a rewritten config pass the resume check.

Each test below pins one of those aliases at the seam that fixed it. The
regression class is the same everywhere: a length was left where the declared
values (or their census) belong.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest


def _repository() -> Path:
    return Path(__file__).resolve().parents[1]


def _script(name: str):
    """Load one standalone scripts/ module by path (they are not importable)."""
    path = _repository() / 'scripts' / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── the declared-value identity itself ─────────────────────────────────────

def test_digest_is_the_declared_value_not_its_byte_length():
    from model_tracks import ablation

    left = {'text': [1, 2, 3], 'graph': [{'sku_id': 'ab'}]}
    right = {'text': [4, 5, 6], 'graph': [{'sku_id': 'cd'}]}
    assert len(json.dumps(left, sort_keys=True)) == len(json.dumps(right, sort_keys=True))
    assert ablation.digest(left) == json.dumps(left, sort_keys=True, ensure_ascii=False)
    assert ablation.digest(left) != ablation.digest(right)


def test_cohort_identity_separates_equal_length_cohorts():
    from model_tracks import ablation

    left = [{'sku_id1': 'a1', 'sku_id2': 'b1', 'label': '1', 'split': 'dev'}]
    right = [{'sku_id1': 'a2', 'sku_id2': 'b2', 'label': '1', 'split': 'dev'}]
    assert ablation.cohort_identity(left) != ablation.cohort_identity(right)
    # order is part of the cohort: the pair index order is what the lane scores
    reordered = [dict(left[0], sku_id1='a1', sku_id2='b1', label='0')]
    assert ablation.cohort_identity(left) != ablation.cohort_identity(reordered)


def test_request_folder_is_the_declared_name_not_a_request_digest():
    from model_tracks import ablation
    from model_tracks.resume import TRAINING_TRACKS

    request = {'track': 'gnn_only', 'checkpoint_role': 'selected'}
    assert ablation.request_folder(request) == 'gnn_only_selected'
    # One folder per (track, role): it must never collide with the fixed
    # per-track template directory the same output dir holds.
    assert ablation.request_folder(request) not in set(TRAINING_TRACKS)
    # and its name does not depend on the (up to 1GB) request payload
    assert ablation.request_folder({**request, 'texts': ['x' * 4096]}) == 'gnn_only_selected'


# ── the ablation lane's derived inputs ─────────────────────────────────────

def test_occurring_graph_fields_name_the_exact_removal():
    from model_tracks import ablation

    records = [{'attribute': {'flavor': 'cola'}, 'numeric': {'pack': 6}}]
    assert ablation.occurring_graph_fields(['attribute.flavor'], records) == ('attribute.flavor',)
    # a declared field no record carries removes nothing: it IS the baseline
    assert ablation.occurring_graph_fields(['attribute.pulp'], records) == ()
    with pytest.raises(ValueError, match='unsupported graph field'):
        ablation.graph_field_target('attribute.unknown_relation')


def test_variant_jobs_do_not_alias_equal_length_graph_removals():
    from model_tracks import ablation, ablation_inputs

    baseline_records = [{'sku_id': 'a', 'split': 'dev',
                         'attribute': {'pulp': 'abc'}, 'numeric': {'pack': 12345}}]

    def removed(field):
        return [ablation.graph_removed(record, [field]) for record in baseline_records]

    variants = [
        {'attribute': None, 'channel': 'baseline', 'changed_listings': 0,
         'text_indices': [], 'records': baseline_records},
        {'attribute': 'juice content', 'channel': 'graph', 'changed_listings': 1, 'text_indices': [],
         'records': removed('attribute.pulp')},
        {'attribute': 'count per unit', 'channel': 'graph', 'changed_listings': 1, 'text_indices': [],
         'records': removed('numeric.pack')},
    ]
    # The two derived record sets render to the SAME number of bytes: that is
    # exactly what the byte-length key aliased into one job (and one tensor set).
    assert len(ablation.digest(variants[1]['records'])) == len(ablation.digest(variants[2]['records']))
    request = {'settings': {'graph_fields': {'juice content': ['attribute.pulp'],
                                             'count per unit': ['numeric.pack']}},
               'variants': variants}
    plan = {'jobs': [], 'variant_jobs': [], 'graph_batches': {}}
    arrays: dict = {}
    ablation_inputs._variant_jobs(request, arrays, plan, payload=None, vocabulary=None, batch_size=1)
    assert plan['variant_jobs'] == [0, 1, 2]
    assert plan['jobs'][0]['graph_key'] == 'baseline'
    assert plan['jobs'][1]['graph_key'] != plan['jobs'][2]['graph_key']
    # the unchanged text is one declared input, stored once, not per variant
    assert {job['text_indices_key'] for job in plan['jobs']} == {'indices/baseline'}


def test_each_variant_keeps_its_own_retrieval_ranks():
    """Every variant block has the same byte size; ranks must still be its own."""
    from model_tracks import ablation

    ids = ['a', 'b']
    variants = [
        {'attribute': None, 'channel': 'baseline', 'changed_listings': 0,
         'text_indices': [], 'records': []},
        {'attribute': 'volume', 'channel': 'text', 'changed_listings': 1,
         'text_indices': [], 'records': []},
        {'attribute': 'flavour', 'channel': 'text', 'changed_listings': 1,
         'text_indices': [], 'records': []},
    ]
    request = {'ids': ids, 'pairs': [{'sku_id1': 'a', 'sku_id2': 'b', 'label': '1'}],
               'variants': variants, 'settings': ablation.Settings().model_dump(),
               'prepared_inputs': {'variant_jobs': [0, 1, 2]}}
    vectors = np.asarray([[[1., 0.], [0., 1.]], [[0., 1.], [0., 1.]], [[0., .5], [0., 1.]]],
                         dtype=np.float32)
    scores = np.asarray([[0.9], [0.1], [0.5]], dtype=np.float32)
    seen: list[tuple] = [tuple(np.round(vectors[0], 3).ravel().tolist())]

    class Retrieval:
        def ranks(self, vec):
            key = tuple(np.round(np.asarray(vec), 3).ravel().tolist())
            if key not in seen:
                seen.append(key)
            return [[len(seen), 9]]

        def ann_hits(self, vec):
            return [{'1': [True, False]}]

    baseline_ranks, baseline_hits = [[1, 9]], [{'1': [True, False]}]
    # every variant block has the SAME byte size: that is what the retired
    # length-keyed memo collapsed onto one cached answer.
    from core.portable_archive import ByteCount
    assert len({ByteCount(vectors[index].tobytes()).total for index in range(len(variants))}) == 1
    rows = ablation._comparison_rows(
        request, vectors, scores, 0.5, ablation.Settings(), Retrieval(),
        {key: index for index, key in enumerate(ids)}, baseline_ranks, baseline_hits,
        {0: (baseline_ranks, baseline_hits)})
    # variant 1 is a new job, so it measures ITS vector; the length-keyed memo
    # used to answer every variant with the baseline's ranks.
    assert [row['ablated_ranks'] for row in rows] == [[2, 9], [3, 9]]


def test_variant_jobs_fail_loud_on_an_incomplete_plan():
    from model_tracks import ablation, ablation_inputs

    assert ablation.variant_jobs({'variants': [{}, {}]}) == [0, 1]
    with pytest.raises(ValueError, match='does not cover every variant'):
        ablation.variant_jobs({'variants': [{}, {}], 'prepared_inputs': {'variant_jobs': [0]}})
    with pytest.raises(ValueError, match='at least the baseline variant'):
        ablation_inputs._variant_jobs({'settings': {}, 'variants': []}, {},
                                      {'jobs': [], 'variant_jobs': [], 'graph_batches': {}},
                                      payload=None, vocabulary=None, batch_size=1)


# ── the text cache's identity ──────────────────────────────────────────────

def test_composition_fingerprint_is_a_per_file_census(tmp_path, monkeypatch):
    from core import common
    from core.common import training_cfg
    from graph_tracks.text_cache import composition_fingerprint

    monkeypatch.setattr(common, 'TRAIN_ROOT', tmp_path)
    for directory in ('src/core', 'src/graph_tracks', 'config'):
        (tmp_path / directory).mkdir(parents=True)
    pinned = [f'config/{name}' for name in training_cfg().packaging.snapshot_pinned_configs]
    tracked = ('src/core/model_input.py', 'src/graph_tracks/text_cache.py', 'src/pipeline.py', *pinned)
    for path in tracked:
        (tmp_path / path).write_text('x')
    swapped, other = 'src/pipeline.py', 'src/graph_tracks/text_cache.py'
    (tmp_path / swapped).write_text('x')
    (tmp_path / other).write_text('y' * 9)
    census = json.loads(composition_fingerprint())
    assert [entry[0] for entry in census] == sorted(tracked)
    assert all(len(entry) == 3 for entry in census)

    def retired_byte_length():
        """The alias this replaced: name bytes + size DIGITS, summed (never again)."""
        return sum(len(path) + len(str((tmp_path / path).stat().st_size)) for path in tracked)

    aliased = retired_byte_length()
    # Redistribute 8 bytes between two tracked files: the summed length cannot
    # see it (both sizes stay one digit), the per-file census can.
    (tmp_path / swapped).write_text('x' * 9)
    (tmp_path / other).write_text('y')
    assert retired_byte_length() == aliased
    assert composition_fingerprint() != json.dumps(census, ensure_ascii=False)


def test_composition_fingerprint_tracks_a_same_size_rewrite(tmp_path, monkeypatch):
    from core import common
    from core.common import training_cfg
    from graph_tracks.text_cache import composition_fingerprint

    monkeypatch.setattr(common, 'TRAIN_ROOT', tmp_path)
    for directory in ('src/core', 'src/graph_tracks', 'config'):
        (tmp_path / directory).mkdir(parents=True)
    pinned = [f'config/{name}' for name in training_cfg().packaging.snapshot_pinned_configs]
    for path in ('src/core/model_input.py', 'src/graph_tracks/text_cache.py',
                 'src/pipeline.py', *pinned):
        (tmp_path / path).write_text('initial')
    before = json.loads(composition_fingerprint())
    rewritten = tmp_path / 'src/pipeline.py'
    rewritten.write_text('changed')  # same byte count
    # The modification time IS part of the identity: a same-size rewrite moves
    # the census where a byte length would not. (A filesystem with a coarse
    # clock can hide writes inside one tick; that is the accepted stat blind spot.)
    info = rewritten.stat()
    os.utime(rewritten, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))
    after = json.loads(composition_fingerprint())
    assert [entry[1] for entry in before] == [entry[1] for entry in after]
    assert after != before


def test_checkpoint_size_reports_an_int_size(tmp_path):
    from graph_tracks.text_cache import checkpoint_size

    folder = tmp_path / 'checkpoint'
    folder.mkdir()
    (folder / 'model.pt').write_bytes(b'weights')
    measured = checkpoint_size(folder)
    # the summed identity is the relative path's bytes plus the member size's
    # DECIMAL DIGITS (never the member bytes themselves)
    assert isinstance(measured, int)
    assert measured == len(b'model.pt') + len(str(len(b'weights')))


# ── the sampling ranks ─────────────────────────────────────────────────────

def test_declared_rank_keys_are_the_declared_values():
    from training import generate_rand_truth, sample_balanced_pairs

    assert generate_rand_truth._stable_rank(7, 'ab', 'c') == ('7', 'ab', 'c')
    assert sample_balanced_pairs._stable_rank(7, 1, 'ab') == ('7', '1', 'ab')
    # equal joined length, different declared inputs: never one key
    assert (generate_rand_truth._stable_rank(1, 'abcd', 'e')
            != generate_rand_truth._stable_rank(1, 'ab', 'cde'))
    assert (sample_balanced_pairs._stable_rank(1, 'abcd', 'e')
            != sample_balanced_pairs._stable_rank(1, 'ab', 'cde'))


def test_stratum_sweep_ranks_equal_length_identities_apart():
    from training import generate_rand_stratum_sweep as sweep

    assert sweep._rank(0, 'identity', '1111') != sweep._rank(0, 'identity', '2222')


def test_hidden_gtin_is_valid_absent_and_declared():
    from core.gtin import is_valid_gtin_checksum
    from training import generate_rand_stratum_sweep as sweep

    item_id = '4006381333931'
    canonical = {item_id, '5012345678900'}
    override = sweep._absent_valid_gtin(item_id, canonical, seed=0)
    assert is_valid_gtin_checksum(override) and override not in canonical
    assert override == sweep._absent_valid_gtin(item_id, canonical, seed=0)


# ── the evidence scripts' recorded censuses ────────────────────────────────

@pytest.mark.parametrize('name', ['finalize_full_evidence_rebuild.py',
                                  'replay_identity_residuals.py'])
def test_evidence_scripts_record_a_structural_census(tmp_path, name):
    module = _script(name)
    module.ROOT = tmp_path
    sample = tmp_path / 'input.csv'
    sample.write_bytes(b'inputs')
    assert module.census([sample]) == [
        {'name': 'input.csv', 'size': len(b'inputs'),
         'mtime_ns': sample.stat().st_mtime_ns}]
