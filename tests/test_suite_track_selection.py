"""Track selection is suite-declared config: one validator, one verdict seam."""
from pathlib import Path

import pytest

from model_tracks.config import SuiteConfig, load_config
from model_tracks.package import derivate_verdict
from model_tracks.post_training_ablation import selected_ablation_tracks


def _base(**updates) -> SuiteConfig:
    suite = load_config(Path('config/model_tracks.yaml'))
    return SuiteConfig.model_validate({**suite.model_dump(), **updates})


def test_default_suite_runs_the_full_cohort():
    assert SuiteConfig(setup_dir='s', text_bundle='b').tracks == ('text', 'gnn_only', 'cascade')


def test_cascade_alone_is_refused_naming_the_trained_lanes():
    with pytest.raises(ValueError, match='trained lanes'):
        _base(tracks=('cascade',))


def test_unknown_duplicate_and_empty_selections_are_refused():
    for tracks in (('gnn',), ('text', 'text'), ()):
        with pytest.raises(ValueError):
            _base(tracks=tracks)


def test_subset_selection_is_kept_as_declared():
    assert _base(tracks=('text',)).tracks == ('text',)
    assert selected_ablation_tracks(_base(tracks=('gnn_only',))) == ('gnn_only',)


#: An identity dict as the package bakes it: only text_loss/tracks are mutable.
_BAKED = {'setup_dir': 'data/model_tracks/shared', 'text_bundle': 'b', 'text_model': 'minilm_l6',
          'text_loss': None, 'tracks': ['text', 'gnn_only', 'cascade'], 'epochs': 10}


def test_derivate_verdict_reuse_derivate_and_foreign():
    assert derivate_verdict(_BAKED, dict(_BAKED)) == 'reuse'
    assert derivate_verdict(_BAKED, {**_BAKED, 'tracks': ['text']}) == 'derivate'
    assert derivate_verdict(_BAKED, {**_BAKED, 'text_loss': 'triplet'}) == 'derivate'
    assert derivate_verdict(_BAKED, {**_BAKED, 'epochs': 3}) == 'foreign'
    assert derivate_verdict(_BAKED, {**_BAKED, 'setup_dir': 'other'}) == 'foreign'
