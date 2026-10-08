from core import common


def test_section_copy_is_isolated_and_observes_source_changes(monkeypatch):
    source = {'training':{'model_input':{'profile':'cleaned'}}}
    monkeypatch.setattr(common,'_load_config_cached',lambda:source)
    section = common.config_section('training','model_input')
    section['profile'] = 'changed by caller'
    assert source['training']['model_input']['profile'] == 'cleaned'
    source['training']['model_input']['profile'] = 'new config'
    assert common.config_section('training','model_input')['profile'] == 'new config'


def test_section_accessor_respects_injected_config_loader():
    override = lambda:{'training':{'model_input':{'profile':'override'}}}
    assert common.config_section('training','model_input',loader=override) == {'profile':'override'}


# ── ONE YAML load-validate home (consolidation audit 2026-10-08, finding 2) ──
# Four ``load_config`` implementations existed: the project-config SSOT
# (core.common.load_config), a suite-manifest loader (model_tracks.config), a
# graph-lane loader (graph_tracks.config) and a DEAD NER loader
# (ner/config_loader.py, zero importers — deleted). The two lane loaders do not
# load the project config; what they duplicated was the read + validate +
# name-the-file dance, and ner/config_loader's ``${base_dir}`` expansion
# (ner/ner.py:_resolve_config_value keeps the ONE pinned copy: the bare Colab
# runtime imports ner.py with core.common deliberately absent, so a shared helper
# would pull core into that runtime). The shared read+validate now lives once,
# in core.common.

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from core import common


class _Doc(BaseModel):
    model_config = ConfigDict(extra='forbid')
    value: int = 0


def test_load_validated_yaml_returns_the_model_and_names_the_file(tmp_path):
    good = tmp_path / 'good.yaml'
    good.write_text('value: 7\n')
    assert common.load_validated_yaml(good, _Doc, label='Test document').value == 7

    bad = tmp_path / 'bad.yaml'
    bad.write_text('value: [not, an, int]\nunexpected: true\n')
    with pytest.raises(ValidationError) as excinfo:
        common.load_validated_yaml(bad, _Doc, label='Test document')
    notes = ' | '.join(getattr(excinfo.value, '__notes__', []))
    assert 'Test document' in notes and str(bad) in notes


def test_the_lane_loaders_delegate_to_the_one_home(monkeypatch, tmp_path):
    from graph_tracks import config as graph_config
    from model_tracks import config as suite_config
    recorded: list = []
    marker = SimpleNamespace(track='text')

    def spy(path, model, *, label):
        recorded.append((Path(path), model, label))
        return marker

    monkeypatch.setattr(common, 'load_validated_yaml', spy)
    lane_yaml = tmp_path / 'lane.yaml'
    lane_yaml.write_text('track: text\noutput_dir: reports\n')

    assert suite_config.load_config(lane_yaml) is marker
    assert graph_config.load_config(lane_yaml, expected_track='text') is marker
    assert graph_config.load_text_config(lane_yaml) is marker
    assert [label for _, _, label in recorded] == [
        'Model-track suite configuration',
        'Graph lane configuration',
        'Text lane configuration',
    ]
    assert [model for _, model, _ in recorded] == [
        suite_config.SuiteConfig, graph_config.GraphConfig, graph_config.TextConfig,
    ]


def test_graph_lane_loaders_keep_their_sharp_messages(tmp_path):
    from graph_tracks.config import load_config as load_graph_config
    from graph_tracks.config import load_text_config

    missing = tmp_path / 'absent.yaml'
    with pytest.raises(FileNotFoundError, match='text lane config'):
        load_text_config(missing)

    unparseable = tmp_path / 'broken.yaml'
    unparseable.write_text('track: ["unclosed"\n')
    with pytest.raises(Exception) as excinfo:
        load_graph_config(unparseable)
    assert 'Graph lane configuration' in ' | '.join(getattr(excinfo.value, '__notes__', []))


def test_the_dead_ner_config_loader_is_gone():
    """ner/config_loader.py had zero importers and duplicated the SSOT read."""
    assert importlib.util.find_spec('ner.config_loader') is None


# ── orchestration stage inventory lives in config/paths.yaml (audit 2026-10-08) ──
# The trace-stage join is the SSOT-sanctioned block: config/paths.yaml declares
# it, and ``core.tracing`` reads it back through its accessors (no code-side copy).

def test_orchestration_stages_read_from_the_config_ssot():
    from core import tracing

    merged = common.load_config()
    declared = {stage: tuple(names)
                for stage, names in merged['orchestration_stages'].items()}
    assert declared == tracing.orchestration_trace_stages()
    assert tuple(merged['orchestration_lane_stages']) == \
        tracing.orchestration_lane_stages()
    # the two row-free stages stay explicit (a coverage gap, not "no work")
    assert declared['gate_census'] == () and declared['discriminator'] == ()
    # the code-side duplicate registry is GONE: config is the ONE home
    assert not hasattr(tracing, 'ORCHESTRATION_TRACE_STAGES')
    assert not hasattr(tracing, 'ORCHESTRATION_LANE_STAGES')


def _paths_config(**overrides):
    from core.schemas import DataConfig

    raw = yaml.safe_load(common.CONFIG_PATH.read_text())
    raw.update(overrides)
    return DataConfig.model_validate(raw)


def test_orchestration_lane_stages_must_name_declared_stages():
    """A lane stage absent from ``orchestration_stages`` can never join its
    rows: the root contract fails loud at load instead of mid-run."""
    with pytest.raises(ValidationError, match='undeclared orchestration'):
        _paths_config(orchestration_lane_stages=['not_a_stage'])


def test_orchestration_lane_stages_reject_duplicates():
    declared = next(iter(_paths_config().orchestration_stages))
    with pytest.raises(ValidationError, match='contains duplicates'):
        _paths_config(orchestration_lane_stages=[declared, declared])


# ── config/training.yaml bundle: block declares every BundleSpec value ──────
# The audit found 11 BundleSpec keys were schema-default-only (no config value);
# a name a sealing/verifying surface reads must be config-owned, not a mirror.

def test_bundle_block_declares_every_bundle_spec_key():
    from core.schemas import BundleSpec

    raw = yaml.safe_load((common.TRAIN_ROOT / 'config/training.yaml').read_text())
    assert set(raw['bundle']) == set(BundleSpec.model_fields), (
        'config/training.yaml bundle: must declare every BundleSpec field; '
        f'missing {sorted(set(BundleSpec.model_fields) - set(raw["bundle"]))}')


def test_bundle_block_matches_the_loaded_spec():
    raw = yaml.safe_load((common.TRAIN_ROOT / 'config/training.yaml').read_text())
    spec = common.training_cfg().bundle
    for key, value in raw['bundle'].items():
        loaded = getattr(spec, key)
        expected = tuple(value) if isinstance(loaded, tuple) else value
        assert loaded == expected, key


# ── prepared setup layout: ONE accessor + config-owned child filenames ──────
# The prepared-setup layout (training.preparation.graph_setup) is read through
# common.prepared_setup_layout(), and its child filenames are config-owned, so
# no lane re-spells ``training_cfg().preparation.graph_setup`` or a literal.

def test_prepared_setup_layout_accessor_is_the_training_cfg_layout():
    layout = common.prepared_setup_layout()
    assert layout is common.training_cfg().preparation.graph_setup
    assert layout.input_manifest == 'input_manifest.json'
    assert layout.listings == 'listings.json'


def test_graph_setup_block_declares_every_layout_field():
    from core.schemas import PreparationGraphSetupSpec

    raw = yaml.safe_load((common.TRAIN_ROOT / 'config/training.yaml').read_text())
    block = raw['preparation']['graph_setup']
    assert set(block) == set(PreparationGraphSetupSpec.model_fields), (
        'config/training.yaml preparation.graph_setup must declare every '
        f'layout field; missing '
        f'{sorted(set(PreparationGraphSetupSpec.model_fields) - set(block))}')


# ── the archive sidecar suffix has ONE home (audit 2026-10-08) ──────────────

def test_the_kaggle_sidecar_suffix_is_a_projection_of_the_bundle_home():
    cfg = common.training_cfg()
    raw = yaml.safe_load((common.TRAIN_ROOT / 'config/training.yaml').read_text())
    # the duplicate kaggle.files.hash_suffix declaration is deleted ...
    assert 'hash_suffix' not in raw['kaggle']['files']
    # ... and the rendered LANE projection carries the ONE home's value.
    assert cfg.kaggle.files.hash_suffix == cfg.bundle.sha256_sidecar_suffix

