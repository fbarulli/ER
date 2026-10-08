"""Smoke tests for the ER dashboard composition root (dashboard/app.py).

Regression cover for the census panel: its budget row builder referenced an
undefined ``kv``, and the surrounding ``except Exception`` swallowed the
NameError, so Finding 05 rendered '' instead of the measured datagen headroom
shortlist. These tests pin the helper's rows AND that the panel renders them.
"""
import importlib.util
from pathlib import Path

import pytest

DASHBOARD = Path(__file__).parents[1] / 'dashboard'


@pytest.fixture(scope='module')
def app_module():
    spec = importlib.util.spec_from_file_location('er_dashboard_app', DASHBOARD / 'app.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CENSUS = {
    'baseline': {'caffeine': {}, 'diets': {}},
    'census': {'keys': {}},
    'datagen_budget': {
        # Below the headroom floor: no minting headroom left, so not a row.
        'spent': {'rows_populated': 10, 'conflict_rate': 0.5, 'headroom_share': 0.02,
                  'kind': 'SET_ENUM', 'veto_candidate': False},
        'wide': {'rows_populated': 3652, 'conflict_rate': 0.4905, 'headroom_share': 0.4905,
                 'kind': 'SET_CATEGORICAL', 'veto_candidate': True},
        'thin': {'rows_populated': 1864, 'conflict_rate': 0.1847, 'headroom_share': 0.1847,
                 'kind': 'SET_CATEGORICAL'},
    },
}


def test_census_budget_rows_rank_by_measured_conflict(app_module):
    rows = app_module.census_budget_rows(CENSUS)
    assert [row[0] for row in rows] == ['wide', 'thin']
    assert rows[0] == ('wide', 3652, 0.4905, 'SET_CATEGORICAL', True)
    assert rows[1][4] is False


def test_census_budget_rows_tolerate_absent_budget(app_module):
    assert app_module.census_budget_rows({}) == []
    assert app_module.census_budget_rows(None) == []
    assert app_module.census_budget_rows({'datagen_budget': {'x': None}}) == []
    assert app_module.census_pending_table({'datagen_budget': {}}) == ''


def test_census_pending_table_renders_escaped_measured_cells(app_module):
    html = app_module.census_pending_table(CENSUS)
    assert all(cell in html for cell in ('wide', '3,652', '49.0%', 'SET_CATEGORICAL', 'veto-grade'))
    assert 'spent' not in html


def test_census_finding_renders_the_measured_shortlist(app_module):
    census, source = app_module._census_payload()
    assert source is not None and census is not None, 'census evidence must be on disk'
    rows = app_module.census_budget_rows(census)
    assert rows, 'populated census must yield ranked rows'
    finding = app_module.census_finding()
    assert 'capture still pending' in finding
    assert 'dimensions with unspent conflict headroom' in finding
    assert rows[0][0] in finding, 'the panel must render the measured rows'
    assert 'not parseable' not in finding
    assert rows[0][0] in app_module.datagen_track()


def test_unreadable_census_is_surfaced_not_swallowed(app_module, tmp_path, monkeypatch, caplog):
    broken = tmp_path / 'attribute_universe_census.json'
    broken.write_text('{not json')
    monkeypatch.setattr(app_module, 'census_sources', lambda: (broken,))
    with caplog.at_level('WARNING'):
        finding = app_module.census_finding()
    assert 'unreadable' in finding
    assert 'JSONDecodeError' in finding, 'the real error must reach the panel'
    assert caplog.records, 'the error must be logged, never silently swallowed'
