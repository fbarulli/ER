"""The comparison surface reads DECLARED prepared-setup names.

``model_comparison`` joins scored pairs back to the prepared catalog. That
catalog filename is a declaration owned by ``training.preparation.graph_setup``
(``common.prepared_setup_layout().catalog``), not a dashboard literal: a
re-pointed layout must move this reader, or the dashboard silently reports on a
file no producer writes.
"""
import csv
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def comparison(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'dashboard'))
    return importlib.import_module('model_comparison')


def _write_catalog(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_pair_truth_reads_the_declared_catalog(comparison, tmp_path, monkeypatch):
    from core import common

    layout = common.prepared_setup_layout()
    monkeypatch.setattr(layout, 'catalog', 'renamed_catalog.csv')
    packaged = tmp_path / 'packaged'
    _write_catalog(packaged / 'renamed_catalog.csv', [
        {'sku_id': 'a', 'gtin': '1', 'brand': 'Alpha', 'sku_name_eng': 'first'},
        {'sku_id': 'b', 'gtin': '1', 'brand': 'Alpha', 'sku_name_eng': 'second'},
    ])
    # No gate results and no local run tree: the join is decided by the catalog.
    monkeypatch.setattr(comparison, 'packaged_setup', lambda run_tag: packaged)
    monkeypatch.setattr(comparison, 'run_dir', lambda run_tag: tmp_path / 'empty')
    models = {'text': {'scored': [{'sku_id1': 'a', 'sku_id2': 'b',
                                   'true_label': '1', 'score': '0.5'}]}}

    keys, truth, catalog = comparison._pair_truth('run-1', models)

    assert keys == [('a', 'b')]
    assert set(catalog) == {'a', 'b'}
    assert truth[('a', 'b')]['label'] == 1
    assert truth[('a', 'b')]['same_gtin'] is True
    assert truth[('a', 'b')]['title_b'] == 'second'
