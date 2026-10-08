import numpy as np
import pandas as pd
import pytest

from core.sku_identity import ProductIdentity
from graph_tracks.report_attributes import identity_attributes, load_inputs, write_inputs, write_reports
from training.attribute_separation import ATTRIBUTE_SOURCES


def test_every_track_uses_existing_attribute_registry_and_support_rules(tmp_path):
    identities = [ProductIdentity(brand=frozenset({'brand'}), volume_ml=frozenset({500.}), pulp=frozenset({'pulp'})),
                  ProductIdentity(brand=frozenset({'brand'}), volume_ml=frozenset({500.}), pulp=frozenset({'pulp'})),
                  ProductIdentity(brand=frozenset({'other'}), volume_ml=frozenset({1000.}))]
    rows = [{'sku_id': str(i), 'attribute': identity_attributes(value)} for i, value in enumerate(identities)]
    write_inputs(tmp_path, rows)
    records = [{'sku_id': str(i)} for i in range(3)]
    pairs = {'dev': (np.array([[0, 1], [0, 2]]), np.array([1., 0.]))}
    for track in ('text', 'gnn_only', 'cascade'):
        output = tmp_path / track
        output.mkdir()
        write_reports(tmp_path / 'listings.json', records, pairs, ['dev'], output, track)
        summary = pd.read_csv(next(output.glob('*attribute_separation_summary.csv')))
        assert set(summary.attribute) == set(ATTRIBUTE_SOURCES)
        pulp = summary.set_index('attribute').loc['pulp']
        assert pulp.n_positive == 1 and pulp.n_negative == 1
        assert not pulp.reportable and not pulp.flagged_weak
        assert summary.set_index('attribute').loc['flavor', 'n_unobservable'] == 2


def test_report_population_mismatch_is_rejected(tmp_path):
    write_inputs(tmp_path, [{'sku_id': 'a', 'attribute': identity_attributes(ProductIdentity())}])
    with pytest.raises(ValueError, match='population mismatch'):
        load_inputs(tmp_path / 'listings.json', [{'sku_id': 'different'}])
