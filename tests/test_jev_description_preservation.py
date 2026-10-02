"""JEV must receive the complete supplied description, including its tail."""
import csv
import importlib
import json
from pathlib import Path


def test_legacy_loader_preserves_identity_evidence_after_300_characters(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'jev'))
    state = importlib.import_module('state')
    description = 'Product information. ' * 40 + 'Contains pulp; six bottles per pack.'
    path = tmp_path / 'canonical_records.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['gtin', 'source_rows'])
        writer.writeheader()
        writer.writerow({'gtin': '123', 'source_rows': json.dumps([
            {'title': 'Juice', 'description': description}
        ])})
    listing = state.load_record_index(path)['123']
    assert state.listing_state('123', listing)['description'] == description
