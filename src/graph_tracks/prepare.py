"""Export graph inputs from shared identity extraction and explicit listing splits.

No splits are invented and no gtin/label edges enter the model. The
listing schema is DERIVED from the shared extractor contract
(core.sku_identity.graph_schema) and its manifest records what was
derived at prepare time; a loader that sees a different schema refuses the
inputs as stale.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from graph_tracks.data import RELATIONS, NUMERIC, file_hash, load_records


def prepare(catalog: Path, splits: Path, pairs: Path, output: Path) -> Path:
    from core.sku_identity import row_identity
    from graph_tracks.train import load_pairs, write_json
    frame = pd.read_csv(catalog, dtype=str, keep_default_na=False, low_memory=False)
    from core.identity_policy import reviewed_row_mask, POLICY_PATH
    from core.common import TRAIN_ROOT
    if reviewed_row_mask(frame).any():
        raise ValueError("catalog contains quarantined identity groups/listings; apply reviewed exclusions before preparing splits")
    assignment = pd.read_csv(splits, dtype=str, keep_default_na=False)
    if 'sku_id' not in frame or set(assignment.columns) != {'sku_id', 'split'}:
        raise ValueError('catalog needs sku_id; split CSV needs exactly sku_id,split')
    if frame.sku_id.duplicated().any() or assignment.sku_id.duplicated().any():
        raise ValueError('listing IDs and split assignments must be unique')
    if set(frame.sku_id) != set(assignment.sku_id):
        raise ValueError('split map must cover exactly the retained catalog')
    lookup = assignment.set_index('sku_id').split.to_dict()
    records = []
    from graph_tracks.report_attributes import FILENAME, identity_attributes, write_inputs
    report_rows = []
    for _, row in frame.iterrows():
        identity = row_identity(row)
        report_rows.append({'sku_id': row.sku_id, 'attribute': identity_attributes(identity)})
        records.append({'sku_id': row.sku_id, 'split': lookup[row.sku_id],
                        'attribute': {key: sorted(getattr(identity, key)) for key in RELATIONS},
                        'numeric': {key: sorted(getattr(identity, key)) for key in NUMERIC}})
    output.mkdir(parents=True, exist_ok=False)
    listing_path = output / 'listings.json'
    write_json(listing_path, {'schema': 'er-graph-listings-v1', 'listings': records})
    write_inputs(output, report_rows)
    records = load_records(listing_path)
    load_pairs(pairs, records)
    (output / 'pairs.csv').write_bytes(pairs.read_bytes())
    write_json(output / 'input_manifest.json', {
        'schema': 'er-graph-inputs-v1', 'catalog_sha256': file_hash(catalog),
        'identity_policy_sha256': file_hash(POLICY_PATH),
        'identity_dimensions_sha256': file_hash(TRAIN_ROOT / 'config' / 'identity_dimensions.yaml'),
        'splits_sha256': file_hash(splits), 'pairs_sha256': file_hash(pairs),
        'listings_sha256': file_hash(listing_path), 'identity_extractor': 'core.sku_identity.row_identity',
        'report_attributes_sha256': file_hash(output / FILENAME),
        'relations': list(RELATIONS), 'numeric': list(NUMERIC),
        'feature_scope': 'derived from core.sku_identity.graph_schema; every extractor descriptor is a model input',
        'excluded_model_inputs': ['gtin', 'verified identity edges', 'raw text'],
    })
    return listing_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('catalog', 'splits', 'pairs', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.catalog, args.splits, args.pairs, args.output)

if __name__ == '__main__':
    main()
