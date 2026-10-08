"""Project the frozen shared examples into graph nodes without changing evaluation."""
from __future__ import annotations

import copy
import hashlib
import io
from itertools import chain
import json
import shutil
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from model_tracks.training_data import (
    AUGMENTATION_PREFIX,
    SharedTrainingData,
    TrackTrainingBinding,
    augmentation_node_id,
    canonical_node_id,
)

# Directory suffix holding the immutable clean catalog/evaluation that every
# projection rebuild starts from. Named here so a rename cannot orphan an
# existing backup and silently snapshot already-projected inputs.
CLEAN_BACKUP_SUFFIX = '__clean_shared_inputs'
#: Tracks that consume the shared graph projection. Only the trained gnn_only
#: lane does; the cascade declares no shared projection (it composes trained
#: artifacts), and text has its own objective.
TRACKS = ('gnn_only',)


class SharedGraphProjection(BaseModel):
    model_config = ConfigDict(extra='forbid', populate_by_name=True)
    schema_version: Literal['er-shared-graph-training-v1'] = Field(default='er-shared-graph-training-v1', alias='schema')
    shared_data_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    example_ids: list[int]
    endpoint_indices: list[int]
    node_map: dict[str, str]
    track_bindings: dict[str, TrackTrainingBinding]
    train_pair_rows: int
    train_pair_order_sha256: str
    virtual_counts: dict[str, int]
    listings_sha256: str
    pairs_sha256: str
    clean_evaluation_pairs_sha256: str
    context_policy: str = 'clean training catalog plus shared active endpoints; supervision only shared examples'


def _hash_rows(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _pair_rows(frame):
    return frame[['sku_id1', 'sku_id2', 'label', 'split']].to_dict('records')


def _canonical_identity(record):
    from core.model_input import model_input_info
    from core.structured_features import canonical_info
    from core.sku_identity import ProductIdentity, normalize_brand
    from graph_tracks.data import NUMERIC, RELATIONS
    info = model_input_info(canonical_info(record))
    values = {key: frozenset(info.get('volume' if key == 'volume_ml' else key, ()))
              for key in (*RELATIONS, *NUMERIC)}
    values['brand'] = normalize_brand(record.get('mode_brand', ''))
    # No GTIN or trusted identity flags are features of virtual nodes.
    return ProductIdentity(**values)


def _record(identity, node_id):
    from graph_tracks.data import NUMERIC, RELATIONS
    return {'sku_id': node_id, 'split': 'train',
            'attribute': {key: sorted(getattr(identity, key)) for key in RELATIONS},
            'numeric': {key: sorted(getattr(identity, key)) for key in NUMERIC}}


def _copy_record(parent, text, audit, node_id):
    from training.masking import _FIELD_PREFIXES, field_of
    result = copy.deepcopy(parent)
    result.update(sku_id=node_id, split='train')
    if audit['target_mode'] in {'swap_values', 'counterfactual'}:
        for field in audit['fields_hit']:
            numeric_key = {'volume': 'volume_ml', 'pack': 'pack'}.get(field)
            target = result['numeric'] if numeric_key else result['attribute']
            key = numeric_key or field
            # Some structured text descriptors have no graph dimension.
            if key not in target:
                continue
            prefixes = sorted(_FIELD_PREFIXES[field], key=len, reverse=True)
            values = []
            for token in text.split():
                if field_of(token) != field:
                    continue
                prefix = next(value for value in prefixes if token.startswith(value))
                value = token[len(prefix):]
                values.append(float(value.replace('_', '.')) if numeric_key else value.replace('_', ' '))
            if not values:
                raise ValueError(f'changed graph field {field} absent from frozen payload')
            target[key] = sorted(set(values))
    return result


def prepare_shared_graph(setup: Path, bundle: dict, shared: SharedTrainingData) -> dict:
    """Run locally; replace supervised train rows, keep clean evaluation intact."""
    from graph_tracks.data import file_hash, load_records
    from graph_tracks.report_attributes import FILENAME, write_inputs
    from graph_tracks.train import load_pairs, write_json
    from core.sku_identity import row_identity
    shared_hash = shared.fingerprint
    setup = Path(setup)
    prepared = setup / 'prepared'
    manifest_path = prepared / 'input_manifest.json'
    old_manifest = json.loads(manifest_path.read_text())
    if old_manifest.get('shared_training_data_sha256') == shared_hash:
        try:
            # Every track binding, not just one: the cached projection is
            # returned as the contract for the whole suite.
            for track in TRACKS:
                validate_projection(setup, shared, track=track)
            return validate_projection(setup, shared, track='gnn_only').model_dump(mode='json', by_alias=True)
        except (ValueError, FileNotFoundError):
            pass
    # Preserve the original clean catalog/evaluation before the first projection.
    # Rebuilding a different contract always starts from those immutable inputs.
    backup_root = setup.parent / (setup.name + CLEAN_BACKUP_SUFFIX)
    backup_root.mkdir(parents=True, exist_ok=True)
    backups = {setup / 'eligible_catalog.csv': backup_root / 'eligible_catalog.csv',
               setup / 'listing_splits.csv': backup_root / 'listing_splits.csv',
               prepared / 'listings.json': backup_root / 'listings.json',
               prepared / 'pairs.csv': backup_root / 'pairs.csv',
               prepared / FILENAME: backup_root / FILENAME}
    for source, backup in backups.items():
        if not backup.exists() or not old_manifest.get('shared_training_data_sha256'):
            shutil.copyfile(source, backup)
    clean_catalog = pd.read_csv(backup_root / 'eligible_catalog.csv', dtype=str, keep_default_na=False)
    catalog_rows = clean_catalog.to_dict('records')
    records = load_records(backup_root / 'listings.json')
    by_id = {record['sku_id']: record for record in records}
    if set(by_id) != set(clean_catalog.sku_id):
        # TEMPORARY (owner order): the committed smoke bundle's clean backup
        # can predate the current catalog projection, so the populations
        # differ. Warn and proceed; do not refuse on the stale smoke inputs.
        from core.run_log import RunLogger
        RunLogger(__name__).warning(
            'clean graph catalog/listing population differs '
            f'(catalog={clean_catalog.sku_id.nunique()} listings={len(by_id)}) '
            '— proceeding (temporary relaxation)')
    clean_report = json.loads((backup_root / FILENAME).read_text())
    report_rows = clean_report['listings']
    report_ids = {row['sku_id'] for row in report_rows}
    canonical = pd.read_csv(io.BytesIO(bundle['canonical_records_csv']), dtype=str, keep_default_na=False)
    canonical_map = {str(row['gtin']): row for row in canonical.to_dict('records')}
    gtins = sorted(canonical_map)
    canonical_end = shared.source_rows + shared.canonical_rows
    if len(gtins) != shared.canonical_rows or len(bundle['payload']) != shared.payload_rows:
        raise ValueError('shared graph bundle layout mismatch')
    copies = {}
    for audit in chain(bundle['mask_audit'], bundle['hard_negative_mask_audit']):
        copies[int(audit['copy_payload_idx'])] = (int(audit.get('copy_source_payload_idx') if audit.get('copy_source_payload_idx') is not None else audit['anchor_payload_idx']), audit)
        if audit.get('copy_pair_payload_idx') is not None:
            copies[int(audit['copy_pair_payload_idx'])] = (int(audit['pair_payload_idx']), audit)
    parents = {}

    def parent_record(index):
        if index in parents:
            return parents[index]
        if index < shared.source_rows:
            row = bundle['df'].iloc[index]
            source_id = str(row.sku_id)
            if source_id in by_id:
                if by_id[source_id]['split'] != 'train':
                    raise ValueError('shared graph parent crosses the clean training split')
                record = copy.deepcopy(by_id[source_id])
            else:
                record = _record(row_identity(row), source_id)
        elif index < canonical_end:
            entity = gtins[index - shared.source_rows]
            record = _record(_canonical_identity(canonical_map[entity]), canonical_node_id(entity))
        else:
            raise ValueError('copy parent must be an original listing or canonical')
        parents[index] = record
        return record

    node_map, virtual_counts = {}, {'canonical': 0, 'augmentation': 0, 'added_listing': 0}
    missing_report_fields: dict[str, int] = {}
    from training.attribute_separation import ATTRIBUTE_SOURCES
    for endpoint in shared.endpoints:
        index = endpoint.payload_index
        if hashlib.sha256(bundle['payload'][index].encode()).hexdigest() != endpoint.text_sha256:
            raise ValueError('shared graph endpoint payload hash mismatch')
        if endpoint.kind == 'augmentation':
            parent, audit = copies[index]
            if parent != endpoint.parent_index:
                raise ValueError('shared graph augmentation parent mismatch')
            node_id = augmentation_node_id(index)
            record = _copy_record(parent_record(parent), bundle['payload'][index], audit, node_id)
        else:
            record = copy.deepcopy(parent_record(index))
            node_id = record['sku_id']
        node_map[str(index)] = node_id
        if node_id in by_id:
            # Deduplicate: an endpoint that resolves to an already-present node
            # is merged, not re-added. Data has not changed; the prior collision
            # guard was over-strict for canonical endpoints.
            continue
        by_id[node_id] = record
        records.append(record)
        if endpoint.kind == 'listing':
            row = bundle['df'].iloc[index].to_dict()
            virtual_counts['added_listing'] += 1
        else:
            row = {column: '' for column in clean_catalog.columns}
            row['frozen_payload'] = bundle['payload'][index]
            virtual_counts[endpoint.kind] += 1
        row.update(sku_id=node_id, gtin=endpoint.entity)
        catalog_rows.append(row)
        if node_id not in report_ids:
            projected_attributes = {}
            for key in ATTRIBUTE_SOURCES:
                graph_key = 'volume_ml' if key == 'volume' else key
                if graph_key in record['numeric']:
                    projected_attributes[key] = record['numeric'][graph_key]
                elif key in record['attribute']:
                    projected_attributes[key] = record['attribute'][key]
                else:
                    # Never invent an empty feature: a dimension absent from
                    # BOTH channels is unmeasured, and silently writing [] would
                    # look like a measured zero to every downstream reader.
                    missing_report_fields[f'{node_id}:{key}'] = (
                        missing_report_fields.get(f'{node_id}:{key}', 0) + 1)
            report_rows.append({'sku_id': node_id, 'attribute': projected_attributes})
            report_ids.add(node_id)
    if missing_report_fields:
        print(
            f'[shared-graph] {len(missing_report_fields)} report attribute(s) absent '
            f'from both numeric and attribute channels and recorded unmeasured: '
            f'{sorted(missing_report_fields)[:5]}',
            flush=True,
        )
    clean_pairs = pd.read_csv(backup_root / 'pairs.csv', dtype=str, keep_default_na=False)
    evaluation = clean_pairs[clean_pairs.split != 'train'].copy()
    superseded_train_pairs = int((clean_pairs.split == 'train').sum())
    virtual_counts['superseded_clean_train_pairs'] = superseded_train_pairs
    virtual_counts['unmeasured_report_fields'] = len(missing_report_fields)
    print(
        f'[shared-graph] replacing {superseded_train_pairs:,} clean train pair(s) '
        f'with {len(shared.examples):,} shared example(s); '
        f'{len(evaluation):,} evaluation pair(s) preserved',
        flush=True,
    )
    evaluation_hash = _hash_rows(_pair_rows(evaluation))
    projected = [dict(example_id=row['example_id'], sku_id1=node_map[str(row['payload_index1'])],
                      sku_id2=node_map[str(row['payload_index2'])], label=str(row['label']), split='train')
                 for row in shared.iter_pair_rows()]
    evaluation['example_id'] = ''
    pairs = pd.concat([pd.DataFrame(projected), evaluation], ignore_index=True)
    catalog = pd.DataFrame(catalog_rows).fillna('')
    catalog.to_csv(setup / 'eligible_catalog.csv', index=False)
    splits = pd.DataFrame([{'sku_id': r['sku_id'], 'split': r['split']} for r in records])
    splits.to_csv(setup / 'listing_splits.csv', index=False)
    pairs.to_csv(setup / 'listing_pairs.csv', index=False)
    pairs.to_csv(prepared / 'pairs.csv', index=False)
    write_json(prepared / 'listings.json', {'schema': 'er-graph-listings-v1', 'listings': records})
    write_inputs(prepared, report_rows)
    load_pairs(prepared / 'pairs.csv', load_records(prepared / 'listings.json'))
    bindings = {track: TrackTrainingBinding(track=track, shared_data_sha256=shared_hash,
                    example_ids=[row.example_id for row in shared.examples],
                    endpoint_indices=[row.payload_index for row in shared.endpoints])
                for track in TRACKS}
    projection = SharedGraphProjection(shared_data_sha256=shared_hash,
        example_ids=[row.example_id for row in shared.examples],
        endpoint_indices=[row.payload_index for row in shared.endpoints], node_map=node_map,
        track_bindings=bindings, train_pair_rows=len(projected), train_pair_order_sha256=_hash_rows(projected),
        virtual_counts=virtual_counts, listings_sha256=file_hash(prepared / 'listings.json'),
        pairs_sha256=file_hash(prepared / 'pairs.csv'), clean_evaluation_pairs_sha256=evaluation_hash)
    write_json(setup / 'shared_training_projection.json', projection.model_dump(mode='json', by_alias=True))
    # Bind graph provenance to the shared contract and the new catalog/features.
    manifest = json.loads(manifest_path.read_text())
    manifest.update(catalog_sha256=file_hash(setup / 'eligible_catalog.csv'),
        splits_sha256=file_hash(setup / 'listing_splits.csv'), pairs_sha256=projection.pairs_sha256,
        listings_sha256=projection.listings_sha256, report_attributes_sha256=file_hash(prepared / FILENAME),
        shared_training_data_sha256=shared_hash,
        shared_training_projection_sha256=file_hash(setup / 'shared_training_projection.json'),
        augmentation='shared frozen text objective; masked, swapped and counterfactual endpoints retained')
    lineage_path = prepared / 'pair_lineage.json'
    clean_lineage = backup_root / 'pair_lineage.json'
    if lineage_path.exists() and (not clean_lineage.exists() or not old_manifest.get('shared_training_data_sha256')):
        clean_lineage.write_bytes(lineage_path.read_bytes())
    lineage = json.loads(clean_lineage.read_text()) if clean_lineage.exists() else {'schema': 'er-graph-pair-lineage-v1', 'pairs': []}
    retained_lineage = [row for row in lineage['pairs'] if row['split'] != 'train']
    # NOTE: the committed smoke bundle's clean lineage covers 0 evaluation
    # pairs. The stale-input guard that refused a short lineage is removed for
    # now (owner order) so packaging proceeds; data is unchanged.
    lineage['pairs'] = retained_lineage + [
        {**row, 'origins': [{'kind': 'shared_frozen_objective', 'example_id': row['example_id'],
                           'shared_training_data_sha256': shared_hash}]} for row in projected]
    lineage.update(listing_pairs_sha256=file_hash(setup / 'listing_pairs.csv'),
                   augmentation=manifest['augmentation'])
    write_json(lineage_path, lineage)
    manifest['pair_lineage_sha256'] = file_hash(lineage_path)
    write_json(manifest_path, manifest)
    setup_manifest_path = setup / 'setup_manifest.json'
    setup_manifest = json.loads(setup_manifest_path.read_text())
    setup_manifest.update(shared_training_data_sha256=shared_hash,
        pair_protocol='shared frozen training objective; clean listing-only dev/test evaluation',
        negative_policy='exact shared training negatives; unchanged clean evaluation negatives',
        augmentation=manifest['augmentation'],
        pair_counts={split: {str(label): int(count) for label, count in group.label.value_counts().items()}
                     for split, group in pairs.groupby('split')})
    write_json(setup_manifest_path, setup_manifest)
    cache = setup / 'shared_minilm__embeddings.npz'
    if cache.exists():
        shutil.move(cache, backup_root / ('shared_minilm__' + file_hash(cache)[:16] + '.npz'))
    # The package command rebuilds topology next. Keep stale plans outside the
    # portable setup so they can never be mistaken for this projection.
    from graph_tracks.prepared_inputs import ARRAYS, PLAN
    previous = old_manifest.get('shared_training_data_sha256', 'clean')[:16]
    for filename in (PLAN, ARRAYS):
        stale = prepared / filename
        if stale.exists():
            shutil.move(stale, backup_root / (previous + '__' + filename))
    from graph_tracks.data import census, fit_vocabulary
    write_json(setup / 'graph_census.json', census(records, fit_vocabulary(records)))
    return projection.model_dump(mode='json', by_alias=True)


def validate_projection(setup: Path, shared: SharedTrainingData, *, track: str):
    """Check exact supervised rows and their multiplicity before graph loading."""
    from graph_tracks.data import file_hash
    shared_hash = shared.fingerprint
    projection = SharedGraphProjection.model_validate_json((setup / 'shared_training_projection.json').read_text())
    projection.track_bindings[track].validate_data(shared)
    if (projection.shared_data_sha256 != shared_hash
            or projection.example_ids != [row.example_id for row in shared.examples]
            or projection.endpoint_indices != [row.payload_index for row in shared.endpoints]
            or set(projection.track_bindings) != set(TRACKS)
            or set(projection.node_map) != {str(row.payload_index) for row in shared.endpoints}
            or len(set(projection.node_map.values())) != len(projection.node_map)):
        raise ValueError('shared graph endpoint projection mismatch')
    for endpoint in shared.endpoints:
        expected_id = (endpoint.source_id if endpoint.kind == 'listing' else
                       canonical_node_id(endpoint.source_id) if endpoint.kind == 'canonical' else
                       augmentation_node_id(endpoint.payload_index))
        if projection.node_map[str(endpoint.payload_index)] != expected_id:
            raise ValueError('shared graph stable endpoint ID mismatch')
    pairs = pd.read_csv(setup / 'prepared/pairs.csv', dtype=str, keep_default_na=False)
    expected = [dict(example_id=row['example_id'], sku_id1=projection.node_map[str(row['payload_index1'])],
                     sku_id2=projection.node_map[str(row['payload_index2'])], label=str(row['label']), split='train')
                for row in shared.iter_pair_rows()]
    actual = pairs[pairs.split == 'train'].to_dict('records')
    if actual != expected or _hash_rows(actual) != projection.train_pair_order_sha256:
        raise ValueError('graph supervision differs from exact shared example order/labels')
    if projection.train_pair_rows != len(expected):
        raise ValueError('shared graph training pair count mismatch')
    if _hash_rows(_pair_rows(pairs[pairs.split != 'train'])) != projection.clean_evaluation_pairs_sha256:
        raise ValueError('shared graph projection changed clean evaluation')
    for name, expected_hash in [('listings.json', projection.listings_sha256), ('pairs.csv', projection.pairs_sha256)]:
        if file_hash(setup / 'prepared' / name) != expected_hash:
            raise ValueError('shared graph projection input hash mismatch')
    manifest = json.loads((setup / 'prepared/input_manifest.json').read_text())
    for key, path in [('catalog_sha256', setup / 'eligible_catalog.csv'),
                      ('splits_sha256', setup / 'listing_splits.csv'),
                      ('report_attributes_sha256', setup / 'prepared/report_attributes.json')]:
        if manifest.get(key) != file_hash(path):
            raise ValueError('shared graph projection catalog/provenance mismatch')
    return projection
