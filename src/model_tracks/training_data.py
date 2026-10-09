"""One validated training population, shared by every model adapter."""
from __future__ import annotations

from core.portable_archive import ByteCount
import io
from itertools import chain
import json
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

class TrainingJSONEncoder(json.JSONEncoder):
    """Walk these plain-field training schemas without copying their large lists.

    This is intentionally scoped to the training contracts below: they have no
    aliases, custom serializers, computed fields or non-JSON scalar types.
    """

    def default(self, value):
        if isinstance(value, BaseModel):
            return {name: getattr(value, name) for name in type(value).model_fields}
        return super().default(value)


# Stable node-ID contract for virtual (non-source) endpoints. These prefixes
# are read by the ANN catalog filter, the packaged text exporter and the graph
# projection validator, so they live here as the single definition instead of
# being repeated as literals that could drift apart silently.
CANONICAL_PREFIX = 'canonical:'
AUGMENTATION_PREFIX = 'augmentation:'
VIRTUAL_PREFIXES = (CANONICAL_PREFIX, AUGMENTATION_PREFIX)


def canonical_node_id(gtin: str) -> str:
    return CANONICAL_PREFIX + str(gtin)


def augmentation_node_id(payload_index: int) -> str:
    return f'{AUGMENTATION_PREFIX}{payload_index}'


def is_virtual_node(sku_id: object) -> bool:
    """Virtual supervision endpoints are not products in the ANN catalog."""
    return str(sku_id).startswith(VIRTUAL_PREFIXES)


FROZEN_TEXT_COLUMN = 'frozen_payload'


def frozen_endpoint_text(sku_id: object, value: object, *, column_present: bool) -> str | None:
    """Authoritative frozen text for a shared virtual endpoint, else None.

    The CSV lane writes one shared column, so an ordinary listing also carries
    an EMPTY ``frozen_payload`` cell: absence cannot be read as "not present",
    or every listing would look like a frozen endpoint. Virtualness decides,
    and for a virtual endpoint the cell is authoritative even when empty (an
    empty frozen text is a real value, never a reason to recompose).
    """
    if not is_virtual_node(sku_id):
        if column_present and str(value or '').strip():
            raise ValueError('frozen payload overrides are only valid for shared virtual endpoints')
        return None
    if not column_present:
        raise ValueError(f'shared virtual endpoint {sku_id} has no {FROZEN_TEXT_COLUMN} column')
    return '' if value is None else str(value)


class TrainingEndpoint(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    payload_index: int = Field(ge=0)
    kind: Literal['listing', 'canonical', 'augmentation']
    entity: str
    source_id: str
    parent_index: int | None = Field(default=None, ge=0)
    text_size: int = Field(ge=0)

    @model_validator(mode='after')
    def lineage(self):
        if (self.kind == 'augmentation') != (self.parent_index is not None):
            raise ValueError('only augmented endpoints must identify a parent')
        if self.parent_index == self.payload_index:
            raise ValueError('augmentation cannot be its own parent')
        return self


class TrainingExample(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    example_id: int = Field(ge=0)
    anchor: int = Field(ge=0)
    positive: int = Field(ge=0)
    negative: int = Field(ge=0)
    population: str

    @model_validator(mode='after')
    def distinct_endpoints(self):
        if len({self.anchor, self.positive, self.negative}) != 3:
            raise ValueError('training triplet endpoints must be distinct')
        return self


class SharedTrainingData(BaseModel):
    """Frozen MNRL relationships, including canonical and minted endpoints.

    Adapters may use different representations and losses, but may not drop
    examples, relabel relationships, or substitute source rows for copies.
    Repeated positive relationships retain their example multiplicity.
    """
    model_config = ConfigDict(extra='forbid', frozen=True)
    schema_version: Literal[1] = 1
    objective: Literal['mnrl'] = 'mnrl'
    source_rows: int = Field(gt=0)
    canonical_rows: int = Field(ge=0)
    payload_rows: int = Field(gt=0)
    endpoints: list[TrainingEndpoint]
    examples: list[TrainingExample] = Field(min_length=1)

    @model_validator(mode='after')
    def coverage(self):
        endpoints = {row.payload_index: row for row in self.endpoints}
        if len(endpoints) != len(self.endpoints):
            raise ValueError('duplicate training endpoint')
        if any(row.example_id != i for i, row in enumerate(self.examples)):
            raise ValueError('training examples must retain frozen row order')
        used = {i for row in self.examples for i in (row.anchor, row.positive, row.negative)}
        if used != set(endpoints):
            raise ValueError('endpoint population must exactly cover shared examples')
        canonical_end = self.source_rows + self.canonical_rows
        if canonical_end > self.payload_rows:
            raise ValueError('canonical block exceeds frozen payload')
        relationships = {}
        for row in self.iter_pair_rows():
            pair = tuple(sorted((row['payload_index1'], row['payload_index2'])))
            if pair in relationships and relationships[pair] != row['label']:
                raise ValueError('shared examples contain conflicting relationship labels')
            relationships[pair] = row['label']
        for index, row in endpoints.items():
            expected = ('listing' if index < self.source_rows else
                        'canonical' if index < canonical_end else 'augmentation')
            if index >= self.payload_rows or row.kind != expected:
                raise ValueError('endpoint does not match frozen payload layout')
            if row.parent_index is not None and row.parent_index >= canonical_end:
                raise ValueError('augmentation parent must be an original listing or canonical')
        return self

    @property
    def fingerprint(self) -> int:
        size = ByteCount()
        encoder = TrainingJSONEncoder(sort_keys=True, ensure_ascii=False)
        for chunk in encoder.iterencode(self):
            size.update(chunk.encode())
        return size.total

    def iter_pair_rows(self):
        """Yield graph relationships in frozen order without a second population."""
        for row in self.examples:
            for label, other in ((1, row.positive), (0, row.negative)):
                yield dict(example_id=f'{row.example_id}:{label}',
                           payload_index1=row.anchor, payload_index2=other, label=label)

    def pair_rows(self) -> list[dict]:
        """Exact pair projection for callers that require a materialized list."""
        return list(self.iter_pair_rows())


def from_bundle(bundle: dict, *, fold_index: int = 0) -> SharedTrainingData:
    """Use already frozen examples; never regenerate supervision here."""
    from training.folds import normalize_gtin
    canonical = pd.read_csv(io.BytesIO(bundle['canonical_records_csv']), dtype=str,
                            keep_default_na=False)
    gtins = sorted(set(canonical.gtin.astype(str)))
    source_rows = len(bundle['df'])
    canonical_end = source_rows + len(gtins)
    if list(map(str, bundle['row_bc'][source_rows:canonical_end])) != gtins:
        raise ValueError('canonical endpoint ordering differs from bundled records')
    copies = {}
    for audit in chain(bundle['mask_audit'], bundle['hard_negative_mask_audit']):
        mappings = [(audit['copy_payload_idx'], audit.get('copy_source_payload_idx') if audit.get('copy_source_payload_idx') is not None else audit['anchor_payload_idx'])]
        if audit.get('copy_pair_payload_idx') is not None:
            mappings.append((audit['copy_pair_payload_idx'], audit['pair_payload_idx']))
        for child, parent in mappings:
            child, parent = int(child), int(parent)
            if child in copies and copies[child] != parent:
                raise ValueError('conflicting augmentation parent')
            copies[child] = parent
    if set(copies) != set(range(canonical_end, len(bundle['payload']))):
        raise ValueError('augmentation lineage does not cover the payload suffix')
    plan = bundle['training_plan']
    if plan['identity']['loss'] != 'mnrl':
        raise ValueError('shared training contract currently requires frozen MNRL triples')
    folds = plan['inputs']['folds']
    if not folds:
        raise ValueError('shared training contract requires a frozen training fold')
    if not 0 <= fold_index < len(folds):
        raise ValueError(
            f'fold_index {fold_index} is outside the frozen training plan '
            f'({len(folds)} fold(s)); the shared population must name the fold '
            'actually being trained'
        )
    fold = folds[fold_index]
    objective = fold['objective']
    triples = objective['triples']
    populations = objective['dataset']['population']
    if len(triples) != len(populations):
        raise ValueError('shared example populations do not align')
    examples = [TrainingExample(example_id=i, anchor=int(a), positive=int(b), negative=int(c), population=pop)
                for i, ((a, b, c), pop) in enumerate(zip(triples, populations, strict=True))]
    used = sorted({index for row in examples for index in (row.anchor, row.positive, row.negative)})
    train_entities = {normalize_gtin(value) for value in fold['tr_bc']}
    endpoints = []
    for index in used:
        entity = str(bundle['row_bc'][index])
        if normalize_gtin(entity) not in train_entities:
            raise ValueError('shared training endpoint crosses the frozen split')
        kind = 'listing' if index < source_rows else 'canonical' if index < canonical_end else 'augmentation'
        source_id = (str(bundle['df'].iloc[index].sku_id) if kind == 'listing' else
                     gtins[index-source_rows] if kind == 'canonical' else augmentation_node_id(index))
        endpoints.append(TrainingEndpoint(payload_index=index, kind=kind, entity=entity,
            source_id=source_id, parent_index=copies.get(index),
            text_size=ByteCount(bundle['payload'][index].encode()).total))
    return SharedTrainingData(source_rows=source_rows, canonical_rows=len(gtins),
        payload_rows=len(bundle['payload']),
        endpoints=endpoints, examples=examples)


class TrackTrainingBinding(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    track: Literal['text', 'gnn_only']
    shared_data_size: int = Field(ge=0)
    example_ids: list[int]
    endpoint_indices: list[int]

    def validate_data(self, shared: SharedTrainingData):
        if (self.shared_data_size != shared.fingerprint or
                self.example_ids != [row.example_id for row in shared.examples] or
                self.endpoint_indices != [row.payload_index for row in shared.endpoints]):
            raise ValueError(f'{self.track} training population differs from shared data')


def retrieval_indices(records: list[dict]) -> list[int]:
    """Virtual supervision endpoints are not products in the ANN catalog."""
    return [index for index, row in enumerate(records) if not is_virtual_node(row['sku_id'])]
