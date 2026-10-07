"""Track-safe artifact names and checkpoint integrity shared by graph workers."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, StrictBool
from core.portable_archive import Digest
from core.step_trace import timed
from graph_tracks.data import file_hash

class GraphExportManifest(BaseModel):
    """Saved catalog forward provenance shared by exporter and CPU reporter."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)
    schema_id: Literal['er-graph-export-v1'] = Field(alias='schema')
    track: Literal['gnn_only', 'hybrid']
    checkpoint_sha256: Digest
    listings_sha256: Digest
    vectors_sha256: Digest
    text_cache_sha256: Digest | None
    graph_context: Literal['training-listings-only']
    vector_kind: Literal['graph-informed']
    ann_reproduces_pair_scorer: Literal[False]
    count: int = Field(gt=0)
    dimension: int = Field(gt=0)
    index_built: StrictBool
    embedding_dtype: Literal['float32']
    performance: dict[str, Any]
    id_kind: Literal['listing_sku_id']


class GraphForwardManifest(GraphExportManifest):
    pairs_sha256: Digest
    split_scores_sha256: Digest
    report_test: StrictBool
    forward_only: Literal[True]


TRACKS = {'gnn_only', 'hybrid'}
_NAME_TRACKS = TRACKS | {'text'}


def _validate_name_track(track: str) -> None:
    """Reject any track outside the set of nameable artifact tracks."""
    if track not in _NAME_TRACKS:
        raise ValueError(f'unknown graph track: {track}')


def name(track: str, stem: str) -> str:
    _validate_name_track(track)
    return f'{track}__{stem}'


def _verified_track(checkpoint: Path, track: str) -> str:
    """Accept `checkpoint` as `track`'s model file, or raise with the reason."""
    marker = checkpoint.parent / name(track, 'checkpoint_manifest.json')
    if not marker.is_file():
        raise ValueError('checkpoint missing completion marker')
    metadata = json.loads(marker.read_text())
    if metadata.get('track') != track or metadata.get('files', {}).get(checkpoint.name) != file_hash(checkpoint):
        raise ValueError('checkpoint track/hash mismatch')
    return track


@timed
def checkpoint_track(checkpoint: Path) -> str:
    for track in TRACKS:
        if checkpoint.name == name(track, 'graph_model.pt'):
            return _verified_track(checkpoint, track)
    raise ValueError('checkpoint filename must identify its graph track')
