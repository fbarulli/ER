"""Track-safe artifact names and checkpoint integrity shared by graph workers."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, StrictBool
from core.portable_archive import Digest
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


class TrackArtifacts:
    """Track-scoped artifact names and checkpoint integrity.

    Single responsibility per method:
      stem            — the track's <track>__<stem> artifact name (validated)
      checkpoint_track — the checkpoint's track, verified against the
                         completion marker (track + file hash)
      _marker_path    — where the completion marker lives
      _marker_sha_ok  — does the marker pin THIS checkpoint's bytes
    """

    def stem(self, track: str, stem: str) -> str:
        if track not in TRACKS | {'text'}:
            raise ValueError(f'unknown graph track: {track}')
        return f'{track}__{stem}'

    def checkpoint_track(self, checkpoint: Path) -> str:
        for track in TRACKS:
            if checkpoint.name == self.stem(track, 'graph_model.pt'):
                marker = self._marker_path(checkpoint, track)
                if not marker.is_file():
                    raise ValueError('checkpoint missing completion marker')
                metadata = json.loads(marker.read_text())
                if metadata.get('track') != track or metadata.get('files', {}).get(checkpoint.name) != file_hash(checkpoint):
                    raise ValueError('checkpoint track/hash mismatch')
                return track
        raise ValueError('checkpoint filename must identify its graph track')

    def _marker_path(self, checkpoint: Path, track: str) -> Path:
        return checkpoint.parent / self.stem(track, 'checkpoint_manifest.json')


_ARTIFACTS = TrackArtifacts()


def name(track: str, stem: str) -> str:
    """The track's <track>__<stem> artifact name — see TrackArtifacts.stem."""
    return _ARTIFACTS.stem(track, stem)


def checkpoint_track(checkpoint: Path) -> str:
    """The checkpoint's verified graph track — see TrackArtifacts."""
    return _ARTIFACTS.checkpoint_track(checkpoint)
