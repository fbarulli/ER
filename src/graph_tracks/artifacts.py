"""Track-safe artifact names and checkpoint integrity shared by graph workers."""
from __future__ import annotations
import json
from pathlib import Path
from graph_tracks.data import file_hash

TRACKS = {'gnn_only', 'hybrid'}


def name(track: str, stem: str) -> str:
    if track not in TRACKS | {'text'}:
        raise ValueError(f'unknown graph track: {track}')
    return f'{track}__{stem}'


def checkpoint_track(checkpoint: Path) -> str:
    for track in TRACKS:
        if checkpoint.name == name(track, 'graph_model.pt'):
            marker = checkpoint.parent / name(track, 'checkpoint_manifest.json')
            if not marker.is_file():
                raise ValueError('checkpoint missing completion marker')
            metadata = json.loads(marker.read_text())
            if metadata.get('track') != track or metadata.get('files', {}).get(checkpoint.name) != file_hash(checkpoint):
                raise ValueError('checkpoint track/hash mismatch')
            return track
    raise ValueError('checkpoint filename must identify its graph track')
