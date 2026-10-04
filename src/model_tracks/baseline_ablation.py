"""Frozen baseline ablation before training, with saved-vector CPU reporting."""
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from graph_tracks.data import file_hash, load_records, load_text_cache
from model_tracks.ablation import checkpoint_identity, report, request_context, write


class BaselineCalibration(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid', allow_inf_nan=False)
    track: Literal['text'] = 'text'
    checkpoint_role: Literal['baseline'] = 'baseline'
    checkpoint_sha256: str
    vectors_sha256: str
    listings_sha256: str
    pairs_sha256: str
    threshold: float
    threshold_source: Literal['dev_youden'] = 'dev_youden'
    calibration_split: Literal['dev'] = 'dev'
    dev_pairs: int = Field(gt=0)
    dev_positives: int = Field(gt=0)
    dev_negatives: int = Field(gt=0)
    test_used_for_selection: Literal[False] = False
    retraining: Literal[False] = False


def forward(output: Path, setup: Path, checkpoint: Path, *, device: str):
    """Reuse text interventions and frozen catalog vectors in this suite session."""
    from model_tracks.staged_ablation import forward as forward_staged
    template = json.loads((setup/'ablation_templates/text/request.json').read_text())
    saved = (setup/'shared_minilm__embeddings.npz'
             if template['settings']['retrieval_catalog'] == 'full' else None)
    return forward_staged(output, setup, 'text', checkpoint, device=device,
                          checkpoint_role='baseline', saved_text=saved)


def complete(output: Path, setup: Path, *, config: Path | None = None):
    """Fit the untrained baseline threshold on dev; consume saved ablation only."""
    from graph_tracks.report import dev_threshold
    from graph_tracks.train import load_pairs
    from model_tracks.ablation import resolve
    request_path = output/'ablation/request.json'
    request = json.loads(request_path.read_text())
    if request['track'] != 'text' or request.get('checkpoint_role') != 'baseline':
        raise ValueError('baseline report requires the frozen baseline ablation')
    records_path, pairs_path = setup/'prepared/listings.json', setup/'prepared/pairs.csv'
    records = load_records(records_path)
    vectors_path = output/'shared_minilm__embeddings.npz'
    vectors, metadata = load_text_cache(vectors_path, [row['sku_id'] for row in records])
    indices, labels = load_pairs(pairs_path, records)['dev']
    scores = (vectors[indices[:, 0]]*vectors[indices[:, 1]]).sum(-1)
    with request_context(request_path):
        checkpoint_sha256 = checkpoint_identity(resolve(request['checkpoint']))
        if metadata.get('checkpoint_sha256') != checkpoint_sha256:
            raise ValueError('baseline calibration vectors differ from frozen checkpoint')
        calibration = BaselineCalibration(checkpoint_sha256=checkpoint_sha256,
            vectors_sha256=file_hash(vectors_path), listings_sha256=file_hash(records_path),
            pairs_sha256=file_hash(pairs_path), threshold=dev_threshold(labels, scores),
            dev_pairs=len(labels), dev_positives=int(labels.sum()),
            dev_negatives=int((labels == 0).sum()))
        binding = request_path.parent/'baseline_threshold.json'
        document = calibration.model_dump(mode='json')
        if binding.exists() and json.loads(binding.read_text()) != document:
            raise ValueError('frozen baseline calibration changed during completion')
        write(binding, document)
        result = report(request_path, request_path.parent/'vectors.npz', calibration.threshold,
                        threshold_source=str(binding), config=config, save=False)
        # Keep this baseline report separate from trained-text dashboard pointers.
        write(request_path.parent/'report.json', result)
        (request_path.parent/'report.sha256').write_text(file_hash(request_path.parent/'report.json')+'\n')
    return request_path.parent/'report.json'
