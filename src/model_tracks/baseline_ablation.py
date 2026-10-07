"""Frozen baseline ablation before training, with saved-vector CPU reporting."""
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from core.run_log import RunLogger
from core.step_trace import timed
from graph_tracks.data import load_records, load_text_cache
from graph_tracks.report import dev_threshold
from graph_tracks.train import load_pairs
from model_tracks.ablation import report, request_context, write
from model_tracks.staged_ablation import forward as forward_staged

_LOG = RunLogger(__name__)


class BaselineCalibration(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid', allow_inf_nan=False)
    track: Literal['text'] = 'text'
    checkpoint_role: Literal['baseline'] = 'baseline'
    checkpoint_sha256: str
    threshold: float
    threshold_source: Literal['dev_youden'] = 'dev_youden'
    calibration_split: Literal['dev'] = 'dev'
    dev_pairs: int = Field(gt=0)
    dev_positives: int = Field(gt=0)
    dev_negatives: int = Field(gt=0)
    test_used_for_selection: Literal[False] = False
    retraining: Literal[False] = False


@timed
def forward(output: Path, setup: Path, checkpoint: Path, *, device: str, text_model=None):
    """Reuse text interventions and frozen catalog vectors in this suite session."""
    with _LOG.section('ablation.baseline.forward'):
        template = json.loads((setup/'ablation_templates/text/request.json').read_text())
        saved = (setup/'shared_minilm__embeddings.npz'
                 if template['settings']['retrieval_catalog'] == 'full' and template['settings'].get('coverage') != 'all' else None)
        return forward_staged(output, setup, 'text', checkpoint, device=device,
                              checkpoint_role='baseline', saved_text=saved, text_model=text_model)


@timed
def _saved_vectors(records, output):
    """Vectors and metadata for the saved catalog snapshot."""
    with _LOG.section('ablation.baseline.saved_vectors'):
        vectors, metadata = load_text_cache(output/'shared_minilm__embeddings.npz',
                                            [row['sku_id'] for row in records])
        return vectors, metadata


@timed
def _dev_scores(pairs_path, records, vectors):
    """Dev-split indices/labels with dot-product scores from saved vectors."""
    with _LOG.section('ablation.baseline.dev_scores'):
        indices, labels = load_pairs(pairs_path, records)['dev']
        return labels, (vectors[indices[:, 0]]*vectors[indices[:, 1]]).sum(-1)


@timed
def _calibrated_calibration(checkpoint_sha256, metadata, labels, scores):
    """The BaselineCalibration for the untrained checkpoint, or identity raise."""
    with _LOG.section('ablation.baseline.calibrate'):
        if metadata.get('checkpoint_sha256') != checkpoint_sha256:
            raise ValueError('baseline calibration vectors differ from frozen checkpoint')
        return BaselineCalibration(checkpoint_sha256=checkpoint_sha256,
            threshold=dev_threshold(labels, scores),
            dev_pairs=len(labels), dev_positives=int(labels.sum()),
            dev_negatives=int((labels == 0).sum()))


@timed
def _frozen_report(request_path, request, calibration, *, config, saved=None):
    """Compute the threshold-frozen report at the baseline calibration threshold."""
    with _LOG.section('ablation.baseline.frozen_report'):
        return report(request_path, request_path.parent/'vectors.npz', calibration.threshold,
                      threshold_source=str(saved), config=config, save=False)


@timed
def _frozen_calibration(request_path, calibration):
    """Seal the calibration document into baseline_threshold.json."""
    with _LOG.section('ablation.baseline.seal'):
        binding = request_path.parent/'baseline_threshold.json'
        write(binding, calibration.model_dump(mode='json'))
        return binding


@timed
def _persist_baseline(request_path, result):
    """Write the baseline report beside its request."""
    with _LOG.section('ablation.baseline.persist'):
        write(request_path.parent/'report.json', result)


@timed
def complete(output: Path, setup: Path, *, config: Path | None = None):
    """Fit the untrained baseline threshold on dev; consume saved ablation only."""
    with _LOG.section('ablation.baseline.load'):
        request_path = output/'ablation/request.json'
        request = json.loads(request_path.read_text())
        if request['track'] != 'text' or request.get('checkpoint_role') != 'baseline':
            raise ValueError('baseline report requires the frozen baseline ablation')
        records_path, pairs_path = setup/'prepared/listings.json', setup/'prepared/pairs.csv'
        records = load_records(records_path)
        vectors, metadata = _saved_vectors(records, output)
        labels, scores = _dev_scores(pairs_path, records, vectors)
    with _LOG.section('ablation.baseline.calibration'):
        with request_context(request_path):
            checkpoint_sha256 = request['sources'][request['checkpoint']]
            calibration = _calibrated_calibration(checkpoint_sha256, metadata, labels, scores)
            binding = _frozen_calibration(request_path, calibration)
            result = _frozen_report(request_path, request, calibration, config=config, saved=binding)
            # Keep this baseline report separate from trained-text dashboard pointers.
            _persist_baseline(request_path, result)
    return request_path.parent/'report.json'
