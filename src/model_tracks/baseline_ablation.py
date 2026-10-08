"""Frozen baseline ablation before training, with saved-vector CPU reporting."""
from core.portable_archive import ByteCount
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from core.bundle import bundle_spec
from core.run_log import RunLogger
from core.step_trace import timed
from core.tracing import flush_stage_trace, stage_trace
from graph_tracks.data import file_size, load_records, load_text_cache
from graph_tracks.report import dev_threshold
from graph_tracks.train import load_pairs
from model_tracks.ablation import checkpoint_identity, report, request_context, resolve, source_name, write
from model_tracks.staged_ablation import forward as forward_staged

_LOG = RunLogger(__name__)


def _setup_layout():
    """The declared prepared-setup layout (training.preparation.graph_setup)."""
    from core.common import prepared_setup_layout
    return prepared_setup_layout()


#: The stage name this module owns in the ONE consolidated pipeline trace.
STAGE = "baseline_ablation"

#: The module's trace writer: the shared shim's slot (``None`` until first use;
#: see :func:`core.tracing.stage_trace`), so importing this module never touches
#: the trace layout.
_TRACE = None


def trace():
    """The ONE writer for the ``baseline_ablation`` stage of the current run."""
    global _TRACE
    _TRACE = stage_trace(STAGE, _TRACE)
    return _TRACE


def flush_trace():
    """Commit this process's baseline-ablation rows once; a no-op while empty."""
    return flush_stage_trace(_TRACE)


class BaselineCalibration(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid', allow_inf_nan=False)
    track: Literal['text'] = 'text'
    checkpoint_role: Literal['baseline'] = 'baseline'
    checkpoint_size: int = Field(ge=0)
    vectors_size: int = Field(ge=0)
    listings_size: int = Field(ge=0)
    pairs_size: int = Field(ge=0)
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
        spec = bundle_spec()
        template_path = setup/spec.ablation_templates_dir/'text'/spec.ablation_request_file
        template = json.loads(template_path.read_text())
        saved = (setup/_setup_layout().shared_embeddings
                 if template['settings']['retrieval_catalog'] == 'full' and template['settings'].get('coverage') != 'all' else None)
        trace().add(
            "forward", "text_template",
            in_count=1, out_count=1, key='text',
            reason='the untrained baseline reuses the frozen text template and its saved catalog vectors',
            detail={'template': source_name(template_path),
                    'retrieval_catalog': template['settings']['retrieval_catalog'],
                    'coverage': template['settings'].get('coverage'),
                    'saved_text': None if saved is None else source_name(saved),
                    'checkpoint': source_name(checkpoint), 'device': device},
            source=source_name(template_path),
        )
        path = forward_staged(output, setup, 'text', checkpoint, device=device,
                              checkpoint_role='baseline', saved_text=saved, text_model=text_model)
    flush_trace()
    return path


@timed
def _saved_vectors(records, output):
    """Vectors and metadata for the saved catalog snapshot."""
    with _LOG.section('ablation.baseline.saved_vectors'):
        vectors_path = output/_setup_layout().shared_embeddings
        vectors, metadata = load_text_cache(vectors_path, [row['sku_id'] for row in records])
        return vectors_path, vectors, metadata


@timed
def _dev_scores(pairs_path, records, vectors):
    """Dev-split indices/labels with dot-product scores from saved vectors."""
    with _LOG.section('ablation.baseline.dev_scores'):
        indices, labels = load_pairs(pairs_path, records)['dev']
        return labels, (vectors[indices[:, 0]]*vectors[indices[:, 1]]).sum(-1)


@timed
def _calibrated_calibration(checkpoint_size, vectors_path, metadata, records_path, pairs_path, labels, scores):
    """The BaselineCalibration for the untrained checkpoint, or identity raise."""
    with _LOG.section('ablation.baseline.calibrate'):
        if metadata.get('checkpoint_size') != checkpoint_size:
            from core.tracing import SCOPE_ENTITY
            trace().add(
                "complete", "calibration_rejected",
                scope=SCOPE_ENTITY, key='text',
                reason='baseline calibration vectors differ from the frozen checkpoint; '
                       'the baseline is quarantined rather than refit',
                detail={'vectors_checkpoint_size': metadata.get('checkpoint_size'),
                        'frozen_checkpoint_size': checkpoint_size,
                        'vectors': source_name(vectors_path)},
                source=source_name(vectors_path),
            )
            flush_trace()
            raise ValueError('baseline calibration vectors differ from frozen checkpoint')
        calibration = BaselineCalibration(checkpoint_size=checkpoint_size,
            vectors_size=file_size(vectors_path), listings_size=file_size(records_path),
            pairs_size=file_size(pairs_path), threshold=dev_threshold(labels, scores),
            dev_pairs=len(labels), dev_positives=int(labels.sum()),
            dev_negatives=int((labels == 0).sum()))
    trace().add(
        "complete", "calibration",
        in_count=int(len(labels)), out_count=1, key='text',
        reason='the untrained baseline threshold is fit on the dev split and never on test',
        detail={'checkpoint_size': checkpoint_size,
                'vectors_size': calibration.vectors_size,
                'listings_size': calibration.listings_size,
                'pairs_size': calibration.pairs_size,
                'threshold': calibration.threshold,
                'dev_pairs': calibration.dev_pairs,
                'dev_positives': calibration.dev_positives,
                'dev_negatives': calibration.dev_negatives,
                'threshold_source': calibration.threshold_source,
                'test_used_for_selection': calibration.test_used_for_selection},
        source=source_name(pairs_path),
    )
    return calibration


@timed
def _frozen_report(request_path, calibration, *, saved=None):
    """Compute the threshold-frozen report at the baseline calibration threshold."""
    with _LOG.section('ablation.baseline.frozen_report'):
        result = report(request_path, request_path.parent/'vectors.npz', calibration.threshold,
                        threshold_source=str(saved), save=False)
    trace().add(
        "complete", "frozen_report",
        # Two different populations (dev calibration pairs vs comparison rows):
        # a validation/reporting row, not a funnel.
        in_count=None, out_count=len(result['rows']), key='text',
        reason='the baseline report is computed at the frozen calibration threshold with no refit',
        detail={'request_path': source_name(request_path),
                'dev_pairs': int(calibration.dev_pairs),
                'threshold': result['threshold'],
                'rows': len(result['rows']),
                'retrieval_catalog_count': result.get('retrieval_catalog_count')},
        source=source_name(request_path.parent/'vectors.npz'),
    )
    return result


@timed
def _frozen_calibration(request_path, calibration):
    """Seal the calibration document into baseline_threshold.json, or refuse drift."""
    with _LOG.section('ablation.baseline.seal'):
        binding = request_path.parent/'baseline_threshold.json'
        document = calibration.model_dump(mode='json')
        if binding.exists() and json.loads(binding.read_text()) != document:
            from core.tracing import SCOPE_ENTITY
            trace().add(
                "complete", "binding_rejected",
                scope=SCOPE_ENTITY, key='text',
                reason='frozen baseline calibration changed during completion; the sealed '
                       'binding is never overwritten',
                detail={'binding': source_name(binding)},
                source=source_name(binding),
            )
            flush_trace()
            raise ValueError('frozen baseline calibration changed during completion')
        write(binding, document)
        trace().add(
            "complete", "binding",
            in_count=1, out_count=1, key='text',
            reason='the untrained baseline checkpoint identity and frozen threshold are sealed together',
            detail={'binding': source_name(binding),
                    'checkpoint_size': calibration.checkpoint_size,
                    'threshold': calibration.threshold,
                    'pre_existed': binding.exists()},
            source=source_name(binding),
        )
        return binding


@timed
def _persist_baseline(request_path, result):
    """Write the baseline report beside its request with the sha sidecar."""
    with _LOG.section('ablation.baseline.persist'):
        path = request_path.parent/'report.json'
        # One serialize + one write. `ablation.write()` streams the very same
        # encoder output through json.dump, which costs one Python-level
        # handle.write per emitted chunk (measured: ~900k calls, 0.26s of the
        # 0.32s) — this payload is produced by the identical encoder arguments
        # and trailing newline, so the bytes on disk are unchanged.
        payload = (json.dumps(result, sort_keys=True, ensure_ascii=False,
                              indent=2, allow_nan=False) + '\n').encode('utf-8')
        path.write_bytes(payload)
        # Hash the bytes we just wrote instead of reading 7 MB back off disk.
        digest = ByteCount(payload).total
        (request_path.parent/'report.size').write_text(str(digest)+'\n')
        trace().add(
            "complete", "persisted",
            in_count=None, out_count=2, key='text',
            reason='the baseline report and its size sidecar are written beside the request',
            detail={'report': source_name(path), 'report_size': digest,
                    'sidecar': source_name(request_path.parent/'report.size'),
                    'rows': len(result['rows']), 'threshold': result['threshold']},
            source=source_name(path),
        )
    return path


@timed
def complete(output: Path, setup: Path, *, config: Path | None = None):
    """Fit the untrained baseline threshold on dev; consume saved ablation only."""
    with _LOG.section('ablation.baseline.load'):
        request_path = output/'ablation'/bundle_spec().ablation_request_file
        request = json.loads(request_path.read_text())
        if request['track'] != 'text' or request.get('checkpoint_role') != 'baseline':
            raise ValueError('baseline report requires the frozen baseline ablation')
        layout = _setup_layout()
        records_path, pairs_path = (setup/layout.prepared_dir/layout.listings,
                                    setup/layout.prepared_dir/'pairs.csv')
        records = load_records(records_path)
        vectors_path, vectors, metadata = _saved_vectors(records, output)
        labels, scores = _dev_scores(pairs_path, records, vectors)
        trace().add(
            "complete", "load",
            # Two different populations (listing records vs dev pair rows):
            # a load row, not a funnel.
            in_count=None, out_count=int(len(labels)), key='text',
            reason='the untrained baseline consumes only the saved catalog vectors and the dev labels',
            detail={'request_path': source_name(request_path), 'track': request['track'],
                    'checkpoint_role': request.get('checkpoint_role'),
                    'checkpoint': request['checkpoint'], 'records': len(records),
                    'dev_pairs': int(len(labels)),
                    'vectors': source_name(vectors_path),
                    'vectors_bytes': vectors_path.stat().st_size},
            source=source_name(request_path),
        )
    with _LOG.section('ablation.baseline.calibration'):
        with request_context(request_path):
            checkpoint_size = checkpoint_identity(resolve(request['checkpoint']))
            calibration = _calibrated_calibration(checkpoint_size, vectors_path, metadata,
                                                  records_path, pairs_path, labels, scores)
            binding = _frozen_calibration(request_path, calibration)
            result = _frozen_report(request_path, calibration, saved=binding)
            # Keep this baseline report separate from trained-text dashboard pointers.
            _persist_baseline(request_path, result)
    flush_trace()
    return request_path.parent/'report.json'
