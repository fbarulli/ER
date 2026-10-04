"""Automatic inference-only ablations after selected model publication."""
import importlib.util
import json
from pathlib import Path
from core.common import TRAIN_ROOT
from core.portable_archive import verify_archive
from graph_tracks.artifacts import name
from model_tracks.ablation import report, checkpoint_identity, source_name, write, resolve, frozen_threshold, verify_threshold_binding


def complete_saved(destination,suite,*,publisher=None):
    """Consume suite GPU exports after shutdown; no provisioning or forwards."""
    from model_tracks.ablation import request_context, validate_vectors
    outputs = {}
    for track in ('text','gnn_only','hybrid'):
        request = destination/track/'ablation/request.json'
        result = request.parent/'vectors.npz'
        if not request.is_file() or not result.is_file():
            raise ValueError('suite lacks prepared GPU ablation export: '+track)
        sources = list((destination/track).rglob('text__completion_manifest.json' if track == 'text' else name(track,'report_manifest.json')))
        sources = [path for path in sources if not any(part.startswith('interrupted-') or '.interrupted-' in part for part in path.parts)]
        if len(sources) != 1:
            raise ValueError('ambiguous baseline calibration manifest: '+track)
        calibration = json.loads(sources[0].read_text())
        threshold = calibration['threshold']
        document = json.loads(request.read_text())
        with request_context(request):
            checkpoint = resolve(document['checkpoint'])
            selected_identity = checkpoint_identity(checkpoint)
            calibrated_identity = calibration['checkpoint_sha256']
            if calibrated_identity != selected_identity:
                raise ValueError('baseline calibration differs from selected ablation checkpoint: '+track)
            binding = request.parent/'baseline_threshold.json' 
            write(binding,{'track':track,'checkpoint_sha256':selected_identity,
                'threshold':threshold,'calibration':{'threshold':threshold},
                'source_calibration':source_name(sources[0]),
                'source_calibration_sha256':__import__('graph_tracks.data',fromlist=['file_hash']).file_hash(sources[0]),
                'threshold_source':'saved dev calibration; no refit'})
            previous = request.parent/'report.json'
            report_digest = previous.with_suffix('.sha256')
            validated = json.loads(previous.read_text()) if previous.exists() else None
            from graph_tracks.data import file_hash
            if validated and report_digest.is_file() and report_digest.read_text().strip() == file_hash(previous) and validated.get('request_sha256') == file_hash(request) and validated.get('result_sha256') == file_hash(result) and validated.get('threshold') == threshold and validated.get('threshold_provenance',{}).get('sha256') == file_hash(binding):
                validate_vectors(request,result)
                attestation = frozen_threshold(str(binding),threshold)
                if validated.get('threshold_binding') != verify_threshold_binding(document,attestation) or validated.get('threshold_provenance') != attestation:
                    raise ValueError('cached ablation calibration binding differs')
            else:
                validated = report(request,result,threshold,threshold_source=str(binding),save=False,config=resolve(suite.ablation_config))
        from model_tracks.ablation import save_report
        save_report(request,validated,config=resolve(suite.ablation_config))
        previous.with_suffix('.sha256').write_text(file_hash(previous)+'\n')
        outputs[track] = {'request':source_name(request),'status':'verified saved GPU result'}
        if publisher is not None:
            artifact = publisher(request,result,validated,str(binding))
            outputs[track]['artifact'] = source_name(artifact)
    receipt = destination/'post_training_ablation.json'
    write(receipt,{'tracks':outputs,'retraining':False,'gpu_reopened':False})
    return receipt


def git_publisher(suite):
    """The git-side ablation publisher for a suite, or None when not publishing.

    Split out of :func:`run` so a caller that must produce the reports BEFORE
    archiving them can still publish the very same artifacts in one pass,
    instead of running :func:`complete_saved` twice.
    """
    if not suite.publish_git:
        return None
    spec = importlib.util.spec_from_file_location(
        'saved_gpu_ablation_publisher', TRAIN_ROOT / 'scripts/run_colab_ablation.py')
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.path.insert(0, str(TRAIN_ROOT / 'scripts'))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module.persist_result


def run(archive, run_tag, suite, *, launcher=None):
    archive_metadata = verify_archive(archive,'suite_bundle_manifest.json')
    destination = archive.parent/run_tag
    if any((destination/track/'ablation/request.json').exists() for track in ('text','gnn_only','hybrid')):
        return complete_saved(destination,suite,publisher=git_publisher(suite))
    raise ValueError('suite lacks staged GPU ablation exports; rebuild prepared inputs before training')
