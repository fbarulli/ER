"""Add locally prepared native training tokens to an existing validated bundle."""
from __future__ import annotations
import argparse
import gzip
import json
import os
import pickle
from pathlib import Path

from core.common import F, load_config, load_local_sentence_transformer, resolve_model
from core.run_log import RunLogger
from core.step_trace import timed
from training.prepared_bundle import load_prepared_bundle, _digest
from training.token_inputs import prepare_training_tokens, validate_training_tokens

_LOG = RunLogger(__name__)
_FROZEN_INPUT_KEYS = ('canonical_records', 'gate_results', 'labeled_pairs')


def _fail_on_stale_frozen(bundle: dict) -> None:
    """Every frozen artifact in the bundle must still match the on-disk F map."""
    for key in _FROZEN_INPUT_KEYS:
        if bundle[key + '_csv'] != Path(F[key]).read_bytes():
            raise ValueError(f'prepared bundle contains stale frozen {key}; rebuild CPU preparation')


def _fail_on_setup_drift(setup: Path, checkpoint: str) -> None:
    """A prepared setup must match the on-disk catalog and the requested checkpoint."""
    from graph_tracks.text_cache import checkpoint_hash
    manifest = json.loads((setup / 'setup_manifest.json').read_text())
    if manifest['source_catalog_sha256'] != _digest(Path(F['dataset_deduped'])):
        raise ValueError('prepared setup contains stale source catalog; rebuild CPU preparation')
    if manifest['text_checkpoint_sha256'] != checkpoint_hash(Path(checkpoint)):
        raise ValueError('prepared setup checkpoint differs from requested native tokenizer checkpoint')


def _verify_or_prepare_tokens(model, bundle: dict) -> None:
    """Reuse an existing native table (verified) or tokenize the payload once."""
    if "training_tokens" in bundle:
        from training.token_inputs import PreparedTokenLookup
        PreparedTokenLookup(model, bundle["training_tokens"], bundle["payload"])
        _LOG.info('[training tokens/local] existing native table verified; no duplicate tokenization')
    else:
        bundle['training_tokens'] = prepare_training_tokens(model, bundle['payload'])


def _atomic_save_bundle(path: Path, header, bundle: dict):
    """Replace the bundle + its sidecar with the token-extended version, atomically."""
    temporary = path.with_suffix(path.suffix + '.tokens.tmp')
    try:
        with gzip.open(temporary, 'wb', compresslevel=6) as stream:
            pickle.dump(bundle, stream, protocol=pickle.HIGHEST_PROTOCOL)
        updated = header.model_copy(update={'sha256': _digest(temporary)})
        temporary.replace(path)
        sidecar = path.with_suffix(path.suffix + '.json')
        sidecar_tmp = sidecar.with_suffix(sidecar.suffix + '.tmp')
        sidecar_tmp.write_text(updated.model_dump_json(indent=2) + '\n')
        sidecar_tmp.replace(sidecar)
    finally:
        temporary.unlink(missing_ok=True)
    return updated


@timed(prefix='prepare_tokens')
def prepare_bundle_tokens(path: Path, checkpoint: str, *, setup: Path | None = None):
    # Fail on frozen input/config drift before spending local tokenizer time.
    os.environ['PREPARED_BUNDLE_DRIFT_STRICT'] = 'true'
    header, bundle = load_prepared_bundle(path)
    _fail_on_stale_frozen(bundle)
    if setup is not None:
        _fail_on_setup_drift(setup, checkpoint)
    model = load_local_sentence_transformer(checkpoint, device='cpu')
    _LOG.info(f'[training tokens/local] rows={len(bundle["payload"]):,} checkpoint={checkpoint}')
    _verify_or_prepare_tokens(model, bundle)
    validate_training_tokens(bundle['training_tokens'])
    from training.run_plan import prepare_run_plan
    bundle['training_plan'] = prepare_run_plan(bundle)
    updated = _atomic_save_bundle(path, header, bundle)
    _LOG.info(f'[training tokens/local] complete unique={len(bundle["training_tokens"]["texts"]):,} sha256={updated.sha256}')
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--setup', type=Path)
    args = parser.parse_args()
    RunLogger.configure_console()
    checkpoint = resolve_model(args.checkpoint or load_config()['training']['base_model'])
    prepare_bundle_tokens(args.bundle, checkpoint, setup=args.setup)


if __name__ == '__main__':
    main()
