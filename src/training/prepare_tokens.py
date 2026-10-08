"""Add locally prepared native training tokens to an existing validated bundle."""
from __future__ import annotations
import argparse
import gzip
import pickle
from pathlib import Path

from core.common import load_config, load_local_sentence_transformer, resolve_model
from core.run_log import RunLogger
from core.step_trace import timed
from training.prepared_bundle import load_prepared_bundle, _size_of
from training.token_inputs import prepare_training_tokens, validate_training_tokens

_LOG = RunLogger(__name__)


def _verify_or_prepare_tokens(model, bundle: dict) -> None:
    """Reuse a compatible native table, else build it once (rebuild silently).

    Integrity only (owner directive 2026-10-08): the recorded token policy and
    payload digest are checked against the bundle's payload. An incompatible
    table is not called stale — it is rebuilt from the payload in place.
    """
    if "training_tokens" in bundle:
        from training.token_inputs import PreparedTokenLookup
        try:
            PreparedTokenLookup(model, bundle["training_tokens"], bundle["payload"])
        except ValueError:
            _LOG.info('[training tokens/local] existing native table is incompatible; rebuilding')
        else:
            _LOG.info('[training tokens/local] existing native table verified; no duplicate tokenization')
            return
    bundle['training_tokens'] = prepare_training_tokens(model, bundle['payload'])


def _atomic_save_bundle(path: Path, header, bundle: dict):
    """Replace the bundle + its sidecar with the token-extended version, atomically."""
    temporary = path.with_suffix(path.suffix + '.tokens.tmp')
    try:
        with gzip.open(temporary, 'wb', compresslevel=6) as stream:
            pickle.dump(bundle, stream, protocol=pickle.HIGHEST_PROTOCOL)
        updated = header.model_copy(update={'size': _size_of(temporary)})
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
    # The bundle's own digest is verified at load; no provenance freshness gate.
    header, bundle = load_prepared_bundle(path)
    model = load_local_sentence_transformer(checkpoint, device='cpu')
    _LOG.info(f'[training tokens/local] rows={len(bundle["payload"]):,} checkpoint={checkpoint}')
    _verify_or_prepare_tokens(model, bundle)
    validate_training_tokens(bundle['training_tokens'])
    from training.run_plan import prepare_run_plan
    bundle['training_plan'] = prepare_run_plan(bundle)
    updated = _atomic_save_bundle(path, header, bundle)
    _LOG.info(f'[training tokens/local] complete unique={len(bundle["training_tokens"]["texts"]):,} size={updated.size}')
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
