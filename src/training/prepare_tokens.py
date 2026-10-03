"""Add locally prepared native training tokens to an existing validated bundle."""
from __future__ import annotations
import argparse
import gzip
import json
import os
import pickle
from pathlib import Path

from core.common import F, load_config, load_local_sentence_transformer, resolve_model
from training.prepared_bundle import load_prepared_bundle, _digest
from training.token_inputs import prepare_training_tokens, validate_training_tokens


def prepare_bundle_tokens(path: Path, checkpoint: str, *, setup: Path | None = None):
    # Fail on frozen input/config drift before spending local tokenizer time.
    os.environ['PREPARED_BUNDLE_DRIFT_STRICT'] = 'true'
    header, bundle = load_prepared_bundle(path)
    for key in ('canonical_records', 'gate_results', 'labeled_pairs'):
        if bundle[key + '_csv'] != Path(F[key]).read_bytes():
            raise ValueError(f'prepared bundle contains stale frozen {key}; rebuild CPU preparation')
    if setup is not None:
        from graph_tracks.text_cache import checkpoint_hash
        manifest = json.loads((setup / 'setup_manifest.json').read_text())
        if manifest['source_catalog_sha256'] != _digest(Path(F['dataset_deduped'])):
            raise ValueError('prepared setup contains stale source catalog; rebuild CPU preparation')
        if manifest['text_checkpoint_sha256'] != checkpoint_hash(Path(checkpoint)):
            raise ValueError('prepared setup checkpoint differs from requested native tokenizer checkpoint')
    model = load_local_sentence_transformer(checkpoint, device='cpu')
    print(f'[training tokens/local] rows={len(bundle["payload"]):,} checkpoint={checkpoint}', flush=True)
    if "training_tokens" in bundle:
        from training.token_inputs import PreparedTokenLookup
        PreparedTokenLookup(model, bundle["training_tokens"], bundle["payload"])
        print('[training tokens/local] existing native table verified; no duplicate tokenization', flush=True)
    else:
        bundle['training_tokens'] = prepare_training_tokens(model, bundle['payload'])
    validate_training_tokens(bundle['training_tokens'])
    from training.run_plan import prepare_run_plan
    bundle['training_plan'] = prepare_run_plan(bundle)
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
    print(f'[training tokens/local] complete unique={len(bundle["training_tokens"]["texts"]):,} sha256={updated.sha256}', flush=True)
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--setup', type=Path)
    args = parser.parse_args()
    checkpoint = resolve_model(args.checkpoint or load_config()['training']['base_model'])
    prepare_bundle_tokens(args.bundle, checkpoint, setup=args.setup)


if __name__ == '__main__':
    main()
