"""Colab worker: encode locally prepared text, never compose or reuse a cache.

RESPONSIBILITY MAP (single-responsibility decomposition; behaviour pinned)
-------------------------------------------------------------------------
- :class:`CheckpointDigest` — directory-order sha256 of the uploaded
  checkpoint (:func:`checkpoint_hash` stays the public face).
- :class:`EncodeRequest` — the request contract: schema, id/text population,
  checkpoint binding, locally prepared token archive binding, and worker
  tokenizer-policy equality (fail-loud on every mismatch).
- :class:`BatchEncoder` — the prepared-token batches -> normalized float32
  embeddings loop.
- :func:`main` — the CLI orchestrator (paths/device + publish).
"""
import argparse
import hashlib
import json
from pathlib import Path

try:
    from contextlib import contextmanager

    from core.run_log import RunLogger
    from training.prepare_all_trace import timed

    _LOG = RunLogger('encode_prepared_embeddings')
except ImportError:  # Colab ships this script standalone; prints stay contract
    from contextlib import contextmanager

    class _Fallback:
        @staticmethod
        def info(message):
            print(message, flush=True)

        @staticmethod
        def progress(iterable, **kwargs):
            return iterable

        @staticmethod
        @contextmanager
        def section(label, **kwargs):
            yield

    def timed(function=None, **kwargs):
        if function is not None:
            return function

        def decorate(candidate):
            return candidate

        return decorate

    _LOG = _Fallback()

REQUEST_SCHEMA = 'er-embedding-request-v2'


class CheckpointDigest:
    """The path-ordered directory digest (order makes it deterministic)."""

    @staticmethod
    def of(path: Path) -> str:
        digest = hashlib.sha256()
        for file in sorted(path.rglob('*')):
            if file.is_file():
                digest.update(str(file.relative_to(path)).encode())
                digest.update(hashlib.sha256(file.read_bytes()).digest())
        return digest.hexdigest()


def checkpoint_hash(path):
    return CheckpointDigest.of(path)


class EncodeRequest:
    """The prepared-text request contract (transport integrity only)."""

    def __init__(self, request_path: Path, checkpoint: Path):
        self.path = request_path
        self.raw = request_path.read_bytes()
        self.values = json.loads(self.raw)
        self._check_schema()
        self._bind_checkpoint(checkpoint)
        self.token_archive = request_path.parent / 'prepared_text.npz'
        self.token_batches = self._bind_token_archive()

    def _check_schema(self) -> None:
        if (self.values['schema'] != REQUEST_SCHEMA
                or len(self.values['ids']) != len(self.values['texts'])):
            raise ValueError('Invalid prepared text request')

    def _bind_checkpoint(self, checkpoint: Path) -> None:
        # Transport integrity only: semantic validation and cache decisions
        # are local.
        if checkpoint_hash(checkpoint) != self.values['metadata']['checkpoint_sha256']:
            raise ValueError('Uploaded checkpoint differs from local request')

    def _bind_token_archive(self) -> list:
        plan = self.values.get('prepared_text')
        if (not plan or hashlib.sha256(self.token_archive.read_bytes()).hexdigest()
                != plan['sha256']):
            raise ValueError('Locally prepared tokens required; missing or corrupt token archive')
        return plan['token_batches']


def _load_token_features():
    """Bind the token-feature loader (standalone bundle name or local core)."""
    try:
        from encoding_inputs import load_token_features
    except ImportError:
        from core.encoding_inputs import load_token_features
    return load_token_features


class BatchEncoder:
    """Prepared token batches -> L2-normalized float32 embedding rows."""

    def __init__(self, model, token_archive: Path, device: str):
        self._model = model
        self._token_archive = token_archive
        self._device = device

    @timed
    def run(self, token_batches: list):
        """One timed pass: every prepared batch, then the stacked chunks."""
        import numpy as np
        import torch
        load_token_features = _load_token_features()
        chunks = []
        total = len(token_batches)
        with np.load(self._token_archive, allow_pickle=False) as data, torch.no_grad():
            for n, batch in enumerate(
                _LOG.progress(token_batches, desc='prepared_encode_batches',
                              unit='batch', total=total), 1
            ):
                chunks.append(self._encode_batch(data, load_token_features, batch))
                _LOG.info(f'[embeddings/{self._device}] batch={n}/{total}')
        return np.concatenate(chunks)

    def _encode_batch(self, data, load_token_features, batch):
        """One prepared batch -> normalized float32 embedding chunk."""
        import numpy as np
        import torch
        features = load_token_features(data, batch, self._device)
        vector = self._model(features)['sentence_embedding']
        return torch.nn.functional.normalize(
            vector, p=2, dim=1).cpu().numpy().astype(np.float32)


def _load_tokenization_policy():
    try:
        from encoding_inputs import tokenization_policy
    except ImportError:
        from core.encoding_inputs import tokenization_policy
    return tokenization_policy


def main():
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA required; refusing CPU fallback')
    if args.output.exists():
        raise FileExistsError('Worker never reuses an existing output')
    request = EncodeRequest(args.request, args.checkpoint)
    _LOG.info(
        f'[embeddings/{args.device}] loading checkpoint; '
        f"prepared texts={len(request.values['texts']):,}")
    model = SentenceTransformer(str(args.checkpoint), device=args.device, local_files_only=True)
    model.eval()
    tokenization_policy = _load_tokenization_policy()
    plan = request.values.get('prepared_text')
    if tokenization_policy(model) != plan['tokenization']:
        raise ValueError('Worker tokenizer policy differs from local preparation')
    _LOG.info(f'[embeddings/{args.device}] encoding prepared batches; truncated=0')
    vectors = BatchEncoder(model, request.token_archive, args.device).run(request.token_batches)
    if len(vectors) != len(request.values['ids']):
        raise ValueError('Prepared token population differs from request IDs')
    metadata = {**request.values['metadata'],
                'request_sha256': hashlib.sha256(request.raw).hexdigest(),
                'embedding_dtype': 'float32'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as handle:
        np.savez_compressed(handle, ids=np.asarray(request.values['ids'], dtype=str),
                            embeddings=vectors, metadata=json.dumps(metadata, sort_keys=True))
    args.output.with_suffix('.sha256').write_text(
        hashlib.sha256(args.output.read_bytes()).hexdigest())
    _LOG.info(f'[embeddings/{args.device}] encoded shape={vectors.shape}')


if __name__ == '__main__':
    main()
