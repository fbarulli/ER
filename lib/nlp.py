"""encode_corpus — inlined from the monorepo training.nlp for the standalone
TRAIN_GPU folder (no external package imports). Loads one bi-encoder,
encodes the payload, L2-normalizes, returns (embeddings, seconds)."""

from __future__ import annotations

import time

import numpy as np
import torch


def encode_corpus(
    model_id: str,
    payload: list[str],
    *,
    device: str = "cpu",
    batch_size: int,
    max_seq_length: int,
    cache_dir: str | None = None,
) -> tuple[np.ndarray, float]:
    """Encode a corpus once. When cache_dir is given, reuse the cached
    embeddings keyed by (model, payload hash) — same convention as the
    repo lane (embeddings_cache/).

    `prompt` param REMOVED (audit 2026-09-09): it was accepted but never
    used in the body — a silent no-op knob (callers passing it got no
    prompt and no error).

    AUDIT FIX (round 2 F12, round 3): batch_size / max_seq_length are now
    REQUIRED keyword-only params. The old `= 256` / `= 128` defaults
    duplicated training.batch_size_embed / max_seq_length — unreachable
    fallback literals (every caller passes runtime() values), and a
    future caller could silently get the default instead of the SSOT.
    """
    from sentence_transformers import SentenceTransformer

    if cache_dir:
        import hashlib
        from pathlib import Path

        key = hashlib.md5(
            (model_id + "\x00" + "\x00".join(payload)).encode("utf-8")
        ).hexdigest()[:16]
        cdir = Path(cache_dir)
        cdir.mkdir(parents=True, exist_ok=True)
        cpath = cdir / f"{key}.npy"
        if cpath.exists():
            emb = np.load(cpath)
            if emb.shape[0] == len(payload):
                return emb, 0.0
            # size mismatch: stale cache entry, re-encode below

    model = SentenceTransformer(
        model_id, device=device, model_kwargs={"torch_dtype": torch.float32}
    )
    model.max_seq_length = max_seq_length
    t0 = time.perf_counter()
    emb = model.encode(
        payload,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    secs = time.perf_counter() - t0
    if cache_dir:
        np.save(cpath, emb)
    return emb, secs


# AUDIT FIX (round 2 F19, round 3): _cosine was a byte-for-byte duplicate
# of lib.common.pair_similarity. lib.common is the SSOT module, so
# pair_similarity is canonical; this alias keeps the historical
# `from lib.nlp import _cosine` import surface working with ONE
# implementation behind it (zero churn for any importer).
from lib.common import pair_similarity as _cosine

__all__ = ["_cosine", "encode_corpus"]
