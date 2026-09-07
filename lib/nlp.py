"""encode_corpus — inlined from broadway.training.nlp for the standalone
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
    batch_size: int = 256,
    max_seq_length: int = 128,
    cache_dir: str | None = None,
    prompt: str | None = None,
) -> tuple[np.ndarray, float]:
    """Encode a corpus once. When cache_dir is given, reuse the cached
    embeddings keyed by (model, payload hash) — same convention as the
    repo lane (embeddings_cache/)."""
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


def _cosine(emb: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Cosine similarity for index-aligned pair arrays (rows of a matrix)."""
    return (emb[pairs[:, 0]] * emb[pairs[:, 1]]).sum(axis=1)
