"""Structured volume/pack features shared by the training and scoring lanes.

The catalog already carries these values in the canonical records.  This
module gives them one representation at both boundaries of the model:

* stable text tokens (``volume_ml_500`` / ``pack_qty_2``), and
* a small numeric vector fused with the encoder output before similarity or
  contrastive loss is calculated.

The values are deliberately derived from the existing structured sets; this
module does not re-infer labels or introduce a second attribute parser.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence

import numpy as np

from core.unit_canonicalization import canonical_pack_count, canonical_volume_ml


def _as_set(value: object, *, kind: str) -> set[float]:
    if value is None:
        return set()
    if isinstance(value, (set, list, tuple, np.ndarray)):
        values = value
    else:
        text = str(value).strip()
        if not text:
            return set()
        try:
            values = ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"invalid structured attribute set: {value!r}") from exc
    if not isinstance(values, (set, list, tuple, np.ndarray)):
        raise ValueError(f"structured attribute must be a sequence: {value!r}")
    canonicalizer = canonical_volume_ml if kind == "volume" else canonical_pack_count
    normalized: set[float] = set()
    for item in values:
        try:
            if float(item) <= 0:
                # Zero is the pipeline's explicit unknown sentinel.
                continue
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid structured {kind} value: {item!r}") from exc
        try:
            normalized.add(float(canonicalizer(item)))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid structured {kind} value: {item!r}") from exc
    return normalized


def info_from_sets(volume: object, pack: object) -> dict[str, set[float]]:
    """Normalize a SKU/canonical record into the shared set representation."""
    return {
        "volume": _as_set(volume, kind="volume"),
        "pack": _as_set(pack, kind="pack"),
    }


def sku_info(title: object, attributes: object) -> dict[str, set[float]]:
    """Parse one source SKU using the pipeline's existing extractor."""
    from pipeline import extract_all

    extracted = extract_all(str(title), str(attributes))
    volume = {float(extracted.get("volume_ml") or 0.0)}
    pack_qty = extracted.get("pack_qty")
    pack = (
        {float(pack_qty)}
        if pack_qty is not None and float(extracted.get("pack_confidence") or 0.0) > 0.0
        else set()
    )
    return info_from_sets(volume, pack)


def canonical_info(record: Mapping[str, object]) -> dict[str, set[float]]:
    return info_from_sets(record.get("volume_set"), record.get("pack_set"))


def _number_token(prefix: str, value: float) -> str:
    rendered = f"{value:g}"
    return f"{prefix}{rendered.replace('.', '_')}"


def text_tokens(info: Mapping[str, object]) -> list[str]:
    """Return deterministic, tokenizer-safe-ish normalized attribute tokens."""
    volumes = sorted(_as_set(info.get("volume"), kind="volume"))
    packs = sorted(_as_set(info.get("pack"), kind="pack"))
    return [*_number_tokens("volume_ml_", volumes), *_number_tokens("pack_qty_", packs)]


def _number_tokens(prefix: str, values: Sequence[float]) -> list[str]:
    return [_number_token(prefix, float(value)) for value in values]


def append_text(text: str, info: Mapping[str, object], *, enabled: bool = True) -> str:
    """Append normalized structured values without changing the base text."""
    if not enabled:
        return str(text)
    tokens = text_tokens(info)
    return " ".join([str(text).strip(), *tokens]).strip()


def vector(info: Mapping[str, object], *, volume_scale_ml: float, pack_scale: float, max_set_size: int) -> list[float]:
    """Encode volume_set/pack_set as a fixed-size, scale-normalized vector.

    Presence, min, max, span and cardinality are retained for each set.  The
    vector is intentionally small and deterministic so it can be fused with
    any MiniLM-sized embedding without another trainable model.
    """
    if volume_scale_ml <= 0 or pack_scale <= 0 or max_set_size <= 0:
        raise ValueError("structured feature scales and max_set_size must be positive")

    def block(values: set[float], scale: float) -> list[float]:
        if not values:
            return [0.0] * 5
        ordered = sorted(values)
        lo, hi = ordered[0], ordered[-1]
        return [
            1.0,
            float(np.log1p(lo) / np.log1p(scale)),
            float(np.log1p(hi) / np.log1p(scale)),
            float(np.log1p(max(0.0, hi - lo)) / np.log1p(scale)),
            float(min(len(ordered), max_set_size) / max_set_size),
        ]

    return block(_as_set(info.get("volume"), kind="volume"), volume_scale_ml) + block(
        _as_set(info.get("pack"), kind="pack"), pack_scale
    )


def fuse_numpy(embeddings: np.ndarray, features: np.ndarray, weight: float) -> np.ndarray:
    """Fuse structured features into normalized embeddings for scoring."""
    emb = np.asarray(embeddings, dtype=np.float32)
    feat = np.asarray(features, dtype=np.float32)
    if len(emb) != len(feat):
        raise ValueError(f"embedding/structured feature length mismatch: {len(emb)} != {len(feat)}")
    if weight <= 0 or feat.shape[1] == 0:
        return emb
    feat_norm = feat / np.maximum(np.linalg.norm(feat, axis=1, keepdims=True), 1e-12)
    fused = np.concatenate([emb, feat_norm * float(weight)], axis=1)
    return fused / np.maximum(np.linalg.norm(fused, axis=1, keepdims=True), 1e-12)


def fuse_torch(embeddings, features, weight: float):
    """Torch equivalent used inside the contrastive loss."""
    import torch.nn.functional as F

    if weight <= 0 or features.shape[-1] == 0:
        return embeddings
    emb = F.normalize(embeddings, p=2, dim=-1)
    feat = F.normalize(features.to(device=emb.device, dtype=emb.dtype), p=2, dim=-1)
    return F.normalize(torch_cat([emb, feat * float(weight)], dim=-1), p=2, dim=-1)


def torch_cat(values, dim: int):
    import torch

    return torch.cat(values, dim=dim)
