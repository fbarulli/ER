"""Masking augmentation (second_masking, corrected for MNRL semantics).

The original second_masking.py fed label=0.0 hard-negative pairs into
MultipleNegativesRankingLoss — MNRL IGNORES labels, so those pairs would be
trained as POSITIVES (pulling different products together). This module
implements the corrected behavior: masking augments POSITIVES only.

For a chosen fraction of positive pairs, the ANCHOR text gets random token
masking (each token replaced with the mask token at mask_prob). The masked
anchor is appended to the payload and paired with the ORIGINAL positive —
same pair semantics, noised anchor. Masked texts are new payload entries
carrying the anchor's barcode, so components/folds are unaffected.
"""

from __future__ import annotations

import random

import numpy as np

MASK_TOKEN = "[MASK]"


def mask_text(
    text: str,
    mask_prob: float | None = None,
    rng: random.Random | None = None,
    lo: float = 0.05,
    hi: float = 0.15,
) -> str:
    """Randomly replace whitespace tokens with the mask token.

    mask_prob=None draws the extent per call from U(lo, hi)
    spec: masking done to different extents varying from 5-15%.
    """
    if rng is None:
        rng = random.Random()
    if mask_prob is None:
        mask_prob = lo + (hi - lo) * rng.random()
    toks = text.split()
    return " ".join(MASK_TOKEN if rng.random() < mask_prob else t for t in toks)


def augment_positives(
    pos: np.ndarray,
    payload: list[str],
    row_bc: np.ndarray,
    *,
    frac: float,
    mask_prob: float | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, list[str], np.ndarray, int]:
    """Append masked-anchor copies of a fraction of positive pairs.

    mask_prob None (default): each masked copy draws its own extent from
    U(0.05, 0.15) — per-pair variation per the owner spec. A fixed float
    keeps the old uniform behavior.

    Returns (pos', payload', row_bc', n_added). No-op when frac <= 0.
    """
    if frac <= 0 or len(pos) == 0:
        return pos, payload, row_bc, 0
    rng = random.Random(seed)
    n_mask = int(len(pos) * min(frac, 1.0))
    mask_idx = rng.sample(range(len(pos)), n_mask) if n_mask else []
    if not mask_idx:
        return pos, payload, row_bc, 0
    new_payload = list(payload)
    new_bc = [str(x) for x in row_bc]
    extra = []
    base = len(payload)
    for i in mask_idx:
        a, b = int(pos[i][0]), int(pos[i][1])
        new_payload.append(mask_text(payload[a], mask_prob, rng))
        new_bc.append(str(row_bc[a]))
        extra.append((base + len(extra), b))
        # per-pair varied extent: next draw differs even for same anchor
    return (
        np.vstack([pos, np.array(extra, dtype=int)]),
        new_payload,
        np.array(new_bc),
        len(extra),
    )
