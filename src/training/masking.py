"""Masking augmentation for positive and labeled hard-negative pairs.

The original second_masking.py fed label=0.0 hard-negative pairs into
MultipleNegativesRankingLoss — MNRL IGNORES labels, so those pairs would be
trained as POSITIVES (pulling different products together). This module
implements label-preserving augmentation for both populations. MNRL callers
must continue to pass masked hard negatives through an explicit negative
channel; this module never changes labels or treats a negative as positive.

For a chosen fraction of positive pairs, the ANCHOR text gets random token
masking (each token replaced with the mask token at mask_prob). The masked
anchor is appended to the payload and paired with the ORIGINAL positive —
same pair semantics, noised anchor. Masked texts are new payload entries
carrying the anchor's barcode, so components/folds are unaffected.
"""

from __future__ import annotations

import random

import numpy as np

from core.common import training_cfg
from core.schemas import MaskingResult

# extent band SSOT — read once at import from config/training.yaml (masking:
# block, validated by MaskingSpec at load). No inline literals (owner Q27:
# the config, not the signature, declares the band).
_MASK_LO = float(training_cfg().masking.mask_lo)
_MASK_HI = float(training_cfg().masking.mask_hi)

MASK_TOKEN = "[MASK]"


def mask_text(
    text: str,
    mask_prob: float | None = None,
    rng: random.Random | None = None,
    lo: float | None = None,
    hi: float | None = None,
) -> tuple[str, float]:  # (masked_text, realized extent)
    """Randomly replace whitespace tokens with the mask token.

    mask_prob=None draws the extent per call from U(lo, hi)
    spec: masking extent is drawn from the configured mask_lo..mask_hi band
    The band is owned by config/training.yaml and is validated at load time.

    GUARANTEE (owner audit 2026-09-07): a masked copy must actually be a
    COPY — when the extent draw masks zero tokens (27.4% of short SKU
    titles at U(0.05,0.15); measured on the real payload), one random
    token is masked so no augmented row is an exact duplicate of its
    anchor. Empty/1-token texts return unchanged (nothing to mask).

    Returns (masked_text, extent) where extent = fraction of tokens
    actually masked (the REALIZED extent, not the draw) — high/low-extent
    effect tracking (owner directive 2026-09-07).
    """
    if rng is None:
        rng = random.Random()
    lo = _MASK_LO if lo is None else lo
    hi = _MASK_HI if hi is None else hi
    if mask_prob is None:
        mask_prob = lo + (hi - lo) * rng.random()
    toks = text.split()
    if len(toks) < 2:
        return text, 0.0
    out = [MASK_TOKEN if rng.random() < mask_prob else t for t in toks]
    if MASK_TOKEN not in out:
        out[rng.randrange(len(out))] = MASK_TOKEN
    extent = out.count(MASK_TOKEN) / len(out)
    return " ".join(out), extent


def augment_pairs(
    pos: np.ndarray,
    payload: list[str],
    row_bc: np.ndarray,
    *,
    frac: float,
    mask_prob: float | None = None,
    seed: int = 0,
    population: str = "positive",
) -> tuple[
    np.ndarray, list[str], np.ndarray, int, list[dict]
]:  # (pos', payload', row_bc', n_added, audit dicts)
    """Append masked-anchor copies of a fraction of labeled pair rows.

    mask_prob None (default): each masked copy draws its own extent from
    the config band U(mask_lo, mask_hi) — per-pair variation per the owner
    spec. A fixed float keeps the old uniform behavior.

    BOUNDARY CONTRACT (lib.schemas.MaskingResult): the return crosses into
    src/training/train + src/training/training — payload'/row_bc' stay length-locked and
    every pos' index is in range of the EXTENDED payload, or the call dies
    here with a named field error. Callers unpack the same 5-tuple as
    before (pos, payload, row_bc, n_added, audit-dicts).
    """
    audit: list[dict] = []
    if frac <= 0 or len(pos) == 0:
        res = MaskingResult(
            pos=pos, payload=list(payload), row_bc=np.asarray(row_bc),
            n_added=0, audit=[],
        )
        return res.pos, res.payload, res.row_bc, res.n_added, audit
    rng = random.Random(seed)
    n_mask = int(len(pos) * min(frac, 1.0))
    mask_idx = rng.sample(range(len(pos)), n_mask) if n_mask else []
    if not mask_idx:
        res = MaskingResult(
            pos=pos, payload=list(payload), row_bc=np.asarray(row_bc),
            n_added=0, audit=[],
        )
        return res.pos, res.payload, res.row_bc, res.n_added, audit
    new_payload = list(payload)
    new_bc = [str(x) for x in row_bc]
    extra = []
    base = len(payload)
    for i in mask_idx:
        a, b = int(pos[i][0]), int(pos[i][1])
        masked, extent = mask_text(payload[a], mask_prob, rng)
        new_payload.append(masked)
        new_bc.append(str(row_bc[a]))
        copy_idx = base + len(extra)
        extra.append((copy_idx, b))
        audit.append(
            {
                "anchor_payload_idx": a,
                "copy_payload_idx": copy_idx,
                "pair_payload_idx": b,
                "barcode": str(row_bc[a]),
                "realized_extent": round(extent, 4),
                "anchor_text": payload[a],
                "masked_text": masked,
                "population": population,
            }
        )
        # per-pair varied extent: next draw differs even for same anchor
    res = MaskingResult(
        pos=np.vstack([pos, np.array(extra, dtype=int)]),
        payload=new_payload,
        row_bc=np.array(new_bc),
        n_added=len(extra),
        audit=audit,
    )
    return res.pos, res.payload, res.row_bc, res.n_added, res.audit_dicts()


def augment_positives(
    pos: np.ndarray,
    payload: list[str],
    row_bc: np.ndarray,
    *,
    frac: float,
    mask_prob: float | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, list[str], np.ndarray, int, list[dict]]:
    """Append masked-anchor copies with positive-pair semantics."""
    return augment_pairs(
        pos, payload, row_bc,
        frac=frac, mask_prob=mask_prob, seed=seed, population="positive",
    )


def augment_hard_negatives(
    neg: np.ndarray,
    payload: list[str],
    row_bc: np.ndarray,
    *,
    frac: float,
    mask_prob: float | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, list[str], np.ndarray, int, list[dict]]:
    """Append masked-anchor copies while preserving label-0 semantics."""
    return augment_pairs(
        neg, payload, row_bc,
        frac=frac, mask_prob=mask_prob, seed=seed, population="hard_negative",
    )
