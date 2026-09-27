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

# Structured-tail field groups (src/core/structured_features.text_tokens).
# The [FIELD_*] markers are configuration-dependent (the shipped 'cleaned'
# composition omits them), but the VALUE tokens below are always present, so
# targeted masking keys on value prefixes, never on markers.
_FIELD_PREFIXES: dict[str, tuple[str, ...]] = {
    "volume": ("volume_ml_",),
    "pack": ("pack_qty_",),
    "package_type": ("package_type_",),
    "flavor": ("flavor_",),
    "carbonation": ("carbonation_",),
    "sweetener": ("sweetener_diet_", "sweetener_"),
    "pulp": ("pulp_",),
}


def field_of(token: str) -> str | None:
    """Structured field group of one payload token, or None for prose."""
    lowered = token.lower()
    for field, prefixes in _FIELD_PREFIXES.items():
        if lowered.startswith(prefixes):
            return field
    return None


def swap_structured_field(
    text: str,
    donor_text: str,
    *,
    field: str,
    rng: random.Random | None = None,
) -> tuple[str, bool]:
    """Replace ``field``'s value tokens in ``text`` with the donor's.

    The counterfactual probe: for a volume-conflict pair (anchor 500 vs
    counterpart 250), swapping the anchor's volume tokens to the donor's
    simulates agreement — a model that decides on volume should score the
    swapped anchor HIGHER against the counterpart. Returns (swapped_text,
    swapped); swapped is False (text unchanged) when either side lacks the
    field or the values already agree — callers must not count those as
    counterfactual evidence. Unknown fields raise.
    """
    if field not in _FIELD_PREFIXES:
        raise ValueError(f"unknown swap field: {field} (known: {sorted(_FIELD_PREFIXES)})")
    mine = [t for t in text.split() if field_of(t) == field]
    theirs = [t for t in donor_text.split() if field_of(t) == field]
    if not mine or not theirs or mine == theirs:
        return text, False
    donor = theirs  # donor values replace the full group, preserving count shape
    out, used = [], 0
    for tok in text.split():
        if field_of(tok) == field:
            out.append(donor[used % len(donor)])
            used += 1
        else:
            out.append(tok)
    return " ".join(out), True


def mask_targeted(
    text: str,
    *,
    fields: list[str] | tuple[str, ...],
    background_prob: float = 0.05,
    rng: random.Random | None = None,
) -> tuple[str, float, list[str]]:
    """Drop-mask the structured tokens of the requested field groups.

    Random masking mostly hits filler prose, teaching the model to ignore
    masking instead of attribute evidence. Targeted masking drops every
    value token of ``fields`` (e.g. all ``volume_ml_*``), plus a light
    random background elsewhere so copies stay true copies and general
    robustness is retained. Returns (masked_text, realized extent,
    fields_hit) — fields_hit names the groups actually masked, which is
    the coverage signal the diet manifest gates on. Unknown field names
    raise (fail loud, never silently mask nothing). Texts with no target
    token fall back to one random mask with fields_hit == [].
    """
    if rng is None:
        rng = random.Random()
    unknown = [f for f in fields if f not in _FIELD_PREFIXES]
    if unknown:
        raise ValueError(f"unknown mask target fields: {unknown} (known: {sorted(_FIELD_PREFIXES)})")
    wanted = set(fields)
    toks = text.split()
    if len(toks) < 2:
        return text, 0.0, []
    hit: set[str] = set()
    out: list[str] = []
    for tok in toks:
        group = field_of(tok)
        if group is not None and group in wanted:
            out.append(MASK_TOKEN)
            hit.add(group)
        elif rng.random() < background_prob:
            out.append(MASK_TOKEN)
        else:
            out.append(tok)
    if MASK_TOKEN not in out:
        out[rng.randrange(len(out))] = MASK_TOKEN
    extent = out.count(MASK_TOKEN) / len(out)
    return " ".join(out), extent, sorted(hit)


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
    lo: float | None = None,
    hi: float | None = None,
    target_fields: list[str] | tuple[str, ...] | None = None,
    background_prob: float = 0.05,
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
    effective_lo = _MASK_LO if lo is None else float(lo)
    effective_hi = _MASK_HI if hi is None else float(hi)
    for i in mask_idx:
        a, b = int(pos[i][0]), int(pos[i][1])
        if target_fields is None:
            masked, extent = mask_text(payload[a], mask_prob, rng, lo=lo, hi=hi)
            fields_hit: list[str] = []
            target_mode = "random"
        else:
            masked, extent, fields_hit = mask_targeted(
                payload[a], fields=list(target_fields),
                background_prob=background_prob, rng=rng,
            )
            target_mode = "targeted"
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
                "configured_mask_lo": effective_lo,
                "configured_mask_hi": effective_hi,
                "mask_prob": mask_prob,
                "anchor_text": payload[a],
                "masked_text": masked,
                "population": population,
                "target_mode": target_mode,
                "fields_hit": fields_hit,
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


def _field_surfaces(text: str) -> dict[str, list[str]]:
    """Ordered surface tokens per structured field group in one text."""
    surfaces: dict[str, list[str]] = {}
    for tok in text.split():
        group = field_of(tok)
        if group is not None:
            surfaces.setdefault(group, []).append(tok)
    return surfaces


def augment_swapped_agreed(
    pairs: np.ndarray,
    payload: list[str],
    row_bc: np.ndarray,
    *,
    frac: float,
    seed: int = 0,
    population: str = "positive",
    pool_size: int | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray, int, list[dict]]:
    """Append counterpart-surface copies for pairs agreeing on a field.

    For a sampled pair, every structured field whose VALUE sets agree
    (case-insensitive) between anchor and counterpart — but whose surface
    order differs — is rewritten in the anchor copy with the counterpart's
    surface form. Labels are preserved by construction: agreed values stay
    agreed, disagreed fields are never touched (a conflict pair keeps its
    conflict; a true pair keeps its match). Pairs with no swappable field
    emit nothing. Returns the standard 5-tuple with
    target_mode="swap_agreed" audits; MaskingResult validates shapes.
    """
    audit: list[dict] = []
    pool = int(pool_size) if pool_size is not None else len(pairs)
    pool = max(0, min(pool, len(pairs)))
    if frac <= 0 or pool == 0:
        res = MaskingResult(
            pos=pairs, payload=list(payload), row_bc=np.asarray(row_bc),
            n_added=0, audit=[],
        )
        return res.pos, res.payload, res.row_bc, res.n_added, audit
    rng = random.Random(seed)
    n_pick = int(pool * min(frac, 1.0))
    picked = rng.sample(range(pool), n_pick) if n_pick else []
    if not picked:
        res = MaskingResult(
            pos=pairs, payload=list(payload), row_bc=np.asarray(row_bc),
            n_added=0, audit=[],
        )
        return res.pos, res.payload, res.row_bc, res.n_added, audit
    new_payload = list(payload)
    new_bc = [str(x) for x in row_bc]
    extra = []
    base = len(payload)
    for i in picked:
        a, b = int(pairs[i][0]), int(pairs[i][1])
        anchor_fields = _field_surfaces(payload[a])
        other_fields = _field_surfaces(payload[b])
        swappable = sorted(
            field
            for field in anchor_fields
            if field in other_fields
            and {t.lower() for t in anchor_fields[field]}
            == {t.lower() for t in other_fields[field]}
            and anchor_fields[field] != other_fields[field]
        )
        if not swappable:
            continue
        donor = {field: other_fields[field] for field in swappable}
        out = []
        for tok in payload[a].split():
            group = field_of(tok)
            if group in donor and donor[group]:
                out.append(donor[group].pop(0))
            else:
                out.append(tok)
        swapped = " ".join(out)
        if swapped == payload[a]:
            continue
        new_payload.append(swapped)
        new_bc.append(str(row_bc[a]))
        copy_idx = base + len(extra)
        extra.append((copy_idx, b))
        audit.append(
            {
                "anchor_payload_idx": a,
                "copy_payload_idx": copy_idx,
                "pair_payload_idx": b,
                "barcode": str(row_bc[a]),
                "realized_extent": 0.0,
                "configured_mask_lo": None,
                "configured_mask_hi": None,
                "mask_prob": None,
                "anchor_text": payload[a],
                "masked_text": swapped,
                "population": population,
                "target_mode": "swap_agreed",
                "fields_hit": swappable,
            }
        )
    res = MaskingResult(
        pos=np.vstack([pairs, np.array(extra, dtype=int)]) if extra else np.asarray(pairs),
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
    lo: float | None = None,
    hi: float | None = None,
    target_fields: list[str] | tuple[str, ...] | None = None,
    background_prob: float = 0.05,
) -> tuple[np.ndarray, list[str], np.ndarray, int, list[dict]]:
    """Append masked-anchor copies while preserving label-0 semantics."""
    mask_cfg = training_cfg().masking
    return augment_pairs(
        neg, payload, row_bc,
        frac=frac,
        mask_prob=mask_prob,
        seed=seed,
        population="hard_negative",
        lo=mask_cfg.hard_negative_mask_lo if lo is None else lo,
        hi=mask_cfg.hard_negative_mask_hi if hi is None else hi,
        target_fields=target_fields,
        background_prob=background_prob,
    )
