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

from core.common import load_config, training_cfg
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
    "sweetener_type": ("sweetener_type_",),
    "sweetening": ("sweetening_",),
    "sweetener": ("sweetener_diet_", "sweetener_"),
    "pulp": ("pulp_",),
    # Captured attribute evidence (pipeline.extract_all structured evidence
    # section): pack material type is the census's donor-capable SET_ENUM
    # (51,703 rows, 5 value-sets, 9.64% conflict — veto band) and juice
    # content is the donor-capable NUMERIC_BAND channel (63,117 rows, 27
    # canonical bands). The registry list grows; every unknown-field raise
    # (swap_structured_field / mask_targeted) fails loud exactly as before.
    "package_material": ("package_material_",),
    "juice_content": ("juice_content_",),
}


def field_of(token: str) -> str | None:
    """Structured field group of one payload token, or None for prose."""
    lowered = token.lower()
    for field, prefixes in _FIELD_PREFIXES.items():
        if lowered.startswith(prefixes):
            return field
    return None


def extend_augmented_features(features, payload, audit):
    """Append features in payload-index order, including symmetric copies.

    Masks retain source features. Value swaps and twins update the numeric
    field they changed, so the text and numeric loss channels agree.
    """
    from core.structured_features import vector

    features = np.asarray(features, dtype=np.float32)
    cfg = training_cfg().training.structured_features
    copies = {}
    for row in audit:
        pairs = [(row["copy_payload_idx"], row["anchor_payload_idx"])]
        if row.get("copy_pair_payload_idx") is not None:
            pairs.append((row["copy_pair_payload_idx"], row["pair_payload_idx"]))
        for destination, source in pairs:
            destination, source = int(destination), int(source)
            if destination in copies or not 0 <= source < len(features):
                raise ValueError("invalid or duplicate augmentation feature lineage")
            result = features[source].copy()
            if row["target_mode"] in {"swap_values", "counterfactual"}:
                for field, prefix, start in (("volume", "volume_ml_", 0), ("pack", "pack_qty_", 5)):
                    if field not in row["fields_hit"]:
                        continue
                    values = {
                        float(token[len(prefix):].replace("_", "."))
                        for token in payload[destination].split()
                        if token.startswith(prefix)
                    }
                    if not values or len(result) != 10:
                        raise ValueError("invalid numeric augmentation feature contract")
                    encoded = vector(
                        {field: values}, volume_scale_ml=cfg.volume_scale_ml,
                        pack_scale=cfg.pack_scale, max_set_size=cfg.max_set_size,
                    )
                    result[start:start + 5] = encoded[start:start + 5]
            copies[destination] = result
    if set(copies) != set(range(len(features), len(payload))):
        raise ValueError("augmentation feature lineage does not cover the payload suffix")
    if not copies:
        return features
    return np.vstack([features, *[copies[i][None, :] for i in sorted(copies)]])


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
        pos=np.vstack([pos, np.array(extra, dtype=int)]) if extra else np.asarray(pos),
        payload=new_payload,
        row_bc=np.array(new_bc),
        n_added=len(extra),
        audit=audit,
    )
    return res.pos, res.payload, res.row_bc, res.n_added, res.audit_dicts()


def augment_declaration_dropout(
    pairs: np.ndarray,
    payload: list[str],
    row_bc: np.ndarray,
    *,
    frac: float,
    seed: int = 0,
    pool_size: int | None = None,
    min_drop: int = 1,
    max_drop: int = 3,
    shared_value_counts: Counter[tuple[str, tuple[str, ...]]] | None = None,
    cap_base: int | None = None,
    max_value_share: float | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray, int, list[dict]]:
    """Remove structured groups from ONE side of a positive pair — the
    missing_both shape duplicate pairs really have.

    The duplicate census (2026-10-01) showed attribute cells differ in 100%
    of same-GTIN groups: each retailer declares a different partial key
    subset. This lane mints that exact shape: copy the anchor and delete
    min_drop..max_drop random declared groups (not just blank them — the
    tokens LEAVE, so the copy under-declares like a thinner retailer feed).
    No tokens are invented and none are replaced, so the copy still agrees
    with its counterpart everywhere both sides ever spoke; label stays 1 by
    construction (removing evidence cannot contradict identity). This is the
    lane that reaches the registry-only weak spots no token group serves
    (health claims, made from, sustainable sourcing...): their claims ride
    in prose, and prose survives pruning of the STRUCTURED tail.

    Footprint caps shared with the value lanes (per group: the pair's
    removed (group, token-signature) draw is bounded by max_value_share of
    the cap base, so one popular subset cannot dominate). Deterministic.
    """
    import math
    from collections import Counter

    if max_value_share is not None and not 0.0 < max_value_share <= 1.0:
        raise ValueError("max_value_share must be in (0, 1]")
    pool = int(pool_size) if pool_size is not None else len(pairs)
    pool = max(0, min(pool, len(pairs)))
    if frac <= 0 or pool == 0:
        res = MaskingResult(
            pos=pairs, payload=list(payload), row_bc=np.asarray(row_bc),
            n_added=0, audit=[],
        )
        return res.pos, res.payload, res.row_bc, res.n_added, []
    rng = random.Random(seed)
    n_pick = int(pool * min(frac, 1.0))
    picked = rng.sample(range(pool), n_pick) if n_pick else []
    if not picked:
        res = MaskingResult(
            pos=pairs, payload=list(payload), row_bc=np.asarray(row_bc),
            n_added=0, audit=[],
        )
        return res.pos, res.payload, res.row_bc, res.n_added, []
    _base = int(cap_base) if cap_base else n_pick
    value_cap = max(1, math.ceil(max_value_share * _base)) if max_value_share else None
    used_values = shared_value_counts if shared_value_counts is not None else Counter()
    for_seen: Counter[str] = Counter()
    out_pairs: list[tuple[int, int]] = []
    out_payload: list[str] = []
    out_bc: list[str] = []
    audit: list[dict] = []
    for i in picked:
        a, b = int(pairs[i][0]), int(pairs[i][1])
        surfaces = _field_surfaces(payload[a])
        if not surfaces:
            continue
        groups = sorted(surfaces)
        n_drop = (rng.randint(min_drop, max_drop) if max_drop >= min_drop else min_drop)
        n_drop = min(n_drop, len(groups))
        if n_drop <= 0:
            continue
        dropped = rng.sample(groups, n_drop)
        dropped_set = set(dropped)
        kept = [
            tok for tok in payload[a].split()
            if field_of(tok) is None or field_of(tok) not in dropped_set
        ]
        del_surfaces = {g: surfaces[g] for g in dropped}
        if value_cap is not None:
            signature_key = (tuple(dropped), tuple(sorted(
                t.lower() for g in dropped for t in del_surfaces[g])))
            if used_values[(signature_key, ())] >= value_cap:
                continue
            used_values[(signature_key, ())] += 1
        copy_text = " ".join(kept)
        if copy_text == payload[a] or copy_text == payload[b]:
            continue
        extent = round(len(del_surfaces and [
            t for g in dropped for t in del_surfaces[g]
        ]) / max(len(payload[a].split()), 1), 4)
        copy_idx = len(payload) + len(out_payload)
        out_payload.append(copy_text)
        out_bc.append(str(row_bc[a]))
        out_pairs.append((copy_idx, b))
        for_seen[tuple(dropped)] += 1
        audit.append({
            "anchor_payload_idx": a,
            "copy_payload_idx": copy_idx,
            "pair_payload_idx": b,
            "barcode": str(row_bc[a]),
            "realized_extent": extent,
            "configured_mask_lo": None,
            "configured_mask_hi": None,
            "mask_prob": None,
            "anchor_text": payload[a],
            "masked_text": copy_text,
            "population": "positive",
            "target_mode": "declaration_dropout",
            "fields_hit": dropped,
            "donor_anchor_payload_idx": None,
            "donor_pair_payload_idx": None,
            "copy_pair_payload_idx": None,
        })
    res = MaskingResult(
        pos=(np.vstack([pairs, np.array(out_pairs, dtype=int)])
             if out_pairs else np.asarray(pairs)),
        payload=[*payload, *out_payload],
        row_bc=np.array([*map(str, row_bc), *out_bc]),
        n_added=len(out_payload),
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


def _field_values_conflict(field: str, left: list[str], right: list[str]) -> bool:
    """Apply the shared attribute compatibility rules to structured tokens."""
    prefixes = _FIELD_PREFIXES[field]

    def values(tokens: list[str]) -> set[str]:
        result = set()
        for token in tokens:
            prefix = next(p for p in prefixes if token.startswith(p))
            result.add(token[len(prefix):].casefold())
        return result

    left_values = values(left)
    right_values = values(right)
    if not left_values or not right_values:
        return False
    if field == "volume":
        from core.critical_attributes import volumes_compatible

        def parse_volume(value: str) -> float:
            return float(value.replace("_", "."))

        cfg = training_cfg()
        # Read from the gate block (the SSOT both cuts now share) rather than
        # re-deriving the absolute cut from the veto lane's own block: two
        # declarations of one tolerance is how the two lanes drifted apart.
        absolute_tolerance = float(cfg.gate.vol_abs_tolerance)
        return not volumes_compatible(
            {parse_volume(value) for value in left_values},
            {parse_volume(value) for value in right_values},
            volume_relative_tolerance=float(cfg.gate.vol_tolerance),
            volume_absolute_tolerance_ml=absolute_tolerance,
        )
    if field == "flavor":
        from core.attribute_conflicts import flavor_overlap_metrics

        return flavor_overlap_metrics(left_values, right_values)[1] == 0.0
    from core.critical_attributes import categorical_conflict

    return categorical_conflict(field, {field: left_values}, {field: right_values})


def build_entity_cluster_map(
    positive_pairs_df, donor_pool_df,
    *,
    anchor_col: str = "anchor_id",
    pair_col: str = "pair_id",
    record_col: str = "record_id",
    barcode_col: str = "barcode",
) -> dict:
    """Deterministic cluster-ID map via transitive closure.

    Nodes are record IDs; edges come from ground-truth positive pairs plus
    shared non-null barcodes (chained within each barcode group, so N rows
    cost N-1 edges). Connected components become CLUSTER_xxxxxx IDs. Only
    connected records are mapped — isolated records are absent, and the
    caller must give them unique fallback keys (never a shared blank key).

    Determinism: components are enumerated in first-seen order over
    insertion-ordered nodes (positive-pair endpoints first, then donor
    rows), so identical inputs always yield identical IDs.
    """
    import networkx as nx

    graph = nx.Graph()
    for _, row in positive_pairs_df.iterrows():
        graph.add_edge(row[anchor_col], row[pair_col])
    pool = donor_pool_df[
        donor_pool_df[barcode_col].notna() & (donor_pool_df[barcode_col] != "")
    ]
    for _, group in pool.groupby(barcode_col, sort=True):
        node_ids = group[record_col].tolist()
        for index in range(len(node_ids) - 1):
            graph.add_edge(node_ids[index], node_ids[index + 1])
    cluster_map: dict = {}
    for cluster_idx, component in enumerate(nx.connected_components(graph)):
        cluster_id = f"CLUSTER_{cluster_idx:06d}"
        for record_id in component:
            cluster_map[record_id] = cluster_id
    return cluster_map


def check_cluster_sizes(
    cluster_map: dict,
    *,
    max_component_size: int,
    max_giant_ratio: float,
    population_size: int | None = None,
) -> dict[str, object]:
    """Circuit breaker over entity-cluster topology (fail loud, not silent).

    A single bad edge (shared placeholder barcode, feed corruption) merges
    two large clusters under union-find with no un-merge short of a full
    rebuild. This check runs at build time in data prep: components are
    tiny by construction (measured max 13 over 38,952 covered rows), so a
    giant component means catalog corruption, not a big product family.
    Raises ValueError naming the offender; returns the stats otherwise.
    """
    from collections import Counter

    if max_component_size < 2:
        raise ValueError("max_component_size must be >= 2")
    if not 0.0 < max_giant_ratio <= 1.0:
        raise ValueError("max_giant_ratio must be in (0, 1]")
    population_size = (
        len(cluster_map) if population_size is None else int(population_size)
    )
    if population_size < len(cluster_map):
        raise ValueError("population_size cannot be smaller than clustered coverage")
    sizes = Counter(cluster_map.values())
    if not sizes:
        return {"clusters": 0, "covered": 0, "max_size": 0, "giant_ratio": 0.0}
    biggest, biggest_size = sizes.most_common(1)[0]
    # Isolated records are still part of the population at risk of being
    # incorrectly merged. For tiny samples, the ratio has too little signal
    # to trip a breaker (the absolute component-size cap remains active).
    giant_ratio = biggest_size / max(population_size, 1)
    stats: dict[str, object] = {
        "clusters": len(sizes),
        "covered": len(cluster_map),
        "max_size": biggest_size,
        "max_cluster": biggest,
        "giant_ratio": giant_ratio,
    }
    if biggest_size > max_component_size:
        raise ValueError(
            "entity-cluster circuit breaker: component "
            f"{biggest} has {biggest_size} rows > max {max_component_size} "
            "(suspect shared placeholder barcode or feed corruption — "
            "audit before merging)"
        )
    min_population_for_ratio = int(1 / max_giant_ratio + 0.999999)
    if population_size >= min_population_for_ratio and giant_ratio > max_giant_ratio:
        raise ValueError(
            "entity-cluster circuit breaker: giant ratio "
            f"{giant_ratio:.4f} > {max_giant_ratio:.4f} "
            "(one component dominates the clustered rows — audit before merging)"
        )
    return stats


def normalize_entity_key(value: object, fallback: str) -> str:
    """Canonical entity key for donor-disjointness checks.

    GTINs fragment across feeds (UPC-12 vs EAN-13, zero padding,
    missing values): strip surrounding whitespace and leading zeros so
    length-variants of one GTIN share a key. Empty/missing values get the
    caller-supplied unique fallback (never the shared "" — one blank key
    would refuse every barcode-less donor at once).
    """
    text = str(value or "").strip()
    if not text:
        return fallback
    return text.lstrip("0") or "0"


def _resolve_entity_keys(
    row_bc: np.ndarray,
    entity_keys: list[str] | None,
    pairs: np.ndarray,
    pool: int,
) -> list[str] | None:
    """Entity keys for donor-disjointness guards, or None for barcodes.

    ``pool`` counts sampled pair rows, not payload rows. Validate every
    endpoint that can be selected so a sparse/high payload index cannot
    escape coverage checks.
    """
    if entity_keys is None:
        return None
    selected = np.asarray(pairs)[:pool]
    if selected.size:
        endpoints = selected.astype(int, copy=False).reshape(-1)
        if np.any(endpoints < 0):
            raise ValueError("selected pair endpoints must be non-negative")
        required = int(endpoints.max()) + 1
    else:
        required = 0
    if len(entity_keys) < required:
        raise ValueError(
            "entity keys must cover selected pair endpoints: "
            f"{len(entity_keys)} < required payload length {required}"
        )
    return list(entity_keys)


def _splice_field(text: str, field: str, donor_tokens: list[str]) -> str:
    """Replace ``field``'s whole value group in ``text`` with donor tokens.

    The group is removed wherever its tokens sit and the donor tokens are
    inserted once, at the first removed position. Donor tokens always come
    from another real payload row, so no value is ever invented.
    """
    out: list[str] = []
    inserted = False
    for tok in text.split():
        if field_of(tok) == field:
            if not inserted:
                out.extend(donor_tokens)
                inserted = True
        else:
            out.append(tok)
    return " ".join(out)


def augment_value_swaps(
    pairs: np.ndarray,
    payload: list[str],
    row_bc: np.ndarray,
    *,
    frac: float,
    seed: int = 0,
    population: str = "positive",
    pool_size: int | None = None,
    symmetric: bool = False,
    entity_keys: list[str] | None = None,
    max_field_share: float | None = None,
    max_value_share: float | None = None,
    shared_value_counts: Counter[tuple[str, tuple[str, ...]]] | None = None,
    cap_base: int | None = None,
    max_donor_overlap: float | None = None,
    field_quota_shares: dict[str, float] | None = None,
    row_retailer: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray, int, list[dict]]:
    """Append copies whose structured VALUE was transplanted from a donor pair.

    This is the value-swap lane (e.g. an anchor saying ``coconut`` is
    reworded to say ``lime`` with a real ``lime`` donor row): a genuine
    semantic change, not a surface reorder. Static and precomputed — the
    donor is chosen here, before training, from the same pair pool, and the
    audit names it. No invented tokens: every transplanted token already
    exists in the corpus payload.

    ``row_retailer`` (same length as payload, retailer string per row):
    donors come from a DIFFERENT retailer first (91.4% of real same-GTIN
    duplicates are cross-retailer — the phrasing gap a minted pair should
    train); after 10 strict tries the loop relaxes so rich lanes never
    mint nothing. Deterministic. ``field_quota_shares`` (field -> share
    of this lane's picks): the measured-conflict-rate slot allocation,
    binding like a per-field cap with the same soft fallback.

    Label safety is structural, and differs per population:

    * positives (``symmetric=True``): BOTH sides are rewritten, the anchor
      from the donor pair's anchor and the counterpart from the donor
      pair's counterpart, and only when the donor pair itself agrees on the
      field. The copy pair agrees exactly where the donor pair agrees, so a
      match stays a match — a synthetic flavor identity, self-consistent on
      both sides. Single-sided value swaps on positives would teach a false
      grounding (lime text = coconut product) and are refused by this flag.
    * hard negatives (``symmetric=False``): only the anchor is rewritten.
      The two endpoints are different products by canonical identity (same-
      canonical rows are excluded upstream), so the pair stays label 0 no
      matter what the text says — even when the transplant heals the very
      conflict the gate found, which is precisely the hardest negative.

    False-negative guards (both populations, applied before anything is
    emitted):

    * the donor pair must share NO entity key with the anchor pair, so a
      duplicate record of the same entity can never donate its own values.
      Keys are canonical-entity GTINs with zero-padding normalized away
      (UPC-12 vs EAN-13 length variants collide); unmapped rows carry
      unique per-row keys and never block each other;
    * the rewritten copy must not be byte-identical to the other side of
      the pair — an identical-text label-0 row would punish the model for
      a distinction absent from the text.

    Concentration caps (anti-dominance): ``max_field_share`` bounds any one
    field to a share of the picks (soft — falls back to an uncapped field
    rather than emitting nothing, so structural fields still flow when no
    semantic field is eligible); ``max_value_share`` bounds any one
    (field, value) transplant to a share of the picks (hard — an
    overused donor value is skipped, so no single string becomes a
    synthetic-generation artifact).

    Pairs with no eligible donor (all donors carry the same values) emit
    nothing. Returns the standard 5-tuple with target_mode="swap_values"
    audits; MaskingResult validates shapes.
    """
    import math

    if max_field_share is not None and not 0.0 < max_field_share <= 1.0:
        raise ValueError("max_field_share must be in (0, 1]")
    if max_value_share is not None and not 0.0 < max_value_share <= 1.0:
        raise ValueError("max_value_share must be in (0, 1]")
    if max_donor_overlap is not None and not 0.0 < max_donor_overlap <= 1.0:
        raise ValueError("max_donor_overlap must be in (0, 1]")
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
    _entity_list = _resolve_entity_keys(row_bc, entity_keys, pairs, pool)
    _barcodes = [str(x) for x in np.asarray(row_bc)]
    entities = _entity_list if _entity_list is not None else _barcodes
    from collections import Counter

    used_fields: Counter[str] = Counter()
    # Value counts may be shared across lanes (pos swaps, neg swaps, twins)
    # so the footprint cap binds the whole bundle, not one call. The caller
    # owns the object and passes the same one to every lane, in fixed order
    # — deterministic. A per-call Counter otherwise.
    used_values = (
        shared_value_counts if shared_value_counts is not None else Counter()
    )
    # Caps bind the caller's pick total (cap_base) when lanes share
    # counters, so one global budget covers the whole bundle;
    # otherwise they bind this call's own picks.
    _base = int(cap_base) if cap_base else n_pick
    field_cap = max(1, math.ceil(max_field_share * _base)) if max_field_share else None
    value_cap = max(1, math.ceil(max_value_share * _base)) if max_value_share else None
    quota_caps = (
        {f: max(1, math.ceil(s * n_pick)) for f, s in field_quota_shares.items() if s > 0}
        if field_quota_shares
        else None
    )
    _ret = np.asarray(row_retailer, dtype=object) if row_retailer is not None else None
    extra = []
    for i in picked:
        a, b = int(pairs[i][0]), int(pairs[i][1])
        anchor_fields = _field_surfaces(payload[a])
        anchor_entities = {entities[a], entities[b]}
        chosen: tuple[str, list[str], list[str] | None, int, int] | None = None
        for _attempt in range(14):
            _relaxed_retailer = _ret is None or _attempt >= 10
            j = rng.randrange(pool)
            if j == i:
                continue
            c, d = int(pairs[j][0]), int(pairs[j][1])
            if anchor_entities & {entities[c], entities[d]}:
                continue
            if not _relaxed_retailer and str(_ret[c]) == str(_ret[a]):
                # strict cross-retailer donor first: real duplicates are 91.4%
                # cross-seller, so a same-store donor teaches the wrong gap
                continue
            if max_donor_overlap is not None:
                anchor_toks = set(payload[a].split())
                donor_toks = set(payload[c].split())
                union = anchor_toks | donor_toks
                if union and len(anchor_toks & donor_toks) / len(union) >= max_donor_overlap:
                    # Near-identical donor from an unmapped row: probable
                    # same-entity relist the cluster map cannot see. Refuse.
                    continue
            donor_anchor = _field_surfaces(payload[c])
            donor_pair = _field_surfaces(payload[d])
            candidates = sorted(
                field
                for field in anchor_fields
                if field in donor_anchor
                and donor_anchor[field] != anchor_fields[field]
                and (not symmetric or (
                    field in donor_pair
                    and {t.lower() for t in donor_pair[field]}
                    == {t.lower() for t in donor_anchor[field]}
                ))
            )
            if not candidates:
                continue
            if quota_caps is not None:
                qunder = [f for f in candidates if f not in quota_caps or used_fields[f] < quota_caps[f]]
                candidates = qunder or candidates
            if field_cap is not None:
                under = [f for f in candidates if used_fields[f] < field_cap]
                candidates = under or candidates
            field = rng.choice(candidates)
            signature = (field, tuple(sorted(t.lower() for t in donor_anchor[field])))
            if value_cap is not None and used_values[signature] >= value_cap:
                continue
            chosen = (field, donor_anchor[field], donor_pair.get(field), c, d)
            break
        if chosen is None:
            continue
        field, donor_a_tokens, donor_b_tokens, c, d = chosen
        swapped_anchor = _splice_field(payload[a], field, donor_a_tokens)
        if swapped_anchor == payload[a]:
            continue
        anchor_toks = len(payload[a].split())
        extent = round(len(anchor_fields[field]) / max(anchor_toks, 1), 4)
        # Position-based indices: a symmetric audit appends TWO rows per
        # pair, so the pair count is not the row count. Index from the
        # payload length actually in hand.
        copy_idx = len(new_payload)
        new_payload.append(swapped_anchor)
        new_bc.append(str(row_bc[a]))
        if symmetric:
            assert donor_b_tokens is not None
            swapped_pair = _splice_field(payload[b], field, donor_b_tokens)
            if swapped_pair == payload[b] or swapped_pair == swapped_anchor:
                new_payload.pop()
                new_bc.pop()
                continue
            pair_copy_idx = len(new_payload)
            new_payload.append(swapped_pair)
            new_bc.append(str(row_bc[b]))
            extra.append((copy_idx, pair_copy_idx))
        else:
            pair_copy_idx = None
            if swapped_anchor == payload[b]:
                # The transplant erased every textual difference. A
                # byte-identical label-0 row would punish the model for a
                # distinction absent from the text — skip it.
                new_payload.pop()
                new_bc.pop()
                continue
            extra.append((copy_idx, b))
        used_fields[field] += 1
        used_values[(field, tuple(sorted(t.lower() for t in donor_a_tokens)))] += 1
        audit.append(
            {
                "anchor_payload_idx": a,
                "copy_payload_idx": copy_idx,
                "pair_payload_idx": b,
                "barcode": str(row_bc[a]),
                "realized_extent": extent,
                "configured_mask_lo": None,
                "configured_mask_hi": None,
                "mask_prob": None,
                "anchor_text": payload[a],
                "masked_text": swapped_anchor,
                "population": population,
                "target_mode": "swap_values",
                "fields_hit": [field],
                "donor_anchor_payload_idx": c,
                "donor_pair_payload_idx": d,
                "copy_pair_payload_idx": pair_copy_idx,
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


def mint_swap_counterpart_positives(
    swap_audit: list[dict],
    payload: list[str],
    row_bc: np.ndarray,
    pos: np.ndarray,
) -> tuple[list[tuple[int, int]], list[str], list[str], list[dict]]:
    """TIER 1(a): give every anchor-side value-swap copy a counterpart positive.

    A hard-negative swap transplants one field into the ANCHOR only
    (``symmetric=False``), so the copy stops agreeing with its own source
    positive. Pairing it with that unchanged positive would train a false
    match, so the MNRL triple builder omitted the row outright and the copy
    never reached a gradient.

    This replays the SAME transplant onto the source's own positive, exactly as
    the positive lane's symmetric swap rewrites both sides, so
    ``(copy, counterpart)`` is a genuine positive. No value is invented: the
    donor tokens come from the recorded donor payload row, so the counterpart
    only ever carries a value that exists elsewhere in the bundle.

    Returns the new positive pairs, their payload rows/barcodes, and audit
    entries whose ``copy_payload_idx`` is the NEW counterpart row and whose
    ``anchor_payload_idx`` is the source positive, so
    ``extend_augmented_features`` derives counterpart features from the right
    row without re-claiming the already-extended swap copy.
    """
    positives_by_anchor: dict[int, list[int]] = {}
    for anchor, positive in np.asarray(pos, dtype=int).reshape(-1, 2):
        positives_by_anchor.setdefault(int(anchor), []).append(int(positive))

    base = len(payload)
    new_pairs: list[tuple[int, int]] = []
    new_payload: list[str] = []
    new_bc: list[str] = []
    audit: list[dict] = []
    for row in swap_audit or []:
        if str(row.get("target_mode")) != "swap_values":
            continue
        fields = row.get("fields_hit") or []
        donor_i = row.get("donor_anchor_payload_idx")
        if donor_i is None or len(fields) != 1:
            continue
        source_i = int(row["anchor_payload_idx"])
        copy_i = int(row["copy_payload_idx"])
        if not (0 <= donor_i < len(payload) and copy_i < len(payload)):
            continue
        field = str(fields[0])
        donor_tokens = _field_surfaces(payload[int(donor_i)]).get(field)
        if not donor_tokens:
            continue
        candidates = positives_by_anchor.get(source_i) or []
        if not candidates:
            continue
        positive_i = int(candidates[0])
        if not 0 <= positive_i < len(payload):
            continue
        counterpart_text = _splice_field(payload[positive_i], field, donor_tokens)
        if counterpart_text == payload[positive_i]:
            # The source positive carries NO token for the transplanted field,
            # so the transplant cannot contradict it — it is silent where the
            # copy now speaks. The ORIGINAL blanket skip discarded these rows
            # for a contradiction that does not exist, and they are 493 of the
            # 2,709 real swap rows. Register the source positive against the
            # COPY anchor directly: both indices already exist, so no new
            # payload row and no feature lineage are involved.
            new_pairs.append((copy_i, positive_i))
            continue
        if counterpart_text == payload[copy_i]:
            # A counterpart equal to the swap copy would make the positive pair
            # self-referential. Dropped rather than minted.
            continue
        counterpart_idx = base + len(new_payload)
        new_payload.append(counterpart_text)
        new_bc.append(str(row_bc[positive_i]))
        new_pairs.append((copy_i, counterpart_idx))
        audit.append(
            {
                "anchor_payload_idx": positive_i,
                "copy_payload_idx": counterpart_idx,
                "pair_payload_idx": copy_i,
                "copy_pair_payload_idx": None,
                "barcode": str(row_bc[source_i]),
                "realized_extent": row.get("realized_extent"),
                "configured_mask_lo": None,
                "configured_mask_hi": None,
                "mask_prob": None,
                "anchor_text": payload[positive_i],
                "masked_text": counterpart_text,
                "population": "swap_counterpart",
                "target_mode": "swap_values",
                "fields_hit": [field],
                "donor_anchor_payload_idx": int(donor_i),
                "donor_pair_payload_idx": row.get("donor_pair_payload_idx"),
            }
        )
    return new_pairs, new_payload, new_bc, audit


def augment_counterfactual_twins(
    pairs: np.ndarray,
    payload: list[str],
    row_bc: np.ndarray,
    *,
    frac: float,
    seed: int = 0,
    pool_size: int | None = None,
    entity_keys: list[str] | None = None,
    max_field_share: float | None = None,
    max_value_share: float | None = None,
    shared_value_counts: Counter[tuple[str, tuple[str, ...]]] | None = None,
    cap_base: int | None = None,
    max_donor_overlap: float | None = None,
    field_quota_shares: dict[str, float] | None = None,
    row_retailer: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray, int, list[dict]]:
    """Mint minimal-flip negatives from positive pairs: (A1', A2) labeled 0.

    For a sampled positive pair ``(A1, A2)`` that AGREES on a structured
    field, the anchor side alone is rewritten with a real donor value for
    that field (``coconut`` -> ``lime``), breaking exactly one previously-
    agreed field and nothing else. The twin ``(A1', A2)`` is labeled 0 by
    construction: it differs from a verified match in one load-bearing
    attribute, so it cannot be the same product. The original ``(A1, A2)``
    stays label 1, and both rows train together — the loss must stop
    leaning on the 90% shared tokens (brand, size, pack) and look at the
    one token that changed.

    Only agreed fields flip: a field the pair already disagrees on is not
    match evidence, so flipping it is not minimal. Guards are shared with
    the value-swap lane — entity-disjoint donors, no invented tokens, the
    twin must differ from the pair side (an identical twin is not a flip),
    and the same field/value concentration caps. Static and precomputed:
    donors are chosen here, before training.

    The returned pair array holds the new ``(copy, pair-side)`` rows for the
    caller to append to its NEGATIVE pool (with a ``counterfactual``
    provenance label); the audit population is ``hard_negative`` so the
    diet gate counts them as augmented negative views.
    """
    import math

    if max_field_share is not None and not 0.0 < max_field_share <= 1.0:
        raise ValueError("max_field_share must be in (0, 1]")
    if max_value_share is not None and not 0.0 < max_value_share <= 1.0:
        raise ValueError("max_value_share must be in (0, 1]")
    if max_donor_overlap is not None and not 0.0 < max_donor_overlap <= 1.0:
        raise ValueError("max_donor_overlap must be in (0, 1]")
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
    _entity_list = _resolve_entity_keys(row_bc, entity_keys, pairs, pool)
    _barcodes = [str(x) for x in np.asarray(row_bc)]
    entities = _entity_list if _entity_list is not None else _barcodes
    from collections import Counter

    used_fields: Counter[str] = Counter()
    used_values = (
        shared_value_counts if shared_value_counts is not None else Counter()
    )
    # Caps bind the caller's pick total (cap_base) when lanes share
    # counters, so one global budget covers the whole bundle;
    # otherwise they bind this call's own picks.
    _base = int(cap_base) if cap_base else n_pick
    field_cap = max(1, math.ceil(max_field_share * _base)) if max_field_share else None
    value_cap = max(1, math.ceil(max_value_share * _base)) if max_value_share else None
    quota_caps = (
        {f: max(1, math.ceil(s * n_pick)) for f, s in field_quota_shares.items() if s > 0}
        if field_quota_shares
        else None
    )
    _ret = np.asarray(row_retailer, dtype=object) if row_retailer is not None else None
    extra = []
    for i in picked:
        a, b = int(pairs[i][0]), int(pairs[i][1])
        anchor_fields = _field_surfaces(payload[a])
        pair_fields = _field_surfaces(payload[b])
        agreed = sorted(
            field
            for field in anchor_fields
            if field in pair_fields
            and {t.lower() for t in anchor_fields[field]}
            == {t.lower() for t in pair_fields[field]}
        )
        if not agreed:
            continue
        anchor_entities = {entities[a], entities[b]}
        chosen: tuple[str, list[str], int, int] | None = None
        for _attempt in range(14):
            _relaxed_retailer = _ret is None or _attempt >= 10
            j = rng.randrange(pool)
            if j == i:
                continue
            c, d = int(pairs[j][0]), int(pairs[j][1])
            if anchor_entities & {entities[c], entities[d]}:
                continue
            if not _relaxed_retailer and str(_ret[c]) == str(_ret[a]):
                # strict cross-retailer donor first: real duplicates are 91.4%
                # cross-seller, so a same-store donor teaches the wrong gap
                continue
            if max_donor_overlap is not None:
                anchor_toks = set(payload[a].split())
                donor_toks = set(payload[c].split())
                union = anchor_toks | donor_toks
                if union and len(anchor_toks & donor_toks) / len(union) >= max_donor_overlap:
                    # Near-identical donor from an unmapped row: probable
                    # same-entity relist the cluster map cannot see. Refuse.
                    continue
            donor_anchor = _field_surfaces(payload[c])
            candidates = sorted(
                field
                for field in agreed
                if field in donor_anchor
                and _field_values_conflict(
                    field, donor_anchor[field], anchor_fields[field]
                )
            )
            if not candidates:
                continue
            if quota_caps is not None:
                qunder = [f for f in candidates if f not in quota_caps or used_fields[f] < quota_caps[f]]
                candidates = qunder or candidates
            if field_cap is not None:
                under = [f for f in candidates if used_fields[f] < field_cap]
                candidates = under or candidates
            field = rng.choice(candidates)
            signature = (field, tuple(sorted(t.lower() for t in donor_anchor[field])))
            if value_cap is not None and used_values[signature] >= value_cap:
                continue
            chosen = (field, donor_anchor[field], c, d)
            break
        if chosen is None:
            continue
        field, donor_tokens, c, d = chosen
        flipped = _splice_field(payload[a], field, donor_tokens)
        if flipped == payload[a] or flipped == payload[b]:
            continue
        anchor_toks = len(payload[a].split())
        extent = round(len(anchor_fields[field]) / max(anchor_toks, 1), 4)
        copy_idx = len(new_payload)
        new_payload.append(flipped)
        new_bc.append(str(row_bc[a]))
        extra.append((copy_idx, b))
        used_fields[field] += 1
        used_values[(field, tuple(sorted(t.lower() for t in donor_tokens)))] += 1
        audit.append(
            {
                "anchor_payload_idx": a,
                "copy_payload_idx": copy_idx,
                "pair_payload_idx": b,
                "barcode": str(row_bc[a]),
                "realized_extent": extent,
                "configured_mask_lo": None,
                "configured_mask_hi": None,
                "mask_prob": None,
                "anchor_text": payload[a],
                "masked_text": flipped,
                "population": "hard_negative",
                "target_mode": "counterfactual",
                "fields_hit": [field],
                "donor_anchor_payload_idx": c,
                "donor_pair_payload_idx": d,
                "copy_pair_payload_idx": None,
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
