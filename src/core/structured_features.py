"""Structured product attributes shared by the training and scoring lanes.

The catalog already carries these values in the canonical records.  This
module gives them one representation at both boundaries of the model:

* stable text tokens (``volume_ml_500`` / ``pack_qty_2`` / ``package_type_can``), and
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
        if text in {"set()", "frozenset()"}:
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


def _as_string_set(value: object, *, kind: str) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (set, frozenset, list, tuple, np.ndarray)):
        values = value
    else:
        text = str(value).strip()
        if not text:
            return set()
        if text in {"set()", "frozenset()"}:
            return set()
        try:
            values = ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"invalid structured {kind} set: {value!r}") from exc
    if not isinstance(values, (set, frozenset, list, tuple, np.ndarray)):
        raise ValueError(f"structured {kind} must be a sequence: {value!r}")
    return {
        normalized
        for item in values
        if (normalized := str(item).strip().casefold())
    }


def info_from_sets(
    volume: object,
    pack: object,
    package_type: object = None,
    *,
    flavor: object = None,
    carbonation: object = None,
    sweetener: object = None,
    sweetener_type: object = None,
    sweetening: object = None,
    pulp: object = None,
    package_material: object = None,
    juice_content: object = None,
) -> dict[str, set[float] | set[str]]:
    """Normalize a SKU/canonical record into the shared set representation.

    ``package_material`` and ``juice_content`` are the captured attribute
    evidence channels (pipeline.extract_all's structured evidence section):
    material rides the EXISTING package_materials/package_material_set
    convention (both payload sides), juice content rides the source row's
    attribute cell (the canonical record stores no such column, so its side
    reads an empty set from absent records — the same omit-when-unobserved
    treatment every categorical field gets).
    """
    return {
        "volume": _as_set(volume, kind="volume"),
        "pack": _as_set(pack, kind="pack"),
        "package_type": _as_string_set(package_type, kind="package_type"),
        "flavor": _as_string_set(flavor, kind="flavor"),
        "carbonation": _as_string_set(carbonation, kind="carbonation"),
        "sweetener": _as_string_set(sweetener, kind="sweetener"),
        "sweetener_type": _as_string_set(sweetener_type, kind="sweetener_type"),
        "sweetening": _as_string_set(sweetening, kind="sweetening"),
        "pulp": _as_string_set(pulp, kind="pulp"),
        "package_material": _as_string_set(package_material, kind="package_material"),
        "juice_content": _as_string_set(juice_content, kind="juice_content"),
    }


def sku_info(
    title: object, attributes: object, description: object = ""
) -> dict[str, set[float] | set[str]]:
    """Parse one source SKU using the pipeline's existing extractor."""
    from pipeline import extract_all

    desc = "" if description is None or (isinstance(description, float) and np.isnan(description)) else str(description)
    extracted = extract_all(str(title), str(attributes), desc)
    flags = _as_string_set(
        extracted.get("attribute_consistency_flags"),
        kind="attribute_consistency_flags",
    )
    volume = (
        {float(extracted.get("volume_ml") or 0.0)}
        if "ambiguous_volume" not in flags
        else set()
    )
    pack_qty = extracted.get("pack_qty")
    # The pipeline's parser uses pack_qty=1 with zero confidence as its
    # explicit "no pack count observed" sentinel.  The model-side structured
    # channel must still emit the same singleton token as canonical records;
    # gate code intentionally keeps using confidence-aware unknown semantics.
    pack = (
        {float(pack_qty)}
        if pack_qty is not None and float(extracted.get("pack_confidence") or 0.0) > 0.0
        else {1.0}
    )
    evidence = extracted.get("attribute_universe_evidence") or {}
    return info_from_sets(
        volume,
        pack,
        extracted.get("package_types"),
        flavor=extracted.get("flavor_set"),
        carbonation=extracted.get("carbonation_set"),
        sweetener=extracted.get("sweetener_set"),
        sweetener_type=extracted.get("sweetener_type_set"),
        sweetening=extracted.get("sweetening_set"),
        pulp=extracted.get("pulp_set"),
        package_material=extracted.get("package_materials"),
        juice_content=evidence.get("juice content"),
    )


def canonical_info(
    record: Mapping[str, object],
) -> dict[str, set[float] | set[str]]:
    from core.critical_attributes import extract_critical_claims

    inferred = extract_critical_claims(
        record.get("canonical", ""), record.get("mode_flavor", "")
    )

    def evidence(field: str, inferred_field: str) -> object:
        value = record.get(field)
        return value if value is not None and str(value).strip() else inferred[inferred_field]

    # Preserve out-of-range measurements in the canonical record for
    # diagnostics, while leaving them out of model evidence.
    flags = _as_string_set(
        record.get("attribute_consistency_flags"), kind="attribute_consistency_flags"
    )
    volume = set() if "ambiguous_volume" in flags else record.get("volume_set")

    return info_from_sets(
        volume,
        record.get("pack_set"),
        record.get("package_type_set"),
        flavor=evidence("flavor_set", "flavor"),
        carbonation=evidence("carbonation_set", "carbonation"),
        sweetener=evidence("sweetener_set", "sweetener"),
        sweetener_type=record.get("sweetener_type_set"),
        sweetening=record.get("sweetening_set"),
        pulp=evidence("pulp_set", "pulp"),
        # package_material_set is already a canonical column (line 1109 of the
        # canonical builder unions it); juice evidence has no canonical
        # column, so records without it stay as empty sets (omit when
        # unobserved — the symmetry contract shared by package_type/flavor/…
        # above).
        package_material=record.get("package_material_set"),
        juice_content=record.get("juice_content_bands"),
    )


def _number_token(prefix: str, value: float) -> str:
    rendered = f"{value:g}"
    return f"{prefix}{rendered.replace('.', '_')}"


def text_tokens(info: Mapping[str, object]) -> list[str]:
    """Return deterministic field-marked tokens appended to model text.

    Existing value tokens are retained (for example ``volume_ml_500``), while
    stable ``[FIELD_*]`` markers make field boundaries explicit.  The encoder
    still receives one ordinary string and the numeric feature vector keeps
    its existing dimensionality.

    The captured evidence groups (``PACK_MATERIAL`` / ``JUICE_CONTENT_BAND``
    — pipeline.extract_all's structured evidence section) sit AFTER every
    legacy group, in that legacy order, so the byte contract of the existing
    token stream is a PREFIX of the new one: an info map without the two new
    keys renders byte-identically to the pre-capture composition (pinned in
    tests/test_universe_capture_wiring.py), and a populated key only appends
    its block.
    """
    volumes = sorted(_as_set(info.get("volume"), kind="volume"))
    packs = sorted(_as_set(info.get("pack"), kind="pack"))
    package_types = sorted(
        _as_string_set(info.get("package_type"), kind="package_type")
    )
    package_materials = sorted(
        _as_string_set(info.get("package_material"), kind="package_material")
    )
    juice_bands = sorted(_as_string_set(info.get("juice_content"), kind="juice_content"))
    categorical = {
        "FLAVOR": sorted(_as_string_set(info.get("flavor"), kind="flavor")),
        "CARBONATION": sorted(
            _as_string_set(info.get("carbonation"), kind="carbonation")
        ),
        "SWEETENER_DIET": sorted(
            _as_string_set(info.get("sweetener"), kind="sweetener")
        ),
        "PULP": sorted(_as_string_set(info.get("pulp"), kind="pulp")),
        "SWEETENER_TYPE": sorted(_as_string_set(info.get("sweetener_type"), kind="sweetener_type")),
        "SWEETENING": sorted(_as_string_set(info.get("sweetening"), kind="sweetening")),
    }
    groups: list[tuple[str, list[str]]] = [
        ("VOLUME", _number_tokens("volume_ml_", volumes)),
        ("PACK_SIZE", _number_tokens("pack_qty_", packs)),
        (
            "PACKAGE_TYPE",
            [f"package_type_{value.replace(' ', '_')}" for value in package_types],
        ),
        *[
            (name, [f"{name.casefold()}_{value.replace(' ', '_')}" for value in values])
            for name, values in categorical.items()
        ],
        # Captured evidence appends LAST so every existing group's byte
        # stream and relative order are unchanged (value values keep the
        # published space->underscore convention; a census value such as
        # "paper / carton" renders as the token package_material_paper_/_carton).
        (
            "PACK_MATERIAL",
            [
                f"package_material_{value.replace(' ', '_')}"
                for value in package_materials
            ],
        ),
        (
            "JUICE_CONTENT_BAND",
            [
                f"juice_content_{value.replace(' ', '_')}"
                for value in juice_bands
            ],
        ),
    ]
    tokens: list[str] = []
    for field, values in groups:
        if values:
            tokens.extend((f"[FIELD_{field}]", *values))
    return tokens


def _number_tokens(prefix: str, values: Sequence[float]) -> list[str]:
    return [_number_token(prefix, float(value)) for value in values]


def append_text(text: str, info: Mapping[str, object], *, enabled: bool = True) -> str:
    """Append normalized structured values without changing the base text."""
    if not enabled:
        return str(text)
    tokens = text_tokens(info)
    return " ".join([str(text).strip(), *tokens]).strip()


def symmetric_info(
    info: Mapping[str, object], *, implicit_pack_qty: float
) -> dict[str, set[float] | set[str]]:
    """Apply the universal implicit-default rule to one side of a pair.

    An attribute that was NOT observed must be represented identically on both
    sides, or the two sides of the same product differ for a reason that has
    nothing to do with the product.

    Audited attribute by attribute against ``sku_info`` / ``canonical_info``:

    * ``volume`` — both sides already omit when unobserved: already symmetric.
    * ``pack`` — the ONLY one-sided implicit default: ``sku_info`` emits the
      ``{1.0}`` "no pack count observed" sentinel while ``canonical_info``
      passes an empty set through. This emits ``implicit_pack_qty`` whenever
      the pack set is empty, on whichever side is being normalized.
    * ``package_type`` / ``flavor`` / ``carbonation`` / ``sweetener`` /
      ``pulp`` — both sides already omit when unobserved (an empty record
      returns empty sets from both extractors), so no implicit default is
      added. They can still disagree as *evidence* for a pair that is not the
      same product; that is information, not a default.

    Idempotent: a pack set that is already non-empty is returned unchanged.
    """
    if implicit_pack_qty <= 0:
        raise ValueError("implicit_pack_qty must be positive")
    normalized = dict(info)
    if not _as_set(info.get("pack"), kind="pack"):
        normalized["pack"] = {float(implicit_pack_qty)}
    return normalized


def vector(info: Mapping[str, object], *, volume_scale_ml: float, pack_scale: float, max_set_size: int) -> list[float]:
    """Encode volume_set/pack_set as a fixed-size, scale-normalized vector.

    Presence, min, max, span and cardinality are retained for each set.  The
    vector is intentionally small and deterministic so it can be fused with
    any MiniLM-sized embedding without another trainable model. Package type
    uses the text-token channel because an unordered categorical vocabulary
    has no meaningful scalar geometry.
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
