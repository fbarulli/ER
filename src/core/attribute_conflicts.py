"""Shared attribute parsing and conflict classification.

Training-time mining, pair dumps, and post-run reports must use the same
structured canonical attributes.  Keeping this logic here prevents a report
from disagreeing with the population that was actually mined.
"""

from __future__ import annotations

import ast
import re
import unicodedata
from collections.abc import Mapping

from core.unit_canonicalization import canonical_pack_count, canonical_volume_ml
from core.critical_attributes import (
    CRITICAL_ATTRIBUTE_DIMENSIONS,
    FLAVOR_ALIASES,
    FLAVOR_LEXICON,
    categorical_conflict,
    extract_critical_claims,
    volumes_compatible,
)


_FLAVOR_NOISE = frozenset(
    {
        "flavor",
        "flavored",
        "flavour",
        "flavoured",
        "profile",
        "taste",
    }
)
def normalized_flavor_tokens(value: object) -> frozenset[str]:
    """Return stable flavor evidence tokens for overlap-based comparison.

    Separators commonly found in catalog exports (commas, underscores,
    slashes, ampersands, and hyphens) are deliberately equivalent.  Generic
    flavor-label words are removed so they cannot create false overlap.
    """
    if isinstance(value, (set, frozenset, list, tuple)):
        return frozenset().union(*(normalized_flavor_tokens(item) for item in value))
    text = unicodedata.normalize("NFKD", str(value or "").casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    tokens = re.findall(r"[a-z0-9]+", text)
    return frozenset(
        FLAVOR_ALIASES.get(token, token)
        for token in tokens
        if token not in _FLAVOR_NOISE
    )


def flavor_overlap_metrics(left: object, right: object) -> tuple[float, float]:
    """Return ``(Jaccard, overlap coefficient)`` for flavor evidence.

    Empty evidence is explicit rather than treated as disagreement: both
    values are zero and callers decide whether missingness needs a separate
    confidence mask.
    """
    left_tokens = normalized_flavor_tokens(left)
    right_tokens = normalized_flavor_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0, 0.0
    intersection = len(left_tokens & right_tokens)
    return (
        intersection / len(left_tokens | right_tokens),
        intersection / min(len(left_tokens), len(right_tokens)),
    )


def _flavor_evidence(*values: object) -> str:
    """Collect all recognized flavor mentions, not just the first regex hit."""
    tokens: set[str] = set()
    for value in values:
        tokens.update(normalized_flavor_tokens(value) & FLAVOR_LEXICON)
    return " ".join(sorted(tokens))


def _value_set(value: object, *, kind: str) -> set[object]:
    """Parse a canonical CSV set field without trusting CSV dtype inference."""
    if value is None:
        return set()
    text = str(value).strip()
    if not text:
        return set()
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"invalid canonical attribute set: {value!r}") from exc
    if isinstance(parsed, (list, tuple, set)):
        canonicalizer = canonical_volume_ml if kind == "volume" else canonical_pack_count
        normalized = set()
        for item in parsed:
            try:
                normalized.add(canonicalizer(item))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid canonical {kind} value: {item!r}"
                ) from exc
        return normalized
    raise ValueError(f"canonical attribute set is not a sequence: {value!r}")


def _string_value_set(value: object, *, kind: str) -> set[str]:
    """Parse a canonical string-set field into normalized values."""
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set, frozenset)):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            return set()
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"invalid canonical {kind} set: {value!r}") from exc
    if not isinstance(parsed, (list, tuple, set, frozenset)):
        raise ValueError(f"canonical {kind} set is not a sequence: {value!r}")
    return {
        normalized
        for item in parsed
        if (normalized := str(item).strip().casefold())
    }


def canonical_attribute_info(record: Mapping[str, object]) -> dict[str, object]:
    """Return the structured attributes used by the canonical record lane."""
    inferred = extract_critical_claims(
        record.get("canonical", ""), record.get("mode_flavor", "")
    )

    def evidence(field: str, inferred_field: str) -> set[str]:
        value = record.get(field)
        if value is None or not str(value).strip():
            return set(inferred[inferred_field])
        return _string_value_set(value, kind=field)

    flavor_set = evidence("flavor_set", "flavor")
    return {
        "volume": _value_set(record.get("volume_set"), kind="volume"),
        "pack": _value_set(record.get("pack_set"), kind="pack"),
        "package_type": _string_value_set(
            record.get("package_type_set"), kind="package_type"
        ),
        "flavor": " ".join(sorted(flavor_set)),
        "flavor_set": flavor_set,
        "carbonation": evidence("carbonation_set", "carbonation"),
        "sweetener": evidence("sweetener_set", "sweetener"),
        "pulp": evidence("pulp_set", "pulp"),
    }


def sku_attribute_info(title: object, attributes: object) -> dict[str, object]:
    """Extract SKU-side attributes using the same parser as the data lane."""
    from pipeline import extract_all

    extracted = extract_all(str(title), str(attributes))
    volume_ml = extracted.get("volume_ml")
    # Zero is the extractor's sentinel for "no volume mention".  It must
    # remain unknown here; turning it into {0.0} makes every known canonical
    # volume look like a real conflict.
    try:
        volume = (
            {canonical_volume_ml(volume_ml)}
            if volume_ml is not None and float(volume_ml) > 0
            else set()
        )
    except (TypeError, ValueError):
        volume = set()
    pack_qty = extracted.get("pack_qty")
    pack_confidence = extracted.get("pack_confidence")
    try:
        pack = (
            {canonical_pack_count(pack_qty)}
            if pack_qty is not None
            and int(pack_qty) >= 1
            and float(pack_confidence or 0.0) > 0.0
            else set()
        )
    except (TypeError, ValueError):
        pack = set()
    flavor_set = set(extracted.get("flavor_set") or set())
    return {
        "volume": volume,
        "pack": pack,
        "package_type": {
            str(value).strip().casefold()
            for value in extracted.get("package_types") or []
            if str(value).strip()
        },
        "flavor": " ".join(sorted(flavor_set)),
        "flavor_set": flavor_set,
        "carbonation": set(extracted.get("carbonation_set") or set()),
        "sweetener": set(extracted.get("sweetener_set") or set()),
        "pulp": set(extracted.get("pulp_set") or set()),
    }


def critical_attribute_evaluation(
    left: Mapping[str, object],
    right: Mapping[str, object],
    *,
    volume_relative_tolerance: float = 0.0,
    volume_absolute_tolerance_ml: float = 0.0,
) -> dict[str, list[str]]:
    """Classify every shared critical dimension as agree/conflict/unknown."""
    conflicts: list[str] = []
    unknown: list[str] = []
    agreements: list[str] = []
    for dimension in CRITICAL_ATTRIBUTE_DIMENSIONS:
        left_value = (
            left.get("flavor_set") or normalized_flavor_tokens(left.get("flavor"))
            if dimension == "flavor"
            else left.get(dimension)
        )
        right_value = (
            right.get("flavor_set") or normalized_flavor_tokens(right.get("flavor"))
            if dimension == "flavor"
            else right.get(dimension)
        )
        left_set = set(left_value or set())
        right_set = set(right_value or set())
        if not left_set or not right_set:
            unknown.append(dimension)
            continue
        if dimension == "volume":
            compatible = any(
                abs(float(a) - float(b))
                <= max(
                    float(volume_absolute_tolerance_ml),
                    float(volume_relative_tolerance) * max(abs(float(a)), abs(float(b))),
                )
                for a in left_set
                for b in right_set
            )
            conflict = not compatible
        elif dimension == "flavor":
            conflict = flavor_overlap_metrics(left_set, right_set)[1] == 0.0
        else:
            conflict = categorical_conflict(dimension, {dimension: left_set}, {dimension: right_set})
        (conflicts if conflict else agreements).append(dimension)
    return {"conflicts": conflicts, "unknown": unknown, "agreements": agreements}


def strict_attribute_gate(
    left: Mapping[str, object],
    right: Mapping[str, object],
    *,
    volume_relative_tolerance: float = 0.0,
    volume_absolute_tolerance_ml: float = 0.0,
) -> bool:
    """Return ``True`` only when every critical dimension is explicit AND agrees.

    SSOT (audit 2026-09-15): this used to be a second public function named
    ``pack_gate`` with a different signature from ``pipeline.pack_gate`` and
    the OPPOSITE answer on identical input — the pipeline gate treats missing
    evidence as unknown/compatible while this one requires full evidence. Two
    functions under one name made "does the pack gate pass?" depend on which
    module asked. The training-label gate keeps the name ``pack_gate``; this
    stricter full-evidence predicate is for callers that must NOT auto-merge
    on partial evidence, and returns ``False`` for unknown as well as for
    conflict. Callers that need to tell those two cases apart use
    :func:`critical_attribute_evaluation` directly.
    """
    result = critical_attribute_evaluation(
        left,
        right,
        volume_relative_tolerance=volume_relative_tolerance,
        volume_absolute_tolerance_ml=volume_absolute_tolerance_ml,
    )
    return not result["conflicts"] and not result["unknown"]


def attribute_conflict_types(
    left: Mapping[str, object],
    right: Mapping[str, object],
    *,
    volume_relative_tolerance: float = 0.0,
    volume_absolute_tolerance_ml: float = 0.0,
) -> list[str]:
    """Classify conflicts using the canonical miner's disjoint-set rules.

    Volume now uses the SHARED tolerance predicate instead of exact set
    intersection, so a pair whose volumes sit inside the gate's
    ``vol_tolerance`` is no longer reported as a conflict by the miner that
    feeds training labels. The default 0.0 keeps exact-match callers (the
    unit canonicalization tests pin that contract) unchanged.
    """
    conflicts: list[str] = []
    if not volumes_compatible(
        left.get("volume"),
        right.get("volume"),
        volume_relative_tolerance=volume_relative_tolerance,
        volume_absolute_tolerance_ml=volume_absolute_tolerance_ml,
    ):
        conflicts.append("volume")
    if left["pack"] and right["pack"] and not (
        set(left["pack"]) & set(right["pack"])
    ):
        conflicts.append("pack")
    left_package_types = set(left.get("package_type") or set())
    right_package_types = set(right.get("package_type") or set())
    if (
        left_package_types
        and right_package_types
        and not (left_package_types & right_package_types)
    ):
        conflicts.append("package_type")
    evaluation = critical_attribute_evaluation(
        left,
        right,
        volume_relative_tolerance=volume_relative_tolerance,
        volume_absolute_tolerance_ml=volume_absolute_tolerance_ml,
    )
    for dimension in ("flavor", "carbonation", "sweetener", "pulp"):
        if dimension in evaluation["conflicts"]:
            conflicts.append(dimension)
    return conflicts


def conflict_columns(
    left: Mapping[str, object], right: Mapping[str, object]
) -> dict[str, object]:
    """Return the report columns for a pair of parsed attribute records."""
    conflicts = attribute_conflict_types(left, right)
    return {
        "volume_conflict": int("volume" in conflicts),
        "pack_conflict": int("pack" in conflicts),
        "package_type_conflict": int("package_type" in conflicts),
        "flavor_conflict": int("flavor" in conflicts),
        "carbonation_conflict": int("carbonation" in conflicts),
        "sweetener_conflict": int("sweetener" in conflicts),
        "pulp_conflict": int("pulp" in conflicts),
        "attribute_conflict_type": "+".join(conflicts) if conflicts else "none",
    }


__all__ = [
    "attribute_conflict_types",
    "canonical_attribute_info",
    "conflict_columns",
    "critical_attribute_evaluation",
    "flavor_overlap_metrics",
    "normalized_flavor_tokens",
    "strict_attribute_gate",
    "sku_attribute_info",
]
