"""Full raw-dimension evidence alongside the shared product identity authority.

This is the only parser/evaluator of the full attribute-key registry. Existing
specialized extractors remain adapters for title and canonical evidence.
All raw fields are compared, but feed differences are not automatic identity
links or vetoes. Exact IDs and established structured conflicts retain their
separate authority in core.sku_identity.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field
import yaml

from core.text import normalized_attribute_text, unicode_casefold
from core.critical_attributes import volumes_compatible, categorical_conflict


class AttributeRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["categorical", "numeric", "interval"]
    unit: str | None = None
    aliases: dict[str, str] = Field(default_factory=dict)


class DimensionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1]
    columns: dict[str, Literal["listing_identifier", "identity_identifier", "offer_context",
                              "descriptor_text", "attribute_container"]]
    attributes: dict[str, AttributeRule]


@lru_cache(maxsize=1)
def dimension_policy() -> DimensionPolicy:
    from core.common import TRAIN_ROOT
    path = TRAIN_ROOT / "config" / "identity_dimensions.yaml"
    return DimensionPolicy.model_validate(yaml.safe_load(path.read_text()))


@dataclass(frozen=True)
class DimensionEvidence:
    attributes: Mapping[str, frozenset[str]]
    columns: Mapping[str, str]
    unclassified_keys: tuple[str, ...]
    malformed_parts: tuple[str, ...]
    context: Mapping | None = None


@lru_cache(maxsize=8)
def _attribute_registry(keys: tuple[str, ...]) -> dict[str, str]:
    """normalized-attribute-key -> declared-name, built once per policy.

    Rebuilding this map per row re-normalized all 37 declared keys for every
    row (measured 179.6 us of row_dimensions' 283.3 us per call).
    """
    return {normalized_attribute_text(k): k for k in keys}


@lru_cache(maxsize=8)
def _declared_names(keys: tuple[str, ...]) -> frozenset[str]:
    """The policy's declared attribute names, as the comparison set they are."""
    return frozenset(keys)


_WHITESPACE_RE = re.compile(r"\s+")


def row_dimensions(row: Mapping[str, object], *, policy: DimensionPolicy | None = None) -> DimensionEvidence:
    policy = policy or dimension_policy()
    registry = _attribute_registry(tuple(policy.attributes))
    attributes: dict[str, set[str]] = {}
    unknown, malformed = set(), []
    for part in str(row.get("attribute", "") or "").split(";"):
        if not part.strip():
            continue
        if ":" not in part:
            malformed.append(part.strip())
            continue
        key, value = part.split(":", 1)
        normalized_key = normalized_attribute_text(key)
        if not normalized_key:
            malformed.append(part.strip())
            continue
        name = registry.get(normalized_key, normalized_key)
        if name not in policy.attributes:
            unknown.add(name)
        rule = policy.attributes.get(name)
        for item in value.split(","):
            normalized = _WHITESPACE_RE.sub(" ", unicode_casefold(item)).strip()
            if not normalized:
                continue
            normalized = rule.aliases.get(normalized, normalized) if rule else normalized
            attributes.setdefault(name, set()).add(normalized)
    columns = {str(k): str(v or "").strip() for k, v in row.items() if k != "attribute"}
    from core.product_context import resolve_context
    frozen_attributes = {k: frozenset(v) for k, v in attributes.items()}
    return DimensionEvidence(frozen_attributes, columns, tuple(sorted(unknown)), tuple(malformed),
                             resolve_context(row, frozen_attributes))


# One compiled reader per declared unit family (unknown units fall back to the
# suffix-less form, exactly like the dict lookup this replaces).
_INTERVAL_RES = {
    unit: re.compile(r"\s*(\d+(?:\.\d+)?)\s*(?:[-–]\s*(\d+(?:\.\d+)?))?\s*" + suffix)
    for unit, suffix in (("percent", r"%?"), ("mg", r"(?:mg)?"), ("ml", r"(?:ml)?"),
                         ("count", ""), ("unspecified", ""))
}


def _interval(value: str, unit: str | None) -> tuple[float, float] | None:
    match = _INTERVAL_RES.get(unit, _INTERVAL_RES["unspecified"]).fullmatch(value)
    if not match:
        return None
    lo = float(match[1])
    hi = float(match[2]) if match[2] else lo
    return (lo, hi) if lo <= hi else None


def evaluate_dimensions(left: DimensionEvidence, right: DimensionEvidence,
                        *, policy: DimensionPolicy | None = None) -> dict[str, dict]:
    """Compare every registered dimension, preserving unknown/overlap states.

    A partial overlap is compatible evidence, not proof of equal formulations.
    Raw non-overlap is reported as a difference requiring review.
    """
    policy = policy or dimension_policy()
    report = {}
    for name in sorted(_declared_names(tuple(policy.attributes)) | set(left.attributes) | set(right.attributes)):
        a, b = left.attributes.get(name, frozenset()), right.attributes.get(name, frozenset())
        rule = policy.attributes.get(name)
        parse_warning = False
        if not a or not b:
            status = "unknown"
        elif rule and rule.kind in {"numeric", "interval"}:
            intervals_a = [_interval(v, rule.unit) for v in a]
            intervals_b = [_interval(v, rule.unit) for v in b]
            parse_warning = any(v is None or (rule.kind == "numeric" and v[0] != v[1])
                                for v in intervals_a + intervals_b)
            if parse_warning:
                status = "unparsed"
            elif a == b:
                status = "equal"
            elif rule.kind == "numeric":
                status = "overlap" if volumes_compatible(
                    {v[0] for v in intervals_a}, {v[0] for v in intervals_b},
                    volume_relative_tolerance=.05 if rule.unit == "ml" else 0.,
                ) else "different"
            else:
                status = "overlap" if any(max(x[0], y[0]) <= min(x[1], y[1])
                    for x in intervals_a for y in intervals_b) else "different"
        elif a == b:
            status = "equal"
        else:
            # Do not apply sugar-claim semantics to raw Sweetener ingredients.
            # Ingredient identity (sugar vs cane sugar) is a separate raw enum.
            status = "different" if categorical_conflict(name, {name: a}, {name: b}) else "overlap"
        report[name] = {"status": status, "left": sorted(a), "right": sorted(b),
                        "review": status in {"different", "unparsed"},
                        "classified": name in policy.attributes, "parse_warning": parse_warning}
    return report


def evaluate_columns(left: DimensionEvidence, right: DimensionEvidence,
                     *, policy: DimensionPolicy | None = None) -> dict[str, dict]:
    policy = policy or dimension_policy()
    out = {}
    for name in sorted(_declared_names(tuple(policy.columns)) | set(left.columns) | set(right.columns)):
        a, b = left.columns.get(name, ""), right.columns.get(name, "")
        role = policy.columns.get(name, "unclassified")
        out[name] = {"role": role, "status": "unknown" if not a or not b else
                     "equal" if normalized_attribute_text(a) == normalized_attribute_text(b) else "different"}
    return out


def evaluate_rows(left: Mapping[str, object], right: Mapping[str, object]) -> dict:
    """Full identity evidence; use established identity authority once per pair."""
    from core.sku_identity import evaluate_sku_identity, row_identity
    return evaluate_sku_identity(row_identity(left), row_identity(right))
