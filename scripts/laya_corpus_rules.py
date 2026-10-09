"""scripts/laya_corpus_rules.py — the corpus's documented scoring rules.

Every label the corpus emits is read off ONE of two things the builder already
has: the standardized attribute string (package evidence) or the six-field
side literals `compose_side` emits. No rule invents a token, and unknown
evidence is never guessed (it is `unknown`/`false`/absent).

Classes:
  * `PackageStateRules` — the `package_state` rule + package-quantity reads.
  * `PairLabelRules`    — the per-field / per-pair agreement labels.
  * `GateReasonRules`   — free-form gate_reason -> its bounded family.
"""
from __future__ import annotations

import re

from scripts.laya_corpus_composer import PAIR_FIELDS

NUMERIC_VALUE_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)?$")
PACK_COUNT_KEYS = frozenset({
    "pack", "pack size", "pack count", "pack quantity", "packaging",
    "units", "number of items", "item count",
})
MULTIPACK_RE = re.compile(
    r"\b[0-9]+\s*[xX]\s*[0-9]+(?:[.,][0-9]+)?\s*(?:ml|l|cl|dl|g|kg|oz)\b")
PACKAGE_STATE_RULE = (
    "package_state=true iff the standardized state carries a measured unit "
    "volume (a 'Volume:' field with a finite numeric value) OR an explicit "
    "numeric pack count (a numeric value under a pack-quantity key: "
    "Pack/Pack Size/Pack Count/Pack Quantity/Packaging/Units/Number of "
    "Items, or an 'NxM<unit>' multipack token); a 'Pack Type:' value alone "
    "is a package form, not a quantity, and does NOT qualify."
)

FIELD_SAME_QIDS = tuple(f"field_same:{field}" for field in PAIR_FIELDS)

GATE_REASON_FAMILIES: dict[str, str] = {
    "Pack blocker": "pack_blocker",
    "Critical attribute mismatch": "critical_attribute",
    "Package material mismatch": "package_material",
    "Contradictory source attribute evidence": "contradictory_evidence",
    "Declared product identity differs or is incomplete":
        "declared_identity_incomplete",
    "Missing flavor evidence with differing supporting attributes":
        "missing_flavor_evidence",
    "Low raw volume confidence": "low_volume_confidence",
    "Low raw pack confidence": "low_pack_confidence",
    "Known critical attributes compatible": "compatible",
}


class PackageStateRules:
    """The package_state rule and the package-quantity evidence it reads."""

    @staticmethod
    def _field(attribute: str, name: str) -> str | None:
        for part in str(attribute).split(";"):
            if ":" in part:
                key, value = part.split(":", 1)
                if key.strip().lower() == name:
                    return value.strip()
        return None

    @staticmethod
    def package_state(attribute: str) -> bool:
        """The package_state rule above, over one standardized attribute string."""
        for part in str(attribute).split(";"):
            if ":" not in part:
                continue
            key, value = part.split(":", 1)
            key = key.strip().lower()
            value = value.strip()
            if key == "volume" and NUMERIC_VALUE_RE.match(value):
                return True
            if key in PACK_COUNT_KEYS and NUMERIC_VALUE_RE.match(value):
                return True
        return bool(MULTIPACK_RE.search(str(attribute)))

    @staticmethod
    def _pack_type_only(attribute: str) -> bool:
        """Diagnostic: carries a Pack Type but no qualifying evidence."""
        return (not PackageStateRules.package_state(attribute)
                and bool(PackageStateRules._field(attribute, "pack type")))

    @staticmethod
    def _package_signature(attribute: str) -> tuple:
        """(volume, pack-count, multipack) evidence read off the attribute.

        Reuses the same keys the `package_state` rule reads (Volume plus the
        PACK_COUNT_KEYS / MULTIPACK_RE vocabulary); an unmeasured field is
        simply absent, never zero.
        """
        parts: list[tuple[str, str]] = []
        for part in str(attribute).split(";"):
            if ":" not in part:
                continue
            key, value = part.split(":", 1)
            key, value = key.strip().lower(), value.strip()
            if key == "volume" and NUMERIC_VALUE_RE.match(value):
                parts.append(("volume", value))
            elif key in PACK_COUNT_KEYS and NUMERIC_VALUE_RE.match(value):
                parts.append(("pack", value))
        match = MULTIPACK_RE.search(str(attribute))
        if match:
            parts.append(("multipack",
                          match.group(0).lower().replace(" ", "")))
        return tuple(sorted(parts))

    @staticmethod
    def pack_volume_equal(attr_one: str, attr_two: str) -> str:
        """true iff both sides carry package-quantity evidence and it agrees."""
        signature_one = PackageStateRules._package_signature(attr_one)
        signature_two = PackageStateRules._package_signature(attr_two)
        return "true" if (signature_one and signature_two
                          and signature_one == signature_two) else "false"

    @staticmethod
    def pack_format_equivalent(attr_one: str, attr_two: str) -> str:
        """true iff both sides measure a Pack Type and the forms agree."""
        one = (PackageStateRules._field(attr_one, "pack type") or "").strip().lower()
        two = (PackageStateRules._field(attr_two, "pack type") or "").strip().lower()
        return "true" if (one and two and one == two) else "false"


class PairLabelRules:
    """Per-field and per-pair labels read off the composed side literals."""

    @staticmethod
    def _field_same_label(side_one: dict[str, str], side_two: dict[str, str],
                          field: str) -> str:
        """same / different / unknown for one slice field, from side literals.

        Both sides measured and equal -> same; both measured and unequal ->
        different; either side unmeasured ('') -> unknown (never guessed).
        """
        one, two = side_one.get(field, ""), side_two.get(field, "")
        if one and two:
            return "same" if one == two else "different"
        return "unknown"

    @staticmethod
    def _has_evidence(side: dict[str, str]) -> bool:
        """Any of the six identity slice fields measured on one side."""
        return any(side.get(field) for field in PAIR_FIELDS)

    @staticmethod
    def evidence_sufficient(side_one: dict[str, str],
                            side_two: dict[str, str]) -> str:
        """true iff both sides carry at least one measured slice field."""
        return "true" if (PairLabelRules._has_evidence(side_one)
                          and PairLabelRules._has_evidence(side_two)) else "false"

    @staticmethod
    def same_brand_only(brand_one: str | None, brand_two: str | None,
                        identity_label: str | None) -> str | None:
        """true iff the brands agree and the pair is NOT the same item.

        `None` (omit the label) when either brand is unknown: the state does
        not carry brand evidence, so the label is only emitted where the
        catalog source supplies both brands AND the GTIN truth supplies the
        identity label.
        """
        if not brand_one or not brand_two or identity_label is None:
            return None
        same = brand_one.strip().lower() == brand_two.strip().lower()
        return "true" if (same and identity_label == "false") else "false"

    @staticmethod
    def _difficulty_slice(side_one: dict[str, str],
                          side_two: dict[str, str]) -> str:
        """A pair's difficulty from the same field-level agreement the
        questions use: how many measured fields disagree (never invented from
        a model)."""
        if not (PairLabelRules._has_evidence(side_one)
                and PairLabelRules._has_evidence(side_two)):
            return "insufficient"
        differing = sum(
            1 for field in PAIR_FIELDS
            if PairLabelRules._field_same_label(side_one, side_two, field)
            == "different")
        if differing == 0:
            return "all_same"
        if differing == 1:
            return "one_diff"
        return "multi_diff"

    @staticmethod
    def _primary_attribute(side_one: dict[str, str],
                           side_two: dict[str, str]) -> str:
        """The row's attribute tag: the first differing measured field, else
        the first measured field in the frozen slice order, else 'none'."""
        for field in PAIR_FIELDS:
            if PairLabelRules._field_same_label(side_one, side_two, field) \
                    == "different":
                return field
        for field in PAIR_FIELDS:
            if side_one.get(field) or side_two.get(field):
                return field
        return "none"


class GateReasonRules:
    """Free-form gate_reason -> its bounded family (text before ':')."""

    @staticmethod
    def gate_reason_family(reason: str) -> str:
        prefix = str(reason).split(":", 1)[0].strip()
        return GATE_REASON_FAMILIES.get(prefix, "unclassified")
