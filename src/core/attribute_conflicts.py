"""Shared attribute parsing and conflict classification.

Training-time mining, pair dumps, and post-run reports must use the same
structured canonical attributes.  Keeping this logic here prevents a report
from disagreeing with the population that was actually mined.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping

from core.text import unicode_casefold

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
    text = unicode_casefold(value)
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


def _has_consistency_flag(record: Mapping[str, object], flag: str) -> bool:
    """Read canonical flag sets from either Python objects or CSV strings."""
    value = record.get("attribute_consistency_flags")
    if value is None:
        return False
    if isinstance(value, (list, tuple, set, frozenset)):
        return flag in {str(item).strip() for item in value}
    text = str(value).strip()
    if not text:
        return False
    if text in {"set()", "frozenset()"}:
        return False
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"invalid canonical attribute flags: {value!r}") from exc
    if not isinstance(parsed, (list, tuple, set, frozenset)):
        raise ValueError(f"canonical attribute flags are not a sequence: {value!r}")
    return flag in {str(item).strip() for item in parsed}


def _universe_evidence_of(record: Mapping[str, object]) -> dict[str, frozenset]:
    """Read a record's captured universe-evidence section, fail-loud.

    Mapping values (in-memory records) are accepted verbatim; string values
    (CSV read-back) must be a Python-literal dict AND every value must be a
    set/sequence of tokens — anything else raises (no silent drop of captured
    evidence: from silence, the census would silently shrink instead of
    failing at its cause). Only registered keys are kept verbatim.
    """
    value = record.get("universe_evidence")
    if value is None:
        return {}
    if isinstance(value, Mapping):
        parsed = dict(value)
    elif str(value).strip() in {"set()", "frozenset()", "{}", ""}:
        return {}
    else:
        text = str(value).strip()
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"invalid canonical universe_evidence (a literal dict of "
                f"token lists is required, like every canonical set column): "
                f"{value!r}"
            ) from exc
    if not isinstance(parsed, Mapping):
        raise ValueError(f"canonical universe_evidence is not a mapping: {value!r}")
    out: dict[str, frozenset] = {}
    from core.attribute_universe import attribute_registry

    registry = attribute_registry()
    for key, item in parsed.items():
        if key not in registry:
            continue
        if not isinstance(item, (set, frozenset, list, tuple)):
            raise ValueError(
                f"universe_evidence[{key!r}] is not a set/sequence: {item!r}"
            )
        out[str(key)] = frozenset(str(token).casefold() for token in item)
    return out


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

    # Full-universe evidence (owner ruling 2026-10-01): canonical records
    # built BEFORE the full-capture wiring carry no such column, so absence
    # stays explicit absence here (an empty mapping, evaluated per-pair as
    # missing_* census states — never invented into agreement). A record
    # with a `universe_evidence` value (mapping on a dict record, Python
    # literal on a CSV record) is parsed with the same fail-loud AST device
    # the other canonical set fields use.
    universe_evidence = _universe_evidence_of(record)
    flavor_set = evidence("flavor_set", "flavor")
    return {
        # Keep the flagged raw values in canonical records for audit, but do
        # not promote them into evidence used to classify a pair.
        "volume": (
            set()
            if _has_consistency_flag(record, "ambiguous_volume")
            else _value_set(record.get("volume_set"), kind="volume")
        ),
        "pack": _value_set(record.get("pack_set"), kind="pack"),
        "package_type": _string_value_set(
            record.get("package_type_set"), kind="package_type"
        ),
        "flavor": " ".join(sorted(flavor_set)),
        "flavor_set": flavor_set,
        "carbonation": evidence("carbonation_set", "carbonation"),
        "sweetener": evidence("sweetener_set", "sweetener"),
        "sweetener_type": _string_value_set(record.get("sweetener_type_set"), kind="sweetener_type"),
        "sweetening": _string_value_set(record.get("sweetening_set"), kind="sweetening"),
        "attribute_consistency_flags": record.get("attribute_consistency_flags"),
        "pulp": evidence("pulp_set", "pulp"),
        # pack material (owner veto 2026-10-01): the canonical-level union
        # (package_material_set). Identity-safe by construction: both sides of
        # a verified true match share one canonical -> the same set -> the
        # set-intersection veto can never fire within it.
        "pack_material": _string_value_set(
            record.get("package_material_set"), kind="package_material"
        ),
        "universe_evidence": universe_evidence,
    }


def sku_attribute_info(title: object, attributes: object, description: object = "") -> dict[str, object]:
    """Extract SKU-side attributes using the same parser as the data lane."""
    from pipeline import extract_all

    desc = "" if description is None or (isinstance(description, float) and description != description) else str(description)
    extracted = extract_all(str(title), str(attributes), desc)
    volume_ml = extracted.get("volume_ml")
    # Zero is the extractor's sentinel for "no volume mention".  It must
    # remain unknown here; turning it into {0.0} makes every known canonical
    # volume look like a real conflict.
    try:
        volume = (
            {canonical_volume_ml(volume_ml)}
            if volume_ml is not None
            and float(volume_ml) > 0
            and "ambiguous_volume" not in set(extracted.get("attribute_consistency_flags") or set())
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
    # Full-universe parse (owner ruling 2026-10-01, ALL ATTRIBUTES): the raw
    # attribute cell is re-parsed by the census SSOT parser so every
    # registered key rides the SKU-side pair record — not only the six
    # extract_all capture keys. frozenset values, registry-keyed verbatim;
    # the `unclassified_keys` bucket is kept for the review lane. Additive
    # key: consumers enumerating the named fields never see it.
    universe_evidence = dict(parse_universe_cell(attributes))
    unclassified = universe_evidence.pop("unclassified_keys", ())
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
        "sweetener_type": set(extracted.get("sweetener_type_set") or set()),
        "sweetening": set(extracted.get("sweetening_set") or set()),
        "attribute_consistency_flags": set(extracted.get("attribute_consistency_flags") or set()),
        "pulp": set(extracted.get("pulp_set") or set()),
        # pack material (owner veto 2026-10-01): the captured attribute/title
        # evidence (pipeline.extract_all package_materials). Additive key for
        # targeted_veto_gate's discrete material clause; shared consumers that
        # enumerate CRITICAL_ATTRIBUTE_DIMENSIONS never read it, so the
        # training gate's decisions are untouched.
        "pack_material": set(extracted.get("package_materials") or set()),
        # full-universe captured evidence (see above), plus the unclassified
        # key bucket the review lane censuses; both stored even when empty so
        # the record is explicit about its own evidence breadth.
        "universe_evidence": universe_evidence,
        "unclassified_keys": tuple(sorted(unclassified)) if unclassified else (),
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
            else set()
            if dimension == "volume" and _has_consistency_flag(left, "ambiguous_volume")
            else left.get(dimension)
        )
        right_value = (
            right.get("flavor_set") or normalized_flavor_tokens(right.get("flavor"))
            if dimension == "flavor"
            else set()
            if dimension == "volume" and _has_consistency_flag(right, "ambiguous_volume")
            else right.get(dimension)
        )
        left_set = set(left_value or set())
        right_set = set(right_value or set())
        if not left_set or not right_set:
            unknown.append(dimension)
            continue
        if dimension == "volume":
            compatible = volumes_compatible(
                left_set, right_set,
                volume_relative_tolerance=volume_relative_tolerance,
                volume_absolute_tolerance_ml=volume_absolute_tolerance_ml,
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
    return critical_attribute_evaluation(
        left,
        right,
        volume_relative_tolerance=volume_relative_tolerance,
        volume_absolute_tolerance_ml=volume_absolute_tolerance_ml,
    )["conflicts"]


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


# ═══════════════════════════════════════════════════════════════════════════
# FULL-ATTRIBUTES DECISION LAYER (owner ruling 2026-10-01)
# "ALL ATTRIBUTES are used to make ALL DECISIONS"
# ═══════════════════════════════════════════════════════════════════════════
# BOTH volume tolerances (relative AND absolute) thread through this whole
# layer. The absolute cut is NOT cosmetic at small volumes: 5% of 14ml is
# 0.7ml, so a relative-only call classifies 14-vs-16ml as a CONFLICT while the
# veto lane — which applies both cuts — calls it compatible. A caller that
# omits `volume_absolute_tolerance_ml` therefore falls back to the parameter
# default of 0.0 and reintroduces that disagreement; pass the gate block's
# value (training_cfg().gate.vol_abs_tolerance).
#
# The critical-7 lane above stays BYTE-STABLE (predicates, call sites, output
# key order): the veto doctrine keeps absence-on-one-side as unknown, and a
# conflict votes only where config permits (rand_matching.targeted_veto_gates
# veto_dimensions). This section EXTENDS, never replaces: every
# AttributeUniverse-registered SET/BAND/ENUM/numeric key enters pair-level
# evaluation with its own measured conflict semantics (the registry's
# FieldSpec.conflict names), while non-critical dimensions are recorded as
# evidence for the audit/trace lanes — the veto still fires only where the
# config lists the dimension.
#
# Prohibitios held here (measured origin):
#   * absence on either side is UNKNOWN, never agreement and never veto
#     (product_identity doctrine 2; pinned by test_full_attribute_decisions);
#   * a value that fails its band/enum grammar stays in evidence but reports
#     `unknown_parse` instead of pretending a set inequality is measured.


# Pair census states, exactly the audit vocabulary the trace column contract
# publishes. `missing_left/right/both` are populated-side observations, not
# decisions: they can never become a conflict (absence is unknown).
DIMENSION_STATES: frozenset[str] = frozenset(
    {"agree", "conflict", "missing_left", "missing_right", "missing_both", "unknown_parse"}
)


def _universe_parser():
    """The ONE AttributeUniverse parser instance, reused (parse is pure).

    AttributeUniverse validates its frame argument loudly, so the sentinel
    empty frame is constructed once here with the canonical column names —
    no silent anything: a registry drift raises at import of the census
    module, exactly where the owner would want it.
    """
    import pandas as pd

    from core.attribute_universe import AttributeUniverse

    frame = pd.DataFrame(
        {"attributes": pd.Series([], dtype=object), "barcode": pd.Series([], dtype=object)}
    )
    return AttributeUniverse(frame)


def parse_universe_cell(cell: object) -> dict[str, object]:
    """One raw attribute cell -> registered {key: parsed value} (census SSOT parse).

    Uses AttributeUniverse.parse, so band canon, delegated flavor/sweetener
    and volume semantics are the census lane's own device — no second parser
    exists. Unknown keys ride the `unclassified_keys` bucket exactly like the
    census and core.product_dimensions do.
    """
    return _universe_parser().parse(str(cell or ""))


def _band_parseable(token: object) -> bool:
    """Whether a canonical band token matches the census band grammar.

    Mirrors core.attribute_universe._canonical_band's acceptance rules — the
    SAME three pinned regexes, imported (never copied) — so a pair with a
    populated non-numeric band token reports `unknown_parse` instead of a
    spurious set inequality. Pinned bidirectionally in
    tests/test_universe_capture_wiring.py, which lives above this lane.
    """
    from core.attribute_universe import _BAND_EXACT_RE, _BAND_PLUS_RE, _BAND_RANGE_RE

    plain = (
        str(token or "").strip().lower().replace("–", "-").replace(" ", "")
    )
    return bool(
        _BAND_RANGE_RE.match(plain)
        or _BAND_PLUS_RE.match(plain)
        or _BAND_EXACT_RE.match(plain)
    )


def _universe_value(record: Mapping[str, object], key: str, spec) -> object:
    """Resolve one registered key's parsed value from a pair record.

    Critical-7 mirrored channels stay FIRST (they are the SSOT evidence the
    gate already reads): volume uses the gate's own ambiguous-volume rule,
    flavour the shared declared-token set, sweetener the RAW ingredient
    identity channel (sweetener_type+sweetening — deliberately NOT the
    critical sweetener_claim set, whose sugar-claim semantics belong to
    critical_attribute_evaluation only). Every other key reads the captured
    `universe_evidence` section (parsed by parse_universe_cell). Absence of a
    key resolves to an empty frozenset — populated-side evidence, evaluated
    as missing by the caller, never invented.
    """
    if key == "flavour":
        value = record.get("flavor_set")
        if value is None:
            value = normalized_flavor_tokens(record.get("flavor"))
        return frozenset(str(item).casefold() for item in (value or ()))
    if key == "volume":
        flags = record.get("attribute_consistency_flags")
        if isinstance(flags, (set, frozenset, list, tuple)):
            if "ambiguous_volume" in {str(item) for item in flags}:
                return frozenset()
        return frozenset(record.get("volume_set") or record.get("volume") or ())
    if key == "sweetener":
        merged: set[str] = set()
        for channel in ("sweetener_type", "sweetening"):
            for item in (record.get(channel) or ()):
                merged.add(str(item).strip().casefold())
        return frozenset(merged)
    if key == "pack material type":
        # The pack material veto rides its OWN channel (owner ruling
        # 2026-10-01): sku_attribute_info/canonical_attribute_info expose the
        # captured material union as `pack_material`/`package_material_set`,
        # NOT the universe_evidence capture (which is referenced by name for
        # the NOT-yet-wired keys only).
        return frozenset(
            record.get("pack_material") or record.get("package_material_set") or ()
        )
    if key == "carbonization":
        # Mirror the critical carbonation channel (SSOT map
        # VETO_CENSUS_KEY_BY_DIMENSION["carbonation"]): the canonical set
        # column is the gate's own evidence, the universe_evidence capture
        # rides behind it for records that carry the column.
        value = (
            record.get("carbonation_set") or record.get("carbonation")
            or _universe_evidence_of(record).get(key) or ()
        )
        return frozenset(_string_value_set(value, kind="carbonation"))
    if key == "pack type":
        # Mirror the critical pack channel the same way.
        value = (
            record.get("pack_set") or record.get("pack")
            or _universe_evidence_of(record).get(key) or ()
        )
        return frozenset(_string_value_set(value, kind="pack type"))
    evidence = record.get("universe_evidence")
    if not isinstance(evidence, Mapping):
        return frozenset()
    return frozenset(evidence.get(key) or ())


def _census_state_for_field(
    left_value: object,
    right_value: object,
    spec,
    *,
    volume_relative_tolerance: float,
    volume_absolute_tolerance_ml: float,
    ) -> str:
    """Classify one registered field pair into the DIMENSION_STATES vocabulary."""
    left_set = frozenset(left_value) if left_value else frozenset()
    right_set = frozenset(right_value) if right_value else frozenset()
    if not left_set and not right_set:
        return "missing_both"
    if not right_set:
        return "missing_right"
    if not left_set:
        return "missing_left"
    if (
        spec.kind == "NUMERIC_BAND"
        and (
            not all(_band_parseable(token) for token in left_set)
            or not all(_band_parseable(token) for token in right_set)
        )
    ):
        return "unknown_parse"

    # The registry's own measured conflict semantics (the SAME predicate
    # identities attribute_universe._conflict_predicates voices, applied
    # with pair tolerances where the field is the volume channel).
    if spec.conflict == "volume_compatible":
        conflict = not volumes_compatible(
            set(left_set),
            set(right_set),
            volume_relative_tolerance=volume_relative_tolerance,
            volume_absolute_tolerance_ml=volume_absolute_tolerance_ml,
        )
    elif spec.conflict == "flavor_overlap":
        conflict = flavor_overlap_metrics(set(left_set), set(right_set))[1] == 0.0
    else:
        conflict = frozenset(left_set) != frozenset(right_set)
    return "conflict" if conflict else "agree"


def full_dimension_states(
    left: Mapping[str, object],
    right: Mapping[str, object],
    *,
    volume_relative_tolerance: float = 0.0,
    volume_absolute_tolerance_ml: float = 0.0,
    registry: Mapping | None = None,
) -> dict[str, str]:
    """ALL-registered-dimensions census for one pair (owner ruling 2026-10-01).

    Returns every AttributeUniverse-registered key sorted, mapped to exactly
    one of DIMENSION_STATES evaluated with the field's own measured conflict
    semantics:
      SET_*      -> frozenset inequality,
      volume     -> volumes_compatible (unit_canonicalization device),
      flavour    -> declared-token overlap coefficient == 0,
      caffeine/weight/bands -> bounded canonical-band inequality.

    Non-critical dimensions CANNOT veto here — states are evidence and the
    veto list stays config-owned; callers that consume critical-7 outputs get
    them from full_attribute_evaluation, which delegates to the unchanged
    critical_attribute_evaluation on top of this census.
    """
    from core.attribute_universe import attribute_registry

    specs = dict(registry) if registry is not None else attribute_registry()
    if not specs:
        raise ValueError("full_dimension_states requires a non-empty registry")
    states: dict[str, str] = {}
    for key in sorted(specs):
        states[key] = _census_state_for_field(
            _universe_value(left, key, specs[key]),
            _universe_value(right, key, specs[key]),
            specs[key],
            volume_relative_tolerance=volume_relative_tolerance,
            volume_absolute_tolerance_ml=volume_absolute_tolerance_ml,
        )
    return states


def full_attribute_evaluation(
    left: Mapping[str, object],
    right: Mapping[str, object],
    *,
    volume_relative_tolerance: float = 0.0,
    volume_absolute_tolerance_ml: float = 0.0,
    registry: Mapping | None = None,
    left_raw: Mapping[str, object] | None = None,
    right_raw: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Critical-7 decision record EXTENDED with the full-decision evidence.

    The first three keys are the UNCHANGED critical-7 contract
    (same predicates, same ordering) — every existing consumer reads them
    byte-stable. `dimension_states` / `dimension_conflicts` carry the
    owner-ruling ALL-ATTRIBUTES census; water type et al. flow into them via
    the registry's own measured semantics, and the veto still votes only
    where rand_matching.targeted_veto_gates.veto_dimensions permits.
    ``dimension_evidence`` is the single decision engine's all-metrics
    adjudication (owner directive: ALL attributes × ALL metrics for the
    entire decision process) — the ordered stack ending in the stage-7
    original-column re-parse when raw rows are supplied.
    """
    critical = critical_attribute_evaluation(
        left,
        right,
        volume_relative_tolerance=volume_relative_tolerance,
        volume_absolute_tolerance_ml=volume_absolute_tolerance_ml,
    )
    states = full_dimension_states(
        left,
        right,
        volume_relative_tolerance=volume_relative_tolerance,
        volume_absolute_tolerance_ml=volume_absolute_tolerance_ml,
        registry=registry,
    )
    conflicts = sorted(key for key, state in states.items() if state == "conflict")
    from core.attribute_decision import AttributeDecisionEngine

    pair_evidence = AttributeDecisionEngine(
        volume_relative_tolerance=float(volume_relative_tolerance),
        volume_absolute_tolerance_ml=float(volume_absolute_tolerance_ml),
    ).evaluate(left, right, left_raw=left_raw, right_raw=right_raw, registry=registry)
    return {
        **critical,
        "dimension_states": states,
        "dimension_conflicts": conflicts,
        "dimension_evidence": pair_evidence,
    }


def dimension_census_columns(states: Mapping[str, str]) -> dict[str, object]:
    """Audit columns (existing key_naming convention) for one pair's census.

    One `<key>_state` column per registered dimension (spaces -> underscores,
    the published value convention), plus the conflict/missing/unknown rollups
    the trace and the pair reports audit against. Column names are
    deterministic so a pair dump is diffable run to run.
    """
    states = dict(states)
    unknown = sorted(set(states.values()) - DIMENSION_STATES)
    if unknown:
        raise ValueError(
            f"dimension states outside the audited vocabulary {sorted(DIMENSION_STATES)}: {unknown}"
        )
    columns: dict[str, object] = {
        key.replace(" ", "_") + "_state": state for key, state in states.items()
    }
    conflicts = sorted(key for key, state in states.items() if state == "conflict")
    columns["dimension_conflicts"] = ",".join(k.replace(" ", "_") for k in conflicts)
    columns["dimension_conflict_count"] = len(conflicts)
    for state in ("missing_left", "missing_right", "missing_both", "unknown_parse"):
        members = sorted(key for key, item in states.items() if item == state)
        columns[f"dimension_{state}"] = ",".join(
            key.replace(" ", "_") for key in members
        )
    return columns


# ── veto-eligibility ledger ───────────────────────────────────────────────────
# Measured veto band (core.attribute_universe VETO_RATE_FLOOR/CEILING 2.5%-15%
# same-GTIN conflict rate). Non-critical keys are REPORTED, never flipped:
# the owner applies the delta (config veto_dimensions + schema allow-list).
VETO_CENSUS_KEY_BY_DIMENSION: dict[str, str] = {
    # critical/veto-lane dimension name -> census registry key (name differs
    # where the two lanes name the same evidence differently, SSOT mapping
    # documented here once).
    "volume": "volume",
    "pack": "pack type",
    "package_type": "",   # title-parsed; no raw attribute-cell census key
    "flavor": "flavour",
    "carbonation": "carbonization",
    "sweetener": "sweetener",
    "pulp": "",           # prose claim lane; no registry key
    "pack_material": "pack material type",
}
CONFIG_MEASURED_VETO_ARGUMENT: dict[str, str] = {
    # config/training.yaml targeted_veto_gates measured origin (TRUE LOST is
    # the identity-safety number the owner's own table leads with).
    "volume": "60 false merges removed, 0 true lost (config measured table)",
    "pack": "20 false merges removed, 0 true lost",
    "package_type": "13 false merges removed, 0 true lost",
    "flavor": "1 false merge removed, 0 true lost (delegated rate 0.09%)",
    "carbonation": "0 false merges removed",
    "pulp": "3 false merges removed, 0 true lost",
    "sweetener": "6 false merges, 74 TRUE LOST — 12:1 against, excluded by the owner",
    "pack_material": "32,641/67,899 disjoint both-populated; same-canonical subset argument",
}

# The decision engine publishes conflicts under CENSUS registry keys ("water
# type"); the veto/audit lanes speak CRITICAL dimension names ("volume",
# "pack_material"). This inversion is the SSOT mapping between the two
# vocabularies — every consumer maps through it, never a second table.
CRITICAL_NAME_BY_CENSUS_KEY: dict[str, str] = {
    census_key: dimension
    for dimension, census_key in VETO_CENSUS_KEY_BY_DIMENSION.items()
    if census_key
}


def veto_eligibility_ledger(
    *,
    census: Mapping[str, Mapping[str, object]] | None = None,
    config: object | None = None,
) -> dict[str, dict[str, object]]:
    """Per-dimension veto-eligibility ledger (evidence class + config delta).

    Reads ONLY evidence and CURRENT config state — never writes either:
      * census  results/attribute_universe_census.json['census']['keys']
        (conflict rates are the evidence class's measured origin);
      * config  rand_matching.targeted_veto_gates.veto_dimensions and its
        schema allow-list (CRITICAL_ATTRIBUTE_DIMENSIONS + pack_material).
    The owner flips config; this ledger reports the exact delta needed.
    """
    from core.attribute_universe import (
        NON_YIELD_KINDS,
        VETO_RATE_CEILING,
        VETO_RATE_FLOOR,
        attribute_registry,
    )
    from core.critical_attributes import CRITICAL_ATTRIBUTE_DIMENSIONS

    if census is None:
        from core.common import RESULTS

        path = RESULTS / "attribute_universe_census.json"
        if not path.exists():
            raise FileNotFoundError(
                f"veto_eligibility_ledger requires {path.as_posix()} (the "
                "attribute census is the ledger's evidence source — no silent fallback)"
            )
        import json

        census = json.loads(path.read_text())["census"]["keys"]

    settings = config
    if settings is None:
        from core.common import training_cfg

        settings = training_cfg().rand_matching.targeted_veto_gates
    vetoed = {str(name).strip() for name in settings.veto_dimensions}

    registry = attribute_registry()
    schema_admitted = set(CRITICAL_ATTRIBUTE_DIMENSIONS) | {"pack_material"}

    ledger: dict[str, dict[str, object]] = {}
    for dimension in sorted(set(registry) | set(schema_admitted) | vetoed):
        census_key = VETO_CENSUS_KEY_BY_DIMENSION.get(dimension, dimension)
        stats = census.get(census_key) if census_key else None
        rate = float(stats["conflict_rate"]) if stats else None
        spec = registry.get(census_key) if census_key else None
        if spec is not None and spec.kind in NON_YIELD_KINDS:
            evidence_class = "no_yield"
        elif stats is None:
            evidence_class = "non_attribute_cell_lane"
        elif rate is not None and rate < VETO_RATE_FLOOR:
            evidence_class = "below_veto_floor"
        elif rate is not None and rate > VETO_RATE_CEILING:
            evidence_class = "review_lane_above_band"
        else:
            evidence_class = "veto_band_candidate"

        if dimension in vetoed:
            config_state = "hard_veto_permitted"
            owner_delta = ""
        else:
            config_state = "audit_only"
            owner_delta = (
                f"add {dimension!r} to rand_matching.targeted_veto_gates.veto_dimensions"
            )
            if dimension not in schema_admitted:
                owner_delta += (
                    " AND extend the targeted-veto schema allow-list "
                    "(core.schemas.TargetedVetoGatesSpec._veto_dimensions_are_critical)"
                )
        identity_safety = CONFIG_MEASURED_VETO_ARGUMENT.get(
            dimension,
            "raw attribute-cell capture only — NOT identity-safe as a veto "
            "until a canonical-union channel exists for it (see pack_material)",
        )
        ledger[dimension] = {
            "conflict_rate": rate,
            "census_key": census_key,
            "registry_kind": spec.kind if spec else None,
            "conflict_semantics": spec.conflict if spec else None,
            "evidence_class": evidence_class,
            "identity_safety": identity_safety,
            "config_state": config_state,
            "in_schema_allow_list": dimension in schema_admitted,
            "owner_delta": owner_delta,
        }
    return ledger


__all__ = [
    "DIMENSION_STATES",
    "VETO_CENSUS_KEY_BY_DIMENSION",
    "attribute_conflict_types",
    "canonical_attribute_info",
    "conflict_columns",
    "critical_attribute_evaluation",
    "dimension_census_columns",
    "flavor_overlap_metrics",
    "full_attribute_evaluation",
    "full_dimension_states",
    "normalized_flavor_tokens",
    "parse_universe_cell",
    "strict_attribute_gate",
    "sku_attribute_info",
    "veto_eligibility_ledger",
]
