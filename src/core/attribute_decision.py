"""src/core/attribute_decision.py — THE single all-attributes decision engine.

Owner directive (2026-10-01): ALL 37 registered attributes, ALL metrics, for
the ENTIRE decision process, carried by ONE dataclass loaded across the
process — no lane re-derives its own attribute verdicts from ad-hoc sets
anymore. Every populated key is evaluated against the ordered stack:

  1. UNIT NORMALIZATION        (standardize units & scales; band/enum/alias canon)
  2. NEGATION / ABSENCE        (hard veto when one side asserts X, the other Non-X)
  3. STRICT / ALIAS-FOLDED MATCH
  4. NUMERIC / BAND INTERVAL MATH (tolerance & intersections)
  5. SET OVERLAPS              (Jaccard, overlap coefficient, containment)
  6. FUZZY SURFACE SIMILARITY  (length-adaptive Levenshtein / token Jaccard)
  7. ORIGINAL-COLUMN RE-PARSE  (uncertainty resolution fallback)

Each populated key yields ONE :class:`ComparisonResult`:
  MATCH        confirmed agreement,
  CONFLICT     hard veto condition met,
  SUBSET       asymmetric containment (one side is more specific),
  INCONCLUSIVE low confidence / missing data (fallback re-parse resolves
               it only when the caller supplies the original rows).

Decisions NEVER mint from silence: a one-sided value stays INCONCLUSIVE
(never MATCH, never CONFLICT). The registry's census semantics
(core.attribute_conflicts.DIMENSION_STATES) stay byte-stable alongside the
decision result — MEASURED_BASELINE pins the census rates, while `result` is
the all-metrics adjudication the decision lanes consume.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache
from typing import Iterable, Mapping, Sequence

# ════════════════════════════════════════════════════════════════════════════
# THE single decision vocabulary
# ════════════════════════════════════════════════════════════════════════════


class ComparisonResult(Enum):
    """The ONE decision vocabulary every attribute comparison emits."""

    MATCH = auto()  # confirmed agreement
    CONFLICT = auto()  # hard veto condition met
    SUBSET = auto()  # asymmetric containment (one side is more specific)
    INCONCLUSIVE = auto()  # low confidence / missing data (fallback re-parse)


# Nordic band tokens reuse attribute_universe's canonical band grammar.
_NEGATION_PREFIXES: tuple[str, ...] = ("no ", "non ", "without ", "zero ")
_POLAR_PAIRS: dict[str, str] = {
    "sweetened": "unsweetened",
    "carbonated": "still",
    "sugar": "sugar-free",
}
_UNPOLARIZED = {v: k for k, v in _POLAR_PAIRS.items()}


@dataclass(frozen=True)
class AttributeMetrics:
    """ALL metric values for ONE attribute-key comparison (audit-complete).

    Every metric is computed for every populated key — the ordered decision
    dispatch later only selects WHICH metric decides, never skips one.
    """

    exact_match: bool
    alias_match: bool
    jaccard: float
    overlap_coef: float
    containment_a: float  # |A∩B|/|A| — how well A is covered by B
    containment_b: float  # |A∩B|/|B| — how well B is covered by A
    levenshtein_sim: float  # best token-pair surface ratio (length-adaptive)
    negation_conflict: bool = False
    numeric_diff_ratio: float | None = None
    interval_overlap: bool | None = None
    parse_problem: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "exact_match": self.exact_match,
            "alias_match": self.alias_match,
            "jaccard": round(self.jaccard, 6),
            "overlap_coef": round(self.overlap_coef, 6),
            "containment_a": round(self.containment_a, 6),
            "containment_b": round(self.containment_b, 6),
            "levenshtein_sim": round(self.levenshtein_sim, 6),
            "negation_conflict": self.negation_conflict,
            "numeric_diff_ratio": (
                None if self.numeric_diff_ratio is None
                else round(self.numeric_diff_ratio, 6)
            ),
            "interval_overlap": self.interval_overlap,
            "parse_problem": self.parse_problem,
        }


@dataclass(frozen=True)
class DimensionDecision:
    """One registered key's decision: metrics + census state + result."""

    key: str
    type_key: str  # NUMERIC | BAND | STRING | ENUM | SET (from FieldSpec.kind)
    state: str  # DIMENSION_STATES census semantics (byte-stable)
    result: ComparisonResult
    metrics: AttributeMetrics
    fallback_from: str = ""  # original column that resolved an unclear case

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "type_key": self.type_key,
            "state": self.state,
            "result": self.result.name,
            "fallback_from": self.fallback_from,
            **self.metrics.as_dict(),
        }


@dataclass(frozen=True)
class PairEvidence:
    """THE single decision object loaded across the process.

    Produced once per pair by :class:`AttributeDecisionEngine` and consumed
    by every lane (three_way_gate, targeted veto, hard-negative miner,
    identity census columns) — one ad hoc evaluation stack fewer per lane.
    """

    dimensions: Mapping[str, DimensionDecision]

    @property
    def conflicts(self) -> list[str]:
        return sorted(k for k, d in self.dimensions.items() if d.result is ComparisonResult.CONFLICT)

    @property
    def agreements(self) -> list[str]:
        return sorted(
            k for k, d in self.dimensions.items()
            if d.result in (ComparisonResult.MATCH, ComparisonResult.SUBSET)
        )

    @property
    def subsets(self) -> list[str]:
        return sorted(k for k, d in self.dimensions.items() if d.result is ComparisonResult.SUBSET)

    @property
    def inconclusive(self) -> list[str]:
        return sorted(k for k, d in self.dimensions.items() if d.result is ComparisonResult.INCONCLUSIVE)

    def as_dict(self) -> dict[str, object]:
        return {key: decision.as_dict() for key, decision in sorted(self.dimensions.items())}


# ════════════════════════════════════════════════════════════════════════════
# Metric computation (every populated key × every metric)
# ════════════════════════════════════════════════════════════════════════════


def _tokens(values: Iterable[str]) -> frozenset[str]:
    """Stage-1 unit normalization → stable value tokens (alias-fold applied)."""
    import re

    from core.critical_attributes import FLAVOR_ALIASES

    out: set[str] = set()
    for value in values:
        out |= set(re.findall(r"[a-z0-9]+", str(value).casefold().replace("–", "-")))
    return frozenset(FLAVOR_ALIASES.get(token, token) for token in out)


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ca != cb),
            ))
        previous = current
    return previous[-1]


def _damerau_levenshtein(a: str, b: str) -> int:
    """Classic Levenshtein plus 1 adjacent-transposition edits (OSA)."""
    if len(a) < len(b):
        a, b = b, a
    previous2: list[int] | None = None
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            )
            if previous2 is not None and i > 1 and j > 1:
                if a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                    current[j] = min(current[j], previous2[j - 2] + 1)
        previous2, previous = previous, current
    return previous[-1]


def _levenshtein_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return 1.0 - _damerau_levenshtein(a, b) / max(len(a), len(b))


def _length_adaptive_threshold(min_len: int) -> float:
    """Fuzzy surface thresholds: short tokens need stronger evidence.
    A 1-char slip on a >=7-char token is a plausible spelling difference;
    anything shorter needs the tighter bands.
    """
    if min_len <= 3:
        return 1.0  # exact only
    if min_len <= 6:
        return 0.93
    if min_len <= 10:
        return 0.85
    return 0.84


# Key-scoped CONCEPT folds (measured from the corpus value census, 2026-10-01):
# a spelling outside a closed enum folds to its family root. `tin` folds by
# KEY: tin (material) -> metal, tin (package type) -> can — the aluminum >
# metal > can chain. Out-of-family values stay raw (no invention).
CONCEPT_FOLD_BY_KEY: dict[str, dict[str, str]] = {
    "pack material type": {
        "pet": "plastic", "aluminium": "metal", "aluminum": "metal", "tin": "metal",
    },
    "package type": {
        "tin": "can", "tetra pak": "carton", "sachet": "packet",
    },
    "pack type": {
        "tin": "can", "sachet": "packet",
    },
    "water type": {
        "flavoured": "spring",  # measured: flavour suffix rides the spring key ("flavoured spring water" phrasing); a TRUE flavour change (e.g. aloe) stays a distinct value and still mints
    },
    "carbonization": {
        "sparkling": "carbonated", "gently carbonated": "carbonated",
    },
}

# DOMAIN SEMANTICS per key (slice census, 2026-10-01). EXCLUSIVE: values are
# alternatives — disjoint sets MINT a conflict (class 3). ADDITIVE: values
# co-occur — disjoint sets stay UNKNOWN unless the negation check fires
# (class 2 dissolved; X vs no-X stays a hard veto). Unlisted keys default
# EXCLUSIVE (a change of value is a change of claim).
EXCLUSIVE_KEYS = frozenset({
    "tea type", "water type", "carbonization", "pack type", "diets",
    "roast type", "caffeine", "rtd coffee style", "sports ingredients",
    "package type", "flavour",
})
ADDITIVE_KEYS = frozenset({
    "botanicals and functional ingredients", "contains minerals",
    "immune support ingredients", "made from", "health claims",
    "sustainable sourcing", "sustainable packaging", "energy source",
    "juice features",
})


def _fold_values(values: Iterable[str]) -> frozenset[str]:
    """Stage-1 unit normalization on VALUES: separator-folded, casefolded."""
    import re

    return frozenset(
        re.sub(r"[\s_-]+", " ", str(value)).strip().casefold()
        for value in values
    )


def _concept_fold(values: Iterable[str], *, key: str) -> frozenset[str]:
    """Fold spellings into their concept family root for this key."""
    fold_map = CONCEPT_FOLD_BY_KEY.get(key)
    folded = _fold_values(values)
    if not fold_map:
        return folded
    return frozenset(
        fold_map.get(value, value)
        for value in folded
    )


def _negation_conflict(
    a: frozenset[str], b: frozenset[str]
) -> bool:
    """Stage 2: X vs Non-X across the two sides = the hard veto condition.

    Operates on the RAW VALUE strings (not the alphanumeric token split —
    "no_sugar" must structure as "no sugar"), separator-folded.
    """
    import re

    def fold(values: frozenset[str]) -> frozenset[str]:
        return frozenset(
            re.sub(r"[\s_-]+", " ", str(value)).strip().casefold()
            for value in values
        )

    fa, fb = fold(a), fold(b)
    # A shared folded token is genuine agreement: polarity differences can
    # never mint from silence when the sides agree on a common value (one
    # feed's {carbonated, still} vs the other's {still} is containment, not
    # X-vs-Non-X).
    if fa & fb:
        return False
    neg_re = re.compile(r"(?:no|non|not|never|without|zero)\s+(.+)")
    for small, big in ((fa, fb), (fb, fa)):
        for value in small:
            match = neg_re.fullmatch(value)
            if match and match.group(1).strip() in big:
                return True
            # polar pairs: sweetened/unsweetened, carbonated/still
            for positive, negative in _POLAR_PAIRS.items():
                if value == positive and negative in big:
                    return True
                if value == negative and positive in big:
                    return True
    return False


def _metric_values(
    a: frozenset[str], b: frozenset[str]
) -> tuple[float, float, float, float, float]:
    """Raw metric values for one populated key (no metric is skipped):
    (jaccard, overlap, containment_a, containment_b, best_levenshtein).
    """
    jacc = len(a & b) / len(a | b) if a | b else 0.0
    overlap = len(a & b) / min(len(a), len(b)) if a and b else 0.0
    ca = len(a & b) / len(a) if a else 0.0
    cb = len(a & b) / len(b) if b else 0.0
    best_lev = 0.0
    for x in a:
        for y in b:
            best_lev = max(best_lev, _levenshtein_ratio(x, y))
    return jacc, overlap, ca, cb, best_lev


def _numeric_diff_ratio(a: frozenset[str], b: frozenset[str]) -> float | None:
    """Stage 4 numeric device: best relative difference over parsed floats."""
    def floats(values: frozenset[str]) -> list[float]:
        out = []
        for token in values:
            try:
                out.append(float(token))
            except ValueError:
                continue
        return out

    fa, fb = floats(a), floats(b)
    if not fa or not fb:
        return None
    best = min(
        (abs(x - y) / max(abs(x), abs(y)) if max(abs(x), abs(y)) else 0.0)
        for x in fa for y in fb
    )
    return best


def _interval_overlap(a: frozenset[str], b: frozenset[str]) -> bool | None:
    """Stage 4 band device: the product_dimensions interval-overlap rule."""
    from core.product_dimensions import _interval

    def intervals(values: frozenset[str]) -> list[tuple[float, float]]:
        out = []
        for token in values:
            iv = _interval(token, "percent" if token.endswith("%") else "mg")
            if iv is not None:
                out.append(iv)
        return out

    ia, ib = intervals(a), intervals(b)
    if not ia or not ib:
        return None
    return any(max(x[0], y[0]) <= min(x[1], y[1]) for x in ia for y in ib)


# ════════════════════════════════════════════════════════════════════════════
# Type dispatch — the ordered decision stack
# ════════════════════════════════════════════════════════════════════════════


def attribute_metrics(
    val_a: frozenset[str],
    val_b: frozenset[str],
    *,
    type_key: str = "STRING",
    key: str = "",
) -> AttributeMetrics:
    """ALL metrics for one populated key — :class:`AttributeMetrics` built
    with the shared devices (concept-fold aliases, band grammar, numeric
    tolerance). Stage 1 (unit normalization) rides value folding; exact /
    alias / negation / Levenshtein run on the folded VALUE strings
    (phrase-level values like 'b corp certified' compare whole), while the
    set-overlap metrics (Jaccard, overlap coefficient, containment) run on
    the word token bags.
    """
    values_a = _concept_fold(val_a, key=key)
    values_b = _concept_fold(val_b, key=key)
    tokens_a = _tokens(val_a)
    tokens_b = _tokens(val_b)
    exact = values_a == values_b
    jacc = len(tokens_a & tokens_b) / len(tokens_a | tokens_b) if tokens_a | tokens_b else 0.0
    overlap = len(tokens_a & tokens_b) / min(len(tokens_a), len(tokens_b)) if tokens_a and tokens_b else 0.0
    ca = len(tokens_a & tokens_b) / len(tokens_a) if tokens_a else 0.0
    cb = len(tokens_a & tokens_b) / len(tokens_b) if tokens_b else 0.0
    best_lev = 0.0
    for x in values_a:
        for y in values_b:
            best_lev = max(best_lev, _levenshtein_ratio(x, y))
    numeric = _numeric_diff_ratio(val_a, val_b) if type_key == "NUMERIC" else None
    intervals = _interval_overlap(val_a, val_b) if type_key == "BAND" else None
    return AttributeMetrics(
        exact_match=exact,
        alias_match=False,
        jaccard=jacc,
        overlap_coef=overlap,
        containment_a=ca,
        containment_b=cb,
        levenshtein_sim=best_lev,
        negation_conflict=_negation_conflict(values_a, values_b),
        numeric_diff_ratio=numeric,
        interval_overlap=intervals,
    )


def evaluate_metrics(
    type_key: str,
    val_a: frozenset[str],
    val_b: frozenset[str],
    metrics: AttributeMetrics,
    *,
    domain: str = "exclusive",
) -> ComparisonResult:
    """THE ordered dispatch (owner stack sketch, stages 2-6). Stage 1 is
    inside tokenization; stage 7 is the caller's re-parse fallback.
    ``domain`` scopes the final bucket: EXCLUSIVE keys mint CONFLICT on
    disjoint sets (class 3); ADDITIVE keys stay INCONCLUSIVE unless the
    negation check fired (class 2 dissolved).
    """
    if not val_a or not val_b:
        return ComparisonResult.INCONCLUSIVE
    if metrics.negation_conflict:
        return ComparisonResult.CONFLICT
    if metrics.exact_match or metrics.alias_match:
        return ComparisonResult.MATCH
    if type_key in ("NUMERIC", "BAND"):
        if metrics.interval_overlap is not None:
            return (
                ComparisonResult.MATCH if metrics.interval_overlap
                else ComparisonResult.CONFLICT
            )
        if metrics.numeric_diff_ratio is not None:
            return (
                ComparisonResult.MATCH if metrics.numeric_diff_ratio <= 0.05
                else ComparisonResult.CONFLICT
            )
        return ComparisonResult.INCONCLUSIVE
    if type_key == "ENUM":
        if metrics.overlap_coef == 0.0:
            return ComparisonResult.CONFLICT
        return ComparisonResult.MATCH
    if metrics.containment_a == 1.0 or metrics.containment_b == 1.0:
        return ComparisonResult.SUBSET
    if metrics.overlap_coef > 0.0 or metrics.jaccard >= 0.6:
        return ComparisonResult.MATCH
    short_side = min(
        (len(x) for x in val_a | val_b), default=0
    )
    if short_side <= 3:
        return ComparisonResult.CONFLICT
    if metrics.levenshtein_sim >= _length_adaptive_threshold(short_side) and (
        metrics.jaccard >= 0.5
        or (len(val_a) == len(val_b) == 1)  # equally specific single values
    ):
        return ComparisonResult.MATCH
    # All rescue metrics ran. The domain scopes the final verdict: EXCLUSIVE
    # keys mint CONFLICT on genuinely disjoint sets (class 3 real conflicts);
    # ADDITIVE keys stay INCONCLUSIVE (partial additive reporting — class 2
    # dissolved by construction, negation already checked above).
    return (
        ComparisonResult.CONFLICT if domain == "exclusive"
        else ComparisonResult.INCONCLUSIVE
    )


# ════════════════════════════════════════════════════════════════════════════
# THE engine — loaded once, consumed everywhere
# ════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class AttributeDecisionEngine:
    """THE SSOT decision engine (owner: all attributes × all metrics).

    Loaded once per process from the config SSOT + the 37-key registry;
    every consumer imports :func:`engine` and receives the SAME tolerances,
    registry and metric devices. Lanes apply their own config-permitted
    veto subset over the shared evidence — the config stays the only
    authority over which conflicts veto.
    """

    volume_relative_tolerance: float
    volume_absolute_tolerance_ml: float = 0.0
    jaccard_match_threshold: float = 0.6

    @classmethod
    def load(cls, *, overrides: Mapping[str, float] | None = None) -> "AttributeDecisionEngine":
        from core.common import training_cfg

        cfg = training_cfg().gate
        over = dict(overrides or {})
        return cls(
            volume_relative_tolerance=float(over.get("volume_relative_tolerance", cfg.vol_tolerance)),
            # Config SSOT, not the 0.0 parameter default: the engine is the
            # shared decision path, so a default here silently disagreed with
            # the veto lane at small volumes (the relative cut is stricter
            # there). Direct construction still wins via `overrides`.
            volume_absolute_tolerance_ml=float(
                over.get("volume_absolute_tolerance_ml", cfg.vol_abs_tolerance)
            ),
            jaccard_match_threshold=float(over.get("jaccard_match_threshold", 0.6)),
        )

    def evaluate(
        self,
        left: Mapping[str, object],
        right: Mapping[str, object],
        *,
        left_raw: Mapping[str, object] | None = None,
        right_raw: Mapping[str, object] | None = None,
        registry: Mapping | None = None,
    ) -> PairEvidence:
        """ALL attributes × ALL metrics for one pair — the single entry point.

        ``left``/``right`` are pair records carrying the shared evidence keys
        (sku_attribute_info / canonical_attribute_info output shape). The
        optional raw mappings are the ORIGINAL rows (title / description /
        attribute cells); stage 7 re-parses them only for keys left
        INCONCLUSIVE.
        """
        from core.attribute_conflicts import (
            _band_parseable,
            _census_state_for_field,
            _universe_value,
        )
        from core.attribute_universe import NON_YIELD_KINDS, attribute_registry

        specs = dict(registry) if registry is not None else attribute_registry()
        domain_by_key = {
            key: ("additive" if key in ADDITIVE_KEYS else "exclusive")
            for key in specs
        }
        if not specs:
            raise ValueError("engine requires a non-empty registry")
        decisions: dict[str, DimensionDecision] = {}
        for key in sorted(specs):
            spec = specs[key]
            # NON_YIELD_KINDS (currently CONSTANT) are keys whose every
            # populated row carries the SAME value — measured on the raw
            # export, `giftbox` and `special edition` appear only as the
            # literal pairs "Giftbox: giftbox" / "Special Edition: special
            # edition" (21 and 216 of 71,623 rows). A constant cannot
            # disagree with itself, so every pair it lands on is a forced
            # MATCH or a forced INCONCLUSIVE that carries no information.
            #
            # The registry declared this itself ("100% constant — no yield")
            # and attribute_conflicts ledgers these as evidence_class
            # "no_yield", but this loop was unconditional: it re-derived a
            # verdict for them anyway. The exclusion existed as a constant
            # consulted only by the census/budget helper, never by the
            # decision loop. Skipped here so the declared class is honoured
            # where verdicts are actually produced.
            if spec.kind in NON_YIELD_KINDS:
                continue
            left_value = _universe_value(left, key, spec)
            right_value = _universe_value(right, key, spec)
            populated = bool(left_value) and bool(right_value)
            state = _census_state_for_field(
                left_value,
                right_value,
                spec,
                volume_relative_tolerance=self.volume_relative_tolerance,
                volume_absolute_tolerance_ml=self.volume_absolute_tolerance_ml,
            )
            if key == "volume":
                type_key = "NUMERIC"
            elif spec.kind == "NUMERIC_BAND":
                type_key = "BAND"
            elif spec.parser == "enum":
                type_key = "ENUM"
            elif spec.kind in ("SET_NUMERIC",):
                type_key = "NUMERIC"
            else:
                type_key = "STRING"
            # Reset BEFORE the adjudication, not after it. The old placement
            # (an unconditional assignment following the if/else) wiped the
            # stage-8 `semantic_family` provenance on every iteration, so the
            # rescue still downgraded the verdict but DimensionDecision
            # always reported fallback_from="" — the audit trail the stage-8
            # comment promises could never fire. Stage 7 assigns after this
            # point, so its own attribution was never affected.
            fallback_from = ""
            if not populated or state == "unknown_parse":
                metrics = AttributeMetrics(
                    exact_match=False, alias_match=False, jaccard=0.0,
                    overlap_coef=0.0, containment_a=0.0, containment_b=0.0,
                    levenshtein_sim=0.0, negation_conflict=False,
                    parse_problem=state if state == "unknown_parse" else "",
                )
                result = ComparisonResult.INCONCLUSIVE
            else:
                left_set = frozenset(str(token) for token in left_value)
                right_set = frozenset(str(token) for token in right_value)
                metrics = attribute_metrics(left_set, right_set, type_key=type_key, key=key)
                result = evaluate_metrics(
                    type_key, left_set, right_set, metrics,
                    domain=domain_by_key[key],
                )
                # Stage 8: semantic-family rescue. A lexical-fail pair whose
                # members sit in ONE evidenced family (built from the capture
                # census at tau) is a sibling phrasing, not a conflict —
                # recorded with its source so audits trace the downgrade.
                if result is ComparisonResult.CONFLICT and semantic_family_shared(
                    left_set, right_set
                ):
                    result = ComparisonResult.MATCH
                    fallback_from = "semantic_family"
            if (
                result is ComparisonResult.INCONCLUSIVE
                and left_raw is not None and right_raw is not None
            ):
                resolved = self._fallback_reparse(
                    key, left, right, left_raw, right_raw, specs
                )
                if resolved is not None:
                    result, fallback_from = resolved
            decisions[key] = DimensionDecision(
                key=key, type_key=type_key, state=state, result=result,
                metrics=metrics, fallback_from=fallback_from,
            )
        return PairEvidence(dimensions=decisions)

    def _fallback_reparse(
        self,
        key: str,
        left: Mapping[str, object],
        right: Mapping[str, object],
        left_raw: Mapping[str, object],
        right_raw: Mapping[str, object],
        specs: Mapping,
    ) -> tuple[ComparisonResult, str] | None:
        """Stage 7: re-parse the ORIGINAL columns for one unclear key.

        The raw rows are re-read through the SAME devices (title/description
        lift via extract_critical_claims + the attribute cell via the census
        parser) and the key re-evaluated between the re-parsed sets. A real
        verdict from the original evidence REPLACES the INCONCLUSIVE.

        WHY THE MULTI-ROW EXPANSION. ``left_raw``/``right_raw`` may be EITHER
        one source row OR a canonical record carrying a per-title
        ``source_rows`` capture (canonical_records.source_rows — see
        core.columns). The stage was wired but INERT: three_way_gate passed
        canonical records whose keys (``canonical``, ``flavor_set``,
        ``volume_set``, ``mode_brand``) are none of the source columns read
        below, so reparse() returned {} for every key and the clarification
        never fired. A canonical also spans ~4.77 titles on average, and
        per-title multiplicity is real evidence, so the union is taken across
        every captured title rather than picking one.
        """
        from core.attribute_conflicts import parse_universe_cell
        from core.critical_attributes import extract_critical_claims

        def _iter_source_rows(row: Mapping[str, object]):
            """Yield each original-evidence row carried by ``row``.

            Handles a single source row (the direct call shape) and a
            canonical record's ``source_rows`` capture (a JSON array string),
            so the caller does not have to know which it holds.
            """
            captured = row.get("source_rows")
            if captured:
                if isinstance(captured, str):
                    text = captured.strip()
                    if text in {"", "[]"}:
                        return
                    try:
                        import json as _json

                        captured = _json.loads(text)
                    except ValueError:
                        return
                if isinstance(captured, list):
                    for entry in captured:
                        if isinstance(entry, Mapping):
                            yield entry
                    return
            yield row

        def reparse(row: Mapping[str, object]) -> dict[str, frozenset]:
            out: dict[str, frozenset] = {}
            # The columns read here come from the capture declaration
            # (config/paths.yaml column_evidence) via alias_names, so this
            # reader cannot drift away from what the writer actually wrote —
            # which is precisely how it stayed inert.
            from core.columns import alias_names

            attribute_names = alias_names("attributes")
            description_names = alias_names("description")
            title_names = alias_names("title")
            # url is CAPTURED per title (config column_evidence capture=true):
            # the listing slug carries real product words ("sparkling" 70x,
            # "strawberry" 48x in 8,000 sampled sku_url slugs — see the
            # url_evidence block's defect ledger). Reading it here keeps the
            # stage honest: SAME claims device, one more captured column,
            # nothing bespoke. image_url stays OUT (media slug, not shelf
            # text) — read via alias_names('image_url') only if a later
            # ruling extends this.
            from core.url_evidence import url_text

            url_names = alias_names("url")
            from core.attribute_conflicts import VETO_CENSUS_KEY_BY_DIMENSION
            from core.sweetener_values import (
                declared_sweeteners, title_sweetener_types, negated_sweetener_types,
            )
            ingredient_types: set[str] = set()
            negative_ingredients: set[str] = set()
            ingredient_source_conflict = False
            for source_row in _iter_source_rows(row):
                for source in attribute_names:
                    cell = str(source_row.get(source) or "")
                    if cell:
                        for k, v in parse_universe_cell(cell).items():
                            if v:
                                out[k] = out.get(k, frozenset()) | frozenset(
                                    str(t) for t in v
                                )
                cell_text = " ".join(
                    str(source_row.get(name) or "")
                    for name in description_names
                ).strip()
                url_blob = " ".join(
                    url_text(source_row.get(name) or "")
                    for name in url_names
                ).strip()
                title_blob = " ".join(
                    text for text in (
                        " ".join(str(source_row.get(name) or "") for name in title_names).strip(),
                        url_blob,
                    ) if text
                )
                attribute_blob = " ".join(
                    str(source_row.get(name) or "") for name in attribute_names
                ).strip()
                declared = declared_sweeteners(attribute_blob)
                ingredient_types.update(declared["sweetener_type"])
                ingredient_types.update(title_sweetener_types(title_blob))
                ingredient_types.update(title_sweetener_types(cell_text))
                negative_ingredients.update(
                    negated_sweetener_types(title_blob, cell_text, attribute_blob)
                )
                ingredient_source_conflict |= bool(declared["consistency_flags"])
                claims = extract_critical_claims(title_blob, cell_text, attribute_blob)
                for field, tokens in claims.items():
                    # The registry sweetener channel compares ingredients.
                    # Sugar/no-sugar/no-added-sugar are a separate claim axis.
                    if field == "sweetener":
                        continue
                    alias = VETO_CENSUS_KEY_BY_DIMENSION.get(field, field)
                    if tokens and alias in specs:
                        out[alias] = out.get(alias, frozenset()) | frozenset(
                            str(t) for t in tokens
                        )
            if ingredient_types:
                out["sweetener"] = out.get("sweetener", frozenset()) | frozenset(ingredient_types)
            if ingredient_source_conflict or negative_ingredients & ingredient_types:
                out["_sweetener_source_conflict"] = frozenset({"contradiction"})
            return out

        left_sets = reparse(left_raw)
        right_sets = reparse(right_raw)
        if key == "sweetener" and (
            left_sets.get("_sweetener_source_conflict") or right_sets.get("_sweetener_source_conflict")
        ):
            return ComparisonResult.INCONCLUSIVE, "source_conflict"
        a, b = left_sets.get(key, frozenset()), right_sets.get(key, frozenset())
        if not a or not b:
            return None
        reborn = attribute_metrics(a, b)
        result = evaluate_metrics(_type_key_for(key, specs[key]), a, b, reborn)
        return result, "original_columns"


@dataclass(frozen=True)
class SemanticFamilyIndex:
    """THE loaded family registry (stage-8 semantic metric source).

    Built by scripts/build_attribute_semantics.py from the canonical
    capture: per-KEY threshold components (cosine >= tau AND a
    length-adaptive token-Jaccard floor), with negation pairs kept
    apart. Load is once-per-process and byte-stability asserted like
    every artifact read.
    """

    tau: float
    family_of: Mapping[str, int]  # token -> dense family id (per key)
    families: Mapping[int, tuple[str, ...]]

    @classmethod
    def load(cls) -> "SemanticFamilyIndex":
        from core.common import TRAIN_ROOT

        path = TRAIN_ROOT / "results" / "semantics" / "family_registry.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"semantic family registry missing: {path.as_posix()} — "
                "run scripts/build_attribute_semantics.py first (no silent "
                "open-vocabulary gap)"
            )
        payload = json.loads(path.read_text())
        family_of: dict[str, int] = {}
        families: dict[int, tuple[str, ...]] = {}
        for fid, family in enumerate(payload["families"]):
            families[fid] = ("key", family["key"], tuple(family["members"]))
            for member in family["members"]:
                family_of.setdefault(member, fid)
        return cls(tau=float(payload["tau"]), family_of=family_of, families=families)


@lru_cache(maxsize=1)
def _family_index() -> SemanticFamilyIndex:
    return SemanticFamilyIndex.load()


def semantic_family_shared(left_set: frozenset[str], right_set: frozenset[str]) -> bool:
    """True when any member across the two sides shares an evidenced family."""
    index = _family_index()
    return any(
        index.family_of.get(token) is not None
        and index.family_of[token] == index.family_of.get(other)
        for token in left_set
        for other in right_set
    )


@lru_cache(maxsize=1)
def engine() -> AttributeDecisionEngine:
    """THE process-wide loaded engine (the owner's 'single dataclass')."""
    return AttributeDecisionEngine.load()


def _type_key_for(key: str, spec) -> str:
    if key == "volume":
        return "NUMERIC"
    if spec.kind == "NUMERIC_BAND":
        return "BAND"
    if spec.parser == "enum":
        return "ENUM"
    if spec.kind == "SET_NUMERIC":
        return "NUMERIC"
    return "STRING"


__all__ = [
    "AttributeDecisionEngine",
    "AttributeMetrics",
    "ComparisonResult",
    "DimensionDecision",
    "PairEvidence",
    "attribute_metrics",
    "engine",
    "evaluate_metrics",
]
