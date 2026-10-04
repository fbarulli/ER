"""src/core/audit_guard.py — fail-closed guards for measurement audits.

An extraction audit can report a number that means nothing while looking
healthy. Three distinct failure modes, each with its own guard:

  assert_vocabulary_overlap  the audit's extraction vocabulary must
                             actually occur in the corpus it scores. An
                             empty or mismatched vocabulary measures the
                             absence of the vocabulary, not the presence of
                             the attribute (the sweetening-harness artifact).
  self_comparison_control    a known-positive control must score its
                             expected result. A listing compared with
                             itself, or an extractor run twice on one
                             input, must agree — a broken join or a
                             comparator that always returns one answer is
                             caught before it produces a rate.
  assert_not_degenerate      an aggregate rate must not land on exactly
                             0% or 100%. Those extremes are almost always
                             harness artifacts (a float/int key mismatch
                             that joins nothing, a self-comparison that
                             trivially matches everything), not real
                             extraction performance. A count whose 0 (or
                             full) value is the INTENDED outcome must be
                             guarded as a rate, not forced through this.

Every guard raises AuditGuardError. Fail-closed: a caller must not publish
an audit report when a guard fires. The guards are pure (no I/O, no global
state) so they wrap any audit script or test.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


class AuditGuardError(RuntimeError):
    """A measurement audit failed a fail-closed precondition/postcondition."""


@dataclass(frozen=True)
class DimensionGuardResult:
    """Per-dimension guard outcome.

    ``passed`` is False only for a hard harness failure (empty/mismatched
    vocabulary, or a self-comparison that reports a row different from
    itself). ``unmeasured`` is True when the dimension is degenerate in the
    audited population (exactly 0% or 100% coverage) — not a harness bug, but
    a dimension the population cannot score, so it must be excluded from any
    reported rate rather than silently published.
    """

    name: str
    passed: bool
    unmeasured: bool
    detail: str = ""


def _clean_terms(vocabulary: Iterable[Any]) -> set[str]:
    return {str(term).strip().lower() for term in vocabulary if str(term).strip()}


def assert_vocabulary_overlap(
    vocabulary: Iterable[Any],
    texts: Iterable[Any],
    *,
    label: str,
    min_terms: int = 1,
) -> set[str]:
    """Assert the audit vocabulary occurs in the corpus it is scored against.

    Returns the set of vocabulary terms found. Raises when fewer than
    ``min_terms`` distinct terms occur — the "exactly 0% overlap" case: a
    vocabulary that matches nothing cannot measure extraction.
    """
    terms = _clean_terms(vocabulary)
    if not terms:
        raise AuditGuardError(f"[{label}] vocabulary is empty")
    corpus = [str(text).lower() for text in texts]
    if not corpus:
        raise AuditGuardError(f"[{label}] corpus is empty")
    matched = {term for term in terms if any(term in text for text in corpus)}
    if len(matched) < min_terms:
        raise AuditGuardError(
            f"[{label}] vocabulary does not overlap corpus: "
            f"{len(matched)}/{len(terms)} terms matched"
        )
    return matched


def self_comparison_control(
    compare: Callable[[Any, Any], Any],
    samples: Iterable[Any],
    *,
    label: str,
    expected: Any = True,
    normalize: Callable[[Any], Any] | None = None,
) -> int:
    """Assert a known-positive control scores its expected result.

    ``compare(a, b)`` is the audit's comparator; it is run on each sample
    against itself. Every result must equal ``expected`` (after ``normalize``
    when supplied). An empty sample set is itself a failure — a control that
    never ran proves nothing. Returns the number of controls checked.
    """
    materialized = list(samples)
    if not materialized:
        raise AuditGuardError(f"[{label}] self-comparison control has no samples")

    def norm(value: Any) -> Any:
        return normalize(value) if normalize is not None else value

    bad: list[tuple[int, Any]] = []
    for index, sample in enumerate(materialized):
        got = norm(compare(sample, sample))
        if got != expected:
            bad.append((index, got))
    if bad:
        raise AuditGuardError(
            f"[{label}] self-comparison control failed on {len(bad)}/"
            f"{len(materialized)} samples (expected {expected!r}): {bad[:3]}"
        )
    return len(materialized)


def assert_not_degenerate(
    name: str,
    value: float,
    *,
    total: int | None = None,
    label: str = "",
) -> float:
    """Assert a metric is not exactly 0% or exactly 100%.

    With ``total``, ``value`` is a count: 0 and ``total`` are rejected.
    Without it, ``value`` is a fraction: 0.0 and 1.0 are rejected. Returns
    the fraction.
    """
    if total is not None:
        if total <= 0:
            raise AuditGuardError(
                f"[{label}] {name}: total must be positive, got {total}"
            )
        if value == 0 or value == total:
            raise AuditGuardError(f"[{label}] {name} is degenerate: {value}/{total}")
        return value / total
    fraction = float(value)
    if fraction == 0.0 or fraction == 1.0:
        raise AuditGuardError(f"[{label}] {name} is degenerate: {fraction}")
    return fraction


def assert_metrics_not_degenerate(
    metrics: Mapping[str, float | tuple[float, int]],
    *,
    label: str = "",
) -> None:
    """Apply :func:`assert_not_degenerate` to a mapping of name -> value.

    A value is either a fraction, or a ``(count, total)`` pair.
    """
    for name, spec in metrics.items():
        if isinstance(spec, tuple):
            count, total = spec
            assert_not_degenerate(name, count, total=total, label=label)
        else:
            assert_not_degenerate(name, spec, label=label)


def guard_dimension(
    name: str,
    *,
    values: Iterable[Any],
    source_texts: Iterable[Any],
    self_compare: Callable[[Any, Any], Any],
    self_samples: Iterable[Any],
    populated: int,
    total: int,
    self_expected: Any = True,
    label: str = "attribute",
) -> DimensionGuardResult:
    """Run all three guards for one dimension and return the outcome.

    Hard failures (returned with ``passed=False``) abort the audit: an empty
    or non-overlapping value vocabulary, or a self-comparison that does not
    score its expected result. A degenerate population (0%/100% coverage) is
    returned with ``unmeasured=True`` instead — the dimension cannot be
    scored here, so the caller must exclude it rather than report it.
    """
    scoped = f"{label}:{name}"
    try:
        assert_vocabulary_overlap(values, source_texts, label=scoped)
    except AuditGuardError as exc:
        return DimensionGuardResult(name, passed=False, unmeasured=False, detail=str(exc))
    try:
        self_comparison_control(
            self_compare, self_samples, label=scoped, expected=self_expected
        )
    except AuditGuardError as exc:
        return DimensionGuardResult(name, passed=False, unmeasured=False, detail=str(exc))
    try:
        assert_not_degenerate("coverage", populated, total=total, label=scoped)
    except AuditGuardError as exc:
        return DimensionGuardResult(name, passed=True, unmeasured=True, detail=str(exc))
    return DimensionGuardResult(name, passed=True, unmeasured=False)


def guard_dimensions(
    specs: Sequence[Mapping[str, Any]],
    *,
    label: str = "attribute",
) -> list[DimensionGuardResult]:
    """Run :func:`guard_dimension` for every spec; raise on hard failures.

    Every spec is checked before any failure is raised, so one bad dimension
    does not hide the others. Returns the full result list (including
    degenerate/unmeasured dimensions) when no hard failure occurred.
    """
    results = [guard_dimension(label=label, **spec) for spec in specs]
    failed = [result for result in results if not result.passed]
    if failed:
        detail = "\n  ".join(f"{result.name}: {result.detail}" for result in failed)
        raise AuditGuardError(
            f"[{label}] {len(failed)}/{len(results)} dimension guard(s) failed:\n  {detail}"
        )
    return results


# The attribute self-comparison sample size (a known-positive control should
# exercise a meaningful slice, not the full corpus). One constant, not a
# per-script magic number: audit_identity_dimensions and
# audit_attribute_readings build their guard specs through
# :func:`attribute_dimension_guard_specs`.
ATTRIBUTE_SELF_SAMPLE = 200


def attribute_dimension_guard_specs(
    dimensions: Iterable[str],
    *,
    values: Mapping[str, Iterable[str]],
    source_texts: Iterable[str],
    populated: Mapping[str, int],
    total: int,
    evaluate_status: Callable[[Any, Any, str], str],
    evidence_samples: Sequence[Any],
) -> list[dict[str, Any]]:
    """Build the per-dimension guard specs shared by the attribute audits.

    ``evaluate_status(left, right, dimension) -> status`` is the audit's
    dimension comparator (typically ``core.product_dimensions
    .evaluate_dimensions(a, b)[dim]["status"])``, injected so this stays a
    pure no-import helper. A self-comparison may legitimately be "equal",
    "unknown" (no data), or "unparsed" (a real parser gap, reported
    separately); only "different" — a row disagreeing with itself — is a
    broken comparator and a hard failure.

    Returns spec dicts ready for :func:`guard_dimensions`, plus the
    parse-gap census over the same sample (dimension -> rows reported
    "unparsed" against themselves), so every audit reports parser gaps the
    same way instead of two drifted copies.
    """
    sample = list(evidence_samples)[:ATTRIBUTE_SELF_SAMPLE]
    attr_texts = list(source_texts)
    specs: list[dict[str, Any]] = []
    for name in sorted(dimensions):
        specs.append({
            "name": name,
            "self_compare": (
                lambda a, b, _n=name: evaluate_status(a, b, _n) != "different"
            ),
            "self_samples": sample,
            "values": set(values.get(name, ())),
            "source_texts": attr_texts,
            "populated": populated.get(name, 0),
            "total": total,
        })
    return specs


def self_comparison_parse_gaps(
    dimensions: Iterable[str],
    *,
    evaluate_status: Callable[[Any, Any, str], str],
    evidence_samples: Sequence[Any],
) -> dict[str, int]:
    """Census rows a dimension comparator cannot parse back from itself.

    A dimension whose own value cannot be parsed back (e.g. Caffeine
    "200+ mg") reports "unparsed" against itself — reported, never silently
    absorbed. Duplicates zero logic: injects the same comparator as
    :func:`attribute_dimension_guard_specs`.
    """
    sample = list(evidence_samples)[:ATTRIBUTE_SELF_SAMPLE]
    gaps: dict[str, int] = {}
    for name in sorted(dimensions):
        count = sum(
            evaluate_status(evidence, evidence, name) == "unparsed"
            for evidence in sample
        )
        if count:
            gaps[name] = count
    return gaps
