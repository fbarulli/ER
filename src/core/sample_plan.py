"""src/core/sample_plan.py — how many validation samples a measurement NEEDS.

WHY THIS MODULE EXISTS
----------------------
The validation population was sized by the component-fold deal, never by the
question "how many samples does a measurement need to be representative?".
This module answers that question ONCE, from declared TARGETS
(``config/sampling.yaml`` -> :class:`core.schemas.SamplingPlanSpec`) and the
REAL prevalences of the dataset's slices, so no lane has to re-derive the
arithmetic (and no lane can quietly disagree about it).

WHAT IT IS NOT
--------------
It PLANS; it does not gate. No data is read or checked here — the census below
turns samples into supports, and the calculator turns supports + targets into
required counts. There is no existence/size/validity verdict anywhere.

A SLICE IS A FILTER OVER SAMPLES
--------------------------------
One row IS one sample (a scored pair). A slice is an attribute column whose
values select subsets: a value's ``support`` is the number of samples that
carry it; its ``share`` is ``support / population``. A target N for a subgroup
is therefore a target for the WHOLE validation: to carry ``n`` samples of a
subgroup that is ``share`` of the population you need ``n / share`` samples in
total. That is what makes a small subgroup the binding constraint.

THE MATH (all stated, none invented)
------------------------------------
1. Proportion estimate to a CI half-width ``h`` at confidence ``c``::

       n = z^2 * p * (1 - p) / h^2,   z = Phi^-1(1 - (1 - c)/2)

   ``p`` defaults to the configured worst case 0.5. Rounded to the nearest
   whole sample (the reference table's convention).

2. Paired McNemar improvement (the SAME validation samples before and after a
   fix): under pure improvement the only discordant pairs are the improvements,
   so psi = delta and::

       N = ( z_{alpha/2} + z_{power} * sqrt(1 - delta) )^2 / delta

   ``mde_paired`` inverts this for the MINIMUM DETECTABLE EFFECT at N;
   ``required_paired_n`` evaluates it for the N at a target effect.

3. Two-proportion / CI-MDE variant (independent samples), ``n`` per group::

       n = ( z_{alpha/2} * sqrt(2 * pbar * (1 - pbar))
             + z_{power} * sqrt(p1(1-p1) + p2(1-p2)) )^2 / (p2 - p1)^2

   ``mde_two_proportion`` inverts it at N.

REFERENCE NUMBERS (reproduced by the calculator, pinned by tests)
-----------------------------------------------------------------
``mde_paired`` (alpha 0.05, power 0.80): N=28 -> 25.75pp, N=100 -> 7.67pp,
N=300 -> 2.60pp, N=400 -> 1.95pp, N=1000 -> 0.78pp, N=2000 -> 0.39pp,
N=5000 -> 0.16pp. ``required_proportion_n``: (p=0.50,h=0.05) -> 384,
(p=0.50,h=0.10) -> 96, (p=0.95,h=0.05) -> 73, (p=0.95,h=0.10) -> 18.

WHERE THE "28 PAIRS" FIGURE COMES FROM
--------------------------------------
28 is the LIVE/REDUCED SCORED census in ``data/final_validation.csv`` (4
positives / 24 negatives) — the scored half after the fold deal and the
straddle policy, NOT the size of the labelled population. The full labelled
census in ``data/labeled_pairs.csv`` is 41 pairs. Both are far below the N a
2pp improvement needs, so the binding constraint is the labelled census
itself, not the split fraction — no split fraction can conjure labelled pairs
that do not exist.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from statistics import NormalDist
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.schemas import SampleValueParse, SamplingPlanSpec

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from core.validation_set import ValidationSelection

#: The declared sample-plan contract lives beside the other config documents.
CONFIG_NAME = "sampling.yaml"

#: The FLEX validation_size unit: a fraction of the component graph (which the
#: fold deal rounds to whole trailing quarters) or a whole-sample row count.
ValidationSizeUnit = Literal["fraction", "rows"]

#: Bisection bounds for an MDE search, in proportion units. The upper bound is
#: below 1 because a detectable improvement cannot exceed the discordant share.
_MDE_LOW = 1e-9
_MDE_HIGH = 0.95
_BISECTION_STEPS = 200


def sampling_plan_config_path() -> Path:
    """The declared sample-plan document (``config/sampling.yaml``)."""
    from core.common import TRAIN_ROOT

    return TRAIN_ROOT / "config" / CONFIG_NAME


def sampling_plan_spec() -> SamplingPlanSpec:
    """Read + validate the sample-plan declaration through the ONE home."""
    from core.common import load_validated_yaml

    return load_validated_yaml(
        sampling_plan_config_path(), SamplingPlanSpec, label="Sample-plan declaration"
    )


def slice_parses(spec: SamplingPlanSpec) -> dict[str, SampleValueParse]:
    """The declared slice name -> value-parse map a census/builder consumes."""
    return {name: declared.parse for name, declared in spec.slices.items()}


# ── domain models ───────────────────────────────────────────────────────────
class SubgroupSupport(BaseModel):
    """One (slice, value) subgroup: the filter's support and its population.

    ``support`` is the number of samples whose slice parse yields ``value``;
    ``population`` is the number of samples the filter range covers.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    slice: str = Field(min_length=1)
    value: str
    support: int = Field(ge=1)
    population: int = Field(ge=1)

    @model_validator(mode="after")
    def _support_within_population(self) -> SubgroupSupport:
        if self.support > self.population:
            raise ValueError(
                f"subgroup {self.slice}={self.value!r} has support {self.support} "
                f"above its population {self.population}"
            )
        return self

    @property
    def share(self) -> float:
        """The subgroup's prevalence: ``support / population``."""
        return self.support / self.population


class SliceCensus(BaseModel):
    """One slice's measured census over a sample population.

    ``populated`` is the number of samples carrying ANY value of the slice (a
    sample may carry several values of a set-valued slice, so ``populated`` is
    NOT the sum of the value supports); ``values`` holds one
    :class:`SubgroupSupport` per observed value.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    slice: str = Field(min_length=1)
    population: int = Field(ge=1)
    populated: int = Field(ge=0)
    values: tuple[SubgroupSupport, ...]

    @model_validator(mode="after")
    def _accounting(self) -> SliceCensus:
        if self.populated > self.population:
            raise ValueError(
                f"slice {self.slice!r} populated {self.populated} exceeds "
                f"population {self.population}"
            )
        if any(
            value.population != self.population or value.slice != self.slice
            for value in self.values
        ):
            raise ValueError(
                f"slice {self.slice!r} carries a subgroup with a different "
                "slice name or population"
            )
        return self


class SubgroupRequirement(BaseModel):
    """The required TOTAL validation N for one subgroup to be measurable.

    ``per_subgroup_n`` is the number of samples the subgroup itself needs (to
    satisfy BOTH the CI target and the target effect); ``required_n`` is the
    total validation size that delivers them (``per_subgroup_n / share``).
    ``measurable`` is False when the population carries fewer than
    ``min_subgroup_support`` samples of the value — nothing can measure it.
    ``needs_oversampling`` is True when PROPORTIONAL sampling at
    ``max_realistic_n`` would leave it below the floor, i.e. it can only be
    measured by stratifying/oversampling the validation toward it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    slice: str
    value: str
    support: int = Field(ge=1)
    share: float = Field(gt=0.0, le=1.0)
    per_subgroup_n: int = Field(ge=1)
    required_n: int = Field(ge=1)
    measurable: bool
    needs_oversampling: bool


class SlicePlan(BaseModel):
    """One slice's plan: its whole-field requirement and every value's floor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    slice: str
    population: int
    populated: int
    #: One requirement per observed value; the builder's coverage targets.
    requirements: tuple[SubgroupRequirement, ...]
    #: Requirement for the slice as a whole (the ``populated`` filter).
    populated_required_n: int | None
    #: The smallest meaningful (support >= floor) value's requirement, if any.
    smallest_meaningful: SubgroupRequirement | None

    @property
    def declared_values(self) -> int:
        return len(self.requirements)


class SamplePlanReport(BaseModel):
    """The emitted plan: the binding floor, the per-slice table, the recommendation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    population: int = Field(ge=1)
    confidence: float
    ci_half_width: float
    alpha: float
    power: float
    target_effect: float
    min_subgroup_support: int = Field(ge=1)
    per_subgroup_n: int
    mde_at_population: float
    current_validation: int | None
    mde_at_current_validation: float | None
    labeled_census: int | None
    mde_at_labeled_census: float | None
    #: The smallest meaningful subgroup's requirement (None ⇒ none is meaningful).
    binding: SubgroupRequirement | None
    slices: tuple[SlicePlan, ...]
    recommended_n: int
    recommended_validation_size: float | int
    validation_size_unit: ValidationSizeUnit
    validation_size_reachable: bool | None
    notes: tuple[str, ...]

    def measurable_targets(self) -> tuple[CoverageTarget, ...]:
        """Every measurable subgroup's coverage demand for the validation set.

        The demand is ``per_subgroup_n`` (the samples the subgroup itself needs
        to satisfy both targets); the total size that delivers them is
        ``recommended_n``. Unmeasurable subgroups are excluded — they are
        reported, never demanded.
        """
        return tuple(
            CoverageTarget(slice=item.slice, value=item.value, demand=item.per_subgroup_n)
            for plan in self.slices
            for item in plan.requirements
            if item.measurable
        )

    def unmeasurable(self) -> tuple[SubgroupRequirement, ...]:
        """Subgroups no realistic validation N can make measurable."""
        return tuple(
            item
            for plan in self.slices
            for item in plan.requirements
            if not item.measurable
        )


class CoverageTarget(BaseModel):
    """One measurable subgroup's demand: samples the validation must carry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    slice: str = Field(min_length=1)
    value: str
    demand: int = Field(ge=1)


# ── the filter-over-samples census ──────────────────────────────────────────
class SubgroupCensus:
    """Turns samples into per-slice supports: a slice/attribute IS a filter.

    The class owns the value-reading rule (a ``scalar`` cell, or a stored
    ``set_literal`` read as a BAG of values) so every consumer filters the same
    way, and it never reads I/O itself — it consumes already-loaded samples, so
    a lane can trace a single sample through the same filter.
    """

    def __init__(
        self,
        slices: Mapping[str, SampleValueParse],
        cells: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self._slices = dict(slices)
        self._cells = {name: tuple(components) for name, components in (cells or {}).items()}

    @classmethod
    def from_config(cls, spec: SamplingPlanSpec | None = None) -> SubgroupCensus:
        """Build the census from the validated slice declaration."""
        resolved = sampling_plan_spec() if spec is None else spec
        return cls(slice_parses(resolved), resolved.cells)

    @staticmethod
    def values(raw: object, parse: SampleValueParse) -> tuple[str, ...]:
        """The values a cell carries under one parse rule (empty ⇒ not populated).

        ``scalar`` lowercases the trimmed cell. ``set_literal`` evaluates the
        stored literal and returns its deduped, lowercased bag; an unparseable
        cell yields nothing rather than a fabricated value.
        """
        if raw is None:
            return ()
        if parse == "scalar":
            text = str(raw).strip().lower()
            return (text,) if text else ()
        try:
            parsed = ast.literal_eval(str(raw))
        except (ValueError, SyntaxError):
            return ()
        items = parsed if isinstance(parsed, (list, tuple, set)) else [parsed]
        return tuple(sorted({str(item).strip().lower() for item in items if str(item).strip()}))

    def census(
        self, samples: Iterable[Mapping[str, object]], *, columns: Sequence[str] | None = None
    ) -> tuple[SliceCensus, ...]:
        """Measure every declared slice over ``samples``.

        ``columns`` optionally restricts the declaration to a subset; a column
        absent from every sample raises (a declared slice whose column is not in
        the frame is a wiring error, not an empty slice).
        """
        rows = list(samples)
        population = len(rows)
        if population == 0:
            raise ValueError("cannot census an empty sample population")
        present = set().union(*(row.keys() for row in rows))
        selected = list(self._slices) if columns is None else list(columns)
        censuses: list[SliceCensus] = []
        for name in selected:
            if name not in self._slices:
                raise KeyError(
                    f"slice {name!r} is not declared; declared slices: "
                    f"{sorted(self._slices)}"
                )
            censuses.append(self._census_slice(name, self._slices[name], rows, population))
        for cell, components in self._cells.items():
            # A cell axis absent from this frame is skipped, not an error: the
            # entity census has no difficulty column, the pair census does.
            if set(components) <= present:
                censuses.append(self._census_cell(cell, components, rows, population))
        return tuple(censuses)

    @staticmethod
    def _cell_values(
        row: Mapping[str, object],
        components: Sequence[str],
        parses: Mapping[str, SampleValueParse],
    ) -> tuple[str, ...]:
        """One sample's composite cell values: the cross of each component's bag.

        A component with no value on the sample contributes nothing, so a cell
        only exists for a sample that carries EVERY component. Values join as
        ``a=v|b=w`` so the components of a cell stay traceable.
        """
        cells: list[str] = [""]
        for name in components:
            values = SubgroupCensus.values(row.get(name), parses[name])
            if not values:
                return ()
            cells = [f"{prefix}{'|' if prefix else ''}{name}={value}"
                     for prefix in cells for value in values]
        return tuple(sorted(cells))

    def _census_cell(
        self,
        cell: str,
        components: Sequence[str],
        rows: Sequence[Mapping[str, object]],
        population: int,
    ) -> SliceCensus:
        if rows and not all(name in rows[0] for name in components):
            raise KeyError(
                f"composite cell {cell!r} references a column absent from the "
                f"sample frame: {[n for n in components if n not in rows[0]]}"
            )
        counts: dict[str, int] = {}
        populated = 0
        for row in rows:
            values = self._cell_values(row, components, self._slices)
            if values:
                populated += 1
            for value in values:
                counts[value] = counts.get(value, 0) + 1
        supports = tuple(
            SubgroupSupport(slice=cell, value=value, support=count, population=population)
            for value, count in sorted(counts.items())
        )
        return SliceCensus(
            slice=cell, population=population, populated=populated, values=supports
        )

    @staticmethod
    def _census_slice(
        name: str,
        parse: SampleValueParse,
        rows: Sequence[Mapping[str, object]],
        population: int,
    ) -> SliceCensus:
        if rows and name not in rows[0]:
            raise KeyError(
                f"slice column {name!r} is not present in the sample frame"
            )
        counts: dict[str, int] = {}
        populated = 0
        for row in rows:
            values = SubgroupCensus.values(row.get(name), parse)
            if values:
                populated += 1
            for value in values:
                counts[value] = counts.get(value, 0) + 1
        supports = tuple(
            SubgroupSupport(slice=name, value=value, support=count, population=population)
            for value, count in sorted(counts.items())
        )
        return SliceCensus(
            slice=name, population=population, populated=populated, values=supports
        )


# ── the calculator / planner ────────────────────────────────────────────────
def _invert_mde(
    required: Callable[[float], float], target_n: int, *, high: float = _MDE_HIGH
) -> float:
    """The effect at which ``required(effect) == target_n``, by bisection.

    ``required`` decreases in the effect, so a target below its value at the
    cap means even the cap is too small to reach — the cap is returned.
    """
    if required(high) > target_n:
        return high
    low = _MDE_LOW
    for _ in range(_BISECTION_STEPS):
        mid = (low + high) / 2.0
        if required(mid) > target_n:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


class SamplePlan:
    """The sample-plan calculator: targets in, required counts out.

    Constructed from the validated config SSOT by :meth:`from_config`; the
    target methods are pure functions of an injected spec, so a caller can
    pin the math without touching config.
    """

    def __init__(self, spec: SamplingPlanSpec) -> None:
        self._spec = spec

    @classmethod
    def from_config(cls) -> SamplePlan:
        """Build from the validated ``config/sampling.yaml`` SSOT."""
        return cls(sampling_plan_spec())

    @property
    def spec(self) -> SamplingPlanSpec:
        return self._spec

    @property
    def slices(self) -> dict[str, SampleValueParse]:
        """The declared slice name -> value-parse map every filter uses."""
        return slice_parses(self._spec)

    def difficulty_labels_path(self) -> Path | None:
        """The declared canonical per-pair difficulty-labels address, or None.

        The labels are the local-only output of ``training.difficulty``; this is
        the ONE place the address is read, so the source can never drift from
        the declaration.
        """
        definition = self._spec.difficulty_definition
        if definition is None or definition.measured_labels_path is None:
            return None
        from core.common import TRAIN_ROOT

        return TRAIN_ROOT / definition.measured_labels_path

    # -------------------------------------------------------------- z helpers
    @staticmethod
    def _z_tail(tail: float) -> float:
        """The standard-normal quantile at ``1 - tail``."""
        return NormalDist().inv_cdf(1.0 - tail)

    def _z_alpha(self, alpha: float | None) -> float:
        return self._z_tail((self._spec.targets.alpha if alpha is None else alpha) / 2.0)

    def _z_power(self, power: float | None) -> float:
        return self._z_tail(1.0 - (self._spec.targets.power if power is None else power))

    # ------------------------------------------------------------ proportions
    def required_proportion_n(
        self, p: float, *, half_width: float | None = None, confidence: float | None = None
    ) -> int:
        """``n = z^2 p(1-p) / h^2`` for a proportion estimate, rounded.

        ``p`` is the expected proportion; the caller passes the configured
        worst case (0.5) when no measured metric supplies one.
        """
        h = self._spec.targets.ci_half_width if half_width is None else half_width
        c = self._spec.targets.confidence if confidence is None else confidence
        z = self._z_tail((1.0 - c) / 2.0)
        return round(z * z * p * (1.0 - p) / (h * h))

    # ------------------------------------------------------- paired (McNemar)
    def _paired_required_n(self, effect: float, alpha: float | None, power: float | None) -> float:
        """``(z_a + z_b sqrt(1-delta))^2 / delta`` — the McNemar requirement."""
        z_a, z_b = self._z_alpha(alpha), self._z_power(power)
        return (z_a + z_b * math.sqrt(1.0 - effect)) ** 2 / effect

    def required_paired_n(
        self, effect: float, *, alpha: float | None = None, power: float | None = None
    ) -> int:
        """The validation N needed to detect ``effect`` (paired McNemar), rounded up."""
        return math.ceil(self._paired_required_n(effect, alpha, power))

    def mde_paired(
        self, n: int, *, alpha: float | None = None, power: float | None = None
    ) -> float:
        """The MINIMUM DETECTABLE EFFECT at validation N (paired McNemar)."""
        return _invert_mde(lambda effect: self._paired_required_n(effect, alpha, power), n)

    # --------------------------------------------------- two-proportion (CI-MDE)
    def _two_proportion_required_n(
        self, p1: float, p2: float, alpha: float | None, power: float | None
    ) -> float:
        z_a, z_b = self._z_alpha(alpha), self._z_power(power)
        pbar = (p1 + p2) / 2.0
        return (
            z_a * math.sqrt(2.0 * pbar * (1.0 - pbar))
            + z_b * math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
        ) ** 2 / (p2 - p1) ** 2

    def required_two_proportion_n(
        self,
        p1: float,
        p2: float,
        *,
        alpha: float | None = None,
        power: float | None = None,
    ) -> int:
        """N PER GROUP for an independent two-proportion test, rounded up."""
        if p1 == p2:
            raise ValueError("two-proportion sample size needs p1 != p2")
        return math.ceil(self._two_proportion_required_n(p1, p2, alpha, power))

    def mde_two_proportion(
        self,
        n: int,
        *,
        baseline: float | None = None,
        alpha: float | None = None,
        power: float | None = None,
    ) -> float:
        """The MINIMUM DETECTABLE EFFECT at n per group (independent, CI-MDE variant)."""
        base = self._spec.targets.two_proportion_baseline if baseline is None else baseline

        def required(effect: float) -> float:
            if base + effect >= 1.0:
                return math.inf
            return self._two_proportion_required_n(base, base + effect, alpha, power)

        # A rate cannot rise past 1.0, so the search cap is the domain edge.
        return _invert_mde(required, n, high=min(_MDE_HIGH, 0.999 * (1.0 - base)))

    # ------------------------------------------------------------------- plan
    def plan(
        self,
        censuses: Sequence[SliceCensus],
        *,
        labeled_census: int | None = None,
        current_validation: int | None = None,
        component_folds: int | None = None,
    ) -> SamplePlanReport:
        """Turn measured censuses + targets into the required-N plan.

        ``labeled_census``/``current_validation`` are the REAL available pair
        counts (the full labelled population and the live scored census), used
        to say whether the recommendation is reachable and to report the
        detectable effect at both. ``component_folds`` is the fold arity whose
        whole trailing quarters the FLEX ``validation_size`` deals, used to
        express the recommendation in the units that knob takes.
        """
        targets = self._spec.targets
        if not censuses:
            raise ValueError("the sample plan needs at least one slice census")
        # Slices may be measured over different populations (attribute
        # prevalences over the entity catalog; difficulty/composite cells over
        # the labelled pair pool). Each SlicePlan carries its own population;
        # the report's headline is the largest.
        population = max(census.population for census in censuses)

        # Samples ONE subgroup needs to satisfy both targets; share=1 means this
        # is also the whole-population requirement.
        per_subgroup_n = max(
            self.required_paired_n(targets.target_effect),
            self.required_proportion_n(targets.worst_case_proportion),
        )
        slice_plans = tuple(
            self._slice_plan(census, per_subgroup_n) for census in censuses
        )
        binding = self._binding(slice_plans)
        recommended_n = max(per_subgroup_n, binding.required_n if binding else 0)
        size, unit, reachable = self._recommend_size(
            recommended_n, labeled_census, component_folds
        )
        return SamplePlanReport(
            population=population,
            confidence=targets.confidence,
            ci_half_width=targets.ci_half_width,
            alpha=targets.alpha,
            power=targets.power,
            target_effect=targets.target_effect,
            min_subgroup_support=targets.min_subgroup_support,
            per_subgroup_n=per_subgroup_n,
            mde_at_population=self.mde_paired(population),
            current_validation=current_validation,
            mde_at_current_validation=(
                None if current_validation is None else self.mde_paired(current_validation)
            ),
            labeled_census=labeled_census,
            mde_at_labeled_census=(
                None if labeled_census is None else self.mde_paired(labeled_census)
            ),
            binding=binding,
            slices=slice_plans,
            recommended_n=recommended_n,
            recommended_validation_size=size,
            validation_size_unit=unit,
            validation_size_reachable=reachable,
            notes=self._notes(
                binding, recommended_n, labeled_census, current_validation, reachable
            ),
        )

    def _slice_plan(self, census: SliceCensus, per_subgroup_n: int) -> SlicePlan:
        """One slice's whole-field requirement and every value's floor."""
        requirements = tuple(
            self._requirement(value, per_subgroup_n) for value in census.values
        )
        smallest = min(
            (item for item in requirements if item.measurable),
            key=lambda item: (item.share, item.value),
            default=None,
        )
        populated_share = census.populated / census.population
        return SlicePlan(
            slice=census.slice,
            population=census.population,
            populated=census.populated,
            requirements=requirements,
            populated_required_n=(
                None if not populated_share else math.ceil(per_subgroup_n / populated_share)
            ),
            smallest_meaningful=smallest,
        )

    def _requirement(self, value: SubgroupSupport, per_subgroup_n: int) -> SubgroupRequirement:
        """The total-N requirement for one subgroup, and its measurability."""
        share = value.share
        floor = self._spec.targets.min_subgroup_support
        return SubgroupRequirement(
            slice=value.slice,
            value=value.value,
            support=value.support,
            share=share,
            per_subgroup_n=per_subgroup_n,
            required_n=math.ceil(per_subgroup_n / share),
            measurable=value.support >= floor,
            needs_oversampling=share * self._spec.targets.max_realistic_n < floor,
        )

    @staticmethod
    def _binding(slice_plans: Sequence[SlicePlan]) -> SubgroupRequirement | None:
        """The SMALLEST meaningful subgroup: the one that sets the N floor."""
        candidates = [
            plan.smallest_meaningful for plan in slice_plans if plan.smallest_meaningful
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item.required_n, item.value))

    def _recommend_size(
        self,
        required_n: int,
        labeled_census: int | None,
        component_folds: int | None,
    ) -> tuple[float | int, ValidationSizeUnit, bool | None]:
        """The FLEX ``split.validation_size`` recommendation.

        With a labelled census and the fold arity the recommendation is the
        smallest fraction of whole trailing quarters whose deal covers
        ``required_n``; when even the largest legal fraction (all but one
        quarter) cannot, the census — not the split — is the binding constraint
        and the max fraction is returned with ``reachable=False``. Without them
        the raw row count is returned.
        """
        folds = component_folds
        if not labeled_census or not folds or folds < 3:
            return required_n, "rows", None
        max_quarters = folds - 1
        max_fraction = max_quarters / folds
        if required_n > labeled_census * max_fraction:
            return max_fraction, "fraction", False
        pairs_per_quarter = labeled_census / folds
        quarters = min(max_quarters, max(1, math.ceil(required_n / pairs_per_quarter)))
        return quarters / folds, "fraction", True

    def _notes(
        self,
        binding: SubgroupRequirement | None,
        recommended_n: int,
        labeled_census: int | None,
        current_validation: int | None,
        reachable: bool | None,
    ) -> tuple[str, ...]:
        """The provenance a reader must not misread the recommendation without."""
        notes = [
            (
                "'current_validation' is the LIVE/REDUCED scored census "
                "(data/final_validation.csv: 4 pos / 24 neg = 28 pairs), not the "
                "labelled population; the full labelled census (data/labeled_pairs.csv) "
                "is its own number."
            ),
        ]
        if binding is None:
            notes.append(
                "no subgroup clears the support floor, so the whole-population "
                f"requirement ({recommended_n}) is binding."
            )
        else:
            notes.append(
                f"binding constraint = smallest meaningful subgroup "
                f"{binding.slice}={binding.value!r} (support {binding.support}, "
                f"share {binding.share:.6g}) -> {binding.required_n} total samples."
            )
        if labeled_census is not None and current_validation is not None and reachable is False:
            notes.append(
                f"required N {recommended_n} exceeds what the largest legal split "
                f"fraction can draw from the {labeled_census}-pair labelled census "
                f"({current_validation} scored today): the CENSUS is the binding "
                "constraint, not the split fraction — stratified/oversampled "
                "validation is required."
            )
        return tuple(notes)

    def build_validation_set(
        self,
        report: SamplePlanReport,
        pool: Sequence[Mapping[str, object]],
        *,
        id_key: str,
        label_key: str,
        positive_label: str = "1",
        source: str = "",
        difficulty_definition: dict[str, object] | None = None,
        difficulty_source: str | None = None,
    ) -> ValidationSelection:
        """Materialize the validation set the report asks for from a pool.

        The plan owns the DECISION (size + per-subgroup demand); the builder
        (``core.validation_set.ValidationSetBuilder``) owns the selection. This
        is the ONE seam a lane calls to produce the correctly-sized, stratified
        validation set.
        """
        from core.validation_set import ValidationSetBuilder

        return ValidationSetBuilder(slice_parses(self._spec), self._spec.cells).build(
            pool,
            report,
            id_key=id_key,
            label_key=label_key,
            positive_label=positive_label,
            source=source,
            difficulty_definition=difficulty_definition,
            difficulty_source=difficulty_source,
        )


__all__ = [
    "CONFIG_NAME",
    "CoverageTarget",
    "SamplePlan",
    "SamplePlanReport",
    "SliceCensus",
    "SlicePlan",
    "SubgroupCensus",
    "SubgroupRequirement",
    "SubgroupSupport",
    "sampling_plan_config_path",
    "sampling_plan_spec",
    "slice_parses",
]
