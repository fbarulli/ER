"""src/core/validation_set.py — materialize the validation set the plan asks for.

The sample plan (``core.sample_plan.SamplePlan``) owns the DECISION: the size
and the per-subgroup demand. This module owns the SELECTION: from a candidate
pool it greedily picks the samples covering the most unmet demand until every
measurable subgroup reaches its demand or the pool is exhausted, preferring the
under-represented label to keep the classes balanced, then fills to the
requested size. It is the ONE seam a lane calls to produce the correctly-sized,
stratified validation set, and it emits a manifest recording the targets, the
required N, the realized N per subgroup, and the binding subgroup.

A slice is a FILTER over samples and a composite cell crosses several slices
(attribute value x difficulty), so a sample carries a cell only when it carries
every component. Values are read through ``SubgroupCensus`` — the same filter
semantics the census measured with, so a single sample traces through the
identical predicate.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from core.sample_plan import (
    SamplePlanReport,
    SampleValueParse,
    SamplingPlanSpec,
    SubgroupCensus,
    sampling_plan_spec,
    slice_parses,
)


class SubgroupCoverage(BaseModel):
    """The realized coverage of one target after selection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    slice: str
    value: str
    demand: int = Field(ge=1)
    realized: int = Field(ge=0)

    @property
    def covered(self) -> bool:
        return self.realized >= self.demand


class ValidationSetManifest(BaseModel):
    """The emitted validation set's manifest: what it targets and what it got."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str
    binding: str | None
    ci_half_width: float
    alpha: float
    power: float
    target_effect: float
    per_subgroup_n: int
    #: The size the plan asked for (the binding subgroup's total-N requirement).
    required_n: int
    #: The size requested of the builder (the pool may be smaller than required).
    requested_size: int
    realized_n: int
    pool_size: int
    positive: int
    negative: int
    coverage: tuple[SubgroupCoverage, ...]
    #: Targets the pool could not cover, as ``slice=value``.
    uncovered: tuple[str, ...]
    #: The canonical difficulty definition (producer + labels + version) and the
    #: source the labels were read from; ``None`` when no difficulty axis is used.
    difficulty_definition: dict[str, object] | None = None
    difficulty_source: str | None = None


class ValidationSelection(BaseModel):
    """A materialized validation set: the selected ids plus its manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ids: tuple[str, ...]
    manifest: ValidationSetManifest


class ValidationSetBuilder:
    """Materializes the validation set the plan asks for."""

    def __init__(
        self,
        slices: Mapping[str, SampleValueParse],
        cells: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self._slices = dict(slices)
        self._cells = {name: tuple(components) for name, components in (cells or {}).items()}

    @classmethod
    def from_config(cls, spec: SamplingPlanSpec | None = None) -> ValidationSetBuilder:
        """Build from the validated slice declaration."""
        resolved = sampling_plan_spec() if spec is None else spec
        return cls(slice_parses(resolved), resolved.cells)

    def build(
        self,
        pool: Sequence[Mapping[str, object]],
        report: SamplePlanReport,
        *,
        id_key: str,
        label_key: str,
        positive_label: str = "1",
        source: str = "",
        difficulty_definition: dict[str, object] | None = None,
        difficulty_source: str | None = None,
    ) -> ValidationSelection:
        """Select the validation set and report its realized coverage."""
        rows = list(pool)
        if not rows:
            raise ValueError("cannot build a validation set from an empty pool")
        targets = report.measurable_targets()
        demand = {(target.slice, target.value): target.demand for target in targets}
        target_slices = sorted({target.slice for target in targets})
        cells = [self._cells_for(row, target_slices) for row in rows]
        size = min(report.recommended_n, len(rows))
        selected = self._cover(rows, cells, demand, size, id_key, label_key, positive_label)
        selected = self._fill(selected, cells, size, len(rows), id_key, rows)
        realized: dict[tuple[str, str], int] = {}
        for index in selected:
            for cell in cells[index]:
                realized[cell] = realized.get(cell, 0) + 1
        coverage = tuple(
            SubgroupCoverage(
                slice=target.slice,
                value=target.value,
                demand=target.demand,
                realized=realized.get((target.slice, target.value), 0),
            )
            for target in targets
        )
        labels = [str(rows[index].get(label_key)) for index in selected]
        binding = report.binding
        manifest = ValidationSetManifest(
            source=source,
            binding=None if binding is None else f"{binding.slice}={binding.value}",
            ci_half_width=report.ci_half_width,
            alpha=report.alpha,
            power=report.power,
            target_effect=report.target_effect,
            per_subgroup_n=report.per_subgroup_n,
            required_n=report.recommended_n,
            requested_size=size,
            realized_n=len(selected),
            pool_size=len(rows),
            positive=sum(1 for label in labels if label == positive_label),
            negative=sum(1 for label in labels if label != positive_label),
            coverage=coverage,
            uncovered=tuple(
                f"{item.slice}={item.value}" for item in coverage if not item.covered
            ),
            difficulty_definition=difficulty_definition,
            difficulty_source=difficulty_source,
        )
        return ValidationSelection(
            ids=tuple(str(rows[index].get(id_key)) for index in selected),
            manifest=manifest,
        )

    def _cells_for(self, row: Mapping[str, object], target_slices: Sequence[str]) -> frozenset:
        """The (slice, value) coverage cells one sample carries.

        A simple slice reads its own value; a composite cell crosses its
        components' values (``a=v|b=w``), so the sample must carry every
        component for the cell to exist.
        """
        covered = set()
        for name in target_slices:
            if name in self._cells:
                covered.update(
                    (name, value)
                    for value in SubgroupCensus._cell_values(row, self._cells[name], self._slices)
                )
            else:
                covered.update(
                    (name, value) for value in SubgroupCensus.values(row.get(name), self._slices[name])
                )
        return frozenset(covered)

    @staticmethod
    def _cover(
        rows: Sequence[Mapping[str, object]],
        cells: Sequence[frozenset],
        demand: Mapping[tuple[str, str], int],
        size: int,
        id_key: str,
        label_key: str,
        positive_label: str,
    ) -> list[int]:
        """Greedy multicover: maximize unmet demand covered, balance the labels.

        O(size * pool * cells); the pool is one labelled census, so this stays
        small — a much larger pool would want an inverted index.
        """
        remaining = dict(demand)
        selected: list[int] = []
        used: set[int] = set()
        positive = negative = 0
        while len(selected) < size:
            best, best_key = -1, None
            for index, row in enumerate(rows):
                if index in used:
                    continue
                gain = sum(1 for cell in cells[index] if remaining.get(cell, 0) > 0)
                if gain == 0:
                    continue
                need = sum(remaining[cell] for cell in cells[index] if remaining.get(cell, 0) > 0)
                is_positive = str(row.get(label_key)) == positive_label
                minority = 0 if (is_positive and positive <= negative) or (
                    not is_positive and negative <= positive
                ) else 1
                key = (-gain, -need, minority, str(row.get(id_key)))
                if best_key is None or key < best_key:
                    best, best_key = index, key
            if best < 0:
                break
            selected.append(best)
            used.add(best)
            if str(rows[best].get(label_key)) == positive_label:
                positive += 1
            else:
                negative += 1
            for cell in cells[best]:
                if remaining.get(cell, 0) > 0:
                    remaining[cell] -= 1
        return selected

    @staticmethod
    def _fill(
        selected: list[int],
        cells: Sequence[frozenset],
        size: int,
        pool_size: int,
        id_key: str,
        rows: Sequence[Mapping[str, object]],
    ) -> list[int]:
        """Top up to ``size`` with the widest-coverage unselected samples."""
        used = set(selected)
        for index in sorted(
            (i for i in range(pool_size) if i not in used),
            key=lambda i: (-len(cells[i]), str(rows[i].get(id_key))),
        ):
            if len(selected) >= size:
                break
            selected.append(index)
        return selected


__all__ = [
    "SubgroupCoverage",
    "ValidationSelection",
    "ValidationSetBuilder",
    "ValidationSetManifest",
]
