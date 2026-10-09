"""SamplePlan public behaviour: the reference math and the sized validation set.

Two public behaviours, one test each: the calculator's formulas (the reference
numbers reproduced) and the builder's coverage (a target margin yields a
validation set that reaches the demand of its binding subgroup).
"""
from __future__ import annotations

from core.sample_plan import SamplePlan, SubgroupCensus
from core.schemas import SamplingPlanSpec


def _spec(**target_overrides) -> SamplingPlanSpec:
    """A self-contained plan spec (no config read) for the public-API tests."""
    targets = {
        "confidence": 0.95,
        "ci_half_width": 0.05,
        "alpha": 0.05,
        "power": 0.80,
        "target_effect": 0.02,
        "worst_case_proportion": 0.50,
        "two_proportion_baseline": 0.50,
        "min_subgroup_support": 5,
        "max_realistic_n": 5000,
    }
    targets.update(target_overrides)
    return SamplingPlanSpec.model_validate(
        {
            "targets": targets,
            "slices": {
                "attr": {"parse": "scalar"},
                "difficulty": {"parse": "scalar"},
            },
            "cells": {"attr_difficulty": ["attr", "difficulty"]},
        }
    )


def test_reference_math_pins():
    """The declared formulas reproduce the reference MDE / CI numbers exactly."""
    plan = SamplePlan(_spec())
    assert plan.required_proportion_n(0.50) == 384
    assert plan.required_proportion_n(0.50, half_width=0.10) == 96
    assert plan.required_proportion_n(0.95) == 73
    assert plan.required_proportion_n(0.95, half_width=0.10) == 18
    assert round(plan.mde_paired(28) * 100, 2) == 25.75
    assert round(plan.mde_paired(100) * 100, 2) == 7.67
    assert round(plan.mde_paired(400) * 100, 2) == 1.95
    assert round(plan.mde_paired(1000) * 100, 2) == 0.78
    assert plan.required_paired_n(0.02) == 391


def test_builder_meets_binding_subgroup_demand():
    """A target margin yields a validation set that reaches the binding demand.

    The smallest (attribute x difficulty) cell — the HARD stratum — sets the
    floor; the built set must reach its per-subgroup demand.
    """
    plan = SamplePlan(_spec(target_effect=0.20, ci_half_width=0.20))
    census = SubgroupCensus(plan.slices, plan.spec.cells)
    pool = (
        [{"id": f"ah{i}", "label": "1" if i % 2 else "0", "attr": "a", "difficulty": "hard"}
         for i in range(50)]
        + [{"id": f"bh{i}", "label": "1" if i % 2 else "0", "attr": "b", "difficulty": "hard"}
           for i in range(50)]
        + [{"id": f"ae{i}", "label": "1" if i % 2 else "0", "attr": "a", "difficulty": "easy"}
           for i in range(150)]
        + [{"id": f"be{i}", "label": "1" if i % 2 else "0", "attr": "b", "difficulty": "easy"}
           for i in range(150)]
    )
    report = plan.plan(census.census(pool))

    binding = report.binding
    assert binding is not None
    assert binding.slice == "attr_difficulty"
    assert "difficulty=hard" in binding.value

    selection = plan.build_validation_set(
        report, pool, id_key="id", label_key="label"
    )
    coverage = {(item.slice, item.value): item for item in selection.manifest.coverage}
    realized = coverage[("attr_difficulty", binding.value)]
    assert realized.realized >= binding.per_subgroup_n
    assert selection.manifest.realized_n == min(report.recommended_n, len(pool))
