"""Pin: the laya corpus splits meet the canonical SamplePlan requirement.

``core.sample_plan.SamplePlan`` (targets from ``config/sampling.yaml``) is the
ONE calculator for "how many validation samples a measurement needs". This pins
that the committed laya corpus splits that feed calibration (``dev``) and the
held-out read (``test``) are at least the per-subgroup N the plan requires, so
the corpus is not silently shrunk below the measurement floor.
"""
from __future__ import annotations

import json
from pathlib import Path

from core.sample_plan import SamplePlan

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "data/laya/receipt.json"


def test_laya_corpus_splits_meet_the_sample_plan_requirement() -> None:
    plan = SamplePlan.from_config()
    targets = plan.spec.targets
    required = max(
        plan.required_paired_n(targets.target_effect),
        plan.required_proportion_n(targets.worst_case_proportion),
    )
    sizes = json.loads(RECEIPT.read_text(encoding="utf-8"))["split_sizes"]
    assert sizes["dev"] >= required, (sizes, required)
    assert sizes["test"] >= required, (sizes, required)
