"""Pin: the laya corpus carve is power-consistent and stratum-covering.

``core.sample_plan.SamplePlan`` (targets from ``config/sampling.yaml``) is the
ONE calculator for "how many validation samples a measurement needs", and the
canonical builder (``scripts/laya_build_dataset.py``) records the carved
corpus's plan in ``data/laya/receipt.json`` (``sample_plan``). This pins the
carve's public behaviour: the plan is REACHABLE for the full corpus, both the
selecting ``dev`` and the validating ``test`` split carry at least the
per-subgroup N, and every MEANINGFUL ``difficulty_slice`` subgroup (support at
the declared floor) is represented in both.
"""
from __future__ import annotations

import json
from pathlib import Path

from core.smart_split import SmartSplit

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "data/laya/receipt.json"


def test_laya_carve_is_reachable_and_covers_every_meaningful_stratum() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    plan = receipt["sample_plan"]
    targets = plan["targets"]
    sizes = receipt["split_sizes"]

    # The carve is the ONE SmartSplit owner: its role map is the receipt's.
    assert receipt["split_roles"] == SmartSplit.from_config().roles

    # The full corpus can deliver the declared MDE (the reachable plan), and
    # dev/validation each carry at least the per-subgroup sample floor.
    assert plan["reachable"] is True, plan["binding"]
    assert sizes["dev"] >= plan["per_subgroup_n"], (sizes, plan["per_subgroup_n"])
    assert sizes["test"] >= plan["per_subgroup_n"], (sizes, plan["per_subgroup_n"])

    # Every difficulty subgroup at/above the declared support floor is present
    # in BOTH the selecting dev split and the validating test split.
    support = receipt["difficulty_slice_census"]
    meaningful = {value for value, count in support.items()
                  if count >= targets["min_subgroup_support"]}
    assert meaningful, support
    for split in ("dev", "test"):
        covered = receipt["strata_coverage"][split]["difficulty_slice"]
        missing = sorted(value for value in meaningful if not covered.get(value))
        assert not missing, (split, missing, covered)
