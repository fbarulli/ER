"""Pin the smart 50/50 setup partition: zero leakage + preserved strata shares.

The partition re-carves an EXISTING prepared setup through the canonical
``core.smart_split.SmartSplit`` (``config/smart_split.yaml``: train 0.5 vs
validate = dev ∪ test 0.5, strata ``difficulty_slice`` × ``attribute``) at the
COMPONENT grain. The fixture is exact by construction (every stratum's weight
divides at the declared ratios), so the carve must land on 50/50 with the
per-axis shares of BOTH halves preserved; the real setup's shares carry the
weighted greedy's one-item seam.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from model_tracks.smart_setup import SmartSetupPartition

#: (difficulty, attribute, components) — the synthetic strata census. Every
#: component is 2 gtins / 2 listings joined by ONE positive pair, so each
#: stratum's listing weight (24/24/16/16) divides exactly at 0.5/0.25/0.25.
STRATA = (("easy", "flavor", 6), ("hard", "flavor", 6),
          ("easy", "volume", 4), ("hard", "volume", 4))

HALVES = {"train": ("train",), "validate": ("dev", "test")}


def _build_setup(root: Path) -> None:
    """A 20-component synthetic setup: 2 gtins (2 listings) per component."""
    catalog, pairs, listings, cohort = [], [], [], []
    gtin = 8900000000000
    components: dict[str, list[list[str]]] = {}
    for difficulty, attribute, count in STRATA:
        for index in range(count):
            skus = []
            for _ in ("a", "b"):
                gtin += 1
                sku = f"sku-{gtin}"
                catalog.append({"sku_id": sku, "gtin": str(gtin)})
                listings.append({"sku_id": sku, "split": "train", "numeric": {},
                                 "attribute": {"flavor": ["lemon"] if attribute == "flavor" else [],
                                               "volume": [] if attribute == "flavor" else [330.0]}})
                skus.append(sku)
            example_id = f"{difficulty}-{attribute}-{index}"
            pairs.append({"example_id": example_id, "sku_id1": skus[0],
                          "sku_id2": skus[1], "label": "1", "split": "train"})
            cohort.append({"example_id": example_id, "sku_id1": skus[0],
                           "sku_id2": skus[1], "difficulty_slice": difficulty,
                           "difficulty_reason": "positive_high_text_overlap"})
            components.setdefault(f"{difficulty}|{attribute}", []).append(skus)
    # Within-stratum negatives: every component pair contributes one cross-gtin
    # negative, so each half keeps negative support after the carve; the ones
    # that land across the halves must be excluded and counted, never relabeled.
    # They stay UNMEASURED (no frozen encoder input) so they cast no difficulty
    # vote — the components keep the stratum their positive pair measured.
    serial = 0
    for group in components.values():
        for left, right in ((a, b) for i, a in enumerate(group) for b in group[i + 1:]):
            serial += 1
            pairs.append({"example_id": f"neg-{serial}", "sku_id1": left[0],
                          "sku_id2": right[1], "label": "0", "split": "train"})
            cohort.append({"example_id": f"neg-{serial}", "sku_id1": left[0],
                           "sku_id2": right[1], "difficulty_slice": "unknown",
                           "difficulty_reason": "no_frozen_encoder_input"})
    (root / "prepared").mkdir(parents=True)
    (root / "ablation_cohort").mkdir()
    pd.DataFrame(catalog).to_csv(root / "eligible_catalog.csv", index=False)
    pd.DataFrame(pairs).to_csv(root / "listing_pairs.csv", index=False)
    (root / "prepared" / "listings.json").write_text(json.dumps(
        {"schema": "er-graph-listings-v1", "listings": listings}))
    pd.DataFrame(cohort).to_csv(root / "ablation_cohort" / "pairs.csv", index=False)


@pytest.fixture()
def partition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The carve of the synthetic setup (hermetic: no labeled census to merge)."""
    import core.identity_policy
    import training.folds
    monkeypatch.setattr(training.folds, "load_labeled_pairs",
                        lambda _: pd.DataFrame({"gtin1": [], "gtin2": [], "true_label": []}))
    monkeypatch.setattr(core.identity_policy, "review_mask",
                        lambda series: pd.Series(False, index=pd.Series(series).index))
    setup = tmp_path / "setup"
    _build_setup(setup)
    return SmartSetupPartition.from_config(setup).partition()


def test_smart_partition_is_zero_leakage_and_stratified(partition) -> None:
    setup = Path(partition.census["source_setup"])
    catalog = pd.read_csv(setup / "eligible_catalog.csv", dtype=str)
    source = pd.read_csv(setup / "listing_pairs.csv", dtype=str)
    split_of_sku = dict(zip(partition.assignments.sku_id, partition.assignments.split))
    gtin_of_sku = dict(zip(catalog.sku_id, catalog.gtin))

    # ZERO leakage: the halves are disjoint on GTIN endpoints...
    gtins = {name: {gtin_of_sku[sku] for sku, split in split_of_sku.items() if split in parts}
             for name, parts in HALVES.items()}
    assert gtins["train"] & gtins["validate"] == set()
    assert len(gtins["train"]) + len(gtins["validate"]) == len(catalog)

    # ...on components (every source positive pair stays inside ONE half)...
    positives = source[source.label == "1"]
    for row in positives.itertuples(index=False):
        assert split_of_sku[row.sku_id1] == split_of_sku[row.sku_id2]

    # ...and on pairs: every kept pair is same-split, positives are never
    # dropped, and the excluded remainder is exactly the cross-half negatives.
    kept = partition.pairs
    assert len(kept[kept.label == "1"]) == len(positives)
    for row in kept.itertuples(index=False):
        assert split_of_sku[row.sku_id1] == row.split == split_of_sku[row.sku_id2]
    assert (len(kept) + partition.census["excluded_cross_split_negatives"]
            == len(source))

    # 50/50 train vs validate at the listing grain...
    shares = partition.assignments.split.value_counts(normalize=True).to_dict()
    half_listings = {name: sum(partition.census["per_split"][split]["listings"]
                               for split in parts) for name, parts in HALVES.items()}
    assert shares["train"] == pytest.approx(0.5, abs=0.01)
    assert shares["dev"] + shares["test"] == pytest.approx(0.5, abs=0.01)
    assert half_listings["train"] == half_listings["validate"] == 20

    # ...with the per-attribute and per-difficulty shares preserved in BOTH
    # halves (each half's share of an axis value stays at the global share).
    for axis in ("attribute", "difficulty_slice"):
        axis_shares = partition.census["axis_shares"][axis]
        values = sorted({value for per_split in axis_shares.values() for value in per_split})
        total = sum(half_listings.values())
        for value in values:
            global_share = sum(
                axis_shares[split].get(value, 0.0) * partition.census["per_split"][split]["listings"]
                for split in axis_shares) / total
            for name, parts in HALVES.items():
                half_share = sum(
                    axis_shares[split].get(value, 0.0)
                    * partition.census["per_split"][split]["listings"]
                    for split in parts) / half_listings[name]
                assert half_share == pytest.approx(global_share, abs=0.05), (axis, value, name)
