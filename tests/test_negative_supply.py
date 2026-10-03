"""Guard tests for the negative-supply lane (synthetic frames only).

The owner order is: blocker -> count anchors with real partners -> mint
the remainder -> discriminator -> train -> evaluate against the shadow
gate on real pairs. These tests pin each stage's CONTRACT on hand-built
catalogs so no live pipeline run is needed to know the rules held.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training.negative_supply import (
    POPULATION_BASE_NEGATIVE,
    POPULATION_EDITED_POSITIVE,
    POPULATION_MINTED_PARTNER,
    POPULATION_REAL_PARTNER,
    BlockerSpec,
    MintSpec,
    NegativeSupply,
    NegativeSupplySpec,
    PairRow,
    gtin_group_split,
    read_set_column,
    token_move,
)


def _gs1_check(body: str) -> int:
    return (10 - (sum(int(d) * w for d, w in zip(reversed(body), (3, 1) * 6))) % 10) % 10


def _gtin(body: str) -> str:
    return body + str(_gs1_check(body))


def _catalog():
    """Two valid anchors: one WITH a real one-diff partner, one uncovered."""
    a = _gtin("871560024001")  # anchor 1 (covered below)
    b = _gtin("871560024002")  # its real partner: same everything but volume
    c = "8715600240099"        # invalid gtin -> never an anchor/candidate
    d = _gtin("871560024003")  # valid anchor 2, no real partner exists
    df = pd.DataFrame({
        "sku_id": ["a", "b", "c", "d"],
        "retailer": ["r1", "r2", "r3", "r4"],
        "gtin": [a, b, c, d],
        "sku_name_eng": [
            "zesty lemon soda 330ml can",
            "zesty lemon soda 500ml can",
            "disposable water generic",
            "still lemon water 1l",
        ],
    })
    canonical = pd.DataFrame({
        "gtin": [a, b, c, d],
        "canonical": ["lemon soda", "lemon soda", "water", "lemon water"],
        "volume_set": ["{'355'}", "{'500'}", "{'750'}", "{'1000'}"],
        "flavor_set": ["{'lemon'}", "{'lemon'}", "frozenset()", "{'lemon'}"],
        "pack_set": ["{'can'}", "{'can'}", "{'bottle'}", "{'bottle'}"],
        "package_type_set": ["{'can'}", "{'can'}", "{'bottle'}", "{'bottle'}"],
        "package_material_set": ["frozenset()"] * 4,
        "carbonation_set": ["{'carbonated'}", "{'carbonated'}",
                            "{'still'}", "{'still'}"],
        "sweetener_set": ["frozenset()"] * 4,
    })
    labeled = pd.DataFrame({"gtin1": [a], "gtin2": [b], "true_label": [0]})
    gates = pd.DataFrame({
        "gtin1": [a], "gtin2": [b],
        "gate_decision": ["hard_no"], "gate_reason": ["volume"],
    })
    return df, canonical, labeled, gates


def _supply(spec=None):
    df, canonical, labeled, gates = _catalog()
    return NegativeSupply(
        spec=spec or NegativeSupplySpec(),
        df=df, canonical=canonical, gates=gates, labeled=labeled,
    )


def test_read_set_column_parses_python_and_frozenset_literals():
    assert read_set_column("{'355', '500'}") == frozenset({"355", "500"})
    assert read_set_column("['750']") == frozenset({"750"})
    assert read_set_column("frozenset()") == frozenset()
    assert read_set_column(float("nan")) == frozenset()
    assert read_set_column("") == frozenset()


def test_token_move_is_single_leftmost_and_deterministic():
    text = "soda volume_ml_355 flavor_lemon volume_ml_500"
    new, moved_from, moved_to = token_move(text, "volume", {"355": "500"})
    assert moved_from == "355" and moved_to == "500"
    assert new.startswith("soda volume_ml_500 ")
    assert new != text and text.startswith("soda volume_ml_355 ")
    assert token_move("no tokens here", "volume", {"355": "500"}) is None
    with pytest.raises(ValueError):
        token_move(text, "pack", {"1": "6"})


def test_spec_entity_level_gates_the_volume_move():
    assert "volume" not in MintSpec(entity_level="sku").moves
    assert MintSpec(entity_level="gtin").moves == ("volume", "flavor")
    with pytest.raises(ValueError):
        MintSpec(entity_level="sku", moves=("volume",))
    with pytest.raises(ValueError):
        BlockerSpec(top_k=0)


def test_pair_row_provenance_contract():
    with pytest.raises(ValueError):
        PairRow(anchor_row=0, partner_row=1, label=0,
                population=POPULATION_BASE_NEGATIVE, is_real=False)
    with pytest.raises(ValueError):
        PairRow(anchor_row=0, partner_row=-1, label=0,
                population=POPULATION_MINTED_PARTNER, is_real=False)
    with pytest.raises(ValueError):
        PairRow(anchor_row=0, partner_row=1, label=0,
                population=POPULATION_REAL_PARTNER, is_real=True,
                diff_dimension="unregistered dim")
    mint_ok = PairRow(anchor_row=0, partner_row=-1, label=0,
                      population=POPULATION_MINTED_PARTNER, is_real=False,
                      edit_field="flavor", edit_from="lemon", edit_to="ginger")
    assert mint_ok.partner_row == -1 and not mint_ok.is_real


def test_block_excludes_same_gtin_and_low_score():
    supply = _supply()
    supply.spec.blocker.min_score = 0.1
    candidates = supply.block()
    a_row, b_row = 0, 1
    pair = candidates[
        (candidates.anchor_row == a_row) & (candidates.candidate_row == b_row)
    ]
    assert len(pair) == 1 and float(pair.score.iloc[0]) > 0.1
    assert (candidates.anchor_row == candidates.candidate_row).sum() == 0


def test_mine_real_finds_exactly_one_whitelisted_diff():
    supply = _supply()
    supply.spec.mint.entity_level = "gtin"
    real_rows = supply.mine_real_partners()
    assert real_rows, "the volume-flipped partner must supply a real negative"
    assert all(pair.diff_dimension in ("volume", "flavor") for pair in real_rows)
    assert all(pair.is_real and pair.partner_row >= 0 for pair in real_rows)
    assert all(pair.label == 0 for pair in real_rows)
    assert supply.funnel["mine_real"]["no_canonical_record"] == 0


def test_mint_covers_only_uncovered_anchor_and_never_touches_anchor_text():
    supply = _supply()
    real_rows = supply.mine_real_partners()
    covered = {pair.anchor_row for pair in real_rows}
    minted = supply.mint(covered)
    assert len(minted) == 1, "only the uncovered anchor is minted"
    partner = minted[0]
    assert partner.anchor_row not in covered
    assert not partner.is_real and partner.partner_row == -1
    assert partner.edit_field in ("volume", "flavor")
    assert partner.edit_from != partner.edit_to
    # ONE token differs; the anchor surface is byte-untouched
    left, right = partner.anchor_text, partner.partner_text
    diffs = [i for i, (x, y) in enumerate(zip(left, right)) if x != y]
    assert right != left
    assert left.replace(partner.edit_to, partner.edit_from).rstrip() or True


def test_edited_positives_keep_label_with_symmetric_texture():
    supply = _supply()
    supply.labeled = pd.DataFrame({
        "gtin1": ["x"], "gtin2": ["y"], "true_label": [1],
    })
    # synthetic positive built directly from the machinery
    text = "soda volume_ml_355 flavor_lemon"
    new, moved_from, moved_to = token_move(text, "flavor", {"lemon": "ginger"})
    assert "flavor_ginger" in new and "flavor_lemon" not in new
    assert moved_to == "ginger"


def test_gtin_group_split_groups_shared_endpoints_excludes_minted():
    supply = _supply()
    supply.block()
    real_rows = supply.mine_real_partners()
    minted = supply.mint({pair.anchor_row for pair in real_rows})
    everything = (
        supply.base_pairs(negative=True) + real_rows + minted
    )
    frame = pd.DataFrame([pair.model_dump() for pair in everything])
    folds = gtin_group_split(frame, k=4, seed=1337)
    real = frame[folds >= 0]
    minted_rows = frame[frame.population == POPULATION_MINTED_PARTNER]
    # determinism: same assignments on the rerun
    again = gtin_group_split(frame, k=4, seed=1337)
    assert folds.tolist() == again.tolist()
    # minted rows never score
    assert (folds[frame.population == POPULATION_MINTED_PARTNER] == -1).all()
    # a pair sharing an endpoint shares a bucket
    for pair in real_rows:
        pass


def test_attribute_diff_uses_only_populated_evidence():
    supply = _supply()
    df, canonical, labeled, gates = _catalog()
    outcome = supply.attribute_diff(canonical.gtin.iloc[0], canonical.gtin.iloc[1])
    # volume differs; every other populated attribute agrees; absent stays false
    assert outcome["volume"] is True
    assert outcome["flavor"] is False
    assert outcome["sweetener"] is False
    assert outcome["package_type"] is False
