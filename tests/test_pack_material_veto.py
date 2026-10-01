"""tests/test_pack_material_veto.py — the pack_material targeted-veto addition.

Owner ruling 2026-10-01, measured full-corpus on the original 13-column feed:
hard_no 92,259 pairs; pack material both-populated 67,899; genuinely disjoint
(mismatch class) 32,641 (48%). Identity-safe BY CONSTRUCTION: the material
union is canonical-level (package_material_set), both sides of a verified true
match share the same canonical -> the same set, so a set-intersection veto can
never lose a true match. Pinned here:

(a) config parses — pack_material is an admissible veto dimension at load;
(b) a both-populated disjoint-material pair vetoes when enabled, does NOT
    when the gate is disabled (absence of a rule is not reviewed as a veto);
(c) the identity-safe invariant: a same-canonical pair's material sets are
    equal (a row-level SKU set is a subset of its canonical union) and the
    veto never fires on it;
(d) material absent on either side -> unknown: no veto, not even a conflict.
"""

from __future__ import annotations

from core.common import load_config
from core.attribute_conflicts import canonical_attribute_info, sku_attribute_info
from core.critical_attributes import (
    CRITICAL_ATTRIBUTE_DIMENSIONS,
    categorical_conflict,
)
from training.rand_matching import targeted_veto_gate

PACK_MATERIAL_DIMENSION = "pack_material"


def _base_info() -> dict[str, object]:
    """A pair whose shared critical dimensions all agree/populated."""
    return {
        "volume": {500.0},
        "pack": {1},
        "package_type": {"bottle"},
        "flavor": set(),
        "flavor_set": set(),
        "carbonation": set(),
        "sweetener": set(),
        "sweetener_type": set(),
        "sweetening": set(),
        "pulp": set(),
    }


def test_config_admits_pack_material_veto_dimension():
    """(a) load_config green: the measured dimension parses and is auditable."""
    settings = load_config()["rand_matching"]["targeted_veto_gates"]
    assert settings["enabled"] is True
    # a member of the veto list, not of the seven shared critical dimensions
    # (the material clause is discrete; see schemas.TargetedVetoGatesSpec).
    assert PACK_MATERIAL_DIMENSION in settings["veto_dimensions"]
    assert PACK_MATERIAL_DIMENSION not in CRITICAL_ATTRIBUTE_DIMENSIONS


def test_disjoint_material_pair_vetoes_when_enabled_and_not_when_disabled():
    """(b) both-populated disjoint material -> veto fires / absent when off."""
    settings = dict(load_config()["rand_matching"]["targeted_veto_gates"])
    left = _base_info() | {"pack_material": {"glass"}}
    right = _base_info() | {"pack_material": {"plastic"}}
    kw = dict(sku_brand="Acme", candidate_brand="Acme", exact_gtin=False)

    gate = targeted_veto_gate(left, right, config=settings, **kw)
    assert gate["targeted_gate_decision"] == "veto"
    assert "pack_material_mismatch" in str(gate["targeted_gate_reason"])
    assert gate["targeted_gate_route"] == "reject"
    assert "pack_material" in str(gate["targeted_critical_conflicts"])
    # the audit column reports it inside the configured veto set
    assert gate["targeted_vetoed_conflicts"] == "pack_material"

    # disabled gate: no veto, conspicuously not a silent same-decision copy
    gate = targeted_veto_gate(
        left, right, config=settings | {"enabled": False}, **kw
    )
    assert gate["targeted_gate_decision"] == "allow"
    assert gate["targeted_gate_reason"] == "disabled"

    # configured out: audited as a conflict, not spent as a veto
    gate = targeted_veto_gate(
        left,
        right,
        config=settings
        | {
            "veto_dimensions": [
                d for d in settings["veto_dimensions"] if d != PACK_MATERIAL_DIMENSION
            ]
        },
        **kw,
    )
    assert gate["targeted_gate_decision"] == "allow"
    assert "pack_material_mismatch" not in str(gate["targeted_gate_reason"])
    assert "pack_material" in str(gate["targeted_critical_conflicts"])
    assert gate["targeted_vetoed_conflicts"] == ""


def test_same_canonical_material_sets_equal_never_veto():
    """(c) identity-safe invariant, pinned: one canonical -> one material set.

    The candidate side reads the canonical-level union
    (canonical_attribute_info -> package_material_set); a member SKU's
    row-level set is a SUBSET of that union, so the intersection is never
    empty and the veto can structurally never cost a true match.
    """
    settings = load_config()["rand_matching"]["targeted_veto_gates"]
    canonical_material = {"glass", "paper / carton"}
    # the sku's own row evidence: subset of its canonical's union
    left = _base_info() | {"pack_material": {"glass"}}
    right = _base_info() | {"pack_material": canonical_material}
    kw = dict(sku_brand="Acme", candidate_brand="Acme", exact_gtin=False)

    assert left["pack_material"] <= right["pack_material"]
    gate = targeted_veto_gate(left, right, config=settings, **kw)
    assert gate["targeted_gate_decision"] == "allow"
    assert gate["targeted_gate_reason"] == "attributes_compatible"
    assert gate["targeted_critical_conflicts"] == ""
    assert gate["targeted_vetoed_conflicts"] == ""
    # the identical-set case (both sides the canonical record itself)
    both = _base_info() | {"pack_material": canonical_material}
    gate = targeted_veto_gate(both, both, config=settings, **kw)
    assert gate["targeted_gate_decision"] == "allow"
    assert gate["targeted_critical_conflicts"] == ""

    # and the same predicate through the shared categorical rule stays
    # false for equal/overlapping sets (the veto's building block)
    assert categorical_conflict(
        PACK_MATERIAL_DIMENSION,
        {PACK_MATERIAL_DIMENSION: canonical_material},
        {PACK_MATERIAL_DIMENSION: canonical_material},
    ) is False


def test_absent_material_is_unknown_and_never_vetoes():
    """(d) absence on either side -> unknown, no veto, no deferral."""
    settings = load_config()["rand_matching"]["targeted_veto_gates"]
    kw = dict(sku_brand="Acme", candidate_brand="Acme", exact_gtin=False)
    populated = _base_info() | {"pack_material": {"metal"}}

    # neither side carries material
    gate = targeted_veto_gate(_base_info(), _base_info(), config=settings, **kw)
    assert gate["targeted_gate_decision"] == "allow"
    assert gate["targeted_gate_reason"] == "attributes_compatible"
    assert gate["targeted_critical_conflicts"] == ""

    # one side carries it, the other does not — still unknown (absence is
    # not contradiction), and material never lands in the deferral list
    # (DEFERRAL_DIMENSIONS remains pack/volume only)
    gate = targeted_veto_gate(populated, _base_info(), config=settings, **kw)
    assert gate["targeted_gate_decision"] == "allow"
    assert "pack_material_mismatch" not in str(gate["targeted_gate_reason"])
    assert gate["targeted_critical_conflicts"] == ""
    gate = targeted_veto_gate(_base_info(), populated, config=settings, **kw)
    assert gate["targeted_gate_decision"] == "allow"
    assert gate["targeted_critical_conflicts"] == ""


def test_evidence_lanes_carry_pack_material():
    """Both shared info builders expose the captured material evidence."""
    sku = sku_attribute_info("Acme glass bottle", "Pack Material Type: Metal")
    assert sku["pack_material"] == {"metal", "glass"}
    canonical = canonical_attribute_info(
        {"package_material_set": "['paper / carton']", "canonical": "Acme soda"}
    )
    assert canonical["pack_material"] == {"paper / carton"}
    assert canonical_attribute_info({})["pack_material"] == set()
