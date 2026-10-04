"""Alias-aware brand gates: the seeded fold holds at every consumption site.

One mechanism (``core.sku_identity.normalize_brand``: config-vocabulary
"brand_aliases", owner ruling 2026-09-29/30, seeded from the within-GTIN
brand-variant census — 71 variant groups, 49 distinct within-group brand
vetoes, 27 dissolved by the 8 reviewed entries) feeds FOUR consumers, and each
one is pinned here on synthetic frames:

* veto (rand-matching brand gate): an alias family pair ("A SHOC" vs
  "Accelerator") must NO LONGER veto a candidate; a distinct brand
  ("Acme" vs "Other") must still veto; a one-sided/absent brand must stay
  unknown (absence is not contradiction); the exact-GTIN lock stays a bypass.
* blocking (record_linkage brand blocks): alias siblings must land in ONE
  candidate-generation block — a recall gain only; the link rule still has to
  pass, so an alias block meeting that fails the same-title rule still emits
  no link.
* negative donors (hard_negatives cross-brand `_brand_surface_variant`): the
  subset/overlap guard now judges the FOLDED family, so an alias member and
  its canonical spelling are spared donor status together with the plain
  surface variants ("Kiju" vs "Kiju Organic").
* evaluation (evaluate_models canon-map join): the UPC-12 zero-prefix latent
  dtype bug — canonical DataFrame built through an int64-coercing frame loses
  the leading zero and the `gtin -> canonical` map join NaNs out — is closed
  by string normalization, and the repair is idempotent on the pre/post row
  counts (0 rows affected TODAY: canonical_records.csv holds no leading-zero
  GTIN; the synthetic join pins the WOULD-BE defect).
* veto asymmetry (owner doctrine): on a fixed synthetic frame, adding alias
  knowledge can never ADD a brand-conflict verdict — conflicts-with-aliases
  <= conflicts-without — because a fold only ADDS tokens (sets can share or
  nest, never go disjoint). A conflict manufactured by folding is THE
  expensive direction: it merges two real products' vetoes.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.sku_identity import (
    brand_aliases,
    brand_conflict,
    normalize_brand,
)

# Frozen at seed time; guards the SSOT content these tests reason about
# (same doctrine as tests/test_brand_aliases.py EXPECTED_KEYS).
_ALIAS_KEYS = frozenset(
    {"accelerator", "adrenaline", "ball", "fitaid", "hi", "kevytolo", "techusa", "ting"}
)


def _settings() -> dict[str, object]:
    """The measured veto opt-in (mirrors tests/test_targeted_ann_gates.py)."""
    return {
        "enabled": True,
        "veto_dimensions": ["volume", "pack", "package_type", "flavor", "carbonation", "pulp"],
        "pack_mismatch_veto": True,
        "volume_mismatch_veto": True,
        "package_type_mismatch_veto": True,
        "brand_mismatch_veto": True,
        "missing_pack_or_volume_route": "human_review",
        "volume_relative_tolerance": 0.05,
        "volume_absolute_tolerance_ml": 5.0,
        "preserve_exact_gtin": True,
    }


def _info(
    *,
    pack: set[float] | int = (),
    volume: set[float] | int = (),
    package_type: set[str] = (),
    flavor: set[str] = (),
    carbonation: set[str] = (),
    sweetener: set[str] = (),
    pulp: set[str] = (),
    canonical: str = "acme_soda_500",
) -> dict[str, object]:
    """The gate-info shape targeted_veto_gate consumes (both readers: the
    SET keys for ``critical_attribute_evaluation`` plus the scalar display
    keys, per the contract test_targeted_ann_gates._info pins).

    The shared canonical name is affirmative identity evidence: since
    50e2d5d auto_merge requires a positive identity signal
    (pair_policy.assess_pair -> positive_identity_missing), so the brand
    pins below test the alias fold in isolation, not the identity policy."""
    by_key = {
        "pack": pack, "volume": volume, "package_type": package_type,
        "flavor_set": flavor, "carbonation_set": carbonation,
        "sweetener_set": sweetener, "pulp_set": pulp,
    }
    info: dict[str, object] = {}
    for key, value in by_key.items():
        info[key] = set(value)
        if key == "flavor_set":
            info["flavor"] = " ".join(sorted(value))
        elif key in {"carbonation_set", "sweetener_set", "pulp_set"}:
            info[key.rsplit("_", 1)[0]] = set(value)
    info["canonical"] = canonical
    return info


# ── veto: the alias pair no longer vetoes, distinct brands still do ────────
def test_alias_family_pair_no_longer_vetoes() -> None:
    """The measured defect: 'A SHOC' != 'Accelerator' raw, one family folded.

    A candidate carrying agreeing pack/volume evidence must reach auto_merge
    with the fold live (before it: brand_mismatch reject — the lane vetoed
    proven-same pairs exactly like these inside the 49 within-GTIN vetoes
    the alias map was seeded from).
    """
    from training.rand_matching import targeted_veto_gate

    gate = targeted_veto_gate(
        _info(pack={6}, volume={750}),
        _info(pack={6}, volume={750}),
        sku_brand="A SHOC",
        candidate_brand="Accelerator",
        exact_gtin=False,
        config=_settings(),
    )
    assert gate["targeted_brand_conflict"] == 0
    assert gate["targeted_gate_decision"] == "allow"
    assert gate["targeted_gate_route"] == "auto_merge"
    # The audit columns carry the FOLDED family key next to the spellings.
    assert "shoc" in str(gate["targeted_brand_a"]).split()
    assert "shoc" in str(gate["targeted_brand_b"]).split()


def test_distinct_brands_still_veto() -> None:
    """Veto asymmetry in its sharpest form: the fold dissolves only the
    measured families. 'Acme' vs 'Other' stays a brand_mismatch veto."""
    from training.rand_matching import targeted_veto_gate

    gate = targeted_veto_gate(
        _info(pack={6}, volume={750}),
        _info(pack={6}, volume={750}),
        sku_brand="Acme",
        candidate_brand="Other",
        exact_gtin=False,
        config=_settings(),
    )
    assert gate["targeted_brand_conflict"] == 1
    assert gate["targeted_gate_decision"] == "veto"
    assert "brand_mismatch" in str(gate["targeted_gate_reason"])
    assert gate["targeted_gate_route"] == "reject"


def test_gate_off_still_reports_and_never_blocks() -> None:
    """The flag still gates: brand_mismatch_veto=False reports the conflict in
    targeted_brand_conflict and targeted_critical_conflicts but vetoes on
    nothing brand-related."""
    from training.rand_matching import targeted_veto_gate

    settings = {**_settings(), "brand_mismatch_veto": False}
    gate = targeted_veto_gate(
        _info(pack={6}, volume={750}),
        _info(pack={6}, volume={750}),
        sku_brand="Acme",
        candidate_brand="Other",
        exact_gtin=False,
        config=settings,
    )
    assert gate["targeted_brand_conflict"] == 1
    assert gate["targeted_gate_route"] != "reject"
    assert "brand_mismatch" not in str(gate["targeted_gate_reason"])


def test_missing_brand_stays_unknown_not_conflict() -> None:
    """Absence is never contradiction: one side with no brand must route by
    the remaining evidence, and must never produce a brand_mismatch reason —
    even against a populated brand the fold knows nothing about."""
    from training.rand_matching import targeted_veto_gate

    for left_brand, right_brand in (("", "A SHOC"), ("Accelerator", "")):
        gate = targeted_veto_gate(
            _info(pack={6}, volume={750}),
            _info(pack={6}, volume={750}),
            sku_brand=left_brand,
            candidate_brand=right_brand,
            exact_gtin=False,
            config=_settings(),
        )
        assert gate["targeted_brand_conflict"] == 0
        assert "brand_mismatch" not in str(gate["targeted_gate_reason"])
        assert gate["targeted_gate_decision"] == "allow"


def test_exact_gtin_bypass_unchanged_by_the_fold() -> None:
    """preserve_exact_gtin stays the top of the gate: a trusted gtin is the
    answer and no text comparison (with or without an alias) is consulted."""
    from training.rand_matching import targeted_veto_gate

    gate = targeted_veto_gate(
        _info(pack={12}, volume={750}),
        _info(),
        sku_brand="A SHOC",
        candidate_brand="Other",
        exact_gtin=True,
        config=_settings(),
    )
    assert gate["targeted_gate_decision"] == "exact_gtin_lock"
    assert gate["targeted_gate_route"] == "auto_merge"


def test_both_sides_route_through_the_same_ssot_fold() -> None:
    """The gate's audit spelling matches the SSOT fold, so 'DiAGNOSIS'-grade
    drift (a lane re-normalizing privately) fails loudly: the printed family
    key IS what the decision consumed."""
    from training.rand_matching import targeted_veto_gate

    gate = targeted_veto_gate(
        _info(pack={6}, volume={750}),
        _info(pack={6}, volume={750}),
        sku_brand="A SHOC",
        candidate_brand="A SHOC",
        exact_gtin=False,
        config=_settings(),
    )
    assert str(gate["targeted_brand_a"]).split() == sorted(normalize_brand("A SHOC"))
    assert str(gate["targeted_brand_b"]).split() == sorted(normalize_brand("A SHOC"))


# ── blocking: alias siblings share a record_linkage block ──────────────────
def test_alias_siblings_share_a_block_key() -> None:
    """Blocking keys must be IDENTICAL for alias siblings.

    The folded token sets are NOT ("A SHOC" -> {a, shoc}, "Accelerator" ->
    {accelerator, shoc}) — a groupby on the fold itself would still split the
    family — so the family key collapses a brand to the alias families its
    fold reaches. Both spellings reach {shoc} here.
    """
    from core.record_linkage import _brand_family_key

    keys = _brand_family_key(pd.Series(["A SHOC", "Accelerator", "a shoc"]))
    assert keys.iloc[0] == keys.iloc[1] == keys.iloc[2] == "shoc"


def test_alias_free_brands_keep_their_pre_alias_block_key() -> None:
    """A brand no alias family reaches keeps its full folded spelling as the
    block key — the pre-alias key, byte-identical, so alias-free blocking
    behavior ("Goat Fuel" vs "Goa Fuel" staying separate) is unchanged."""
    from core.record_linkage import _brand_family_key

    keys = _brand_family_key(pd.Series(["Goat Fuel", "Goa Fuel", "ACME"]))
    assert keys.iloc[0] == "fuel goat" and keys.iloc[1] == "fuel goa"
    assert keys.iloc[2] == "acme"


def test_blank_brand_never_enters_any_block() -> None:
    """Rows with no brand at all must stay OUT of every block: the empty fold
    keeps the blank sentinel and must never collapse brand-less rows into one
    shared block."""
    from core.record_linkage import _brand_family_key

    keys = _brand_family_key(pd.Series(["", None, "A SHOC", float("nan")]))
    assert list(keys) == ["", "", "shoc", ""]


def test_alias_siblings_link_across_retailers_end_to_end() -> None:
    """The recall gain, end to end: two cross-retailer rows whose ONLY brand
    difference is the alias family land in one block and LINK when the same
    pack-stripped title passes the match rule. The link rule (the verifier)
    still has to pass — blocking asserts nothing."""
    from core.record_linkage import link_gtin_less

    df = pd.DataFrame(
        [
            ("a", "shoc energy drink", "A SHOC", "", "Shop A"),
            ("b", "shoc energy drink", "Accelerator", "", "Shop B"),
        ],
        columns=["sku_id", "sku_name_eng", "brand", "gtin", "retailer"],
        index=["r0", "r1"],
    )
    clusters, census = link_gtin_less(df)
    assert clusters["r0"] == clusters["r1"]
    assert census["exact_title_pairs_linked"] == 1


def test_alias_sibling_block_does_not_weaken_the_verifier() -> None:
    """A wider block adds CANDIDATE PAIRS, not links: an alias block holding
    two different titles still emits no link when the match rule refuses
    (different retailers, IDF-weighted Jaccard below the 0.7 threshold).
    The candidate census (pair_checks) shows the pair was generated; the
    verifier's no is the final word."""
    from core.record_linkage import link_gtin_less

    df = pd.DataFrame(
        [
            ("a", "shoc orange soda", "A SHOC", "", "Shop A"),
            ("b", "shoc grape water", "Accelerator", "", "Shop B"),
        ],
        columns=["sku_id", "sku_name_eng", "brand", "gtin", "retailer"],
        index=["r0", "r1"],
    )
    clusters, census = link_gtin_less(df)
    assert clusters["r0"] != clusters["r1"]
    assert census["pair_checks"] == 1
    assert census["fuzzy_title_pairs_linked"] == 0
    assert census["exact_title_pairs_linked"] == 0


# ── hard_negatives: folded acceptance judges the family, not the spelling ──
def test_alias_siblings_are_spared_as_cross_brand_donors() -> None:
    """``_brand_surface_variant`` folds both sides BEFORE the subset test.

    "A SHOC" {a, shoc} and "Accelerator" {accelerator, shoc} SHARE the folded
    `shoc` token, so the pair is a surface variant of one brand and must be
    spared. Before the fold, both spellings were raw-disjoint token sets and
    the pair entered the cross-brand donor pool like any two real brands.
    """
    from core.hard_negatives import _brand_surface_variant

    assert _brand_surface_variant("A SHOC", "Accelerator")
    assert _brand_surface_variant("Accelerator", "A SHOC")
    # the rest of the seeded family inside its own spellings too
    assert _brand_surface_variant("A SHOC", "Adrenaline Shoc")


def test_alias_accents_and_punctuation_still_one_family() -> None:
    """The guard keeps its original folded-normalization job: `Peet""S`-grade
    OCR damage and accents collapse before the subset test, so the folded
    acceptance ADDS to the plain-surface acceptance it never removes."""
    from core.hard_negatives import _brand_surface_variant

    assert _brand_surface_variant("Kiju Organic", "Kiju")
    assert _brand_surface_variant("the london essence co", "london essence co")


def test_alias_member_and_its_canonical_are_not_mineable_negatives() -> None:
    """End-to-end: a cross-brand mining window must not mine the alias family
    into the donor pool. One family member (mode_brand spelled 'A SHOC') and
    its canonical spelling ('Accelerator') each share volume/package evidence
    with a competitor; the family member may donate against the competitor,
    and the canonical-spelling sibling of the SAME family... also may — but
    the pair (member, canonical-sibling) itself must never surface, because
    the surface-variant guard now judges the folded family it does not mine."""
    from core.hard_negatives import CrossBrandMiningFunnel, mine_cross_brand_negatives

    gtins = {
        "a_shoc": "4006381333931",
        "accelerator": "4006381340250",
        "bolt": "4006381340366",
        "crisp": "4006381340373",
    }
    records = []
    for gtin, brand in (
        (gtins["a_shoc"], "A SHOC"),
        (gtins["accelerator"], "Accelerator"),
        (gtins["bolt"], "Bolt"),
        (gtins["crisp"], "Crisp"),
    ):
        records.append(
                {
                    "gtin": gtin,
                    # One canonical text per item, folded spelling distinct:
                    # the collapsed `X and 'cola'` f-string fixture made ALL
                    # FOUR rows read canonical "cola cola", so every pair
                    # died at the same-canonical guard before the
                    # surface-variant guard could ever fire.
                    "canonical": f"{' '.join(sorted(normalize_brand(brand)))} cola",
                    "mode_brand": brand,
                "mode_flavor": "cola",
                "volume_set": "[330]",
                "pack_set": "[]",
                "package_type_set": "['can']",
                "flavor_set": "['cola']",
                "carbonation_set": "['carbonated']",
                "sweetener_set": "['sugar']",
                "pulp_set": "[]",
            }
        )
    canon = pd.DataFrame(records)
    df = pd.DataFrame(
        {
            # Deliberately DIFFERENT titles: a same-title pair would be the
            # conflicting-gtin label error the miner excludes.
            "gtin": [gtins["a_shoc"], gtins["accelerator"], gtins["bolt"], gtins["crisp"]],
            "sku_name_eng": ["A SHOC cola can", "Accelerator cola can", "Bolt cola can", "Crisp cola can"],
        }
    )
    gtin_to_row = {g: i for i, g in enumerate(df["gtin"].astype(str))}
    gtin_to_canon_idx = {
        g: len(df) + i for i, g in enumerate(sorted(canon["gtin"].astype(str)))
    }
    funnel = CrossBrandMiningFunnel()
    pairs, _ = mine_cross_brand_negatives(
        df, canon, gtin_to_row, gtin_to_canon_idx,
        n_target=100, require_agreement=("volume", "package_type"),
        min_similarity=0.0, funnel=funnel,
    )
    # Pairs are (source-sku ROW index, other-canonical PAYLOAD index) — the
    # canonical side lives OUTSIDE df, so its coordinate resolves through the
    # sorted-GTIN construction of gtin_to_canon_idx, never df.iat.
    row_gtins = [str(b) for b in df["gtin"]]
    canon_idx_gtins = {
        len(df) + i: g for i, g in enumerate(sorted(canon["gtin"].astype(str)))
    }
    endpoints = {
        frozenset(
            (
                row_gtins[int(source)],
                canon_idx_gtins[int(other_canonical)],
            )
        )
        for source, other_canonical in pairs
    }
    shoc_pair = frozenset({gtins["a_shoc"], gtins["accelerator"]})
    assert shoc_pair not in endpoints
    assert funnel.dropped_candidates_brand_surface_variant >= 1


# ── evaluate_models: the zero-prefixed canonical join, repaired loudly ─────
def _zero_prefixed_canonical_frame(n_rows: int = 3) -> pd.DataFrame:
    """A canonical_records-shaped frame whose sole GTIN is a real checksum-
    valid UPC-12 (036000291452, tests/test_gtin_integrity's laboratory value).
    The zero-prefixed spelling is what the canonical lane writes; the int64
    read path is exactly how any untyped consumer mangles it.
    """
    return pd.DataFrame(
        {
            "gtin": ["036000291452"] * n_rows,
            "canonical": ["acme cola"] * n_rows,
            "mode_brand": ["Acme"] * n_rows,
        }
    )


def test_canonical_map_join_resolves_a_zero_prefixed_upc12() -> None:
    """The labeled-side duplicate of evaluate_models' canon-map join.

    The int64-losing path: the canonical map built from a frame the dtype was
    coerced on (pandas read_csv without dtype=str doing that on its own;
    any parquet->pandas round trip too). The string-normalized map reproduces
    the stage's contract: keys byte-spelled, join exact.
    """
    canon_int = _zero_prefixed_canonical_frame().assign(
        # simulate the coercion: the leading zero is silently destroyed
        gtin=lambda frame: frame["gtin"].astype("int64")
    )
    gtin_to_canon_int = dict(
        zip(canon_int["gtin"].astype(str), canon_int["canonical"].astype(str), strict=True)
    )
    # The defect, reproduced: '036000291452' was mangled to 36000291452
    # somewhere upstream (e.g. re-indexing through the int64 spelling), so the
    # raw-spelled lookup misses, a pair endpoint NaNs out and the row thins.
    assert "036000291452" not in gtin_to_canon_int
    assert "36000291452" in gtin_to_canon_int

    # The repair is the stage's own SSOT path (evaluate_models doctrine:
    # the canonical GTIN column is read as dtype=str, so the map keys stay
    # BYTE-spelled and the join needs no GTIN rewriting). The map is
    # exercised over the frame's INDEXED rows — 3 rows in -> 3 indexed
    # (key, canonical) entries, nothing dropped — because a gtin-keyed dict
    # collapses duplicate spellings BY DESIGN (these 3 rows are one
    # canonical item); the row-count pin lives on the indexed structure,
    # the join pin on the dict. The SSOT cleaner confirms the zero-prefix
    # holds: every row survives normalize_and_validate_gtin (its UPC-12 ->
    # GTIN-13 canonicalization is prefixing the 13-digit canonical spelling,
    # not erasing a digit).
    from core.gtin import normalize_and_validate_gtin

    canon_str = _zero_prefixed_canonical_frame()
    facts = normalize_and_validate_gtin(pd.Series(canon_str["gtin"], dtype="string"))
    assert facts["gtin_clean"].notna().all()
    indexed_rows = list(
        zip(canon_str["gtin"].astype(str), canon_str["canonical"].astype(str), strict=True)
    )
    # 0->3 rows: every canonical row contributes, no silent thinning.
    assert len(indexed_rows) == 3
    gtin_to_canon = dict(indexed_rows)
    assert len(gtin_to_canon) == 1  # duplicate keys collapse by design
    assert "036000291452" in gtin_to_canon
    labeled = pd.DataFrame({"gtin1": ["036000291452"], "gtin2": ["036000291452"]})
    df = labeled.copy()
    df["canon1"] = df["gtin1"].map(gtin_to_canon)
    df["canon2"] = df["gtin2"].map(gtin_to_canon)
    assert df["canon1"].notna().all() and df["canon2"].notna().all()


def test_frozen_canonical_records_have_no_leading_zero_gtins() -> None:
    """Pre/post row-count identity on the REAL data (fast path).

    The stage's measured claim, pinned against the artifact it measured:
    canonical_records.csv holds ZERO leading-zero GTINs today, so the
    string-normalized join changes nothing — rows in == rows out, 0 affected.
    If this ever fails, the int64 hazard is LIVE on the corpus and the stage
    must be rerun with the repair path (never silently compiled away).
    """
    canon = pd.read_csv("data/canonical_records.csv", dtype={"gtin": str}, keep_default_na=False)
    # RE-PINNED 2026-10-01: 13,225 -> 13,216 canonicals (-9). Source:
    # results/manifests/data_prep.json (run 2026-09-30 23:43-23:45):
    # dataset.csv SHA 539c2479 rows=197,783 -> canonical_records.csv
    # rows=13,216 SHA a655b27d; row_accounting closure 71,623 in ==
    # 13,216 kept + 12,995 collapsed_same_gtin + 45,412 dropped
    # (gtin_missing_or_nan 41,545 / gtin_checksum_failed 3,715 /
    # identity_review_quarantined 152). The old 13,225 was measured
    # before the latest dedupe chain rebuild (dataset_deduped.csv SHA
    # 73a94016, 63,079 rows; the 9-canonical shortfall rides that
    # rebuild). Zero leading-zero GTINs: unchanged (re-measured 0 on
    # the new export). If this fails again, the int64 hazard is live
    # (or the export narrowed again): rerun data_prep with the repair
    # path; never silently compile the guard away.
    assert len(canon) == 13216
    leading_zero = canon["gtin"].str.startswith("0").sum()
    assert int(leading_zero) == 0
    # every key byte-spells into the string-normalized map with no loss
    from core.gtin import normalize_and_validate_gtin

    facts = normalize_and_validate_gtin(canon["gtin"])
    mapped = facts["gtin_clean"].astype("string").str.zfill(14).notna()
    clean = facts["gtin_clean"].astype("string").fillna(canon["gtin"].astype("string"))
    joined = dict(zip(clean, canon["canonical"], strict=True))
    assert len(joined) == len(canon)
    del mapped  # extra probe retained for symmetry; the row count is the pin


# ── veto-asymmetry property: folds may never manufacture a conflict ────────
def test_veto_asymmetry_property_on_a_fixed_synthetic_frame() -> None:
    """conflicts_WITH_aliases <= conflicts_WITHOUT, exhaustively over a fixed
    synthetic brand frame crossed with the seeded map.

    The frame: 8 real measured spellings + 8 generic competitors, every
    unordered pair. The fold only ADDS tokens (set sharing/nesting, never new
    disjointness), so no pair may GAIN a conflict, and the alias-family pairs
    must lose theirs.
    """
    from itertools import combinations

    spellings = [
        "A SHOC", "Accelerator", "Adrenaline Shoc",
        "DG", "Ting",
        "Hi Ball", "Hiball Energy",
        "FitAid", "LifeAid",
        "Olvi", "Kevytolo",
        "BioTech USA", "Bio Techusa",
        "Acme", "Bolt", "Crisp",
    ]
    assert set(brand_aliases()) == _ALIAS_KEYS

    def raw_conflict(left: str, right: str) -> bool:
        a = left.strip().casefold()
        b = right.strip().casefold()
        if not a or not b:
            return False
        if a == b or a in b or b in a:
            return False
        return True

    pairs = list(combinations(spellings, 2))
    raw_vetoes = sum(1 for a, b in pairs if raw_conflict(a, b))
    alias_vetoes = sum(
        1 for a, b in pairs
        if brand_conflict(normalize_brand(a), normalize_brand(b))
    )
    assert alias_vetoes <= raw_vetoes, (
        f"folding manufactured vetoes: raw={raw_vetoes}, alias-aware={alias_vetoes}"
    )
    # …and the fold only helps where the map speaks: every dissolved veto is
    # exactly a seeded-family pair (shared token AFTER the fold).
    for a, b in pairs:
        if raw_conflict(a, b) and not brand_conflict(normalize_brand(a), normalize_brand(b)):
            shared = normalize_brand(a) & normalize_brand(b)
            assert shared, f"{a!r}/{b!r} dissolved without sharing a folded token"
            assert shared & _family_targets(), (
                f"{a!r}/{b!r} dissolved on a token the map does not own: {sorted(shared)}"
            )


def _family_targets() -> frozenset[str]:
    return frozenset(brand_aliases().values())


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("A SHOC", "Accelerator", False),
        ("Accelerator", "adrenaline shoc", False),
        ("TechUSA", "BioTech USA", False),
        ("Kevytolo", "Olvi", False),
        ("FitAid", "LifeAid", False),
        ("Ting", "DG", False),
        ("ball", "Hiball Energy", False),
        ("hi", "Hi Ball", False),
        ("Acme", "Other", True),
        ("Goat Fuel", "Sioux City", True),
    ],
)
def test_the_seeded_families_and_the_declined_stays_on_the_record(
    left: str, right: str, expected: bool
) -> None:
    """The whole seeded star map, both consumption-site directions, plus the
    measured decline set: the eight granted families dissolve, real
    competitors keep firing — on the SAME predicate every gate above uses."""
    assert brand_conflict(normalize_brand(left), normalize_brand(right)) is expected


def test_veto_asymmetry_never_creates_a_conflict_against_outsiders() -> None:
    """The structural one-liner behind the property: a fold can only ADD the
    family key, so against any outside brand the alias-aware verdict always
    BEATS the raw one — it can dissolve a raw conflict (a family member vs
    its own canonical is no conflict), never MINT one (a conflict the raw
    spelling did not already carry). Distinct-brand vetoes must therefore
    keep firing: dissolving is the doctrine's ONLY permitted movement."""
    insiders = ("A SHOC", "Accelerator", "Adrenaline Shoc", "Ting", "Hi Ball")
    outsiders = ("Acme", "Bolt", "Crisp Terra", "Unrelated Ltd")

    def raw_conflict(left: str, right: str) -> bool:
        a = left.strip().casefold()
        b = right.strip().casefold()
        if not a or not b:
            return False
        if a == b or a in b or b in a:
            return False
        return True

    for left in insiders:
        for right in outsiders:
            verdict = brand_conflict(normalize_brand(left), normalize_brand(right))
            assert not (verdict and not raw_conflict(left, right)), (
                f"{left!r} vs {right!r}: the fold minted a conflict against an"
                " outsider — veto asymmetry broken"
            )
