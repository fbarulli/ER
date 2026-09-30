#!/usr/bin/env python3
"""Recreate the 2026-09-29 within-GTIN brand-variant measurement and seed
config/vocabulary.json "brand_aliases" from it (SSOT write).

Owner doctrine (TODO.md:668-676; owner-ruled measurement 2026-09-29: "62/62
groups survive case/accent folding"): brand variants are SEMANTIC rebrands /
parent-company / sister-name filings, NOT typos. The alias map is therefore
an EXPLICIT reviewed pair list in config — never edit-distance fuzzy matching.
Reproduced on the current corpus (71,623 rows): 71 checksum-valid GTIN groups
carry 2+ distinct non-empty normalized brand strings (65 two-brand + 6
three-brand, 341 rows), and 49 distinct within-group brand-string pairs fire
core.product_identity.brand_conflict at current HEAD. The reconciled owner/HEAD
figure (49 veto-firing pair GTINs) is the alias population this map addresses
plus genuinely mixed barcodes, which stay vetoed by design.

Fold semantics (veto-asymmetry doctrine, conservative):
- Only tokens ALREADY present on a brand side are ADDED; nothing is dropped,
  so a fold can make two brand token sets share a token or nest — it can
  never make them disjoint. Every fold can only dissolve a spurious brand
  veto, never mint one.
- One token-keyed star per measured family: KEY -> family canonical; keys that
  appear in the fold of both sides of a measured pair resolve that pair.
- Corollary (checked by _rarity_audit before any write): a key token must
  mark, across ALL 2,261 distinct normalized brand strings of the corpus, the
  territory of its own family only. A key that orbits unrelated brand strings
  (measured: 'energy' 13 strings, 'bio' 14, 'life' 8, the 1-token mistype
  trap for single-token brands) is REFUSED — the family loses that key, or is
  declined outright. This is why the map contains no generic common token.
- Whole-GTIN multi-brand groups (e.g. 8414100000013 la casera/may tea/sunny
  delight) get NO fold: an alias would fold real competitors. Their vetoes
  stay live.

Pure: prints the derived map + counts; writes ONLY the "brand_aliases" key
and its "brand_aliases_provenance" provenance block, then re-reads through
core.common and asserts identical. Run:

    PYTHONPATH=src .venv/bin/python scripts/seed_brand_aliases.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

TRAIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN_ROOT / "src"))

from core.common import DATA_PATH, load_dataset, _read_vocabulary, VOCABULARY_CONFIG_PATH
from core.gtin import normalize_and_validate_gtin
from core.manifest import sha256_file
from core.product_identity import brand_conflict, normalize_brand
from core.text import normalized_attribute_text

VOCAB_PATH = VOCABULARY_CONFIG_PATH

# ── the measured permission grants ─────────────────────────────────────────
# key: the family's canonical fold token — every family member's token set
# ends up containing it. sources: the distinct normalized brand strings of
# the family inside the within-GTIN census. gtins: the checksum-valid
# barcodes whose rows EVIDENCE the pair (dashboard/evidence/identity/
# 03_measurement_and_packaging_context.json brand_variation_groups; titles
# verified as one brand written two ways, never two competing brands).
FAMILIES: tuple[dict[str, object], ...] = (
    {
        "key": "shoc",
        "sources": ("a shoc", "adrenaline shoc", "accelerator"),
        "gtins": (
            "0810014530017", "0810014530031", "0810014530048", "0810014530055",
            "0810014530062", "0810014530130", "0810014530178", "0810014530314",
            "0810014530345", "0810014530369", "0810014530475", "0810014530482",
            "0810014530499", "0810014530505", "0810014530543", "0810014530673",
            "0810014530680", "0810014530710",
        ),
        "note": (
            "A SHOC -> Adrenaline Shoc -> Accelerator: one product line's "
            "rebrand chain, retailers still file the same barcode under all "
            "three names (14 distinct within-group pairs, all spurious "
            "brand vetoes before seeding)"
        ),
    },
    {
        "key": "dg",
        "sources": ("dg", "ting"),
        "gtins": ("0858629001065",),
        "note": (
            "DG Ting Grapefruit Soda: retailers filed the soda name 'Ting' "
            "as the brand; DG and Ting are one label family"
        ),
    },
    {
        "key": "hiball",
        "sources": ("hi ball", "hiball energy"),
        "gtins": (
            "0852421006075", "0852421006150", "0897351000328",
            "0897351000427", "0897351000823", "0897351000878",
        ),
        "note": (
            "amazon splits the brand into 'Hi- Ball'; token folds hi->hiball "
            "and ball->hiball close the split across 6 GTINs"
        ),
    },
    {
        "key": "lifeaid",
        "sources": ("fitaid", "lifeaid"),
        "gtins": ("0857886006004", "0857886006424"),
        "note": (
            "LIFEAID Bev Co files its FITAID recovery drink under either "
            "brand; titles carry both strings on one barcode (sister brands)"
        ),
    },
    {
        "key": "olvi",
        "sources": ("kevytolo", "olvi"),
        "gtins": (
            "6419800052678", "6419800052807", "6419800053071",
            "6419806053204",
        ),
        "note": (
            "Olvi owns KevytOlo; the same barcode is filed under either name "
            "(parent company, 4 GTINs)"
        ),
    },
    {
        "key": "biotech",
        "sources": ("bio techusa", "biotech usa"),
        "gtins": ("5999076206513",),
        "note": "BioTechUSA spacing variant across Cdiscount/Notino exports",
    },
)
# Whole-group DECLINES (no fold granted): the group mixes genuinely distinct
# brands on one barcode (whole-barcode review holds, not alias pairs), the
# family is unverifiable (hierarchy/identity unresolved), or — for families
# granted but with a key token lost to the rarity audit — the fold key would
# be a common word whose territory escapes the family.
DECLINED_REASONS: dict[str, str] = {
    "0850010701257": (
        "goat fuel vs sioux city over one barcode; energy-drink and root-beer "
        "titles differ — distinct pairs, not a rebrand"
    ),
    "0865891000184": (
        "sunniva (seller/roaster string) vs kitu super espresso (product "
        "brand); hierarchy unverified (also no conflict fires: disjoint "
        "multi-token sides both nest nothing — exempt via subset rule only "
        "on one side; no fold without owner ruling)"
    ),
    "0868784000346": (
        "3d vs blue energy ('3D Blue Energy...' title split); each name is "
        "the BRAND word of other real brands (measured: 'blue' marks 9 brand "
        "strings, 'energy' 13) — folding on d/blue/energy keys would "
        "collapse unrelated brands; owner ruling required"
    ),
    "11982760": "reviewed mixed-GTIN junk group (Mat Smart; held lineage)",
    "3301591000040": (
        "coteaux nantais vs planet bio on one barcode; distinct juice brands"
    ),
    "5010889010040": (
        "KA vs 'K A': the family's only fold keys are 'k'/'a', whose "
        "measured territories escape the family ('k' also marks 'k classic', "
        "'a' also marks 'a shoc'). Rarity audit refuses them; the 2-GTIN "
        "veto stays until the owner rules on a two-token key schema"
    ),
    "8410055000009": "distinct Spanish water brands share one barcode",
    "8410128270070": "distinct brands share one barcode (bifrutas/pascual)",
    "8410128776718": "distinct brands share one barcode (bifrutas/pascual)",
    "8410171000006": (
        "alcampo rows mixing; 'via nature' is a separate brand line"
    ),
    "8410749000001": "distinct brands share one barcode (lambda/mondariz)",
    "8414100000013": (
        "WHOLE-BARCODE multi-brand group (la casera / may tea / sunny "
        "delight): an alias would fold real competitors — conflicts stay"
    ),
    "8426633001405": (
        "eco+ vs int-salim: folding 'eco' would also fold ecomil/ecor/"
        "ecosana (measured terrace of 5+ 'eco'-prefixed strings) — unsafe"
    ),
    "8429359000004": "distinct brands share one barcode (primavera/tampico)",
    "8433963000008": (
        "WHOLE-BARCODE multi-brand group (aquabona / nordic mist / royal "
        "bliss): conflicts stay live"
    ),
    "8858947400108": (
        "eloa is aloe drink for life's newcomer name, but every safe fold key "
        "outside the brand's own words is a COMMON token (measured: 'aloe' 8 "
        "strings, 'life' 8, 'drink' 11); the only family-unique token is "
        "'eloa' itself, whose side already shares nothing — folding would "
        "require an unattested 'eloa' KEY written by hand, not measured; "
        "declined pending owner review of aloe vs eloa"
    ),
    "0818972021318": (
        "mixed GTIN: '100' percent vs 'Coco Goods' split one number across "
        "two brands; identity unresolved (review hold lineage)"
    ),
    "5741000164532": (
        "faxe kondi carries 'Booster' inside its product title while a "
        "different 'Booster' brand exists elsewhere (census: token-shared "
        "against unrelated strings) — declined as unsafe"
    ),
    "8904061111090": (
        "KA vs 'K A': same key-territory refusal as 5010889010040"
    ),
}


def _measurement(df) -> dict[str, list[str]]:
    """Checksum-valid GTIN groups carrying 2+ distinct non-empty normalized
    brand strings (the raw string level the owner adjudicated on)."""
    facts = normalize_and_validate_gtin(df["barcode"])
    mask = facts["gtin_structurally_valid"].to_numpy()
    valid = df[mask].assign(_g=facts["gtin_clean"][mask].to_numpy())
    groups = valid.groupby("_g")["brand"].apply(
        lambda s: sorted({b for b in (normalized_attribute_text(x) for x in s) if b})
    )
    return {g: b for g, b in groups.items() if len(b) > 1}


def _pre_seed_veto_census(variant: dict[str, list[str]]) -> int:
    """Within-group brand vetoes with the alias map OFF.

    Re-run stability: the file may already carry a seeded map (idempotent
    re-runs), so the pre-seed lens temporarily hides brand_aliases from
    vocabulary() and clears the lru_cache, folding through normalize_brand
    exactly as the brand-fresh world did. Restored after counting.
    """
    import core.product_identity as pi

    original_vocabulary = pi.vocabulary

    def _without_aliases():
        return {
            k: v for k, v in original_vocabulary().items()
            if k != "brand_aliases"
        }

    pi.vocabulary = _without_aliases
    try:
        pi.brand_aliases.cache_clear()
        count = 0
        for brands in variant.values():
            folds = [normalize_brand(b) for b in brands]
            for i, fa in enumerate(folds):
                for fb in folds[i + 1:]:
                    count += bool(brand_conflict(fa, fb))
        return count
    finally:
        pi.vocabulary = original_vocabulary
        pi.brand_aliases.cache_clear()


def _territories(df) -> dict[str, set[str]]:
    """Token -> set of distinct normalized brand strings containing it."""
    out: dict[str, set[str]] = defaultdict(set)
    strings = {b for b in map(normalized_attribute_text, df["brand"]) if b}
    for brand in strings:
        for token in normalize_brand(brand):
            out[token].add(brand)
    return out


def main() -> None:
    df = load_dataset()
    variant = _measurement(df)
    n_two = sum(1 for b in variant.values() if len(b) == 2)
    n_three = len(variant) - n_two
    grant_gtins = {g for fam in FAMILIES for g in fam["gtins"]}
    print(
        f"[seed] measured: {len(variant)} within-GTIN brand-variant groups "
        f"({n_two} two-brand / {n_three} three-brand), "
        f"{sum(len(b) for b in variant.values())} rows"
    )

    # Pre-seed veto census (provenance evidence): within-group brand-string
    # pairs that fire a conflict with the alias map OFF (re-run stable).
    pre_veto_pairs = _pre_seed_veto_census(variant)
    print(f"[seed] pre-seed within-group brand vetoes: {pre_veto_pairs}")

    territories = _territories(df)
    # Derive the map: every family token whose territory is EXACTLY the
    # family's own strings becomes a KEY folding to the canonical token.
    candidates: dict[str, str] = {}
    family_strings: dict[str, set[str]] = {}
    for fam in FAMILIES:
        key = str(fam["key"])
        strings = set(fam["sources"])
        family_strings[key] = strings
        for token in sorted({t for s in strings for t in normalize_brand(s)}):
            if token == key:
                continue  # self-binding is a no-op fold; never seeded
            if territories.get(token, set()) <= strings:
                candidates[token] = key
            else:
                print(
                    f"[seed] drop key {token!r}: territory escapes its family "
                    f"({sorted(territories[token] - strings)!r}) — "
                    "veto-asymmetry guard"
                )
    # Every family must keep at least one key, and every MEASURED within-
    # family pair must still become conflict-free after folding; else abort.
    for fam in FAMILIES:
        key = str(fam["key"])
        kept = [k for k, v in candidates.items() if v == key]
        assert kept, f"family {key} lost every fold key to the rarity audit"
    alias_map = dict(sorted(candidates.items()))

    # Post-seed check with the WRITTEN map live: every granted family pair
    # must be conflict-free now (the dissolved count is hard provenance).
    import core.product_identity as pi

    original_vocabulary = pi.vocabulary
    pi.vocabulary = lambda: _read_vocabulary(VOCAB_PATH)
    try:
        pi.brand_aliases.cache_clear()
        still_firing = [
            (gtin, a, b)
            for gtin, brands in variant.items()
            for i, a in enumerate(brands)
            for b in brands[i + 1:]
            if brand_conflict(normalize_brand(a), normalize_brand(b))
        ]
        assert all(
            g not in grant_gtins for g, _a, _b in still_firing if False
        )
        for gtin, a, b in still_firing:
            assert gtin not in grant_gtins, (
                f"seeded family pair still vetoing: {gtin} {a!r} x {b!r}"
            )
        print(
            "[seed] post-seed within-group brand vetoes still firing: "
            f"{len(still_firing)} (all declined/no-fold groups — the right "
            "world for veto-asymmetry)"
        )
        for gtin, a, b in still_firing:
            print(f"    still vetoed: {gtin} {a!r} x {b!r}")
    finally:
        pi.vocabulary = original_vocabulary
        pi.brand_aliases.cache_clear()

    document = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    document["brand_aliases"] = alias_map
    document["brand_aliases_provenance"] = {
        "generated_by": "scripts/seed_brand_aliases.py",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_sha256": sha256_file(DATA_PATH),
        "measurement_date": "2026-09-29",
        "measured_at_commit": "0452692",
        "current_corpus_at_head": "5001027",
        "measurement": (
            "within-GTIN brand-variant census: checksum-valid barcodes with "
            "2+ distinct non-empty normalized brand names. Owner-ruled "
            "2026-09-29 as 62 groups; reproduction on the current corpus: "
            f"{len(variant)} groups ({n_two} two-brand / {n_three} "
            "three-brand), 341 rows, "
            f"{pre_veto_pairs} distinct within-group brand vetoes before "
            "seeding"
        ),
        "counts": {
            "variant_groups_measured": len(variant),
            "variant_groups_two_brand": n_two,
            "variant_groups_three_brand": n_three,
            "within_group_brand_vetoes_before_seeding": pre_veto_pairs,
            "alias_families": len(FAMILIES),
            "alias_entries": len(alias_map),
            "declined_group_gtins": len(DECLINED_REASONS),
        },
        "veto_asymmetry": (
            "conservative token-keyed star folds: normalize_brand ADDS alias "
            "keys and drops nothing, so a fold can only make two brand token "
            "sets share a token or nest — never disjoint. Key tokens are "
            "corpus-rare (audited against all brand strings before write); "
            "whole-barcode multi-brand groups are NOT seeded and their brand "
            "conflicts stay live. No edit-distance fuzzy matching anywhere."
        ),
        "families": [
            {
                "key": fam["key"],
                "sources": list(fam["sources"]),
                "evidence_gtins": list(fam["gtins"]),
                "note": fam["note"],
            }
            for fam in FAMILIES
        ],
        "declined_groups": DECLINED_REASONS,
    }
    VOCAB_PATH.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[seed] wrote brand_aliases ({len(alias_map)} entries) + provenance")
    print("[seed] final alias map:")
    for token, target in alias_map.items():
        print(f"  {token} -> {target}")
    # Re-read through the validated SSOT loader: schema-clean + identical.
    reloaded = _read_vocabulary(VOCAB_PATH)
    assert dict(reloaded["brand_aliases"]) == alias_map
    assert (
        reloaded["brand_aliases_provenance"]["counts"]["alias_families"]
        == len(FAMILIES)
    )
    print("[seed] re-read via core.common: identical, validation clean")


if __name__ == "__main__":
    main()
