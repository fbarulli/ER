"""06: Tiered exact-duplicate dedupe — deliberate representatives, not blind drops.

05b found 13,519 rows in retailer+title duplicate groups, but only 4,439 also
match on price — so ~9,080 same-title rows at one retailer are DISTINCT
marketplace offers (Gittigidiyor-style sellers), not scrape glitches.
Collapsing them is PRICE-AGGREGATION, not noise-removal, so the representative
is chosen deliberately: has barcode (trusted identity) > most complete >
lowest price.

Tiers (each operates on the rows surviving the previous tier):
  T1 retailer+barcode     same product at one retailer -> 1 row (barcode is
                          ground truth, highest confidence).
  T2 retailer+title+price identical everything -> 1 row (lossless).
  T3 retailer+title       varying price -> 1 deliberate representative; the
                          group is FLAGGED as an ambiguous offer (pv) so the
                          price-collapse is auditable, never silent.

The deduped dataset is the MATCHING-stage input (step 03+); the raw export
stays the source of truth (load_dataset unchanged).

Writes:
  data/dataset_deduped.csv             deduped dataset (pipeline input)
  data/sku_to_rep.csv                  raw SKU (product_id) -> rep_id
  results/06_dedupe_summary.csv        per-tier counts
  results/06_ambiguous_offer_groups.csv  retailer+title >1 price
  results/06_dedupe_removals.csv       one review row per removed raw SKU
  results/manifests/dedupe.json        per-stage manifest, written LAST
                                        (SILENT_DROPS task 4 — the stage's
                                        completion marker: closure
                                        input_rows == output_rows + dropped,
                                        every output sha256-pinned, all
                                        writes atomic)
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from core.common import DATA_PATH, SEED, F, ensure_parent, load_dataset
from core.manifest import atomic_write_csv, begin_manifest, finish_manifest

# SSOT normalization reused from the record-linkage lane (safe: strips pack/
# count noise, PRESERVES flavor/roast/brand tokens) for the T1.5 product-
# identity check on malformed-barcode groups.
from core.record_linkage import strip_pack_multiplicity  # noqa: E402
from pipeline import normalize_text  # noqa: E402
from core.critical_attributes import (  # noqa: E402
    DECLARED_FLAVOR_LEXICON,
    FLAVOR_LEXICON,
)

# output paths from the config SSOT (files.*) — were hardcoded here, the
# only filenames in the tree outside config/paths.yaml
CSV_SUMMARY = F["dedupe_summary"]
CSV_OFFERS = F["ambiguous_offer_groups"]
CSV_REMOVALS = F["removals"]
DEDUPED_PATH = F["dataset_deduped"]
SKU_TO_REP_PATH = F["sku_to_rep"]

HELPERS = ["_price", "_nonnull", "_has_bc", "_t2_bc", "_bc_valid"]


# Diet/zero/sugar-free title markers. A group is a genuine split ONLY when
# these markers are MIXED across rows (some rows diet, some regular) — a
# marker present in EVERY row (all sugar-free) is a shared product trait, not
# a conflict. Deliberately EXCLUDES "caffeinated"/"decaf"/"cold brew"/"light"
# as noise qualifiers (verified: caffeinated etc. are listing noise, not
# product splitters for these malformed-barcode rows).
_DIET_MARKER = re.compile(
    r"\b(diet|zero|sugar\s*free|no\s*sugar|low\s*cal|unsweetened)\b", re.I
)


def _attr_token_set(attr: str, key: str) -> set[str]:
    """Extract a canonical token SET from a `Key: value; ...` attribute cell.

    NORMALIZE FIRST (verified order of operations): the value is normalize_text'd
    then split into word tokens, so "coffee, vanilla" and "vanilla coffee" are
    the SAME set. Comparing sets (not ordered strings) is what lets a reworded
    title (Gevalia "Cold Brew Vanilla Concentrate" vs "Vanilla Cold Brew
    Concentrate") collapse correctly.
    """
    m = re.search(key + r"\s*:\s*([^;]+)", str(attr), re.I)
    if not m:
        return set()
    return set(re.findall(r"[a-z0-9]+", normalize_text(m.group(1))))


def _flavor_roast_sets(sub: pd.DataFrame) -> list[set[str]]:
    """Per-row identity sets = declared flavor tokens UNION roast-type tokens.

    Roast (French/Italian/espresso) is a coffee product splitter that lives in
    the `Roast Type` attribute, NOT the `Flavour` field — Cool Brew "French
    Roast" vs "Vanilla" both carry Flavour: coffee, so flavor alone cannot
    separate them. Unioning roast-type catches the genuine split while a
    missing field (empty set) stays compatible.
    """
    attr_col = "attributes" if "attributes" in sub.columns else "attribute"
    return [
        _attr_token_set(a, r"flavou?r") | _attr_token_set(a, r"roast\s*type")
        for a in sub[attr_col]
    ]


def _subset_compatible(sets: list[set[str]]) -> bool:
    """True if every non-empty identity set is subset-compatible.

    Two rows conflict ONLY when each has a token the other lacks (e.g. lemon
    vs lime, french vs vanilla) — then it is a genuine product split. An
    OMISSION (one title/attr is a subset of the other, e.g. "lemon" vs
    "lemon, lime") is the truncated-listing noise we must ALLOW, not veto.
    """
    nonempty = [s for s in sets if s]
    if not nonempty:
        return True
    for i in range(len(nonempty)):
        for j in range(i + 1, len(nonempty)):
            a, b = nonempty[i], nonempty[j]
            if (a - b) and (b - a):
                return False
    return True


# STEP-2 adjudicated verdicts (owner review 2026-09-29). The 10 residual
# groups that Step 1 (normalized brand+flavor+roast+diet) cannot decide
# unambiguously were reviewed on their ORIGINAL (unnormalized) data — full
# title, attributes, brand, category, price. 8 are genuine product splits
# (keep separate), 2 are same-product where a listing qualified a shared
# trait (collapse). Keyed by (retailer, barcode) so the T1.5 loop can look up
# the verdict directly.
_STEP2_COLLAPSE = {  # (retailer, barcode) -> same product, collapse
    # L&A All Cranberry Juice: 3 listings, all L&A cranberry juice 32oz;
    # "No Sugar Added" on one is a shared listing qualifier, not a split.
    ("amazon", "41755098003"),
    # Orange Crush Sugar Free Singles: both are Orange Crush sugar-free
    # orange drink-mix singles; the second title is listing spam ("...Fashion
    # Accessories") over the same product.
    ("amazon", "72392329915"),
}
_STEP2_KEEP = {  # (retailer, barcode) -> genuine split, keep separate
    # Whole Foods nutrient water vs Zero-Calorie Lemonade variant (lemon).
    ("Wholefoods", "99482464950"),
    # Ozarka sparkling water: Lemon vs Lime.
    ("amazon", "22592446530"),
    # Cool Brew cold-brew concentrate: French Roast vs Vanilla.
    ("amazon", "53721632036"),
    # Montellier mineral water: Lemon vs Lime.
    ("amazon", "56918000304"),
    # Pennsylvania Dutch birch beer: regular vs Diet.
    ("amazon", "71573024687"),
    # Zephyrhills sparkling water: Lemon vs Lime vs Spring.
    ("amazon", "73430910713"),
    # Stewart's root beer: Original vs Diet.
    ("amazon", "98794313048"),
    # Thick & Easy cranberry: Hormel vs Thick & Easy (different brands).
    ("amazon", "99429158133"),
}


def _same_product_by_title(sub: pd.DataFrame, retailer: str, barcode: str) -> bool:
    """Two-step product-identity decision for a (retailer, barcode) group.

    STEP 1 — normalize first, then collapse on the normalized signal:
      brand agrees (normalized) + flavor ∪ roast attribute sets are
      subset-compatible (omissions allowed, genuine lemon-vs-lime / french-vs-
      vanilla vetoed) + diet/zero/sugar-free markers are NOT mixed across rows.
      Category is deliberately NOT a veto: it is as noisy as the attributes on
      these malformed-barcode rows (Reconstituted vs Not-from-Concentrate
      juice are the same product).

    STEP 2 — residual adjudication on ORIGINAL data: the small set of groups
    Step 1 routes to review are decided explicitly by owner (see _STEP2_*).

    `sub` must contain columns: title, brand, attribute. Returns True when the
    group is the same product and should collapse.
    """
    key = (retailer, barcode)
    if key in _STEP2_KEEP:
        return False
    if key in _STEP2_COLLAPSE:
        return True

    # STEP 1
    brands = set(sub["brand"].fillna("").str.lower())
    if len(brands) != 1:
        return False
    if not _subset_compatible(_flavor_roast_sets(sub)):
        return False
    diet_flags = [bool(_DIET_MARKER.search(t)) for t in sub["title"].astype(str)]
    if len(set(diet_flags)) > 1:
        return False
    return True


def main() -> None:
    for _out in (CSV_SUMMARY, CSV_OFFERS, CSV_REMOVALS, DEDUPED_PATH, SKU_TO_REP_PATH):
        ensure_parent(_out)
    # Stage manifest (SILENT_DROPS task 4) — begin BEFORE the work: the
    # raw export is hashed now (53MB, chunked) so the record pins exactly
    # what this stage read. Seed = the SSOT seed (lib.common.SEED); the
    # tiered collapse below is deterministic, no RNG is consumed.
    manifest = begin_manifest("dedupe", inputs=[DATA_PATH], seed=SEED)
    df = load_dataset()
    n0 = len(df)
    work = df.assign(
        _price=pd.to_numeric(df["price"], errors="coerce"),
        _nonnull=df.notna().sum(axis=1),
        _has_bc=(df["barcode"].fillna("").str.len() > 0).astype(int),
    )

    # parent[i] = original row index of the surviving representative for row i.
    # Updated per tier and resolved transitively at the end (a T1 survivor may
    # itself be collapsed by T2/T3).
    parent = {i: i for i in work.index}

    def collapse(frame: pd.DataFrame, groups: list[str], sort_cols: list[str],
                 ascending: list[bool]) -> tuple[pd.DataFrame, pd.Index]:
        """Collapse each group to one representative; record the mapping.

        ONE sorted pass + ONE groupby-min reduction: sort by the
        representative-preference columns, then every row learns its
        group's minimum position (= the first row in preference order —
        the same first-of-group semantics drop_duplicates keep="first"
        had, with groupby dropna=False so NaN==NaN grouping matches).
        The old form ran drop_duplicates AND a per-group Python loop that
        re-walked every group just to fill parent[]; measured 465ms ->
        27ms on the 71.6k-row corpus.
        """
        ordered = frame.sort_values(sort_cols, ascending=ascending,
                                    na_position="last")
        pos = pd.Series(np.arange(len(ordered)), index=ordered.index)
        rep_pos = pos.groupby(
            [ordered[c] for c in groups], dropna=False
        ).transform("min")
        survivors = ordered.loc[rep_pos == pos]
        parent.update(
            dict(
                zip(
                    ordered.index,
                    ordered.index.to_numpy()[rep_pos.to_numpy()],
                    strict=True,
                )
            )
        )
        dropped = frame.index.difference(survivors.index)
        return survivors, dropped

    summary = []

    # T1: retailer+barcode -> one row (ground-truth identity), ONLY for rows
    # that actually have a barcode. Rows with a MISSING barcode are NOT
    # collapsed ("no barcode" is not "same barcode"). And only for rows
    # whose barcode PASSES the GS1 checksum (owner ruling, src/core/gtin.py):
    # an invalid barcode is export noise, not identity — 103 retailer+
    # barcode groups carried >1 distinct title on a checksum-fail barcode
    # and would silently merge different products. Invalid-barcode rows
    # are not dropped: they fall through to T2/T3 title-based tiers.
    from core.gtin import barcode_validity

    work = work.assign(
        _bc_valid=barcode_validity(
            work["barcode"].fillna("").astype(str).str.strip()
        ).to_numpy()
    )
    with_bc = work[(work["_has_bc"] == 1) & (work["_bc_valid"])]
    t1_bc_invalid = work[(work["_has_bc"] == 1) & (~work["_bc_valid"])]
    no_bc = work[work["_has_bc"] == 0]
    n_t1_skipped = len(t1_bc_invalid)
    t1, dropped1 = collapse(with_bc, ["retailer", "barcode"],
                            ["_nonnull", "_price"], [False, True])
    work = pd.concat([t1, t1_bc_invalid, no_bc])
    summary.append({"tier": "T1 retailer+barcode",
                    "dropped_rows": len(dropped1),
                    "skipped_checksum_invalid": n_t1_skipped})

    # T1.5: same retailer + same MALFORMED (checksum-invalid) barcode + same
    # product -> one row. T1 refuses to collapse on an invalid barcode because
    # "invalid barcode is export noise, not identity" — but a malformed barcode
    # that is byte-identical at one retailer is still a strong candidate, and
    # the title is the arbiter. We collapse a group ONLY when the normalized,
    # pack-stripped titles agree on product identity (no flavor/roast/brand
    # token differs — see _same_product_by_title). This recovers the 97 groups
    # measured in the dedupe invalid-barcode audit (2026-09-29) while never
    # merging genuinely different products (Cool Brew French Roast vs Vanilla,
    # Montellier Lemon vs Lime, Ginseng Up vs Natural Ginger Ale, ...).
    t15_dropped = []
    t15_kept = []
    t15_groups = 0
    for (retailer, barcode), sub in t1_bc_invalid.groupby(
        ["retailer", "barcode"], sort=False
    ):
        if len(sub) <= 1:
            t15_kept.append(sub)
            continue
        if _same_product_by_title(sub, retailer, barcode):
            # Representative: most-complete, then lowest price (same preference
            # order as the other tiers), first in that order wins.
            order = sub.sort_values(
                ["_nonnull", "_price"], ascending=[False, True],
                na_position="last",
            )
            rep = order.iloc[[0]]
            rep_idx = order.index[0]
            for idx in sub.index:
                parent[idx] = rep_idx
            t15_dropped.extend(sub.index.difference([rep_idx]))
            t15_kept.append(rep)
            t15_groups += 1
        else:
            t15_kept.append(sub)
    t15_kept = pd.concat(t15_kept)
    work = pd.concat([t1, t15_kept, no_bc])
    summary.append({"tier": "T1.5 retailer+malformed-barcode+same-product",
                    "dropped_rows": len(t15_dropped),
                    "collapsed_groups": t15_groups})

    # T2: retailer+title+price(+barcode) -> one row (lossless), ONLY for rows
    # that HAVE a price. NaN != NaN in the real world, so two missing-price rows
    # are not "identical everything" and must flow to T3's auditable
    # price-aggregation. "Lossless" is ENFORCED, not assumed: rows collapse only
    # when their barcodes agree (same non-empty barcode, or both missing) —
    # same title+price with DIFFERENT barcodes is a different product and
    # flows to T3's auditable path instead of being silently merged. The group
    # key uses NUMERIC _price so "10.0" and "10.00" are one price, matching
    # T3's numeric aggregation.
    with_price = work[work["_price"].notna()].copy()
    with_price["_t2_bc"] = with_price["barcode"].fillna("").astype(str)
    no_price = work[work["_price"].isna()]

    # Split T2 by barcode agreement WITHIN each (retailer,title,price) group:
    # an all-same-barcode (or all-missing) group collapses losslessly; a group
    # with >1 distinct barcode carries genuinely different products — those
    # rows all flow to T3 (kept here via keep_idx exclusion from collapse).
    bc_sig = with_price.groupby(["retailer", "title", "_price"], sort=False, dropna=False)["_t2_bc"].transform(
        lambda s: "1" if s.nunique() <= 1 else "0")
    t2_clean = with_price[bc_sig == "1"]
    t2_conflict = with_price[bc_sig == "0"]

    t2, dropped2 = collapse(t2_clean, ["retailer", "title", "_price", "_t2_bc"],
                            ["_nonnull", "_price"], [False, True])
    work = pd.concat([t2, t2_conflict.drop(columns=["_t2_bc"]), no_price])
    summary.append({"tier": "T2 retailer+title+price+barcode",
                    "dropped_rows": len(dropped2),
                    "deferred_to_t3": len(t2_conflict)})

    # T3: retailer+title with varying price -> deliberate representative + flag
    # "count" is GROUP SIZE ("size"), not pandas' NaN-excluding count: a group
    # with rows [10, 20, NaN] has 3 rows, and the audit CSV must say so.
    pv = work.groupby(["retailer", "title"])["_price"].agg(
        ["size", "nunique", "min", "max"]).rename(columns={"size": "count"})
    ambiguous = pv[(pv["count"] > 1) & (pv["nunique"] > 1)]
    t3, dropped3 = collapse(work, ["retailer", "title"],
                            ["_has_bc", "_nonnull", "_price"], [False, False, True])
    work = t3
    summary.append({"tier": "T3 retailer+title (price-aggregation)",
                    "dropped_rows": len(dropped3)})

    # ---- outputs --------------------------------------------------------------
    # Resolve representative pointers transitively (T1/T2 survivors may be
    # dropped by a later tier), then emit the raw-SKU -> rep mapping so the
    # deliverable notebook derives ITEM_ID from this step instead of re-deduping.
    for i in df.index:
        while parent[parent[i]] != parent[i]:
            parent[i] = parent[parent[i]]

    # errors="ignore": _t2_bc exists only on T2 survivors; a tier upstream may
    # legitimately not produce it.
    deduped = work.drop(columns=HELPERS, errors="ignore").reset_index(drop=True)
    atomic_write_csv(deduped, DEDUPED_PATH, index=False)
    print(f"wrote {DEDUPED_PATH} ({len(deduped):,} rows)")

    rep_pos = {idx: pos for pos, idx in enumerate(work.index)}
    sku_to_rep = pd.DataFrame({
        "product_id": df["product_id"].to_numpy(),
        "rep_id": [rep_pos[parent[i]] for i in df.index],
    })
    atomic_write_csv(sku_to_rep, SKU_TO_REP_PATH, index=False)
    print(f"wrote {SKU_TO_REP_PATH} ({len(sku_to_rep):,} rows)")

    # sanity: no (retailer,title) duplicates may remain, and every raw SKU
    # resolves to a valid representative.
    dups = int(deduped.duplicated(subset=["retailer", "title"]).sum())
    if dups:
        raise AssertionError(f"sanity FAILED: {dups} retailer+title dupes remain")
    if sku_to_rep["rep_id"].isna().any():
        raise AssertionError("sanity FAILED: some SKUs map to no representative")
    if set(sku_to_rep["rep_id"].unique()) != set(range(len(deduped))):
        raise AssertionError("sanity FAILED: rep_id coverage is not 0..n-1")
    print("  [PASS] no retailer+title duplicates remain; SKU->rep mapping complete")

    # One review row for every raw SKU deliberately collapsed by a tier.
    # Capture the tier at its direct parent update, before transitive
    # representative resolution obscures where the removal happened.
    removal_tier = {
        **{idx: "T1 retailer+barcode" for idx in dropped1},
        **{idx: "T1.5 retailer+malformed-barcode+same-product" for idx in t15_dropped},
        **{idx: "T2 retailer+title+price+barcode" for idx in dropped2},
        **{idx: "T3 retailer+title (price-aggregation)" for idx in dropped3},
    }
    removals = sku_to_rep.loc[list(removal_tier)].copy()
    removals["tier"] = [removal_tier[idx] for idx in removals.index]
    removals = removals[["product_id", "rep_id", "tier"]].reset_index(drop=True)
    expected_removals = n0 - len(deduped)
    if len(removals) != expected_removals:
        raise AssertionError(
            f"sanity FAILED: removals has {len(removals):,} rows, expected "
            f"{expected_removals:,}"
        )
    atomic_write_csv(removals, CSV_REMOVALS, index=False)
    print(f"wrote {CSV_REMOVALS} ({len(removals):,} review rows)")

    summary.append({"tier": "TOTAL dropped", "dropped_rows": n0 - len(deduped)})
    summary.append({"tier": "TOTAL remaining", "dropped_rows": len(deduped)})
    summary_df = pd.DataFrame(summary)
    atomic_write_csv(summary_df, CSV_SUMMARY, index=False)
    print(f"wrote {CSV_SUMMARY} (display table, {len(summary)} rows)")

    ambiguous_out = ambiguous.rename(
        columns={"count": "rows", "nunique": "distinct_prices"}).reset_index()
    atomic_write_csv(ambiguous_out, CSV_OFFERS, index=False)
    print(f"wrote {CSV_OFFERS} (display table, {len(ambiguous_out)} rows) "
          f"— {len(ambiguous_out):,} ambiguous-offer groups flagged")

    # ---- row accounting (SILENT_DROPS task 4; capture-only) ────────────────
    # The three tier counters partition the frame at each step, so the
    # closure input_rows == output_rows + sum(dropped) holds by
    # construction (finish_manifest asserts it before publishing):
    # 71,623 == 61,529 + (1,943 + 2,245 + 5,906).
    #
    # The two "deferred" populations are NOT drops and deliberately
    # excluded from `dropped`:
    #   skipped_checksum_invalid (T1) — 3,715 checksum-fail barcode rows
    #     are concatenated BACK into the work frame; T1.5 collapses the
    #     same-product ones (counted under dropped.t1_5_*), the rest fall
    #     through to the title tiers.
    #   deferred_to_t3 (T2) — 465 barcode-conflicting rows likewise
    #     re-enter the frame and are settled by T3's counter.
    # Recording them under their own keys (outside `dropped`) keeps the
    # audit trail complete without breaking the closure invariant.
    row_accounting = {
        "input_rows": n0,
        "output_rows": len(deduped),
        "dropped": {
            "t1_retailer_barcode": len(dropped1),
            "t1_5_retailer_malformed_barcode_same_product": len(t15_dropped),
            "t2_retailer_title_price_barcode": len(dropped2),
            "t3_retailer_title_price_aggregation": len(dropped3),
        },
        "skipped_checksum_invalid": n_t1_skipped,
        "deferred_to_t3": len(t2_conflict),
        "ambiguous_offer_groups": len(ambiguous_out),
    }
    manifest_path = finish_manifest(
        manifest,
        outputs=[DEDUPED_PATH, SKU_TO_REP_PATH, CSV_SUMMARY, CSV_OFFERS, CSV_REMOVALS],
        row_accounting=row_accounting,
        expected_outputs=[
            F["dataset_deduped"].name, F["sku_to_rep"].name,
            F["dedupe_summary"].name, F["ambiguous_offer_groups"].name,
            F["removals"].name,
        ],
    )
    print(f"wrote {manifest_path} — stage manifest (closure "
          f"{row_accounting['input_rows']:,} == {row_accounting['output_rows']:,} "
          f"+ {sum(row_accounting['dropped'].values()):,} dropped)")

    print(f"\n{len(df):,} rows -> {len(deduped):,} after tiered dedupe "
          f"(dropped {n0 - len(deduped):,}); "
          f"{len(ambiguous_out):,} retailer+title groups were "
          "price-varying offers (flagged, not silently merged)")


if __name__ == "__main__":
    main()
