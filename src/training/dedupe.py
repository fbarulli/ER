"""06: Tiered exact-duplicate dedupe — deliberate representatives, not blind drops.

05b found 13,519 rows in retailer+title duplicate groups, but only 4,439 also
match on price — so ~9,080 same-title rows at one retailer are DISTINCT
marketplace offers (Gittigidiyor-style sellers), not scrape glitches.
Collapsing them is PRICE-AGGREGATION, not noise-removal, so the representative
is chosen deliberately using gtin presence and descriptor completeness;
price does not rank representatives.

Tiers (each operates on the rows surviving the previous tier):
  T1  retailer+gtin      same product at one retailer -> 1 row (gtin is
                           ground truth, highest confidence).
  T1.5 retailer+malformed   same retailer + same CHECKSUM-INVALID gtin +
      gtin               same product -> 1 row. Identity decided by the
                           descriptor bundle (core.sku_identity), never by
                           price or URL.
  T2  retailer+title+       matching identity partition -> 1 row. Price is
      identity partition    NOT part of the key (it is a seller attribute);
                           same retailer+title+gtin at two prices is the
                           same product offered twice, which is T3's
                           price-aggregation and is flagged, not silent.
  T3  retailer+title +      varying price -> one deliberate representative per
      identity partition   IDENTITY PARTITION, flagged as an ambiguous offer
                           (pv) so the price-collapse is auditable.

The identity partition in T3 is load-bearing, not cosmetic (measured
2026-09-30): keying the collapse on (retailer,title) ALONE merged two distinct
checksum-valid products whenever one retailer listed the same title string
under two gtins. 692 groups, 1,086 product-listings deleted, and 264
products lost their ONLY row in the corpus — present in canonical_records.csv,
absent from the deduped output, with a representative carrying a sibling's
gtin. T2 already detects exactly this hazard and deferred to T3; T3 then
merged them anyway. The partition closes it: rows carrying DIFFERENT trusted
gtins never collapse together. Rows without a gtin may share a populated
title for price aggregation; rows with missing titles stay separate in T2/T3.

Identity is decided by `core.sku_identity` (the SSOT) and by nothing else.
Representative choice deliberately ignores `price`, `url` and `image_url`:
price moves with the seller, and the URL columns are export noise — the
completeness score counts DESCRIPTOR fields only.

Writes:
  data/dataset_deduped.csv             deduped dataset (pipeline input)
  data/sku_to_rep.csv                  raw SKU (sku_id) -> rep_id
  results/06_dedupe_summary.csv        per-tier counts
  results/06_ambiguous_offer_groups.csv  retailer+title >1 price
  results/06_dedupe_removals.csv       one review row per removed raw SKU
  results/training/dedupe_conflicts.csv  descriptor conflicts left unresolved
  results/manifests/dedupe.json        per-stage manifest, written LAST
                                        (SILENT_DROPS task 4 — the stage's
                                        completion marker: closure
                                        input_rows == output_rows + dropped,
                                        every output sha256-pinned, all
                                        writes atomic)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.common import DATA_PATH, SEED, F, ensure_parent, load_dataset, data_cfg
from core.manifest import atomic_write_csv, begin_manifest, finish_manifest
from core.deduplication import collapse_representatives
from core.sku_identity import (
    completeness_frame,
    evaluate_sku_identity,
    identity_conflict,
    row_identity,
)
from core.progress import tracked

# output paths from the config SSOT (files.*) — were hardcoded here, the
# only filenames in the tree outside config/paths.yaml
CSV_SUMMARY = F["dedupe_summary"]
CSV_OFFERS = F["ambiguous_offer_groups"]
CSV_REMOVALS = F["removals"]
DEDUPED_PATH = F["dataset_deduped"]
SKU_TO_REP_PATH = F["sku_to_rep"]
CSV_CONFLICTS = F["dedupe_conflicts"]

HELPERS = ["_price", "_complete", "_has_bc", "_t2_bc", "_bc_valid", "_ident"]

# STEP-2 adjudicated verdicts (owner review 2026-09-29). The 10 residual
# groups that the descriptor bundle cannot decide unambiguously were reviewed
# on their ORIGINAL (unnormalized) data — full title, attributes, brand,
# category, price. 8 are genuine product splits (keep separate), 2 are
# same-product where a listing qualified a shared trait (collapse). Keyed by
# (retailer, gtin) in config/paths.yaml dedupe_adjudications so the T1.5
# loop can look up the reviewed verdict directly.
#
# This table is the reason the text predicate is allowed to decide T1.5 at
# all: measured on gtin-labeled ground truth (2026-09-30) the descriptor
# predicate alone merges 59.0% of provably-different pairs, because a
# missing descriptor reads as agreement. Absence of a conflict is therefore
# NOT sufficient — a byte-identical malformed gtin at one retailer plus a
# descriptor verdict is, and anything the verdict cannot settle is escalated
# here rather than merged.


def _same_product_by_title(sub: pd.DataFrame, retailer: str, gtin: str) -> bool:
    """Two-step product-identity decision for a (retailer, gtin) group.

    STEP 0 — config-owned adjudicated overrides win outright.
    STEP 1 — the descriptor bundle decides: no dimension may PROVE the rows are
      different products. `core.sku_identity` owns that comparison, so the
      dedupe, the gate and the vetoes cannot drift apart. Category is
      deliberately not a veto: it is as noisy as the attribute cell on these
      rows (Reconstituted vs Not-from-Conjugate juice are the same product).
    STEP 2 — a group the bundle cannot settle is NOT merged. The absence of a
      conflict is not evidence of identity (measured 59.0% false-merge rate on
      gtin-labeled hard negatives), so the safe answer is to keep the rows
      apart and let the link lane adjudicate.

    `sub` must contain the descriptor columns. Returns True when the group is
    the same product and should collapse.
    """
    key = (retailer, gtin)
    for reviewed in data_cfg().dedupe_adjudications:
        if key == (reviewed.retailer, reviewed.gtin):
            return reviewed.decision == "collapse"

    identities = [row_identity(row) for row in sub.to_dict("records")]
    # Compatibility is not transitive: A={lemon,lime} can overlap B={lemon}
    # and C={lime} while B conflicts with C. Check every pair before collapse.
    from itertools import combinations
    return all(evaluate_sku_identity(left, right)["decision"] in {
        "same", "compatible_unverified"
    } for left, right in combinations(identities, 2))


def _protect_missing_titles(frame: pd.DataFrame) -> pd.DataFrame:
    """Give untitled listings separate partitions before title-based collapse.

    T1 has already consumed trustworthy retailer/gtin identity. An absent
    title cannot prove identity for the remaining T2/T3 rows, even when both
    also lack a gtin. Use row positions rather than a possibly missing or
    repeated listing ID, preserving every such source row independently.
    """
    missing = frame["sku_name_eng"].fillna("").astype(str).str.strip().eq("")
    if not missing.any():
        return frame
    frame = frame.copy()
    positions = np.flatnonzero(missing.to_numpy())
    frame.loc[missing, "_ident"] = [f"untitled-row:{i}" for i in positions]
    return frame


def _protect_untrusted_title_conflicts(frame: pd.DataFrame) -> pd.DataFrame:
    """Do not let T2/T3 undo descriptor splits left unresolved by T1.5.

    Empty identity partitions mean no trusted key. Before price aggregation,
    require every pair to pass the same descriptor review as T1.5. Preserve
    the whole group on a conflict: compatibility is not transitive.
    """
    candidates = frame[frame['_ident'].eq('')]
    protected = []
    for (_, _), group in candidates.groupby(['retailer', 'sku_name_eng'], sort=False):
        if len(group) < 2:
            continue
        from itertools import combinations
        identities = [row_identity(row) for row in group.to_dict('records')]
        if any(evaluate_sku_identity(a, b)['decision'] not in {'same', 'compatible_unverified'}
               for a, b in combinations(identities, 2)):
            protected.extend(group.index)
    if not protected:
        return frame
    result = frame.copy()
    protected = set(protected)
    for position, index in enumerate(result.index):
        if index in protected:
            result.at[index, '_ident'] = f'descriptor-review-row:{position}'
    return result



def main() -> None:
    for _out in (CSV_SUMMARY, CSV_OFFERS, CSV_REMOVALS, CSV_CONFLICTS,
                 DEDUPED_PATH, SKU_TO_REP_PATH):
        ensure_parent(_out)
    # Stage manifest (SILENT_DROPS task 4) — begin BEFORE the work: the
    # raw export is hashed now (53MB, chunked) so the record pins exactly
    # what this stage read. Seed = the SSOT seed (lib.common.SEED); the
    # tiered collapse below is deterministic, no RNG is consumed.
    manifest = begin_manifest("dedupe", inputs=[DATA_PATH], seed=SEED)
    from core.identity_policy import apply_identity_links, reviewed_row_mask
    df = apply_identity_links(load_dataset())
    n0 = len(df)
    work = df.assign(
        _price=pd.to_numeric(df["sku_last_price"], errors="coerce"),
        # Descriptor completeness, NOT `df.notna().sum()`: the old score
        # counted `url` and `image_url`, so a listing survived on the strength
        # of two export-noise columns while a complete title with no image lost
        # (measured 2026-09-30). price is excluded too — it is a seller
        # attribute, not a description of the product.
        _complete=completeness_frame(df),
        _has_bc=(df["gtin"].fillna("").str.len() > 0).astype(int),
    )

    # The ambiguous-offer audit is computed from the RAW frame, before any tier
    # can consume the rows. It used to be derived from whatever survived to T3,
    # so a group T2 collapsed first silently vanished from the audit trail.
    pv = work.groupby(["retailer", "sku_name_eng"])["_price"].agg(
        ["size", "nunique", "min", "max"]).rename(columns={"size": "count"})
    ambiguous = pv[(pv["count"] > 1) & (pv["nunique"] > 1)]

    # parent[i] = original row index of the surviving representative for row i.
    # Updated per tier and resolved transitively at the end (a T1 survivor may
    # itself be collapsed by T2/T3).
    parent = {i: i for i in work.index}

    summary = []

    # T1: retailer+gtin -> one row (ground-truth identity), ONLY for rows
    # that actually have a gtin. Rows with a MISSING gtin are NOT
    # collapsed ("no gtin" is not "same gtin"). And only for rows
    # whose gtin PASSES the GS1 checksum (owner ruling, src/core/gtin.py):
    # an invalid gtin is export noise, not identity — 103 retailer+
    # gtin groups carried >1 distinct title on a checksum-fail gtin
    # and would silently merge different products. Invalid-gtin rows
    # are not dropped: they fall through to T2/T3 title-based tiers.
    from core.gtin import gtin_validity

    bc_stripped = work["gtin"].fillna("").astype(str).str.strip()
    before_valid_gtins = set(bc_stripped[gtin_validity(bc_stripped)]) - {""}
    held = reviewed_row_mask(work)
    work = work.assign(
        _bc_valid=(gtin_validity(bc_stripped) & ~held).to_numpy()
    )
    # The identity partition: a row's TRUSTED gtin, or "" when it has none.
    # T1 collapses within a partition by construction; T3 reuses it so two
    # different products sharing a title string can never collapse together.
    work = work.assign(
        _ident=np.where(work["_bc_valid"],
                        work["gtin"].fillna("").astype(str).str.strip(), ""),
    )
    work.loc[held, "_ident"] = "review:" + work.loc[held, "sku_id"].astype(str)
    with_bc = work[(work["_has_bc"] == 1) & (work["_bc_valid"])]
    t1_bc_invalid = work[(work["_has_bc"] == 1) & (~work["_bc_valid"])]
    no_bc = work[work["_has_bc"] == 0]
    n_t1_skipped = len(t1_bc_invalid)
    t1, dropped1 = collapse_representatives(with_bc, ["retailer", "gtin"],
                            ["_complete"], [False], parent=parent)
    work = pd.concat([t1, t1_bc_invalid, no_bc])
    summary.append({"tier": "T1 retailer+gtin",
                    "dropped_rows": len(dropped1),
                    "skipped_checksum_invalid": n_t1_skipped})

    # T1.5: same retailer + same MALFORMED (checksum-invalid) gtin + same
    # product -> one row. T1 refuses to collapse on an invalid gtin because
    # "invalid gtin is export noise, not identity" — but a malformed gtin
    # that is byte-identical at one retailer is still a strong candidate, and
    # the descriptor bundle is the arbiter (`_same_product_by_title`, which
    # delegates to core.sku_identity and escalates anything it cannot
    # settle to the owner-adjudicated table). This recovers the 97 groups
    # measured in the dedupe invalid-gtin audit (2026-09-29) while never
    # merging genuinely different products (Cool Brew French Roast vs Vanilla,
    # Montellier Lemon vs Lime, Ginseng Up vs Natural Ginger Ale, ...).
    t15_dropped = []
    t15_kept = []
    t15_groups = 0
    t15_unresolved = 0
    conflict_rows = []
    for (retailer, gtin), sub in tracked(
        list(t1_bc_invalid.groupby(["retailer", "gtin"], sort=False)),
        desc="T1.5 malformed-gtin groups",
    ):
        if len(sub) <= 1:
            t15_kept.append(sub)
            continue
        if _same_product_by_title(sub, retailer, gtin):
            # Representative: most complete descriptors, then a trusted
            # gtin, then the most informative title. NOT price — the
            # cheapest listing is not the most truthful one.
            order = sub.sort_values(
                ["_complete", "_ident", "sku_name_eng"],
                ascending=[False, False, True],
                na_position="last", kind="stable",
            )
            rep = order.iloc[[0]]
            rep_idx = order.index[0]
            for idx in sub.index:
                parent[idx] = rep_idx
            t15_dropped.extend(sub.index.difference([rep_idx]))
            t15_kept.append(rep)
            t15_groups += 1
        else:
            # Unresolved OR a proven split: keep every row and record WHY, so
            # the identity question is a review queue rather than a silent
            # either/or. `label` distinguishes "descriptor proves different"
            # from "descriptor had nothing to say".
            identities = [row_identity(r) for r in sub.to_dict("records")]
            anchor = identities[0]
            for other in identities[1:]:
                evaluation = evaluate_sku_identity(anchor, other)
                reasons = evaluation["identity_conflicts"]
                reviews = evaluation["review_dimensions"] + evaluation["unclassified_keys"]
                conflict_rows.append({
                    "tier": "T1.5", "retailer": retailer, "gtin": gtin,
                    "label": "proven_split" if reasons else "unresolved",
                    "reasons": "|".join(reasons or [f"review:{r}" for r in reviews]) or "-",
                    "title_a": sub["sku_name_eng"].iat[0], "title_b": sub["sku_name_eng"].iat[1],
                    "brand_a": anchor.brand and " ".join(sorted(anchor.brand)) or "",
                    "brand_b": other.brand and " ".join(sorted(other.brand)) or "",
                })
                if not reasons:
                    t15_unresolved += 1
            t15_kept.append(sub)
    t15_kept = pd.concat(t15_kept)
    work = pd.concat([t1, t15_kept, no_bc])
    summary.append({"tier": "T1.5 retailer+malformed-gtin+same-product",
                    "dropped_rows": len(t15_dropped),
                    "collapsed_groups": t15_groups,
                    "unresolved_groups": t15_unresolved})

    # T2: retailer+title+gtin -> one row (lossless). `price` is NOT part of
    # the key: it is a seller attribute, and two rows at one retailer with the
    # same title and the same product gtin but different prices are the same
    # product offered twice — that is T3's price-aggregation, and the
    # ambiguous-offer audit (computed from the raw frame above) still records
    # it. Rows with a MISSING price are not "identical everything" in the old
    # sense, but price is no longer part of identity, so they no longer need
    # their own lane.
    #
    # "Lossless" is still ENFORCED by the gtin-agreement guard: rows
    # collapse only when their trusted gtins agree (same non-empty
    # identity, or both without one). Two DIFFERENT trusted gtins under one
    # title are different products and must not be merged here.
    work = _protect_missing_titles(work)
    work = _protect_untrusted_title_conflicts(work)
    with_price = work.copy()
    with_price["_t2_bc"] = with_price["_ident"]
    no_price = work.iloc[0:0]

    # Split T2 by gtin agreement WITHIN each (retailer,title) group: an
    # all-same-identity group collapses losslessly; a group with >1 distinct
    # trusted gtin carries genuinely different products — those rows all
    # flow to T3 (kept here via the conflict mask).
    bc_agrees = with_price.groupby(
        ["retailer", "sku_name_eng"], sort=False, dropna=False
    )["_t2_bc"].transform("nunique").le(1)
    t2_clean = with_price[bc_agrees]
    t2_conflict = with_price[~bc_agrees]

    t2, dropped2 = collapse_representatives(t2_clean, ["retailer", "sku_name_eng", "_t2_bc"],
                            ["_complete"], [False], parent=parent)
    work = pd.concat([t2, t2_conflict.drop(columns=["_t2_bc"]), no_price])
    summary.append({"tier": "T2 retailer+title+gtin",
                    "dropped_rows": len(dropped2),
                    "deferred_to_t3": len(t2_conflict)})
    print(f"[dedupe] T2 complete: {len(dropped2):,} dropped, {len(t2_conflict):,} deferred to T3", flush=True)

    # T3: retailer+title WITHIN AN IDENTITY PARTITION -> one deliberate
    # representative + flag. The `_ident` partition is the fix for the measured
    # identity loss: keying on (retailer,title) alone deleted 1,086
    # product-listings carrying a valid gtin and erased 264 products from
    # the corpus entirely (they survived in canonical_records.csv, absent from
    # the output, represented by a sibling's gtin). Rows sharing a title
    # with NO trusted gtin still collapse together — that is the genuine
    # price-aggregation case, and it is flagged rather than silent.
    t3, dropped3 = collapse_representatives(work, ["retailer", "sku_name_eng", "_ident"],
                            ["_has_bc", "_complete", "sku_name_eng"],
                            [False, False, True], parent=parent)
    work = t3
    summary.append({"tier": "T3 retailer+title+identity-partition (price-aggregation)",
                    "dropped_rows": len(dropped3)})
    print(f"[dedupe] T3 complete: {len(dropped3):,} dropped", flush=True)


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
        "sku_id": df["sku_id"].to_numpy(),
        "rep_id": [rep_pos[parent[i]] for i in df.index],
    })
    atomic_write_csv(sku_to_rep, SKU_TO_REP_PATH, index=False)
    print(f"wrote {SKU_TO_REP_PATH} ({len(sku_to_rep):,} rows)")

    # sanity: no (retailer,title,identity-partition) duplicates may remain, and
    # every raw SKU resolves to a valid representative. The partition is part
    # of the key on PURPOSE: two genuinely different products may share a title
    # string at one retailer, but no two rows sharing a trusted gtin can
    # survive as duplicates. Checked on `work` because `deduped` drops the
    # helper columns.
    dups = int(
        work.duplicated(subset=["retailer", "sku_name_eng", "_ident"]).sum()
    )
    if dups:
        raise AssertionError(
            f"sanity FAILED: {dups} retailer+title+identity dupes remain"
        )
    if sku_to_rep["rep_id"].isna().any():
        raise AssertionError("sanity FAILED: some SKUs map to no representative")
    if set(sku_to_rep["rep_id"].unique()) != set(range(len(deduped))):
        raise AssertionError("sanity FAILED: rep_id coverage is not 0..n-1")
    print("  [PASS] no retailer+title+identity duplicates remain; "
          "SKU->rep mapping complete")

    # IDENTITY INVARIANT (2026-09-30): no product may lose its last row. Every
    # trusted gtin present in the input must still be present in the output,
    # or a product has been erased from the matching input. Measured 264
    # products failing this before the T3 identity partition landed. This is
    # now a hard gate, not a report: the failure mode it catches is silent and
    # unrecoverable downstream, and no future tier may be allowed to reintroduce
    # it.
    after = set(deduped["gtin"].fillna("").astype(str).str.strip())
    lost = before_valid_gtins - after
    if lost:
        raise AssertionError(
            f"sanity FAILED: {len(lost):,} trusted gtins lost their last "
            f"row (e.g. {sorted(lost)[:3]}) — a product was deleted from the "
            f"matching input, not de-duplicated"
        )
    print(f"  [PASS] identity invariant: all {len(before_valid_gtins):,} "
          f"trusted gtins still have a representative")

    # One review row for every raw SKU deliberately collapsed by a tier.
    # Capture the tier at its direct parent update, before transitive
    # representative resolution obscures where the removal happened.
    removal_tier = {
        **{idx: "T1 retailer+gtin" for idx in dropped1},
        **{idx: "T1.5 retailer+malformed-gtin+same-product" for idx in t15_dropped},
        **{idx: "T2 retailer+title+gtin" for idx in dropped2},
        **{idx: "T3 retailer+title+identity-partition (price-aggregation)" for idx in dropped3},
    }
    removals = sku_to_rep.loc[list(removal_tier)].copy()
    removals["tier"] = [removal_tier[idx] for idx in removals.index]
    removals = removals[["sku_id", "rep_id", "tier"]].reset_index(drop=True)
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
    print(f"wrote {CSV_OFFERS} (display table, {len(ambiguous_out):,} rows) "
          f"— {len(ambiguous_out):,} ambiguous-offer groups flagged")

    conflicts = pd.DataFrame(
        conflict_rows,
        columns=["tier", "retailer", "gtin", "label", "reasons",
                 "title_a", "title_b", "brand_a", "brand_b"],
    )
    atomic_write_csv(conflicts, CSV_CONFLICTS, index=False)
    n_unresolved = int((conflicts["label"] == "unresolved").sum()) if len(conflicts) else 0
    print(f"wrote {CSV_CONFLICTS} ({len(conflicts):,} rows) — "
          f"{n_unresolved:,} identity questions the descriptor bundle could "
          f"not settle, escalated instead of guessed")


    # ---- row accounting (SILENT_DROPS task 4; capture-only) ────────────────
    # The three tier counters partition the frame at each step, so the
    # closure input_rows == output_rows + sum(dropped) holds by
    # construction (finish_manifest asserts it before publishing):
    # 71,623 == 61,529 + (1,943 + 2,245 + 5,906).
    #
    # The two "deferred" populations are NOT drops and deliberately
    # excluded from `dropped`:
    #   skipped_checksum_invalid (T1) — 3,715 checksum-fail gtin rows
    #     are concatenated BACK into the work frame; T1.5 collapses the
    #     same-product ones (counted under dropped.t1_5_*), the rest fall
    #     through to the title tiers.
    #   deferred_to_t3 (T2) — 465 gtin-conflicting rows likewise
    #     re-enter the frame and are settled by T3's counter.
    # Recording them under their own keys (outside `dropped`) keeps the
    # audit trail complete without breaking the closure invariant.
    row_accounting = {
        "input_rows": n0,
        "output_rows": len(deduped),
        "dropped": {
            "t1_retailer_gtin": len(dropped1),
            "t1_5_retailer_malformed_gtin_same_product": len(t15_dropped),
            "t2_retailer_title_gtin": len(dropped2),
            "t3_retailer_title_identity_partition": len(dropped3),
        },
        "skipped_checksum_invalid": n_t1_skipped,
        "deferred_to_t3": len(t2_conflict),
        "ambiguous_offer_groups": len(ambiguous_out),
        "unresolved_identity_review_rows": len(conflicts),
    }
    manifest_path = finish_manifest(
        manifest,
        outputs=[DEDUPED_PATH, SKU_TO_REP_PATH, CSV_SUMMARY, CSV_OFFERS,
                 CSV_REMOVALS, CSV_CONFLICTS],
        row_accounting=row_accounting,
        expected_outputs=[
            F["dataset_deduped"].name, F["sku_to_rep"].name,
            F["dedupe_summary"].name, F["ambiguous_offer_groups"].name,
            F["removals"].name, F["dedupe_conflicts"].name,
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
