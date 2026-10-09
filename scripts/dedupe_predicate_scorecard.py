"""dedupe_predicate_scorecard.py — measured verdicts for EXTRA identity surfaces.

Owner question: "attributes are noisy, but regex cleaning handles noise — was
this measured on the cleaned surface?"
Recorded answer (code comments): YES — sku_identity = the cleaned typed surface
shared with the gate (dedupe.py:93-98, sku_identity.py:6-11); the measured
residual is ABSENCE (missing descriptor reads as agreement; 59.0% false-merge
on the gtin-labeled slice). This scorecard quantifies which ADDITIONAL
surfaces recover discriminant evidence for those absent cases.

Read-only probe: repo untouched; only this script + scratch notes are new.
No prepare_all / training.data_prep is run; no src/ edit; no commit/push.

Stacks
------
S0  typed-only  — the CURRENT `row_identity` surface (cleaned, regex-parsed
    claims of sku_name_eng/attribute/description_short_eng/urls/breadcrumbs),
    compared via `identity_conflict` with the rule-4 gtin lane DISABLED so the
    hypothesis is measured on the text surface alone (on label-0 pairs both
    gtins are valid and DIFFERENT, so rule 4 would fire "gtin" and the text
    surface would never be consulted — the 59% owner figure is about the TEXT
    surface).
S1  S0 + composed-row-surface — the row surface EXTENDED so claims parsed
    from description_short_eng + sku_name_eng + breadcrumbs_eng participate as
    conflict evidence using the SAME lexicons (extract_critical_claims /
    row_identity fields) — measured in conflicts recovered and false-merge cut.
S2  S1 + exact image_url match (same retailer) — identical URL = strong
    same-product EVIDENCE (positive lane, not a veto lane). Counted on both
    labels; pure positive evidence can only reduce false VETOES not absences,
    so it cannot reduce the absence-driven false-merges.
S3  GTIN-sibling closure — for label-1 families (same gtin), claims from
    canonical_records.csv (the per-gtin SSOT aggregation with per-key sources)
    resolve missing fields of a sibling row. Conflict resolution measured as
    S1-style blocked-pair delta. Label-0 pairs share no family by construction,
    so sibling closure is measured as label-1 recovery only.

Verdict logic (parameter surface X: false-merge ceiling to qualify as a
measured VETO): X ∈ {0.05, 0.10, 0.20}, default aggregate = 0.10 (reviewer's
bar). Per stack: VETO-CLASS if false_merge <= X; SUGGESTION-CLASS if strictly
better than S0 but still > X; REJECTED if worse than S0 or equal-and-unhelpful.
"""

from __future__ import annotations

import ast
from core.portable_archive import ByteCount
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from core.project_root import find_project_root

WORK = find_project_root(Path(__file__))
sys.path.insert(0, str(WORK / "src"))

from core.common import SEED                                          # noqa: E402
from core.gtin import gtin_validity                                   # noqa: E402
from core.sku_identity import identity_conflict, row_identity         # noqa: E402
from core.critical_attributes import extract_critical_claims          # noqa: E402

DATASET_CSV = WORK / "dataset.csv"
CANONICAL_CSV = WORK / "data" / "canonical_records.csv"
SCRATCH = WORK / "scratch"
SCRATCH.mkdir(exist_ok=True)

# budget caps for the pairwise-parse loop — deterministic, seed-bound.
CAP_PAIRS_PER_GROUP = 8
CAP_PAIRS_TOTAL = 4_000
SIBLING_CAP = 1_000
BARS = (0.05, 0.10, 0.20)


def size_of(path: Path) -> int:
    h = ByteCount()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.total


def load_dataset() -> pd.DataFrame:
    """dataset.csv: SSOT-loader-equal row set, read directly.

    Pins are removed (2026-10-06 owner ruling): no drift gate and no
    expected-sha comparison. The byte identity of the export this probe
    ran on is recorded in the probe's own outputs."""
    columns = ["sku_id", "retailer", "country", "sku_name_eng",
               "description_short_eng", "breadcrumbs_eng", "sku_url",
               "image_url", "sku_last_price", "gtin", "brand", "category",
               "attribute"]
    df = pd.read_csv(DATASET_CSV, usecols=columns, dtype="string",
                     keep_default_na=False, na_values=[""])
    if len(df) < 10:
        raise SystemExit("dataset.csv row collapse")
    return df


def _drop_gtin_trust(identity):
    """A ProductIdentity view with rule-4 gtin lane DISABLED so the predicate
    has to decide on the typed text surface (S0-S3's measured target)."""
    return type(identity)(**{**identity.__dict__, "gtin_trusted": False,
                             "gtin_key": ""})


@dataclass
class StackMetrics:
    false_merge: float
    true_merge: float
    recovered_pairs: int = 0
    dims: Counter = field(default_factory=Counter)


def _text_conflict(a: dict, b: dict) -> frozenset[str]:
    """identity_conflict on the de-trusted identity view (text surface)."""
    return frozenset(identity_conflict(_drop_gtin_trust(a), _drop_gtin_trust(b)))


def _extended_conflict(a_id, b_id, a_extra: frozenset, b_extra: frozenset,
                       base: frozenset) -> frozenset[str]:
    """S1: union current-surface conflicts with composed-surface claim-dims.

    The claim-dims come from extract_critical_claims over the composed
    surface text (title + attribute + description + breadcrumbs), because the
    current surface already parses those SAME TEXTS via sku_info but only
    routes them through the row_identity dimensions that have a dedicated
    extractor. To measure what ADDITIONAL evidence brings, extra claims are
    compared as claim-set DISJOINTNESS per dimension (both sides populated).
    """
    extra_conflicts = set()
    for dim, claim_a, claim_b in (
        ("flavor", a_extra.get("flavor"), b_extra.get("flavor")),
        ("carbonation", a_extra.get("carbonation"), b_extra.get("carbonation")),
        ("sweetener", a_extra.get("sweetener"), b_extra.get("sweetener")),
        ("pulp", a_extra.get("pulp"), b_extra.get("pulp")),
    ):
        if claim_a and claim_b and not (claim_a & claim_b):
            extra_conflicts.add(dim)
    # conflicts from the typed surface that remain: flavor/carbonation dims of
    # extract_all claims are also present since extract_all uses the same
    # extractor, so add them in case base didn't fire (extract_critical_claims
    # reads title+attribute+description+breadcrumbs in ONE normalized pass,
    # while the current surface splits them across sku_info/critical/row_dims).
    for dim in ("flavor", "carbonation", "sweetener", "pulp"):
        la = set(a_extra.get(dim) or frozenset()) | set(getattr(a_id, dim) or frozenset())
        lb = set(b_extra.get(dim) or frozenset()) | set(getattr(b_id, dim) or frozenset())
        if la and lb and not (la & lb):
            extra_conflicts.add(dim)
    return frozenset(set(base) | extra_conflicts)


def main() -> None:
    t_start = time.time()
    df = load_dataset()
    print(f"[data] rows={len(df):,} size={size_of(DATASET_CSV)}...")

    valid = gtin_validity(df["gtin"].astype(str)).to_numpy()
    # gtin_validity only quarantines the 34 held GTIN KEYS; the dedupe protocol
    # also excludes the per-sku reviewed listings this mask cannot see.
    from core.identity_policy import reviewed_row_mask
    valid = valid & ~reviewed_row_mask(df).to_numpy()
    df["valid_gtin"] = valid
    df["g"] = df["gtin"].astype(str).str.strip()
    df["g_key"] = df["g"].str.zfill(14)
    print(f"[data] gtin-valid TRUSTED rows (held excluded): {int(valid.sum()):,}")
    # SEED is consumed by deterministic down-sampling (np.linspace index
    # caps) and the optional embedding-band ordering; no RNG draw escapes
    # reproducibility (core.common.SEED = 42 is the SSOT).
    print(f"[seed] core.common.SEED={SEED}")

    # ── label-0 pool: same (retailer, title) with >1 DISTINCT valid gtin ───
    # Across rows in one group: distinct rows carry distinct gtins (a
    # checksum-valid gtin is row-inconsistent within a title otherwise).
    # gtin distinctness uses the 14-digit-zerofilled canonical key so a raw
    # cell variant never fakes 'different'.
    t0 = time.time()
    eligible = df[df["valid_gtin"]].copy()
    multi_gtin = eligible.groupby(["retailer", "sku_name_eng"])["g_key"].transform("nunique") > 1
    eligible0 = eligible[multi_gtin].copy()
    label0: list[tuple[dict, dict]] = []
    for (retailer, title), grp in eligible0.groupby(["retailer", "sku_name_eng"], sort=False):
        rows = list(grp.sort_values("sku_id").to_dict("records"))
        # anchor = each row paired with the NEXT row carrying a DIFFERENT gtin.
        pairs = []
        for i in range(len(rows) - 1):
            if rows[i]["g_key"] != rows[i + 1]["g_key"]:
                pairs.append((rows[i], rows[i + 1]))
        if len(pairs) > CAP_PAIRS_PER_GROUP:
            keep = np.linspace(0, len(pairs) - 1, CAP_PAIRS_PER_GROUP).astype(int)
            pairs = [pairs[i] for i in keep]
        label0.extend(pairs)
    if len(label0) > CAP_PAIRS_TOTAL:
        keep = np.linspace(0, len(label0) - 1, CAP_PAIRS_TOTAL).astype(int)
        label0 = [label0[i] for i in keep]
    label0 = [(a, b) for a, b in sorted(label0, key=lambda p: (p[0]["sku_id"], p[1]["sku_id"]))]
    print(f"label-0 built: {len(label0):,} pairs "
          f"(from {int(eligible[multi_gtin].groupby(['retailer','sku_name_eng']).ngroups):,} "
          f"multi-gtin title groups) | {time.time()-t0:.1f}s")

    # ── label-1 pool: duplicate rows of the SAME checksum-valid gtin ───────
    # Family links MAY cross retailers (the gtin is the answer): pair every
    # row with the next sibling of the same g_key in sorted (retailer,sku_id)
    # order, so cross-retailer same-gtin pairs enter the label-1 supply.
    t1 = time.time()
    sizes = eligible.groupby("g_key")["sku_id"].transform("size")
    eligible1 = eligible[sizes > 1].copy()
    label1: list[tuple[dict, dict]] = []
    for g_key, grp in eligible1.groupby("g_key", sort=False):
        rows = list(grp.sort_values(["retailer", "sku_id"]).to_dict("records"))
        # chain-pairs: first row with each successive sibling.
        pairs = [(rows[0], r) for r in rows[1:]]
        if len(pairs) > CAP_PAIRS_PER_GROUP:
            keep = np.linspace(0, len(pairs) - 1, CAP_PAIRS_PER_GROUP).astype(int)
            pairs = [pairs[i] for i in keep]
        label1.extend(pairs)
    if len(label1) > CAP_PAIRS_TOTAL:
        keep = np.linspace(0, len(label1) - 1, CAP_PAIRS_TOTAL).astype(int)
        label1 = [label1[i] for i in keep]
    label1 = [(a, b) for a, b in sorted(label1, key=lambda p: (p[0]["sku_id"], p[1]["sku_id"]))]
    n_l1_groups = int(eligible1.groupby("g_key").ngroups)
    print(f"label-1 built: {len(label1):,} pairs "
          f"(from {n_l1_groups:,} same-gtin dup families) "
          f"| {time.time()-t1:.1f}s")

    # ── parse once per distinct row ────────────────────────────────────────
    t2 = time.time()
    identities: dict[str, object] = {}
    for pairs in (label0, label1):
        for a, b in pairs:
            for row in (a, b):
                if row["sku_id"] not in identities:
                    identities[row["sku_id"]] = row_identity(row)
    print(f"parsed {len(identities):,} distinct rows through row_identity "
          f"in {time.time()-t2:.1f}s")

    # ── extended-claims per distinct row (composed surface, S1) ────────────
    EXT_COLS = ("sku_name_eng", "attribute", "description_short_eng", "breadcrumbs_eng")
    extended: dict[str, dict] = {}
    distinct_rows: dict[str, dict] = {}
    for pairs in (label0, label1):
        for a, b in pairs:
            for r in (a, b):
                distinct_rows[r["sku_id"]] = r
    for row_id, row in distinct_rows.items():
        extended[row_id] = extract_critical_claims(
            *[str(row.get(c, "") or "") for c in EXT_COLS]
        )

    # ══ S0: typed-only (current surface, gtin lane off) ═══════════════════
    t3 = time.time()
    s0_conf_l0 = [_text_conflict(identities[a["sku_id"]], identities[b["sku_id"]])
                  for a, b in label0]
    s0_conf_l1 = [_text_conflict(identities[a["sku_id"]], identities[b["sku_id"]])
                  for a, b in label1]
    s0 = StackMetrics(
        false_merge=sum(1 for c in s0_conf_l0 if not c) / max(1, len(s0_conf_l0)),
        true_merge=sum(1 for c in s0_conf_l1 if not c) / max(1, len(s0_conf_l1)),
    )
    s0_dims = Counter(d for c in s0_conf_l0 for d in c)
    print(f"S0 false_merge={s0.false_merge:.4f} true_merge={s0.true_merge:.4f} "
          f"conflict-dims-on-label-0={dict(s0_dims)} | {time.time()-t3:.1f}s")

    # ══ S1: extended composed surface ════════════════════════════════════
    t4 = time.time()
    s1_conf_l0 = []
    recovered0 = Counter()
    for (a, b), base in zip(label0, s0_conf_l0):
        full = _extended_conflict(identities[a["sku_id"]], identities[b["sku_id"]],
                                  extended[a["sku_id"]], extended[b["sku_id"]], base)
        s1_conf_l0.append(full)
        if full and not base:
            recovered0.update(full - set(base))
    s1_conf_l1 = [
        _extended_conflict(identities[a["sku_id"]], identities[b["sku_id"]],
                           extended[a["sku_id"]], extended[b["sku_id"]],
                           base)
        for (a, b), base in zip(label1, s0_conf_l1)
    ]
    s1 = StackMetrics(
        false_merge=sum(1 for c in s1_conf_l0 if not c) / max(1, len(s1_conf_l0)),
        true_merge=sum(1 for c in s1_conf_l1 if not c) / max(1, len(s1_conf_l1)),
        recovered_pairs=sum(1 for base, full in zip(s0_conf_l0, s1_conf_l0) if full and not base),
        dims=recovered0,
    )
    print(f"S1 false_merge={s1.false_merge:.4f} true_merge={s1.true_merge:.4f} "
          f"recovered={s1.recovered_pairs} dims={dict(s1.dims)} | {time.time()-t4:.1f}s")

    # ══ S2: S1 + exact image_url match (same retailer; positive lane) ═════
    t5 = time.time()
    img_same_l0 = img_same_l1 = img_same_cross_l1 = 0
    for a, b in label0:
        if a["image_url"] and a["image_url"] == b["image_url"]:
            img_same_l0 += 1
    for a, b in label1:
        if a["image_url"] and a["image_url"] == b["image_url"]:
            img_same_l1 += 1
            if a["retailer"] != b["retailer"]:
                img_same_cross_l1 += 1
    # An IDENTICAL image URL is byte-level same-listing evidence. On the
    # WITHIN-retailer negative slice it is instead a hard CONTAMINATION
    # signal (retailers reuse one marketing image across a product's variant
    # pages: 35 label-0 pairs matched), so it stays a ranker/positive surface,
    # never a merge decider.
    # The surface is a POSITIVE lane by construction: it adds no veto, so the
    # false-merge carries over from S1 unchanged and is reported for parity.
    s2 = StackMetrics(false_merge=s1.false_merge, true_merge=s1.true_merge,
                      recovered_pairs=img_same_l0, dims=Counter())
    print(f"S2 image-identical: label0={img_same_l0}/{len(label0)} "
          f"label1={img_same_l1}/{len(label1)} (cross-retailer {img_same_cross_l1}) "
          f"| {time.time()-t5:.1f}s")

    # ══ S3: GTIN-sibling closure from canonical_records.csv ══════════════
    # A same-gtin family's canonical claim-set pools EVERY sibling listing's
    # evidence (per-key sources: extract_all per row, unioned per gtin). The
    # closure direction measured: a row-surface claim that is EMPTY but the
    # FAMILY canonical record HOLDS gets restored — absence becomes evidence.
    # Within a label-1 family every restore is label-consistent by definition,
    # so we count FILLS (absence-recovery), not new conflicts. The fill supply
    # is what a future absence-recovery rule could consult; it does NOT change
    # the S1 false-merge on the different-gtin slice (sibling closure is inert
    # there by construction: no two different gtins share a family).
    # NOTE the closure pool is canonical-family-wide: a restores from ANY
    # sibling of either row, per-family, not pairwise sibling-to-sibling.
    # Measured fills (label-1 first SIBLING_CAP pairs): the rate at which a
    # missing row-surface claim would have been repaired had this surface
    # been available to T1.5's descriptor bundle.
    t6 = time.time()
    s3 = StackMetrics(false_merge=s1.false_merge, true_merge=s1.true_merge,
                      recovered_pairs=0, dims=Counter())
    if not CANONICAL_CSV.exists():
        print(f"S3 SKIPPED — missing artifact: {CANONICAL_CSV} (see report notes)")
    else:
        canon = pd.read_csv(
            CANONICAL_CSV,
            usecols=["gtin", "flavor_set", "carbonation_set", "sweetener_set",
                     "pulp_set"],
            dtype="string", keep_default_na=False,
        )
        canon["gtin_key"] = canon["gtin"].astype(str).str.zfill(14)
        canon = canon.set_index("gtin_key")
        dims = ("flavor_set", "carbonation_set", "sweetener_set", "pulp_set")
        filled_pairs = 0
        dim_counter: Counter = Counter()
        for a, b in label1[:SIBLING_CAP]:
            ga, gb = a["g_key"], b["g_key"]
            if ga not in canon.index or gb not in canon.index:
                continue
            for dim, row_dim in zip(dims, ("flavor", "carbonation", "sweetener", "pulp")):
                # row-level claim sets from the CURRENT surface (S1's data)
                claim_a = frozenset(getattr(identities[a["sku_id"]], row_dim, frozenset()) or frozenset())
                claim_b = frozenset(getattr(identities[b["sku_id"]], row_dim, frozenset()) or frozenset())
                if claim_a and claim_b:
                    continue  # both populated: closure has nothing to fill
                try:
                    family = ast.literal_eval(canon.at[ga, dim] or "[]")
                except (ValueError, SyntaxError):
                    continue
                if not family:
                    continue
                filled_pairs += 1
                dim_counter[row_dim] += 1
        s3.recovered_pairs = filled_pairs
        s3.dims = dim_counter
        print(f"S3 sibling-fill label-1 pairs (first {min(SIBLING_CAP, len(label1))}): "
              f"{filled_pairs} dims={dict(dim_counter)} | {time.time()-t6:.1f}s")

    # ── verdict per stack, at all three bars ──────────────────────────────
    rows = []
    for stack, m, better_than_s0, note in (
        ("S0", s0, None, "typed-only current surface (gtin lane off)"),
        ("S1", s1, s1.false_merge < s0.false_merge, "S0 + composed-row claims"),
        ("S2", s2, s2.false_merge < s0.false_merge,
         "S1 + exact image_url (same retailer; positive/recall lane)"),
        ("S3", s3, s3.false_merge < s0.false_merge,
         "S1 + GTIN-sibling closure from canonical_records"),
    ):
        for x in BARS:
            if m.false_merge <= x:
                verdict = "VETO-CLASS"
            elif better_than_s0 is True and m.recovered_pairs > 0:
                verdict = "SUGGESTION-CLASS"
            else:
                verdict = "REJECTED"
            rows.append({"stack": stack, "X": x,
                         "false_merge": round(m.false_merge, 4),
                         "true_merge": round(m.true_merge, 4),
                         "recovered_pairs": m.recovered_pairs,
                         "recovered_dims": sorted(m.dims.items(), key=lambda kv: str(kv[0])),
                         "verdict": verdict, "note": note})
    # S0 has no "improvement" semantics at any X — force-baseline naming
    for r in rows:
        if r["stack"] == "S0":
            r["verdict"] = f"BASELINE (false_merge={s0.false_merge:.4f})"

    out = pd.DataFrame(rows)
    out_path = SCRATCH / "dedupe_predicate_scorecard.results.csv"
    out.to_csv(out_path, index=False)
    print(f"\nresults written: {out_path}\n")
    print(out.to_string(index=False))

    summary = {
        "dataset_rows": int(len(df)),
        "label0_pairs": len(label0),
        "label1_pairs": len(label1),
        "distinct_rows_parsed": len(identities),
        "runtime_seconds": round(time.time() - t_start, 1),
        "S0": {"false_merge": s0.false_merge, "true_merge": s0.true_merge,
               "dims": dict(s0_dims)},
        "S1": {"false_merge": s1.false_merge, "true_merge": s1.true_merge,
               "recovered": s1.recovered_pairs, "recovered_dims": dict(s1.dims)},
        "S2": {"false_merge": s2.false_merge, "true_merge": s2.true_merge,
               "image_identical_l0": img_same_l0, "image_identical_l1": img_same_l1,
               "image_identical_cross_retailer_l1": img_same_cross_l1},
        "S3": {"false_merge": s3.false_merge, "true_merge": s3.true_merge,
               "recovered": s3.recovered_pairs, "recovered_dims": dict(s3.dims),
               "canonical_available": CANONICAL_CSV.exists()},
        "bars": list(BARS),
    }
    summary_path = SCRATCH / "dedupe_predicate_scorecard.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"summary written: {summary_path}")
    print(f"total runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
