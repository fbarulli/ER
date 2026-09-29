"""Dedupe signal from MALFORMED/INVALID barcodes (T1.5 recovery sizing).

TODO.md dedupe audit (2026-09-29) left one signal on the table: the
invalid-checksum barcode groups. T1 skips them (a checksum-fail barcode is
treated as export noise, not identity), so two rows that share the SAME
malformed barcode at the SAME retailer are NOT collapsed — even when they
are the same product. This script sizes the safe T1.5 recovery precisely.

Lesson learned (this audit): the raw `attribute` field on these malformed-
barcode rows is UNRELIABLE — its structured conflicts (pack material, juice
content, sweetener, carbonization) are correlated export noise. Of 34
structured-conflict groups, 28 were the same product (false vetoes) and only
6 were genuine product splits. The ground truth is the record-linkage SSOT
normalized, pack-stripped TITLE with a descriptor (flavor/roast/brand) check:
a differing descriptor token (Cool Brew French Roast vs Vanilla, Montellier
Lemon vs Lime, Ozarka/Zephyrhills Lemon vs Lime, Ginseng Up vs Natural Ginger
Ale) proves a genuine split even when fuzzy similarity is high.

Method: start from the malformed/invalid barcode rows, group by
(retailer, raw malformed barcode), and for every multi-row group decide
same-product using `training.dedupe._same_product_by_title` (the EXACT logic
the T1.5 tier in dedupe.py runs). We report the raw structured verdict AND
the corrected same-product verdict side by side so the owner sees how many
structured conflicts were false vetoes.

Writes:
  results/training/dedupe_invalid_barcode_groups.csv   per-group table
  results/training/dedupe_invalid_barcode_recovery.csv recovery-vs-fuzzy-floor
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

from core.common import load_raw_export
from core.gtin import barcode_validity

# The T1.5 product-identity decision is the SSOT — import it from dedupe so
# this audit and the tier it sizes can never drift apart.
from training.dedupe import _same_product_by_title  # noqa: E402

RESULTS = Path(os.environ.get("EUROMONITOR_RESULTS_DIR", "results")) / "training"
OUT_GROUPS = RESULTS / "dedupe_invalid_barcode_groups.csv"
OUT_RECOVERY = RESULTS / "dedupe_invalid_barcode_recovery.csv"

# Text columns that contribute fuzzy signal (title plus all free prose).
FUZZY_TEXT_COLS = [
    "title", "description_short_eng", "breadcrumbs_eng", "brand", "category",
]


def _norm_tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(s).lower()))


def jaccard(a: str, b: str) -> float:
    sa, sb = _norm_tokens(a), _norm_tokens(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / len(union)


def levenshtein(a: str, b: str) -> float:
    """Normalized Levenshtein ratio (1.0 == identical)."""
    a, b = str(a).lower(), str(b).lower()
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b))


def fuzzy_score(a: str, b: str) -> float:
    return 0.5 * jaccard(a, b) + 0.5 * levenshtein(a, b)


def main() -> None:
    OUT_GROUPS.parent.mkdir(parents=True, exist_ok=True)

    df = load_raw_export().rename(
        columns={"sku_name_eng": "title", "gtin": "barcode"}
    )
    raw_bc = df["barcode"].fillna("").str.strip()
    has_bc = raw_bc.str.len() > 0
    valid = barcode_validity(raw_bc).to_numpy()

    # START HERE: the malformed/invalid barcode population.
    malformed = df[has_bc & ~valid].copy()
    malformed["barcode"] = raw_bc[has_bc & ~valid]

    rows = []
    for (retailer, barcode), sub in malformed.groupby(
        ["retailer", "barcode"], sort=False
    ):
        if len(sub) <= 1:
            continue

        same = _same_product_by_title(sub, retailer, barcode)

        # Fuzzy across ALL text cols + the attribute string, worst-pair (min)
        # and mean over the group.
        pairs = [(i, j) for i in range(len(sub)) for j in range(i + 1, len(sub))]
        min_fuzzy = 1.0
        mean_fuzzy = 0.0
        for i, j in pairs:
            s = 0.0
            n = 0
            for col in FUZZY_TEXT_COLS:
                s += fuzzy_score(sub[col].iloc[i], sub[col].iloc[j])
                n += 1
            s += fuzzy_score(sub["attribute"].iloc[i], sub["attribute"].iloc[j])
            n += 1
            fs = s / n
            min_fuzzy = min(min_fuzzy, fs)
            mean_fuzzy += fs
        mean_fuzzy /= max(len(pairs), 1)

        rows.append({
            "retailer": retailer,
            "barcode": barcode,
            "rows": len(sub),
            "distinct_titles": sub["title"].nunique(),
            "same_product": same,
            "min_fuzzy": round(min_fuzzy, 4),
            "mean_fuzzy": round(mean_fuzzy, 4),
            "titles": " | ".join(sub["title"].astype(str).unique()),
        })

    groups = pd.DataFrame(rows)
    groups = groups.sort_values(
        ["rows", "distinct_titles"], ascending=[False, False]
    ).reset_index(drop=True)
    groups.to_csv(OUT_GROUPS, index=False)

    # ---- Recovery sizing ------------------------------------------------------
    # A group is SAFELY collapsible when it is the same product (T1.5's rule).
    # Report recovery at several MIN-fuzzy floors so the owner sees how the
    # fuzzy corroboration correlates with the same-product verdict — but the
    # SAME-product count is the floor that T1.5 actually applies.
    floors = [0.5, 0.6, 0.7, 0.8, 0.9]
    recovery_rows = []
    for floor in floors:
        cond = groups["min_fuzzy"] >= floor
        recovery_rows.append({
            "min_fuzzy_floor": floor,
            "same_product_groups": int((cond & groups["same_product"]).sum()),
            "same_product_rows": int(
                groups.loc[cond & groups["same_product"], "rows"].sum()
            ),
            "groups_total": int(cond.sum()),
            "rows_total": int(groups.loc[cond, "rows"].sum()),
        })
    recovery_df = pd.DataFrame(recovery_rows)
    recovery_df.to_csv(OUT_RECOVERY, index=False)

    # ---- Console summary ------------------------------------------------------
    n_same = int(groups["same_product"].sum())
    n_diff = int((~groups["same_product"]).sum())
    print(f"malformed/invalid barcode rows (source dataset.csv): {len(malformed):,}")
    print(f"multi-row (retailer, malformed barcode) groups: {len(groups):,}")
    print(f"  rows inside them: {int(groups['rows'].sum()):,}")
    print(f"  SAME-product groups (T1.5 collapsible): {n_same:,}")
    print(f"  DIFFERENT-product groups (kept separate): {n_diff:,}")
    print("\nRecovery by min-fuzzy floor:")
    for r in recovery_rows:
        print(f"  >= {r['min_fuzzy_floor']}: {r['same_product_groups']} same-product "
              f"groups / {r['same_product_rows']} rows")

    print(f"\nwrote {OUT_GROUPS}, {OUT_RECOVERY}")


if __name__ == "__main__":
    main()
