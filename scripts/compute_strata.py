#!/usr/bin/env python3
import json, ast
from collections import Counter, defaultdict
import pandas as pd
from core.common import RESULTS, F
from training.folds import normalize_gtin as n

can = pd.read_csv(RESULTS / F["canonical_records"], dtype=str, keep_default_na=False).drop_duplicates("gtin")

def parse(v):
    try:
        x = ast.literal_eval(str(v))
    except Exception:
        return set()
    return {str(t).strip().lower() for t in x if str(t).strip()}

GATE = ["volume_set", "pack_set", "package_type_set", "flavor_set"]

r = json.load(open("results/permutation_census.json"))

# Cell values are already strings from the census script (parsed list literals)
# Don't convert to tuple - they're already individual values like '900.0'
marg = {f: {vv for vv, cnt in r["marginals"][f].items() if cnt > 50} for f in GATE}
marg_none = {f for f in GATE if "<none>" in marg[f]}

def n_common(cell_dict):
    """Count how many of the 4 gate components are individually common (>50 support).
    
    cell_dict[f] is a string like '900.0' or '<none>', marg[f] is a set of strings.
    """
    return sum(
        1
        for f in GATE
        if cell_dict.get(f, "<none>") in marg[f]
        or (cell_dict.get(f, "<none>") == "<none>" and f in marg_none)
    )

blind = [c for c in r["reachable_absent_cells"] if n_common(c["cell"]) >= 4]
partial = [c for c in r["reachable_absent_cells"] if 3 <= n_common(c["cell"]) < 4]

# weak spots: observed cells with 1-5 support where all 4 components common
observed_support = Counter()
for row in can.itertuples(index=False):
    c = tuple(
        tuple(sorted(parse(getattr(row, f)))) if parse(getattr(row, f)) else ("<none>",)
        for f in GATE
    )
    observed_support[c] += 1

weak = []
for c, s in observed_support.items():
    if 1 <= s <= 5 and n_common(dict(zip(GATE, (v[0] if isinstance(v, tuple) and v else "<none>" for v in c)))) >= 4:
        weak.append((c, s))

print(f"Blind spots: {len(blind)}")
print(f"Partial blind spots: {len(partial)}")
print(f"Weak spots: {len(weak)}")
print()
print("=== PRODUCTS NEEDED PER TIER ===")
print(f"Blind spots: {len(blind)} products (1 per cell)")
print(f"Partial blind spots: {len(partial)} products (1 per cell)")
weak_needed = sum(max(0, 10 - s) for c, s in weak)
print(f"Weak spots: {weak_needed} products (to increase support to 10 per cell)")
print(f"Total products: {len(blind) + len(partial) + weak_needed}")
print()

blind_partial = blind + [c for c in partial]
per_strata = defaultdict(Counter)
for c in blind_partial:
    for f in GATE:
        per_strata[f][str(c["cell"][f])] += 1

print("=== BLIND + PARTIAL PER STRATA ===")
for f in GATE:
    print(f"{f}:")
    for v, cnt in per_strata[f].most_common():
        print(f"  {v}: {cnt}")
print()

weak_strata = defaultdict(Counter)
for c, s in weak:
    cd = dict(zip(GATE, c))
    for f in GATE:
        weak_strata[f][str(cd[f])] += 1

print("=== WEAK SPOTS PER STRATA ===")
for f in GATE:
    print(f"{f}: {len(weak_strata[f])} values, {sum(weak_strata[f].values())} total cells")