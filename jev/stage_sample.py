"""Stage the stratified JEV audit sample (doubled: original + swapped order).

    python jev/stage_sample.py            # writes sample_doubled.json + summary
    python jev/stage_sample.py --mock     # offline wiring check, no network
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

from core.project_root import find_project_root

sys.path.insert(0, str(Path(__file__).parent))

from state import load_record_index, listing_state, load_pairs

ER_ROOT = find_project_root(Path(__file__))
GATE_CSV = ER_ROOT / "data" / "gate_results.csv"


def parse(s: str) -> set:
    try:
        v = ast.literal_eval(s)
        return set(v) if isinstance(v, (list, tuple, set, frozenset)) else set()
    except Exception:
        return set()


def stratum_of(mf_a: str, mf_b: str) -> str:
    if mf_a and mf_b:
        return "proceed_mf_diff" if mf_a != mf_b else "proceed_mf_same"
    if not mf_a and not mf_b:
        return "proceed_mf_both_empty"
    return "proceed_mf_one_empty"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "sample_doubled.json"))
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    rows_df = {r["gtin"]: r for r in csv.DictReader(open(ER_ROOT / "data" / "canonical_records.csv"))}
    gate = list(csv.DictReader(open(GATE_CSV)))
    rng = random.Random(29)

    # stratify proceeds by mode_flavor bucket
    buckets: dict[str, list[dict]] = {}
    for g in gate:
        if g["gate_decision"] == "proceed":
            fa = rows_df[g["gtin1"]]["mode_flavor"].strip().lower()
            fb = rows_df[g["gtin2"]]["mode_flavor"].strip().lower()
            buckets.setdefault(stratum_of(fa, fb), []).append(g)
        elif g["gate_decision"] == "fallback":
            buckets.setdefault("fallback", []).append(g)
        elif g["gate_decision"] == "hard_no":
            # split hard negatives by similarity band
            sim = float(g["similarity"] or 0)
            band = "hardno_simhi" if sim >= 0.8 else "hardno_simmid"
            buckets.setdefault(band, []).append(g)

    alloc = {
        "proceed_mf_diff": 20,
        "proceed_mf_same": 20,
        "proceed_mf_one_empty": 5,
        "proceed_mf_both_empty": 5,
        "fallback": 10,
        "hardno_simhi": 10,
        "hardno_simmid": 10,
    }
    sample = []
    for name, n in alloc.items():
        pool = buckets.get(name, [])
        pick = rng.sample(pool, min(n, len(pool)))
        for g in pick:
            sample.append({"gtin1": g["gtin1"], "gtin2": g["gtin2"], "gate": g["gate_decision"],
                           "similarity": g["similarity"], "stratum": name})

    # DOUBLE: original order + swapped order, interleaved to avoid ordering artifacts
    doubled = []
    for i, s in enumerate(sample):
        doubled.append({**s, "copy": "a_order"})
        doubled.append({**s, "gtin1": s["gtin2"], "gtin2": s["gtin1"], "copy": "b_swapped"})
    interleave = []
    for s in sample:
        interleave.append({**s, "copy": "a_order"})
        interleave.append({**s, "gtin1": s["gtin2"], "gtin2": s["gtin1"], "copy": "b_swapped"})
    doubled = interleave
    json.dump(doubled, open(args.out, "w"), indent=1)

    print(f"staged {len(doubled)} calls ({len(sample)} unique pairs x2 orders)")
    print(Counter(s["stratum"] for s in doubled))

    if args.mock:
        from client import MockJevClient
        records = load_record_index()
        client = MockJevClient()
        for s in doubled[:6]:
            p = client.ask_noul({"record_a": listing_state(s["gtin1"], records[s["gtin1"]]),
                                 "record_b": listing_state(s["gtin2"], records[s["gtin2"]])})
            print("MOCK OK", s["stratum"], s["copy"], p)
    else:
        print("staged only — awaiting explicit approval to send")


if __name__ == "__main__":
    main()
