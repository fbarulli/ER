"""Runner: evaluate labeled GTIN pairs through JEV (zen / openrouter / mock).

Usage:
    python jev/run_eval.py --adapter mock --limit 20
    python jev/run_eval.py --adapter zen   --limit 20
    python jev/run_eval.py --adapter openrouter --limit 20
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from client import JevClient, MockJevClient
from state import load_pairs, load_record_index, pair_state, sample_pairs

MATCH_ABOVE = 0.9   # auto-match band start (from refund-policy sketch)
MATCH_BELOW = 0.1   # auto-decline band end; in-between = human review
MAX_WORKERS = 4


def resolve_pair(client, gtin1, gtin2, a, b) -> dict:
    probs = client.ask_noul(pair_state(gtin1, a, gtin2, b))
    p = probs["is_same_product"]
    if p >= MATCH_ABOVE:
        verdict = "MATCH"
    elif p <= MATCH_BELOW:
        verdict = "NON_MATCH"
    else:
        verdict = "REVIEW"
    return {
        "gtin1": gtin1,
        "gtin2": gtin2,
        "p_same": round(p, 4),
        "pred": int(p >= 0.5),
        "verdict": verdict,
    }


def summarize(rows: list[dict]) -> dict:
    decided = [r for r in rows if r["verdict"] != "REVIEW"]
    res = {
        "n": len(rows),
        "n_decided": len(decided),
        "n_review": len(rows) - len(decided),
    }
    if decided:
        tp = sum(1 for r in decided if r["pred"] == 1 and r["label"] == 1)
        fp = sum(1 for r in decided if r["pred"] == 1 and r["label"] == 0)
        fn = sum(1 for r in decided if r["pred"] == 0 and r["label"] == 1)
        res["accuracy"] = (tp + len([r for r in decided if r["pred"] == 0 and r["label"] == 0])) / len(decided)
        res["precision"] = tp / (tp + fp) if tp + fp else 0.0
        res["recall"] = tp / (tp + fn) if tp + fn else 0.0
        p, rc = res["precision"], res["recall"]
        res["f1"] = 2 * p * rc / (p + rc) if p + rc else 0.0
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", choices=["zen", "openrouter", "mock"], default="zen")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--no-balance", action="store_true")
    ap.add_argument("--out", default=None, help="results csv path")
    args = ap.parse_args()

    records = load_record_index()
    pairs = sample_pairs(load_pairs(), args.limit, balanced=not args.no_balance, seed=args.seed)
    missing = [(p["gtin1"], p["gtin2"]) for p in pairs if p["gtin1"] not in records or p["gtin2"] not in records]
    if missing:
        sys.exit(f"gtins missing from canonical_records.csv: {missing[:5]}")

    print(f"adapter={args.adapter} pairs={len(pairs)} balanced={not args.no_balance}", flush=True)
    if args.adapter == "mock":
        client = MockJevClient()
    else:
        client = JevClient(args.adapter)  # constructs key + live client, but no request made yet

    rows: list[dict] = []
    t0 = time.monotonic()
    def job(p):
        try:
            r = resolve_pair(client, p["gtin1"], p["gtin2"], records[p["gtin1"]], records[p["gtin2"]])
            r["label"] = p["label"]
            return r
        except Exception as e:
            return {"gtin1": p["gtin1"], "gtin2": p["gtin2"], "label": p["label"], "error": str(e)[:200]}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(job, p) for p in pairs]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            rows.append(r)
            if i % 5 == 0 or i == len(pairs):
                ok = sum(1 for x in rows if "error" not in x)
                print(f"  {i}/{len(pairs)} done ({ok} ok, {time.monotonic() - t0:.1f}s)", flush=True)

    errors = [r for r in rows if "error" in r]
    if errors:
        print(f"\n{len(errors)} errors, first:", errors[0]["error"])
    rows_ok = [r for r in rows if "error" not in r]
    for r in rows:
        r.setdefault("p_same", "")
        r.setdefault("pred", "")
        r.setdefault("verdict", "ERROR" if "error" in r else "")

    if rows_ok:
        s = summarize(rows_ok)
        print(f"\n== metrics (on {s['n_decided']}/{s['n']} decided) ==")
        for k in ("accuracy", "precision", "recall", "f1", "n_review"):
            if k in s:
                print(f"  {k:9s} {s[k]:.2%}" if isinstance(s[k], float) else f"  {k:9s} {s[k]}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else Path(__file__).parent / f"results_{args.adapter}_{stamp}.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["gtin1", "gtin2", "label", "p_same", "pred", "verdict"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
