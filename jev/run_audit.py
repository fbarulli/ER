"""Run the staged doubled audit sample through a live JEV adapter.

    python jev/run_audit.py --adapter openrouter
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from client import JevClient
from state import load_record_index, listing_state

AUDIT_JSON = Path(__file__).parent / "sample_doubled.json"
OUT_JSONL = Path(__file__).parent / "audit_results.jsonl"


def load_done(output: Path = OUT_JSONL) -> set[tuple[str, str]]:
    if not output.exists():
        return set()
    keys = set()
    for line in open(output, encoding="utf-8").read().splitlines():
        try:
            r = json.loads(line)
            if r.get("status") == "ok":
                keys.add((r["gtin1"], r["gtin2"]))
        except (json.JSONDecodeError, KeyError):
            continue
    return keys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", choices=["zen", "openrouter"], default="openrouter")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--staging", type=Path, default=AUDIT_JSON)
    ap.add_argument("--out", type=Path, default=OUT_JSONL)
    args = ap.parse_args()
    if args.workers < 1:
        ap.error("--workers must be positive")
    if args.staging != AUDIT_JSON and args.out == OUT_JSONL:
        ap.error("custom --staging requires --out to keep checkpoints separate")

    with args.staging.open(encoding="utf-8") as fh:
        sample = json.load(fh)
    done = load_done(args.out)
    todo = [s for s in sample if (s["gtin1"], s["gtin2"]) not in done]
    print(f"adapter={args.adapter} staged={len(sample)} done={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        return

    records = load_record_index()
    client = JevClient(args.adapter)
    lock = threading.Lock()

    def job(s):
        try:
            probs = client.ask_noul({"record_a": listing_state(s["gtin1"], records[s["gtin1"]]),
                                     "record_b": listing_state(s["gtin2"], records[s["gtin2"]])})
            row = {**s, "noul": probs.get("is_same_product"), "status": "ok"}
        except Exception as e:
            row = {**s, "status": f"error:{str(e)[:160]}"}
        with lock:
            with open(args.out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
        return row

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(job, s) for s in todo]
        for i, f in enumerate(as_completed(futs), 1):
            f.result()
            if i % 20 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} ({time.monotonic()-t0:.0f}s)", flush=True)
    print(f"audit checkpoint now has {len(load_done(args.out))} completed calls")


if __name__ == "__main__":
    main()
