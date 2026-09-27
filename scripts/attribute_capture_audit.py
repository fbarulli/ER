#!/usr/bin/env python3
"""Attribute capture audit: what do the parsers actually catch?

For every SKU (deduped dataset) and every canonical record, run the SAME
parsers the lanes consume (sku_attribute_info / canonical_attribute_info)
and report, per attribute dimension:
  * coverage (% rows with non-empty capture),
  * where the evidence is missing despite large raw text (top unparsed
    examples — "largest text with nothing captured shows what we missed"),
  * value cardinality (distinct values — a dimension with 3 values or
    30,000 values behaves differently as a matching signal),
  * the zero/empty sentinel rate (volume {0.0} etc. must stay "unknown",
    never a value).

Writes results/attribute_capture.json. Read-only over inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from core.attribute_conflicts import canonical_attribute_info, sku_attribute_info
from core.common import F, TRAIN_ROOT

DIMS = ("volume", "pack", "package_type", "flavor", "carbonation", "sweetener", "pulp")


def _nonempty(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, (set, frozenset, list, tuple)):
        return len(value) > 0
    return bool(str(value).strip())


def _raw_len(title: object, attributes: object) -> int:
    return len(str(title or "")) + len(str(attributes or ""))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str,
                        default=str(TRAIN_ROOT / "results" / "attribute_capture.json"))
    parser.add_argument("--top-misses", type=int, default=5)
    args = parser.parse_args(argv)

    sku = pd.read_csv(F["dataset_deduped"], dtype=str, keep_default_na=False)
    canon = pd.read_csv(F["canonical_records"], dtype=str, keep_default_na=False)
    canon_records = canon.to_dict("records")

    report: dict[str, dict] = {"sku_rows": len(sku), "canon_rows": len(canon_records), "dims": {}}
    for dim in DIMS:
        sku_hits = 0
        values: set[str] = set()
        misses: list[tuple[int, str]] = []  # (raw_len, sku_id/title snippet)
        for _, row in sku.iterrows():
            info = sku_attribute_info(str(row.get("title", "")), str(row.get("attributes", "")))
            got = info.get("flavor_set") if dim == "flavor" else info.get(dim)
            if dim == "flavor":
                got = set(info.get("flavor_set") or set()) | (
                    {info["flavor"]} if info.get("flavor") else set())
            if _nonempty(got):
                sku_hits += 1
                if isinstance(got, (set, frozenset, list, tuple)):
                    values.update(str(v) for v in got)
                else:
                    values.add(str(got))
            else:
                misses.append((_raw_len(row.get("title"), row.get("attributes")),
                               f"{row.get('product_id', '?')}: {str(row.get('title', ''))[:90]}"))
        misses.sort(reverse=True)
        canon_hits = 0
        canon_values: set[str] = set()
        for record in canon_records:
            info = canonical_attribute_info(record)
            got = info.get("flavor_set") if dim == "flavor" else info.get(dim)
            if dim == "flavor":
                got = set(info.get("flavor_set") or set())
            if _nonempty(got):
                canon_hits += 1
                if isinstance(got, (set, frozenset, list, tuple)):
                    canon_values.update(str(v) for v in got)
                else:
                    canon_values.add(str(got))
        report["dims"][dim] = {
            "sku_coverage": sku_hits / len(sku),
            "canon_coverage": canon_hits / len(canon_records),
            "sku_distinct_values": len(values),
            "canon_distinct_values": len(canon_values),
            "top_unparsed": [{"raw_len": n, "sku": s} for n, s in misses[: int(args.top_misses)]],
        }
        print(f"[capture] {dim:<14} sku_cov {sku_hits / len(sku):.3f} "
              f"canon_cov {canon_hits / len(canon_records):.3f} "
              f"distinct sku/canon {len(values):,}/{len(canon_values):,}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")
    print("\nTop unparsed (largest raw text, nothing captured) per dimension:")
    for dim in DIMS:
        print(f"--- {dim} ---")
        for m in report["dims"][dim]["top_unparsed"][:3]:
            print(f"  len={m['raw_len']:>6}  {m['sku'][:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
