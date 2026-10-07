"""Lane C micro-benchmark: CPU-time A/B for the five lane-C owned modules.

Workload: the 11,441-row prefix of the 10k-cohort eligible catalog, fed to the
same five per-column entry points the pipeline uses (see
src/pipeline.py:harvest_evidence_channels / __init__), so each target function
sees the same call counts as the recorded cProfile attribution.

Why CPU time: the host is shared by several lanes, so wall clock is noisy.
time.process_time() measures only this process, and the reported number is the
best (minimum) of repeated passes, which is the most stable estimator of the
work actually removed.

Also emits a sha256 digest per target over its full output. A change that
speeds a target up but alters its output is a regression, and the digest makes
that visible in the same run as the timing.

Usage:
    python artifacts/abl_opt/lane_c/micro_bench.py --out results.json [--label r15]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
os.environ.setdefault("EUROMONITOR_PROJECT_ROOT", str(ROOT))

ROWS = 11441
CATALOG = Path(os.environ.get("LANE_C_CATALOG", "/tmp/opc/eligible_catalog.csv"))


def load_rows() -> list[dict[str, str]]:
    with CATALOG.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = rows[:ROWS]
    columns = ("sku_name_eng", "attribute", "description_short_eng",
               "breadcrumbs_eng", "category", "sku_url", "image_url", "brand")
    return [{col: (row.get(col) or "") for col in columns} for row in rows]


def canonical(value):
    """Order-insensitive canonical form for the digest.

    Sets/frozensets have no insertion-order contract, and repr(frozenset) order
    depends on the hash layout, so a raw repr would report a "difference"
    between two implementations that returned the same set. Lists and tuples
    keep their order (the date ledger's order is data). Dicts are compared by
    key, since every producer here builds a fixed-key literal.
    """
    if isinstance(value, (set, frozenset)):
        return ("<set>", sorted(canonical(item) for item in value))
    if isinstance(value, dict):
        return ("<dict>", sorted((repr(key), canonical(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return ("<seq>", [canonical(item) for item in value])
    if hasattr(value, "start") and hasattr(value, "end") and hasattr(value, "label"):
        # ner.Candidate: a frozen dataclass, so repr is already canonical.
        return repr(value)
    return value


class Sink:
    """Cheap digest of every observed result (correctness gate)."""

    __slots__ = ("_hash", "count")

    def __init__(self) -> None:
        self._hash = hashlib.sha256()
        self.count = 0

    def add(self, value: object) -> None:
        self._hash.update(repr(canonical(value)).encode("utf-8", "surrogatepass"))
        self._hash.update(b"\x1e")
        self.count += 1

    def digest(self) -> str:
        return self._hash.hexdigest()[:16]


class CountingSink:
    """Timing sink: counts samples and nothing else.

    Hashing inside the timed region would add a constant per-sample cost to
    every target and dilute the effect being measured, so the digest pass runs
    separately and is not timed.
    """

    __slots__ = ("count",)

    def __init__(self) -> None:
        self.count = 0

    def add(self, value: object) -> None:
        self.count += 1

    def digest(self) -> str:
        return ""


def build_targets(rows):
    """(name, callable(sink)) pairs — one full corpus pass per call."""
    from core import critical_attributes as ca
    from core import date_evidence as de
    from core import sweetener_values as sv
    from core import url_evidence as ue
    from ner import ner_product_attributes as ner

    urls = [row["sku_url"] for row in rows]
    images = [row["image_url"] for row in rows]
    titles = [row["sku_name_eng"] for row in rows]
    attrs = [row["attribute"] for row in rows]
    descs = [row["description_short_eng"] for row in rows]
    crumbs = [row["breadcrumbs_eng"] for row in rows]
    cats = [row["category"] for row in rows]
    brands = [row["brand"] for row in rows]

    def date_evidence(sink):
        for column in (titles, attrs, descs, crumbs, cats):
            for text in column:
                sink.add(de.extract_date_evidence(text))

    def url_text(sink):
        for column in (urls, images):
            for url in column:
                sink.add(ue.url_text(url))

    def is_noise(sink):
        for url in urls:
            for token in ue.url_text(url).split():
                sink.add(ue._is_noise(token))

    def critical_claims(sink):
        for title, attr in zip(titles, attrs, strict=True):
            sink.add(ca.extract_critical_claims(title, attr))

    def description_claims(sink):
        for desc in descs:
            sink.add(ca.extract_description_claims(desc))

    def flavor_tokens(sink):
        for title in titles:
            sink.add(ca.extract_flavor_tokens(title))

    def declared_flavor(sink):
        for attr in attrs:
            sink.add(ca.extract_declared_flavor_tokens(attr))
        for title, attr in zip(titles, attrs, strict=True):
            sink.add(ca.extract_declared_flavor_tokens(title, attr))

    def made_from(sink):
        for title, attr in zip(titles, attrs, strict=True):
            sink.add(ca.extract_made_from_tokens(title, attr))

    def consistency_flags(sink):
        for title, attr in zip(titles, attrs, strict=True):
            sink.add(ca.source_consistency_flags(
                attr, title, ca.extract_critical_claims(title, attr)["sweetener"]))

    def sweetening_status(sink):
        for title, attr, desc in zip(titles, attrs, descs, strict=True):
            sink.add(sv.extract_sweetening_status(title, attr, desc))

    def title_sweeteners(sink):
        for text in titles:
            sink.add(sv.title_sweetener_types(text))

    def negated_sweeteners(sink):
        for title, attr, desc in zip(titles, attrs, descs, strict=True):
            sink.add(sv.negated_sweetener_types(title, attr, desc))

    def declared_sweeteners(sink):
        for attr in attrs:
            sink.add(sv.declared_sweeteners(attr))

    def ner_details(sink):
        for attr in attrs:
            sink.add(ner.parse_attribute_details(attr))

    def ner_titles(sink):
        for title in titles:
            sink.add(ner.extract_title_attributes(title))

    def ner_candidates(sink):
        for title in titles:
            sink.add(ner._candidates_from_package_details(title))
            sink.add(ner._candidates_from_measurements(title))

    def ner_brand(sink):
        for title, brand in zip(titles, brands, strict=True):
            sink.add(ner.find_brand_span(title, brand))

    return [
        ("date_evidence.extract_date_evidence", date_evidence),
        ("url_evidence.url_text", url_text),
        ("url_evidence._is_noise", is_noise),
        ("critical.extract_critical_claims", critical_claims),
        ("critical.extract_description_claims", description_claims),
        ("critical.extract_flavor_tokens", flavor_tokens),
        ("critical.extract_declared_flavor_tokens", declared_flavor),
        ("critical.extract_made_from_tokens", made_from),
        ("critical.source_consistency_flags", consistency_flags),
        ("sweetener.extract_sweetening_status", sweetening_status),
        ("sweetener.title_sweetener_types", title_sweeteners),
        ("sweetener.negated_sweetener_types", negated_sweeteners),
        ("sweetener.declared_sweeteners", declared_sweeteners),
        ("ner.parse_attribute_details", ner_details),
        ("ner.extract_title_attributes", ner_titles),
        ("ner._candidates_from_package_details", ner_candidates),
        ("ner.find_brand_span", ner_brand),
    ]


def measure(target, sink, min_seconds: float, max_reps: int):
    best = float("inf")
    reps = 0
    while reps < max_reps:
        start = time.process_time()
        target(sink)
        elapsed = time.process_time() - start
        reps += 1
        best = min(best, elapsed)
        if best >= min_seconds:
            break
    return best, reps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="results JSON path")
    parser.add_argument("--label", default="", help="round label recorded in the file")
    parser.add_argument("--min-seconds", type=float, default=0.35,
                        help="grow reps until one pass costs at least this much CPU time")
    parser.add_argument("--max-reps", type=int, default=12)
    parser.add_argument("--only", default="", help="substring filter on target name")
    args = parser.parse_args()

    rows = load_rows()
    results = {"label": args.label, "catalog": str(CATALOG), "rows": len(rows),
               "targets": {}}
    total = 0.0
    for name, target in build_targets(rows):
        if args.only and args.only not in name:
            continue
        counted = CountingSink()
        best, reps = measure(target, counted, args.min_seconds, args.max_reps)
        digest_sink = Sink()
        target(digest_sink)
        results["targets"][name] = {
            "cpu_seconds": round(best, 6), "reps": reps,
            "samples_per_pass": counted.count, "digest": digest_sink.digest(),
        }
        total += best
        print(f"{best:9.4f}s x{reps:<3d} {counted.count:>8d} samples  {name}"
              f"  [{digest_sink.digest()}]", flush=True)
    results["total_cpu_seconds"] = round(total, 6)
    print(f"{total:9.4f}s TOTAL")
    Path(args.out).write_text(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
