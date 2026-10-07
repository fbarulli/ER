"""Build the laya play dataset: data/laya/play_500.csv.

Reads the repo-root export `dataset.csv` (the 10k cohort, 13 columns)
and writes a DETERMINISTIC 500-row play sample (seed 1729):
  * stratified by `category`, approximately proportional to the cohort;
  * the FIRST 40 sampled rows carry `attribute == ""` (the blank filter
    locates them; the laya decision lane's empty-state inputs);
  * the rest of the draw keeps its (filled) attribute;
  * ~30 of the rest carry the LONGEST available attribute strings
    (>140 chars) — the top-30 longest rows of the cohort.

Deterministic: the category allocation is a largest-remainder split over
the draw pool's category sizes and the picks come from one
random.Random(1729) stream over categories in sorted-name order, so a
rerun reproduces the file byte-for-byte.
"""
from __future__ import annotations

import csv
import hashlib
import random
from pathlib import Path

from core.common import TRAIN_ROOT

SAMPLE_ROWS = 500
BLANK_ROWS = 40
LONG_ROWS = 30
LONG_THRESHOLD = 140
SEED = 1729
OUTPUT = TRAIN_ROOT / "data" / "laya" / "play_500.csv"


def main() -> None:
    source = TRAIN_ROOT / "dataset.csv"
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames)
        rows = list(reader)
    if len(rows) < SAMPLE_ROWS:
        raise RuntimeError(
            f"dataset.csv has {len(rows)} rows; the play sample wants "
            f"{SAMPLE_ROWS}")
    pools: dict[str, list[dict]] = {}
    for row in rows:
        pools.setdefault(row["category"], []).append(row)
    longest = sorted(rows, key=lambda row: (-len(row["attribute"]),
                                            row["sku_id"]))[:LONG_ROWS]
    long_keys = {(row["category"], row["sku_id"]) for row in longest}
    pool = {category: [row for row in members
                       if (category, row["sku_id"]) not in long_keys
                       and len(row["attribute"]) <= LONG_THRESHOLD]
            for category, members in pools.items()}
    pool = {category: members for category, members in pool.items()
            if members}
    total = sum(len(members) for members in pool.values())
    raw = {category: len(members) / total for category, members in pool.items()}
    allocation = {category: int(len(pool[category]) / total * (
        SAMPLE_ROWS - len(longest))) for category in pool}
    remainder = (SAMPLE_ROWS - len(longest)) - sum(allocation.values())
    for category in sorted(pool, key=lambda c: (-(raw[c] - allocation[c]), c)):
        if remainder <= 0:
            break
        if allocation[category] < len(pool[category]):
            allocation[category] += 1
            remainder -= 1
    if remainder:
        raise RuntimeError(f"category allocation undershot by {remainder}")
    rng = random.Random(SEED)
    draw: list[dict] = []
    for category in sorted(pool):
        draw.extend(rng.sample(pool[category], allocation[category]))
    blanks = [dict(row, attribute="") for row in draw[:BLANK_ROWS]]
    body = draw[BLANK_ROWS:]
    ordered = blanks + body + longest
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(ordered)
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    with OUTPUT.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    empty = sum(1 for row in written if row["attribute"] == "")
    long_attribute = sum(1 for row in written
                         if len(row["attribute"]) > LONG_THRESHOLD)
    print(f"[laya-play-sample] rows={len(written)} "
          f"attribute-empty={empty} long-attribute={long_attribute} "
          f"sha256={digest} -> {OUTPUT}")


if __name__ == "__main__":
    main()
