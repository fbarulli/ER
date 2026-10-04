#!/usr/bin/env python3
"""Review sheet for the SOURCE-defect consistency flags (fcc2c07).

Four flags mark internal source contradictions (the extractor is faithful;
the DATA is wrong), measured 2026-10-03:

  caffeine_source_conflict     positive caffeine band + declared "no caffeine"
  caffeine_without_source      positive caffeine band, no named caffeine
                               source in title/attributes
  no_sugar_with_sugar          "no sugar" claim beside a sugar ingredient
  no_aspartame_with_aspartame  "no aspartame" claim beside aspartame

Doctrine (owner, 2026-10-03): REVIEW-NOT-GUESS. The flags are never
silently dropped or "corrected"; a human reads the sheet and records a
verdict. Verdicts merge through scripts/apply_reading_verdicts.py (same
``"<field>|<id>"`` key format, here ``"<flag>|<gtin>"``).

Read-only over the shared canonical artifact, loaded through the validated
`core.common.canonical_records_frame` (SSOT path + column contract), then
guarded: the flags column must exist and every cell must parse (a literal
failure is surfaced, not swallowed).
"""

from __future__ import annotations

import argparse
import ast
import csv
from collections import Counter
from pathlib import Path

from core.audit_guard import AuditGuardError
from core.common import RESULTS, canonical_records_frame

SOURCE_DEFECT_FLAGS: tuple[str, ...] = (
    "caffeine_source_conflict",
    "caffeine_without_source",
    "no_sugar_with_sugar",
    "no_aspartame_with_aspartame",
)


def _parse_flag_cell(cell: object) -> set[str]:
    """One canonical attribute_consistency_flags cell -> normalized set.

    Accepts Python set/frozenset objects already (fast path) or their CSV
    literal text; ``set()``/empty stay empty. Malformed literals FAIL
    (fail-closed provenance: a flags cell that cannot be parsed is exactly
    what this review exists to catch).
    """
    if cell is None:
        return set()
    if isinstance(cell, (set, frozenset, list, tuple)):
        parsed = cell
    else:
        text = str(cell).strip()
        if not text or text in {"set()", "frozenset()"}:
            return set()
        parsed = ast.literal_eval(text)  # raises on malformed text
    if not isinstance(parsed, (set, frozenset, list, tuple)):
        raise AuditGuardError(
            f"attribute_consistency_flags cell is not a sequence: {cell!r}"
        )
    return {str(item).strip() for item in parsed if str(item).strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=RESULTS / "source_consistency_review"
    )
    args = parser.parse_args()

    frame = canonical_records_frame()
    if "attribute_consistency_flags" not in frame.columns:
        raise AuditGuardError("canonical records carry no attribute_consistency_flags column")
    rows: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    for record in frame.to_dict("records"):
        flags = _parse_flag_cell(record.get("attribute_consistency_flags"))
        defects = sorted(set(flags) & set(SOURCE_DEFECT_FLAGS))
        for flag in defects:
            counts[flag] += 1
        if not defects:
            continue
        for flag in defects:
            rows.append({
                "flag": flag,
                "gtin": record.get("gtin", ""),
                "canonical": record.get("canonical", ""),
                "mode_flavor": record.get("mode_flavor", ""),
                "attribute": " ".join(str(record.get("attribute", "") or "").split()),
                "source_rows": record.get("source_rows", ""),
                "verdict": "",
                "note": "",
            })
    rows.sort(key=lambda row: (row["flag"], row["gtin"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sheet = args.output_dir / "source_defect_sheet.csv"
    columns = ["flag", "gtin", "canonical", "mode_flavor", "attribute",
               "source_rows", "verdict", "note"]
    with sheet.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "canonical_records": len(frame),
        "flagged_gtins": sum(counts.values()),
        "per_flag": dict(sorted(counts.items())),
        "sheet_rows": len(rows),
        "verdict_key_format": "<flag>|<gtin>",
    }
    (args.output_dir / "review_manifest.json").write_text(
        __import__("json").dumps(report, indent=2) + "\n"
    )
    print(__import__("json").dumps(report, indent=2))
    print(f"sheet -> {sheet}")


if __name__ == "__main__":
    main()
