#!/usr/bin/env python3
"""Per-field flip validity, prose contradiction, and transplant concentration.

Answers three open data gaps from ER/TODO.md without any model in the loop:

  flip validity      counterfactual twins assume a flipped field breaks the
                     product identity. If the OLD value's surface form still
                     appears in the prose after the splice (title says
                     "coconut", structured token now says flavor_watermelon),
                     the text is self-contradictory and the label-0 claim
                     rests on a token the prose contradicts — the decorative
                     value -> label noise risk. Measured per field.
  swap contradiction the same prose-retention measure for the value-swap
                     lanes (positive symmetric + hard-negative single-sided):
                     structured tokens change while prose keeps the old word.
  concentration      realized (field, donor value) transplant distribution
                     per lane against the configured field/value caps — the
                     low-cardinality monitor.

Usage:
  PYTHONPATH=src .venv/bin/python scripts/flip_validity_audit.py \
      --bundle data/prepared/full/worker_1_baseline.pkl.gz \
      --out results/flip_validity_audit.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from training.masking import _FIELD_PREFIXES, field_of

_NUMERIC_FIELDS = {"volume", "pack"}


def _value_strings(field: str, token: str) -> list[str]:
    """Human surface forms of one structured token's value."""
    prefixes = _FIELD_PREFIXES[field]
    prefix = next(p for p in prefixes if token.lower().startswith(p))
    raw = token[len(prefix):].replace("_", ".").casefold()
    if field not in _NUMERIC_FIELDS:
        return [raw]
    try:
        number = float(raw)
    except ValueError:
        return [raw]
    if number <= 0 or number != int(number):
        return [raw]
    ml = int(number)
    forms = [str(ml)]
    if ml % 1000 == 0:
        forms.append(str(ml // 1000))
    if ml % 10 == 0:
        forms.append(str(ml // 10))
    litres = ml / 1000.0
    text = f"{litres:.3f}".rstrip("0").rstrip(".")
    if text:
        forms.extend([text.replace(".", ","), text])
    seen, ordered = set(), []
    for form in forms:
        if form not in seen:
            seen.add(form)
            ordered.append(form)
    return ordered


def _prose(text: str) -> str:
    """Non-structured tokens of one payload text, joined."""
    return " ".join(tok for tok in text.split() if field_of(tok) is None)


def _prose_hit(field: str, values: list[str], prose: str) -> bool:
    """True when any value's surface form appears in the prose."""
    if not prose:
        return False
    for value in values:
        if field in _NUMERIC_FIELDS:
            pattern = r"(?<![0-9.,])" + re.escape(value) + r"(?![0-9])"
        else:
            pattern = r"(?<![a-z0-9])" + re.escape(value) + r"(?![a-z0-9])"
        if re.search(pattern, prose):
            return True
    return False


def _token_values(field: str, tokens: list[str]) -> list[str]:
    out: list[str] = []
    for token in tokens:
        prefix = next(
            p for p in _FIELD_PREFIXES[field] if token.lower().startswith(p)
        )
        out.append(token[len(prefix):].replace("_", ".").casefold())
    return out


def _audit_rows(data: dict) -> list[tuple[str, dict]]:
    """(lane, row) pairs: positive swaps, negative swaps, twins."""
    lanes: list[tuple[str, dict]] = []
    for row in data.get("mask_audit", []):
        if row["target_mode"] == "swap_values":
            lanes.append(("swap_positive", row))
    for row in data.get("hard_negative_mask_audit", []):
        if row["target_mode"] == "swap_values":
            lanes.append(("swap_negative", row))
        elif row["target_mode"] == "counterfactual":
            lanes.append(("twin", row))
    return lanes


def _classify(row: dict) -> dict:
    """Prose-contradiction classification for one swap/twin audit row."""
    field = row["fields_hit"][0]
    anchor_fields = [
        tok for tok in row["anchor_text"].split() if field_of(tok) == field
    ]
    copy_fields = [
        tok for tok in row["masked_text"].split() if field_of(tok) == field
    ]
    old_values = _value_strings(field, anchor_fields[0]) if anchor_fields else []
    new_values = _value_strings(field, copy_fields[0]) if copy_fields else []
    prose = _prose(row["masked_text"])
    old_in_prose = _prose_hit(field, old_values, prose)
    new_in_prose = _prose_hit(field, new_values, prose)
    if old_in_prose and new_in_prose:
        verdict = "both"
    elif old_in_prose:
        verdict = "contradicted"
    elif new_in_prose:
        verdict = "supported"
    else:
        verdict = "opaque"
    return {
        "field": field,
        "old_value": " ".join(_token_values(field, anchor_fields)),
        "new_value": " ".join(_token_values(field, copy_fields)),
        "old_in_prose": old_in_prose,
        "new_in_prose": new_in_prose,
        "verdict": verdict,
        "prose_after": _prose(row["masked_text"])[:200],
    }


def _per_field(rows: list[dict]) -> dict:
    """Per-field verdict shares plus one contradicted example each."""
    by_field: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_field[row["field"]].append(row)
    report: dict[str, dict] = {}
    for field, field_rows in sorted(by_field.items()):
        counts = Counter(r["verdict"] for r in field_rows)
        total = len(field_rows)
        example = next(
            (r for r in field_rows if r["verdict"] == "contradicted"), None
        )
        report[field] = {
            "n": total,
            "contradicted": round(counts["contradicted"] / total, 4),
            "supported": round(counts["supported"] / total, 4),
            "opaque": round(counts["opaque"] / total, 4),
            "both": round(counts["both"] / total, 4),
            "example": (
                {
                    "old": example["old_value"],
                    "new": example["new_value"],
                    "prose_after": example["prose_after"],
                }
                if example
                else None
            ),
        }
    return report


def _concentration(rows: list[tuple[str, dict]], field_cap: float, value_cap: float) -> dict:
    """Realized transplant distribution per lane and globally against the caps.

    Per-lane shares are diagnostic only: the caps bind against the SHARED
    budget (cap_base across all lanes that share the value counter), so the
    global section is the cap surface and the per-lane sections show where
    the picks come from.
    """
    report: dict[str, dict] = {}
    lanes = sorted({lane for lane, _ in rows})
    for lane in lanes + ["global"]:
        lane_rows = [row for l, row in rows if lane == "global" or l == lane]
        n = len(lane_rows)
        if not n:
            continue
        field_counts: Counter[str] = Counter()
        value_counts: Counter[tuple[str, str]] = Counter()
        for row in lane_rows:
            field = row["field"]
            field_counts[field] += 1
            value_counts[(field, row["new_value"])] += 1
        per_field = {}
        for field, count in field_counts.most_common():
            field_values = [
                ((f, v), c) for (f, v), c in value_counts.items() if f == field
            ]
            top_value, top_count = max(field_values, key=lambda kv: kv[1])
            per_field[field] = {
                "share": round(count / n, 4),
                "share_vs_field_cap": round(count / n / field_cap, 3),
                "unique_values": len(field_values),
                "top_value": top_value[1],
                "top_value_share": round(top_count / n, 4),
                "top_value_share_vs_value_cap": round(
                    top_count / n / value_cap, 3
                ),
            }
        report[lane] = {"n": n, "fields": per_field}
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/flip_validity_audit.json"))
    args = parser.parse_args(argv)

    from core.common import load_config
    from training.prepared_bundle import load_prepared_bundle

    masking = load_config()["masking"]
    field_cap = float(masking["swap_max_field_share"])
    value_cap = float(masking["swap_max_value_share"])

    _, data = load_prepared_bundle(args.bundle)
    lanes = _audit_rows(data)
    classified = [(lane, _classify(row)) for lane, row in lanes]

    flip_report: dict[str, dict] = {}
    for lane in ("twin", "swap_positive", "swap_negative"):
        rows = [c for l, c in classified if l == lane]
        if rows:
            flip_report[lane] = _per_field(rows)

    concentration = _concentration(classified, field_cap, value_cap)

    out = {
        "bundle": str(args.bundle),
        "caps": {"swap_max_field_share": field_cap, "swap_max_value_share": value_cap},
        "flip_validity": flip_report,
        "concentration": concentration,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))

    for lane, report in flip_report.items():
        print(f"[{lane}]")
        for field, stats in sorted(report.items()):
            print(
                f"  {field:<14} n={stats['n']:>5}  "
                f"contradicted={stats['contradicted']:.2f}  "
                f"supported={stats['supported']:.2f}  "
                f"opaque={stats['opaque']:.2f}"
            )
    print("[concentration] field share (cap 0.35) / top value share (cap 0.03)")
    for lane, report in concentration.items():
        for field, stats in report["fields"].items():
            print(
                f"  {lane:<14} {field:<14} share={stats['share']:.3f}  "
                f"top='{stats['top_value']}' {stats['top_value_share']:.3f}  "
                f"unique_values={stats['unique_values']}"
            )
    print(f"[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
