#!/usr/bin/env python3
"""Sanity gate for newly admitted proceed pairs (threshold-lowering guard).

When pairs.proceed_sim_threshold drops, the newly admitted band
(old_floor <= sim < new_floor) must be nearly perfectly clean: every pair
is checked for canonical-attribute agreement (volumes within tolerance,
zero categorical conflicts on every dimension). Exit 0 PASS, exit 2 FAIL
naming the agreement rate. Run BEFORE rebuilding labeled_pairs.csv so a
dirty band can never become labels.

Usage:
  PYTHONPATH=src .venv/bin/python scripts/check_proceed_precision.py \\
      --old-floor 0.80 --min-agreement 0.98
  (new floor and thresholds read from config/training.yaml pairs.*)
"""

from __future__ import annotations

import argparse
import ast
import math
import sys
from pathlib import Path

import pandas as pd

from core.common import F, load_config
from core.critical_attributes import categorical_conflict, volumes_compatible


def _as_set(value: object, column: str) -> set:
    """Parse a canonical set, keeping blank cells as absent evidence.

    Invalid syntax, scalar/dict values, and values of the wrong element type
    must not become empty sets: that would make corrupt records pass the
    compatibility checks as though the source simply lacked evidence.
    """
    if value is None or (
        not isinstance(value, (list, tuple, set, dict)) and pd.isna(value)
    ):
        return set()
    if isinstance(value, str):
        if not value.strip():
            return set()
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"invalid {column} literal {value!r}") from exc
    else:
        parsed = value
    if parsed is None:
        return set()
    if not isinstance(parsed, (set, list, tuple)):
        raise ValueError(
            f"{column} must contain a set/list/tuple, got {type(parsed).__name__}"
        )

    is_numeric = column in {"volume_set", "pack_set"}
    normalized = set()
    for item in parsed:
        if is_numeric:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError(f"{column} contains non-numeric value {item!r}")
            if not math.isfinite(float(item)) or float(item) <= 0:
                raise ValueError(f"{column} contains invalid numeric value {item!r}")
            if column == "pack_set" and not float(item).is_integer():
                raise ValueError(f"{column} contains fractional pack count {item!r}")
            normalized.add(float(item))
        else:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"{column} contains invalid categorical value {item!r}")
            normalized.add(item)
    return normalized


DIMENSION_COLUMNS = (
    ("pack", "pack_set"),
    ("package_type", "package_type_set"),
    ("flavor", "flavor_set"),
    ("carbonation", "carbonation_set"),
    ("sweetener", "sweetener_set"),
    ("pulp", "pulp_set"),
)
ALL_COLUMNS = ("volume_set",) + tuple(column for _, column in DIMENSION_COLUMNS)


def _record(frame_row) -> dict:
    return {column: _as_set(frame_row[column], column) for column in ALL_COLUMNS}


def pair_agrees(left: dict, right: dict) -> bool:
    """True when two canonical records show no attribute conflict."""
    if not volumes_compatible(
        left["volume_set"], right["volume_set"],
        volume_relative_tolerance=0.05, volume_absolute_tolerance_ml=5.0,
    ):
        return False
    for dimension, column in DIMENSION_COLUMNS:
        if categorical_conflict(
            dimension, {dimension: left[column]}, {dimension: right[column]}
        ):
            return False
    return True


def agreement_rate(
    gate: pd.DataFrame, canonicals: pd.DataFrame, lo: float, hi: float
) -> dict[str, object]:
    """Agreement over proceed pairs with lo <= sim < hi (canonical sets)."""
    canon = canonicals.set_index("gtin")
    if canon.index.has_duplicates:
        raise ValueError("canonical records contain duplicate GTINs")
    sub = gate[
        (gate["gate_decision"] == "proceed")
        & (gate["similarity"] >= lo)
        & (gate["similarity"] < hi)
    ]
    checked = agreed = missing = 0
    for row in sub.itertuples():
        try:
            left, right = canon.loc[row.gtin1], canon.loc[row.gtin2]
        except KeyError:
            missing += 1
            continue
        checked += 1
        agreed += pair_agrees(_record(left), _record(right))
    return {
        "n": int(len(sub)), "checked": checked, "missing_canon": missing,
        "agreed": agreed,
        "rate": (agreed / checked) if checked else float("nan"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-floor", type=float, required=True)
    parser.add_argument("--min-agreement", type=float, default=0.98)
    parser.add_argument("--gate-csv", type=Path, default=F["gate_results"])
    parser.add_argument("--canonical-csv", type=Path, default=F["canonical_records"])
    args = parser.parse_args(argv)

    pairs_cfg = load_config()["pairs"]
    new_floor = float(pairs_cfg["proceed_sim_threshold"])
    if not args.old_floor > new_floor:
        print(
            f"old floor {args.old_floor} must exceed new floor {new_floor}",
            file=sys.stderr,
        )
        return 2
    gate = pd.read_csv(
        args.gate_csv, dtype={"gtin1": str, "gtin2": str}, keep_default_na=False
    )
    gate["similarity"] = pd.to_numeric(gate["similarity"], errors="raise")
    canonicals = pd.read_csv(args.canonical_csv, dtype=str, keep_default_na=False)
    report = agreement_rate(gate, canonicals, new_floor, args.old_floor)
    print(
        f"[proceed-precision] band [{new_floor:g}, {args.old_floor:g}): "
        f"n={report['n']} checked={report['checked']} "
        f"missing_canon={report['missing_canon']} "
        f"agreed={report['agreed']} rate={report['rate']:.4f} "
        f">= {args.min_agreement:.4f}",
        flush=True,
    )
    if report["missing_canon"]:
        print(
            f"PRECISION FAIL: {report['missing_canon']} candidate pairs have missing canonical endpoints",
            flush=True,
        )
        return 2
    if not report["checked"]:
        print("PRECISION FAIL: no pairs checked", flush=True)
        return 2
    if not report["rate"] >= args.min_agreement:
        print(
            f"PRECISION FAIL: {report['rate']:.4f} < {args.min_agreement:.4f}",
            flush=True,
        )
        return 2
    print("PRECISION PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
