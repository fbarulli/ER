"""scripts/laya_compare.py — score models against the laya holdout, honestly.

Joins each model's pair predictions to the component-disjoint holdout
(``data/laya/holdout.csv``) and reports precision/recall/F1/PR-AUC with
component-clustered bootstrap CIs, overall and per gate-difficulty stratum.

A prediction file is CSV with columns ``gtin1,gtin2,score`` (order-insensitive
join via ``training.folds.normalize_gtin``); the score is P(same item). Only
labelled holdout rows (real listing pairs + P0) enter the truth metrics — the
gate verdicts are label-less difficulty tags, reported as score summaries so a
model that only looks good on the easy bucket is visible.

Usage:
    python scripts/laya_compare.py \
        --predictions laya=path/to/laya_scores.csv \
        --predictions tracks=path/to/tracks_scores.csv \
        --out results/laya_lane/compare.json
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from core.holdout_eval import DEFAULT_BOOTSTRAP, DEFAULT_SEED, stratified_report
from training.folds import normalize_gtin

ROOT = Path(__file__).resolve().parent.parent
HOLDOUT_PATH = ROOT / "data/laya/holdout.csv"


def _pair_key(gtin1: object, gtin2: object) -> tuple[str, str]:
    a, b = normalize_gtin(gtin1), normalize_gtin(gtin2)
    return (a, b) if a <= b else (b, a)


def load_predictions(path: Path) -> dict[tuple[str, str], float]:
    """Read ``gtin1,gtin2,score`` into an order-insensitive score map."""
    scores: dict[tuple[str, str], float] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                score = float(row["score"])
            except (KeyError, TypeError, ValueError):
                continue
            scores[_pair_key(row.get("gtin1"), row.get("gtin2"))] = score
    return scores


def load_holdout(path: Path = HOLDOUT_PATH) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _gate_score_summary(rows: list[dict], threshold: float) -> dict:
    by_stratum: dict[str, list[float]] = {}
    for row in rows:
        if row.get("stratum", "").startswith("gate_"):
            by_stratum.setdefault(row["stratum"], []).append(row["_score"])
    return {
        name: {"n": len(values),
               "mean": statistics.fmean(values),
               "median": statistics.median(values),
               "above_threshold": sum(v >= threshold for v in values)}
        for name, values in sorted(by_stratum.items())
    }


def run_comparison(holdout: list[dict],
                   predictions: dict[str, dict[tuple[str, str], float]], *,
                   threshold: float = 0.5, n_boot: int = DEFAULT_BOOTSTRAP,
                   seed: int = DEFAULT_SEED) -> dict:
    labelled = [row for row in holdout if row.get("label") in ("0", "1")]
    models: dict[str, dict] = {}
    for name, scores in predictions.items():
        joined, unmatched = [], 0
        for row in labelled:
            key = _pair_key(row.get("gtin1"), row.get("gtin2"))
            if key not in scores:
                unmatched += 1
                continue
            joined.append({**row, "_score": scores[key]})
        report = stratified_report(
            joined,
            component_of=lambda r: r.get("component") or r.get("gtin1"),
            stratum_of=lambda r: r.get("stratum", "unknown"),
            label_of=lambda r: int(r["label"]),
            score_of=lambda r: r["_score"],
            threshold=threshold, n_boot=n_boot, seed=seed)
        gate_rows = [{**row, "_score": scores[_pair_key(row.get("gtin1"),
                                                       row.get("gtin2"))]}
                     for row in holdout
                     if _pair_key(row.get("gtin1"), row.get("gtin2")) in scores]
        models[name] = {
            "matched_labelled": len(joined),
            "unmatched_labelled": unmatched,
            "truth": report,
            "gate_score_summary": _gate_score_summary(gate_rows, threshold),
        }
    return {"threshold": threshold, "alpha": 1 - 0.95,
            "n_boot": n_boot, "seed": seed, "models": models}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=Path, default=HOLDOUT_PATH)
    parser.add_argument("--predictions", action="append", default=[],
                        metavar="NAME=PATH",
                        help="one or more model prediction CSVs (repeatable)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    predictions: dict[str, dict] = {}
    for spec in args.predictions:
        name, _, path = spec.partition("=")
        if not path:
            parser.error(f"--predictions must be NAME=PATH, got {spec!r}")
        predictions[name] = load_predictions(Path(path))
    report = run_comparison(load_holdout(args.holdout), predictions,
                            threshold=args.threshold, n_boot=args.n_boot,
                            seed=args.seed)
    text = json.dumps(report, indent=2, default=str)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
