"""scripts/laya_verify.py — verify a laya checkpoint on the honest holdout.

Ties the pieces together: compose each holdout pair into laya's identity state,
score it with a fine-tuned checkpoint, and report precision/recall/F1/PR-AUC with
component-clustered CIs per gate stratum (``core.holdout_eval``).

This is the missing link between ``scripts/laya_holdout.py`` (which assembles
the component-disjoint holdout) and ``scripts/laya_compare.py`` (which needs
``gtin1,gtin2,score`` predictions). It writes those predictions AND prints the
stratified report in one pass.

The model call is isolated in :func:`score_states` (lazy ``laya`` import); the
holdout composition and reporting are model-agnostic so they are unit-testable
without laya/torch.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

from scripts.laya_holdout import build_holdout, _read_csv
from training.folds import normalize_gtin

ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = ROOT / "data/track_setup/eligible_catalog.csv"

#: The one question this verifier scores: does the paired evidence support the
#: same grocery item? (`noul`; the positive option returns P(same).)
IDENTITY_QUESTION = {
    "identity_claim": {
        "type": "noul",
        "instructions": "Does the paired attribute evidence support the two rows "
                        "carrying the same grocery item?",
    },
}


def _composer():
    """The corpus pair composer (byte-identical states; reused, never copied)."""
    path = Path(__file__).resolve().parent / "laya_metrics_pairs.py"
    spec = importlib.util.spec_from_file_location("laya_metrics_pairs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def holdout_states(rows: list[dict], catalog: list[dict]) -> tuple[list[dict], list[dict]]:
    """One ``(row, state)`` per labelled holdout pair.

    Returns ``(scored, skipped)``: rows carrying a real 0/1 label and both
    endpoints present in the catalog; everything else (label-less gate strata,
    unknown gtins) is returned in ``skipped`` with a reason.
    """
    composer = _composer()
    by_gtin = {}
    for row in catalog:
        by_gtin.setdefault(normalize_gtin(row.get("gtin")), row)
    scored, skipped = [], []
    for row in rows:
        if row.get("label") not in ("0", "1"):
            skipped.append({**row, "reason": "label-less stratum"})
            continue
        g1, g2 = normalize_gtin(row.get("gtin1")), normalize_gtin(row.get("gtin2"))
        one, two = by_gtin.get(g1), by_gtin.get(g2)
        if one is None or two is None:
            skipped.append({**row, "reason": "endpoint absent from catalog"})
            continue
        side_one = composer.compose_side(one["attribute"])
        side_two = composer.compose_side(two["attribute"])
        scored.append({**row, "state": composer.compose_state(side_one, side_two)})
    return scored, skipped


def predictions_report(scored: list[dict], scores: list[float], *,
                       threshold: float, n_boot: int, seed: int) -> dict:
    """Join scores to labels and compute the clustered, stratified report."""
    from core.holdout_eval import stratified_report

    joined = [{**row, "_score": score} for row, score in zip(scored, scores)]
    return stratified_report(
        joined,
        component_of=lambda r: r.get("component") or r["gtin1"],
        stratum_of=lambda r: r.get("stratum", "unknown"),
        label_of=lambda r: int(r["label"]),
        score_of=lambda r: r["_score"],
        threshold=threshold, n_boot=n_boot, seed=seed)


def score_states(checkpoint: Path, states: list[str], *, device: str = "cpu",
                 batch_size: int = 16, positive_index: int = 1) -> list[float]:
    """Score each identity state with a fine-tuned checkpoint -> P(same).

    ``positive_index`` is the softmax slot of the "true"/positive option (laya's
    noul option order); default 1. Lazy ``laya`` import keeps this module
    importable (and the other helpers testable) without laya/torch installed.
    """
    import numpy as np
    import torch
    from laya import train as laya_train

    model, tok, cfg = laya_train.load_checkpoint(str(checkpoint))
    model = model.to(torch.device(device)).eval()
    max_len = int(cfg.get("max_len", 512))
    head_max_len = int(cfg.get("head_max_len", 192))
    parallel = laya_train.uses_parallel_layout(cfg)
    rows = [{"state": state, "questions": IDENTITY_QUESTION, "expected": {}}
            for state in states]
    items, skipped = laya_train.items_from_rows(
        tok, rows, max_len, head_max_len, label_smoothing=0.0)
    if len(items) != len(states):
        raise RuntimeError(
            f"identity items {len(items)} != states {len(states)} "
            f"(skipped {skipped!r}); the holdout states were not consumed 1:1")
    records = laya_train.calibration_records(
        model, tok, items, device, max_len, head_max_len,
        batch_size=batch_size, parallel=parallel)
    scores: list[float] = []
    for _qtype, logits, _target, _k in records:
        z = np.asarray(logits, dtype=float)
        z = z - z.max()
        p = np.exp(z)
        p = p / p.sum()
        scores.append(float(p[positive_index]))
    return scores


def write_predictions(scored: list[dict], scores: list[float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gtin1", "gtin2", "score"])
        for row, score in zip(scored, scores):
            writer.writerow([row.get("gtin1", ""), row.get("gtin2", ""), score])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="fine-tuned laya checkpoint dir (rl_agent_config.json)")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "results/laya_lane/verify/predictions.csv")
    parser.add_argument("--holdout", type=Path, default=None)
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    if args.holdout is not None:
        rows = _read_csv(args.holdout)
    else:
        rows, _receipt = build_holdout()
    catalog = _read_csv(args.catalog)
    scored, skipped = holdout_states(rows, catalog)
    scores = score_states(args.checkpoint, [row["state"] for row in scored],
                          device=args.device)
    write_predictions(scored, scores, args.out)
    report = predictions_report(scored, scores, threshold=args.threshold,
                                n_boot=args.n_boot, seed=args.seed)
    print(json.dumps({
        "predictions": str(args.out),
        "scored": len(scored),
        "skipped": len(skipped),
        "report": report,
    }, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
