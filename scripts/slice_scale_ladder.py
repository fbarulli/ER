#!/usr/bin/env python3
"""Slice-scale ladder — how many field-focused samples until the model picks
the attribute up?

Custom experiment (2026-10-01). Isolates ONE structured field group at a
time and measures the smallest training population at which the model
separates a verified positive from its minimal flip (the one-field twin):

  rung k      a nested, seeded sample of that slice's pairs, trained as
              MNRL triples (anchor = A1, positive = pair side, negative = twin)
  eval        held-out twins of the same slice, excluded from every rung
  pickup      flipped rate >= min_flipped AND margin >= min_margin
              (margin floor pin: lift off the observed ~0.008 — the
              pre-registered sweetener failure signature inverted)
  k*          smallest ladder rung passing on BOTH seeds

The curve answers "how thin is this slice really": strong fields should
pick the break up at the first rungs; prose-carried keys should state how
many declared pairs the signal takes before it becomes learnable.

Usage:
  PYTHONPATH=src .venv/bin/python scripts/slice_scale_ladder.py \
      --bundle data/prepared/full/worker_1_baseline.pkl.gz \
      --model minilm_l6 --out results/slice_scale_ladder.json
  # subset / custom ladder / one slice for probing:
  #   --slices sweetener,flavor --ladder 25,50,100
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--model", default="minilm_l6")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ladder", default="25,50,100,200,400,800",
                        help="ascending sample counts per slice (nested)")
    parser.add_argument("--slices", default=None,
                        help="comma list; default: every observed flip field")
    parser.add_argument("--max-steps-per-pair", type=float, default=2.0,
                        help="MNRL gradient steps per ladder pair")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--min-flipped", type=float, default=0.80)
    parser.add_argument("--min-margin", type=float, default=0.008)
    parser.add_argument("--out", type=Path, default=Path("results/slice_scale_ladder.json"))
    args = parser.parse_args()

    import numpy as np
    import torch

    from core.common import load_local_sentence_transformer
    from training.prepared_bundle import load_prepared_bundle

    _, data = load_prepared_bundle(args.bundle)
    payload = list(data["payload"])
    structured_features = np.asarray(data["structured_features"], dtype=np.float32)

    # Slice triples from the counterfactual audit: each twin row is
    # (A1', pair side) label 0, source (A1, pair side) label 1. MNRL triple
    # = (A1, pair-side, twin).
    slices: dict[str, list[tuple[int, int, int]]] = {}
    for row in data.get("hard_negative_mask_audit", []):
        if row.get("target_mode") != "counterfactual":
            continue
        fields = list(row.get("fields_hit") or [])
        if len(fields) != 1 or fields[0] == "unknown":
            continue
        slices.setdefault(fields[0], []).append(
            (int(row["anchor_payload_idx"]), int(row["pair_payload_idx"]),
             int(row["copy_payload_idx"]))
        )
    wanted = (
        [s.strip() for s in args.slices.split(",") if s.strip()]
        if args.slices else sorted(slices)
    )
    missing = [s for s in wanted if s not in slices]
    if missing:
        raise SystemExit(f"no twins minted for slice(s): {missing} (have {sorted(slices)})")

    ladder_rungs = sorted(int(x) for x in args.ladder.split(","))
    seeds = [int(x) for x in args.seeds.split(",")]

    def evaluate(model, triples, holdout_pick: set[int]) -> dict[str, float]:
        """Held-out twin pickup on the SAME slice twins for every rung."""
        idx = sorted(holdout_pick)
        if not idx:
            return {"flipped_rate": float("nan"), "margin_mean": float("nan"), "n": 0}
        vectors = _fused_vectors(
            model, payload, structured_features,
            {t for tri in (triples[i] for i in idx) for t in tri},
        )
        margins = [
            float(np.dot(vectors[a], vectors[p])) - float(np.dot(vectors[t], vectors[p]))
            for a, p, t in (triples[i] for i in idx)
        ]
        margins_arr = np.asarray(margins)
        return {
            "flipped_rate": float((margins_arr > 0).mean()),
            "margin_mean": float(margins_arr.mean()),
            "margin_std": float(margins_arr.std()),
            "n": len(idx),
        }

    from datasets import Dataset

    from sentence_transformers.sentence_transformer import (
        SentenceTransformerTrainingArguments,
        SentenceTransformerTrainer,
    )
    from sentence_transformers.sentence_transformer import losses as _losses

    def train_rung(train_triples: list[tuple[int, int, int]], seed: int):
        """Fresh model per rung; MNRL triple loss on that slice's pairs only."""
        model = load_local_sentence_transformer(args.model, device=args.device)
        args_hf = SentenceTransformerTrainingArguments(
            output_dir=str(args.out.parent / "_ladder_runs"),
            max_steps=math.ceil(len(train_triples) * args.max_steps_per_pair),
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.lr,
            seed=seed,
            reporting_to=[],
            disable_tqdm=True,
            save_strategy="no",
            logging_steps=1e9,
        )
        trainer = SentenceTransformerTrainer(
            model=model,
            args=args_hf,
            loss=_losses.MultipleNegativesRankingLoss(model),
            train_dataset=Dataset.from_dict({
                "anchor": [payload[a] for a, _, _ in train_triples],
                "positive": [payload[p] for _, p, _ in train_triples],
                "negative": [payload[t] for _, _, t in train_triples],
            }),
        )
        trainer.train()
        return model

    report: dict[str, dict] = {}
    for slice_name in wanted:
        triples = slices[slice_name]
        # Held-out eval set reserved ONCE per slice (nested rungs draw only
        # from the remainder), so every rung meets the SAME twins.
        shuffle = random.Random(777)
        order = list(range(len(triples)))
        shuffle.shuffle(order)
        n_hold = max(50, math.ceil(len(order) * 0.15))
        holdout_pick = set(order[:n_hold])
        pool_order = order[n_hold:]
        nested_rng = random.Random(999)
        chosen: list[int] = []
        nested: dict[int, list[int]] = {}
        for k in ladder_rungs:
            while len(chosen) < min(k, len(pool_order)):
                idx = pool_order[nested_rng.randrange(len(pool_order))]
                if idx not in set(chosen):
                    chosen.append(idx)
            nested[k] = list(chosen)
        curves: dict[int, dict] = {}
        for k in ladder_rungs:
            rials = [triples[i] for i in nested[k]]
            rungs = []
            for seed in seeds:
                model = train_rung(rials, seed)
                ev = evaluate(model, triples, holdout_pick)
                ev["ladder_k"], ev["seed"] = k, seed
                del model
                rungs.append(ev)
            flipped = [r["flipped_rate"] for r in rungs]
            margins = [r["margin_mean"] for r in rungs]
            curves[k] = {
                "flipped_rate_mean": math.fsum(flipped) / len(flipped),
                "margin_mean_mean": math.fsum(margins) / len(margins),
                "picked_up": all(
                    r["flipped_rate"] >= args.min_flipped
                    and r["margin_mean"] >= args.min_margin
                    for r in rungs
                ),
                "per_seed": rungs,
            }
        first_pick = next((k for k in ladder_rungs if curves[k]["picked_up"]), None)
        report[slice_name] = {
            "twins_minted": len(triples),
            "holdout_twins": len(holdout_pick),
            "k_star": first_pick,
            "ladder": {str(k): curves[k] for k in ladder_rungs},
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "note": "slice-scale ladder: smallest field-focused sample the model separates from its minimal flip",
        "bundle": str(args.bundle), "model": args.model,
        "ladder": ladder_rungs, "seeds": seeds,
        "pickup_rule": {"min_flipped": args.min_flipped, "min_margin": args.min_margin},
        "slices": report,
    }
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(f"slice-scale ladder · model={args.model} · seeds={seeds}", flush=True)
    print(f"{'slice':<20}{'k*':>7}{'twins':>7}{'holdout':>9}", flush=True)
    for s in wanted:
        row = report[s]
        print(f"{s:<20}{str(row['k_star']):>7}{row['twins_minted']:>7}{row['holdout_twins']:>9}", flush=True)
    print(f"report: {args.out}", flush=True)


if __name__ == "__main__":
    main()
