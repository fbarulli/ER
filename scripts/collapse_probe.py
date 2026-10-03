#!/usr/bin/env python3
"""Collapse probe (diagnostic): unrelated-pair cosine distribution per encoder.

Reuses the lane's own uniformity sampler (brand/category/token-disjoint
pairs) and collapse-guardrail SSOT (operating_threshold, crossing_rate
ceiling, pair count). Run the same probe on zero-shot and fine-tuned
encoders: a healthy encoder sits low (zero-shot MiniLM: median ~0.34);
a collapsed one compresses upward (observed medians up to 0.83).

Usage:
  python scripts/collapse_probe.py --encoders minilm_l6 CKPT=/path/to/checkpoint
Writes results/collapse_probe.json. Exits nonzero on missing bundle/rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from core.attribute_conflicts import sku_attribute_info
from core.common import F, SEED, TRAIN_ROOT, load_config, load_local_sentence_transformer, resolve_model
from core.model_input import build_sku_text
from training.uniformity import select_unrelated_pairs

_RESULTS = TRAIN_ROOT / "results" / "collapse_probe.json"


def _fail(message: str) -> int:
    print(f"collapse probe ABORT: {message}", file=sys.stderr, flush=True)
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoders", nargs="+", default=["minilm_l6"],
                        help="registry keys and/or local paths (NAME=spec or bare spec)")
    parser.add_argument("--pool", type=int, default=2000,
                        help="deduped head rows mined for unrelated pairs")
    parser.add_argument("--out", type=str, default=str(_RESULTS))
    args = parser.parse_args(argv)

    cfg = load_config()
    guard = cfg["collapse_guardrail"]
    n_pairs = int(guard["unrelated_pairs"])
    max_tf = float(guard["max_token_frequency"])
    thr = float(guard["operating_threshold"])
    ceil = float(guard["crossing_rate_ceiling"])

    pool = pd.read_csv(F["dataset_deduped"], dtype=str, keep_default_na=False).head(int(args.pool))
    if "brand" not in pool.columns or "category" not in pool.columns:
        return _fail("deduped frame lacks brand/category columns")
    infos = [sku_attribute_info(str(r.get("sku_name_eng", "")), str(r.get("attribute", "")))
             for _, r in pool.iterrows()]
    sys.path.insert(0, str(TRAIN_ROOT / "scripts"))
    from sid_phase0_report import _text_info  # noqa: E402
    payload = [build_sku_text(row, _text_info(info)) for (_, row), info in zip(pool.iterrows(), infos)]
    pairs = select_unrelated_pairs(pool, payload, n_pairs=n_pairs, seed=int(SEED), max_token_frequency=max_tf)
    if len(pairs) < n_pairs:
        print(f"[probe] WARN: only {len(pairs)}/{n_pairs} unrelated pairs (pool too overlapping)", flush=True)
    if not pairs:
        return _fail("zero unrelated pairs — cannot probe collapse")

    specs: list[tuple[str, str]] = []
    for raw in args.encoders:
        if "=" in raw:
            name, spec = raw.split("=", 1)
        else:
            name, spec = raw.replace("/", "_"), raw
        specs.append((name, spec))

    report: dict[str, dict] = {}
    for name, spec in specs:
        try:
            resolve_model(spec)
            model = load_local_sentence_transformer(spec, device=DEVICE)
            model.max_seq_length = int(cfg["training"]["max_seq_length"])  # SSOT
            batch_size = int(cfg["training"]["batch_size_embed"])  # SSOT
            emb = np.asarray(model.encode(
                payload, batch_size=batch_size, show_progress_bar=False,
                normalize_embeddings=True, convert_to_numpy=True), dtype=np.float64)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (FileNotFoundError, KeyError, ValueError, OSError, RuntimeError) as exc:
            return _fail(f"encoder {name!r} ({spec}) missing or unloadable — {exc}")
        ai = np.asarray([p[0] for p in pairs])
        bi = np.asarray([p[1] for p in pairs])
        scores = np.einsum("ij,ij->i", emb[ai], emb[bi])
        crossing = float(np.mean(scores >= thr))
        report[name] = {
            "spec": spec,
            "n_pairs": len(pairs),
            "median": float(np.median(scores)),
            "p90": float(np.quantile(scores, 0.90)),
            "std": float(np.std(scores)),
            "crossing_rate_at_operating": crossing,
            "ceiling": ceil,
            "collapsed": bool(crossing > ceil),
        }
        print(f"[probe] {name:<28} median {np.median(scores):.3f} "
              f"p90 {np.quantile(scores, 0.90):.3f} cross@{thr} {crossing:.3f} "
              f"(ceiling {ceil}) {'COLLAPSED' if crossing > ceil else 'healthy'}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"threshold": thr, "ceiling": ceil, "encoders": report}, indent=2) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
