#!/usr/bin/env python3
"""Field-sliced hard-negative harness (eval_slice_by_field.json).

The organic gate sample is a monoculture (pack_blocker only): per-field
model behavior is unmeasurable on it. This harness replaces it with a
synthetic-supported, field-balanced benchmark built from a prepared
bundle's counterfactual twins:

  33% flavor twins / 33% volume twins / 34% package-pack-type twins
  (package bucket = package_type + pack flips)

plus an equal count of organic gate negatives, plus every selected twin's
source positive. Partition is deterministic (sorted by copy index, first K
per bucket where K = the smallest bucket); unknown flipped fields are
excluded, never forced into a bucket.

Usage:
  PYTHONPATH=src .venv/bin/python scripts/build_field_slice.py \\
      --bundle data/prepared/full/worker_1_baseline.pkl.gz \\
      --model minilm_l6 --out results/eval_slice_by_field.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

if __package__:
    from .minimal_flip_slice import _fused_vectors, precision_at_recall
else:
    from minimal_flip_slice import _fused_vectors, precision_at_recall


BUCKET_FIELDS = {
    "flavor": {"flavor"},
    "volume": {"volume"},
    "package": {"package_type", "pack"},
}


def select_field_buckets(
    twin_infos: list[dict], headings: tuple[str, str, str] = ("flavor", "volume", "package")
) -> dict[str, list[dict]]:
    """Deterministic 33/33/34 partition over flipped-field buckets.

    Each bucket keeps its first K twins in copy-index order, where K is the
    smallest bucket size — no RNG, no replacement, no padding. Twins whose
    flipped field maps to no bucket are reported as excluded.
    """
    grouped: dict[str, list[dict]] = {name: [] for name in headings}
    excluded = 0
    for info in sorted(twin_infos, key=lambda row: int(row["copy_idx"])):
        placed = False
        for name in headings:
            if info["field"] in BUCKET_FIELDS[name]:
                grouped[name].append(info)
                placed = True
                break
        excluded += not placed
    sizes = {name: len(rows) for name, rows in grouped.items()}
    smallest = min(sizes.values()) if sizes else 0
    return {
        name: rows[:smallest]
        for name, rows in grouped.items()
    } | {"_excluded_unknown_field": excluded, "_per_bucket": smallest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--model", default="minilm_l6")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("results/eval_slice_by_field.json"))
    args = parser.parse_args()

    from core.common import load_local_sentence_transformer
    from training.prepared_bundle import load_prepared_bundle

    _, data = load_prepared_bundle(args.bundle)
    payload = list(data["payload"])
    structured_features = np.asarray(data["structured_features"], dtype=np.float32)
    neg = np.asarray(data["neg"], dtype=int)
    neg_sources = np.asarray(data["neg_sources"], dtype=object)
    neg_audit = list(data.get("hard_negative_mask_audit", []))

    twins = [
        {
            "copy_idx": int(row["copy_payload_idx"]),
            "anchor_idx": int(row["anchor_payload_idx"]),
            "pair_idx": int(row["pair_payload_idx"]),
            "field": (list(row.get("fields_hit") or ["unknown"]))[0],
        }
        for row in neg_audit
        if row.get("target_mode") == "counterfactual"
    ]
    if not twins:
        raise SystemExit(f"bundle has no counterfactual twins: {args.bundle}")
    partition = select_field_buckets(twins)
    per_bucket = int(partition["_per_bucket"])
    if per_bucket == 0:
        raise SystemExit("a field bucket is empty; cannot balance 33/33/34")
    selected = [
        info for name in ("flavor", "volume", "package") for info in partition[name]
    ]
    gate_idx = [
        i for i, source in enumerate(neg_sources) if str(source) == "gate"
    ][: len(selected)]

    rows: list[dict] = []
    need: set[int] = set()
    for info in selected:
        rows.append({
            "anchor_text_idx": info["anchor_idx"], "other_text_idx": info["pair_idx"],
            "label": 1, "kind": "source_positive", "field": info["field"],
        })
        rows.append({
            "anchor_text_idx": info["copy_idx"], "other_text_idx": info["pair_idx"],
            "source_anchor_idx": info["anchor_idx"],
            "label": 0, "kind": "twin", "field": info["field"],
        })
        need.update({info["anchor_idx"], info["pair_idx"], info["copy_idx"]})
    for i in gate_idx:
        rows.append({
            "anchor_text_idx": int(neg[i][0]), "other_text_idx": int(neg[i][1]),
            "label": 0, "kind": "gate_negative", "field": None,
        })
        need.update(map(int, neg[i]))

    model = load_local_sentence_transformer(args.model, device=args.device)
    vectors = _fused_vectors(model, payload, structured_features, need)

    def sim(left_idx: int, right_idx: int) -> float:
        return float(np.dot(vectors[left_idx], vectors[right_idx]))

    for row in rows:
        row["score"] = sim(row["anchor_text_idx"], row["other_text_idx"])

    def _report(subset: list[dict]) -> dict[str, float]:
        rep = precision_at_recall(
            np.array([row["score"] for row in subset]),
            np.array([row["label"] for row in subset]),
        )
        return {"precision_at_recall_95": float(rep["precision"]), "n": len(subset)}

    by_bucket = {}
    for name in ("flavor", "volume", "package"):
        sub = [row for row in rows if row["field"] in BUCKET_FIELDS[name]]
        margins = [
            sim(row["source_anchor_idx"], row["other_text_idx"]) - row["score"]
            for row in sub if row["kind"] == "twin"
        ]
        by_bucket[name] = _report(sub) | {
            "n_twins": sum(1 for row in sub if row["kind"] == "twin"),
            "margin_mean": float(np.mean(margins)) if margins else float("nan"),
        }
    report = {
        "bundle": str(args.bundle),
        "model": str(args.model),
        "buckets": {name: len(partition[name]) for name in ("flavor", "volume", "package")},
        "n_gate_negatives": len(gate_idx),
        "overall": _report(rows),
        "by_bucket": by_bucket,
        "rows": [
            {
                "anchor_text_idx": row["anchor_text_idx"],
                "other_text_idx": row["other_text_idx"],
                "source_anchor_idx": row.get("source_anchor_idx"),
                "label": row["label"], "kind": row["kind"],
                "field": row["field"], "score": row["score"],
            }
            for row in rows
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"field slice P@R95={report['overall']['precision_at_recall_95']:.4f} "
          f"(n={report['overall']['n']})", flush=True)
    for name in ("flavor", "volume", "package"):
        row = report["by_bucket"][name]
        print(f"  {name:<8} P@R95={row['precision_at_recall_95']:.4f} "
              f"(n={row['n']}, twins={row['n_twins']})", flush=True)
    print(f"report: {args.out}", flush=True)


if __name__ == "__main__":
    main()
