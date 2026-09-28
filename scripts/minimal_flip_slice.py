#!/usr/bin/env python3
"""Minimal-flip stress slice: measure what macro metrics wash out.

Overall AUC/F1 are dominated by easy negatives, so single-token effects
(counterfactual twins, cross-brand lookalikes) disappear in the average.
This script scores ISOLATED slices from a prepared bundle with one encoder:

  twin slice       every counterfactual twin (label 0) + its source
                   positive (label 1) — pairs differing in exactly one
                   agreed field. Metric: Precision @ Recall 95 + the twin
                   margin sim(anchor, pos) - sim(anchor, twin), overall and
                   per flipped field (the field-bias watch).
  cross-brand slice  eval negatives with a cross_brand_conflict source
                   (label 0) + a matched count of positives (label 1).
  gate slice         eval negatives with a gate source (label 0) + matched
                   positives — the organic background for comparison.

It also reports two full-run diagnostics:

  donor uniformity   reuse distribution over donor rows AND over
                   (field, value) transplants for the swap lanes. A single
                   value above ~3% of a field's transplants is a
                   transplant footprint (the model can memorize the donor
                   string instead of the conflict rule).
  subset split       mean cosine sim for twin-0 vs gate-0 vs pos-1 rows —
                   the twin-vs-easy divergence watch, tracked separately
                   per subset instead of inside one pooled loss number.

Same script scores any encoder the registry resolves — a base model for the
pre-training baseline, or a checkpoint path for the training curve. Re-run
per checkpoint to draw the twin P@R95 curve over epochs.

Usage:
  PYTHONPATH=src .venv/bin/python scripts/minimal_flip_slice.py \\
      --bundle /tmp/opencode/swap_probe3.pkl.gz --model minilm_l6 \\
      --out results/minimal_flip_slice.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def precision_at_recall(
    scores: np.ndarray, labels: np.ndarray, target: float = 0.95
) -> dict[str, float | int]:
    """Precision/recall/threshold at a target recall (07-series convention).

    Threshold = the HIGHEST observed score achieving the target recall. Every
    row tied at that threshold is accepted, matching threshold-based scoring.
    """
    ordered_scores = np.asarray(scores, dtype=float)
    ordered_labels = np.asarray(labels, dtype=int)
    if len(ordered_scores) != len(ordered_labels):
        raise ValueError(
            f"scores/labels length mismatch: {len(ordered_scores)} != {len(ordered_labels)}"
        )
    order = np.argsort(-ordered_scores, kind="stable")
    ranked_scores = ordered_scores[order]
    ranked_labels = ordered_labels[order]
    n_pos = int((ordered_labels == 1).sum())
    if n_pos == 0 or len(ordered_scores) == 0:
        return {
            "precision": float("nan"), "recall": float("nan"),
            "threshold": float("nan"), "tp": 0, "fp": 0,
            "n_pos": n_pos, "n": int(len(ordered_scores)),
        }
    tp_cum = np.cumsum(ranked_labels == 1)
    rank = int(np.searchsorted(tp_cum, int(np.ceil(target * n_pos))))
    rank = min(rank, len(ranked_scores) - 1)
    threshold = float(ranked_scores[rank])
    accepted = ranked_scores >= threshold
    tp = int(np.sum((ranked_labels == 1) & accepted))
    fp = int(np.sum((ranked_labels == 0) & accepted))
    return {
        "precision": float(tp / (tp + fp)) if tp + fp else float("nan"),
        "recall": float(tp / n_pos),
        "threshold": threshold,
        "tp": tp, "fp": fp, "n_pos": n_pos, "n": int(len(ordered_scores)),
    }


def donor_uniformity(
    audits: list[dict], payload: list[str],
) -> dict[str, object]:
    """Reuse distribution over donor rows and transplanted values.

    Returns per-lane draws, unique donors, the max single-donor share, the
    top-10 share, and — per field — the max single-VALUE share with the
    offending value named (donor value read back from the payload, not
    guessed from text). The transplant-footprint rule: any one value
    above ~3% of its field's transplants deserves a cap or a wider pool.
    """
    from training.masking import _field_surfaces

    rows = [row for row in audits if row.get("donor_anchor_payload_idx") is not None]
    donor_counts: Counter[int] = Counter(
        int(row["donor_anchor_payload_idx"]) for row in rows
    )
    draws = len(rows)
    by_field: dict[str, Counter[tuple[str, ...]]] = {}
    for row in rows:
        fields = list(row.get("fields_hit") or [])
        if len(fields) != 1:
            continue
        donor_text = payload[int(row["donor_anchor_payload_idx"])]
        value = tuple(sorted(
            t.lower() for t in _field_surfaces(donor_text).get(fields[0], [])
        ))
        by_field.setdefault(fields[0], Counter())[value] += 1
    field_report = {}
    for field, counts in sorted(by_field.items()):
        total = sum(counts.values())
        (top_value, top_count) = counts.most_common(1)[0]
        field_report[field] = {
            "transplants": total,
            "unique_values": len(counts),
            "max_value_share": (top_count / total) if total else 0.0,
            "max_value_preview": " ".join(top_value)[:80],
        }
    top_rows = donor_counts.most_common(10)
    return {
        "draws": draws,
        "n_donor_rows": len(donor_counts),
        "max_row_share": (top_rows[0][1] / draws) if draws else 0.0,
        "top10_row_share": (sum(c for _, c in top_rows) / draws) if draws else 0.0,
        "by_field": field_report,
    }


def _cosine(model, texts: list[str]) -> np.ndarray:
    vectors = np.asarray(
        model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
        dtype=float,
    )
    return vectors


def _fused_vectors(
    model, payload: list[str], features: np.ndarray, rows: set[int]
) -> dict[int, np.ndarray]:
    """Encode distinct payload rows and apply the production structured fusion."""
    from core.common import load_config
    from core.structured_features import fuse_numpy

    ordered = sorted(rows)
    embeddings = _cosine(model, [payload[row] for row in ordered])
    cfg = load_config()["training"]["structured_features"]
    weight = (
        float(cfg["embedding_weight"])
        if bool(cfg["enabled"]) and bool(cfg["feed_to_loss"])
        else 0.0
    )
    fused = fuse_numpy(embeddings, features[ordered], weight)
    return {row: vec for row, vec in zip(ordered, fused, strict=True)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--model", default="minilm_l6")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("results/minimal_flip_slice.json"))
    parser.add_argument("--gate-sample", type=int, default=500)
    parser.add_argument("--pos-sample", type=int, default=500)
    args = parser.parse_args()

    from core.common import load_local_sentence_transformer
    from training.prepared_bundle import load_prepared_bundle

    _, data = load_prepared_bundle(args.bundle)
    payload = list(data["payload"])
    structured_features = np.asarray(data["structured_features"], dtype=np.float32)
    pos = np.asarray(data["pos"], dtype=int)
    neg = np.asarray(data["neg"], dtype=int)
    neg_sources = np.asarray(data["neg_sources"], dtype=object)
    neg_audit = list(data.get("hard_negative_mask_audit", []))
    mask_audit = list(data.get("mask_audit", []))

    twins = [row for row in neg_audit if row.get("target_mode") == "counterfactual"]
    if not twins:
        raise SystemExit(f"bundle has no counterfactual twins: {args.bundle}")

    model = load_local_sentence_transformer(args.model, device=args.device)

    # Twin slice: each twin (0) with its source positive (1).
    twin_scores: list[float] = []
    twin_labels: list[int] = []
    margins: list[float] = []
    field_groups: dict[str, dict[str, list]] = {}
    need_rows: set[int] = set()
    for row in twins:
        need_rows.update(
            int(row[k])
            for k in ("anchor_payload_idx", "pair_payload_idx", "copy_payload_idx")
        )
    # Background slices share the same encode batch.
    gate_idx = [i for i, s in enumerate(neg_sources) if str(s) == "gate"][: args.gate_sample]
    for i in gate_idx:
        need_rows.update(map(int, neg[i]))
    cross_idx = [
        i for i, s in enumerate(neg_sources) if str(s) == "cross_brand_conflict"
    ][: args.gate_sample]
    for i in cross_idx:
        need_rows.update(map(int, neg[i]))
    pos_idx = list(range(min(args.pos_sample, len(pos))))
    for i in pos_idx:
        need_rows.update(map(int, pos[i]))

    vectors = _fused_vectors(model, payload, structured_features, need_rows)

    def sim(left: int, right: int) -> float:
        return float(np.dot(vectors[left], vectors[right]))

    for row in twins:
        anchor = int(row["anchor_payload_idx"])
        other = int(row["pair_payload_idx"])
        twin = int(row["copy_payload_idx"])
        pos_sim = sim(anchor, other)
        twin_sim = sim(twin, other)
        twin_scores += [pos_sim, twin_sim]
        twin_labels += [1, 0]
        margins.append(pos_sim - twin_sim)
        field = (list(row.get("fields_hit") or ["unknown"]))[0]
        group = field_groups.setdefault(field, {"scores": [], "labels": []})
        group["scores"] += [pos_sim, twin_sim]
        group["labels"] += [1, 0]

    twin_report = precision_at_recall(np.array(twin_scores), np.array(twin_labels))
    margins_arr = np.array(margins, dtype=float)
    by_field = {}
    for field in sorted(field_groups):
        group = field_groups[field]
        rep = precision_at_recall(np.array(group["scores"]), np.array(group["labels"]))
        group_margins = np.array([
            s - t for s, t, lab in zip(
                group["scores"][0::2], group["scores"][1::2], group["labels"][0::2],
                strict=True,
            )
        ])
        by_field[field] = {
            "n_twins": len(group_margins),
            "precision_at_recall_95": rep["precision"],
            "margin_mean": float(group_margins.mean()),
            "margin_frac_positive": float((group_margins > 0).mean()),
        }

    # Background slices use the same fused representation as the twin slice.
    gate_scores = [sim(int(a), int(b)) for a, b in neg[gate_idx]]
    pos_scores = [sim(int(a), int(b)) for a, b in pos[pos_idx]]
    cross_scores = [sim(int(a), int(b)) for a, b in neg[cross_idx]]
    matched_pos_scores = pos_scores[: min(len(pos_scores), len(cross_scores))]
    matched_cross_scores = cross_scores[: len(matched_pos_scores)]
    cross_report = precision_at_recall(
        np.array(matched_pos_scores + matched_cross_scores),
        np.array([1] * len(matched_pos_scores) + [0] * len(matched_cross_scores)),
    )

    swap_audits = [
        row for row in mask_audit + neg_audit
        if row.get("target_mode") in {"swap_values", "counterfactual"}
    ]
    report = {
        "bundle": str(args.bundle),
        "model": str(args.model),
        "n_twins": len(twins),
        "twin_slice": {
            "precision_at_recall_95": twin_report["precision"],
            "threshold": twin_report["threshold"],
            "n": twin_report["n"],
        },
        "twin_margin": {
            "mean": float(margins_arr.mean()),
            "median": float(np.median(margins_arr)),
            "min": float(margins_arr.min()),
            "frac_positive": float((margins_arr > 0).mean()),
        },
        "by_flipped_field": by_field,
        "cross_brand_slice": {
            "precision_at_recall_95": cross_report["precision"],
            "n": cross_report["n"],
        },
        "subset_mean_sim": {
            "twin_0": float(np.mean([s for s, lab in zip(twin_scores, twin_labels) if lab == 0])),
            "gate_0": float(np.mean(gate_scores)) if gate_scores else float("nan"),
            "cross_brand_0": float(np.mean(cross_scores)) if cross_scores else float("nan"),
            "pos_1": float(np.mean(pos_scores)) if pos_scores else float("nan"),
        },
        "donor_uniformity": donor_uniformity(swap_audits, payload),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"twin slice P@R95={report['twin_slice']['precision_at_recall_95']:.4f} "
          f"(n={report['twin_slice']['n']})", flush=True)
    print(f"twin margin mean={report['twin_margin']['mean']:.4f} "
          f"frac>0={report['twin_margin']['frac_positive']:.3f}", flush=True)
    print(f"subset mean sim: {json.dumps(report['subset_mean_sim'], sort_keys=True)}", flush=True)
    print(f"donor uniformity: draws={report['donor_uniformity']['draws']} "
          f"donors={report['donor_uniformity']['n_donor_rows']} "
          f"max_row_share={report['donor_uniformity']['max_row_share']:.4f}", flush=True)
    for field in sorted(report["by_flipped_field"]):
        row = report["by_flipped_field"][field]
        print(f"  {field:<14} n={row['n_twins']:<4} P@R95={row['precision_at_recall_95']:.4f} "
              f"margin={row['margin_mean']:.4f}", flush=True)
    for field in sorted(report["donor_uniformity"]["by_field"]):
        row = report["donor_uniformity"]["by_field"][field]
        print(f"  donor {field:<14} values={row['unique_values']:<4} "
              f"max_value_share={row['max_value_share']:.4f}", flush=True)
    print(f"report: {args.out}", flush=True)


if __name__ == "__main__":
    main()
