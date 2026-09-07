"""rerank.py — two-stage rerank :
cross-encoder re-scores the bi-encoder's confusion band post-training).

First, a fast model (bi‑encoder) looks at a pair of product descriptions and gives a similarity score between 0 and 1.

High score = probably the same product.

Low score = probably different.

But there’s a middle zone (0.50–0.75) where the fast model is unsure.

Second, a slower but smarter model (cross‑encoder) re‑checks only the pairs in that middle zone.

It reads both descriptions together, so it can catch subtle differences (like “sugar free” vs “no sugar”) that the fast model missed.

For pairs outside that zone, we trust the fast model’s score.

"""

from __future__ import annotations

import numpy as np

from lib.common import RESULTS, pair_auc


def rerank_stage(
    args,
    rows,
    run_tag,
    row_bc,
    payload,
    df,
    dev_override,
    folds_override,
    seed,
    payload_variant="full",
) -> None:
    """07e mirror: two-stage rerank on the fine-tuned model's confusion band.

    Stage 1 (bi-encoder): the best per-fold checkpoint scores test positives
    and negatives. Stage 2 (cross-encoder): only pairs in the application
    band (0.50-0.75 cosine, 07e's precision-spend slice) are re-scored;
    outside it the bi-encoder score stands. Reports bi-only vs hybrid AUC.
    """
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    test_bc = (
        folds_override
        if isinstance(folds_override, (set, frozenset))
        else folds_override[0]
    )
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if not ok_rows:
        print("[rerank] no trained fold to rerank — skipping", flush=True)
        return
    ckpt = RESULTS / f"_checkpoints/r{run_tag}_f{ok_rows[0]['fold']}"
    if not ckpt.exists():
        # HF Trainer keeps the final/best model in the checkpoint dir root
        # only with save_only_model; else look one level up for best dir
        cands = sorted(ckpt.glob("**/model.safetensors"))
        if not cands:
            print(f"[rerank] no checkpoint at {ckpt} — skipping", flush=True)
            return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bi = SentenceTransformer(str(ckpt), device=dev)
    ce = CrossEncoder(args.rerank, device=dev, max_length=512)

    # rebuild the pair pools the same way the trainer did
    from importlib import util
    from pathlib import Path

    spec = util.spec_from_file_location(
        "train_entry", Path(__file__).resolve().parent / "05_train.py"
    )
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from lib.hard_negatives import pairs_in_set

    d = mod.load_training_data(df, payload_variant=payload_variant)
    pos, neg = d["pos"], d["neg"]
    test_pos = pos[pairs_in_set(pos, row_bc, set(test_bc))]
    test_neg = (
        neg[pairs_in_set(neg, row_bc, set(test_bc))]
        if neg is not None and len(neg)
        else np.empty((0, 2), dtype=int)
    )

    def cos(a, b):
        e = bi.encode(
            [payload[a], payload[b]], convert_to_numpy=True, normalize_embeddings=True
        )
        return float(e[0] @ e[1])

    pairs = [(a, b, 1) for a, b in test_pos] + [(a, b, 0) for a, b in test_neg]
    bi_s = np.array([cos(a, b) for a, b, _ in pairs])
    y = np.array([t for _, _, t in pairs])
    in_band = (bi_s >= 0.50) & (bi_s <= 0.75)
    hyb = bi_s.copy()
    if in_band.any():
        ce_s = np.array(
            ce.predict(
                [
                    [payload[a], payload[b]]
                    for a, b, _ in [p for p, ib in zip(pairs, in_band) if ib]
                ]
            )
        )
        hyb[in_band] = ce_s

    # ── the A/B protocol (owner spec): PR-AUC primary, P/R/F1 at a
    # threshold chosen ON VALIDATION, ROC-AUC secondary; the hybrid must
    # CLEARLY beat bi-only or the cross-encoder is not worth its latency.
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_fscore_support,
    )

    def report(scores: np.ndarray) -> dict:
        pr = average_precision_score(y, scores)
        auc = (
            pair_auc(scores[y == 1], scores[y == 0]) if (y == 0).any() else float("nan")
        )
        # Youden threshold — chosen on this validation set, per protocol
        order = np.argsort(-scores)
        tps = np.cumsum(y[order])
        fps = np.cumsum(1 - y[order])
        j = tps / max((y == 1).sum(), 1) - fps / max((y == 0).sum(), 1)
        thr = float(scores[order][int(np.argmax(j))])
        pred = (scores >= thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y, pred, average="binary", zero_division=0
        )
        return {
            "pr_auc": pr,
            "roc_auc": auc,
            "thr": thr,
            "precision": p,
            "recall": r,
            "f1": f1,
        }

    bi_m = report(bi_s)
    hyb_m = report(hyb)
    d_pr = hyb_m["pr_auc"] - bi_m["pr_auc"]
    d_f1 = hyb_m["f1"] - bi_m["f1"]
    verdict = (
        "HYBRID WINS — keep the cross-encoder"
        if d_pr > 0.005 or d_f1 > 0.005
        else "NO CLEAR WIN — drop the cross-encoder (latency not justified)"
    )
    print(
        f"[07e rerank] band pairs re-scored: {int(in_band.sum()):,} of {len(pairs):,}",
        flush=True,
    )
    print(
        f"[07e rerank] bi:     PR-AUC {bi_m['pr_auc']:.4f}  "
        f"F1@{bi_m['thr']:.2f} {bi_m['f1']:.4f} "
        f"(P {bi_m['precision']:.4f} / R {bi_m['recall']:.4f})  "
        f"ROC-AUC {bi_m['roc_auc']:.4f}",
        flush=True,
    )
    print(
        f"[07e rerank] hybrid: PR-AUC {hyb_m['pr_auc']:.4f}  "
        f"F1@{hyb_m['thr']:.2f} {hyb_m['f1']:.4f} "
        f"(P {hyb_m['precision']:.4f} / R {hyb_m['recall']:.4f})  "
        f"ROC-AUC {hyb_m['roc_auc']:.4f}",
        flush=True,
    )
    print(
        f"[07e rerank] Δ PR-AUC {d_pr:+.4f} | Δ F1 {d_f1:+.4f} → {verdict}",
        flush=True,
    )
