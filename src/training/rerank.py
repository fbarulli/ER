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

from core import common
from core.common import load_config, pair_auc


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
    test_bc = (
        folds_override
        if isinstance(folds_override, (set, frozenset))
        else folds_override[0]
    )
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if not ok_rows:
        print("[rerank] no trained fold to rerank — skipping", flush=True)
        return
    model_tag = args.model.rstrip("/").rsplit("/", 1)[-1]
    ckpt = common.artifact(
        "checkpoint_repo",
        {
            "model_tag": model_tag,
            "run_tag": run_tag,
            "fold": int(ok_rows[0]["fold"]),
            "step": 0,
        },
    ).parent
    if not ckpt.exists():
        # HF Trainer keeps the final/best model in the checkpoint dir root
        # only with save_only_model; else look one level up for best dir
        cands = sorted(ckpt.glob("**/model.safetensors"))
        if not cands:
            print(f"[rerank] no checkpoint at {ckpt} — skipping", flush=True)
            return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bi = common.load_local_sentence_transformer(str(ckpt), device=dev)
    # max_length SSOT: training.rerank_max_length (config/training.yaml) —
    # was an inline 512 the config could not steer (audit, owner Q27).
    from core.common import runtime as _runtime

    ce = common.load_local_cross_encoder(
        args.rerank, device=dev, max_length=int(_runtime("rerank_max_length"))
    )

    # rebuild the pair pools the same way the trainer did — now a plain
    # module import (the importlib hack existed only because "src/training/train"
    # is not a legal module name)
    import training.train as mod
    from core.hard_negatives import pairs_in_set

    d = mod.load_training_data(df, payload_variant=payload_variant)
    pos, neg = d["pos"], d["neg"]
    test_pos = pos[pairs_in_set(pos, row_bc, set(test_bc))]
    test_neg = (
        neg[pairs_in_set(neg, row_bc, set(test_bc))]
        if neg is not None and len(neg)
        else np.empty((0, 2), dtype=int)
    )
    # HOLDOUT DISCIPLINE (owner audit 2026-09-07): the A/B operating
    # threshold is picked on the DEV pairs and applied to the TEST pairs —
    # the old code chose Youden on the test scores themselves (optimistic
    # leak on both arms; the comparison stayed fair, the absolutes did not).
    # dev_bc comes from dev_override (the holdout split's dev set — the same
    # pool the trainer early-stopped on). In CV mode no dev pool is
    # available to this stage (the trainer carves it per fold from train,
    # never exported) — rather than pick a threshold on trained-on or test
    # pairs, the stage falls back to the fixed SSOT threshold and SAYS SO.
    if dev_override is not None:
        dev_bc_set = set(dev_override)
    else:
        dev_bc_set = set()
    dev_pos = (
        pos[pairs_in_set(pos, row_bc, dev_bc_set)]
        if dev_bc_set
        else np.empty((0, 2), dtype=int)
    )
    dev_neg = (
        neg[pairs_in_set(neg, row_bc, dev_bc_set)]
        if neg is not None and len(neg) and dev_bc_set
        else np.empty((0, 2), dtype=int)
    )

    def cos(a, b):
        e = bi.encode(
            [payload[a], payload[b]], convert_to_numpy=True, normalize_embeddings=True
        )
        return float(e[0] @ e[1])

    dev_pairs = [(a, b, 1) for a, b in dev_pos] + [(a, b, 0) for a, b in dev_neg]
    dev_bi_s = (
        np.array([cos(a, b) for a, b, _ in dev_pairs]) if dev_pairs else np.empty(0)
    )
    dev_y = np.array([t for _, _, t in dev_pairs]) if dev_pairs else np.empty(0)

    pairs = [(a, b, 1) for a, b in test_pos] + [(a, b, 0) for a, b in test_neg]
    bi_s = np.array([cos(a, b) for a, b, _ in pairs])
    y = np.array([t for _, _, t in pairs])
    # application band from the config SSOT (bands.rerank_band) — was a
    # second inline 0.50/0.75 declaration the config could not steer
    from core.common import band as _band

    _rb = _band("rerank_band")
    in_band = (bi_s >= _rb[0]) & (bi_s <= _rb[1])
    hyb = bi_s.copy()
    if in_band.any():
        ce_s = np.array(
            ce.predict(
                [
                    [payload[a], payload[b]]
                    for a, b, _ in [p for p, ib in zip(pairs, in_band, strict=True) if ib]
                ]
            )
        )
        hyb[in_band] = ce_s

    # dev-side hybrid scores (for the dev-picked Youden): same band rule
    dev_in_band = (
        (dev_bi_s >= _rb[0]) & (dev_bi_s <= _rb[1]) if len(dev_bi_s) else None
    )
    dev_hyb = dev_bi_s.copy() if len(dev_bi_s) else dev_bi_s
    if dev_in_band is not None and dev_in_band.any():
        dev_ce_s = np.array(
            ce.predict(
                [
                    [payload[a], payload[b]]
                    for a, b, ib in zip(dev_pairs, dev_in_band, strict=True) if ib
                ]
            )
        )
        dev_hyb[dev_in_band] = dev_ce_s

    # ── the A/B protocol (owner spec): PR-AUC primary, P/R/F1 at a
    # threshold chosen ON VALIDATION (dev), ROC-AUC secondary; the hybrid
    # must CLEARLY beat bi-only or the cross-encoder is not worth its latency.
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_fscore_support,
    )

    def _youden(scores: np.ndarray, labels: np.ndarray) -> float:
        order = np.argsort(-scores)
        tps = np.cumsum(labels[order])
        fps = np.cumsum(1 - labels[order])
        j = tps / max((labels == 1).sum(), 1) - fps / max((labels == 0).sum(), 1)
        return float(scores[order][int(np.argmax(j))])

    def report(scores: np.ndarray, dev_scores: np.ndarray) -> dict:
        pr = average_precision_score(y, scores)
        auc = (
            pair_auc(scores[y == 1], scores[y == 0]) if (y == 0).any() else float("nan")
        )
        # HOLDOUT DISCIPLINE: threshold picked on DEV, applied to TEST.
        # No dev pool (CV mode) -> the FIXED SSOT threshold, never a
        # threshold fitted on these test scores.
        if len(dev_scores) and dev_y.size and int((dev_y == 1).sum()) and int((dev_y == 0).sum()):
            thr = _youden(dev_scores, dev_y)
        else:
            # NO FALLBACK (owner Q27): split.fixed_threshold hard-indexed —
            # was .get(0.55), an inline literal the YAML could diverge from.
            thr = float(load_config()["split"]["fixed_threshold"])
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

    bi_m = report(bi_s, dev_bi_s)
    hyb_m = report(hyb, dev_hyb)
    # decision rule SSOT: rerank.min_delta_pr_auc / rerank.min_delta_f1
    # (config/training.yaml) — the quantitative "clearly improve" criterion,
    # defined BEFORE looking at test results (was inline 0.005s).
    from core.common import rerank_cfg as _rerank_cfg

    _rule = _rerank_cfg()
    d_pr = hyb_m["pr_auc"] - bi_m["pr_auc"]
    d_f1 = hyb_m["f1"] - bi_m["f1"]
    verdict = (
        "HYBRID WINS — keep the cross-encoder"
        if d_pr > _rule["min_delta_pr_auc"] or d_f1 > _rule["min_delta_f1"]
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

    # ── 07b four-population score distribution (owner ruling: emit here) ──
    # The trained bi-encoder + test pools are live in this scope; the plot
    # reads population/cosine rows. Populations per the 07b contract:
    # in_country_pos / cross_country_pos / hard_neg / random_neg.
    _emit_four_pop(bi, payload, df, row_bc, test_pos, test_neg, pos, neg)


def _emit_four_pop(bi, payload, df, row_bc, test_pos, test_neg, pos, neg) -> None:
    """07b_four_pop_scores.csv — four-population cosines (fine-tuned)."""
    import pandas as pd

    country = df["country"].fillna("").astype(str).to_numpy()
    if len(row_bc) > len(country):
        by_barcode: dict[str, list[str]] = {}
        for barcode, value in zip(
            df["barcode"].fillna("").astype(str), country, strict=True
        ):
            if barcode:
                by_barcode.setdefault(barcode, []).append(value)
        mode_country = {
            barcode: max(set(values), key=values.count)
            for barcode, values in by_barcode.items()
        }
        padded = np.asarray(
            [mode_country.get(str(barcode), "") for barcode in row_bc[len(country):]],
            dtype=object,
        )
        country = np.concatenate([country.astype(object), padded])
    if len(country) < len(payload):
        country = np.pad(
            country.astype(object),
            (0, len(payload) - len(country)),
            constant_values="",
        )

    def pop_scores(pairs, name):
        if pairs is None or not len(pairs):
            return []
        e = bi.encode(
            [payload[a] for a, _ in pairs] + [payload[b] for _, b in pairs],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        n = len(pairs)
        return (e[:n] * e[n:]).sum(axis=1)

    pos_s = pop_scores(test_pos, "pos")
    # in-country vs cross-country split of TEST positives
    a, b = test_pos[:, 0], test_pos[:, 1]
    ca, cb = country[a], country[b]
    both = (ca != "") & (cb != "")
    same = both & (ca == cb)
    cross = both & (ca != cb)
    rows = (
        [("in_country_pos", s) for s in pos_s[same]]
        + [("cross_country_pos", s) for s in pos_s[cross]]
        + [("hard_neg", s) for s in pop_scores(test_neg, "neg")]
    )
    # random negatives: barcode-known-different sample (build_pairs SSOT).
    # Caps read from the SSOT pairs block (config/training.yaml
    # pairs.max_pos_per_group / pairs.n_neg) — were hardcoded 4 / 2_000
    # inline, a second declaration the config could not steer.
    from core.blocking import build_pairs
    from core.common import SEED
    from core.common import load_config as _lc_pairs

    _pairs_cfg = _lc_pairs()["pairs"]
    _, rnd_neg = build_pairs(
        df, SEED, int(_pairs_cfg["max_pos_per_group"]), int(_pairs_cfg["n_neg"])
    )
    rows += [("random_neg", s) for s in pop_scores(rnd_neg, "random_neg")]

    out = pd.DataFrame(rows, columns=["population", "cosine"])
    common.ensure_parent(common.F["four_pop_scores"])
    out.to_csv(common.F["four_pop_scores"], index=False)
    print(
        f"[07b] four-population scores written ({len(out):,} rows) — "
        f"report_plots panel 7 is live",
        flush=True,
    )
