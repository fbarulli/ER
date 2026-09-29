#!/usr/bin/env python3
"""SID Phase 1 (analysis-only): does hybrid = alpha*cosine + beta*overlap beat
bi-encoder cosine alone on FROZEN embeddings?

A/B protocol (mirrors the 07e lane in src/training/rerank.py):
  * pairs: true (same GTIN) = 1 vs volume/pack-conflict + random-easy = 0.
  * threshold picked by Youden ON DEV pairs, applied to TEST pairs (holdout
    discipline — never fitted on the scores it rates).
  * retrieval: per-SKU ranking over ALL canonicals, recall@K of the true GTIN.
  * verdict: hybrid wins iff it clears the SSOT 07e margins
    (rerank.min_delta_pr_auc / rerank.min_delta_f1 from load_config).

Inputs/encoders/splits follow scripts/sid_phase0_report.py exactly (frozen
minilm_l6, train-side-only codebooks). No lane/config/schema changes; alpha
and beta are CLI args because no `sid:` config block exists yet by design.

 honest-scope note (printed with the verdict): frozen embeddings are the
 HEALTHY regime — this A/B measures the fusion's value-add, not its ability
 to rescue a collapsed fine-tuned checkpoint. That second verdict needs a
 Colab-trained checkpoint and is explicitly out of scope here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from core.attribute_conflicts import (
    attribute_conflict_types,
    canonical_attribute_info,
    sku_attribute_info,
)
from core.common import F, SEED, TRAIN_ROOT, load_config, load_local_sentence_transformer, resolve_model
from core.model_input import build_canonical_text, build_sku_text
from training.folds import derive_holdout
from training.semantic_ids import (
    add_collision_tidbits,
    assign_sids,
    fit_rq_kmeans,
)
from training.sid_hybrid import hybrid_matrix, veto_matrix

sys.path.insert(0, str(TRAIN_ROOT / "scripts"))
from sid_phase0_report import _pair_graph, _text_info  # noqa: E402

_SID_OUT_DIR = TRAIN_ROOT / "artifacts" / "sid"
_SMOKE_FILES = {
    128: TRAIN_ROOT / "data" / "dataset_deduped_smoke_128.csv",
    1000: TRAIN_ROOT / "data" / "dataset_deduped_smoke_1000.csv",
}
_DEFAULT_SAMPLE = 1000
_MAX_EDGES_PER_BARCODE = 8


def _fail(message: str) -> int:
    print(f"SID hybrid eval ABORT: {message}", file=sys.stderr, flush=True)
    return 2


def _load_sku_frame(sample: int, sku_csv: str | None) -> pd.DataFrame:
    if sku_csv:
        path = Path(sku_csv)
        if not path.exists():
            raise FileNotFoundError(f"--sku-csv not found: {path}")
        return pd.read_csv(path, dtype=str).head(sample).reset_index(drop=True)
    hit = _SMOKE_FILES.get(sample)
    if hit is not None and hit.exists():
        return pd.read_csv(hit, dtype=str)
    path = F["dataset_deduped"]
    if not path.exists():
        raise FileNotFoundError(f"deduped dataset missing: {path}")
    return pd.read_csv(path, dtype=str).head(sample).reset_index(drop=True)


def _youden(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores)
    tps = np.cumsum(labels[order] == 1)
    fps = np.cumsum(labels[order] == 0)
    youden_j = tps / max(int((labels == 1).sum()), 1) - fps / max(int((labels == 0).sum()), 1)
    return float(scores[order][int(np.argmax(youden_j))])


def _pair_report(
    name: str,
    scores: np.ndarray,
    labels: np.ndarray,
    dev_scores: np.ndarray,
    dev_labels: np.ndarray,
) -> dict:
    pr_auc = float(average_precision_score(labels, scores))
    roc_auc = float(roc_auc_score(labels, scores))
    thr = _youden(dev_scores, dev_labels)
    pred = (scores >= thr).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, pred, average="binary", zero_division=0
    )
    return {
        "arm": name,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "thr_dev_youden": thr,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n_test": int(len(labels)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=_DEFAULT_SAMPLE)
    parser.add_argument("--sku-csv", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--n-clusters", type=int, default=256,
                        help="codebook size per level (sweep coarse granularity)")
    parser.add_argument("--tag", type=str, default=None,
                        help="suffix for the output JSON (sweeps must not overwrite each other)")
    parser.add_argument("--gammas", type=str, default="0.05,0.10,0.20",
                        help="veto penalties swept in one run (one encode)")
    parser.add_argument("--max-pairs", type=int, default=2000)
    parser.add_argument("--conflict-pool", type=int, default=512)
    parser.add_argument("--easy-per-true", type=int, default=2)
    parser.add_argument("--recall-ks", type=str, default="1,5,20")
    parser.add_argument("--out-dir", type=str, default=str(_SID_OUT_DIR))
    args = parser.parse_args(argv)
    recall_ks = [int(k) for k in str(args.recall_ks).split(",") if k.strip()]
    gammas = [float(g) for g in str(args.gammas).split(",") if g.strip()]
    if any(g < 0.0 for g in gammas):
        return _fail(f"--gammas must be non-negative, got {args.gammas}")

    cfg = load_config()
    split_cfg = cfg["split"]
    model_key = str(cfg["training"]["base_model"])
    rule = cfg["rerank"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        sku = _load_sku_frame(int(args.sample), args.sku_csv)
    except FileNotFoundError as exc:
        return _fail(f"SKU input missing — {exc}")
    canon_path = F["canonical_records"]
    if not canon_path.exists():
        return _fail(f"canonical records missing: {canon_path}")
    canon = pd.read_csv(canon_path, dtype=str, keep_default_na=False)

    try:
        full_barcodes = pd.read_csv(F["dataset_deduped"], dtype=str, usecols=["barcode"])
        graph_pos, graph_bc = _pair_graph(full_barcodes)
    except (FileNotFoundError, ValueError):
        graph_pos, graph_bc = _pair_graph(sku)
    try:
        train_bc, dev_bc, test_bc = derive_holdout(
            graph_pos, graph_bc, split_cfg, seed=int(SEED)
        )
    except ValueError as exc:
        return _fail(f"holdout split rejected by the split contract — {exc}")
    dev_bc, test_bc = set(dev_bc), set(test_bc)

    canon_records = canon.to_dict("records")
    canon_infos = [canonical_attribute_info(r) for r in canon_records]
    canon_texts = [build_canonical_text(r, _text_info(i)) for r, i in zip(canon_records, canon_infos)]
    canon_gtins = [str(r["gtin"]) for r in canon_records]
    gtin_to_idx = {g: i for i, g in enumerate(canon_gtins)}

    sku_infos = [sku_attribute_info(str(r.get("title", "")), str(r.get("attributes", ""))) for _, r in sku.iterrows()]
    sku_texts = [build_sku_text(row, _text_info(info)) for (_, row), info in zip(sku.iterrows(), sku_infos)]
    sku_barcodes = sku["barcode"].fillna("").astype(str).tolist()

    try:
        resolve_model(model_key)
        model = load_local_sentence_transformer(model_key, device=DEVICE)
        model.max_seq_length = int(cfg["training"]["max_seq_length"])  # SSOT
        batch_size = int(cfg["training"]["batch_size_embed"])  # SSOT
        canon_emb = np.asarray(
            model.encode(canon_texts, batch_size=batch_size, show_progress_bar=False,
                         normalize_embeddings=True, convert_to_numpy=True), dtype=np.float64)
        sku_emb = np.asarray(
            model.encode(sku_texts, batch_size=batch_size, show_progress_bar=False,
                         normalize_embeddings=True, convert_to_numpy=True), dtype=np.float64)
    except (FileNotFoundError, KeyError, ValueError, OSError, RuntimeError) as exc:
        return _fail(f"encoder bundle for {model_key!r} missing or unloadable — {exc}")

    fit_idx = [i for i, g in enumerate(canon_gtins) if g in set(
        b for b in graph_bc[: len(graph_bc)] if b in train_bc)]
    if not fit_idx:
        fit_idx = list(range(len(canon_gtins)))
        print("[sid] WARN: no train-split tag on canonicals — fit on full catalog", flush=True)
    codebooks = fit_rq_kmeans(canon_emb[np.asarray(fit_idx)],
                             n_clusters=int(args.n_clusters))
    canon_sids = assign_sids(canon_emb, codebooks)
    sku_sids = assign_sids(sku_emb, codebooks)
    _ = add_collision_tidbits(canon_sids)  # uniqueness guard (unused downstream)

    rng = np.random.default_rng(int(SEED))
    order = np.arange(len(canon_gtins))

    def _build_split(rows: list[int]) -> tuple[list, list, list]:
        true = [(i, gtin_to_idx[sku_barcodes[i]]) for i in rows
                if sku_barcodes[i] and sku_barcodes[i] in gtin_to_idx][: int(args.max_pairs)]
        conflicts: list[tuple[int, int]] = []
        for sku_row in [t[0] for t in true]:
            if len(conflicts) >= int(args.max_pairs):
                break
            for cand in (int(c) for c in rng.permutation(order)[: int(args.conflict_pool)]):
                if canon_gtins[cand] == sku_barcodes[sku_row]:
                    continue
                hits = attribute_conflict_types(sku_infos[sku_row], canon_infos[cand])
                if "volume" in hits or "pack" in hits:
                    conflicts.append((sku_row, cand))
                    break
        easy: list[tuple[int, int]] = []
        for sku_row, _ in true:
            others = rng.permutation(order)
            for cand in (int(c) for c in others):
                if canon_gtins[int(cand)] == sku_barcodes[sku_row]:
                    continue
                easy.append((sku_row, int(cand)))
                if len([e for e in easy if e[0] == sku_row]) >= int(args.easy_per_true):
                    break
        return true, conflicts, easy

    dev_rows = [i for i, b in enumerate(sku_barcodes) if b in dev_bc]
    test_rows = [i for i, b in enumerate(sku_barcodes) if b in test_bc]
    if not dev_rows or not test_rows:
        return _fail(f"dev/test SKU rows empty (dev={len(dev_rows)} test={len(test_rows)}) — sample too small for split")
    dev_true, dev_conf, dev_easy = _build_split(dev_rows)
    test_true, test_conf, test_easy = _build_split(test_rows)
    if not dev_true or not test_true or not test_conf or not dev_conf:
        return _fail("a pair class is empty in dev or test — widen --max-pairs/--conflict-pool")

    def _cos(pairs: list[tuple[int, int]]) -> np.ndarray:
        a = sku_emb[np.asarray([p[0] for p in pairs])]
        b = canon_emb[np.asarray([p[1] for p in pairs])]
        return np.einsum("ij,ij->i", a, b)

    def _hyb(pairs: list[tuple[int, int]]) -> np.ndarray:
        a = sku_sids[np.asarray([p[0] for p in pairs])]
        b = canon_sids[np.asarray([p[1] for p in pairs])]
        eq = a == b
        overlap = np.logical_and.accumulate(eq, axis=1).mean(axis=1)
        return float(args.alpha) * _cos(pairs) + float(args.beta) * overlap

    def _veto(pairs: list[tuple[int, int]], gamma: float) -> np.ndarray:
        ai = np.asarray([p[0] for p in pairs])
        bi = np.asarray([p[1] for p in pairs])
        mismatch = (sku_sids[ai, 0] != canon_sids[bi, 0]).astype(np.float64)
        return _cos(pairs) - float(gamma) * mismatch

    dev_pairs = dev_true + dev_conf + dev_easy
    test_pairs = test_true + test_conf + test_easy
    dev_labels = np.asarray([1] * len(dev_true) + [0] * (len(dev_conf) + len(dev_easy)))
    test_labels = np.asarray([1] * len(test_true) + [0] * (len(test_conf) + len(test_easy)))

    def _agree(pairs: list[tuple[int, int]], depth: int) -> float:
        if not pairs:
            return float("nan")
        return float(np.mean(
            [bool(np.array_equal(sku_sids[i][:depth], canon_sids[j][:depth]))
             for i, j in pairs]))
    agreement = {
        f"L{d}": {"true": _agree(test_true, d + 1), "conflict": _agree(test_conf, d + 1)}
        for d in range(int(codebooks.shape[0]))}
    dev_bi, test_bi = _cos(dev_pairs), _cos(test_pairs)
    dev_hy, test_hy = _hyb(dev_pairs), _hyb(test_pairs)

    bi_m = _pair_report("bi", test_bi, test_labels, dev_bi, dev_labels)
    hy_m = _pair_report(f"hybrid(a={args.alpha},b={args.beta})", test_hy, test_labels, dev_hy, dev_labels)
    veto_ms: list[dict] = []
    veto_pair_mats: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for gamma in gammas:
        dev_v, test_v = _veto(dev_pairs, gamma), _veto(test_pairs, gamma)
        veto_pair_mats[gamma] = (dev_v, test_v)
        veto_ms.append(_pair_report(f"veto(g={gamma})", test_v, test_labels, dev_v, dev_labels))

    # retrieval: rank ALL canonicals per test-true SKU
    t_idx = np.asarray([p[0] for p in test_true])
    c_idx = np.asarray([p[1] for p in test_true])
    cos_mat = sku_emb[t_idx] @ canon_emb.T
    hyb_mat = hybrid_matrix(cos_mat, sku_sids[t_idx], canon_sids,
                            alpha=float(args.alpha), beta=float(args.beta))
    recall: dict[str, dict[str, float]] = {}
    mats: dict[str, np.ndarray] = {"bi": cos_mat, "hybrid": hyb_mat}
    for gamma in gammas:
        mats[f"veto(g={gamma})"] = veto_matrix(
            cos_mat, sku_sids[t_idx], canon_sids, gamma=gamma)
    for name, mat in mats.items():
        ranks = np.argsort(-mat, axis=1)
        hits = (ranks == c_idx[:, None])
        positions = np.argmax(hits, axis=1) + 1
        recall[name] = {f"recall@{k}": float(np.mean(positions <= k)) for k in recall_ks}

    arms = [bi_m, hy_m, *veto_ms]
    results = []
    for m in arms:
        d_pr = m["pr_auc"] - bi_m["pr_auc"]
        d_f1 = m["f1"] - bi_m["f1"]
        wins = (m is not bi_m) and (
            d_pr > float(rule["min_delta_pr_auc"]) or d_f1 > float(rule["min_delta_f1"]))
        results.append({"arm": m["arm"], "d_pr_auc": d_pr, "d_f1": d_f1,
                        "wins": bool(wins)})
    winners = [r for r in results if r["wins"]]
    verdict = ("VETO WINS: " + ", ".join(r["arm"] for r in winners)
               if winners else "NO CLEAR WIN")

    metrics = {
        "model_key": model_key, "device": DEVICE, "alpha": float(args.alpha), "beta": float(args.beta),
        "n_clusters": int(args.n_clusters),
        "gammas": gammas,
        "n_dev_pairs": len(dev_pairs), "n_test_pairs": len(test_pairs),
        "n_test_true": len(test_true), "n_test_conflict": len(test_conf), "n_test_easy": len(test_easy),
        "arms": arms, "recall": recall, "comparisons": results,
        "agreement": agreement,
        "rule": {"min_delta_pr_auc": float(rule["min_delta_pr_auc"]), "min_delta_f1": float(rule["min_delta_f1"])},
        "verdict": verdict,
        "scope": "frozen-embedding A/B only; NOT a verdict on rescuing collapsed fine-tuned checkpoints",
    }
    out_name = f"sid_hybrid_eval_{args.tag}.json" if args.tag else "sid_hybrid_eval.json"
    (out_dir / out_name).write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"\nSID hybrid A/B — frozen {model_key} "
          f"(k={int(args.n_clusters)} alpha={args.alpha} beta={args.beta} gammas={gammas})")
    print(f"pairs: dev={len(dev_pairs):,} (true={len(dev_true):,}) "
          f"test={len(test_pairs):,} (true={len(test_true):,} conf={len(test_conf):,} easy={len(test_easy):,})")
    for m in arms:
        print(f"{m['arm']:<22} PR-AUC {m['pr_auc']:.4f} ROC-AUC {m['roc_auc']:.4f} "
              f"F1@{m['thr_dev_youden']:.2f} {m['f1']:.4f} (P {m['precision']:.4f}/R {m['recall']:.4f})")
    print("agree " + "  ".join(
        f"{lv}:T{t['true']:.3f}/C{t['conflict']:.3f}" for lv, t in agreement.items()))
    for k in recall_ks:
        row = "  ".join(f"{n} {recall[n][f'recall@{k}']:.4f}" for n in mats)
        print(f"recall@{k:<3} {row}")
    for r in results[1:]:
        print(f"{r['arm']:<22} dPR {r['d_pr_auc']:+.4f} (need >{rule['min_delta_pr_auc']})  "
              f"dF1 {r['d_f1']:+.4f} (need >{rule['min_delta_f1']})  {'WIN' if r['wins'] else '---'}")
    print(f"VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
