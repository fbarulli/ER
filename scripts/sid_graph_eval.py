#!/usr/bin/env python3
"""Graph + SID joint inference vs pointwise top-1 (spike, analysis-only).

Fixture: results/rand_truth/gtin_stratum_sweep.csv (both_equal / different /
one_missing, 2 SKUs per identity). GTIN locks apply ONLY where the fixture's
source_gtin equals the true identity (both_equal) — blind rows get no lock,
which is the whole point of the fixture.

Arms (same encoder, same SIDs, same candidate pool):
  top1      pointwise top-1 cosine @ --top1-thr, below -> unmatched
            (mirrors how the submission note produces assignments).
  graph-cos veto-constrained clustering on cosine edges only (no SID).
  graph+sid same + SID-admitted edges (L0 agree lowers the cosine bar).

Edges: SKU-SKU corroboration (>= --sku-thr, no conflict), SKU-canon top-K
(>= --cos-thr), LOCKS (+inf, both_equal only), VETO cannot-links
(volume/pack conflict or brand mismatch with evidence on both sides).

Metrics per stratum: assignment accuracy + unmatched rate. Writes
artifacts/sid/sid_graph_eval.json. Exits nonzero with the exact missing
piece instead of degrading silently.
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

from core.attribute_conflicts import (
    attribute_conflict_types,
    canonical_attribute_info,
    sku_attribute_info,
)
from core.common import F, SEED, TRAIN_ROOT, load_config, load_local_sentence_transformer, resolve_model
from core.model_input import build_canonical_text, build_sku_text
from training.folds import derive_holdout
from training.semantic_ids import assign_sids, fit_rq_kmeans
from training.sid_graph import INF_WEIGHT, assign_clusters_to_canonicals, greedy_constrained_clusters

sys.path.insert(0, str(TRAIN_ROOT / "scripts"))
from sid_phase0_report import _pair_graph, _text_info  # noqa: E402

_SID_OUT_DIR = TRAIN_ROOT / "artifacts" / "sid"


def _fail(message: str) -> int:
    print(f"SID graph eval ABORT: {message}", file=sys.stderr, flush=True)
    return 2


def _norm_brand(value: object) -> str:
    return str(value or "").strip().lower()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=str, default=str(TRAIN_ROOT / "results" / "rand_truth" / "gtin_stratum_sweep.csv"))
    parser.add_argument("--n-clusters", type=int, default=64)
    parser.add_argument("--cos-thr", type=float, default=0.50)
    parser.add_argument("--cos-lo-thr", type=float, default=0.30)
    parser.add_argument("--sku-thr", type=float, default=0.80)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top1-thr", type=float, default=0.80)
    parser.add_argument("--out-dir", type=str, default=str(_SID_OUT_DIR))
    args = parser.parse_args(argv)

    cfg = load_config()
    model_key = str(cfg["training"]["base_model"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fixture_path = Path(args.fixture)
    if not fixture_path.exists():
        return _fail(f"fixture missing: {fixture_path} (generate it first)")
    fixture = pd.read_csv(fixture_path, dtype=str, keep_default_na=False)
    for col in ("SKU_ID", "true_item_id", "source_gtin", "gtin_status"):
        if col not in fixture.columns:
            return _fail(f"fixture lacks column {col}")

    dedup_path = F["dataset_deduped"]
    dedup = pd.read_csv(dedup_path, dtype=str, keep_default_na=False)
    dedup["SKU_ID"] = dedup["product_id"].astype(str).str.strip()
    sku_rows = fixture.merge(dedup, on="SKU_ID", how="left", validate="many_to_one")
    if sku_rows["title"].isna().any():
        return _fail("fixture SKU_IDs missing from deduped dataset")
    n = len(sku_rows)

    canon = pd.read_csv(F["canonical_records"], dtype=str, keep_default_na=False)
    # Canonical frame carries the brand under mode_brand (deduped SKUs: brand).
    if "mode_brand" not in canon.columns or "brand" not in sku_rows.columns:
        return _fail("need 'mode_brand' on canonical frame and 'brand' on SKU frame")
    canon_records = canon.to_dict("records")
    canon_infos = [canonical_attribute_info(r) for r in canon_records]
    canon_texts = [build_canonical_text(r, _text_info(i)) for r, i in zip(canon_records, canon_infos)]
    canon_gtins = [str(r["gtin"]) for r in canon_records]
    canon_brands = [_norm_brand(r.get("mode_brand") or r.get("brand")) for r in canon_records]
    gtin_to_idx = {g: i for i, g in enumerate(canon_gtins)}
    m = len(canon_gtins)

    sku_infos = [sku_attribute_info(str(r.get("title", "")), str(r.get("attributes", "")))
                 for _, r in sku_rows.iterrows()]
    sku_texts = [build_sku_text(row, _text_info(info)) for (_, row), info in zip(sku_rows.iterrows(), sku_infos)]
    sku_brands = [_norm_brand(r.get("brand")) for _, r in sku_rows.iterrows()]
    true_idx = np.asarray([gtin_to_idx[str(t)] for t in sku_rows["true_item_id"]])
    statuses = sku_rows["gtin_status"].tolist()

    try:
        resolve_model(model_key)
        model = load_local_sentence_transformer(model_key, device=DEVICE)
        model.max_seq_length = int(cfg["training"]["max_seq_length"])  # SSOT
        batch_size = int(cfg["training"]["batch_size_embed"])  # SSOT
        canon_emb = np.asarray(model.encode(
            canon_texts, batch_size=batch_size, show_progress_bar=False,
            normalize_embeddings=True, convert_to_numpy=True), dtype=np.float64)
        sku_emb = np.asarray(model.encode(
            sku_texts, batch_size=batch_size, show_progress_bar=False,
            normalize_embeddings=True, convert_to_numpy=True), dtype=np.float64)
    except (FileNotFoundError, KeyError, ValueError, OSError, RuntimeError) as exc:
        return _fail(f"encoder bundle for {model_key!r} missing or unloadable — {exc}")

    # codebooks fit on train-side canonicals only (same split contract)
    full_barcodes = pd.read_csv(F["dataset_deduped"], dtype=str, usecols=["barcode"])
    graph_pos, graph_bc = _pair_graph(full_barcodes)
    split_cfg = cfg["split"]
    train_bc, _, _ = derive_holdout(graph_pos, graph_bc, split_cfg, seed=int(SEED))
    train_bc = set(train_bc)
    fit_idx = [i for i, g in enumerate(canon_gtins) if g in train_bc] or list(range(m))
    codebooks = fit_rq_kmeans(canon_emb[np.asarray(fit_idx)], n_clusters=int(args.n_clusters))
    canon_sids = assign_sids(canon_emb, codebooks)
    sku_sids = assign_sids(sku_emb, codebooks)

    cos_sc = sku_emb @ canon_emb.T
    cos_ss = sku_emb @ sku_emb.T
    topk = np.argsort(-cos_sc, axis=1)[:, : int(args.top_k)]

    def _conflict(a_info: dict, b_info: dict) -> bool:
        hits = attribute_conflict_types(a_info, b_info)
        return "volume" in hits or "pack" in hits

    def _brand_clash(ba: str, bb: str) -> bool:
        return bool(ba and bb and ba != bb)

    # ---- shared edge sets -------------------------------------------------
    locks: dict[int, int] = {}
    for pos in range(n):
        row = sku_rows.iloc[pos]
        if str(row["gtin_status"]) == "both_equal" and str(row["source_gtin"]) == str(row["true_item_id"]):
            locks[pos] = int(gtin_to_idx[str(row["true_item_id"])])

    veto: set[tuple[int, int]] = set()
    for i in range(n):
        for j in topk[i].tolist():
            if _conflict(sku_infos[i], canon_infos[j]) or _brand_clash(sku_brands[i], canon_brands[j]):
                veto.add((i, n + int(j)))
        for k in range(i + 1, n):
            if cos_ss[i, k] >= float(args.sku_thr) and (
                    _conflict(sku_infos[i], sku_infos[k]) or _brand_clash(sku_brands[i], sku_brands[k])):
                veto.add((i, k))

    def _edges(use_sid: bool) -> list[tuple[int, int, float]]:
        edges: list[tuple[int, int, float]] = []
        for i, j in locks.items():
            edges.append((i, n + j, INF_WEIGHT))
        for i in range(n):
            for j in topk[i].tolist():
                c = float(cos_sc[i, j])
                l0 = bool(sku_sids[i][0] == canon_sids[j][0])
                if c >= float(args.cos_thr):
                    edges.append((i, n + int(j), c))
                elif use_sid and l0 and c >= float(args.cos_lo_thr):
                    edges.append((i, n + int(j), c))
        for i in range(n):
            for k in range(i + 1, n):
                c = float(cos_ss[i, k])
                l0 = bool(sku_sids[i][0] == sku_sids[k][0])
                if c >= float(args.sku_thr):
                    edges.append((i, k, c))
                elif use_sid and l0 and c >= float(args.sku_thr) - 0.10:
                    edges.append((i, k, c))
        return edges

    w_sc = np.zeros((n, m))
    for i in range(n):
        w_sc[i, topk[i]] = cos_sc[i, topk[i]]

    results: dict[str, dict] = {}

    # arm 1: pointwise top-1 (submission-style: exact-GTIN lock bypasses the
    # threshold entirely, everything else needs top-1 cosine >= --top1-thr).
    best = topk[np.arange(n), 0]
    best_cos = cos_sc[np.arange(n), best]
    top1_pred = [locks[i] if i in locks
                 else (int(b) if c >= float(args.top1_thr) else None)
                 for i, (b, c) in enumerate(zip(best.tolist(), best_cos.tolist()))]
    results["top1"] = {"pred": top1_pred}

    # arms 2-3: constrained clustering, cosine-only vs +SID
    for arm, use_sid in (("graph-cos", False), ("graph+sid", True)):
        labels = greedy_constrained_clusters(n + m, _edges(use_sid), veto)
        sku_lab, canon_lab = labels[:n], labels[n:]
        pred_map = assign_clusters_to_canonicals(sku_lab, canon_lab, w_sc, locks)
        results[arm] = {"pred": [pred_map[i] for i in range(n)]}

    # ---- per-stratum accuracy ----------------------------------------------
    table: dict[str, dict[str, float | int]] = {}
    for arm, res in results.items():
        pred = res["pred"]
        row: dict[str, float | int] = {"n": n}
        for status in ("both_equal", "different", "one_missing"):
            idx = [i for i, s in enumerate(statuses) if s == status]
            hits = sum(1 for i in idx if pred[i] is not None and int(pred[i]) == int(true_idx[i]))
            unm = sum(1 for i in idx if pred[i] is None)
            row[f"{status}/acc"] = hits / len(idx) if idx else float("nan")
            row[f"{status}/unmatched"] = unm / len(idx) if idx else float("nan")
        row["overall/acc"] = sum(
            1 for i in range(n) if pred[i] is not None and int(pred[i]) == int(true_idx[i])) / n
        table[arm] = row

    metrics = {
        "model_key": model_key, "device": DEVICE, "n_clusters": int(args.n_clusters),
        "fixture": str(fixture_path), "n_sku": n, "n_canon": m,
        "n_pos_edges_cos": len(_edges(False)), "n_pos_edges_sid": len(_edges(True)),
        "n_veto": len(veto), "n_locks": len(locks),
        "arms": table,
    }
    (out_dir / "sid_graph_eval.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nSID graph A/B — frozen {model_key} (k={args.n_clusters}, fixture n={n})")
    print(f"edges: cos-only={len(_edges(False)):,}  +sid={len(_edges(True)):,}  veto={len(veto):,}  locks={len(locks):,}")
    header = f"{'arm':<10}{'overall':>9}" + "".join(f"{s[:9]:>18}" for s in ("both_equal", "different", "one_missing"))
    print(header)
    for arm, row in table.items():
        cells = "".join(f"{row[f'{s}/acc']:>9.3f}(u{row[f'{s}/unmatched']:.2f})" for s in ("both_equal", "different", "one_missing"))
        print(f"{arm:<10}{row['overall/acc']:>9.3f}{cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
