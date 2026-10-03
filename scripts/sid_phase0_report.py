#!/usr/bin/env python3
"""SID Phase 0 (analysis-only): do frozen-embedding Semantic IDs separate true
pairs from volume/pack-conflict pairs where cosine collapses?

Pipeline (all SSOT, no inline knobs):
  1. SKU sample (default smoke ``data/dataset_deduped_smoke_128.csv``) +
     full ``data/canonical_records.csv``.
  2. Canonical/SKU texts via ``core.model_input`` (``build_canonical_text`` /
     ``build_sku_text``).  The ``info`` arg originates from
     ``core.attribute_conflicts`` (``canonical_attribute_info`` /
     ``sku_attribute_info``) — the same parse the gate mines conflicts from —
     adapted to the structured-text contract (see ``_text_info``).
  3. Split via ``training.folds.derive_holdout`` on the deduped pair graph
     (q0+q1 = train); RQ codebooks fit on TRAIN-side canonicals only.
  4. FROZEN zero-shot ``minilm_l6`` encode (the ``zero_shot_sims.py``
     pattern: ``max_seq_length`` / ``batch_size_embed`` from ``load_config``,
     ``normalize_embeddings=True``).
  5. Cosine (einsum, as in zero-shot) vs SID prefix overlap for true
     (same-GTIN) vs volume/pack-conflict pairs; writes ``artifacts/sid/*``.

Exits nonzero with the exact missing piece when the encoder bundle, the
input CSVs, or either pair class is absent — no fallback, no silent skip.
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
    normalized_flavor_tokens,
    sku_attribute_info,
)
from core.common import (
    F,
    SEED,
    TRAIN_ROOT,
    load_config,
    load_local_sentence_transformer,
    resolve_model,
)
from core.model_input import build_canonical_text, build_sku_text
from training.folds import derive_holdout
from training.semantic_ids import (
    add_collision_tidbits,
    assign_sids,
    codebook_usage,
    fit_rq_kmeans,
    prefix_overlap,
    save_codebooks,
    save_sid_table,
    unique_ids_proportion,
)

_SID_OUT_DIR = TRAIN_ROOT / "artifacts" / "sid"
_SMOKE_SKU_CSV = TRAIN_ROOT / "data" / "dataset_deduped_smoke_128.csv"
_DEFAULT_SAMPLE = 128
_MAX_EDGES_PER_BARCODE = 8

# GO rule (Phase-0 heuristic, printed verbatim in the report): SIDs earn a GO
# only when coarse-prefix agreement separates the classes by a wide margin on
# a non-collapsed L0 codebook.
_GO_MARGIN_L0 = 0.20
_GO_MIN_L0_USAGE = 0.10


def _fail(message: str) -> int:
    print(f"SID Phase 0 ABORT: {message}", file=sys.stderr, flush=True)
    return 2


def _text_info(conflict_info: dict[str, object]) -> dict[str, object]:
    """Adapt an attribute-conflicts parse to the structured-text contract.

    ``canonical_attribute_info`` / ``sku_attribute_info`` store ``flavor`` as
    a space-joined STRING, but the model-input text channel
    (``structured_features.text_tokens`` via ``_as_string_set``) only accepts
    sequences — a bare string raises ``ValueError``.  Normalize through the
    gate's own tokenizer (``normalized_flavor_tokens``) plus the parsed
    ``flavor_set`` so the encoder text is built from exactly the evidence the
    conflict miner sees.  All other dimensions already carry sets.
    """
    flavor = set(normalized_flavor_tokens(conflict_info.get("flavor")))
    flavor |= set(conflict_info.get("flavor_set") or set())
    info = {
        key: conflict_info.get(key, set())
        for key in (
            "volume",
            "pack",
            "package_type",
            "carbonation",
            "sweetener",
            "pulp",
        )
    }
    info["flavor"] = flavor
    return info


def _pair_graph(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Positive-pair edges between row indices sharing a gtin (singletons: none)."""
    gtins = frame["gtin"].fillna("").astype(str).to_numpy()
    edges: list[list[int]] = []
    for gtin in sorted(set(gtins)):
        if not gtin:
            continue
        rows = np.flatnonzero(gtins == gtin)[: _MAX_EDGES_PER_BARCODE + 1]
        edges.extend([ [int(rows[i]), int(rows[i + 1])] for i in range(len(rows) - 1)])
    pos = np.asarray(edges, dtype=int).reshape(-1, 2) if edges else np.zeros((0, 2), dtype=int)
    return pos, np.asarray(gtins, dtype=str)


def _load_sku_frame(sample: int, sku_csv: str | None) -> pd.DataFrame:
    if sku_csv:
        path = Path(sku_csv)
        if not path.exists():
            raise FileNotFoundError(f"--sku-csv not found: {path}")
        frame = pd.read_csv(path, dtype=str)
        return frame.head(sample).reset_index(drop=True)
    if sample == _DEFAULT_SAMPLE and _SMOKE_SKU_CSV.exists():
        return pd.read_csv(_SMOKE_SKU_CSV, dtype=str)
    path = F["dataset_deduped"]
    if not path.exists():
        raise FileNotFoundError(f"deduped dataset missing: {path}")
    return pd.read_csv(path, dtype=str).head(sample).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=_DEFAULT_SAMPLE,
                        help="SKU rows to analyze (128 = the smoke file)")
    parser.add_argument("--sku-csv", type=str, default=None,
                        help="override SKU input CSV (default: smoke file at --sample 128, else dataset head)")
    parser.add_argument("--max-pairs", type=int, default=2000,
                        help="cap per pair class (deterministic first-N)")
    parser.add_argument("--conflict-pool", type=int, default=512,
                        help="shuffled canonicals scanned per SKU for a volume/pack conflict")
    parser.add_argument("--out-dir", type=str, default=str(_SID_OUT_DIR))
    args = parser.parse_args(argv)

    cfg = load_config()
    split_cfg = cfg["split"]
    model_key = str(cfg["training"]["base_model"])
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
    if "gtin" not in canon.columns or "gtin" not in sku.columns:
        return _fail("expected 'gtin' in canonical_records.csv and 'gtin' in the SKU frame")

    # ---- split: component-aware, full-graph when available -----------------
    try:
        full_gtins = pd.read_csv(
            F["dataset_deduped"], dtype=str, usecols=["gtin"]
        )
        graph_pos, graph_bc = _pair_graph(full_gtins)
        graph_source = "dataset_deduped.csv"
    except (FileNotFoundError, ValueError):
        graph_pos, graph_bc = _pair_graph(sku)
        graph_source = "sample frame (full deduped unavailable)"
    try:
        train_bc, dev_bc, test_bc = derive_holdout(
            graph_pos, graph_bc, split_cfg, seed=int(SEED)
        )
    except ValueError as exc:
        return _fail(f"holdout split rejected by the split contract — {exc}")
    print(
        f"[split] graph={graph_source} train={len(train_bc):,} "
        f"dev={len(dev_bc):,} test={len(test_bc):,} gtins",
        flush=True,
    )

    # ---- texts (model-input SSOT; infos from the conflict parsers) ---------
    canon_records = canon.to_dict("records")
    canon_infos = [canonical_attribute_info(record) for record in canon_records]
    canon_texts = [
        build_canonical_text(record, _text_info(info))
        for record, info in zip(canon_records, canon_infos)
    ]
    canon_gtins = [str(record["gtin"]) for record in canon_records]
    gtin_to_idx = {gtin: i for i, gtin in enumerate(canon_gtins)}

    sku_infos = [
        sku_attribute_info(str(row.get("sku_name_eng", "")), str(row.get("attribute", "")))
        for _, row in sku.iterrows()
    ]
    sku_texts = [
        build_sku_text(row, _text_info(info))
        for (_, row), info in zip(sku.iterrows(), sku_infos)
    ]
    sku_gtins = sku["gtin"].fillna("").astype(str).tolist()

    # ---- frozen encode (zero_shot_sims.py pattern; no fallback) ------------
    try:
        resolve_model(model_key)
        model = load_local_sentence_transformer(model_key, device=DEVICE)
        model.max_seq_length = int(cfg["training"]["max_seq_length"])  # SSOT
        batch_size = int(cfg["training"]["batch_size_embed"])  # SSOT
        canon_emb = np.asarray(
            model.encode(
                canon_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ),
            dtype=np.float64,
        )
        sku_emb = np.asarray(
            model.encode(
                sku_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ),
            dtype=np.float64,
        )
    except (FileNotFoundError, KeyError, ValueError, OSError, RuntimeError) as exc:
        return _fail(
            f"encoder bundle for {model_key!r} missing or unloadable — {exc}. "
            "Materialize the Git-shipped bundle under artifacts/models; "
            "external model downloads are disabled, and Phase 0 has no fallback encoder."
        )

    # ---- fit on TRAIN-side canonicals only ----------------------------------
    fit_idx = [i for i, gtin in enumerate(canon_gtins) if gtin in train_bc]
    if not fit_idx:
        return _fail("no canonical GTIN falls in the train split — codebooks cannot be fit")
    codebooks = fit_rq_kmeans(canon_emb[np.asarray(fit_idx)])
    k_effective = int(codebooks.shape[1])
    if k_effective < 256:
        print(
            f"[sid] NOTE: fit pool n={len(fit_idx)} < 256 clusters → "
            f"effective k={k_effective} (smoke-sized fit; shape carries the degradation)",
            flush=True,
        )
    canon_sids = assign_sids(canon_emb, codebooks)
    sku_sids = assign_sids(sku_emb, codebooks)
    canon_full = add_collision_tidbits(canon_sids)

    # ---- pairs: true (same GTIN) vs volume/pack-conflict --------------------
    true_rows = [
        i for i, gtin in enumerate(sku_gtins)
        if gtin and gtin in gtin_to_idx
    ][: int(args.max_pairs)]
    if not true_rows:
        return _fail("no sample SKU gtin matches a canonical GTIN — zero true pairs")
    rng = np.random.default_rng(int(SEED))
    order = np.arange(len(canon_gtins))
    conflict_pairs: list[tuple[int, int]] = []
    for sku_row in true_rows:
        if len(conflict_pairs) >= int(args.max_pairs):
            break
        shuffled = rng.permutation(order)[: int(args.conflict_pool)]
        for cand in (int(c) for c in shuffled):
            if canon_gtins[cand] == sku_gtins[sku_row]:
                continue
            conflicts = attribute_conflict_types(sku_infos[sku_row], canon_infos[cand])
            if "volume" in conflicts or "pack" in conflicts:
                conflict_pairs.append((sku_row, cand))
                break
    if not conflict_pairs:
        return _fail(
            "no volume/pack-conflict pair mined — widen --conflict-pool or --max-pairs"
        )
    true_pairs = [(i, gtin_to_idx[sku_gtins[i]]) for i in true_rows]

    def _cosine(pairs: list[tuple[int, int]]) -> np.ndarray:
        a = sku_emb[np.asarray([p[0] for p in pairs])]
        b = canon_emb[np.asarray([p[1] for p in pairs])]
        return np.einsum("ij,ij->i", a, b)  # normalized → cosine (zero-shot pattern)

    def _agree(pairs: list[tuple[int, int]], depth: int) -> float:
        return float(
            np.mean(
                [bool(np.array_equal(sku_sids[i][:depth], canon_sids[j][:depth]))
                 for i, j in pairs]
            )
        )

    cos_true, cos_conf = _cosine(true_pairs), _cosine(conflict_pairs)
    pre_true = np.mean([prefix_overlap(sku_sids[i], canon_sids[j]) for i, j in true_pairs])
    pre_conf = np.mean([prefix_overlap(sku_sids[i], canon_sids[j]) for i, j in conflict_pairs])
    agree = {
        f"L{d}": (_agree(true_pairs, d + 1), _agree(conflict_pairs, d + 1))
        for d in range(3)
    }
    usage = codebook_usage(canon_sids, 256)
    uniqueness = unique_ids_proportion(canon_full)
    margin_l0 = agree["L0"][0] - agree["L0"][1]
    verdict = (
        "GO"
        if (margin_l0 >= _GO_MARGIN_L0 and usage[0] >= _GO_MIN_L0_USAGE)
        else "STOP"
    )

    # ---- artifacts -----------------------------------------------------------
    save_codebooks(out_dir / "codebooks.npz", codebooks)
    save_sid_table(out_dir / "canonical_sids.csv", canon_gtins, canon_full)
    health = {
        "n_canonicals": len(canon_gtins),
        "n_fit_train_canonicals": len(fit_idx),
        "n_levels": int(codebooks.shape[0]),
        "n_clusters_requested": 256,
        "n_clusters_effective": k_effective,
        "codebook_usage": [float(u) for u in usage],
        "unique_ids_proportion": float(uniqueness),
    }
    (out_dir / "sid_health.json").write_text(json.dumps(health, indent=2) + "\n")
    metrics = {
        "model_key": model_key,
        "device": DEVICE,
        "sku_rows": len(sku),
        "n_true_pairs": len(true_pairs),
        "n_conflict_pairs": len(conflict_pairs),
        "cosine_true_mean": float(np.mean(cos_true)),
        "cosine_true_std": float(np.std(cos_true)),
        "cosine_conflict_mean": float(np.mean(cos_conf)),
        "cosine_conflict_std": float(np.std(cos_conf)),
        "prefix_overlap_true_mean": float(pre_true),
        "prefix_overlap_conflict_mean": float(pre_conf),
        "agreement": {
            level: {"true": t, "conflict": c, "margin": t - c}
            for level, (t, c) in agree.items()
        },
        "health": health,
        "go_rule": (
            f"GO iff (true L0 − conflict L0) >= {_GO_MARGIN_L0} "
            f"and L0 usage >= {_GO_MIN_L0_USAGE}"
        ),
        "verdict": verdict,
    }
    (out_dir / "sid_phase0_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    # ---- GO/STOP table --------------------------------------------------------
    print(f"\nSID Phase 0 — frozen {model_key} + RQ-KMeans(3 x {k_effective})")
    print(f"pairs: true={len(true_pairs):,} conflict={len(conflict_pairs):,} "
          f"| fit canonicals={len(fit_idx):,} (train-side)")
    print(f"{'level':<6}{'true':>10}{'conflict':>10}{'margin':>10}")
    for level in ("L0", "L1", "L2"):
        t, c = agree[level]
        print(f"{level:<6}{t:>10.3f}{c:>10.3f}{(t - c):>10.3f}")
    print(f"cosine  true {np.mean(cos_true):.3f}±{np.std(cos_true):.3f}  "
          f"conflict {np.mean(cos_conf):.3f}±{np.std(cos_conf):.3f}")
    print(f"health  usage L0/L1/L2 "
          f"{'/'.join(f'{u:.3f}' for u in usage)}  unique_ids {uniqueness:.4f}")
    print(f"VERDICT: {verdict}  (rule: {metrics['go_rule']})")
    print(f"artifacts: {out_dir}/{{codebooks.npz,canonical_sids.csv,"
          "sid_health.json,sid_phase0_metrics.json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
