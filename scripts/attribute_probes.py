#!/usr/bin/env python3
"""Attribute specialist probes: can a SMALL model generalize each attribute?

Your original idea (one model per attribute) starts with the cheapest
possible specialist: a linear probe on FROZEN embeddings. For each attribute
dimension we train logistic regression on |e_anchor - e_other| to predict
agree/disagree, on held-out identities:

  high AUC -> the attribute is already linearly decodable; a specialist head
              is trivially cheap AND the monolith has no representation
              excuse for missing it (its failure is training, not data).
  low AUC  -> the attribute needs representation learning (fine-tuned or
              dedicated encoder) to become separable; specialists need GPU.

Plus a monolith probe (same/different product) as the reference point.
Writes results/attribute_probes.json. Fail-loud on missing bundle/inputs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from core.attribute_conflicts import (
    attribute_conflict_types,
    canonical_attribute_info,
    sku_attribute_info,
)
from core.common import F, SEED, TRAIN_ROOT, load_config, load_local_sentence_transformer, resolve_model
from core.model_input import build_canonical_text, build_sku_text

sys.path.insert(0, str(TRAIN_ROOT / "scripts"))
from sid_phase0_report import _text_info  # noqa: E402

ATTRS = ("volume", "pack", "package_type", "flavor", "carbonation", "sweetener", "pulp")


def _fail(message: str) -> int:
    print(f"attribute probes ABORT: {message}", file=sys.stderr, flush=True)
    return 2


def _agree_on(info_a: dict, info_b: dict, attr: str) -> bool | None:
    """True/False when both sides carry the attribute, else None (unknown)."""
    if attr == "flavor":
        from core.attribute_conflicts import normalized_flavor_tokens
        ta = normalized_flavor_tokens(info_a.get("flavor")) | set(info_a.get("flavor_set") or set())
        tb = normalized_flavor_tokens(info_b.get("flavor")) | set(info_b.get("flavor_set") or set())
    else:
        ta, tb = set(info_a.get(attr) or set()), set(info_b.get(attr) or set())
    if not ta or not tb:
        return None
    return bool(ta & tb)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=128)
    parser.add_argument("--encoder", type=str, default=None)
    parser.add_argument("--max-pairs", type=int, default=400)
    parser.add_argument("--conflict-pool", type=int, default=512)
    parser.add_argument("--out", type=str, default=str(TRAIN_ROOT / "results" / "attribute_probes.json"))
    args = parser.parse_args(argv)

    cfg = load_config()
    model_key = str(args.encoder or cfg["training"]["base_model"])
    rng = random.Random(int(SEED))

    smoke = TRAIN_ROOT / "data" / "dataset_deduped_smoke_128.csv"
    if int(args.sample) == 128 and smoke.exists():
        sku = pd.read_csv(smoke, dtype=str)
    else:
        path = F["dataset_deduped"]
        if not path.exists():
            return _fail(f"deduped dataset missing: {path}")
        sku = pd.read_csv(path, dtype=str).head(int(args.sample)).reset_index(drop=True)
    canon = pd.read_csv(F["canonical_records"], dtype=str, keep_default_na=False)
    canon_records = canon.to_dict("records")
    canon_infos = [canonical_attribute_info(r) for r in canon_records]
    canon_texts = [build_canonical_text(r, _text_info(i)) for r, i in zip(canon_records, canon_infos)]
    canon_gtins = [str(r["gtin"]) for r in canon_records]
    gtin_to_idx = {g: i for i, g in enumerate(canon_gtins)}

    sku_infos = [sku_attribute_info(str(r.get("title", "")), str(r.get("attributes", "")))
                 for _, r in sku.iterrows()]
    sku_texts = [build_sku_text(row, _text_info(info)) for (_, row), info in zip(sku.iterrows(), sku_infos)]
    sku_barcodes = sku["barcode"].fillna("").astype(str).tolist()
    sku_identity = [b if b and b in gtin_to_idx else None for b in sku_barcodes]

    # split identities: even hash -> test (no pair straddles the split)
    test_ids = {b for b in set(sku_barcodes) if b and (hash((int(SEED), b)) % 2 == 0)}

    # collect (anchor_text_idx, other_text_idx, identity_a, identity_b, infos...) per class
    order = np.arange(len(canon_gtins))
    attr_pairs: dict[str, list[tuple[int, int, int]]] = {a: [] for a in ATTRS}  # (sku_row, canon_idx, label)
    mono_pairs: list[tuple[int, int, int]] = []
    for i, b in enumerate(sku_barcodes):
        if not b or b not in gtin_to_idx:
            continue
        j = gtin_to_idx[b]
        mono_pairs.append((i, j, 1))
        for a in ATTRS:
            v = _agree_on(
                {"volume": sku_infos[i].get("volume"), "pack": sku_infos[i].get("pack"),
                 "package_type": sku_infos[i].get("package_type"), "flavor": sku_infos[i].get("flavor"),
                 "flavor_set": sku_infos[i].get("flavor_set"), "carbonation": sku_infos[i].get("carbonation"),
                 "sweetener": sku_infos[i].get("sweetener"), "pulp": sku_infos[i].get("pulp")},
                {"volume": canon_infos[j].get("volume"), "pack": canon_infos[j].get("pack"),
                 "package_type": canon_infos[j].get("package_type"), "flavor": canon_infos[j].get("flavor"),
                 "flavor_set": canon_infos[j].get("flavor_set"), "carbonation": canon_infos[j].get("carbonation"),
                 "sweetener": canon_infos[j].get("sweetener"), "pulp": canon_infos[j].get("pulp")},
                a)
            if v is not None:
                attr_pairs[a].append((i, j, int(v)))
    # disagree examples: mined conflicts (cap per attribute)
    for i in [k for k, b in enumerate(sku_barcodes) if b and b in gtin_to_idx]:
        for cand in (int(c) for c in rng.sample(list(order), min(int(args.conflict_pool), len(order)))):
            if canon_gtins[cand] == sku_barcodes[i]:
                continue
            hits = set(attribute_conflict_types(sku_infos[i], canon_infos[cand]))
            if hits:
                mono_pairs.append((i, cand, 0))
                for a in hits & set(ATTRS):
                    if len([p for p in attr_pairs[a] if p[2] == 0]) < int(args.max_pairs):
                        attr_pairs[a].append((i, cand, 0))
                break

    try:
        resolve_model(model_key)
        model = load_local_sentence_transformer(model_key, device=DEVICE)
        model.max_seq_length = int(cfg["training"]["max_seq_length"])  # SSOT
        batch_size = int(cfg["training"]["batch_size_embed"])  # SSOT

        def _enc(texts: list[str]) -> np.ndarray:
            return np.asarray(model.encode(
                texts, batch_size=batch_size, show_progress_bar=False,
                normalize_embeddings=True, convert_to_numpy=True), dtype=np.float64)

        needed_sku = sorted({p[0] for ps in list(attr_pairs.values()) + [mono_pairs] for p in ps})
        needed_canon = sorted({p[1] for ps in list(attr_pairs.values()) + [mono_pairs] for p in ps})
        es = _enc([sku_texts[i] for i in needed_sku])
        ec = _enc([canon_texts[j] for j in needed_canon])
    except (FileNotFoundError, KeyError, ValueError, OSError, RuntimeError) as exc:
        return _fail(f"encoder bundle for {model_key!r} missing or unloadable — {exc}")
    emap_s = {row: k for k, row in enumerate(needed_sku)}
    emap_c = {col: k for k, col in enumerate(needed_canon)}

    def _splitless_auc(pairs: list[tuple[int, int, int]]) -> dict | None:
        tr = [p for p in pairs if sku_barcodes[p[0]] not in test_ids]
        te = [p for p in pairs if sku_barcodes[p[0]] in test_ids]
        ytr = np.asarray([p[2] for p in tr])
        yte = np.asarray([p[2] for p in te])
        if len(set(ytr.tolist())) < 2 or len(set(yte.tolist())) < 2 or len(te) < 10:
            return None
        Xtr = np.abs(es[[emap_s[p[0]] for p in tr]] - ec[[emap_c[p[1]] for p in tr]])
        Xte = np.abs(es[[emap_s[p[0]] for p in te]] - ec[[emap_c[p[1]] for p in te]])
        clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
        return {"auc": float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])),
                "n_train": len(tr), "n_test": len(te)}

    out: dict[str, dict | None] = {"monolith_same_product": _splitless_auc(mono_pairs)}
    for a in ATTRS:
        out[f"head:{a}"] = _splitless_auc(attr_pairs[a])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"encoder": model_key, "heads": out}, indent=2) + "\n")

    print(f"\nattribute probes — {model_key} (linear, frozen, held-out identities)")
    print(f"{'head':<26}{'AUC':>7}{'n_tr':>7}{'n_te':>7}")
    for name, res in out.items():
        if res is None:
            print(f"{name:<26}{'n/a':>7}")
        else:
            print(f"{name:<26}{res['auc']:>7.3f}{res['n_train']:>7d}{res['n_test']:>7d}")
    print("read: head AUC >> 0.5 = linearly decodable = cheap specialist territory")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
