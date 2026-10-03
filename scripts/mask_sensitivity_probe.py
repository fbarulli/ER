#!/usr/bin/env python3
"""Mask-sensitivity probe: does dropping the discriminative field move decisions?

For true pairs, dropping a field (e.g. volume) should barely move the score
(evidence lost, identity kept). For volume/pack-conflict pairs, dropping the
CONFLICTING field removes the disagreement evidence, so the score should
RISE. Random masking of equal extent is the control: if targeted shifts
dwarf random shifts, masking hits real evidence (solid); if they tie,
masking is decorative.

Usage: python scripts/mask_sensitivity_probe.py [--sample 128] [--encoder minilm_l6]
Writes results/mask_sensitivity.json. Fail-loud on missing bundle/inputs.
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from core.attribute_conflicts import (
    attribute_conflict_types,
    canonical_attribute_info,
    sku_attribute_info,
)
from core.common import F, SEED, TRAIN_ROOT, load_config, load_local_sentence_transformer, resolve_model
from core.model_input import build_canonical_text, build_sku_text
from training.masking import field_of, mask_targeted, swap_structured_field

ALL_FIELDS = ("volume", "pack", "package_type", "flavor", "carbonation", "sweetener", "pulp")

sys.path.insert(0, str(TRAIN_ROOT / "scripts"))
from sid_phase0_report import _text_info  # noqa: E402


def _fail(message: str) -> int:
    print(f"mask-sensitivity ABORT: {message}", file=sys.stderr, flush=True)
    return 2


def _sibling_gtin_index(canon_gtins: list[str]) -> dict[str, int]:
    """Exact-spelling membership map (first lookup tier of `_resolve_identity`)."""
    return {g: i for i, g in enumerate(canon_gtins)}


def _resolve_identity(
    gtins: list[str], exact: dict[str, int], canon_gtins: list[str]
) -> tuple[list[int | None], int]:
    """Row -> canonical index, exact first then sibling-equivalent; misses loud.

    Try the EXACT spelling first, then the UPC-12↔EAN-13 sibling via
    ``core.gtin.gtin_equivalent`` (zero-prefix fold + checksum). A spelled
    gtin that hits NEITHER is counted and reported loudly — never dropped
    in silence — so the probe's identity coverage is a visible number, not a
    shrunk population.
    """
    from core.gtin import gtin_equivalent

    resolved: list[int | None] = []
    misses = 0
    for b in gtins:
        if b and b in exact:
            resolved.append(exact[b])
            continue
        hit = None
        if b:
            for j, g in enumerate(canon_gtins):
                if gtin_equivalent(b, g):
                    hit = j
                    break
        if hit is None and b:
            misses += 1
        resolved.append(hit)
    return resolved, misses


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=128)
    parser.add_argument("--encoder", type=str, default=None)
    parser.add_argument("--max-pairs", type=int, default=200)
    parser.add_argument("--conflict-pool", type=int, default=512)
    parser.add_argument("--out", type=str, default=str(TRAIN_ROOT / "results" / "mask_sensitivity.json"))
    args = parser.parse_args(argv)

    cfg = load_config()
    model_key = str(args.encoder or cfg["training"]["base_model"])
    thr = float(cfg["split"]["fixed_threshold"])
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
    gtin_to_idx = _sibling_gtin_index(canon_gtins)

    sku_infos = [sku_attribute_info(str(r.get("sku_name_eng", "")), str(r.get("attribute", "")))
                 for _, r in sku.iterrows()]
    sku_texts = [build_sku_text(row, _text_info(info)) for (_, row), info in zip(sku.iterrows(), sku_infos)]
    sku_gtins = sku["gtin"].fillna("").astype(str).tolist()

    # sibling-tolerant + LOUD: exact spelling first, then the UPC-12↔EAN-13
    # sibling via core.gtin.gtin_equivalent; a spelled gtin that hits
    # NEITHER is counted and reported — never silently thinned out of the
    # probe population.
    identity, identity_misses = _resolve_identity(sku_gtins, gtin_to_idx, canon_gtins)
    print(
        f"[identity] gtin -> canonical: resolved "
        f"{sum(v is not None for v in identity):,}/{len(sku_gtins):,} "
        f"({identity_misses:,} spelled gtins miss BOTH the exact and the "
        "sibling-equivalent lookup)"
    )

    true_rows = [i for i, v in enumerate(identity) if v is not None][: int(args.max_pairs)]
    if not true_rows:
        return _fail("no sample SKU gtin matches a canonical GTIN")
    order = np.arange(len(canon_gtins))
    pairs: list[tuple[int, int, str, frozenset]] = []  # (sku_row, canon_idx, class, conflict_hits)
    for sku_row in true_rows:
        pairs.append((sku_row, identity[sku_row], "true", frozenset()))
        if len([p for p in pairs if p[2] == "conflict"]) >= int(args.max_pairs):
            continue
        for cand in (int(c) for c in rng.sample(list(order), min(int(args.conflict_pool), len(order)))):
            if canon_gtins[cand] == sku_gtins[sku_row]:
                continue
            hits = set(attribute_conflict_types(sku_infos[sku_row], canon_infos[cand]))
            if hits & {"volume", "pack", "flavor", "package_type", "carbonation", "sweetener", "pulp"}:
                pairs.append((sku_row, cand, "conflict", frozenset(hits)))
                break
    n_conf = len([p for p in pairs if p[2] == "conflict"])
    if not n_conf:
        return _fail("no attribute-conflict pair mined")

    rng2 = random.Random(int(SEED) + 1)
    # Per (pair, field): drop variant (background 0 to isolate the field) and
    # swap variant (anchor takes the counterpart's value — counterfactual
    # agreement). Only counterfactual swaps (values actually differed) count.
    variants: list[tuple[int, str, str]] = []  # (pair_idx, op, field)
    drop_texts: list[str] = []
    swap_texts: list[str] = []
    for k, (i, j, _cls, _hits) in enumerate(pairs):
        anchor_fields = {field_of(t) for t in sku_texts[i].split()} - {None}
        other_fields = {field_of(t) for t in canon_texts[j].split()} - {None}
        for field in ALL_FIELDS:
            if field in anchor_fields:
                t, _, _ = mask_targeted(sku_texts[i], fields=[field],
                                        background_prob=0.0, rng=rng2)
                variants.append((k, "drop", field))
                drop_texts.append(t)
            if field in anchor_fields and field in other_fields:
                s, swapped = swap_structured_field(sku_texts[i], canon_texts[j],
                                                   field=field, rng=rng2)
                if swapped:
                    variants.append((k, "swap", field))
                    swap_texts.append(s)
    anchors = [sku_texts[i] for i, _, _, _ in pairs]
    others = [canon_texts[j] for _, j, _, _ in pairs]

    try:
        resolve_model(model_key)
        model = load_local_sentence_transformer(model_key, device=DEVICE)
        model.max_seq_length = int(cfg["training"]["max_seq_length"])  # SSOT
        batch_size = int(cfg["training"]["batch_size_embed"])  # SSOT
        def _enc(texts: list[str]) -> np.ndarray:
            return np.asarray(model.encode(
                texts, batch_size=batch_size, show_progress_bar=False,
                normalize_embeddings=True, convert_to_numpy=True), dtype=np.float64)
        ea, eo = _enc(anchors), _enc(others)
        ed = _enc(drop_texts) if drop_texts else np.zeros((0, ea.shape[1]))
        es = _enc(swap_texts) if swap_texts else np.zeros((0, ea.shape[1]))
    except (FileNotFoundError, KeyError, ValueError, OSError, RuntimeError) as exc:
        return _fail(f"encoder bundle for {model_key!r} missing or unloadable — {exc}")

    base = np.einsum("ij,ij->i", ea, eo)
    # align variant scores back to pair order (drops and swaps packed separately)
    shift: dict[tuple[int, str, str], float] = {}
    di = si = 0
    for (k, op, field) in variants:
        i, j, _, _ = pairs[k]
        if op == "drop":
            shift[(k, op, field)] = float(ed[di] @ eo[k])
            di += 1
        else:
            shift[(k, op, field)] = float(es[si] @ eo[k])
            si += 1
    delta = {key: score - float(base[key[0]]) for key, score in shift.items()}

    out: dict[str, dict[str, dict]] = {}
    for cls in ("true", "conflict"):
        out[cls] = {}
        for field in ALL_FIELDS:
            for op in ("drop", "swap"):
                vals = np.asarray([delta[key] for key in delta
                                   if pairs[key[0]][2] == cls and key[1] == op and key[2] == field])
                if len(vals) == 0:
                    continue
                out[cls][f"{op}:{field}"] = {
                    "n": len(vals),
                    "shift_mean": float(np.mean(vals)),
                    "big_move_rate": float(np.mean(np.abs(vals) > 0.05)),
                }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"encoder": model_key, "fixed_threshold": thr, "classes": out}, indent=2) + "\n")

    n_true = len([p for p in pairs if p[2] == "true"])
    n_conf = len([p for p in pairs if p[2] == "conflict"])
    print(f"\nmask-sensitivity — {model_key} (n_true={n_true}, n_conf={n_conf})")
    print(f"{'class':<9}{'op:field':<16}{'n':>5}{'shift':>8}{'big':>7}")
    for cls in ("true", "conflict"):
        for key in sorted(out[cls]):
            c = out[cls][key]
            print(f"{cls:<9}{key:<16}{c['n']:>5d}{c['shift_mean']:>+8.3f}{c['big_move_rate']:>7.3f}")
    print("read: conflict swap:+ means the field decides (agreement raises score); "
          "drop:- means shared-evidence counting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
