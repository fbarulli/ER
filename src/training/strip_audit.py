"""strip_audit.py — per-SKU transformation ladder with FULL removal visibility.

Owner directive 2026-09-08: "I want to see what was removed from these
entries to see if we removed too much." Every stage of clean_sku_text (the
exact text the model trains on) is materialized here, with the tokens each
stage REMOVED — nothing strips silently anymore.

Stages (identical order + code paths as pipeline.clean_sku_text — this
module REUSES those functions, never reimplements them):
  s0_raw_title / s0_raw_attribute   the inputs as stored
  s1_normalize        lowercase, ×->x, non-alnum->space, whitespace collapse
  s2_join             title + ' ' + attribute
  s3_volume_pack      _VOLUME_PACK_RE strip (numbers+units, pack counts)
  s4_stopwords        MINIMAL_STOPWORDS + single-char tokens
  s5_number_tokens    number-token reference strip (semantic digits survive)
  final               clean_sku_text() — must equal s5

Usage:
  python src/training/strip_audit.py --gtin 747519482230        one SKU, full ladder
  python src/training/strip_audit.py --sample 200              random SKUs, summary
  python src/training/strip_audit.py --sample 200 --levels 8   + similarity ladder

The similarity ladder (--levels): bucket pairs (same-GTIN sku vs its
canonical model text) by lexical Jaccard, then sample from EVERY bucket —
"view samples from all levels of similarity" so over-stripping at any
similarity band is visible, not averaged away. Semantic similarity (ST
cosine) computed when --semantic is given (CPU; 5k sample ~= minutes).
"""

from __future__ import annotations

import argparse
import random
from collections import Counter

import pandas as pd

from pipeline import (
    _VOLUME_PACK_RE,
    MINIMAL_STOPWORDS,
    canonical_model_text,
    clean_sku_text,
    normalize_text,
    strip_schema_words,
)
from core.common import (
    DATA_DIR,
    SEED,
    artifact,
    ensure_parent,
    load_dataset_deduped,
    resolve_model,
    runtime,
    strip_ladder_bands,
    training_cfg,
    trace_artifact,
)
from core.nlp import encode_corpus

# AUDIT 2026-09-09: RESULTS used to be re-derived here from
# paths.results_dir with its own abs/rel resolution — a duplicate of
# lib.common.RESULTS that could silently diverge. lib.common is the one
# SSOT for every config-derived path.


def ladder(title: str, attribute: str = "", brand: str = "") -> dict:
    """Replay clean_sku_text stage by stage, recording removed tokens."""
    from pipeline import spell_numeric_brand, strip_number_tokens

    s1 = normalize_text(title)
    s1a = normalize_text(attribute or "")
    s2 = (s1 + " " + s1a).strip()
    s3 = _VOLUME_PACK_RE.sub(" ", s2)
    toks4 = [t for t in s3.split() if t not in MINIMAL_STOPWORDS and len(t) > 1]
    s4 = " ".join(toks4)
    s5 = strip_number_tokens(s4, spell_numeric_brand(brand or ""))
    final = clean_sku_text(title, attribute, brand)
    return {
        "raw_title": str(title),
        "raw_attribute": str(attribute or ""),
        "s1_normalize": s2,
        "removed_normalize": sorted(
            (set(str(title).lower().split()) | set(str(attribute or "").lower().split()))
            - set(s2.split())
        ),
        "s3_volume_pack": s3,
        "removed_volume_pack": sorted(set(s2.split()) - set(s3.split())),
        "s4_stopwords": s4,
        "removed_stopwords": sorted(set(s3.split()) - set(s4.split())),
        "s5_number_tokens": s5,
        "removed_number_tokens": sorted(set(s4.split()) - set(s5.split())),
        "final": final,
        "final_matches": s5 == final,  # must ALWAYS be True
    }


def jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def build_audit(n: int | None, seed: int) -> list[dict]:
    df = load_dataset_deduped()
    if n:
        df = df.sample(n=min(n, len(df)), random_state=seed)
    rows = []
    for _, r in df.iterrows():
        e = ladder(
            r.get("title", ""), r.get("attribute", ""), r.get("brand", "")
        )
        e["sku_id"] = str(r.get("product_id", ""))
        e["barcode"] = str(r.get("barcode", ""))
        rows.append(e)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gtin", help="audit every SKU with this GTIN")
    ap.add_argument("--sample", type=int, help="random N SKUs")
    ap.add_argument("--levels", type=int, default=0,
                    help="similarity-ladder sample count per band")
    ap.add_argument("--semantic", action="store_true",
                    help="add semantic (MiniLM) similarity to the ladder "
                    "(CPU; minutes on 5k)")
    args = ap.parse_args()

    if args.gtin:
        df = load_dataset_deduped()
        sub = df[df["barcode"].astype(str) == args.gtin]
        if not len(sub):
            raise SystemExit(f"GTIN {args.gtin}: no rows in dataset_deduped")
        for _, r in sub.iterrows():
            e = ladder(r.get("title", ""), r.get("attribute", ""), r.get("brand", ""))
            print(f"\n=== SKU {r.get('product_id')} (GTIN {args.gtin}) ===")
            print(f"  s1 normalize   : {e['s1_normalize'][:100]}")
            print(f"  s3 volume/pack : {e['s3_volume_pack'][:100]}   removed: {e['removed_volume_pack']}")
            print(f"  s4 stopwords   : {e['s4_stopwords'][:100]}   removed: {e['removed_stopwords']}")
            print(f"  s5 numbers     : {e['s5_number_tokens'][:100]}   removed: {e['removed_number_tokens']}")
            print(f"  FINAL          : {e['final'][:100]}")
            if not e["final_matches"]:
                print("  !!! LADDER MISMATCH — clean_sku_text diverged from stages")
        return

    # default sample SSOT: config/training.yaml audit.strip_audit_sample
    # (was inline 200, then EDA/eda.yaml strip_audit_sample)
    from core.common import training_cfg as _training_cfg

    n = args.sample or int(_training_cfg().audit.strip_audit_sample)
    rows = build_audit(n, SEED)
    out = artifact("visibility", {"name": "strip_audit.csv"})
    ensure_parent(out)
    pd.DataFrame(rows).to_csv(out, index=False)
    trace_artifact("visibility", out)
    n_mismatch = sum(1 for e in rows if not e["final_matches"])
    print(f"[strip-audit] {len(rows):,} SKUs -> results/logs/strip_audit.csv")
    print(f"  ladder mismatches (must be 0): {n_mismatch}")
    all_removed = Counter()
    for e in rows:
        for t in e["removed_volume_pack"]:
            all_removed[f"volume_pack:{t}"] += 1
        for t in e["removed_stopwords"]:
            all_removed[f"stopword:{t}"] += 1
        for t in e["removed_number_tokens"]:
            all_removed[f"number:{t}"] += 1
    print("\n  top removed tokens by stage:")
    for t, c in all_removed.most_common(20):
        print(f"    {c:>5,}  {t}")

    if args.levels:
        # similarity ladder: sku final text vs its GTIN's canonical model text
        # AUDIT 2026-09-09: was RESULTS.parent / "results" / ... — manual path
        # re-derivation; now lib.common RESULTS + the F[] file registry.
        from core.common import F

        canon = pd.read_csv(F["canonical_records"],
                            dtype={"gtin": str})
        canon_map = {
            g: strip_schema_words(canonical_model_text(c))
            for g, c in zip(canon["gtin"], canon["canonical"], strict=True)
        }
        entries = []
        for e in rows:
            c = canon_map.get(e["barcode"])
            if c and e["final"]:
                entries.append((e, c, jaccard(e["final"], c)))
        # AUDIT (SSOT move, this round): the ladder's band edges were an
        # inline literal list here — now audit.strip_ladder_bands in
        # config/training.yaml (validated by AuditSpec: contiguous ascending
        # cover of [0, 1+eps]), read via lib.common.strip_ladder_bands.
        bands = strip_ladder_bands()
        rng = random.Random(SEED)
        sims = None
        if args.semantic:
            # AUDIT 2026-09-09: was a hardcoded hub id + inline batch/seq
            # literals + manual cache path. Registry key (config
            # training.base_model via resolve_model; knobs via runtime();
            # cache via lib.common DATA_DIR (the one embeddings cache).
            texts = [e["final"] for e, _, _ in entries] + [c for _, c, _ in entries]
            emb, _ = encode_corpus(
                resolve_model(str(training_cfg().training.base_model)), texts,
                batch_size=runtime("batch_size_embed"),
                max_seq_length=runtime("max_seq_length"), device="cpu",
                cache_dir=str(DATA_DIR / "embeddings_cache"),
            )
            import numpy as np

            n_e = len(entries)
            sims = np.einsum("ij,ij->i", emb[:n_e], emb[n_e:])
        print(f"\n  similarity ladder ({len(entries):,} SKUs with canonicals):")
        for lo, hi in bands:
            band = [(e, c, j) for e, c, j in entries if lo <= j < hi]
            if not band:
                print(f"    [{lo:.1f},{hi:.1f}): 0 entries")
                continue
            sem = ""
            if sims is not None:
                idx = [entries.index((e, c, j)) for e, c, j in band]
                sem = f" | mean semantic cos {float(sims[idx].mean()):.3f}"
            print(f"    [{lo:.1f},{hi:.1f}): {len(band):,} entries{sem}")
            for e, c, j in rng.sample(band, min(args.levels, len(band))):
                print(f"      lex {j:.3f} sku: {e['final'][:60]}")
                print(f"               canon: {c[:60]}")
                got = set(e["final"].split())
                want = set(c.split())
                print(f"               sku-only: {sorted(got - want)[:6]} | canon-only: {sorted(want - got)[:6]}")


if __name__ == "__main__":
    main()
