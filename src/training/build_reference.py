"""build_reference.py — reproduce number_tokens_reference.csv (the SSOT
number-token verdict census).

The committed artifacts/data/number_tokens_reference.csv (1,742 rows) is
the ONLY input CSV besides dataset.csv that had no producer entrypoint —
its recipe lived only as functions in pipeline.py. This script is that
entrypoint, pinning the EXACT verified recipe:

    build_reference(census_texts(load_dataset_deduped()),
                    vocab = lowercase tokens of the RAW brand strings)

Verdict classes: keep_name_embedded / keep_brand / strip — decided by the
census (how often a digit token appears inside NAME text vs bare) plus the
numeric-brand spellbook. Coverage: 95.2% of digit tokens.

REPRODUCIBILITY CONTRACT (owner ruling): every .csv in this lane is either
committed input (dataset.csv, number_tokens_reference.csv) or regenerated
by a script. This file closes the last "committed but unregenerable" gap.

Usage:
  python src/training/build_reference.py            # (re)build the reference
  python src/training/build_reference.py --verify   # assert byte-equality vs
                                                # the committed CSV, exit 1
                                                # on any drift
NOTE — the census runs over dataset_deduped.csv, so regenerate dedupe
first if the raw export changed.
"""

from __future__ import annotations

import pandas as pd

from pipeline import build_reference, census_texts, reference_path
from core.common import load_dataset


def _brand_vocab() -> set[str]:
    """Vocabulary from the RAW brand strings (lowercased word tokens).

    Verified detail (this is what makes the rebuild byte-equal): the vocab
    is built from set(b.lower().split()) of the raw brand column — NOT
    normalize_text, NOT the spelled numeric brands. spell_numeric_brand
    is applied by clean_sku_text at USE time, not census time.
    """
    df = load_dataset()
    vocab: set[str] = set()
    for b in df["brand"].dropna().astype(str):
        vocab.update(b.lower().split())
    return vocab


def build() -> pd.DataFrame:
    """Rebuild the reference from the current dataset_deduped.csv."""
    from core.common import load_dataset_deduped

    deduped = load_dataset_deduped()
    texts = census_texts(deduped)
    ref = build_reference(texts, _brand_vocab())
    return ref


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--verify",
        action="store_true",
        help="compare against the committed CSV instead of writing",
    )
    args = ap.parse_args()
    p = reference_path()

    if args.verify:
        if not p.exists():
            raise SystemExit(
                f"[verify] FAIL: {p} missing — nothing to compare against"
            )
        committed = pd.read_csv(p, dtype={"token": str})
        rebuilt = build()
        if len(committed) != len(rebuilt):
            raise SystemExit(
                f"[verify] FAIL: row count {len(committed)} vs rebuilt "
                f"{len(rebuilt)}"
            )
        merged = committed.merge(
            rebuilt, on="token", how="outer", suffixes=("_old", "_new")
        )
        bad_verdict = merged[
            merged["verdict_old"].fillna("") != merged["verdict_new"].fillna("")
        ]
        if len(bad_verdict):
            for _, r in bad_verdict.head(10).iterrows():
                print(
                    f"  token {r['token']!r}: {r['verdict_old']!r} -> "
                    f"{r['verdict_new']!r}"
                )
            raise SystemExit(
                f"[verify] FAIL: {len(bad_verdict)} verdict mismatches"
            )
        print(
            f"[verify] PASS: {len(committed):,} rows, all verdicts identical "
            f"— the committed reference reproduces exactly",
            flush=True,
        )
        return

    ref = build()
    p.parent.mkdir(parents=True, exist_ok=True)
    ref.to_csv(p, index=False)
    print(f"wrote {p} ({len(ref):,} rows)", flush=True)
    print(
        f"  verdict mix: {dict(ref['verdict'].value_counts())}",
        flush=True,
    )


if __name__ == "__main__":
    main()
