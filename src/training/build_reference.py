"""build_reference.py — reproduce number_tokens_reference.csv (the SSOT
number-token verdict census).

The committed data/number_tokens_reference.csv (1,742 rows) is
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

import argparse
from pathlib import Path

import pandas as pd

from pipeline import build_reference, census_texts, reference_path
from core.common import load_dataset
from core.run_log import RunLogger
from core.step_trace import timed

log = RunLogger(__name__)


def _brand_vocab() -> set[str]:
    """Vocabulary from the RAW brand strings (lowercased word tokens).

    Verified detail (this is what makes the rebuild byte-equal): the vocab
    is built from set(b.lower().split()) of the raw brand column — NOT
    normalize_text, NOT the spelled numeric brands. spell_numeric_brand
    is applied by clean_sku_text at USE time, not census time.
    """
    df = load_dataset(columns=["brand"])
    vocab: set[str] = set()
    brands = df["brand"].dropna().astype(str)
    for brand in log.progress(brands, desc="brand_vocab", unit="brand"):
        vocab.update(brand.lower().split())
    return vocab


@timed
def build() -> pd.DataFrame:
    """Rebuild the reference from the current dataset_deduped.csv."""
    from core.common import load_dataset_deduped

    deduped = load_dataset_deduped()
    texts = census_texts(deduped)
    ref = build_reference(texts, _brand_vocab())
    return ref


def _verdict_mismatches(committed: pd.DataFrame, rebuilt: pd.DataFrame) -> pd.DataFrame:
    """Rows whose verdict drifts between the committed and rebuilt census."""
    merged = committed.merge(
        rebuilt, on="token", how="outer", suffixes=("_old", "_new")
    )
    return merged[
        merged["verdict_old"].fillna("") != merged["verdict_new"].fillna("")
    ]


def _require_committed_csv(path: Path) -> pd.DataFrame:
    """Load the committed reference, failing with the pinned verify message."""
    if not path.exists():
        raise SystemExit(
            f"[verify] FAIL: {path} missing — nothing to compare against"
        )
    return pd.read_csv(path, dtype={"token": str})


def _rebuilt_reference() -> pd.DataFrame:
    """The rebuilt reference, reusing the preparation run's object when live."""
    from training.preparation_run import active_preparation

    run = active_preparation()
    rebuilt = run._objects.get("number_reference") if run is not None else None
    return rebuilt if rebuilt is not None else build()


def _verify(committed: pd.DataFrame, rebuilt: pd.DataFrame) -> None:
    """Assert the committed reference reproduces exactly; SystemExit on drift."""
    if len(committed) != len(rebuilt):
        raise SystemExit(
            f"[verify] FAIL: row count {len(committed)} vs rebuilt "
            f"{len(rebuilt)}"
        )
    bad_verdict = _verdict_mismatches(committed, rebuilt)
    if len(bad_verdict):
        for _, row in log.progress(
            bad_verdict.head(10).iterrows(),
            desc="verdict_drift", unit="token",
            total=min(10, len(bad_verdict)),
        ):
            log.info(
                f"  token {row['token']!r}: {row['verdict_old']!r} -> "
                f"{row['verdict_new']!r}"
            )
        raise SystemExit(
            f"[verify] FAIL: {len(bad_verdict)} verdict mismatches"
        )
    log.info(
        f"[verify] PASS: {len(committed):,} rows, all verdicts identical "
        f"— the committed reference reproduces exactly"
    )


def _register_in_preparation(ref: pd.DataFrame) -> None:
    """Publish the freshly built frame to a live preparation run, if any."""
    from training.preparation_run import active_preparation

    run = active_preparation()
    if run is not None:
        run._objects["number_reference"] = ref


def _write_reference(ref: pd.DataFrame, path: Path) -> None:
    """Persist the rebuilt reference and register it with a live run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ref.to_csv(path, index=False)
    _register_in_preparation(ref)
    log.info(f"wrote {path} ({len(ref):,} rows)")
    log.info(f"  verdict mix: {dict(ref['verdict'].value_counts())}")


def _parse_args() -> argparse.Namespace:
    """The lane's only switch: --verify compares instead of writing."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--verify",
        action="store_true",
        help="compare against the committed CSV instead of writing",
    )
    return ap.parse_args()


@timed
def main() -> None:
    args = _parse_args()
    path = reference_path()

    if args.verify:
        _verify(_require_committed_csv(path), _rebuilt_reference())
        return

    _write_reference(build(), path)


if __name__ == "__main__":
    main()
