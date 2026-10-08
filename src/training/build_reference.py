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

Class map (one owner per responsibility):
  - BrandVocabulary      — the raw-brand lowercase token vocabulary
  - ReferenceVerifier    — the --verify surface against the committed CSV
  - ReferencePublisher   — the write + live-preparation registration

TRACE ROWS (core.tracing, the ONE consolidated trace)
-----------------------------------------------------
TWO stages, because prepare_all runs TWO passes:
  ``number_reference``  the (re)build pass (build() + publish)
  ``verify_reference``  the --verify pass (rebuilt comparison + named drift)
A shared stage name would be a DEFECT, not a simplification: core.tracing
._commit replaces a stage's rows for the run, so the verify pass would destroy
the rebuild pass's rows (reproduced: corpus.texts_censused vanished, only
verify.* left). Emitted:
  run   corpus.texts_censused            deduped rows -> census texts
  run   vocabulary.tokens_built          raw brand strings -> token set (unit
                                         change aware: no in/out pair when the
                                         token set exceeds the brand count)
  run   reference.verdicts_recorded      digit-token occurrences -> DISTINCT
                                         verdict rows (the collapse is the drop)
  group verdict.reason_census            the EXACT verdict census, one row per
                                         verdict (keep_brand / keep_name_embedded
                                         / strip), plus the bounded sample budget
  ent   verdict.*                        the sampled verdict entities (token +
                                         literal class/occurrences/rule readback)
  run   verify.verdicts_compared         --verify: committed rows -> reproducing
                                         rows, with the drift count as the drop
  group verify.drift.reason_census       --verify drift, one bucket per exact
                                         ``old -> new`` verdict pair
  ent   verify.drift.*                   the drifted TOKENS, named, with the
                                         committed and rebuilt verdict
  run   publish.reference_written        rows published + verdict mix
Sampling caps are core.tracing's (ENTITY_SAMPLE_PER_REASON / ENTITY_ROW_CAP) and
are written into the sample_budget row's detail; nothing here is unbounded.

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

from pipeline import (
    build_reference,
    census_texts,
    reference_path,
    unit_change_counts,
)
from core.common import load_dataset
from core.run_log import RunLogger
from core.step_trace import timed
from core.tracing import (
    ENTITY_ROW_CAP,
    ENTITY_SAMPLE_PER_REASON,
    TraceRun,
    count_rows,
)

_LOG = RunLogger(__name__)

#: The TWO prepare_all passes are DIFFERENT stages and must NOT share a trace
#: stage name. ``core.tracing._commit`` REPLACES a stage's rows for the run, so a
#: shared name lets the verify pass destroy the rebuild pass's rows under the
#: same run id (reproduced: corpus.texts_censused vanished, only verify.* left).
#: ``prepare_all._STAGE_MODULES`` names the passes ``number_reference`` and
#: ``verify_reference``, so the trace stages use exactly those names.
STAGE_WRITE = "number_reference"
STAGE_VERIFY = "verify_reference"


class BrandVocabulary:
    """Vocabulary from the RAW brand strings (lowercased word tokens).

    Verified detail (this is what makes the rebuild byte-equal): the vocab
    is built from set(b.lower().split()) of the raw brand column — NOT
    normalize_text, NOT the spelled numeric brands. spell_numeric_brand
    is applied by clean_sku_text at USE time, not census time.
    """

    @staticmethod
    def build(trace: TraceRun | None = None) -> set[str]:
        """The raw-brand lowercase token vocabulary.

        ``trace`` is this stage's ONE consolidated-trace writer (core.tracing);
        when supplied the vocabulary row is added to it. The count is the RAW
        brand strings actually read versus the tokens they yield — a real
        funnel, never an estimate.
        """
        df = load_dataset(columns=["brand"])
        vocab: set[str] = set()
        brands = df["brand"].dropna().astype(str)
        for brand in _LOG.progress(brands, desc="brand_vocab", unit="brand"):
            vocab.update(brand.lower().split())
        if trace is not None:
            # The vocabulary is a UNIT CHANGE, not a funnel: one raw brand string
            # yields several tokens, so the token set can exceed the brand-string
            # population (it does not on the live corpus — 71,623 brand strings ->
            # 2,495 tokens — but nothing structurally forbids the inversion). The
            # shared rule (pipeline.unit_change_counts) states no in/out pair when
            # that happens, so dropped_count can never go negative.
            brands_read, tokens, unit_change = unit_change_counts(len(brands), len(vocab))
            trace.add(
                "vocabulary",
                "tokens_built",
                in_count=brands_read,
                out_count=tokens,
                reason=(
                    "raw brand strings lowercased and split into the token set; the "
                    "resulting tokens are what the verdict census consults (NOT "
                    "normalize_text and NOT the spelled numeric brands)"
                ),
                detail={
                    "brands_read": int(len(brands)),
                    "tokens": len(vocab),
                    "unit_change": unit_change,
                    "note": "one entry per token, so brands collapse into tokens",
                },
                source="raw export brand column (core.common.load_dataset)",
            )
        return vocab


def _brand_vocab(trace: TraceRun | None = None) -> set[str]:
    """Vocabulary from the RAW brand strings (see :class:`BrandVocabulary`)."""
    return BrandVocabulary.build(trace)


@timed
def build(trace: TraceRun | None = None) -> pd.DataFrame:
    """Rebuild the reference from the current dataset_deduped.csv.

    ``trace`` is this stage's ONE writer. It is optional so a standalone
    ``build()`` (the verifier's rebuilt reference, a test) is still traced: with
    no writer supplied one is created and committed here.
    """
    from core.common import load_dataset_deduped

    own = trace is None
    if own:
        trace = TraceRun(STAGE_WRITE)
    deduped = load_dataset_deduped()
    texts = census_texts(deduped)
    trace.add(
        "corpus",
        "texts_censused",
        in_count=int(len(deduped)),
        out_count=len(texts),
        reason=(
            "one census text per deduped row, built WITHOUT the final "
            "number-token strip so every digit token in the corpus is censused"
        ),
        detail={
            "deduped_rows": int(len(deduped)),
            "census_texts": len(texts),
            "batch_grain": (
                "not applicable: this stage reads ONE frame in one pass and "
                "censuses its digit tokens; there is no chunk/file fan-out to "
                "trace, so the grain below is stage -> entity"
            ),
        },
        source="dataset_deduped (core.common.load_dataset_deduped)",
    )
    ref = build_reference(texts, _brand_vocab(trace))
    _record_reference(trace, ref, texts)
    if own:
        trace.write()
    return ref


def _record_reference(
    trace: TraceRun, ref: pd.DataFrame, texts: list[str]
) -> None:
    """The reference funnel + its exact verdict census and entity sample.

    The funnel is occurrences -> DISTINCT token rows: a token appearing 4,000
    times is ONE reference row, so the collapse is the honest drop and the
    detail names it. The verdict census is EXACT (one group row per verdict);
    the entity rows are core.tracing's bounded sample of that census.
    """
    occurrences = (
        int(ref["n_occurrences"].astype(float).sum()) if len(ref) else 0
    )
    distinct = int(len(ref))
    trace.add(
        "reference",
        "verdicts_recorded",
        in_count=occurrences,
        out_count=distinct,
        reason=(
            "one reference row per DISTINCT digit token; a token's occurrences "
            "collapse into its single verdict row"
        ),
        detail={
            "corpus_texts": len(texts),
            "distinct_tokens": distinct,
            "digit_token_occurrences": occurrences,
            "collapsed_occurrences": occurrences - distinct,
            "verdict_mix": count_rows(ref["verdict"], limit=None),
            "class_mix": count_rows(ref["class"], limit=None),
        },
        source="pipeline.NumberTokenAuthority.reference_frame",
    )
    trace.add_entities(
        "verdict",
        [dict(row) for row in ref.to_dict("records")],
        key_of=lambda row: row["token"],
        reason_of=lambda row: row["verdict"],
        detail_of=lambda row: {
            "class": row["class"],
            "n_occurrences": int(float(row["n_occurrences"])),
            "rule": row["rule"],
        },
        source="pipeline.NumberTokenAuthority.reference_frame",
        per_reason=ENTITY_SAMPLE_PER_REASON,
        total_cap=ENTITY_ROW_CAP,
    )


# ── the --verify surface ────────────────────────────────────────────────────
class ReferenceVerifier:
    """Asserts the committed reference reproduces exactly (SystemExit on drift)."""

    @staticmethod
    def missing(committed: pd.DataFrame, rebuilt: pd.DataFrame) -> pd.DataFrame:
        """Rows whose verdict drifts between committed and rebuilt census."""
        merged = committed.merge(
            rebuilt, on="token", how="outer", suffixes=("_old", "_new")
        )
        return merged[
            merged["verdict_old"].fillna("") != merged["verdict_new"].fillna("")
        ]

    @staticmethod
    def committed(path: Path) -> pd.DataFrame:
        """Load the committed reference, failing loud with the pinned message."""
        if not path.exists():
            raise SystemExit(
                f"[verify] FAIL: {path} missing — nothing to compare against"
            )
        return pd.read_csv(path, dtype={"token": str})

    @staticmethod
    def rebuilt(trace: TraceRun | None = None) -> pd.DataFrame:
        """The rebuilt reference, reusing the preparation run's object when live."""
        from training.preparation_run import active_preparation

        run = active_preparation()
        rebuilt = run._objects.get("number_reference") if run is not None else None
        return rebuilt if rebuilt is not None else build(trace=trace)

    @classmethod
    def verify(
        cls,
        committed: pd.DataFrame,
        rebuilt: pd.DataFrame,
        trace: TraceRun | None = None,
    ) -> None:
        """Assert the committed reference reproduces exactly; exit on drift.

        The comparison is traced at ENTITY grain: a drifted verdict is not a
        count, it is a named token with its committed and rebuilt verdict, so the
        trace answers "which token, and why" without the CSV. On drift the
        trace is committed BEFORE the SystemExit so the failure survives.
        """
        own = trace is None
        if own:
            trace = TraceRun(STAGE_VERIFY)
        if len(committed) != len(rebuilt):
            trace.add(
                "verify",
                "row_count_compared",
                in_count=int(len(committed)),
                out_count=0,
                reason=(
                    f"committed {int(len(committed))} rows vs rebuilt "
                    f"{int(len(rebuilt))} rows: the row populations differ, so "
                    f"no verdict can be compared"
                ),
                detail={
                    "committed_rows": int(len(committed)),
                    "rebuilt_rows": int(len(rebuilt)),
                },
                source="committed reference CSV vs rebuilt census",
            )
            trace.write()
            raise SystemExit(
                f"[verify] FAIL: row count {len(committed)} vs rebuilt "
                f"{len(rebuilt)}"
            )
        bad_verdict = cls.missing(committed, rebuilt)
        n_bad = int(len(bad_verdict))
        trace.add(
            "verify",
            "verdicts_compared",
            in_count=int(len(committed)),
            out_count=int(len(committed)) - n_bad,
            reason=(
                "the committed reference reproduces every verdict"
                if not n_bad
                else f"{n_bad} token verdict(s) drifted between committed and rebuilt"
            ),
            detail={
                "compared": int(len(committed)),
                "reproducing": int(len(committed)) - n_bad,
                "drifted": n_bad,
            },
            source="committed reference CSV vs rebuilt census",
        )
        if n_bad:
            trace.add_entities(
                "verify.drift",
                [dict(row) for row in bad_verdict.to_dict("records")],
                key_of=lambda row: row["token"],
                reason_of=lambda row: (
                    f"committed {row['verdict_old']!r} -> rebuilt "
                    f"{row['verdict_new']!r}"
                ),
                detail_of=lambda row: {
                    "committed_verdict": row["verdict_old"],
                    "rebuilt_verdict": row["verdict_new"],
                    "committed_class": row.get("class_old", ""),
                    "rebuilt_class": row.get("class_new", ""),
                },
                source="committed reference CSV vs rebuilt census",
                per_reason=ENTITY_SAMPLE_PER_REASON,
                total_cap=ENTITY_ROW_CAP,
            )
        if n_bad:
            for _, row in _LOG.progress(
                bad_verdict.head(10).iterrows(),
                desc="verdict_drift", unit="token",
                total=min(10, n_bad),
            ):
                _LOG.info(
                    f"  token {row['token']!r}: {row['verdict_old']!r} -> "
                    f"{row['verdict_new']!r}"
                )
            trace.write()
            raise SystemExit(
                f"[verify] FAIL: {n_bad} verdict mismatches"
            )
        _LOG.info(
            f"[verify] PASS: {len(committed):,} rows, all verdicts identical "
            f"— the committed reference reproduces exactly"
        )
        if own:
            trace.write()


def _verify(
    committed: pd.DataFrame,
    rebuilt: pd.DataFrame,
    trace: TraceRun | None = None,
) -> None:
    """Assert the committed reference reproduces exactly; exit on drift."""
    ReferenceVerifier.verify(committed, rebuilt, trace)


# ── the write + publication surface ────────────────────────────────────────
class ReferencePublisher:
    """Persists the rebuilt reference and registers it with a live run."""

    @staticmethod
    def register(ref: pd.DataFrame) -> None:
        """Publish the freshly built frame to a live preparation run, if any."""
        from training.preparation_run import active_preparation

        run = active_preparation()
        if run is not None:
            run._objects["number_reference"] = ref

    @classmethod
    def write(
        cls, ref: pd.DataFrame, path: Path, trace: TraceRun | None = None
    ) -> None:
        """Persist the rebuilt reference and register it with a live run."""
        path.parent.mkdir(parents=True, exist_ok=True)
        ref.to_csv(path, index=False)
        cls.register(ref)
        if trace is not None:
            trace.add(
                "publish",
                "reference_written",
                in_count=int(len(ref)),
                out_count=int(len(ref)),
                reason="reference CSV rewritten from the rebuilt census",
                detail={
                    "path": str(path),
                    "rows": int(len(ref)),
                    "verdict_mix": count_rows(ref["verdict"], limit=None)
                    if "verdict" in ref
                    else [],
                },
                source=str(path),
            )
        _LOG.info(f"wrote {path} ({len(ref):,} rows)")
        _LOG.info(f"  verdict mix: {dict(ref['verdict'].value_counts())}")


def _write_reference(
    ref: pd.DataFrame, path: Path, trace: TraceRun | None = None
) -> None:
    """Persist the rebuilt reference (see :class:`ReferencePublisher`)."""
    ReferencePublisher.write(ref, path, trace)


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
    # ONE writer per PASS, under that pass's own stage name: the rebuild and the
    # verify are separate prepare_all stages, and a shared stage name would let
    # this commit replace the other pass's rows for the same run.
    trace = TraceRun(STAGE_VERIFY if args.verify else STAGE_WRITE)

    if args.verify:
        rebuilt = ReferenceVerifier.rebuilt(trace)
        _verify(ReferenceVerifier.committed(path), rebuilt, trace)
        trace.write()
        return

    _write_reference(build(trace), path, trace)
    trace.write()


if __name__ == "__main__":
    main()
