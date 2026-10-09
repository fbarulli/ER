"""Build labeled_pairs.csv from gate_results.csv.

true_label=1: proceed & similarity >= 0.8 (
              gate-confirmed same volume+pack, near-identical canonical).
true_label=0: hard_no & similarity >= 0.8 (text-similar but gate-proven
              different size/pack — the hard-negative class).
fallback pairs stay OUT: that tier is 'uncertain' by design and would inject
label noise into both classes. The exclusion is COUNTED below (no silent
drops) — pins removed 2026-10-06 by owner ruling; the four-bucket partition
recorded in the stage manifest is the record.

MANIFEST (SILENT_DROPS task 7): the stage now snapshots gate_results.csv
(begin_manifest), writes labeled_pairs.csv atomically (atomic_write_csv —
a crash never leaves a truncated CSV on the final path) and publishes
results/manifests/labeled_pairs.json LAST. The row accounting records the
EXACT four-bucket partition of gate_results rows:
  pos (proceed & sim>=thr) + neg (hard_no & sim>=thr)
  + fallback_gate_pairs (any similarity — fallback is counted FIRST, so a
    low-similarity fallback pair is counted once, here, never twice)
  + below_similarity_threshold (proceed/hard_no rows under their threshold)
Every gate row lands in exactly one bucket; the closure
input == output + sum(dropped) is asserted by finish_manifest before the
manifest is published.

TRACE ROWS (core.tracing, the ONE consolidated trace)
-----------------------------------------------------
Stage ``labeled_pairs``. Emitted:
  run   gate_rows.split               gate_results rows -> labeled rows, with the
                                      EXACT four-bucket partition in detail
  group partition.reason_census       one EXACT census row per bucket label —
                                      the SAME labels the manifest's `dropped`
                                      keys carry, so the two join by key
  ent   partition.*                   the sampled GATE PAIRS behind each label
                                      (gtin1|gtin2 + its decision + similarity)
  run   labeled_rows.published        the atomic write under the frame contract
Sampling caps are core.tracing's (ENTITY_SAMPLE_PER_REASON / ENTITY_ROW_CAP) and
appear in the sample_budget row's detail; nothing here is unbounded.

Class map (one owner per responsibility):
  - SimilarityThresholds — the pairs-config thresholds
  - GateSplit            — gate CSV load, partition masks, labeled frame
  - SplitLedger          — the four-bucket partition + row accounting
  - LabeledWriter        — the frame contract + stage closure lines

The entrypoint is import-safe so the preparation run can call main directly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.common import (
    SEED,
    TRAINING_CONFIG_PATH,
    F,
    ensure_parent,
    load_config,
)
from core.manifest import atomic_write_csv, begin_manifest, finish_manifest
from core.pair_identity import PairIdentity
from core.run_log import RunLogger
from core.schemas import check_labeled_pairs_frame
from core.step_trace import timed
from core.tracing import ENTITY_ROW_CAP, ENTITY_SAMPLE_PER_REASON, TraceRun

_LOG = RunLogger(__name__)

#: The pipeline stage these rows belong to (core.tracing ``stage`` column).
STAGE = "labeled_pairs"

_LABELED_DECISIONS = ("proceed", "hard_no")

#: The partition's bucket labels. These are the SAME strings the manifest's
#: row_accounting ``dropped`` keys carry, so the trace's reason census and the
#: manifest close over one vocabulary (a reader joins them by key, no mapping).
BUCKET_POS = "pos_labeled"
BUCKET_NEG = "hard_neg_labeled"
BUCKET_FALLBACK = "fallback_gate_pairs"
BUCKET_BELOW = "below_similarity_threshold"
BUCKET_OTHER = "other_gate_decision"


class SimilarityThresholds:
    """The (proceed, hard-no) similarity thresholds from the pairs config."""

    @staticmethod
    def load() -> tuple[float, float]:
        """The (proceed, hard-no) thresholds."""
        cfg = load_config()
        pairs_cfg = cfg["pairs"]
        return (
            float(pairs_cfg["proceed_sim_threshold"]),
            float(pairs_cfg["hardneg_sim_threshold"]),
        )


class GateSplit:
    """gate_results.csv under its pinned dtype, split into labeled classes."""

    @staticmethod
    def load_gate_csv(path: Path) -> pd.DataFrame:
        """gate_results.csv under its pinned string-dtype contract."""
        return pd.read_csv(
            path,
            dtype={"gtin1": str, "gtin2": str},
            keep_default_na=False,
        )

    @staticmethod
    def partition_masks(
        g: pd.DataFrame, pos_sim: float, neg_sim: float
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """The gate row masks: (positives, hard_negatives, fallback, kept)."""
        pos = (g.gate_decision == "proceed") & (g.similarity >= pos_sim)
        neg = (g.gate_decision == "hard_no") & (g.similarity >= neg_sim)
        fallback = g.gate_decision == "fallback"
        return pos, neg, fallback, pos | neg

    @staticmethod
    def labeled_frame(g: pd.DataFrame, pos_mask, neg_mask) -> pd.DataFrame:
        """pos + hard-neg gate rows projected to the labeled contract.

        ``pair_id`` is stamped here (the ONE key, core.pair_identity), not
        carried from the gate frame: this stage reads the gate CSV from disk,
        so it must work whether or not that artifact already carries the key.
        """
        pos = g[pos_mask].copy()
        pos["true_label"] = 1
        neg = g[neg_mask].copy()
        neg["true_label"] = 0
        out = pd.concat(
            [pos[["gtin1", "gtin2", "true_label"]],
             neg[["gtin1", "gtin2", "true_label"]]],
            ignore_index=True,
        )
        out["pair_id"] = PairIdentity.column(out["gtin1"], out["gtin2"])
        return out


# ── the four-bucket partition ledger ────────────────────────────────────────
class SplitLedger:
    """Capture-only accounting of every gate row (SILENT_DROPS task 7)."""

    @staticmethod
    def dropped_buckets(
        g: pd.DataFrame, is_fallback, is_kept
    ) -> dict[str, int]:
        """The exact four-bucket partition of gate_results rows.

        Computed from the same frame the split used (never from
        assumptions):
          fallback        = gate_decision == "fallback" — counted FIRST and
                            exclusively, so a fallback pair below the sim
                            threshold is dropped here ONCE, never twice
          below_threshold = the remaining non-fallback rows whose decision
                            was proceed/hard_no but whose similarity sat
                            under that class's threshold
          other           = any non-fallback, non-kept row that is NOT below
                            threshold (an unknown gate_decision value —
                            zero today; kept as its own key so a future
                            gate tier shows up as a number instead of
                            vanishing)
        """
        below_thr = (~is_fallback) & (~is_kept)
        return {
            BUCKET_FALLBACK: int(is_fallback.sum()),
            BUCKET_BELOW: int(
                (below_thr & g.gate_decision.isin(_LABELED_DECISIONS)).sum()
            ),
            BUCKET_OTHER: int(
                (below_thr & ~g.gate_decision.isin(_LABELED_DECISIONS)).sum()
            ),
        }

    @classmethod
    def reason_buckets(
        cls, g: pd.DataFrame, pos, neg, is_fallback, is_kept
    ) -> pd.Series:
        """One bucket label per gate row — the PER-ENTITY form of the partition.

        Precedence: kept-over-below (a kept pair is pos/neg even if a future
        threshold change made it ambiguous), then the two labeled classes, then
        the fallback tier, then below-threshold, then anything else. It yields
        exactly :meth:`dropped_buckets` folded against the labeled classes, which
        is what makes the trace's census and the manifest's closure one statement.
        """
        below_thr = (~is_fallback) & (~is_kept)
        labels = pd.Series(BUCKET_OTHER, index=g.index, dtype="string")
        labels.loc[below_thr & g.gate_decision.isin(_LABELED_DECISIONS)] = BUCKET_BELOW
        labels.loc[is_fallback] = BUCKET_FALLBACK
        labels.loc[neg] = BUCKET_NEG
        labels.loc[pos] = BUCKET_POS
        return labels

    @classmethod
    def row_accounting(
        cls,
        g: pd.DataFrame,
        out: pd.DataFrame,
        is_fallback,
        is_kept,
        pos_sim: float,
        neg_sim: float,
    ) -> dict:
        """Capture-only row accounting.

        The four buckets partition gate_results EXACTLY; closure:
        input == output + sum(dropped), asserted by finish_manifest.
        """
        return {
            "input_rows": len(g),
            "output_rows": len(out),
            "dropped": cls.dropped_buckets(g, is_fallback, is_kept),
            # population detail (outside `dropped`; not part of the closure)
            "pos_labeled": int((out.true_label == 1).sum()),
            "hard_neg_labeled": int((out.true_label == 0).sum()),
            "pos_sim_threshold": pos_sim,
            "hardneg_sim_threshold": neg_sim,
        }


# ── publication ─────────────────────────────────────────────────────────────
class LabeledWriter:
    """The frame contract write + the stage's closure lines."""

    @staticmethod
    def labeled_csv(out: pd.DataFrame) -> None:
        """Write under the FRAME CONTRACT (lib.schemas): columns, label
        domain, GTIN endpoints, no duplicate (gtin1, gtin2) — asserted at
        the boundary."""
        check_labeled_pairs_frame(out)
        atomic_write_csv(out, ensure_parent(F["labeled_pairs"]), index=False)

    @staticmethod
    def closure(manifest_path, row_accounting: dict, out: pd.DataFrame) -> None:
        """The stage's completion lines with their closure arithmetic."""
        dropped = row_accounting["dropped"]
        _LOG.info(
            f"[manifest] labeled_pairs complete -> {manifest_path} | "
            f"closure {row_accounting['input_rows']:,} == "
            f"{row_accounting['output_rows']:,} kept + "
            f"{sum(dropped.values()):,} dropped "
            f"(fallback {dropped['fallback_gate_pairs']:,} / "
            f"below-thr {dropped['below_similarity_threshold']:,} / "
            f"other {dropped['other_gate_decision']:,})"
        )
        _LOG.info(
            f"labeled_pairs.csv: {len(out):,} rows "
            f"({(out.true_label == 1).sum():,} pos / "
            f"{(out.true_label == 0).sum():,} hard-neg)"
        )


def _similarity_thresholds() -> tuple[float, float]:
    """The pairs-config thresholds (see :class:`SimilarityThresholds`)."""
    return SimilarityThresholds.load()


def _row_accounting(
    g: pd.DataFrame,
    out: pd.DataFrame,
    is_fallback,
    is_kept,
    pos_sim: float,
    neg_sim: float,
) -> dict:
    """Capture-only row accounting (see :class:`SplitLedger`)."""
    return SplitLedger.row_accounting(g, out, is_fallback, is_kept, pos_sim, neg_sim)


@timed
def main() -> None:
    """Read gate_results.csv, split pos/hard-neg, write labeled_pairs.csv.

    Pure pandas over the gate CSV (~135k rows, <1s); the manifest wraps the
    whole flow — begin at stage start, finish LAST. ONE consolidated-trace
    writer commits the partition census, the sampled gate pairs and the
    publication as one flow.
    """
    pos_sim, neg_sim = _similarity_thresholds()
    gate_csv = F["gate_results"]
    trace = TraceRun(STAGE)
    # Seed: the SSOT seed (lib.common.SEED) — this stage is deterministic
    # (no RNG consumed), recorded so the manifest's environment block
    # pins which seed the lane runs under.
    manifest = begin_manifest(
        "labeled_pairs", inputs=[gate_csv, TRAINING_CONFIG_PATH], seed=SEED
    )

    with _LOG.section("labeled_pairs.split"):
        g = GateSplit.load_gate_csv(gate_csv)
        _LOG.info(f"[labeled] loaded {len(g):,} gate pairs")
        pos_mask, neg_mask, is_fallback, is_kept = GateSplit.partition_masks(
            g, pos_sim, neg_sim
        )
        out = GateSplit.labeled_frame(g, pos_mask, neg_mask)
        _LOG.info(f"[labeled] positives: {int(pos_mask.sum()):,} | "
                  f"hard negatives: {int(neg_mask.sum()):,}")

    # ── NO SILENT DROPS (owner doctrine): the fallback exclusion above is a
    # data drop by construction — count it LOUDLY and record it in the stage
    # manifest's row accounting (the four-bucket partition). Pins are removed
    # (2026-10-06); drift across runs is visible via provenance, not asserts.
    n_fallback_excluded = int(is_fallback.sum())
    _LOG.info(f"[labeled] excluded {n_fallback_excluded:,} fallback-gate pairs "
              f"(gate=fallback — not labelable as pos/hard-neg)")
    assert isinstance(n_fallback_excluded, int) and n_fallback_excluded >= 0, (
        f"n_fallback_excluded must be a non-negative int, got "
        f"{n_fallback_excluded!r}"
    )

    with _LOG.section("labeled_pairs.write"):
        LabeledWriter.labeled_csv(out)
        row_accounting = SplitLedger.row_accounting(
            g, out, is_fallback, is_kept, pos_sim, neg_sim
        )
    _record_partition(
        trace, g, out, pos_mask, neg_mask, is_fallback, is_kept,
        row_accounting, pos_sim, neg_sim, gate_csv,
    )
    manifest_path = finish_manifest(
        manifest,
        outputs=[F["labeled_pairs"]],
        row_accounting=row_accounting,
        expected_outputs=[F["labeled_pairs"]],
    )
    LabeledWriter.closure(manifest_path, row_accounting, out)
    trace.write()


def _record_partition(
    trace: TraceRun,
    g: pd.DataFrame,
    out: pd.DataFrame,
    pos_mask,
    neg_mask,
    is_fallback,
    is_kept,
    row_accounting: dict,
    pos_sim: float,
    neg_sim: float,
    gate_csv: object,
) -> None:
    """The stage row, the exact per-bucket census, the named pairs, the write.

    The census is the manifest's own four-bucket partition expressed per row, so
    the two agree by construction and a reader can join them on the bucket label
    itself. Every bucket whose rows were DROPPED carries its pairs at ENTITY
    grain (the gtin pair is the entity here), which is what answers "which pair
    was left out, and why" without opening gate_results.csv.
    """
    trace.add(
        "gate_rows",
        "split",
        in_count=int(len(g)),
        out_count=int(len(out)),
        reason=(
            "a labeled row must be gate-confirmed: proceed with similarity >= "
            f"{pos_sim} (positive) or hard_no with similarity >= {neg_sim} "
            "(hard negative); the fallback tier is uncertain by design and stays out"
        ),
        detail={
            **row_accounting,
            "bucket_labels": [
                BUCKET_POS, BUCKET_NEG, BUCKET_FALLBACK, BUCKET_BELOW, BUCKET_OTHER
            ],
        },
        source=str(gate_csv),
    )
    labels = SplitLedger.reason_buckets(
        g, pos_mask, neg_mask, is_fallback, is_kept
    )
    records = [
        {
            "pair": PairIdentity.of(row["gtin1"], row["gtin2"]),
            "label": str(labels.loc[index]),
            "gate_decision": str(row["gate_decision"]),
            "similarity": row["similarity"],
            "gate_reason": str(row.get("gate_reason", "")),
        }
        for index, row in g[['gtin1', 'gtin2', 'gate_decision', 'similarity', 'gate_reason']].iterrows()
    ]
    trace.add_entities(
        "partition",
        records,
        key_of=lambda record: record["pair"],
        reason_of=lambda record: record["label"],
        detail_of=lambda record: {
            "gate_decision": record["gate_decision"],
            "similarity": record["similarity"],
            "gate_reason": record["gate_reason"],
        },
        source=str(gate_csv),
        per_reason=ENTITY_SAMPLE_PER_REASON,
        total_cap=ENTITY_ROW_CAP,
    )
    trace.add(
        "labeled_rows",
        "published",
        in_count=int(len(out)),
        out_count=int(len(out)),
        reason="labeled_pairs.csv written atomically under the frame contract",
        detail={
            "path": str(F["labeled_pairs"]),
            "rows": int(len(out)),
            "positives": int((out.true_label == 1).sum()),
            "hard_negatives": int((out.true_label == 0).sum()),
            "pos_sim_threshold": pos_sim,
            "hardneg_sim_threshold": neg_sim,
        },
        source=str(F["labeled_pairs"]),
    )


if __name__ == "__main__":
    main()
