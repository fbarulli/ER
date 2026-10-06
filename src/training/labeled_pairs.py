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

The entrypoint is import-safe so the preparation run can call main directly.
"""

from __future__ import annotations

import pandas as pd

from core.common import (
    SEED,
    TRAINING_CONFIG_PATH,
    F,
    ensure_parent,
    load_config,
)
from core.manifest import atomic_write_csv, begin_manifest, finish_manifest
from core.run_log import RunLogger
from core.schemas import check_labeled_pairs_frame
from core.step_trace import timed

log = RunLogger(__name__)

_LABELED_DECISIONS = ("proceed", "hard_no")


def _similarity_thresholds() -> tuple[float, float]:
    """The (proceed, hard-no) similarity thresholds from the pairs config."""
    cfg = load_config()
    pairs_cfg = cfg["pairs"]
    return (
        float(pairs_cfg["proceed_sim_threshold"]),
        float(pairs_cfg["hardneg_sim_threshold"]),
    )


def _load_gate_csv(path) -> pd.DataFrame:
    """gate_results.csv under its pinned string-dtype contract."""
    return pd.read_csv(
        path,
        dtype={"gtin1": str, "gtin2": str},
        keep_default_na=False,
    )


def _gate_partition_masks(
    g: pd.DataFrame, pos_sim: float, neg_sim: float
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """The gate row masks: (positives, hard_negatives, fallback, kept)."""
    pos = (g.gate_decision == "proceed") & (g.similarity >= pos_sim)
    neg = (g.gate_decision == "hard_no") & (g.similarity >= neg_sim)
    fallback = g.gate_decision == "fallback"
    return pos, neg, fallback, pos | neg


def _labeled_frame(g: pd.DataFrame, pos_mask, neg_mask) -> pd.DataFrame:
    """pos + hard-neg gate rows projected to the labeled contract."""
    pos = g[pos_mask].copy()
    pos["true_label"] = 1
    neg = g[neg_mask].copy()
    neg["true_label"] = 0
    return pd.concat(
        [pos[["gtin1", "gtin2", "true_label"]], neg[["gtin1", "gtin2", "true_label"]]],
        ignore_index=True,
    )


def _dropped_buckets(g: pd.DataFrame, is_fallback, is_kept) -> dict[str, int]:
    """The exact four-bucket partition of gate_results rows.

    Computed from the same frame the split used (never from assumptions):
      fallback        = gate_decision == "fallback" — counted FIRST and
                        exclusively, so a fallback pair below the sim
                        threshold is dropped here ONCE, never twice
      below_threshold = the remaining non-fallback rows whose decision
                        was proceed/hard_no but whose similarity sat
                        under that class's threshold
      other           = any non-fallback, non-kept row that is NOT below
                        threshold (an unknown gate_decision value — zero
                        today; kept as its own key so a future gate tier
                        shows up as a number instead of vanishing)
    """
    below_thr = (~is_fallback) & (~is_kept)
    return {
        "fallback_gate_pairs": int(is_fallback.sum()),
        "below_similarity_threshold": int(
            (below_thr & g.gate_decision.isin(_LABELED_DECISIONS)).sum()
        ),
        "other_gate_decision": int(
            (below_thr & ~g.gate_decision.isin(_LABELED_DECISIONS)).sum()
        ),
    }


def _row_accounting(
    g: pd.DataFrame,
    out: pd.DataFrame,
    is_fallback,
    is_kept,
    pos_sim: float,
    neg_sim: float,
) -> dict:
    """Capture-only row accounting (SILENT_DROPS task 7).

    The four buckets below partition gate_results EXACTLY; closure:
    input == output + sum(dropped), asserted by finish_manifest.
    """
    return {
        "input_rows": len(g),
        "output_rows": len(out),
        "dropped": _dropped_buckets(g, is_fallback, is_kept),
        # population detail (outside `dropped`; not part of the closure)
        "pos_labeled": int((out.true_label == 1).sum()),
        "hard_neg_labeled": int((out.true_label == 0).sum()),
        "pos_sim_threshold": pos_sim,
        "hardneg_sim_threshold": neg_sim,
    }


def _write_labeled_csv(out: pd.DataFrame) -> None:
    """Write under the FRAME CONTRACT (lib.schemas): columns, label domain,
    GTIN endpoints, no duplicate (gtin1, gtin2) — asserted at the boundary."""
    check_labeled_pairs_frame(out)
    atomic_write_csv(out, ensure_parent(F["labeled_pairs"]), index=False)


def _log_closure(manifest_path, row_accounting: dict, out: pd.DataFrame) -> None:
    """The stage's completion lines with their closure arithmetic."""
    dropped = row_accounting["dropped"]
    log.info(
        f"[manifest] labeled_pairs complete -> {manifest_path} | "
        f"closure {row_accounting['input_rows']:,} == "
        f"{row_accounting['output_rows']:,} kept + "
        f"{sum(dropped.values()):,} dropped "
        f"(fallback {dropped['fallback_gate_pairs']:,} / "
        f"below-thr {dropped['below_similarity_threshold']:,} / "
        f"other {dropped['other_gate_decision']:,})"
    )
    log.info(
        f"labeled_pairs.csv: {len(out):,} rows "
        f"({(out.true_label == 1).sum():,} pos / {(out.true_label == 0).sum():,} hard-neg)"
    )


@timed
def main() -> None:
    """Read gate_results.csv, split pos/hard-neg, write labeled_pairs.csv.

    Pure pandas over the gate CSV (~135k rows, <1s); the manifest wraps the
    whole flow — begin at stage start, finish LAST.
    """
    pos_sim, neg_sim = _similarity_thresholds()
    gate_csv = F["gate_results"]
    # Seed: the SSOT seed (lib.common.SEED) — this stage is deterministic
    # (no RNG consumed), recorded so the manifest's environment block
    # pins which seed the lane runs under.
    manifest = begin_manifest("labeled_pairs", inputs=[gate_csv, TRAINING_CONFIG_PATH], seed=SEED)

    with log.section("labeled_pairs.split"):
        g = _load_gate_csv(gate_csv)
        log.info(f"[labeled] loaded {len(g):,} gate pairs")
        pos_mask, neg_mask, is_fallback, is_kept = _gate_partition_masks(
            g, pos_sim, neg_sim
        )
        out = _labeled_frame(g, pos_mask, neg_mask)
        log.info(f"[labeled] positives: {int(pos_mask.sum()):,} | "
                 f"hard negatives: {int(neg_mask.sum()):,}")

    # ── NO SILENT DROPS (owner doctrine): the fallback exclusion above is a
    # data drop by construction — count it LOUDLY and record it in the stage
    # manifest's row accounting (the four-bucket partition). Pins are removed
    # (2026-10-06); drift across runs is visible via provenance, not asserts.
    n_fallback_excluded = int(is_fallback.sum())
    log.info(f"[labeled] excluded {n_fallback_excluded:,} fallback-gate pairs "
             f"(gate=fallback — not labelable as pos/hard-neg)")
    assert isinstance(n_fallback_excluded, int) and n_fallback_excluded >= 0, (
        f"n_fallback_excluded must be a non-negative int, got {n_fallback_excluded!r}"
    )

    with log.section("labeled_pairs.write"):
        _write_labeled_csv(out)
        row_accounting = _row_accounting(
            g, out, is_fallback, is_kept, pos_sim, neg_sim
        )
    manifest_path = finish_manifest(
        manifest,
        outputs=[F["labeled_pairs"]],
        row_accounting=row_accounting,
        expected_outputs=[F["labeled_pairs"]],
    )
    _log_closure(manifest_path, row_accounting, out)


if __name__ == "__main__":
    main()
