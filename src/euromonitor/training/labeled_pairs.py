"""Build labeled_pairs.csv from gate_results.csv.

true_label=1: proceed & similarity >= 0.8 (
              gate-confirmed same volume+pack, near-identical canonical).
true_label=0: hard_no & similarity >= 0.8 (text-similar but gate-proven
              different size/pack — the hard-negative class).
fallback pairs stay OUT: that tier is 'uncertain' by design and would inject
label noise into both classes. The exclusion is COUNTED + PINNED below
(no silent drops): the fallback count must equal the census pin
lib.common.PINNED_GATE_FALLBACK_PAIRS (same universe as the selftest
oracle).

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

INVOCATION NOTE: this module deliberately keeps its module-level flow
(it has always run at import; the repo invokes it as a subprocess —
cli/colab.py's data-prep chain runs scripts by file path, no other module
imports it). The work moved into main() called at module level, so BOTH
`python src/euromonitor/training/labeled_pairs.py` and
`python -m euromonitor.training.labeled_pairs` (and any import) behave
identically — nothing that calls this script changes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import pandas as pd

from euromonitor.core.common import (
    PINNED_GATE_FALLBACK_PAIRS,
    RESULTS,
    SEED,
    F,
    load_config,
)
from euromonitor.core.manifest import atomic_write_csv, begin_manifest, finish_manifest
from euromonitor.core.schemas import check_labeled_pairs_frame

# thresholds from the SSOT (src/euromonitor/training/training.yaml pairs.*) — were hardcoded 0.8
# twice; a config change must not silently diverge from build_training_data.
# NO FALLBACK (owner doctrine): a missing key crashes here, loudly, at import
# time — never a silent inline default that can drift from the YAML.
_pairs_cfg = load_config()["pairs"]
POS_SIM = float(_pairs_cfg["proceed_sim_threshold"])
NEG_SIM = float(_pairs_cfg["hardneg_sim_threshold"])


def main() -> None:
    """Read gate_results.csv, split pos/hard-neg, write labeled_pairs.csv.

    Pure pandas over the gate CSV (~135k rows, <1s); the manifest wraps the
    whole flow — begin at stage start, finish LAST.
    """
    gate_csv = RESULTS / F["gate_results"]
    # Seed: the SSOT seed (lib.common.SEED) — this stage is deterministic
    # (no RNG consumed), recorded so the manifest's environment block
    # pins which seed the lane runs under.
    manifest = begin_manifest("labeled_pairs", inputs=[gate_csv], seed=SEED)

    g = pd.read_csv(
        gate_csv,
        dtype={"gtin1": str, "gtin2": str},
        keep_default_na=False,
    )
    pos = g[(g.gate_decision == "proceed") & (g.similarity >= POS_SIM)].copy()
    pos["true_label"] = 1
    neg = g[(g.gate_decision == "hard_no") & (g.similarity >= NEG_SIM)].copy()
    neg["true_label"] = 0

    # ── NO SILENT DROPS (owner doctrine): the fallback exclusion above is a
    # data drop by construction — count it LOUDLY and pin it. This script reads
    # the SAME transductive-census universe (gate_results.csv) the selftest
    # oracle pins, so the excluded count must equal the pinned census fallback
    # count exactly; anything else means the universe or the gate drifted and
    # the selftest pin (oracle_pinned_counts) is now lying.
    n_fallback_excluded = int((g.gate_decision == "fallback").sum())
    print(f"[labeled] excluded {n_fallback_excluded:,} fallback-gate pairs (gate=fallback — not labelable as pos/hard-neg)")
    assert isinstance(n_fallback_excluded, int) and n_fallback_excluded >= 0, (
        f"n_fallback_excluded must be a non-negative int, got {n_fallback_excluded!r}"
    )
    assert n_fallback_excluded == PINNED_GATE_FALLBACK_PAIRS, (
        f"census drift: excluded {n_fallback_excluded:,} fallback-gate pairs but "
        f"lib.common.PINNED_GATE_FALLBACK_PAIRS == {PINNED_GATE_FALLBACK_PAIRS:,} "
        "(the selftest oracle pins the same universe — update BOTH pins together "
        "on intentional drift)"
    )

    out = pd.concat(
        [pos[["gtin1", "gtin2", "true_label"]], neg[["gtin1", "gtin2", "true_label"]]],
        ignore_index=True,
    )
    # FRAME CONTRACT (lib.schemas): columns, label domain, GTIN endpoints, no
    # duplicate (gtin1, gtin2) — asserted at the write boundary.
    check_labeled_pairs_frame(out)
    out_path = RESULTS / F["labeled_pairs"]
    atomic_write_csv(out, out_path, index=False)

    # ── row accounting (SILENT_DROPS task 7; capture-only) ───────────────────
    # The four buckets below partition gate_results EXACTLY, computed from
    # the same frame the split used (never from assumptions):
    #   kept            = pos + neg (the rows written out)
    #   fallback        = gate_decision == "fallback" — counted FIRST and
    #                     exclusively, so a fallback pair below the sim
    #                     threshold is dropped here ONCE, never twice
    #   below_threshold = the remaining non-fallback rows whose decision
    #                     was proceed/hard_no but whose similarity sat
    #                     under that class's threshold
    #   other           = any non-fallback, non-kept row that is NOT below
    #                     threshold (an unknown gate_decision value — zero
    #                     today; kept as its own key so a future gate tier
    #                     shows up as a number instead of vanishing)
    # Closure: input == output + sum(dropped), asserted by finish_manifest.
    kept = len(out)
    is_fallback = g.gate_decision == "fallback"
    is_kept = (
        (g.gate_decision == "proceed") & (g.similarity >= POS_SIM)
    ) | (
        (g.gate_decision == "hard_no") & (g.similarity >= NEG_SIM)
    )
    below_thr = (~is_fallback) & (~is_kept)
    n_below = int((below_thr & (g.gate_decision.isin(["proceed", "hard_no"]))).sum())
    n_other = int((below_thr & ~g.gate_decision.isin(["proceed", "hard_no"])).sum())
    dropped = {
        "fallback_gate_pairs": int(is_fallback.sum()),
        "below_similarity_threshold": n_below,
        "other_gate_decision": n_other,
    }
    row_accounting = {
        "input_rows": len(g),
        "output_rows": kept,
        "dropped": dropped,
        # population detail (outside `dropped`; not part of the closure)
        "pos_labeled": int((out.true_label == 1).sum()),
        "hard_neg_labeled": int((out.true_label == 0).sum()),
        "pos_sim_threshold": POS_SIM,
        "hardneg_sim_threshold": NEG_SIM,
    }
    manifest_path = finish_manifest(
        manifest,
        outputs=[out_path],
        row_accounting=row_accounting,
        expected_outputs=[F["labeled_pairs"]],
    )
    print(
        f"[manifest] labeled_pairs complete -> {manifest_path} | "
        f"closure {row_accounting['input_rows']:,} == "
        f"{row_accounting['output_rows']:,} kept + "
        f"{sum(dropped.values()):,} dropped "
        f"(fallback {dropped['fallback_gate_pairs']:,} / "
        f"below-thr {dropped['below_similarity_threshold']:,} / "
        f"other {dropped['other_gate_decision']:,})"
    )
    print(
        f"labeled_pairs.csv: {len(out):,} rows "
        f"({(out.true_label == 1).sum():,} pos / {(out.true_label == 0).sum():,} hard-neg)"
    )


main()  # module-level flow preserved: this script has always run at import
