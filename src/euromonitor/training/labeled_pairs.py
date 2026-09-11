"""Build labeled_pairs.csv from gate_results.csv.

true_label=1: proceed & similarity >= 0.8 (
              gate-confirmed same volume+pack, near-identical canonical).
true_label=0: hard_no & similarity >= 0.8 (text-similar but gate-proven
              different size/pack — the hard-negative class).
fallback pairs stay OUT: that tier is 'uncertain' by design and would inject
label noise into both classes. The exclusion is COUNTED + PINNED below
(no silent drops): the fallback count must equal the census pin
lib.common.PINNED_GATE_FALLBACK_PAIRS (13,768; same universe as the
selftest oracle).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import pandas as pd

from euromonitor.core.common import PINNED_GATE_FALLBACK_PAIRS, RESULTS, F, load_config
from euromonitor.core.schemas import check_labeled_pairs_frame

# thresholds from the SSOT (src/euromonitor/training/training.yaml pairs.*) — were hardcoded 0.8
# twice; a config change must not silently diverge from build_training_data.
# NO FALLBACK (owner doctrine): a missing key crashes here, loudly, at import
# time — never a silent inline default that can drift from the YAML.
_pairs_cfg = load_config()["pairs"]
POS_SIM = float(_pairs_cfg["proceed_sim_threshold"])
NEG_SIM = float(_pairs_cfg["hardneg_sim_threshold"])

g = pd.read_csv(
    RESULTS / F["gate_results"],
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
out.to_csv(RESULTS / F["labeled_pairs"], index=False)
print(
    f"labeled_pairs.csv: {len(out):,} rows "
    f"({(out.true_label == 1).sum():,} pos / {(out.true_label == 0).sum():,} hard-neg)"
)
