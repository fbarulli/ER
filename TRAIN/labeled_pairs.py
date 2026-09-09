"""Build labeled_pairs.csv from gate_results.csv.

true_label=1: proceed & similarity >= 0.8 (
              gate-confirmed same volume+pack, near-identical canonical).
true_label=0: hard_no & similarity >= 0.8 (text-similar but gate-proven
              different size/pack — the hard-negative class).
fallback pairs stay OUT: that tier is 'uncertain' by design and would inject
label noise into both classes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import pandas as pd

from lib.common import RESULTS, F, load_config
from lib.schemas import check_labeled_pairs_frame

# thresholds from the SSOT (TRAIN/training.yaml pairs.*) — were hardcoded 0.8
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
