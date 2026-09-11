"""training-data composition plot — the owner's single overview figure.

One bar per population with ABSOLUTE counts on top (the % share alone
means little): gate-confirmed hard positives (canonical-anchored),
gate-labeled hard negatives (gate hard_no ∩ sim≥0.8 — deterministic
gate rules, not human-verified), augmented positives (masked-anchor
copies), silver eval pool (mined in-band eval-only pairs) and TOTAL.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from euromonitor.core.common import RESULTS, load_config, load_dataset_deduped, plot_dpi

_cfg = load_config()
# NO FALLBACKS (owner Q27): masking + training sections hard-indexed —
# missing keys crash; the old .get(0.15) silently contradicted masking.frac=1.00.
_mask_cfg = _cfg["masking"]


def main() -> None:
    import euromonitor.pipeline as dp

    d = dp.build_training_data(load_dataset_deduped())
    pos, neg = d["pos"], d["neg"]
    n_pos, n_neg = len(pos), len(neg)
    n_masked = int(n_pos * float(_mask_cfg["frac"]))
    # mined in-band (sku,sku) eval pool — src/euromonitor/training/train lane's mining target
    # (SSOT: training.n_target_mining; was a hardcoded 20_000 duplicate)
    silver = int(_cfg["training"]["n_target_mining"])

    # Label-semantics honesty (2026-09-12): these labels are derived from
    # deterministic gate rules over the transductive census, NOT human-
    # verified ground truth — worded "gate-confirmed"/"gate-labeled"
    # accordingly, never as human-verified claims.
    bars = [
        ("hard positives\n(gate-confirmed, same-GTIN)", n_pos, "#4C72B0"),
        ("hard negatives\n(gate-labeled, gate hard-no)", n_neg, "#C44E52"),
        ("augmented positives\n(masked anchors)", n_masked, "#8172B2"),
        ("silver eval pool\n(mined in-band, eval-only)", silver, "#937860"),
        (
            "TOTAL training pairs\n(pos + masked + neg)",
            n_pos + n_masked + n_neg,
            "#55A868",
        ),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    names = [b[0] for b in bars]
    vals = [b[1] for b in bars]
    colors = [b[2] for b in bars]
    ax.bar(names, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.015, f"{v:,}", ha="center", fontsize=10)
    ax.set_ylabel("pairs")
    ax.set_title("Training data composition — counts per population")
    ax.grid(axis="y", alpha=0.3)
    out = RESULTS / "training_data_composition.png"
    fig.savefig(out, dpi=plot_dpi())
    plt.close(fig)
    print(f"[plot] {out}", flush=True)


if __name__ == "__main__":
    main()
