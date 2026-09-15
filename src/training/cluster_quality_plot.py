"""Plot ANN pairwise and clustering quality from a generated report.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("report_json", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    payload = json.loads(args.report_json.read_text())
    metric = payload["metrics"][sorted(payload["metrics"])[0]]
    values = {
        "Pairwise\nprecision": metric["calibration_pairwise_precision"],
        "Pairwise\nrecall": metric["calibration_pairwise_recall"],
        "Over-merge\nrate": metric["calibration_over_merge_rate"],
        "Under-merge\nrate": metric["calibration_under_merge_rate"],
    }
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    bars = ax.bar(values.keys(), values.values(), color=["#4c72b0", "#55a868", "#c44e52", "#8172b2"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate")
    ax.set_title("ANN pairwise precision/recall and clustering error rates")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, min(value + 0.025, 1.02), f"{value:.3f}", ha="center")
    out = args.out or args.report_json.with_name("ann_cluster_quality.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
