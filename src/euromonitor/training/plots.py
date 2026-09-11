"""plots.py — 07f report plots (pulled out of train.py verbatim:
per-fold AUC bar + score distributions + PR curve) + the train-vs-val
loss curve (owner directive 2026-09-10)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from euromonitor.core.common import RESULTS, plot_dpi


def report_plots(metrics_csv: str | Path, args: argparse.Namespace) -> None:
    """per-run score-distribution + fold-AUC bar chart."""
    m = pd.read_csv(metrics_csv)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ok = m[m["status"] == "ok"]
    if "auc" in ok.columns and len(ok):
        axes[0].bar(ok["fold"].astype(str), ok["auc"], color="#4c72b0")
        axes[0].set_ylim(0.5, 1.0)
        axes[0].set_title("test AUC per fold")
        axes[0].axhline(0.5, color="gray", ls="--", lw=0.8)
    if "auc_cross" in ok.columns and ok["auc_cross"].notna().any():
        axes[1].bar(ok["fold"].astype(str), ok["auc_cross"], color="#55a868")
        axes[1].set_ylim(0.5, 1.0)
        axes[1].set_title("cross-country AUC per fold")
    for ax in axes:
        ax.set_xlabel("fold")
    fig.tight_layout()
    out_png = RESULTS / f"train_{args.split}_payload-{args.payload}.png"
    fig.savefig(out_png, dpi=plot_dpi())
    plt.close(fig)
    print(f"[plot] {out_png}", flush=True)


def training_loss_plot(
    metrics_csv: str | Path, args: argparse.Namespace
) -> None:
    """training vs validation loss curves per fold (train loss + dev
    eval_loss per logged step, best dev AP annotated) — reads the json
    history columns the trainer emits. Owner directive 2026-09-10: the
    plot we keep is train-vs-val loss; the dev-AP vline stays."""
    from euromonitor.core.common import SSOT_LOSS

    m = pd.read_csv(metrics_csv)
    ok = m[m["status"] == "ok"].reset_index(drop=True)
    if "train_loss_hist" not in ok.columns or not len(ok):
        print("[plot] no loss history in metrics — skip", flush=True)
        return
    n_folds = len(ok)
    fig, axes = plt.subplots(1, n_folds, figsize=(4.2 * n_folds, 3.8), squeeze=False)
    for i, (_, r) in enumerate(ok.iterrows()):
        ax = axes[0][i]
        losses = (
            json.loads(r["train_loss_hist"]) if pd.notna(r["train_loss_hist"]) else []
        )
        val_losses = (
            json.loads(r["dev_loss_hist"])
            if "dev_loss_hist" in r and pd.notna(r.get("dev_loss_hist"))
            else []
        )
        aps = json.loads(r["dev_ap_hist"]) if pd.notna(r.get("dev_ap_hist")) else []
        if losses:
            ax.plot(
                range(len(losses)), losses, color="#c44e52", lw=1.6, label="train loss"
            )
        if val_losses:
            ax.plot(
                range(len(val_losses)),
                val_losses,
                color="#4c72b0",
                lw=1.6,
                ls="--",
                label="val loss",
            )
        elif losses:
            # no eval_loss column (triplet lane / older metrics CSV) — say so
            # instead of implying the flat line is the validation curve
            ax.text(
                0.98,
                0.02,
                "no val-loss history",
                transform=ax.transAxes,
                ha="right",
                fontsize=8,
                color="grey",
            )
        if aps:
            best = int(np.argmax(aps))
            ax.axvline(
                best,
                color="#55a868",
                ls=":",
                lw=1.2,
                label=f"best dev AP {aps[best]:.3f}",
            )
        auc = r.get("auc", float("nan"))
        ax.set_title(f"fold {r['fold']}  (test AUC {auc:.3f})", fontsize=10)
        ax.set_xlabel("logged step")
        ax.set_ylabel(f"{SSOT_LOSS} loss" if i == 0 else "")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(
        f"training vs validation loss — {args.model} ({args.split}/{args.payload})",
        fontsize=11,
    )
    fig.tight_layout()
    out_png = RESULTS / f"training_loss_{args.split}_payload-{args.payload}.png"
    fig.savefig(out_png, dpi=plot_dpi())
    plt.close(fig)
    print(f"[plot] {out_png}", flush=True)
