"""TRAIN_GPU/run_all.py — the owner's GPU training plan, in order.

  STEP 1  embeddings   full-corpus embeddings per model, saved LOCALLY as
                       npz (titles + vectors) for further analysis
  STEP 2  sweep-2k     full sweep pass on ONE model, 2,000-row sample
                       (smoke-scale full chain: holdout 50/25/25, hard-negs,
                       mining, MNRL, early stop, plot)
  STEP 3  sweep-full   the same pass on FULL data
  STEP 4  ablation     the 07-series mirrors on ALL 3 models:
                       payload variants (07c), train-frac curve (07d),
                       rerank (07e), plots (07f)

Steps run in order; --only N runs one step; --from N starts mid-way.
Everything logs to TRAIN_GPU/logs/<step>.log AND appends to
TRAIN_GPU/train_manifest.csv (one row per invocation).
Every subprocess is resumable: rerunning a step appends, never destroys.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
from lib.common import _path, load_config

_cfg = load_config()
MODELS = dict(_cfg["models"])
PY = sys.executable

LOGS = _path(_cfg["paths"]["logs_dir"])
LOGS.mkdir(parents=True, exist_ok=True)
EMB_OUT = _path(_cfg["paths"]["embeddings_dir"])

_model_dirs = [
    _path(_cfg["paths"]["models_dir"]),
    _path(_cfg["paths"]["models_dir_sibling"]),
]


def _resolve_model(sub: str) -> str:
    """Local models/ dir first (config models_dir / models_dir_sibling),
    else the hub id (last resort — offline GPU runs should not hit this)."""
    for d in _model_dirs:
        cand = d / sub
        if cand.exists():
            return str(cand.resolve())
    if "deberta" in sub:
        return "microsoft/deberta-v3-base"
    return f"sentence-transformers/{sub}"


def _sh(cmd: list[str], log: Path) -> None:
    """Run a subprocess, streaming output to BOTH the log file and stdout."""
    print(f"[cmd] {' '.join(cmd)}", flush=True)
    with log.open("w") as fh:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(HERE),
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
            print(line, end="", flush=True)
        rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"[fail] rc={rc} — see {log}")


def step1_embeddings() -> None:
    """Full-corpus embeddings per model → TRAIN_GPU/embeddings/<model>.npz."""
    import numpy as np

    from lib.common import load_dataset_deduped
    from lib.nlp import encode_corpus

    df = load_dataset_deduped()
    payload = (
        df["title"].fillna("")
        + " | "
        + df["brand"].fillna("")
        + " | "
        + df["category"].fillna("")
    ).tolist()
    out_dir = EMB_OUT
    out_dir.mkdir(exist_ok=True)
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for key, sub in MODELS.items():
        model_id = _resolve_model(sub)
        npz = out_dir / f"{key}.npz"
        if npz.exists():
            print(f"[step1] {key}: {npz.name} exists — skip", flush=True)
            continue
        emb, encode_s = encode_corpus(
            model_id, payload, batch_size=256, max_seq_length=128, device=device
        )
        np.savez_compressed(
            npz, titles=np.array(payload), embeddings=emb, model=model_id
        )
        print(
            f"[step1] {key}: {len(payload):,} rows → {npz.name} ({encode_s:.0f}s)",
            flush=True,
        )


def step2_sweep_2k() -> None:
    """Full-chain sweep on ONE model (default L12 multilingual) at 2k rows."""
    model = _resolve_model(MODELS["multilingual_l12"])
    _sh(
        [
            PY,
            "TRAIN/05_train.py",
            "--model",
            model,
            "--sample",
            "2000",
            "--epochs",
            "3",
            "--split",
            "holdout",
            "--plot",
        ],
        LOGS / "step2_sweep_2k.log",
    )


def step3_sweep_full() -> None:
    """The full-data training pass (holdout 50/25/25)."""
    model = _resolve_model(MODELS["multilingual_l12"])
    _sh(
        [
            PY,
            "TRAIN/05_train.py",
            "--model",
            model,
            "--epochs",
            "3",
            "--split",
            "holdout",
            "--plot",
        ],
        LOGS / "step3_sweep_full.log",
    )


def step4_ablation() -> None:
    """07-series mirrors across ALL 3 models.

    Per model: 07c payload variants, 07d train-frac curve, then the
    base full run's --plot. Rerank (07e) runs on the best base model
    afterwards (needs a trained checkpoint).
    """
    for key, sub in MODELS.items():
        model = _resolve_model(sub)
        # 07c: payload variants (skip 'full' — that IS step 3)
        for variant in ("title_only",):
            _sh(
                [
                    PY,
                    "TRAIN/05_train.py",
                    "--model",
                    model,
                    "--epochs",
                    "3",
                    "--split",
                    "holdout",
                    "--payload",
                    variant,
                    "--plot",
                ],
                LOGS / f"step4_{key}_07c_{variant}.log",
            )
        # 07d: train-frac scaling curve
        for frac in ("0.25", "0.50", "0.75"):
            _sh(
                [
                    PY,
                    "TRAIN/05_train.py",
                    "--model",
                    model,
                    "--epochs",
                    "3",
                    "--split",
                    "holdout",
                    "--train-frac",
                    frac,
                    "--plot",
                ],
                LOGS / f"step4_{key}_07d_frac{frac}.log",
            )


STEPS = {
    "1": ("embeddings", step1_embeddings),
    "2": ("sweep-2k", step2_sweep_2k),
    "3": ("sweep-full", step3_sweep_full),
    "4": ("ablation", step4_ablation),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", type=str, default=None, help="run ONE step (1-4)")
    ap.add_argument(
        "--from",
        dest="start",
        type=str,
        default="1",
        help="start at step N (default 1)",
    )
    args = ap.parse_args()
    LOGS.mkdir(exist_ok=True)
    manifest = HERE / "train_manifest.csv"
    order = sorted(STEPS) if args.only is None else [args.only]
    start = args.start if args.start in STEPS else "1"
    for n in order:
        if int(n) < int(start) and args.only is None:
            continue
        name, fn = STEPS[n]
        t0 = time.time()
        print(f"\n{'=' * 70}\nSTEP {n}: {name}\n{'=' * 70}", flush=True)
        try:
            fn()
            status = "ok"
        except SystemExit as e:
            status = f"failed: {e}"
        with manifest.open("a") as fh:
            fh.write(f'"{n}","{name}","{status}",{time.time() - t0:.0f}\n')


if __name__ == "__main__":
    main()
