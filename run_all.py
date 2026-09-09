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
Everything logs to <paths.logs_dir>/<step>.log AND appends to
TRAIN_GPU/train_manifest.csv (one row per invocation).
Every subprocess is resumable: rerunning a step appends, never destroys
(step logs append too, with a run separator — audit round 2 F16).
--stop-on-fail halts the chain at the first failed step (default: record
and continue, the historical resumable behavior).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
from lib.common import _path, load_config, resolve_model, sweep_cfg

_cfg = load_config()
MODELS = dict(_cfg["models"])
PY = sys.executable

LOGS = _path(_cfg["paths"]["logs_dir"])
LOGS.mkdir(parents=True, exist_ok=True)
EMB_OUT = _path(_cfg["paths"]["embeddings_dir"])

# _resolve_model moved to lib.common.resolve_model (2026-09-08): run_all
# now resolves model ids through ONE registry-aware helper
# instead of per-file copies — local bundle dir first, hub id fallback.


def _sh(cmd: list[str], log: Path) -> None:
    """Run a subprocess, streaming output to BOTH the log file and stdout.

    AUDIT FIX (round 2 F16a, round 3): the log opens in APPEND mode with a
    run-separator line — the old "w" truncated the previous run's log on a
    re-run while the docstring sold resumability (artifacts were append-
    safe; logs were not). Same cwd/echo behavior otherwise.
    """
    print(f"[cmd] {' '.join(cmd)}", flush=True)
    with log.open("a") as fh:
        fh.write(f"\n{'=' * 70}\n[rerun {time.strftime('%Y-%m-%d %H:%M:%S')}] {' '.join(cmd)}\n{'=' * 70}\n")
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
    """Full-corpus embeddings per model → artifacts/embeddings/<model>.npz.

    NOTE (audit 2026-09-07): these npz dumps have ZERO downstream consumers
    — every later step (TRAIN/train, report_plots) encodes through lib.nlp's
    payload-keyed cache instead. The dump is kept as a standalone analysis
    artifact for the deliverable notebook (vectors + titles in one file,
    loadable without re-encoding); it is NOT part of any step's input. The
    raw `title | brand | category` payload here is the ANALYSIS payload and
    deliberately differs from the training payload — do not "unify" them
    without checking cache-key semantics first.
    """
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
    from lib.common import runtime as _runtime

    for key, sub in MODELS.items():
        model_id = resolve_model(sub)
        npz = out_dir / f"{key}.npz"
        if npz.exists():
            print(f"[step1] {key}: {npz.name} exists — skip", flush=True)
            continue
        emb, encode_s = encode_corpus(
            model_id,
            payload,
            batch_size=_runtime("batch_size_embed"),
            max_seq_length=_runtime("max_seq_length"),
            device=device,
        )
        np.savez_compressed(
            npz, titles=np.array(payload), embeddings=emb, model=model_id
        )
        print(
            f"[step1] {key}: {len(payload):,} rows → {npz.name} ({encode_s:.0f}s)",
            flush=True,
        )


def step2_sweep_2k() -> None:
    """Full-chain sweep on ONE model (default L12 multilingual) at the
    config sweep-sample size."""
    # AUDIT 2026-09-09: no --epochs literal — the SSOT (training.epochs = 10)
    # is train.py's argparse default now; a hardcoded 3 here silently
    # diverged from the config on every full run. The sample size is the
    # SSOT sweep.sweep_sample (was inline 2000).
    model = resolve_model(MODELS["multilingual_l12"])
    _sh(
        [
            PY,
            "TRAIN/train.py",
            "--model",
            model,
            "--sample",
            str(sweep_cfg()["sweep_sample"]),
            "--split",
            "holdout",
            "--plot",
        ],
        LOGS / "step2_sweep_2k.log",
    )


def step3_sweep_full() -> None:
    """The full-data training pass (holdout 50/25/25)."""
    model = resolve_model(MODELS["multilingual_l12"])
    _sh(
        [
            PY,
            "TRAIN/train.py",
            "--model",
            model,
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
    _sw = sweep_cfg()
    for key, sub in MODELS.items():
        model = resolve_model(sub)
        # 07c: payload variants (skip 'full' — that IS step 3) — SSOT
        # sweep.payload_variants (was inline ("title_only",))
        for variant in _sw["payload_variants"]:
            _sh(
                [
                    PY,
                    "TRAIN/train.py",
                    "--model",
                    model,
                    "--split",
                    "holdout",
                    "--payload",
                    variant,
                    "--plot",
                ],
                LOGS / f"step4_{key}_07c_{variant}.log",
            )
        # 07d: train-frac scaling curve — SSOT sweep.train_fracs
        # (was inline ("0.25", "0.50", "0.75"))
        for frac in (f"{f:g}" for f in _sw["train_fracs"]):
            _sh(
                [
                    PY,
                    "TRAIN/train.py",
                    "--model",
                    model,
                    "--split",
                    "holdout",
                    "--train-frac",
                    frac,
                    "--plot",
                ],
                LOGS / f"step4_{key}_07d_frac{frac}.log",
            )
    # 07e rerank + 07b four-population CSV: the docstring promised this
    # "runs on the best base model afterwards (needs a trained checkpoint)"
    # but the invocation was never wired — 07b_four_pop_scores.csv had no
    # producer in this repo until now. Runs on the L12 base (the lane's
    # default trainer) after step3's checkpoint exists.
    model = resolve_model(MODELS["multilingual_l12"])
    _sh(
        [
            PY,
            "TRAIN/train.py",
            "--model",
            model,
            "--split",
            "holdout",
            "--rerank",
            _sw["rerank_model"],  # SSOT sweep.rerank_model
        ],
        LOGS / "step4_07e_rerank.log",
    )
    _sh([PY, "TRAIN/report_plots.py"], LOGS / "step4_07f_plots.log")


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
    # AUDIT FIX (round 2 F16b, round 3): --stop-on-fail halts the chain on
    # the first failed step. Default False preserves the historical
    # resumable-orchestrator behavior (record failed, continue) — but a
    # failed data_prep meant downstream steps burned GPU on stale inputs,
    # so the strict mode is now one flag away.
    ap.add_argument(
        "--stop-on-fail",
        action="store_true",
        help="stop the chain at the first failed step (default: record and continue)",
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
        finally:
            with manifest.open("a") as fh:
                fh.write(f'"{n}","{name}","{status}",{time.time() - t0:.0f}\n')
        if status != "ok" and args.stop_on_fail:
            print(
                f"\n[stop-on-fail] step {n} ({name}) failed — halting the chain "
                f"(remaining steps skipped).",
                flush=True,
            )
            raise SystemExit(1)


if __name__ == "__main__":
    main()
