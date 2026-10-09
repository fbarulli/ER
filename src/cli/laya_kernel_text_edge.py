"""Embedded kernel-script text for the DECISION / EVAL / NOTEBOOK / HOLDOUT payloads.

These triple-quoted constants are the in-source bodies the laya lane token-
substitutes and stages as kaggle kernel scripts. They are pure text (no module
dependencies), kept apart from the orchestration in ``cli.laya_lane``.
"""
from __future__ import annotations


# The laya payloads ride the ATTACHED er-laya-requests dataset, never a
# repo clone: the shipped core.runtime_inputs.checkout_preflight_script
# emits `_runtime_root = Path(root)` for clone lanes and a laya payload
# defines no root (NameError at line 36 killed the first remote boot).
# This dedicated template instead verifies THE ATTACHED INPUTS: files
# land under /kaggle/input/<slug>/ and resolve_input rglob's by name.
# REPOSITORY/BRANCH/REVISION/_runtime_files stay assigned at top level
# so the laya push gate still literal-evals them (_staged_laya_push_
# preflight — the push gate lives in this lane; the shared clone-lane
# staged_kernel_preflight checked the git tree for the attached-inputs
# inventory, which never matches a dataset-carried payload).
LAYA_RUNTIME_PREFLIGHT = '''\
_runtime_files = ("@DECISION_CSV@", @QUESTION_SCHEMA_FILE@)
INPUT_ROOT = Path("/kaggle/input")


def laya_runtime_preflight():
    """Verify the ATTACHED dataset inputs (the er-laya-requests dataset
    mounts under /kaggle/input/<slug>/ and resolve_input searches INPUTS
    recursively by name); fail loud before pip touches anything."""
    missing = [name for name in _runtime_files
               if not any(INPUT_ROOT.rglob(name))]
    if missing:
        raise FileNotFoundError(
            "Runtime preflight missing attached inputs: "
            + ", ".join(missing))
    print("[runtime-preflight] verified %d required files"
          % len(_runtime_files), flush=True)


laya_runtime_preflight()
'''


# The holdout kernel attaches the staged holdout dataset and verifies the ONE
# `holdout.jsonl` (rglob finds it under /kaggle/input/<slug>/). The questions
# schema is embedded PER ROW in that JSONL, so there is no separate schema file
# to attach — unlike the DECISION kernel's inventory (decision CSV + schema).
HOLDOUT_RUNTIME_PREFLIGHT = '''\
_runtime_files = ("@HOLDOUT_JSONL@",)
INPUT_ROOT = Path("/kaggle/input")


def laya_runtime_preflight():
    """Verify the ATTACHED holdout JSONL (the holdout dataset mounts under
    /kaggle/input/<slug>/ and rglob searches recursively by name); fail loud
    before pip touches anything."""
    missing = [name for name in _runtime_files
               if not any(INPUT_ROOT.rglob(name))]
    if missing:
        raise FileNotFoundError(
            "Runtime preflight missing attached inputs: "
            + ", ".join(missing))
    print("[runtime-preflight] verified %d required files"
          % len(_runtime_files), flush=True)


laya_runtime_preflight()
'''


DECISION_KERNEL_SCRIPT = '''\
"""ER typed decisions via laya on a Kaggle GPU session (cli.laya_lane).

Single T4 per owner ruling (2xT4 -> 1xT4; never requests the double
accelerator): pins one CUDA device, installs laya over pip, reads the
question schema + decision CSV attached as the kaggle dataset inputs,
runs the router's typed questions per row, and writes the decision
results + receipt into /kaggle/working for hash-verified fetch-back.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

LAYA_PACKAGE = "@LAYA_PACKAGE@"
CHECKPOINT_HUB = "@CHECKPOINT_HUB@"
DECISION_KIND = "@DECISION_KIND@"
RUN_TAG = "@RUN_TAG@"
DECISION_CSV = "@DECISION_CSV@"
STATE_COLUMN = "@STATE_COLUMN@"
BATCH_SIZE = @BATCH_SIZE@
MIN_CONFIDENCE = @MIN_CONFIDENCE@
QUESTION_SCHEMA_FILE = @QUESTION_SCHEMA_FILE@

REPOSITORY = "@REPOSITORY@"
BRANCH = "@BRANCH@"
REVISION = "@REVISION@"
@RUNTIME_PREFLIGHT@

WORKING = Path("/kaggle/working")
INPUTS = Path("/kaggle/input")


def log(line):
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print("[laya-lane " + stamp + "] " + line, flush=True)


def pip_upgrade_laya():
    """laya installs over pip; torch must already be 2.14 cu13x."""
    command = [sys.executable, "-m", "pip", "install", "-q", "--no-input",
               "--disable-pip-version-check", LAYA_PACKAGE]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def pick_device():
    """SINGLE T4 ruling: pin the FIRST cuda device only (never 2xT4)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("cuda unavailable: the session is not a T4")
    log("device pinned: " + torch.cuda.get_device_name(0)
        + " (single GPU, never a second one)")
    return "cuda"


def resolve_input(name):
    for candidate in sorted(INPUTS.rglob(name)):
        return candidate
    raise FileNotFoundError(
        "attached inputs carried no " + name + " (expected the staged "
        "laya_lane payload dataset)")


def predict_batch(agent, states, questions):
    """One batched call per chunk; min_confidence only when gated."""
    if MIN_CONFIDENCE > 0:
        return agent.predict_batch(states, questions,
                                   min_confidence=MIN_CONFIDENCE)
    return agent.predict_batch(states, questions)


def main():
    pip_upgrade_laya()
    device = pick_device()
    import laya
    questions_path = resolve_input(QUESTION_SCHEMA_FILE)
    questions = json.loads(questions_path.read_text())["questions"]
    decision_csv = resolve_input(DECISION_CSV)
    agent = laya.load(CHECKPOINT_HUB, device=device)
    rows = []
    with decision_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    log("loaded " + str(len(rows)) + " " + DECISION_KIND
        + " state rows from " + decision_csv.name)
    results = []
    for start in range(0, len(rows), BATCH_SIZE):
        chunk = rows[start:start + BATCH_SIZE]
        states = [row.get(STATE_COLUMN, "") for row in chunk]
        answers = predict_batch(agent, states, questions)
        for row, answer in zip(chunk, answers):
            answer["_row"] = {key: value for key, value in row.items()
                              if key != STATE_COLUMN}
            results.append(answer)
        log("decided " + str(min(start + BATCH_SIZE, len(rows))) + "/"
            + str(len(rows)) + " rows")
    WORKING.mkdir(parents=True, exist_ok=True)
    out = WORKING / (DECISION_KIND + ".decisions.jsonl")
    with out.open("w", encoding="utf-8") as handle:
        handle.write("".join(json.dumps(item) + "\\n" for item in results))
    log("wrote " + str(out) + " (" + str(len(results)) + " decisions)")
    receipt = {
        "gpu_kind": DECISION_KIND,
        "gpu": "T4 (single)",
        "run_tag": RUN_TAG,
        "laya_package": LAYA_PACKAGE,
        "checkpoint_hub": CHECKPOINT_HUB,
        "batch_size": BATCH_SIZE,
        "min_confidence": MIN_CONFIDENCE,
        "question_schema_sha256": hashlib.sha256(
            questions_path.read_bytes()).hexdigest(),
        "decision_csv_sha256": hashlib.sha256(
            decision_csv.read_bytes()).hexdigest(),
    }
    (WORKING / "laya_decision.receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\\n", encoding="utf-8")
    with tarfile.open(WORKING / "laya_decision.tar.gz", "w:gz",
                      compresslevel=1) as tar:
        for item in sorted(WORKING.iterdir()):
            if item.name != "laya_decision.tar.gz":
                tar.add(item, arcname=item.name)
    log("staged laya_decision.tar.gz + receipt in /kaggle/working")


if __name__ == "__main__":
    main()
'''


EVAL_KERNEL_SCRIPT = '''\
"""laya-evals harness score on a Kaggle GPU session (cli.laya_lane).

Single T4 per owner ruling; installs laya over pip, derives the laya-evals
JSONL (state, questions, expected) from the staged identity decision CSV +
question schema, runs `laya-evals run`, and stages the report.json +
report.md + receipt into /kaggle/working.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

LAYA_PACKAGE = "@LAYA_PACKAGE@"
RUN_TAG = "@RUN_TAG@"
DECISION_CSV = "@DECISION_CSV@"
QUESTION_SCHEMA_FILE = @QUESTION_SCHEMA_FILE@

REPOSITORY = "@REPOSITORY@"
BRANCH = "@BRANCH@"
REVISION = "@REVISION@"
@RUNTIME_PREFLIGHT@

WORKING = Path("/kaggle/working")
INPUTS = Path("/kaggle/input")


def log(line):
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print("[laya-lane " + stamp + "] " + line, flush=True)


def resolve_input(name):
    for candidate in sorted(INPUTS.rglob(name)):
        return candidate
    raise FileNotFoundError(
        "attached inputs carried no " + name
        + " (expected the staged laya_lane payload dataset)")


def main():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "--no-input", "--disable-pip-version-check",
                    LAYA_PACKAGE], check=True)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    questions_path = resolve_input(QUESTION_SCHEMA_FILE)
    questions = json.loads(questions_path.read_text())["questions"]
    decision_csv = resolve_input(DECISION_CSV)
    WORKING.mkdir(parents=True, exist_ok=True)
    dataset_jsonl = WORKING / "identity_evals.jsonl"
    with decision_csv.open(newline="") as handle, dataset_jsonl.open(
            "w", encoding="utf-8") as out:
        for row in csv.DictReader(handle):
            expected = int(row["true_label"])
            out.write(json.dumps({
                "state": row.get("attribute_pairs", ""),
                "questions": questions,
                "expected": {"identity_claim": expected},
            }) + "\\n")
    log("staged " + str(dataset_jsonl) + " from " + decision_csv.name)
    command = [sys.executable, "-m", "laya.evals", "run",
               str(dataset_jsonl), "--json", str(WORKING / "report.json")]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)
    receipt = {
        "gpu_kind": "laya-cli-eval",
        "gpu": "T4 (single)",
        "run_tag": RUN_TAG,
        "laya_package": LAYA_PACKAGE,
        "question_schema_sha256": hashlib.sha256(
            questions_path.read_bytes()).hexdigest(),
        "evals_dataset_sha256": hashlib.sha256(
            dataset_jsonl.read_bytes()).hexdigest(),
    }
    (WORKING / "laya_evals.receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\\n", encoding="utf-8")
    log("staged laya-evals report + receipt in /kaggle/working")


if __name__ == "__main__":
    main()
'''


NOTEBOOK_SCRIPT = '''\
"""Laya typed-decision Colab payload (cli.laya_lane; delivery contract).

BOOK-END CONTRACT ONLY — no cli.colab import, NO session call. Operator
instruction set for the notebook wrapper:
  1. pip install laya (torch stack per laya's docs; python >= 3.10);
  2. upload the staged laya.question.json + the staged decision CSV as
     the notebook's own payload mounts;
  3. run the decision loop on a SINGLE GPU (never the double accelerator);
  4. write the receipt json + the export tar under the notebook's own
     export mount, then read it back into results/laya_lane/colab/<op>/.
"""

from pathlib import Path

LAYA_PACKAGE = "@LAYA_PACKAGE@"
DECISION_KIND = "@DECISION_KIND@"
RUN_TAG = "@RUN_TAG@"
DECISION_CSV = "@DECISION_CSV@"
STATE_COLUMN = "@STATE_COLUMN@"

WORKING = Path("/content/laya_out")


def main() -> None:
    print("[laya-lane] colab payload staged; delivery contract only",
          flush=True)
    WORKING.mkdir(parents=True, exist_ok=True)
    print("[laya-lane] decision kind: " + DECISION_KIND
          + " batch: " + str(len(list(WORKING.iterdir()))), flush=True)


if __name__ == "__main__":
    main()
'''


HOLDOUT_EVAL_KERNEL_SCRIPT = '''\
"""ER laya holdout verification on Kaggle (cli.laya_lane).

Single T4: installs laya over pip, attaches the staged component-disjoint
holdout (JSONL of composed identity states + labels + strata) and the fine-tuned
checkpoint dataset, scores each pair's identity_claim with the checkpoint, and
writes holdout_report.json: overall + per-stratum precision/recall/F1/PR-AUC at
the configured threshold, each with a COMPONENT-clustered bootstrap CI (real
held-out verification, never the in-sample training eval). NO training, NO Hub.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

LAYA_PACKAGE = "@LAYA_PACKAGE@"
RUN_TAG = "@RUN_TAG@"
HOLDOUT_JSONL = "@HOLDOUT_JSONL@"
CKPT_DIR_HINT = "@CKPT_DIR@"
BATCH_SIZE = @BATCH_SIZE@
THRESHOLD = @THRESHOLD@
N_BOOT = @N_BOOT@
SEED = @SEED@

REPOSITORY = "@REPOSITORY@"
BRANCH = "@BRANCH@"
REVISION = "@REVISION@"
@RUNTIME_PREFLIGHT@

WORKING = Path("/kaggle/working")
INPUTS = Path("/kaggle/input")


def log(line):
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print("[laya-lane " + stamp + "] " + line, flush=True)


def pip_install_laya():
    command = [sys.executable, "-m", "pip", "install", "-q", "--no-input",
               "--disable-pip-version-check", LAYA_PACKAGE]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def pick_device():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("cuda unavailable: the session is not a T4")
    log("device pinned: " + torch.cuda.get_device_name(0))
    return "cuda"


def resolve_input(name):
    for candidate in sorted(INPUTS.rglob(name)):
        return candidate
    raise FileNotFoundError("attached inputs carried no " + name)


def resolve_checkpoint():
    if CKPT_DIR_HINT:
        for found in sorted(INPUTS.rglob(CKPT_DIR_HINT)):
            if found.is_dir() and (found / "rl_agent_config.json").is_file():
                return found
    for found in sorted(INPUTS.rglob("rl_agent_config.json")):
        return found.parent
    raise FileNotFoundError("attached inputs carried no checkpoint")


def sha256_of(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def binary_metrics(labels, scores, threshold):
    tp = fp = tn = fn = 0
    for y, s in zip(labels, scores):
        pred = 1 if s >= threshold else 0
        if pred and y:
            tp += 1
        elif pred and not y:
            fp += 1
        elif not pred and y:
            fn += 1
        else:
            tn += 1
    def pr(tp, fp):
        return tp / (tp + fp) if (tp + fp) else 0.0
    def rc(tp, fn):
        return tp / (tp + fn) if (tp + fn) else 0.0
    precision, recall = pr(tp, fp), rc(tp, fn)
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    n = tp + fp + tn + fn
    return {"n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": (tp + tn) / n if n else None,
            "precision": precision, "recall": recall, "f1": f1}


def pr_auc(labels, scores):
    import numpy as np
    labels = np.asarray(labels)
    if len(set(labels.tolist())) < 2:
        return None
    order = np.argsort(-np.asarray(scores, dtype=float))
    hits = labels[order]
    cum = np.cumsum(hits)
    precision = cum / (np.arange(len(hits)) + 1)
    return float((precision * hits).sum() / hits.sum())


def bootstrap_ci(labels, scores, components, stat, n_boot, seed, alpha=0.05):
    import numpy as np
    components = np.asarray(components, dtype=object)
    labels, scores = np.asarray(labels), np.asarray(scores)
    unique = np.unique(components)
    rows_of = {c: np.where(components == c)[0] for c in unique}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(int(n_boot)):
        picks = rng.choice(unique, size=unique.size, replace=True)
        idx = np.concatenate([rows_of[c] for c in picks])
        value = stat(labels[idx], scores[idx])
        if value is not None:
            samples.append(float(value))
    point = stat(labels, scores)
    if not samples:
        return {"point": point, "lo": None, "hi": None}
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "lo": float(lo), "hi": float(hi)}


def block(labels, scores, components, threshold):
    m = binary_metrics(labels, scores, threshold)
    def _f1(y, s):
        return binary_metrics(y, s, threshold)["f1"]
    def _precision(y, s):
        return binary_metrics(y, s, threshold)["precision"]
    def _recall(y, s):
        return binary_metrics(y, s, threshold)["recall"]
    return {
        "metrics": m,
        "pr_auc": pr_auc(labels, scores),
        "cis": {
            "precision": bootstrap_ci(labels, scores, components, _precision, N_BOOT, SEED),
            "recall": bootstrap_ci(labels, scores, components, _recall, N_BOOT, SEED),
            "f1": bootstrap_ci(labels, scores, components, _f1, N_BOOT, SEED),
            "pr_auc": bootstrap_ci(labels, scores, components, pr_auc, N_BOOT, SEED),
        },
    }


def main():
    pip_install_laya()
    device = pick_device()
    import numpy as np
    import torch
    from laya import train as laya_train
    holdout = resolve_input(HOLDOUT_JSONL)
    checkpoint = resolve_checkpoint()
    log("holdout: " + str(holdout))
    log("checkpoint: " + str(checkpoint))
    rows = [json.loads(line) for line in holdout.read_text().splitlines()
            if line.strip()]
    model, tok, cfg = laya_train.load_checkpoint(str(checkpoint))
    model = model.to(torch.device(device)).eval()
    max_len = int(cfg.get("max_len", 512))
    head_max_len = int(cfg.get("head_max_len", 192))
    parallel = laya_train.uses_parallel_layout(cfg)
    items, skipped = laya_train.items_from_rows(
        tok, rows, max_len, head_max_len, label_smoothing=0.0)
    if len(items) != len(rows):
        raise SystemExit("holdout items %d != rows %d (skipped %s)"
                         % (len(items), len(rows), repr(skipped)))
    records = laya_train.calibration_records(
        model, tok, items, device, max_len, head_max_len,
        batch_size=BATCH_SIZE, parallel=parallel)
    scores = []
    for _qt, logits, _t, _k in records:
        z = np.asarray(logits, dtype=float)
        z = z - z.max()
        p = np.exp(z)
        p = p / p.sum()
        scores.append(float(p[1]))
    labels = [1 if str(r.get("expected", {}).get("identity_claim")).lower()
              in ("true", "1") else 0 for r in rows]
    strata = [str(r.get("stratum") or "unknown") for r in rows]
    components = [str(r.get("component") or r["state"][:24]) for r in rows]
    by_stratum = {}
    for name in sorted(set(strata)):
        idx = [i for i, s in enumerate(strata) if s == name]
        by_stratum[name] = block([labels[i] for i in idx],
                                 [scores[i] for i in idx],
                                 [components[i] for i in idx], THRESHOLD)
    report = {
        "eval_mode": "holdout",
        "is_held_out": True,
        "checkpoint": str(checkpoint),
        "run_tag": RUN_TAG,
        "rows": len(rows),
        "skipped": skipped,
        "threshold": THRESHOLD,
        "n_boot": N_BOOT,
        "overall": block(labels, scores, components, THRESHOLD),
        "by_stratum": by_stratum,
    }
    WORKING.mkdir(parents=True, exist_ok=True)
    out = WORKING / "holdout_report.json"
    out.write_text(json.dumps(report, indent=2) + "\\n", encoding="utf-8")
    receipt = {
        "gpu_kind": "holdout-eval",
        "run_tag": RUN_TAG,
        "device": device,
        "checkpoint": str(checkpoint),
        "holdout": str(holdout),
        "holdout_sha256": sha256_of(holdout),
        "rows": len(rows),
        "overall_accuracy": report["overall"]["metrics"]["accuracy"],
        "overall_f1": report["overall"]["metrics"]["f1"],
        "report": str(out),
    }
    (WORKING / "laya_holdout-eval.receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\\n", encoding="utf-8")
    with tarfile.open(WORKING / "laya_holdout-eval.tar.gz", "w:gz",
                      compresslevel=1) as tar:
        for item in sorted(WORKING.iterdir()):
            if item.name != "laya_holdout-eval.tar.gz":
                tar.add(item, arcname=item.name)
    log("overall accuracy=" + str(report["overall"]["metrics"]["accuracy"])
        + " f1=" + str(report["overall"]["metrics"]["f1"]))
    print("[holdout-eval] report: " + json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
'''
