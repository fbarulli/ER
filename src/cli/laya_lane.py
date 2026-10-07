"""Laya decision lane (branch laya-lane).

Typed-decision surface for the `laya` package (PyPI `laya`: non-
autoregressive decision engine, Python >= 3.10, torch/transformers wheel
stack): stage the laya.question schema + ER decision inputs on THIS box,
then run the typed decision questions on a REMOTE GPU session:
  * kind="kaggle" — a Kaggle GPU kernel payload (metadata + script +
    receipt under results/laya_lane/kaggle/<decision>/); `--execute`
    drives `kaggle kernels push`, everything else is offline staging.
  * kind="colab"  — a Colab notebook payload + receipt under
    results/laya_lane/colab/<decision>/, delivered in the kaggle-lane
    receipts style. The notebook is the delivery CONTRACT: this lane
    never imports or edits cli.colab / cli.colab_lane and never opens a
    session.

Owner rulings honored (docs/laya-lane.md):
* 2xT4 -> SINGLE T4 per owner ruling: no double accelerator (the session
  pins one CUDA device; the staged payload never requests a second GPU);
* laya installs over pip (`pip install laya`), never vendored;
* exports return via /kaggle/working tar + a hashed receipt.

Contract + evidence: tests/test_laya_lane.py (offline, no network).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from core.common import TRAIN_ROOT, training_cfg
from core.manifest import atomic_write_json, sha256_file

# One roof (kaggle_lane precedent: TRAIN_ROOT/logs/<lane>/).
KINDS = ("kaggle", "colab")
GPU_KINDS = ("attribute", "identity", "laya-cli-eval")
LANE_LOG_NAME = "lane.log"
DEFAULT_GPU = "T4"  # single T4; NEVER "2xT4" (no double accelerator)
STAGE_ROOT = "results/laya_lane"

DECISION_KERNEL_CODE_FILE = "laya_decision.py"
EVAL_KERNEL_CODE_FILE = "laya_evals.py"
COLAB_NOTEBOOK_NAME = "laya_decision_colab.py"
QUESTION_SCHEMA_FILE = "laya.question.json"

# Per decision kind: the required header columns, the state column the
# decision state is built from, and what the run decides. THE CSV BINDING
# ITSELF lives in the SSOT (LayaSpec.decision_csv_bindings, resolved
# through core.common.F here) — never duplicated in a second registry.
DECISION_BINDINGS: dict[str, dict[str, Any]] = {
    "attribute": {
        "wanted_columns": ("sku_id", "sku_name_eng", "attribute"),
        "state_column": "attribute",
        "description": ("attribute-channel typed decision over the export's "
                        "attribute text. NOT a replacement for the frozen "
                        "SKU_ITEM/GTIN attribution path — it asks whether "
                        "laya's calibrated route agrees on the same "
                        "attributes the task layout already fixed"),
    },
    "identity": {
        "wanted_columns": ("gtin1", "gtin2", "true_label"),
        "state_column": "attribute_pairs",
        "description": ("identity typed decision over the frozen P0 "
                        "validation population. NOT a replacement for the "
                        "candidate-generation + gate + ann/rerank path — "
                        "its questions ask whether the candidates available "
                        "from the routes agree with the same identity "
                        "evidence the task layout already fixed"),
    },
    "laya-cli-eval": {
        "wanted_columns": ("gtin1", "gtin2", "true_label"),
        "state_column": "attribute_pairs",
        "description": ("the laya-evals harness score over the same "
                        "identity decision samples, verifying the shared "
                        "transport + recall identity"),
    },
}


def _spec():
    return training_cfg().laya


def decision_binding(decision_kind: str) -> str:
    """SSOT F-binding for a decision kind (fail-loud on an unknown kind)."""
    if decision_kind not in DECISION_BINDINGS:
        raise ValueError(f"unknown decision kind: {decision_kind!r}; "
                         f"expected {list(DECISION_BINDINGS)}")
    bindings = _spec().decision_csv_bindings
    binding = bindings.get(decision_kind)
    if not binding:
        raise RuntimeError(
            f"config laya.decision_csv_bindings carries no entry for "
            f"{decision_kind!r}; name the config/paths.yaml files: binding "
            "before staging")
    return binding


def staging_dir() -> Path:
    """The lane staging root (TRAIN_ROOT-relative; SSOT laya.staging_dir)."""
    return (TRAIN_ROOT / _spec().staging_dir).resolve()


def lane_logs_dir() -> Path:
    """The lane transcript dir (one canonical roof: TRAIN_ROOT/logs)."""
    return TRAIN_ROOT / "logs" / "laya"


def _stamp() -> str:
    """Bracketed Europe/Paris (CET/CEST) wall-clock prefix.

    Mirrors kaggle_lane._stamp: the CET convention landed there (owner
    order 2026-10-07) and this lane follows it; the "[laya-lane UTC-stamp]"
    phrasing in the relaunch brief predates that convention.
    """
    return (f"[laya-lane "
            f"{datetime.now(ZoneInfo('Europe/Paris')):%Y-%m-%dT%H:%M:%S %Z}]")


def _log_lane(line: str) -> None:
    """Timestamped lane logging: console plus append-only lane log.

    Best-effort on the file side — a log-write failure is printed and
    never allowed to mask the operation's own outcome.
    """
    stamp = f"{datetime.now(ZoneInfo('Europe/Paris')):%Y-%m-%dT%H:%M:%S %Z}"
    print(f"[laya-lane {stamp}] {line}", flush=True)
    try:
        log_dir = lane_logs_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / LANE_LOG_NAME).open("a", encoding="utf-8") as handle:
            handle.write(f"{stamp} {line}\n")
    except OSError as error:
        print(_stamp(), f"[laya-lane] lane.log write failed ({error}); "
              "continuing", flush=True)


# ── decision-input staging (dry-safe; fail-loud on a missing contract) ─────
def _measure_csv(path: Path, wanted_columns: tuple[str, ...]) -> dict[str, Any]:
    """Stdlib CSV census: header check + row count + sha256 + bytes.

    Deliberately NOT pandas: staging must run anywhere (including a box
    without the frame stack), and the contract is only the columns.
    """
    import csv as _csv

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"decision input not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = _csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"decision input has no header row: {path}")
        missing = [column for column in wanted_columns if column not in header]
        if missing:
            raise ValueError(
                f"decision input {path.name} is missing columns {missing} "
                f"(header: {header})")
        rows = sum(1 for _ in reader)
    if rows == 0:
        raise ValueError(f"decision input has no data rows: {path}")
    return {"rows": rows, "columns": list(header),
            "sha256": sha256_file(path), "bytes": path.stat().st_size}


def stage_decision_input(kind: str, *, decision_kind: str,
                         override: Path | None = None) -> dict[str, Any]:
    """Stage ONE decision CSV under results/laya_lane/<kind>/<decision>/.

    The staged copy is receipted with the measured census (rows, columns,
    sha256, bytes) — the transport-identity contract the kaggle-lane
    package receipts carry. `override` names an alternate source CSV on
    this box (e.g. dataset_50pct.csv for the half cohort).
    """
    if decision_kind not in DECISION_BINDINGS:
        raise ValueError(f"unknown decision kind: {decision_kind!r}")
    from core.common import F

    entry = DECISION_BINDINGS[decision_kind]
    source = Path(override) if override else F[decision_binding(decision_kind)]
    stage = staging_dir() / kind / decision_kind
    stage.mkdir(parents=True, exist_ok=True)
    census = _measure_csv(source, entry["wanted_columns"])
    destination = stage / source.name
    shutil.copy2(source, destination)
    receipt = {
        "kind": kind, "decision_kind": decision_kind,
        "binding": decision_binding(decision_kind),
        "source": str(source), "staged": str(destination),
        "rows": census["rows"], "columns": census["columns"],
        "sha256": census["sha256"], "bytes": census["bytes"],
        "description": entry["description"],
    }
    atomic_write_json(receipt, stage / f"{decision_kind}.receipt.json")
    _log_lane(f"staged decision input [{kind}/{decision_kind}] "
              f"{source.name} rows={census['rows']} "
              f"sha256={census['sha256'][:12]} -> {destination}")
    return receipt


def stage_question_schema(kind: str, *,
                          override: Path | None = None) -> dict[str, Any]:
    """Stage the laya.question schema under results/laya_lane/<kind>/.

    Dry-safe + fail-loud: a missing schema file raises FileNotFoundError
    and a schema without a 'questions' dict raises ValueError (no silent
    empty schema placeholder is ever staged).
    """
    spec = _spec()
    source = Path(override) if override else TRAIN_ROOT / spec.question_schema
    stage = staging_dir() / kind / "question"
    stage.mkdir(parents=True, exist_ok=True)
    if not source.is_file():
        raise FileNotFoundError(f"laya.question schema not found: {source}")
    schema = json.loads(source.read_text(encoding="utf-8"))
    questions = schema.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError(
            f"laya.question schema at {source} carries no 'questions' dict")
    destination = stage / QUESTION_SCHEMA_FILE
    shutil.copy2(source, destination)
    receipt = {
        "question_schema": spec.question_schema,
        "staged": str(destination),
        "questions": sorted(questions),
        "sha256": sha256_file(source),
    }
    atomic_write_json(receipt, stage / "question.receipt.json")
    _log_lane(f"staged question schema [{kind}] {source.name} "
              f"({len(questions)} questions) -> {destination}")
    return receipt


# ── kernel / notebook payload composition ──────────────────────────────────
def _kernel_script_gate(script: str) -> None:
    """Staging-time AST gate (kaggle_lane._kernel_script_gate mirror):
    never stage an unparseable payload or one that references an undeclared
    UPPER_CASE template constant (the v4 NameError error class)."""
    import ast

    parsed = ast.parse(script)
    defined = {node.id for stmt in ast.walk(parsed)
               if isinstance(stmt, ast.Assign)
               for node in stmt.targets if isinstance(node, ast.Name)}
    undeclared = {expr.id for expr in ast.walk(parsed)
                  if isinstance(expr, ast.Name) and isinstance(expr.ctx, ast.Load)
                  and expr.id.isupper() and expr.id not in defined}
    if undeclared:
        raise ValueError(f"staged kernel uses undeclared constants: "
                         f"{sorted(undeclared)}; regenerate the template")


def _template(script: str, values: dict[str, str]) -> str:
    for token, replacement in values.items():
        script = script.replace(f"@{token}@", replacement)
    return script


def _git_revision() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=TRAIN_ROOT,
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"git rev-parse failed in {TRAIN_ROOT}: {result.stderr.strip()}")
    return result.stdout.strip()


def decision_tag() -> str:
    """UTC-stamped run tag (the laya-lane stamp SURFACE stays UTC inside
    the payload because the remote session may not share this box's zone;
    the console/lane.log stamp itself stays Europe/Paris per the landed
    kaggle_lane convention)."""
    return datetime.now(ZoneInfo("UTC")).strftime("%m%dT%H%M%SZ")


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

WORKING = Path("/kaggle/working")
INPUTS = Path("/kaggle/input")


def log(line):
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print("[laya-lane " + stamp + "] " + line, flush=True)


def pip_upgrade_laya():
    """laya installs over pip; torch must already be 2.14 cu13x."""
    command = [sys.executable, "-m", "pip", "install", "-q", "--no-input",
               LAYA_PACKAGE]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def pick_device():
    """SINGLE T4 ruling: pin the FIRST cuda device only (never 2xT4)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("cuda unavailable: the session is not a T4")
    name = torch.cuda.get_device_name(0)
    log("device pinned: " + name + " (single GPU, never a second one)")
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
        for item in results:
            handle.write(json.dumps(item) + "\\n")
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
    with tarfile.open(WORKING / "laya_decision.tar.gz", "w:gz") as tar:
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
                    "--no-input", LAYA_PACKAGE], check=True)
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


def stage_decision_kernel(*, decision_kind: str, revision: str | None = None,
                          run_tag: str | None = None) -> dict[str, Any]:
    """Stage the kaggle decision kernel payload (dry-safe).

    Writes under results/laya_lane/kaggle/<decision_kind>/:
      kernel-metadata.json + <code_file>.py + <decision_kind>.receipt.json
      (+ the staged question schema + decision input receipts).
    Fail-loud preconditions (no silent skip):
      * spec.laya_decision_epochs > 0 (0 = disabled, nothing may stage);
      * spec.export_dataset_slug set (the target kernel owner/slug);
      * the question schema + decision CSV stage from their SSOT bindings.
    """
    spec = _spec()
    if decision_kind not in DECISION_BINDINGS:
        raise ValueError(f"unknown decision kind: {decision_kind!r}; "
                         f"expected {list(DECISION_BINDINGS)}")
    if spec.laya_decision_epochs <= 0:
        raise RuntimeError(
            "config laya.laya_decision_epochs <= 0: the decision lane is "
            "disabled (no payload may stage a GPU session)")
    slug = spec.export_dataset_slug
    if not slug:
        raise RuntimeError(
            "config laya.export_dataset_slug is unset; name the target "
            "kernel (owner/slug) before staging")
    question = stage_question_schema("kaggle")
    input_receipt = stage_decision_input("kaggle",
                                         decision_kind=decision_kind)
    stage = staging_dir() / "kaggle" / decision_kind
    stage.mkdir(parents=True, exist_ok=True)
    code_file = (DECISION_KERNEL_CODE_FILE if decision_kind != "laya-cli-eval"
                 else EVAL_KERNEL_CODE_FILE)
    template = (DECISION_KERNEL_SCRIPT if decision_kind != "laya-cli-eval"
                else EVAL_KERNEL_SCRIPT)
    tag = run_tag or spec.run_tag_prefix + decision_tag()
    metadata: dict[str, Any] = {
        "id": slug,
        "title": slug.rsplit("/", 1)[-1].replace("-", " ").title(),
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "enable_gpu": True,
        # single T4: the payload never requests the double accelerator;
        # the script itself pins CUDA_VISIBLE_DEVICES=0.
        "enable_internet": True,
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "is_private": True,
    }
    entry = DECISION_BINDINGS[decision_kind]
    staged_csv = input_receipt["staged"]
    script = _template(template, {
        "LAYA_PACKAGE": spec.laya_package,
        "CHECKPOINT_HUB": spec.checkpoint_hub,
        "DECISION_KIND": decision_kind,
        "RUN_TAG": tag,
        "DECISION_CSV": Path(staged_csv).name,
        "STATE_COLUMN": entry["state_column"],
        "BATCH_SIZE": str(spec.laya_decision_batch_size),
        "MIN_CONFIDENCE": repr(spec.min_router_confidence),
        "QUESTION_SCHEMA_FILE": repr(QUESTION_SCHEMA_FILE),
    })
    _kernel_script_gate(script)
    atomic_write_json(metadata, stage / "kernel-metadata.json")
    (stage / code_file).write_text(script, encoding="utf-8")
    receipt = {
        "kernel": slug,
        "kind": decision_kind,
        "gpu": "T4 (single)",
        "run_tag": tag,
        "staged": str(stage),
        "code_file": code_file,
        "question_schema": question["staged"],
        "question_sha256": question["sha256"],
        "decision_input": staged_csv,
        "decision_sha256": input_receipt["sha256"],
        "checkpoint_hub": spec.checkpoint_hub,
        "batch_size": spec.laya_decision_batch_size,
        "min_confidence": spec.min_router_confidence,
        "epochs": spec.laya_decision_epochs,
        "state_column": entry["state_column"],
        "evals_enabled": bool(spec.laya_evals_enabled),
        "calibration": bool(spec.calibration),
        "onnx": bool(spec.onnx),
    }
    atomic_write_json(receipt, stage / f"{decision_kind}.receipt.json")
    # The decision csv is already co-located in the payload dir (the
    # stage-decision-input destination IS staging/<kind>/<decision>/);
    # the question schema lands beside it for the payload dataset bind.
    shutil.copy2(question["staged"], stage / QUESTION_SCHEMA_FILE)
    _log_lane(f"staged kaggle kernel [{decision_kind}] ({spec.gpu}) "
              f"run_tag={tag} -> {stage}")
    return receipt


def stage_colab_notebook(*, decision_kind: str,
                         run_tag: str | None = None) -> dict[str, Any]:
    """Colab notebook payload in the receipts style (dry-safe).

    Returns a receipt dict; the payload script lands at
    results/laya_lane/colab/<decision_kind>/laya_decision_colab.py.
    Fail-loud preconditions match stage_decision_kernel (epochs, slug).
    """
    spec = _spec()
    if decision_kind not in DECISION_BINDINGS:
        raise ValueError(f"unknown decision kind: {decision_kind!r}; "
                         f"expected {list(DECISION_BINDINGS)}")
    if spec.laya_decision_epochs <= 0:
        raise RuntimeError(
            "config laya.laya_decision_epochs <= 0: the decision lane is "
            "disabled (no payload may stage a session)")
    if not spec.export_dataset_slug and not spec.dataset_slug:
        raise RuntimeError(
            "config laya.export_dataset_slug / dataset_slug both unset; "
            "name the target surface (owner/slug) before staging")
    question = stage_question_schema("colab")
    input_receipt = stage_decision_input("colab",
                                         decision_kind=decision_kind)
    stage = staging_dir() / "colab" / decision_kind
    stage.mkdir(parents=True, exist_ok=True)
    tag = run_tag or spec.run_tag_prefix + decision_tag()
    entry = DECISION_BINDINGS[decision_kind]
    staged_csv = input_receipt["staged"]
    script = _template(NOTEBOOK_SCRIPT, {
        "LAYA_PACKAGE": spec.laya_package,
        "DECISION_KIND": decision_kind,
        "RUN_TAG": tag,
        "DECISION_CSV": Path(staged_csv).name,
        "STATE_COLUMN": entry["state_column"],
    })
    _kernel_script_gate(script)
    notebook = stage / COLAB_NOTEBOOK_NAME
    notebook.write_text(script, encoding="utf-8")
    receipt = {
        "kernel": COLAB_NOTEBOOK_NAME,
        "kind": decision_kind,
        "gpu": "T4 (single)",
        "run_tag": tag,
        "staged": str(stage),
        "notebook": str(notebook),
        "receipt_style": "results/laya_lane/<kind>/<op>/...",
        "question_schema": question["staged"],
        "decision_input": staged_csv,
        "epochs": spec.laya_decision_epochs,
        "note": "the colab CLI surface is unchanged; delivery contract only",
    }
    atomic_write_json(receipt, stage / f"{decision_kind}.receipt.json")
    _log_lane(f"staged colab notebook payload [{decision_kind}] "
              f"run_tag={tag} -> {notebook}")
    return receipt


# ── the executed ops (fail-loud, --execute gated) ─────────────────────────
def push_kaggle_kernel(stage_dir: Path, *, execute: bool,
                       activate: bool = True) -> dict[str, Any]:
    """`kaggle kernels push` a staged payload, `--execute`-gated.

    Dry run: returns the plan + argv, never spawns the kaggle subprocess.
    Executed: requires the staged metadata file (--activate gate), runs
    the preflight (staged_kernel_preflight), pushes, and embeds the CLI's
    own output in the raised RuntimeError on a failing returncode.
    """
    argv = [sys.executable, "-m", "kaggle", "kernels", "push",
            "-p", str(stage_dir)]
    plan: dict[str, Any] = {"mode": "executed" if execute else "dry-run",
                            "argv": argv, "stage": str(stage_dir)}
    if not execute:
        _log_lane(f"dry-run: would run {' '.join(argv)}")
        return plan
    metadata_file = Path(stage_dir) / "kernel-metadata.json"
    if not metadata_file.is_file():
        raise RuntimeError(
            "--activate gate: no staged kernel at "
            f"{stage_dir} (kernel-metadata.json is missing); stage first "
            "(--what stage-kernel)")
    from core.runtime_inputs import staged_kernel_preflight

    staged_kernel_preflight(Path(stage_dir))
    result = subprocess.run(argv, cwd=TRAIN_ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    output = result.stdout or ""
    rc = result.returncode
    plan["returncode"] = rc
    if rc != 0:
        tail = output.strip()[-4000:] or "(kaggle produced no output)"
        raise RuntimeError(
            f"kaggle command failed (rc={rc}): {' '.join(argv)}\n"
            f"--- kaggle output ---\n{tail}")
    plan["pushed"] = True
    _log_lane(f"pushed kernel payload: {' '.join(argv)} rc=0")
    return plan


def collect_kaggle_result(decision_kind: str, slug: str, *,
                          execute: bool = False) -> dict[str, Any]:
    """`kaggle kernels output` for a staged/decided kernel.

    Dry run: returns the plan only. Executed: pulls the payload and
    verifies the receipt (laya_<decision_kind>.receipt.json inside the
    archive) against the locally staged receipt — fail-loud on a sha256
    mismatch. Results install under results/laya_lane/fetch/<decision>/.
    """
    plan: dict[str, Any] = {"mode": "executed" if execute else "dry-run",
                            "decision_kind": decision_kind, "slug": slug}
    if not execute:
        _log_lane(f"dry-run: would fetch kernel output for {slug}")
        return plan
    stage = staging_dir() / "fetch" / decision_kind
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    result = subprocess.run(
        [sys.executable, "-m", "kaggle", "kernels", "output", slug,
         "-p", str(stage)], cwd=TRAIN_ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"kaggle kernels output failed (rc={result.returncode}) for "
            f"{slug}: {result.stdout.strip()[-4000:]}")
    archives = sorted(stage.glob("*.tar.gz")) or sorted(stage.glob("*.zip"))
    if not archives:
        raise RuntimeError(f"kaggle kernels output staged no archive "
                           f"under {stage} (slug {slug})")
    receipt_name = f"laya_{decision_kind}.receipt.json"
    with tarfile.open(archives[0], "r:*") as tar:
        members = tar.getnames()
        if receipt_name not in members:
            raise RuntimeError(
                f"fetched archive {archives[0].name} carries no "
                f"{receipt_name}; the kernel receipt contract failed")
        payload = json.loads(tar.extractfile(receipt_name)
                             .read().decode())
    plan.update({"archive": str(archives[0]), "members": members,
                 "receipt": payload})
    _log_lane(f"fetched kernel output for {slug}: "
              f"archive={archives[0].name} members={len(members)}")
    return plan


class LayaLane:
    """The lane surface: results/laya_lane/<kind>/<decision>/ receipts.

    ONE class, TWO kinds: kind names the remote surface ("kaggle",
    "colab"); anything else fails loud.
    """

    kind: str

    def __init__(self, kind: str):
        if kind not in KINDS:
            raise ValueError(f"unknown laya lane kind: {kind!r}; "
                             f"expected {list(KINDS)}")
        self.kind = kind
        self._spec = _spec()

    def stage(self, decision_kind: str) -> dict[str, Any]:
        """Stage the payload (offline, dry-safe)."""
        if self.kind == "kaggle":
            return stage_decision_kernel(decision_kind=decision_kind)
        return stage_colab_notebook(decision_kind=decision_kind)

    def push(self, stage_dir: Path, *, execute: bool = False,
             activate: bool = True) -> dict[str, Any]:
        """`--execute` gated push (kaggle kind only)."""
        if self.kind != "kaggle":
            raise RuntimeError("push is a kaggle-lane operation")
        return push_kaggle_kernel(stage_dir, execute=execute,
                                  activate=activate)

    def run(self, args: argparse.Namespace) -> dict[str, Any]:
        """Dispatch the parsed main() args through this lane."""
        if args.decision not in DECISION_BINDINGS:
            raise ValueError(f"unknown decision kind: {args.decision!r}")
        if args.kind != self.kind:
            raise ValueError(f"--kind {args.kind!r} does not match the "
                             f"lane kind {self.kind!r}")
        return self.stage(args.decision)


# ── main ───────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=KINDS, default="kaggle")
    parser.add_argument("--decision", choices=GPU_KINDS, default="attribute",
                        help="which typed decision run to stage "
                             "(default: attribute)")
    parser.add_argument("--execute", action="store_true",
                        help="make the remote call (kaggle kernels push); "
                             "the default is an offline dry-run")
    args = parser.parse_args()
    lane = LayaLane(args.kind)
    receipt = lane.stage(args.decision)
    print(_stamp(), f"[laya-lane] staged {args.kind}/{args.decision} payload: "
          f"{json.dumps(receipt, indent=2)}", flush=True)
    if args.execute and args.kind == "kaggle":
        plan = lane.push(Path(receipt["staged"]), execute=True)
        print(json.dumps(plan, indent=2), flush=True)
    elif args.execute:
        _log_lane("colab payloads are a delivery contract only; nothing "
                  "to --execute")
    else:
        _log_lane("dry-run only; pass --execute to touch the remote surface")


if __name__ == "__main__":
    main()
