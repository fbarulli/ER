"""Kaggle dataset/export transport lane (branch kaggle-lane).

Parallel dataset/transport lane alongside the Colab GPU lane. NEW-file only
(owner ruling 8 precedent): this module never edits or imports the working
colab lane (`cli.colab`, `cli.colab_bundle`, `cli.colab_data_bundle_prep`) —
it shares only `core.common` config/path primitives and `core.manifest`
atomic-write helpers, exactly like every other lane.

What it owns (SSOT: config/training.yaml `kaggle:` block):

* packaging  — a cohort export CSV becomes a Kaggle dataset payload: the
  archive the kaggle CLI uploads plus a config manifest recording the
  measured rows + sha256 census (the same census shape the repo's audit
  pins use — measured at package time, never hardcoded).
* upload     — `kaggle datasets create`/`version` driven through the
  configured executable; fail-loud (RuntimeError) on missing credentials,
  missing executable, or unset `kaggle.slug` — never a silent skip.
* download   — `kaggle datasets download` fetch-back that verifies the
  archive sha256 against the receipt written at package time (the
  transport-identity contract the suite recovery machinery uses).
* submission — validate/format a finished SKU_ITEM frame through the
  EXISTING `scripts.format_submission.format_submission` (imported, never
  duplicated) into the external two-column contract.

Every network-touching command runs ONLY when the caller passes
`--execute`; the default is a dry run that does everything up to and
excluding the kaggle subprocess. That keeps this lane runnable on this box
(no kaggle credentials here) while the owner's live invocation needs one
explicit flag. No default flips anywhere.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from core.common import TRAIN_ROOT, training_cfg
from core.manifest import atomic_write_json, sha256_file
from core.runtime_inputs import checkout_members, checkout_inventory, checkout_preflight_script


class ExportCensus(BaseModel):
    """Measured cohort-export census recorded at package time."""

    model_config = ConfigDict(extra="forbid")

    rows: int = Field(ge=0)
    bytes: int = Field(ge=1)
    sha256: str = Field(min_length=64, max_length=64)
    columns: list[str] = Field(min_length=1)


class KagglePackage(BaseModel):
    """One packaged dataset payload: archive + config + measured census."""

    model_config = ConfigDict(extra="forbid")

    export_path: str
    archive_path: str
    metadata_path: str
    census: ExportCensus


def cohort_label(dataset_csv: Path) -> str:
    """Cohort tag mirroring cli.colab_data_bundle_prep.cohort_label values.

    `full` for the SSOT default export, `50pct` for the half-cohort, else a
    sanitized stem. The SAME tags keep the two lanes' transcripts mutually
    attributable without importing the colab module.
    """
    name = Path(dataset_csv).name
    if name == _spec().export_csvs[0]:
        return "full"
    if "50pct" in name:
        return "50pct"
    return "".join(
        ch if ch.isalnum() or ch in "-_" else "_" for ch in name.rsplit(".", 1)[0]
    )


def _spec():
    return training_cfg().kaggle


def staging_dir() -> Path:
    return (TRAIN_ROOT / _spec().staging_dir).resolve()


def _measure_export(csv_path: Path) -> ExportCensus:
    import pandas as pd

    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"cohort export not found: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if frame.empty:
        raise ValueError(f"cohort export has no data rows: {path}")
    return ExportCensus(
        rows=int(len(frame)),
        bytes=int(path.stat().st_size),
        sha256=sha256_file(path),
        columns=[str(column) for column in frame.columns],
    )


def package_export(csv_path: Path, *, config: dict[str, Any] | None = None) -> KagglePackage:
    """Package one cohort export into a Kaggle dataset upload payload.

    Dry-safe: writes the zip archive + `dataset_metadata.json` + census
    receipt under the configured staging root; no network, no kaggle
    subprocess. Re-packaging the same export bytes is deterministic apart
    from the zip member timestamps.
    """
    spec = _spec()
    source = Path(csv_path).resolve()
    census = _measure_export(source)
    label = cohort_label(source)
    stage = staging_dir() / label
    stage.mkdir(parents=True, exist_ok=True)
    archive = stage / f"{label}{spec.payload_archive_suffix}"
    data_name = spec.packaged_data_name
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(source, arcname=data_name)
    metadata = dict(config) if config is not None else {
        "title": f"euromonitor-reconciliation-{label}",
        "id": spec.slug or f"euromonitor-reconciliation-{label}",
        "licenses": [{"name": "other"}],
    }
    metadata_path = stage / spec.metadata_file
    atomic_write_json(metadata, metadata_path)
    receipt = {
        "cohort": label,
        "export": str(source),
        "export_bytes": census.bytes,
        "export_rows": census.rows,
        "export_sha256": census.sha256,
        "export_columns": census.columns,
        "archive": str(archive),
        "archive_bytes": int(archive.stat().st_size),
        "archive_sha256": sha256_file(archive),
        "metadata": metadata,
    }
    receipt_path = stage / f"{label}{spec.receipt_suffix}"
    atomic_write_json(receipt, receipt_path)
    return KagglePackage(
        export_path=str(source),
        archive_path=str(archive),
        metadata_path=str(metadata_path),
        census=census,
    )


def _require_kaggle_executable(executable: str) -> str:
    resolved = shutil.which(executable)
    if resolved is None:
        raise RuntimeError(
            f"kaggle executable {executable!r} not found on PATH; install the "
            "kaggle CLI and place credentials at ~/.kaggle/kaggle.json "
            "(never inside this repository)"
        )
    return resolved


def upload_dataset(package: KagglePackage, *, execute: bool) -> dict[str, Any]:
    """Create-or-version the Kaggle dataset from a packaged payload.

    `execute=False` (default) stops before any kaggle subprocess and
    returns the plan; `execute=True` runs the real upload and requires
    both the executable and a configured `kaggle.slug`.
    """
    spec = _spec()
    plan: dict[str, Any] = {
        "mode": "executed" if execute else "dry-run",
        "archive": package.archive_path,
        "metadata": package.metadata_path,
        "slug": spec.slug,
        "rows": package.census.rows,
        "export_sha256": package.census.sha256,
    }
    if not execute:
        return plan
    slug = spec.slug
    if not slug:
        raise RuntimeError(
            "config/training.yaml kaggle.slug is unset; name the target "
            "dataset (owner/slug) before an executed upload"
        )
    executable = _require_kaggle_executable(spec.kaggle_executable)
    command = [executable, "datasets", "create", "--dir-mode", "skip",
               "-r", str(package.archive_path)]
    _, _ = _run_kaggle(command)
    plan["returncode"] = 0
    return plan


def download_dataset(package: KagglePackage, *, execute: bool) -> dict[str, Any]:
    """Fetch the published dataset back and verify it against the receipt.

    The receipt written by `package_export` is the transport-identity
    contract: a fetch-back whose sha256 differs from the packaged archive
    raises RuntimeError instead of silently accepting drift.
    """
    spec = _spec()
    receipt_path = (staging_dir() / cohort_label(Path(package.export_path))
                    / f"{cohort_label(Path(package.export_path))}{spec.receipt_suffix}")
    if not receipt_path.is_file():
        raise FileNotFoundError(
            f"no package receipt at {receipt_path}; run package_export first"
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    plan: dict[str, Any] = {
        "mode": "executed" if execute else "dry-run",
        "slug": spec.slug,
        "expected_archive_sha256": receipt["archive_sha256"],
    }
    if not execute:
        return plan
    slug = spec.slug
    if not slug:
        raise RuntimeError(
            "config/training.yaml kaggle.slug is unset; name the target "
            "dataset (owner/slug) before an executed download"
        )
    executable = _require_kaggle_executable(spec.kaggle_executable)
    stage = staging_dir() / cohort_label(Path(package.export_path))
    command = [executable, "datasets", "download", slug, "--path", str(stage)]
    _, _ = _run_kaggle(command)
    fetched_candidates = sorted(
        stage.glob("*.zip"), key=lambda path: path.stat().st_mtime, reverse=True
    )
    if not fetched_candidates:
        raise RuntimeError(f"kaggle download produced no archive under {stage}")
    fetched = fetched_candidates[0]
    observed = sha256_file(fetched)
    if observed != receipt["archive_sha256"]:
        raise RuntimeError(
            "fetched dataset archive sha256 mismatch: "
            f"expected {receipt['archive_sha256']} observed {observed}"
        )
    plan["fetched_archive"] = str(fetched)
    plan["verified"] = True
    return plan


def package_submission(input_path: Path, output_path: Path) -> Path:
    """Format + verify a finished prediction frame for the external contract.

    Reuses scripts.format_submission.format_submission verbatim (single
    source of truth for the two-column rule) and adds the lane's receipt:
    rows, unique items, and the unmatched count under the configured
    prefix from the SSOT.
    """
    from scripts.format_submission import format_submission
    from training.rand_matching import _unmatched_prefix

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    format_submission(Path(input_path), output_path)
    import pandas as pd

    frame = pd.read_csv(output_path, dtype=str, keep_default_na=False)
    spec = _spec()
    expected = list(spec.submission_id_columns)
    if [column.lower() for column in frame.columns] != expected:
        raise RuntimeError(
            f"submission columns {[str(c) for c in frame.columns]} do not "
            f"match the external contract {expected}"
        )
    unmatched = int(
        frame[expected[1]].str.startswith(_unmatched_prefix()).sum()
    )
    stage = staging_dir()
    stage.mkdir(parents=True, exist_ok=True)
    receipt = {
        "submission": str(output_path),
        "rows": int(len(frame)),
        "unique_items": int(frame[expected[1]].nunique()),
        "unmatched_items": unmatched,
        "sha256": sha256_file(output_path),
        "columns": expected,
    }
    atomic_write_json(
        receipt,
        stage / f"submission{spec.receipt_suffix}",
    )
    print(
        f"[kaggle-lane] submission packaged: rows={receipt['rows']} "
        f"unique_items={receipt['unique_items']} "
        f"unmatched={unmatched} -> {output_path}",
        flush=True,
    )
    return output_path


# ── remote bundle-generation kernel (owner ruling 2026-10-06: CPU-only) ─────

CREDENTIALS_PATH = Path.home() / ".kaggle" / "kaggle.json"
# kaggle CLI 2.x consumes this token file (OAuth-free path); the legacy
# kaggle.json stays alongside for the 1.x/hub readers.
ACCESS_TOKEN_PATH = Path.home() / ".kaggle" / "access_token"
BUNDLE_KERNEL_CODE_FILE = "bundle_cpu.py"

BUNDLE_KERNEL_SCRIPT = '''\
"""ER bundle generation on a Kaggle CPU session (generated by cli.kaggle_lane).

Owner ruling 2026-10-06: bundle generation is CPU-only and runs here, not
locally; the GPU session only trains. Clones the pinned revision (partial +
sparse checkout), installs the worker requirements, runs
training.prepare_all end-to-end, and stages the launch package plus its
receipt into /kaggle/working for hash-verified fetch-back.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPOSITORY = "@REPOSITORY@"
BRANCH = "@BRANCH@"
REVISION = "@REVISION@"
CHECKOUT_PATHS = @CHECKOUT_PATHS@
REQUIREMENTS = "@REQUIREMENTS@"
COHORT = "@COHORT@"
COHORT_DATASET = "@COHORT_DATASET@"

SCRATCH = (Path("/kaggle/tmp") if Path("/kaggle/tmp").is_dir()
           else Path(tempfile.gettempdir()) / "er_bundle")
WORKING = Path("/kaggle/working")


def sh(command, **kwargs):
    print("+ " + " ".join(str(part) for part in command), flush=True)
    # Route child stderr (including tqdm) through the notebook output stream.
    input_text = kwargs.pop("input", None)
    kwargs.pop("text", None)
    with subprocess.Popen([str(part) for part in command],
                          stdin=subprocess.PIPE if input_text is not None else None,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1,
                          **kwargs) as process:
        if input_text is not None:
            process.stdin.write(input_text)
            process.stdin.close()
        for line in process.stdout:
            print(line, end="", flush=True)
        rc = process.wait()
    if rc:
        raise subprocess.CalledProcessError(rc, command)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


SCRATCH.mkdir(parents=True, exist_ok=True)
root = SCRATCH / "ER"
if root.exists():
    shutil.rmtree(root)
sh(["git", "clone", "--filter=blob:none", "--no-checkout", "--depth", "1",
    "--branch", BRANCH, REPOSITORY, root])
head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                      capture_output=True, text=True, check=True).stdout.strip()
if head != REVISION:
    raise SystemExit(
        "branch tip moved: cloned %s but the kernel pins %s; regenerate the "
        "kernel (cli.kaggle_lane --what bundle-kernel)" % (head, REVISION))
# Use the same explicit runtime inventory and sparse mode as Colab.
sh(["git", "-C", str(root), "sparse-checkout", "set", "--no-cone", "--stdin"],
   input="\\n".join("/" + member for member in CHECKOUT_PATHS) + "\\n", text=True)
sh(["git", "-C", str(root), "checkout", BRANCH])
@RUNTIME_PREFLIGHT@
if COHORT != "full":
    source = root / COHORT_DATASET
    if not source.is_file():
        raise SystemExit(
            f"cohort export {COHORT_DATASET} not in the branch checkout for "
            f"cohort {COHORT}; commit it at the repository root or attach it "
            "as a dataset")
    shutil.copy2(source, root / "dataset.csv")
    print(f"[bundle-cpu] cohort remap: {COHORT_DATASET} -> dataset.csv", flush=True)
sh([sys.executable, "-m", "pip", "install", "-q", "-r", REQUIREMENTS], cwd=root)
env = {**os.environ, "PYTHONPATH": str(root / "src"), "PYTHONUNBUFFERED": "1"}
sh([sys.executable, "-u", "-m", "training.prepare_all",
    "--tracks-config", "config/model_tracks.yaml"], cwd=root, env=env)
runs = sorted((root / "results" / "training_prep").glob("*/all_tracks_inputs.tar.zst"),
              key=lambda path: path.stat().st_mtime)
if not runs:
    raise SystemExit("prepare_all produced no all_tracks_inputs.tar.zst")
run_dir = runs[-1].parent
destination = WORKING / "bundle"
destination.mkdir(parents=True, exist_ok=True)
shutil.copy2(runs[-1], destination / "all_tracks_inputs.tar.zst")
for name in ("manifest.json", "timings.json"):
    sidecar = run_dir / name
    if sidecar.is_file():
        shutil.copy2(sidecar, destination / name)
receipt = {
    "revision": REVISION,
    "branch": BRANCH,
    "run_dir": run_dir.name,
    "cohort": COHORT,
    "cohort_dataset": COHORT_DATASET,
    "archive": "all_tracks_inputs.tar.zst",
    "archive_bytes": (destination / "all_tracks_inputs.tar.zst").stat().st_size,
    "archive_sha256": sha256_file(destination / "all_tracks_inputs.tar.zst"),
}
(destination / "bundle.receipt.json").write_text(
    json.dumps(receipt, indent=2) + "\\n", encoding="utf-8")
print("[bundle-cpu] receipt: " + json.dumps(receipt, indent=2), flush=True)
'''

# ── remote GPU kernels (owner ruling 2026-10-06: GPU trains, never preps) ──
#
# Two GPU objectives, one skeleton each (pinned sparse clone -> attach the
# CPU bundle kernel output as kernel_source -> verify -> run -> manifest-
# backed tar.zst + sha256 sidecar into /kaggle/working):
#
#   train_gpu  — track train + ablation: model_tracks.run under
#                ER_GPU_TRAINING_ONLY=1 (baseline export + data gate + three
#                parallel CUDA workers under MPS; the Kaggle T4 image ships
#                nvidia-cuda-mps-control at /opt/bin — kernel er-mps-probe
#                verified daemon start + client context on this image).
#   embed_gpu  — embedding forwards for hybrid: encode_prepared_embeddings
#                against the git-shipped checkpoint (artifacts/models is in
#                checkout_paths), request + prepared_text.npz attached as a
#                Kaggle dataset; emits vectors.npz + .sha256.

def cohort_export_csv(cohort: str) -> str:
    """Root-relative export filename for a cohort tag (inverse cohort_label).

    `full` = the SSOT default export; any other tag must name an export_csvs
    entry whose filename carries the tag (50pct). Fail-loud on unknown.
    """
    export_csvs = _spec().export_csvs
    if cohort == "full":
        return export_csvs[0]
    for name in export_csvs:
        if cohort in name:
            return name
    raise RuntimeError(
        f"no export_csvs entry matches cohort {cohort!r}; add it to config "
        "kaggle.export_csvs (root-relative, self-describing filename)")


TRAIN_KERNEL_CODE_FILE = "train_gpu.py"
EMBED_KERNEL_CODE_FILE = "embed_gpu.py"
TRAIN_KERNEL_SHARED = '''\
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPOSITORY = "@REPOSITORY@"
BRANCH = "@BRANCH@"
REVISION = "@REVISION@"
CHECKOUT_PATHS = @CHECKOUT_PATHS@
REQUIREMENTS = "@REQUIREMENTS@"
RUN_TAG = "@RUN_TAG@"
SUITE_CONFIG = "@SUITE_CONFIG@"
BUNDLE_KERNEL_SLUG = "@BUNDLE_KERNEL_SLUG@"

WORKING = Path("/kaggle/working")
INPUTS = Path("/kaggle/input")
SCRATCH = (Path("/kaggle/tmp") if Path("/kaggle/tmp").is_dir()
           else Path(tempfile.gettempdir()) / "er_gpu")
BUNDLE_RECEIPT = "bundle.receipt.json"
BUNDLE_ARCHIVE = "all_tracks_inputs.tar.zst"


def sh(command, **kwargs):
    print("+ " + " ".join(str(part) for part in command), flush=True)
    # Route child stderr (including tqdm) through the notebook output stream.
    input_text = kwargs.pop("input", None)
    kwargs.pop("text", None)
    with subprocess.Popen([str(part) for part in command],
                          stdin=subprocess.PIPE if input_text is not None else None,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1,
                          **kwargs) as process:
        if input_text is not None:
            process.stdin.write(input_text)
            process.stdin.close()
        for line in process.stdout:
            print(line, end="", flush=True)
        rc = process.wait()
    if rc:
        raise subprocess.CalledProcessError(rc, command)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_input_archive():
    """Locate the attached bundle output (receipt + archive) under /kaggle/input."""
    candidates = list(INPUTS.rglob(BUNDLE_RECEIPT))
    if not candidates:
        raise SystemExit("attached kernel output contained no " + BUNDLE_RECEIPT)
    receipt_dir = candidates[0].parent
    archive = receipt_dir / BUNDLE_ARCHIVE
    if not archive.is_file():
        raise SystemExit("attached kernel output missing " + BUNDLE_ARCHIVE)
    return archive, receipt_dir


def clone_pinned():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    root = SCRATCH / "ER"
    if root.exists():
        shutil.rmtree(root)
    sh(["git", "clone", "--filter=blob:none", "--no-checkout", "--depth", "1",
        "--branch", BRANCH, REPOSITORY, root])
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                          capture_output=True, text=True, check=True).stdout.strip()
    if head != REVISION:
        raise SystemExit("branch tip moved: cloned %s but the kernel pins %s"
                         % (head, REVISION))
    sh(["git", "-C", str(root), "sparse-checkout", "set", "--no-cone", "--stdin"],
       input="\\n".join("/" + member for member in CHECKOUT_PATHS) + "\\n", text=True)
    sh(["git", "-C", str(root), "checkout", BRANCH])
@RUNTIME_PREFLIGHT@
    sh([sys.executable, "-m", "pip", "install", "-q", "-r", REQUIREMENTS], cwd=root)
    sys.path.insert(0, str(root / "src"))
    return root


def stage_result_archive(output: Path, *, kind: str, extra: dict) -> str:
    """Manifest-backed tar.zst + .sha256 sidecar (colab result-archive mirror)."""
    result_archive = WORKING / f"{kind}.tar.zst"
    files = sorted(p for p in output.rglob("*") if p.is_file())
    root_name = output.name
    from core.archive_reader import tar_archive
    with tar_archive(result_archive, "w") as tar:
        for path in files:
            arcname = Path(root_name) / path.relative_to(output)
            tar.add(path, arcname=str(arcname))
    result_manifest = {
        "kind": kind,
        "run_tag": RUN_TAG,
        "revision": REVISION,
        "files_count": len(files),
        "files": {
            str(Path(root_name) / p.relative_to(output)): {
                "bytes": p.stat().st_size, "sha256": sha256_file(p)}
            for p in files
        },
        **extra,
    }
    (WORKING / f"{kind}.manifest.json").write_text(
        json.dumps(result_manifest, indent=2), encoding="utf-8")
    digest = sha256_file(result_archive)
    (WORKING / f"{kind}.tar.zst.sha256").write_text(digest + "\\n", encoding="utf-8")
    print(f"[{kind}] staged: {result_archive} sha256={digest}", flush=True)
    return digest
'''

TRAIN_KERNEL_BODY = '''
archive_path, receipt_dir = locate_input_archive()
receipt = json.loads((receipt_dir / BUNDLE_RECEIPT).read_text())
if receipt.get("archive_sha256") and sha256_file(archive_path) != receipt["archive_sha256"]:
    raise SystemExit("attached bundle sha256 mismatch against kernel receipt")

root = clone_pinned()
env = {**os.environ, "PYTHONPATH": str(root / "src"), "PYTHONUNBUFFERED": "1",
       "WANDB_MODE": "disabled"}

# Install the prepared package exactly where the archive declares members,
# then verify the suite preflight contract file landed at TRAIN_ROOT
# (model_tracks.run ER_GPU_TRAINING_ONLY preflight reads it, run.py:68).
sys.path.insert(0, str(root / "src"))
from core.portable_archive import verified_archive
with verified_archive(archive_path, "model_tracks_package.json") as (archive, _):
    archive.extractall(root)
package_manifest = root / "model_tracks_package.json"
if not package_manifest.is_file():
    raise SystemExit("package install produced no model_tracks_package.json")
(root / "model_tracks_package.json").write_text(package_manifest.read_text(),
                                                encoding="utf-8")

output = root / "results" / "model_tracks" / RUN_TAG
env["ER_GPU_TRAINING_ONLY"] = "1"
sh([sys.executable, "-m", "model_tracks.run",
    "--config", SUITE_CONFIG,
    "--output", str(output), "--run-tag", RUN_TAG], cwd=root, env=env)

stage_result_archive(output, kind="result_bundle", extra={
    "bundle_receipt": receipt,
    "bundle_kernel": BUNDLE_KERNEL_SLUG,
    "checkout_paths": CHECKOUT_PATHS,
})
'''

EMBED_KERNEL_BODY = '''
# Embedding objective: encode the prepared token archive against the
# git-shipped checkpoint (artifacts/models rides the sparse checkout).
# The request dataset (request.json + prepared_text.npz) is attached as a
# Kaggle dataset: scripts/encode_prepared_embeddings.py refuses any
# re-composition, so the bytes must match the locally prepared request.
candidates = list(INPUTS.rglob("request.json"))
if not candidates:
    raise SystemExit("attached dataset contained no request.json")
request_dir = candidates[0].parent
root = clone_pinned()
checkpoint = root / "@CHECKPOINT@"
if not checkpoint.is_dir():
    raise SystemExit("git-shipped checkpoint missing: " + str(checkpoint))
env = {**os.environ, "PYTHONPATH": str(root / "src"), "PYTHONUNBUFFERED": "1",
       "WANDB_MODE": "disabled"}
output = root / "results" / "embedding_job" / RUN_TAG
output.mkdir(parents=True, exist_ok=True)
sh([sys.executable, "scripts/encode_prepared_embeddings.py",
    "--request", str(request_dir / "request.json"),
    "--checkpoint", str(checkpoint),
    "--output", str(output / "vectors.npz"),
    "--device", "cuda"], cwd=root, env=env)
stage_result_archive(output, kind="vectors", extra={
    "request_dir": str(request_dir),
    "checkpoint": "@CHECKPOINT@",
})
'''


def _git_revision() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=TRAIN_ROOT,
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"git rev-parse failed in {TRAIN_ROOT}: {result.stderr.strip()}")
    return result.stdout.strip()


def write_credentials(*, key_env: str | None = None, execute: bool) -> dict[str, Any]:
    """Materialize ~/.kaggle/kaggle.json from the environment.

    The token never enters this repository or argv: it is read from the
    config-named environment variable (overridable with --key-env) and
    written to the standard credential path with 0600 permissions. Dry run
    by default; --execute writes the file.
    """
    spec = _spec()
    resolved_env = key_env or spec.api_key_env
    if not spec.username:
        raise RuntimeError(
            "config kaggle.username is unset; name the Kaggle account before "
            "writing credentials")
    token = os.environ.get(resolved_env, "").strip()
    plan: dict[str, Any] = {
        "mode": "executed" if execute else "dry-run",
        "target": str(CREDENTIALS_PATH),
        "username": spec.username,
        "key_env": resolved_env,
        "key_present": bool(token),
    }
    if not execute:
        return plan
    if not token:
        raise RuntimeError(
            f"environment variable {resolved_env!r} is empty or unset; export "
            "the Kaggle API token (credentials never live in this repository)")
    CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CREDENTIALS_PATH.write_text(
        json.dumps({"username": spec.username, "key": token}) + "\n",
        encoding="utf-8")
    CREDENTIALS_PATH.chmod(0o600)
    # The 2.x CLI loads its token from here on every fresh process; no
    # trailing newline — CLI readers do not strip reliably.
    ACCESS_TOKEN_PATH.write_text(token, encoding="utf-8")
    ACCESS_TOKEN_PATH.chmod(0o600)
    plan["written"] = True
    plan["access_token_written"] = str(ACCESS_TOKEN_PATH)
    return plan


def _log_lane(line: str) -> None:
    """Timestamped lane logging: console plus append-only lane log.

    Best-effort on the file side — a log-write failure is printed and never
    allowed to mask the operation's own outcome.
    """
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[kaggle-lane {stamp}] {line}", flush=True)
    try:
        log_dir = staging_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / "lane.log").open("a", encoding="utf-8") as handle:
            handle.write(f"{stamp} {line}\n")
    except OSError as error:
        print(f"[kaggle-lane] lane.log write failed ({error}); continuing",
              flush=True)


def _run_kaggle(command: list[str]) -> tuple[int, str]:
    """Run the kaggle CLI with full logging; never swallow its output.

    stdout and stderr are captured together, echoed line by line, appended
    to the lane log, and — on a failing returncode — embedded verbatim in
    the raised RuntimeError so Kaggle's own diagnostics always surface.
    """
    printable = " ".join(command)
    _log_lane(f"$ {printable}")
    started = time.monotonic()
    result = subprocess.run(command, cwd=TRAIN_ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    elapsed = time.monotonic() - started
    output = result.stdout or ""
    for line in output.splitlines():
        print(f"[kaggle] {line}", flush=True)
    _log_lane(f"rc={result.returncode} seconds={elapsed:.1f}")
    if result.returncode != 0:
        tail = output.strip()[-4000:] or "(kaggle produced no output)"
        raise RuntimeError(
            f"kaggle command failed (rc={result.returncode}): {printable}\n"
            f"--- kaggle output ---\n{tail}")
    return result.returncode, output


def stage_bundle_kernel(*, revision: str | None = None,
                        cohort: str = "full") -> dict[str, Any]:
    """Stage the CPU bundle-generation kernel: metadata + script + receipt.

    Dry-safe: writes into the staging area only; `push_bundle_kernel` makes
    the network call. The revision is pinned at staging time so the kernel
    clones exactly the source the package provenance will record. The
    cohort chooses which root-level export drives prepare_all in the
    kernel (colab replace-on-checkout pattern, no upload round-trip).
    """
    spec = _spec()
    slug = spec.cpu_kernel_slug
    if not slug:
        raise RuntimeError(
            "config kaggle.cpu_kernel_slug is unset; name the CPU kernel "
            "(owner/slug) before staging")
    pinned = revision or _git_revision()
    cohort_dataset = cohort_export_csv(cohort)
    stage = staging_dir() / "bundle_kernel"
    stage.mkdir(parents=True, exist_ok=True)
    metadata = {
        "id": slug,
        "title": slug.rsplit("/", 1)[-1].replace("-", " ").title(),
        "code_file": BUNDLE_KERNEL_CODE_FILE,
        "language": "python",
        "kernel_type": "script",
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "is_private": True,
    }
    script = (BUNDLE_KERNEL_SCRIPT
              .replace("@REPOSITORY@", spec.repository)
              .replace("@BRANCH@", spec.branch)
              .replace("@REVISION@", pinned)
              .replace("@CHECKOUT_PATHS@",
                       json.dumps(checkout_members((*spec.checkout_paths, cohort_dataset))))
              .replace("@RUNTIME_PREFLIGHT@", checkout_preflight_script(
                  checkout_inventory((*spec.checkout_paths, cohort_dataset))))
              .replace("@REQUIREMENTS@", spec.bundle_requirements)
              .replace("@COHORT@", cohort)
              .replace("@COHORT_DATASET@", cohort_dataset))
    atomic_write_json(metadata, stage / "kernel-metadata.json")
    (stage / BUNDLE_KERNEL_CODE_FILE).write_text(script, encoding="utf-8")
    receipt = {
        "kernel": slug,
        "gpu": False,
        "branch": spec.branch,
        "revision": pinned,
        "cohort": cohort,
        "cohort_dataset": cohort_dataset,
        "staged": str(stage),
        "code_file": BUNDLE_KERNEL_CODE_FILE,
    }
    atomic_write_json(receipt, stage / "bundle_kernel.receipt.json")
    return receipt


def push_bundle_kernel(stage_dir: Path) -> dict[str, Any]:
    """Push the staged CPU kernel via the configured kaggle executable."""
    spec = _spec()
    slug = spec.cpu_kernel_slug
    if not slug:
        raise RuntimeError(
            "config kaggle.cpu_kernel_slug is unset; name the CPU kernel "
            "(owner/slug) before pushing")
    from core.runtime_inputs import staged_kernel_preflight
    staged_kernel_preflight(Path(stage_dir))
    executable = _require_kaggle_executable(spec.kaggle_executable)
    command = [executable, "kernels", "push", "-p", str(stage_dir)]
    _, _ = _run_kaggle(command)
    return {"mode": "executed", "kernel": slug, "pushed": True,
            "staged": str(stage_dir)}


def kernel_status(slug: str | None = None, *, which: str = "cpu") -> dict[str, Any]:
    spec = _spec()
    resolved = slug or (spec.gpu_kernel_slug if which == "gpu"
                        else spec.cpu_kernel_slug)
    if not resolved:
        raise RuntimeError(
            f"config kaggle.{which}_kernel_slug is unset; pass a slug or name "
            f"the {which} kernel in config")
    executable = _require_kaggle_executable(spec.kaggle_executable)
    command = [executable, "kernels", "status", resolved]
    _, output = _run_kaggle(command)
    # CLI 2.x prints  "KernelWorkerStatus.COMPLETE"; 1.x printed bare words.
    normalized = output.replace("KernelWorkerStatus.", "")
    status = "unknown"
    for candidate in ("cancelAcknowledged", "cancelRequested", "complete",
                      "running", "queued", "error"):
        if candidate in normalized.lower():
            status = candidate
            break
    return {"kernel": resolved, "status": status, "raw": output.strip()}


GPU_KERNEL_KINDS = {"train": (TRAIN_KERNEL_CODE_FILE, TRAIN_KERNEL_BODY),
                    "embed": (EMBED_KERNEL_CODE_FILE, EMBED_KERNEL_BODY)}


def stage_gpu_kernel(*, kind: str, slug: str | None = None,
                     revision: str | None = None,
                     run_tag: str | None = None,
                     checkpoint: str | None = None,
                     checkout_paths: list[str] | None = None) -> dict[str, Any]:
    """Stage a GPU kernel (train | embed) with the CPU bundle attached.

    Dry-safe: metadata + generated script + receipt under the staging area;
    `--execute` makes push_kernel() perform the network call. The train
    kernel attaches the CPU bundle kernel via kernel_sources (Kaggle mounts
    its output under /kaggle/input); the embed kernel additionally needs an
    embedding request dataset slug (kaggle.embedding_dataset_slug).
    """
    spec = _spec()
    if kind not in GPU_KERNEL_KINDS:
        raise RuntimeError(f"unknown GPU kernel kind: {kind}")
    code_file, body = GPU_KERNEL_KINDS[kind]
    resolved_slug = slug or (spec.embedding_kernel_slug if kind == "embed"
                             else spec.gpu_kernel_slug)
    if not resolved_slug:
        raise RuntimeError(
            f"config kaggle.{kind}_kernel_slug is unset; name the {kind} "
            "kernel (owner/slug) before staging")
    bundle_slug = spec.cpu_kernel_slug
    if not bundle_slug:
        raise RuntimeError("config kaggle.cpu_kernel_slug is unset; the GPU "
                           "kernel attaches the CPU bundle kernel output")
    pinned = revision or _git_revision()
    tag = run_tag or (spec.run_tag_prefix + time.strftime("%m%dT%H%M%S", time.gmtime()))
    resolved_checkpoint = checkpoint or spec.checkpoint
    stage = staging_dir() / f"{kind}_kernel"
    stage.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "id": resolved_slug,
        "title": resolved_slug.rsplit("/", 1)[-1].replace("-", " ").title(),
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "is_private": True,
    }
    if kind == "embed":
        request_dataset = spec.embedding_dataset_slug
        if not request_dataset:
            raise RuntimeError(
                "config kaggle.embedding_dataset_slug is unset; package + "
                "upload the embedding request dataset first (--what package "
                "--dataset-csv ... ; then set the slug)")
        metadata["dataset_sources"] = [request_dataset]
    else:
        metadata["kernel_sources"] = [bundle_slug]
    template = TRAIN_KERNEL_SHARED + body
    script = (template
              .replace("@REPOSITORY@", spec.repository)
              .replace("@BRANCH@", spec.branch)
              .replace("@REVISION@", pinned)
              .replace("@CHECKOUT_PATHS@",
                       json.dumps(checkout_members(checkout_paths or spec.checkout_paths, lane="training")))
              .replace("@RUNTIME_PREFLIGHT@", "\n".join(
                  "    " + line for line in checkout_preflight_script(
                      checkout_inventory(checkout_paths or spec.checkout_paths, lane="training")).splitlines()))
              .replace("@REQUIREMENTS@", spec.bundle_requirements)
              .replace("@RUN_TAG@", tag)
              .replace("@SUITE_CONFIG@", spec.train_suite_config)
              .replace("@BUNDLE_KERNEL_SLUG@", bundle_slug)
              .replace("@CHECKPOINT@", resolved_checkpoint))
    atomic_write_json(metadata, stage / "kernel-metadata.json")
    (stage / code_file).write_text(script, encoding="utf-8")
    receipt = {
        "kernel": resolved_slug,
        "kind": kind,
        "gpu": True,
        "branch": spec.branch,
        "revision": pinned,
        "run_tag": tag,
        "bundle_kernel": bundle_slug,
        "checkpoint": resolved_checkpoint if kind == "embed" else None,
        "staged": str(stage),
        "code_file": code_file,
    }
    atomic_write_json(receipt, stage / f"{kind}_kernel.receipt.json")
    return receipt


def push_kernel(stage_dir: Path) -> dict[str, Any]:
    """Push any staged kernel (bundle | train | embed) via the CLI."""
    spec = _spec()
    from core.runtime_inputs import staged_kernel_preflight
    staged_kernel_preflight(Path(stage_dir))
    executable = _require_kaggle_executable(spec.kaggle_executable)
    metadata = json.loads((Path(stage_dir) / "kernel-metadata.json").read_text())
    command = [executable, "kernels", "push", "-p", str(stage_dir)]
    _, _ = _run_kaggle(command)
    return {"mode": "executed", "kernel": metadata["id"], "pushed": True,
            "staged": str(stage_dir)}


def fetch_kernel_output(*, kind: str = "bundle", execute: bool,
                        cohort: str | None = None) -> dict[str, Any]:
    """Download a kernel's output and hash-verify its manifest archive.

    bundle — bundle.receipt.json contract (all_tracks_inputs.tar.zst)
    train  — result_manifest.json contract (result_bundle.tar.zst)
    embed  — result_manifest.json contract (vectors.tar.zst)
    Verified artifacts install under staging_dir/<cohort>/<kind>/; the
    destination cohort tag comes from the explicit override (for the bundle
    lane's per-cohort fetch) or from the config dataset binding, matching
    the kernel receipt's own cohort tag when it declares one.
    """
    spec = _spec()
    if kind == "bundle":
        slug = spec.cpu_kernel_slug
    elif kind == "train":
        slug = spec.gpu_kernel_slug
    else:
        slug = spec.embedding_kernel_slug
    if not slug:
        raise RuntimeError(f"config kaggle kernel slug for {kind!r} is unset")
    manifest_name = ("bundle.receipt.json" if kind == "bundle"
                     else f"{kind}.manifest.json")
    archive_name = ("all_tracks_inputs.tar.zst" if kind == "bundle"
                    else f"{kind}.tar.zst")
    stage = staging_dir() / f"{kind}_fetch"
    plan: dict[str, Any] = {
        "mode": "executed" if execute else "dry-run",
        "kernel": slug,
        "kind": kind,
        "stage": str(stage),
    }
    if not execute:
        return plan
    executable = _require_kaggle_executable(spec.kaggle_executable)
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    command = [executable, "kernels", "output", slug, "-p", str(stage)]
    _, _ = _run_kaggle(command)
    manifests = sorted(stage.rglob(manifest_name))
    if not manifests:
        zips = sorted(stage.glob("*.zip"))
        if zips:
            with zipfile.ZipFile(zips[0]) as bundle:
                bundle.extractall(stage / "unpacked")
            manifests = sorted((stage / "unpacked").rglob(manifest_name))
    if not manifests:
        raise RuntimeError(
            f"kernel output under {stage} contained no {manifest_name}")
    manifest_dir = manifests[0].parent
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    archive = manifest_dir / archive_name
    if not archive.is_file():
        raise FileNotFoundError(
            f"kernel output manifest names {archive_name} but it is missing "
            f"under {manifest_dir}")
    observed = sha256_file(archive)
    sidecar = manifest_dir / f"{archive_name}.sha256"
    expected = (sidecar.read_text().strip() if sidecar.is_file()
                else manifest.get("archive_sha256"))
    if not expected or observed != expected:
        raise RuntimeError(
            f"fetched {kind} sha256 mismatch: expected {expected} observed "
            f"{observed}")
    from core.common import F

    receipt_cohort = (manifest if kind == "bundle"
                      else json.loads(manifests[0].read_text())).get("cohort")
    resolved_cohort = cohort or receipt_cohort or cohort_label(Path(F["dataset"]))
    destination = staging_dir() / resolved_cohort / kind
    destination.mkdir(parents=True, exist_ok=True)
    installed = {}
    shutil.copy2(archive, destination / archive.name)
    installed[archive_name] = str(destination / archive.name)
    sidecar_names = [manifest_name, f"{archive_name}.sha256"]
    if kind == "bundle":
        sidecar_names += ["manifest.json", "timings.json"]
    for name in sidecar_names:
        sidecar_path = manifest_dir / name
        if sidecar_path.is_file():
            shutil.copy2(sidecar_path, destination / name)
            installed[name] = str(destination / name)
    plan.update({
        "fetched_archive": str(archive),
        "archive_sha256": observed,
        "verified": True,
        "cohort": resolved_cohort,
        "installed": installed,
    })
    return plan


def stop_kernel(slug: str | None = None, *, which: str = "cpu",
                execute: bool) -> dict[str, Any]:
    """Stop a kernel's running session and release its compute quota.

    Kaggle's official CLI exposes no cancel verb and the cancel-session API
    needs a session id the public surfaces never report, so the reliable
    mechanism is a version replace: push a trivial stub that prints and
    exits — the platform tears down the current session to run version N+1.
    Dry-run by default; --execute performs the replace.
    """
    spec = _spec()
    resolved = slug or (spec.embedding_kernel_slug if which == "embed"
                        else spec.gpu_kernel_slug if which == "gpu"
                        else spec.cpu_kernel_slug)
    if not resolved:
        raise RuntimeError(
            f"config kaggle.{which}_kernel_slug is unset; pass a slug or "
            f"name the {which} kernel in config")
    stage = staging_dir() / f"{which}_stop"
    plan: dict[str, Any] = {
        "mode": "executed" if execute else "dry-run",
        "kernel": resolved,
        "staged": str(stage),
    }
    if not execute:
        return plan
    stage.mkdir(parents=True, exist_ok=True)
    title = resolved.rsplit("/", 1)[-1].replace("-", " ").title()
    atomic_write_json({
        "id": resolved,
        "title": title,
        "code_file": "cancel_stub.py",
        "language": "python",
        "kernel_type": "script",
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "is_private": True,
    }, stage / "kernel-metadata.json")
    (stage / "cancel_stub.py").write_text(
        'print("[kaggle-lane] run cancelled by owner; session released")\n',
        encoding="utf-8")
    executable = _require_kaggle_executable(spec.kaggle_executable)
    command = [executable, "kernels", "push", "-p", str(stage)]
    _run_kaggle(command)
    plan.update(stopped=True)
    return plan


def kernel_logs(*, slug: str, poll_seconds: float | None = None, follow: bool,
                execute: bool) -> dict[str, Any]:
    """Poll kernel status; on terminal states pull output logs locally.

    Colab streams VM stdout into local transcripts; Kaggle exposes no live
    stream, so this is the honest equivalent: status polling with the
    configured executable (cadence from config kaggle.logs_poll_seconds)
    and, on terminal states, `kernels output` fetch of the kernel's own log
    file into the staging logs dir (kaggle.logs_dir).
    """
    spec = _spec()
    resolved_poll = poll_seconds if poll_seconds is not None else spec.logs_poll_seconds
    log_dir = staging_dir() / spec.logs_dir
    plan: dict[str, Any] = {
        "kernel": slug,
        "poll_seconds": resolved_poll,
        "follow": follow,
        "log_dir": str(log_dir),
        "mode": "executed" if execute else "dry-run",
    }
    if not execute:
        return plan
    history: list[dict[str, Any]] = []
    while True:
        status = kernel_status(slug)
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _log_lane(f"[{slug}] status={status['status']}")
        history.append({"at": stamp, "status": status["status"]})
        if status["status"] in {"complete", "error", "cancelAcknowledged"} or not follow:
            break
        time.sleep(resolved_poll)
    log_dir.mkdir(parents=True, exist_ok=True)
    fetch = [executable, "kernels", "output", slug, "-p", str(log_dir / slug.replace("/", "__"))]
    try:
        _, _ = _run_kaggle(fetch)
        plan["log_fetched"] = True
    except RuntimeError as error:
        plan["log_fetched"] = False
        plan["log_error"] = str(error)[-800:]
    plan["history"] = history
    return plan



def fetch_bundle_output(*, execute: bool) -> dict[str, Any]:
    """Back-compat entry point — delegates to the generalized fetcher."""
    return fetch_kernel_output(kind="bundle", execute=execute)
def fetch_bundle_output(*, execute: bool) -> dict[str, Any]:
    """Back-compat entry point — delegates to the generalized fetcher."""
    return fetch_kernel_output(kind="bundle", execute=execute)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--what", choices=["package", "upload", "download", "submission",
                        "credentials", "bundle-kernel", "bundle-fetch", "kernel-status",
                        "train-kernel", "embed-kernel", "kernel-logs", "fetch-results",
                        "stop"],
                        default="package")
    parser.add_argument("--dataset-csv", type=Path, default=None,
                        help="cohort export to package (default: the SSOT "
                             "dataset binding)")
    parser.add_argument("--config-json", type=Path, default=None,
                        help="optional dataset_metadata.json content override")
    parser.add_argument("--submission-input", type=Path, default=None)
    parser.add_argument("--submission-output", type=Path, default=None)
    # One explicit flag flips the lane from local dry-run to the live
    # kaggle subprocess. Default stays dry-run: this box has no kaggle
    # credentials and nothing here may silently reach the network.
    parser.add_argument("--execute", action="store_true",
                        help="actually invoke the kaggle CLI (requires "
                             "credentials + configured kaggle.slug)")
    parser.add_argument("--kernel", choices=["cpu", "gpu", "embed"], default="cpu",
                        help="which configured kernel slug kernel-status "
                             "resolves (default: cpu)")
    parser.add_argument("--kind", choices=["bundle", "train", "embed"], default=None,
                        help="fetch-results: which kernel output to fetch and "
                             "verify (default: bundle)")
    parser.add_argument("--slug", default=None,
                        help="kernel-logs: explicit owner/slug (default: "
                             "resolved from --kernel)")
    parser.add_argument("--follow", action="store_true",
                        help="kernel-logs: poll until a terminal status")
    parser.add_argument("--checkpoint", default=None,
                        help="embed-kernel: git-shipped checkpoint path "
                             "(default: config kaggle.checkpoint)")
    parser.add_argument("--run-tag", default=None,
                        help="run tag for GPU kernels (default: from "
                             "config kaggle.run_tag_prefix + UTC stamp)")
    parser.add_argument("--cohort", choices=["full", "50pct", "10k"], default=None,
                        help="bundle-kernel/bundle-fetch: which root-level "
                             "cohort export the CPU kernel remaps onto "
                             "dataset.csv (default: full)")
    parser.add_argument("--key-env", default=None,
                        help="environment variable holding the Kaggle API "
                             "token (default: the configured kaggle.api_key_env)")
    parser.add_argument("--revision", default=None,
                        help="pin the bundle kernel to this git revision "
                             "(default: the current HEAD)")
    args = parser.parse_args()
    if args.what == "credentials":
        print(json.dumps(write_credentials(key_env=args.key_env, execute=args.execute),
                         indent=2), flush=True)
        if not args.execute:
            print("[kaggle-lane] dry-run only; pass --execute to write the "
                  "credential file", flush=True)
        return
    cohort_resolved = args.cohort or "full"
    if args.what == "bundle-kernel":
        receipt = stage_bundle_kernel(revision=args.revision, cohort=cohort_resolved)
        print(f"[kaggle-lane] staged bundle kernel ({cohort_resolved}): "
              f"{json.dumps(receipt, indent=2)}", flush=True)
        if args.execute:
            print(json.dumps(push_bundle_kernel(Path(receipt["staged"])), indent=2),
                  flush=True)
        else:
            print("[kaggle-lane] dry-run only; pass --execute to push the kernel",
                  flush=True)
        return
    if args.what == "bundle-fetch":
        plan = fetch_kernel_output(kind="bundle", execute=args.execute,
                                   cohort=cohort_resolved if args.cohort else None)
        print(json.dumps(plan, indent=2), flush=True)
        if not args.execute:
            print("[kaggle-lane] dry-run only; pass --execute to download the "
                  "kernel output", flush=True)
        return
    if args.what == "train-kernel" or args.what == "embed-kernel":
        kind = "train" if args.what == "train-kernel" else "embed"
        receipt = stage_gpu_kernel(
            kind=kind,
            slug=args.slug,
            revision=args.revision,
            run_tag=args.run_tag,
            checkpoint=args.checkpoint,
        )
        print(f"[kaggle-lane] staged {kind} kernel: {json.dumps(receipt, indent=2)}",
              flush=True)
        if args.execute:
            print(json.dumps(push_kernel(Path(receipt["staged"])), indent=2),
                  flush=True)
        else:
            print("[kaggle-lane] dry-run only; pass --execute to push the kernel",
                  flush=True)
        return
    if args.what == "kernel-logs":
        spec = _spec()
        resolved = args.slug or (
            spec.embedding_kernel_slug if args.kernel == "embed"
            else spec.gpu_kernel_slug if args.kernel == "gpu"
            else spec.cpu_kernel_slug)
        print(json.dumps(kernel_logs(slug=resolved, follow=args.follow,
                                     execute=args.execute), indent=2), flush=True)
        return
    if args.what == "fetch-results":
        kind = args.kind or "train"
        print(json.dumps(fetch_kernel_output(kind=kind, execute=args.execute),
                         indent=2), flush=True)
        if not args.execute:
            print("[kaggle-lane] dry-run only; pass --execute to download and "
                  "verify the result archive", flush=True)
        return
    if args.what == "kernel-status":
        print(json.dumps(kernel_status(which=args.kernel), indent=2), flush=True)
        return
    if args.what == "stop":
        spec = _spec()
        resolved = args.slug or (
            spec.embedding_kernel_slug if args.kernel == "embed"
            else spec.gpu_kernel_slug if args.kernel == "gpu"
            else spec.cpu_kernel_slug)
        print(json.dumps(stop_kernel(slug=resolved, which=args.kernel,
                                     execute=args.execute), indent=2), flush=True)
        return
    spec = _spec()
    if args.what == "submission":
        if args.submission_input is None or args.submission_output is None:
            parser.error("--what submission needs --submission-input and --submission-output")
        package_submission(args.submission_input, args.submission_output)
        return
    from core.common import F

    dataset_csv = args.dataset_csv or Path(F["dataset"])
    if args.what == "package":
        config = (json.loads(args.config_json.read_text(encoding="utf-8"))
                  if args.config_json else None)
        package = package_export(dataset_csv, config=config)
        print(
            f"[kaggle-lane] packaged {package.export_path} "
            f"rows={package.census.rows} sha256={package.census.sha256[:12]} "
            f"-> {package.archive_path}",
            flush=True,
        )
        return
    package = package_export(dataset_csv)
    if args.what == "upload":
        plan = upload_dataset(package, execute=args.execute)
    else:
        plan = download_dataset(package, execute=args.execute)
    print(f"[kaggle-lane] {args.what}: {json.dumps(plan, indent=2)}", flush=True)
    if not args.execute:
        print(
            "[kaggle-lane] dry-run only; pass --execute (with credentials "
            "and kaggle.slug configured) to touch the network",
            flush=True,
        )


if __name__ == "__main__":
    main()
