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
import zipfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from core.common import TRAIN_ROOT, training_cfg
from core.manifest import atomic_write_json, sha256_file


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
    print(f"[kaggle-lane] executing: {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=TRAIN_ROOT)
    if result.returncode != 0:
        raise RuntimeError(
            f"kaggle datasets upload failed (rc={result.returncode}); "
            "see the streamed kaggle output above"
        )
    plan["returncode"] = result.returncode
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
    result = subprocess.run(command, cwd=TRAIN_ROOT)
    if result.returncode != 0:
        raise RuntimeError(
            f"kaggle datasets download failed (rc={result.returncode}); "
            "see the streamed kaggle output above"
        )
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

SCRATCH = (Path("/kaggle/tmp") if Path("/kaggle/tmp").is_dir()
           else Path(tempfile.gettempdir()) / "er_bundle")
WORKING = Path("/kaggle/working")


def sh(command, **kwargs):
    print("+ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], check=True, **kwargs)


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
sh(["git", "clone", "--filter=blob:none", "--sparse", "--depth", "1",
    "--branch", BRANCH, REPOSITORY, root])
head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                      capture_output=True, text=True, check=True).stdout.strip()
if head != REVISION:
    raise SystemExit(
        "branch tip moved: cloned %s but the kernel pins %s; regenerate the "
        "kernel (cli.kaggle_lane --what bundle-kernel)" % (head, REVISION))
# Cone-mode sparse checkout keeps every root-level file (dataset.csv,
# requirements.txt, pyproject.toml) and adds the configured directories.
sh(["git", "-C", str(root), "sparse-checkout", "set", *CHECKOUT_PATHS])
sh([sys.executable, "-m", "pip", "install", "-q", "-r", REQUIREMENTS], cwd=root)
env = {**os.environ, "PYTHONPATH": str(root / "src"), "PYTHONUNBUFFERED": "1"}
sh([sys.executable, "-m", "training.prepare_all",
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
    "archive": "all_tracks_inputs.tar.zst",
    "archive_bytes": (destination / "all_tracks_inputs.tar.zst").stat().st_size,
    "archive_sha256": sha256_file(destination / "all_tracks_inputs.tar.zst"),
}
(destination / "bundle.receipt.json").write_text(
    json.dumps(receipt, indent=2) + "\\n", encoding="utf-8")
print("[bundle-cpu] receipt: " + json.dumps(receipt, indent=2), flush=True)
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
    plan["written"] = True
    return plan


def stage_bundle_kernel(*, revision: str | None = None) -> dict[str, Any]:
    """Stage the CPU bundle-generation kernel: metadata + script + receipt.

    Dry-safe: writes into the staging area only; `push_bundle_kernel` makes
    the network call. The revision is pinned at staging time so the kernel
    clones exactly the source the package provenance will record.
    """
    spec = _spec()
    slug = spec.cpu_kernel_slug
    if not slug:
        raise RuntimeError(
            "config kaggle.cpu_kernel_slug is unset; name the CPU kernel "
            "(owner/slug) before staging")
    pinned = revision or _git_revision()
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
              .replace("@CHECKOUT_PATHS@", json.dumps(list(spec.checkout_paths)))
              .replace("@REQUIREMENTS@", spec.bundle_requirements))
    atomic_write_json(metadata, stage / "kernel-metadata.json")
    (stage / BUNDLE_KERNEL_CODE_FILE).write_text(script, encoding="utf-8")
    receipt = {
        "kernel": slug,
        "gpu": False,
        "branch": spec.branch,
        "revision": pinned,
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
    executable = _require_kaggle_executable(spec.kaggle_executable)
    command = [executable, "kernels", "push", "-p", str(stage_dir)]
    print(f"[kaggle-lane] executing: {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=TRAIN_ROOT)
    if result.returncode != 0:
        raise RuntimeError(
            f"kaggle kernels push failed (rc={result.returncode}); see the "
            "streamed kaggle output above")
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
    result = subprocess.run([executable, "kernels", "status", resolved],
                            cwd=TRAIN_ROOT, capture_output=True, text=True)
    output = ((result.stdout or "") + (result.stderr or "")).strip()
    status = "unknown"
    for candidate in ("cancelAcknowledged", "cancelRequested", "complete",
                      "running", "queued", "error"):
        if candidate in output:
            status = candidate
            break
    return {"kernel": resolved, "returncode": result.returncode,
            "status": status, "raw": output}


def fetch_bundle_output(*, execute: bool) -> dict[str, Any]:
    """Download the CPU kernel output and hash-verify the bundle archive.

    The kernel-written bundle.receipt.json is the transport-identity
    contract: the fetched all_tracks_inputs.tar.zst must match its recorded
    sha256, exactly like the dataset fetch-back. Verified artifacts install
    under staging_dir/<cohort>/bundle/.
    """
    spec = _spec()
    slug = spec.cpu_kernel_slug
    if not slug:
        raise RuntimeError(
            "config kaggle.cpu_kernel_slug is unset; name the CPU kernel "
            "(owner/slug) before fetching its output")
    stage = staging_dir() / "bundle_fetch"
    plan: dict[str, Any] = {
        "mode": "executed" if execute else "dry-run",
        "kernel": slug,
        "stage": str(stage),
    }
    if not execute:
        return plan
    executable = _require_kaggle_executable(spec.kaggle_executable)
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    command = [executable, "kernels", "output", slug, "-p", str(stage)]
    print(f"[kaggle-lane] executing: {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=TRAIN_ROOT)
    if result.returncode != 0:
        raise RuntimeError(
            f"kaggle kernels output failed (rc={result.returncode}); see the "
            "streamed kaggle output above")
    receipts = sorted(stage.rglob("bundle.receipt.json"))
    if not receipts:
        # Older kaggle CLI versions wrap kernel output in a single zip.
        zips = sorted(stage.glob("*.zip"))
        if zips:
            with zipfile.ZipFile(zips[0]) as bundle:
                bundle.extractall(stage / "unpacked")
            receipts = sorted((stage / "unpacked").rglob("bundle.receipt.json"))
    if not receipts:
        raise RuntimeError(
            f"kernel output under {stage} contained no bundle.receipt.json")
    receipt_dir = receipts[0].parent
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    archive = receipt_dir / receipt["archive"]
    if not archive.is_file():
        raise FileNotFoundError(
            f"kernel output receipt names {receipt['archive']} but it is "
            f"missing under {receipt_dir}")
    observed = sha256_file(archive)
    if observed != receipt["archive_sha256"]:
        raise RuntimeError(
            "fetched bundle sha256 mismatch: expected "
            f"{receipt['archive_sha256']} observed {observed}")
    from core.common import F

    cohort = cohort_label(Path(F["dataset"]))
    destination = staging_dir() / cohort / "bundle"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(archive, destination / archive.name)
    installed = {"bundle": str(destination / archive.name)}
    for name in ("manifest.json", "timings.json", "bundle.receipt.json"):
        sidecar = receipt_dir / name
        if sidecar.is_file():
            shutil.copy2(sidecar, destination / name)
            installed[name] = str(destination / name)
    plan.update({
        "fetched_archive": str(archive),
        "archive_sha256": observed,
        "verified": True,
        "cohort": cohort,
        "installed": installed,
    })
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--what", choices=["package", "upload", "download", "submission",
                        "credentials", "bundle-kernel", "bundle-fetch", "kernel-status"],
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
    parser.add_argument("--kernel", choices=["cpu", "gpu"], default="cpu",
                        help="which configured kernel slug kernel-status "
                             "resolves (default: cpu)")
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
    if args.what == "bundle-kernel":
        receipt = stage_bundle_kernel(revision=args.revision)
        print(f"[kaggle-lane] staged bundle kernel: {json.dumps(receipt, indent=2)}",
              flush=True)
        if args.execute:
            print(json.dumps(push_bundle_kernel(Path(receipt["staged"])), indent=2),
                  flush=True)
        else:
            print("[kaggle-lane] dry-run only; pass --execute to push the kernel",
                  flush=True)
        return
    if args.what == "bundle-fetch":
        print(json.dumps(fetch_bundle_output(execute=args.execute), indent=2), flush=True)
        if not args.execute:
            print("[kaggle-lane] dry-run only; pass --execute to download the "
                  "kernel output", flush=True)
        return
    if args.what == "kernel-status":
        print(json.dumps(kernel_status(which=args.kernel), indent=2), flush=True)
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
