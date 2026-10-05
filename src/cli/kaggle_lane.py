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
    if name == Path(DATA_FILE_DEFAULT).name:
        return "full"
    if "50pct" in name:
        return "50pct"
    return "".join(
        ch if ch.isalnum() or ch in "-_" else "_" for ch in name.rsplit(".", 1)[0]
    )


# The SSOT default export binding ("dataset" in config/paths.yaml files:).
# Resolved lazily to keep the module import-light for --help paths.
def _default_export_name() -> str:
    from core.common import F

    return Path(F["dataset"]).name


DATA_FILE_DEFAULT = "dataset.csv"


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
    fetched = stage / f"{slug.split('/')[-1]}.zip"
    if not fetched.is_file():
        raise RuntimeError(f"kaggle download produced no archive at {fetched}")
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--what", choices=["package", "upload", "download", "submission"],
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
    args = parser.parse_args()
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
