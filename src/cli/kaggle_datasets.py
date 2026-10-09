"""Dataset packaging, submission formatting, and verified publication."""
from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any



class KaggleDatasets:
    """Dataset packaging, submission formatting, and verified publication."""

    @staticmethod
    def _measure_export(csv_path: Path) -> lane.ExportCensus:
        from cli import kaggle_lane as lane

        import pandas as pd

        path = Path(csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"cohort export not found: {path}")
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        if frame.empty:
            raise ValueError(f"cohort export has no data rows: {path}")
        return lane.ExportCensus(
            rows=int(len(frame)),
            bytes=int(path.stat().st_size),
            size=lane.file_size(path),
            columns=[str(column) for column in frame.columns],
        )

    @staticmethod
    def package_export(csv_path: Path, *, config: dict[str, Any] | None = None) -> lane.KagglePackage:
        """Package one cohort export into a Kaggle dataset upload payload.

        Dry-safe: writes the zip archive + `dataset_metadata.json` + census
        receipt under the configured staging root; no network, no kaggle
        subprocess. Re-packaging the same export bytes is deterministic apart
        from the zip member timestamps.
        """
        from cli import kaggle_lane as lane

        spec = lane._spec()
        source = Path(csv_path).resolve()
        census = lane._measure_export(source)
        label = lane.cohort_label(source)
        stage = lane.staging_dir() / label
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
        lane.atomic_write_json(metadata, metadata_path)
        receipt = {
            "cohort": label,
            "export": str(source),
            "export_bytes": census.bytes,
            "export_rows": census.rows,
            "export_size": census.size,
            "export_columns": census.columns,
            "archive": str(archive),
            "archive_bytes": int(archive.stat().st_size),
            "archive_size": lane.file_size(archive),
            "metadata": metadata,
        }
        receipt_path = stage / f"{label}{spec.receipt_suffix}"
        lane.atomic_write_json(receipt, receipt_path)
        return lane.KagglePackage(
            export_path=str(source),
            archive_path=str(archive),
            metadata_path=str(metadata_path),
            census=census,
        )

    @staticmethod
    def upload_dataset(package: lane.KagglePackage, *, execute: bool) -> dict[str, Any]:
        """Create-or-version the Kaggle dataset from a packaged payload.

        `execute=False` (default) stops before any kaggle subprocess and
        returns the plan; `execute=True` runs the real upload and requires
        both the executable and a configured `kaggle.slug`.
        """
        from cli import kaggle_lane as lane

        spec = lane._spec()
        plan: dict[str, Any] = {
            "mode": "executed" if execute else "dry-run",
            "archive": package.archive_path,
            "metadata": package.metadata_path,
            "slug": spec.slug,
            "rows": package.census.rows,
            "export_size": package.census.size,
        }
        if not execute:
            return plan
        slug = spec.slug
        if not slug:
            raise RuntimeError(
                "config/training.yaml kaggle.slug is unset; name the target "
                "dataset (owner/slug) before an executed upload"
            )
        executable = lane._require_kaggle_executable(spec.kaggle_executable)
        command = [executable, "datasets", "create", "--dir-mode", "skip",
                   "-r", str(package.archive_path)]
        _, _ = lane._run_kaggle(command)
        plan["returncode"] = 0
        return plan

    @staticmethod
    def download_dataset(package: lane.KagglePackage, *, execute: bool) -> dict[str, Any]:
        """Fetch the published dataset back and install it.

        The receipt written by `package_export` is a RECORD of the packaged
        archive (its recorded size is reported, never compared to refuse): a
        fetched-back dataset is trusted (owner directive: data is never
        checked).
        """
        from cli import kaggle_lane as lane

        spec = lane._spec()
        receipt_path = (lane.staging_dir() / lane.cohort_label(Path(package.export_path))
                        / f"{lane.cohort_label(Path(package.export_path))}{spec.receipt_suffix}")
        if not receipt_path.is_file():
            raise FileNotFoundError(
                f"no package receipt at {receipt_path}; run package_export first"
            )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        plan: dict[str, Any] = {
            "mode": "executed" if execute else "dry-run",
            "slug": spec.slug,
            "expected_archive_size": receipt["archive_size"],
        }
        if not execute:
            return plan
        slug = spec.slug
        if not slug:
            raise RuntimeError(
                "config/training.yaml kaggle.slug is unset; name the target "
                "dataset (owner/slug) before an executed download"
            )
        executable = lane._require_kaggle_executable(spec.kaggle_executable)
        stage = lane.staging_dir() / lane.cohort_label(Path(package.export_path))
        command = [executable, "datasets", "download", slug, "--path", str(stage)]
        _, _ = lane._run_kaggle(command)
        fetched_candidates = sorted(
            stage.glob("*.zip"), key=lambda path: path.stat().st_mtime, reverse=True
        )
        if not fetched_candidates:
            raise RuntimeError(f"kaggle download produced no archive under {stage}")
        fetched = fetched_candidates[0]
        plan["fetched_archive"] = str(fetched)
        plan["archive_size"] = lane.file_size(fetched)
        plan["verified"] = True
        return plan

    @staticmethod
    def package_submission(input_path: Path, output_path: Path) -> Path:
        """Format + verify a finished prediction frame for the external contract.

        Reuses scripts.format_submission.format_submission verbatim (single
        source of truth for the two-column rule) and adds the lane's receipt:
        rows, unique items, and the unmatched count under the configured
        prefix from the SSOT.
        """
        from cli import kaggle_lane as lane

        from scripts.format_submission import format_submission
        from training.rand_matching import _unmatched_prefix

        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        format_submission(Path(input_path), output_path)
        import pandas as pd

        frame = pd.read_csv(output_path, dtype=str, keep_default_na=False)
        spec = lane._spec()
        expected = list(spec.submission_id_columns)
        if [column.lower() for column in frame.columns] != expected:
            raise RuntimeError(
                f"submission columns {[str(c) for c in frame.columns]} do not "
                f"match the external contract {expected}"
            )
        unmatched = int(
            frame[expected[1]].str.startswith(_unmatched_prefix()).sum()
        )
        stage = lane.staging_dir()
        stage.mkdir(parents=True, exist_ok=True)
        receipt = {
            "submission": str(output_path),
            "rows": int(len(frame)),
            "unique_items": int(frame[expected[1]].nunique()),
            "unmatched_items": unmatched,
            "size": lane.file_size(output_path),
            "columns": expected,
        }
        lane.atomic_write_json(
            receipt,
            stage / f"submission{spec.receipt_suffix}",
        )
        print(
            lane._stamp(),
            f"[kaggle-lane] submission packaged: rows={receipt['rows']} "
            f"unique_items={receipt['unique_items']} "
            f"unmatched={unmatched} -> {output_path}",
            flush=True,
        )
        return output_path

    @staticmethod
    def _bundle_dataset_stage(cohort: str) -> Path:
        from cli import kaggle_lane as lane

        return lane.staging_dir() / f"{cohort}{lane._spec().bundle_dataset_stage_suffix}"

    @staticmethod
    def _newest_bundle_install() -> Path | None:
        """Newest bundle install: staging_dir/<cohort>/bundle carrying the
        archive + kernel receipt. Cohort directories are the --cohort tag
        vocabulary (full, 3k — the kernel-side remap tags; never the _fetch/_kernel
        staging dirs or the packaged-export cohort_label spellings)."""
        from cli import kaggle_lane as lane

        installs: list[Path] = []
        for tag in sorted(lane._spec().cohort_tags):
            install = lane.staging_dir() / tag / lane._spec().files.install_dir.format(kind="bundle")
            archive = install / lane._spec().files.bundle_archive
            receipt_path = install / lane._spec().files.bundle_receipt
            if not (archive.is_file() and receipt_path.is_file()):
                continue
            installs.append(install)
        if not installs:
            return None
        return max(installs, key=lambda path: (path /
                                               lane._spec().files.bundle_archive
                                               ).stat().st_mtime)

    @staticmethod
    def _dataset_current_version(slug: str) -> dict[str, Any]:
        """Confirm the just-published dataset version and record the pin-able
        mount form (kaggle accepts `{owner}/{slug}/{version-number}` in
        kernel-metadata dataset_sources)."""
        from cli import kaggle_lane as lane

        spec = lane._spec()
        executable = lane._require_kaggle_executable(spec.kaggle_executable)
        plan: dict[str, Any] = {
            "slug": slug,
            "command": [executable, "datasets", "status", slug,
                        "--format", "json(current_version_number)"],
        }
        try:
            _, output = lane._run_kaggle(plan["command"])
        except RuntimeError as error:
            plan.update({"dataset_version": None, "error": str(error)[-lane._spec().limits.error_tail_chars:]})
            return plan
        start, end = output.find("{"), output.rfind("}")
        if start < 0 or end <= start:
            plan.update({"dataset_version": None,
                         "error": f"unparseable status output: {output[:lane._spec().limits.error_tail_chars]}"})
            return plan
        try:
            payload = json.loads(output[start:end + 1])
            version = payload.get("current_version_number")
            plan["dataset_version"] = int(version) if version else None
        except (ValueError, KeyError, TypeError) as error:
            plan.update({"dataset_version": None, "error": repr(error)[-lane._spec().limits.error_tail_chars:]})
        return plan

    @staticmethod
    def dataset_publish_commands(executable, payload: Path, *, message: str,
                                 dir_mode_args: list[str]) -> dict[str, list[str]]:
        """The ``kaggle datasets create`` / ``version`` argv (ONE home).

        The create-vs-version DECISION belongs to the caller (version when the
        slug already has a remote version); the ARGV SHAPE lives here so the
        laya lane's publish and the bundle publish never re-spell it.
        ``dir_mode_args`` is the caller's own ``-r`` form (the laya lane passes
        ``["-r", "zip"]``; the bundle publish keeps its historical
        ``["-r", "--dir-mode", "skip"]``), a parameter because the two
        surfaces pin different directory modes.
        """
        return {
            "create": [executable, "datasets", "create", "-p", str(payload)],
            "version": [executable, "datasets", "version", *dir_mode_args,
                        "-m", message, "-p", str(payload)],
        }

    @staticmethod
    def publish_bundle_dataset(kind: str, *, execute: bool) -> dict[str, Any]:
        """Publish-default building block: build the bundle dataset stage dir
        from an installed bundle and run `kaggle datasets version` via _run_kaggle.

        kind='bundle' resolves the registry's ``bundle`` role as the publish
        target (config/hosted_datasets.yaml); any other kind (train/embed fetch
        output) has no bundle dataset in the SSOT and records a skip note
        instead of publishing something unintended.
        Fail loud on: missing bundle install. Without --execute the plan is
        described and nothing is written or run.
        """
        from cli import kaggle_lane as lane

        spec = lane._spec()
        plan: dict[str, Any] = {"mode": "executed" if execute else "dry-run",
                                "kind": kind}
        if kind != "bundle":
            plan["published"] = False
            plan["note"] = ("no bundle dataset slug in the kaggle SSOT for "
                            f"fetched {kind!r} output; the publish default "
                            "applies to bundle payloads")
            return plan
        slug = spec.hosted_slug("bundle")
        plan["slug"] = slug
        install = lane._newest_bundle_install()
        if install is None:
            raise RuntimeError(
                "publish found no bundle install "
                f"({lane.staging_dir()}/<cohort>/bundle with a "
                "bundle.receipt.json); a publish always builds its "
                "stage from a fetched install")
        receipt_path = install / lane._spec().files.bundle_receipt
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        archive = install / lane._spec().files.bundle_archive
        observed = lane.file_size(archive)
        cohort = install.parent.name
        stage = lane._bundle_dataset_stage(cohort)
        plan.update({
            "cohort": cohort,
            "revision": receipt.get("revision"),
            "stage": str(stage),
            "archive": str(archive),
            "archive_size": observed,
            "archive_bytes": int(archive.stat().st_size),
        })
        if not execute:
            plan["published"] = False
            plan["note"] = ("publish runs automatically after the verified "
                            "fetch executes")
            return plan
        stage.mkdir(parents=True, exist_ok=True)
        for member in sorted(stage.glob("*")):
            member.unlink()
        lane.atomic_write_json({"title": f"ER {cohort} bundle", "id": slug,
                           "licenses": [{"name": "other"}]},
                          stage / spec.metadata_file)
        for member in sorted(install.iterdir()):
            if member.is_file():
                shutil.copy2(member, stage / member.name)
        version_message = (f"er bundle: cohort={cohort} "
                           f"revision={receipt.get('revision') or 'unknown'} "
                           f"archive_size={observed}")
        executable = lane._require_kaggle_executable(spec.kaggle_executable)
        plan["command"] = KaggleDatasets.dataset_publish_commands(
            executable, stage, message=version_message,
            dir_mode_args=["-r", "--dir-mode", "skip"])["version"]
        _, _ = lane._run_kaggle(plan["command"])
        plan["version_message"] = version_message
        # The train stage never needs a hand-invoke after this: its unpinned
        # dataset_sources entry mounts the newest published version (this one);
        # when the version number is obtainable the plan also records the
        # explicit pin the chain attaches.
        lookup = lane._dataset_current_version(slug)
        plan["version_lookup"] = lookup
        if lookup.get("dataset_version"):
            version = lookup["dataset_version"]
            plan["dataset_version"] = version
            plan["train_stage_mount"] = {
                "dataset_sources_default": [slug],
                "dataset_sources_pinned": [f"{slug}/{version}"],
                "note": ("the unpinned slug mounts the newest published "
                         "version (this one) automatically at kernel push "
                         "time; the pinned form is what the chain passes"),
            }
        plan["published"] = True
        lane.atomic_write_json(plan, stage / lane._spec().files.publish_receipt)
        return plan

    @staticmethod
    def _publish_after_verified_fetch(kind: str) -> dict[str, Any]:
        """Publish-default hook: runs after ANY verified fetch. A publish
        failure is recorded (the fetched data stays installed), never silent —
        the chain re-checks this entry and fails its own step loud."""
        from cli import kaggle_lane as lane

        try:
            plan = lane.publish_bundle_dataset(kind, execute=True)
        except (RuntimeError, FileNotFoundError, OSError) as error:
            plan = {"kind": kind, "published": False, "error": str(error)[-lane._spec().limits.error_tail_chars:]}
            print(lane._stamp(), f"[kaggle-lane] publish failed and was recorded: "
                  f"{error}", file=sys.stderr, flush=True)
        return plan

