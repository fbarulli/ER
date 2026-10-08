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
from datetime import datetime
from zoneinfo import ZoneInfo
import zipfile
from pathlib import Path
from typing import Any, Sequence
from pydantic import BaseModel, ConfigDict, Field
from core.common import TRAIN_ROOT, training_cfg
from core.manifest import atomic_write_json, atomic_write_text, sha256_file
from core.runtime_inputs import checkout_members, checkout_inventory, checkout_preflight_script
from cli.log_capture import progress_frames_to_lines
from cli.kaggle_lifecycle import KernelLifecycle
from cli.kaggle_kernel_templates import (
    KernelTemplates, BUNDLE_KERNEL_SCRIPT,
    TRAIN_KERNEL_SHARED, TRAIN_KERNEL_BODY,
    FINALIZE_KERNEL_BODY,
    EMBED_KERNEL_BODY,
)

KAGGLE_LANE_LOGS_SUBDIR = Path(training_cfg().kaggle.logs_dir).name

LANE_LOG_NAME = training_cfg().kaggle.files.lane_log

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

CREDENTIALS_PATH = Path.home() / training_cfg().kaggle.files.credentials_file

ACCESS_TOKEN_PATH = Path.home() / training_cfg().kaggle.files.access_token_file

#: The finalize job is the second bundle_steps role and runs as a second version
#: of the SAME bundling CPU kernel slug (one Kaggle kernel; Kaggle mounts one
#: code file per pushed version). Its two names are config-owned
#: (``kaggle.files.code_files.finalize`` / ``kaggle.files.result_names.finalize``;
#: the KaggleSpec validators require the base keys and accept finalize as the
#: optional extra), so this lane spells no literal and every surface resolves
#: them through ``kernel_identity`` below.
FINALIZE_KERNEL_KIND = "finalize"


class KernelIdentity(BaseModel):
    """One lane kernel identity, resolved against the config SSOT.

    ``kind`` is the fetch/stage/receipt kind (bundle | train | embed |
    finalize), ``which`` the watcher identity the ``--kernel`` flag takes,
    ``slug_attr`` the ``KaggleSpec`` field holding the configured slug, and
    ``code_file``/``result_name``/``manifest``/``archive`` the pushed script
    and artifact names. Bundle names its own manifest+archive pair; the others
    template ``kaggle.files`` with their result name. Every surface that used
    to rebuild a kind->slug or kind->code-file dict reads this registry.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str
    which: str
    slug_attr: str
    code_file: str
    result_name: str | None = None
    manifest: str | None = None
    archive: str | None = None
    bundle_role: str | None = None

    def slug(self, spec: Any) -> str | None:
        return getattr(spec, self.slug_attr)

    def manifest_name(self, files: Any) -> str:
        return self.manifest or files.result_manifest.format(kind=self.result_name)

    def archive_name(self, files: Any) -> str:
        return self.archive or files.result_archive.format(kind=self.result_name)


def _kernel_identities(files: Any) -> dict[str, KernelIdentity]:
    """The four lane identities, with the code/result names ``kaggle.files`` declares."""
    return {
        "bundle": KernelIdentity(
            kind="bundle", which="cpu", slug_attr="cpu_kernel_slug",
            code_file=files.code_files["bundle"],
            manifest=files.bundle_receipt, archive=files.bundle_archive,
            bundle_role="inputs"),
        "train": KernelIdentity(
            kind="train", which="gpu", slug_attr="gpu_kernel_slug",
            code_file=files.code_files["train"],
            result_name=files.result_names["train"],
            # The train kernel ships the suite's own sealed result Bundle (see
            # KaggleKernels.TRAIN_RESULT_BUNDLE_SHIP), so its fetched output is
            # role-loaded at the same boundary the finalize job consumes; the
            # embed output is not a Bundle role.
            bundle_role="result"),
        "embed": KernelIdentity(
            kind="embed", which="embed", slug_attr="embedding_kernel_slug",
            code_file=files.code_files["embed"],
            result_name=files.result_names["embed"]),
        FINALIZE_KERNEL_KIND: KernelIdentity(
            kind=FINALIZE_KERNEL_KIND, which="finalize",
            slug_attr="cpu_kernel_slug",
            code_file=files.code_files["finalize"],
            result_name=files.result_names["finalize"],
            bundle_role="result"),
    }


def kernel_identities(spec: Any | None = None) -> dict[str, KernelIdentity]:
    """The lane's kernel identities resolved from ``spec`` (default: the SSOT)."""
    spec = _spec() if spec is None else spec
    return _kernel_identities(spec.files)


def kernel_identity(ref: str, spec: Any | None = None) -> KernelIdentity:
    """The ONE identity named by ``ref`` (a kind, or a watcher ``which``)."""
    identities = kernel_identities(spec)
    index = {identity.kind: identity for identity in identities.values()}
    index.update({identity.which: identity for identity in identities.values()})
    if ref not in index:
        raise RuntimeError(
            f"unknown kernel identity {ref!r}; known: {sorted(index)}")
    return index[ref]


#: Import-time identities (module constants only; a caller holding its own spec
#: resolves through ``kernel_identity(..., spec)``).
KERNEL_IDENTITIES = _kernel_identities(training_cfg().kaggle.files)

BUNDLE_KERNEL_CODE_FILE = KERNEL_IDENTITIES["bundle"].code_file
TRAIN_KERNEL_CODE_FILE = KERNEL_IDENTITIES["train"].code_file
EMBED_KERNEL_CODE_FILE = KERNEL_IDENTITIES["embed"].code_file

GPU_KERNEL_KINDS = {"train": (TRAIN_KERNEL_CODE_FILE, TRAIN_KERNEL_BODY),
                    "embed": (EMBED_KERNEL_CODE_FILE, EMBED_KERNEL_BODY)}

COHORT_TAGS = training_cfg().kaggle.cohort_tags

# kind/which -> canonical watcher identity, one entry per kind AND per watcher
# alias. `finalize` is its own watcher identity so its receipt
# (autowatch_finalize.receipt.json) can never be mistaken for the generation
# step's (autowatch_bundle.receipt.json) — both run the CPU slug.
AUTOWATCH_WHICH = {identity.kind: identity.which
                   for identity in KERNEL_IDENTITIES.values()}
AUTOWATCH_WHICH.update({identity.which: identity.which
                        for identity in KERNEL_IDENTITIES.values()})

CHAIN_MAX_POLLS = training_cfg().kaggle.limits.max_polls  # supervise's harvest ceiling, in poll ticks
from cli.kaggle_cli import KaggleCLI

from cli.kaggle_runtime import KaggleRuntime
from cli.kaggle_datasets import KaggleDatasets
from cli.kaggle_kernels import KaggleKernels
from cli.kaggle_outputs import KaggleOutputs
from cli.kaggle_monitor import KaggleMonitor
from cli.kaggle_chain import KaggleChain

# Compatibility bindings keep existing entry points and callers on the same
# class implementations; no parallel procedural implementations are retained.
_spec = KaggleRuntime._spec
staging_dir = KaggleRuntime.staging_dir
lane_logs_dir = KaggleRuntime.lane_logs_dir
cohort_label = KaggleRuntime.cohort_label
cohort_export_csv = KaggleRuntime.cohort_export_csv
_git_revision = KaggleRuntime._git_revision
_stamp = KaggleRuntime._stamp
_log_lane = KaggleRuntime._log_lane
_require_kaggle_executable = KaggleRuntime._require_kaggle_executable
_run_kaggle = KaggleRuntime._run_kaggle
_env_dot_value = KaggleRuntime._env_dot_value
write_credentials = KaggleRuntime.write_credentials
_measure_export = KaggleDatasets._measure_export
package_export = KaggleDatasets.package_export
upload_dataset = KaggleDatasets.upload_dataset
download_dataset = KaggleDatasets.download_dataset
package_submission = KaggleDatasets.package_submission
_bundle_dataset_stage = KaggleDatasets._bundle_dataset_stage
_cohort_marker_conflict = KaggleDatasets._cohort_marker_conflict
_newest_bundle_install = KaggleDatasets._newest_bundle_install
_dataset_current_version = KaggleDatasets._dataset_current_version
publish_bundle_dataset = KaggleDatasets.publish_bundle_dataset
_publish_after_verified_fetch = KaggleDatasets._publish_after_verified_fetch
_kernel_script_gate = KaggleKernels._kernel_script_gate
_attachment_gate = KaggleKernels._attachment_gate
stage_bundle_kernel = KaggleKernels.stage_bundle_kernel
push_bundle_kernel = KaggleKernels.push_bundle_kernel
stage_gpu_kernel = KaggleKernels.stage_gpu_kernel
stage_finalize_kernel = KaggleKernels.stage_finalize_kernel
embed_objective = KaggleKernels.embed_objective
require_embed_objective = KaggleKernels.require_embed_objective
push_kernel = KaggleKernels.push_kernel
kernel_status = KaggleKernels.kernel_status
stop_kernel = KaggleKernels.stop_kernel
fetch_kernel_output = KaggleOutputs.fetch_kernel_output
fetch_failed_kernel_log = KaggleOutputs.fetch_failed_kernel_log
fetch_bundle_output = KaggleOutputs.fetch_bundle_output
_spawn_autowatch = KaggleMonitor._spawn_autowatch
autowatch_kernel = KaggleMonitor.autowatch_kernel
supervise_kernels = KaggleMonitor.supervise_kernels
stream_kernel_logs = KaggleMonitor.stream_kernel_logs
clear_kernel_session_id = KaggleMonitor.clear_kernel_session_id
capture_kernel_session_id = KaggleMonitor.capture_kernel_session_id
kernel_logs = KaggleMonitor.kernel_logs
_await_autowatch_receipt = KaggleChain._await_autowatch_receipt
_clear_stale_autowatch_receipt = KaggleChain._clear_stale_autowatch_receipt
_verify_chain_step = KaggleChain._verify_chain_step
run_chain = KaggleChain.run_chain
main = KaggleCLI.run

if __name__ == "__main__":
    # The CLI dispatcher imports the canonical module name.
    sys.modules["cli.kaggle_lane"] = sys.modules[__name__]
    main()
