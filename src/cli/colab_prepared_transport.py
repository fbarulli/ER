"""GPU lane prepared-bundle transport (split phase of cli.colab).

The GPU train lane's bundle half, and transport ONLY: resolve the masking
profiles one lane's workers are configured for, and push the bundles the CPU
bundle lane produced (``ColabGPULane.prepared_bundles``) to the VM.

The bundle BUILD is not here and never was the launcher's business: the CPU
bundle lane runs ``training.prepare_all`` on the CPU VM and its delivery
installs the bundles (``config colab.full_prepared_bundles``).  The local
``training.train --prepare-bundle`` subprocess, its concurrent prewarm thread
and its diet gate were deleted with the legacy CPU prepping (owner directive:
all bundling and prepping goes to the Colab CPU bundle process, all GPU
training to the GPU VM, and the local machine prepares nothing).

Split from cli/colab.py (the kaggle_lane.py owner-module pattern) exactly like
colab_runtime / colab_result_sync / colab_launch.

Collaborators still owned by cli.colab (config constants, ``RESULTS``/``F``,
the input resolver, transport) are re-read through ``colab_hub.hub()`` at call
time, so the legacy ``from cli import colab`` monkeypatch surface keeps driving
every phase and the running colab identity never sees a stale second copy.
"""
from __future__ import annotations

from pathlib import Path

from cli.colab_hub import hub


def _expand_worker_profiles(raw: str, workers: int, label: str) -> list[str]:
    """Resolve one profile or one explicit profile per concurrent worker."""
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if len(values) == 1:
        return values * workers
    if len(values) != workers:
        raise ValueError(
            f"{label} profile count must be 1 or exactly {workers}; got {len(values)}"
        )
    return values


def _training_bundle_profiles(masking_profile: str | None, workers: int) -> list[str]:
    """Resolve the masking profiles one lane's prepared bundles are built for.

    Single definition shared by `run_train` (which sends the profiles to the
    remote workers) and `main`, so the two can never disagree.
    """
    return hub()._expand_worker_profiles(
        masking_profile or hub()._MASKING_PROFILE, workers, "masking"
    )


def _upload_prepared_bundles(
    *,
    run_id: str,
    bundles: list[Path],
) -> list[str]:
    """Upload only the prepared bundles and their manifests."""
    surface = hub()
    remote_dir = f"{surface.REMOTE_ROOT}/prepared_training/{run_id}"
    worker_dirs = [
        f"{remote_dir}/worker_{number}" for number in range(1, len(bundles) + 1)
    ]
    # One exec for every worker directory. Each notebook exec carries ~1.5 s of
    # round trip (measured 1.43-1.63 s in the 2026-09-15 T4 history), so the
    # former per-file mkdir made setup latency grow with the worker count while
    # creating exactly the same directories.
    surface.run_colab_exec_stream(
        surface.SESSION,
        surface._BOOTSTRAP
        + f"""
import pathlib
for worker_dir in {worker_dirs!r}:
    pathlib.Path(worker_dir).mkdir(parents=True, exist_ok=True)
""",
        timeout=120,
        log_name="prepared_bundle_mkdir",
        retry_safe=False,
    )
    remote_paths: list[str] = []
    for number, bundle in enumerate(bundles, start=1):
        for source in (bundle, bundle.with_suffix(bundle.suffix + ".json")):
            remote = f"{remote_dir}/worker_{number}/{source.name}"
            print(surface._stamp(), f"[upload] prepared bundle file={source} -> {remote}", flush=True)
            surface._upload_with_retries(
                source, remote, timeout=surface._RESULT_DOWNLOAD_TIMEOUT_SECONDS
            )
        remote_paths.append(f"{remote_dir}/worker_{number}/{bundle.name}")
    return remote_paths
