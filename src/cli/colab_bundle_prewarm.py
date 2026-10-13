"""Local prepared-bundle build + prewarm (split phase of cli.colab).

The GPU train lane's pure-local CPU half: resolve one masking profile per
worker, build + diet-gate + validate the bundles, upload them, and run the
whole build concurrently with the VM dependency install.  Split from
cli/colab.py (the kaggle_lane.py owner-module pattern) exactly like
colab_runtime / colab_result_sync / colab_launch.

There is exactly ONE prepared bundle for the full training set (owner
directive 2026-10-08): any input change rebuilds the whole bundle, so no build
is reused across runs and no cross-run cache exists.

Collaborators still owned by cli.colab (config constants, ``RESULTS``/``F``,
the input resolver, the legacy validation gates, transport) are re-read through
``colab_hub.hub()`` at call time, so the legacy ``from cli import colab`` monkeypatch
surface keeps driving every phase and the running colab identity never sees a
stale second copy.  The in-flight prewarm slot stays on the hub
(``_BUNDLE_PREWARM``) for the same reason.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from core.common import TRAIN_ROOT, training_cfg
from cli.colab_hub import hub, timed_colab


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

    Single definition shared by `run_train` (which builds the bundles) and
    `main` (which may start that build early), so a prewarm can never be built
    for a different request than the run asks for.
    """
    return hub()._expand_worker_profiles(
        masking_profile or hub()._MASKING_PROFILE, workers, "masking"
    )


def _bundle_request_key(request: dict) -> tuple:
    """Identity of one local bundle request, for prewarm reuse checks."""
    return (
        tuple(request["profiles"]),
        request["model"],
        request["sample"],
        request.get("dataset_csv"),
        request.get("payload", "full"),
    )


class _BundlePrewarm:
    """One local bundle build running while the VM provisions and installs.

    The build is pure local CPU work over immutable local inputs; nothing on
    the VM feeds it, so it can overlap the remote deps stage instead of
    following it.  Measured on the 2026-09-15 T4 launch: 33.3 s of local
    bundle construction sat strictly after a 62.7 s remote install.
    """

    def __init__(self, request: dict) -> None:
        self.request = request
        self.key = hub()._bundle_request_key(request)
        self.bundles: list[Path] | None = None
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._build, daemon=True)

    def _build(self) -> None:
        try:
            self.bundles = hub()._build_local_training_bundles(**self.request)
        except BaseException as exc:  # re-raised in the owning run_train call
            self.error = exc
            # Also print: a prewarm abandoned by a mismatched request would
            # otherwise fail invisibly (no silent drops).
            print(hub()._stamp(), f"[local-prepare] concurrent build failed: {exc!r}", flush=True)

    def join(self) -> list[Path]:
        self.thread.join()
        if self.error is not None:
            raise self.error
        if self.bundles is None:
            raise RuntimeError("the concurrent bundle build returned no bundles")
        return self.bundles


@timed_colab("step")
def start_local_bundle_prewarm(**request) -> None:
    """Build this lane's prepared bundles while the VM installs its runtime."""
    surface = hub()
    prewarm = _BundlePrewarm(request)
    surface._BUNDLE_PREWARM = prewarm
    prewarm.thread.start()
    print(
        surface._stamp(),
        "[local-prepare] building "
        f"{len(request['profiles'])} bundle(s) concurrently with the VM "
        "dependency install",
        flush=True,
    )


def drain_local_bundle_prewarm() -> None:
    """Never leave a prewarm thread writing into a closing live log."""
    surface = hub()
    prewarm, surface._BUNDLE_PREWARM = surface._BUNDLE_PREWARM, None
    if prewarm is not None and prewarm.thread.is_alive():
        print(surface._stamp(), "[local-prepare] waiting for the concurrent build to finish ...", flush=True)
        prewarm.thread.join()


def _take_prewarmed_bundles(**request) -> list[Path] | None:
    """Hand over the in-flight build when it matches this exact request."""
    surface = hub()
    prewarm, surface._BUNDLE_PREWARM = surface._BUNDLE_PREWARM, None
    if prewarm is None:
        return None
    if prewarm.key != surface._bundle_request_key(request):
        print(
            surface._stamp(),
            "[local-prepare] concurrent build was started for a different "
            f"request {prewarm.key}; rebuilding for {surface._bundle_request_key(request)}",
            flush=True,
        )
        return None
    print(surface._stamp(), "[local-prepare] joining the build started before the VM setup", flush=True)
    return prewarm.join()


def _lane_bundle_request(args: argparse.Namespace) -> dict | None:
    """The exact local bundle request a lane's `run_train` call will make.

    Mirrors the per-lane worker/sample arguments in `main`; the profile
    derivation itself comes from `_training_bundle_profiles`, the same
    function `run_train` uses.  Returns None for lanes that prepare no
    bundles (sims, mixed, hpo, stop).
    """
    if args.what == "dual-train":
        workers, sample = 2, args.sample
        dataset_csv = None
    elif args.what == "train":
        if (
            args.workers == 1
            and args.model is None
            and args.sample is None
            and args.resume_run is None
        ):
            return None
        workers, sample = args.workers, args.sample
        dataset_csv = None
    else:
        return None
    request = {
        "profiles": hub()._training_bundle_profiles(args.masking_profile, workers),
        "model": args.model,
        "sample": sample,
    }
    if dataset_csv is not None:
        request["dataset_csv"] = dataset_csv
    return request


def _run_diet_gate(bundle: Path) -> int:
    """The bundle diet gate (scripts/diet_manifest.py), exit 0 pass / 2 fail."""
    if str(TRAIN_ROOT) not in sys.path:
        sys.path.insert(0, str(TRAIN_ROOT))
    from scripts.diet_manifest import main as diet_manifest_main

    return diet_manifest_main([sys.argv[0], str(bundle)])


def _bundle_manifest(bundle: Path):
    from training.prepared_bundle import load_prepared_bundle

    return load_prepared_bundle(bundle)[0]


def _build_local_training_bundles(
    *,
    profiles: list[str],
    model: str | None,
    sample: int | None,
    dataset_csv: str | None = None,
    payload: str = "full",
) -> list[Path]:
    """Build and validate one complete input bundle per worker locally.

    The single bundle builder.  It deliberately does NOT consult the prewarm:
    it is what the prewarm thread itself runs, so looking the prewarm up here
    would make that thread join itself.

    Every build lands under its own timestamped ``results/prepared_training/<run>``
    directory.  There is exactly ONE prepared bundle for the full training set,
    so an input change rebuilds the whole bundle and no build is ever reused
    across runs (owner directive 2026-10-08; cross-run caching is forbidden).
    """
    surface = hub()
    if sample is not None:
        raise ValueError(
            'legacy sampled preparation does not preserve the shared component split; '
            'use --what tracks --tracks-config '
            'results/model_tracks/smoke_20261001_128/suite.yaml --gpu CPU')
    model_key = model or str(training_cfg().training.base_model)
    training_dataset = surface._validation_input_path(
        dataset_csv or surface._COLAB.training_dataset_csv
    )
    stamp = datetime.now(timezone.utc).strftime("%m%dT%H%M%S%fZ")
    root = surface.RESULTS / "prepared_training" / stamp
    root.mkdir(parents=True, exist_ok=False)
    bundles: list[Path] = []
    for number, profile in enumerate(profiles, start=1):
        bundle = root / f"worker_{number}_{profile}.pkl.gz"
        command = [
            sys.executable,
            "-u",
            "-m",
            "training.train",
            "--model",
            model_key,
            "--dataset",
            str(training_dataset),
            "--payload",
            payload,
            "--masking-profile",
            profile,
            "--collapse-guardrail-profile",
            surface._COLLAPSE_GUARDRAIL_PROFILE,
            "--prepare-bundle",
            str(bundle),
            "--no-mask-effect",
            "--no-plot",
        ]
        if sample is not None:
            command.extend(["--sample", str(sample)])
        env = {
            **os.environ,
            "PYTHONPATH": str(TRAIN_ROOT / "src"),
            "WANDB_MODE": "offline",
            "EUROMONITOR_RUN_ID": f"local-prepare-{stamp}-worker_{number}",
        }
        print(
            surface._stamp(),
            f"[local-prepare] worker={number} profile={profile} "
            f"bundle={bundle}",
            flush=True,
        )
        subprocess.run(command, cwd=TRAIN_ROOT, env=env, check=True)
        manifest = surface._bundle_manifest(bundle)
        if surface._run_diet_gate(bundle) != 0:
            raise SystemExit(
                f"[local-prepare] bundle {bundle} FAILED the diet gate; the "
                "rebuild refuses to ship training inputs that violate "
                "config/training.yaml's diet contract"
            )
        print(
            surface._stamp(),
            f"[local-prepare] validated worker={number} "
            f"rows={manifest.n_df:,} payload={manifest.n_payload:,} "
            f"pos={manifest.n_pos:,} neg={manifest.n_neg:,} "
            f"size={manifest.size}",
            flush=True,
        )
        bundles.append(bundle)
    surface._legacy_validation_sources()
    return bundles


def _prepare_local_training_bundles(
    *,
    profiles: list[str],
    model: str | None,
    sample: int | None,
    dataset_csv: str | None = None,
    payload: str = "full",
) -> list[Path]:
    """Bundles for one training call: the in-flight build when it matches.

    The only entry point that consumes a prewarm, so the build it hands over
    runs once and the overlap in `start_local_bundle_prewarm` is real.
    """
    surface = hub()
    request = {
        "profiles": profiles,
        "model": model,
        "sample": sample,
        "payload": payload,
    }
    if dataset_csv is not None:
        request["dataset_csv"] = dataset_csv
    prewarmed = surface._take_prewarmed_bundles(**request)
    if prewarmed is not None:
        return prewarmed
    return surface._build_local_training_bundles(**request)


def _upload_prepared_bundles(
    *,
    run_id: str,
    bundles: list[Path],
) -> list[str]:
    """Upload only the locally prepared bundles and their manifests."""
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
