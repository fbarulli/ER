"""Local prepared-bundle cache + prewarm (split phase of cli.colab).

The GPU train lane's pure-local CPU half: resolve one masking profile per
worker, content-address a bundle request, serve a byte-identical cache hit,
build + diet-gate + validate the bundles, upload them, and run the whole build
concurrently with the VM dependency install.  Split from cli/colab.py (the
kaggle_lane.py owner-module pattern) exactly like colab_runtime /
colab_result_sync / colab_launch.

Collaborators still owned by cli.colab (config constants, ``RESULTS``/``F``,
the input resolver, the legacy validation gates, transport) are re-read through
``_hub()`` at call time, so the legacy ``from cli import colab`` monkeypatch
surface keeps driving every phase and the running colab identity never sees a
stale second copy.  The in-flight prewarm slot stays on the hub
(``_BUNDLE_PREWARM``) for the same reason.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from core.common import TRAIN_ROOT, resolve_model, training_cfg
from core.manifest import sha256_file


def _hub():
    """The RUNNING cli.colab module (never a second import copy)."""
    hub = sys.modules.get("__colab_runtime_self__")
    if hub is not None:
        return hub
    import cli.colab as surface

    return surface


def _timed_colab(kind: str):
    """Lazy step-timing shim: cli.colab owns ``_timed_colab`` at call time."""
    def decorate(function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            return _hub()._timed_colab(kind)(function)(*args, **kwargs)
        return wrapped
    return decorate


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
    return _hub()._expand_worker_profiles(
        masking_profile or _hub()._MASKING_PROFILE, workers, "masking"
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
        self.key = _hub()._bundle_request_key(request)
        self.bundles: list[Path] | None = None
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._build, daemon=True)

    def _build(self) -> None:
        try:
            self.bundles = _hub()._build_local_training_bundles(**self.request)
        except BaseException as exc:  # re-raised in the owning run_train call
            self.error = exc
            # Also print: a prewarm abandoned by a mismatched request would
            # otherwise fail invisibly (no silent drops).
            print(_hub()._stamp(), f"[local-prepare] concurrent build failed: {exc!r}", flush=True)

    def join(self) -> list[Path]:
        self.thread.join()
        if self.error is not None:
            raise self.error
        if self.bundles is None:
            raise RuntimeError("the concurrent bundle build returned no bundles")
        return self.bundles


@_timed_colab("step")
def start_local_bundle_prewarm(**request) -> None:
    """Build this lane's prepared bundles while the VM installs its runtime."""
    surface = _hub()
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
    surface = _hub()
    prewarm, surface._BUNDLE_PREWARM = surface._BUNDLE_PREWARM, None
    if prewarm is not None and prewarm.thread.is_alive():
        print(surface._stamp(), "[local-prepare] waiting for the concurrent build to finish ...", flush=True)
        prewarm.thread.join()


def _take_prewarmed_bundles(**request) -> list[Path] | None:
    """Hand over the in-flight build when it matches this exact request."""
    surface = _hub()
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
        "profiles": _hub()._training_bundle_profiles(args.masking_profile, workers),
        "model": args.model,
        "sample": sample,
    }
    if dataset_csv is not None:
        request["dataset_csv"] = dataset_csv
    return request


_BUNDLE_CACHE_DIRNAME = "_cache"
# Every file whose content can change what a prepared bundle contains.  A miss
# on any of them must invalidate the cache: a stale bundle would train on data
# the operator did not ask for, which is the silent-staleness defect class this
# repository treats as a bug rather than an inconvenience.
_BUNDLE_SOURCE_DIRS = ("src",)
_BUNDLE_SOURCE_FILES = ("scripts/diet_manifest.py",)


def _tree_digest() -> str:
    """Digest every config file and bundle-producing source file."""
    digest = hashlib.sha256()
    paths = sorted(
        [path for name in _BUNDLE_SOURCE_DIRS for path in (TRAIN_ROOT / name).rglob("*.py")]
        + [TRAIN_ROOT / name for name in _BUNDLE_SOURCE_FILES]
        + sorted((TRAIN_ROOT / "config").glob("*"))
    )
    for path in paths:
        if not path.is_file():
            continue
        digest.update(path.relative_to(TRAIN_ROOT).as_posix().encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _bundle_model_digest(model_key: str) -> str:
    from graph_tracks.text_cache import checkpoint_hash
    return checkpoint_hash(Path(resolve_model(model_key)))


def _bundle_cache_dir(
    *,
    profiles: list[str],
    model_key: str,
    sample: int | None,
    payload: str,
    training_dataset: Path,
) -> Path | None:
    """Content address for one bundle request, or None when caching is off."""
    surface = _hub()
    if not surface._CACHE_PREPARED_BUNDLES:
        return None
    request = json.dumps(
        {
            "profiles": list(profiles),
            "model": model_key,
            "model_checkpoint_sha256": surface._bundle_model_digest(model_key),
            "sample": sample,
            "payload": payload,
            "collapsed_guardrail": surface._COLLAPSE_GUARDRAIL_PROFILE,
            "dataset_sha256": sha256_file(training_dataset),
            # These inputs are read by preparation and frozen into the bundle.
            # Dataset/source identity alone cannot detect label or canonical
            # edits made since the previous build.
            "frozen_inputs_sha256": {
                name: sha256_file(Path(surface.F[name]))
                for name in ("labeled_pairs", "canonical_records", "gate_results", "number_reference")
            },
            "sources_sha256": surface._tree_digest(),
        },
        sort_keys=True,
    )
    key = hashlib.sha256(request.encode("utf-8")).hexdigest()[:32]
    return surface.RESULTS / "prepared_training" / _BUNDLE_CACHE_DIRNAME / key


def _run_diet_gate(bundle: Path) -> int:
    """The bundle diet gate (scripts/diet_manifest.py), exit 0 pass / 2 fail."""
    if str(TRAIN_ROOT) not in sys.path:
        sys.path.insert(0, str(TRAIN_ROOT))
    from scripts.diet_manifest import main as diet_manifest_main

    return diet_manifest_main([sys.argv[0], str(bundle)])


def _bundle_manifest(bundle: Path):
    from training.prepared_bundle import load_prepared_bundle

    return load_prepared_bundle(bundle)[0]


def _cached_bundles(cache_dir: Path, *, profiles: list[str]) -> list[Path] | None:
    """The cached bundles for this request, when every worker's pair is intact.

    `load_prepared_bundle` is the validation: it re-checks the manifest against
    the current encoder-text contract and refuses a mismatch, so a cached
    bundle cannot outlive the model-input spec even if the digest missed it.
    """
    surface = _hub()
    expected = [
        cache_dir / f"worker_{number}_{profile}.pkl.gz"
        for number, profile in enumerate(profiles, start=1)
    ]
    for bundle in expected:
        if not bundle.is_file() or not bundle.with_suffix(bundle.suffix + ".json").is_file():
            return None
        try:
            surface._bundle_manifest(bundle)
        except Exception as exc:
            print(
                surface._stamp(),
                f"[local-prepare] cached bundle {bundle} is not reusable "
                f"({exc!r}); rebuilding",
                flush=True,
            )
            return None
        if surface._run_diet_gate(bundle) != 0:
            print(
                surface._stamp(),
                f"[local-prepare] cached bundle {bundle} fails the diet "
                "gate; rebuilding",
                flush=True,
            )
            return None
    return expected


def _populate_bundle_cache(cache_dir: Path, bundles: list[Path]) -> None:
    """Publish freshly built bundles under their content address."""
    surface = _hub()
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        for bundle in bundles:
            for source in (bundle, bundle.with_suffix(bundle.suffix + ".json")):
                shutil.copy2(source, cache_dir / source.name)
    except OSError as exc:
        print(surface._stamp(), f"[local-prepare] could not populate {cache_dir} ({exc!r})", flush=True)
        return
    print(surface._stamp(), f"[local-prepare] cached {len(bundles)} bundle(s) at {cache_dir}", flush=True)


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

    The build is deterministic in its inputs and expensive (531 s measured on
    this host once the payload stage is cold), and every launch rebuilt it from
    scratch.  A content-keyed cache under `results/prepared_training/_cache`
    now serves a bundle whose inputs — dataset bytes, every bundle-producing
    source file, every config file, and the requested model/profile/payload —
    are byte-identical to one already built.
    """
    surface = _hub()
    if sample is not None:
        raise ValueError(
            'legacy sampled preparation does not preserve the shared component split; '
            'use --what tracks --tracks-config '
            'results/model_tracks/smoke_20261001_128/suite.yaml --gpu CPU')
    model_key = model or str(training_cfg().training.base_model)
    training_dataset = surface._validation_input_path(
        dataset_csv or surface._COLAB.training_dataset_csv
    )
    cache_dir = surface._bundle_cache_dir(
        profiles=profiles, model_key=model_key, sample=sample, payload=payload,
        training_dataset=training_dataset,
    )
    cached = surface._cached_bundles(cache_dir, profiles=profiles) if cache_dir else None
    if cached is not None:
        for number, bundle in enumerate(cached, start=1):
            manifest = surface._bundle_manifest(bundle)
            print(
                surface._stamp(),
                f"[local-prepare] cache hit worker={number} "
                f"rows={manifest.n_df:,} payload={manifest.n_payload:,} "
                f"pos={manifest.n_pos:,} neg={manifest.n_neg:,} "
                f"sha256={manifest.sha256} bundle={bundle}",
                flush=True,
            )
        surface._legacy_validation_sources()
        surface._validate_legacy_bundle_partitions(cached)
        return cached
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
        from training.prepared_bundle import load_prepared_bundle

        manifest, _ = load_prepared_bundle(bundle)
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
            f"sha256={manifest.sha256}",
            flush=True,
        )
        bundles.append(bundle)
    if cache_dir is not None:
        surface._populate_bundle_cache(cache_dir, bundles)
    surface._legacy_validation_sources()
    surface._validate_legacy_bundle_partitions(bundles)
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
    surface = _hub()
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
    surface = _hub()
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
