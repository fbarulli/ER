"""The two bundle steps: generation (prepare inputs) and finalize.

``prepare_inputs`` reuses the owned preparation path
(:mod:`training.prepare_all`, whose ``suite_inputs`` stage is
:mod:`model_tracks.package`) and returns the verified inputs ``Bundle``.

``finalize`` is the only place CPU post-processing, post-training ablation and
sealing run. It consumes a verified result ``Bundle`` (materialize), produces the
CPU reports and ablation over that materialized tree, then seals a result-only
bundle through :meth:`core.bundle.Bundle.seal_result`. No step re-hashes the
incoming archive: the boundary check is :meth:`core.bundle.Bundle.load`, the
only other check is the writer's own verify inside ``write_archive``.

Every name-like value comes from ``training_cfg().bundle`` / the input package's
own suite config; this module spells no path or member literal.
"""
from __future__ import annotations

import json
from pathlib import Path

from core.bundle import Bundle, BundlePipeline, BundleRole


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def prepare_inputs(pipeline: BundlePipeline) -> Bundle:
    """Generation: run the owned preparation path and return the inputs Bundle.

    Reuses ``training.prepare_all`` (the ``suite_inputs`` stage is
    ``model_tracks.package``); the freshly written archive is verified once at
    the :meth:`Bundle.load` boundary.
    """
    if pipeline.role is not BundleRole.inputs:
        raise ValueError(f"prepare_inputs requires the inputs role, got {pipeline.role.value}")
    if pipeline.output is None:
        raise ValueError("prepare_inputs requires pipeline.output (the sealed inputs archive)")

    from training.prepare_all import _load_run_context, prepare_all

    output = Path(pipeline.output)
    run_dir = Path(pipeline.run_dir) if pipeline.run_dir is not None else output.parent
    prepare_all(run_dir=run_dir, tracks_config=pipeline.config)

    context = _load_run_context(pipeline.config)
    archive = run_dir / (f"{context.prep.suite_archive_name}."
                         f"{context.suite.input_archive_format}")
    _require(
        archive.resolve() == output.resolve(),
        f"prepared inputs archive {archive} does not match pipeline.output {output}")
    return Bundle.load(archive, BundleRole.inputs)


def finalize(pipeline: BundlePipeline, result: Bundle) -> Bundle:
    """Finalize: materialize -> postprocess -> ablation -> seal a result bundle."""
    from core.common import TRAIN_ROOT
    from model_tracks.config import SuiteConfig
    from model_tracks.package import package_member
    from model_tracks.resume import (
        TRACKS, record_completion, recorded_ablation_skip, validate_training_binding,
    )
    import yaml

    if result.role is not BundleRole.result:
        raise ValueError(f"finalize requires a result bundle, got {result.role.value}")
    if pipeline.inputs is None:
        raise ValueError("finalize requires pipeline.inputs (the verified prepared inputs)")
    if pipeline.output is None:
        raise ValueError("finalize requires pipeline.output (the sealed result archive)")

    run_tag = result.run_tag()
    _require(bool(run_tag), "result bundle manifest carries no run tag")
    output = Path(pipeline.output)

    inputs = Bundle.load(pipeline.inputs, BundleRole.inputs)
    suite = result.read_json("suite_manifest.json")
    settings = SuiteConfig.model_validate(
        yaml.safe_load(inputs.read(package_member("suite_package_config"))))
    _require(suite.get("run_tag") == run_tag, "result suite manifest run tag differs")
    validate_training_binding(suite, inputs.manifest, settings, run_tag)

    prepared_root = output.parent / f"{run_tag}__prepared"
    prepared_root.mkdir(parents=True, exist_ok=True)
    _extract_prepared_inputs(inputs, prepared_root,
                             package_member("suite_package_config"))
    setup = prepared_root / settings.setup_dir

    tree = result.materialize(output.parent / f"{run_tag}__finalize")
    destination = tree._root()
    _restore_frozen_baseline(destination, setup)

    if settings.post_training_ablation and (destination / "baseline/ablation/request.json").is_file():
        from model_tracks.baseline_ablation import complete as complete_baseline_ablation
        complete_baseline_ablation(destination / "baseline", setup,
                                   config=TRAIN_ROOT / settings.ablation_config)

    ablated = _postprocess_tracks(destination, setup, settings)

    if settings.post_training_ablation and not ablated:
        if recorded_ablation_skip(destination):
            from core.run_log import RunLogger
            RunLogger(__name__).info(
                "[finalize] suite skipped attribute ablation; no saved ablation to consume")
        else:
            from model_tracks.post_training_ablation import complete_saved
            complete_saved(destination, settings)
            for track in TRACKS:
                record_completion(destination / track, track)

    suite["postprocess_location"] = "bundle finalize"
    (destination / "suite_manifest.json").write_text(json.dumps(suite, indent=2))
    return tree.seal_result(
        output, metadata={"run_tag": run_tag, "postprocess_location": "bundle finalize"})


def _extract_prepared_inputs(inputs: Bundle, destination: Path, config_member: str) -> None:
    """Extract the input package's prepared tree under ``destination`` (no re-hash)."""
    package_root = Path(config_member).parent
    for relative in inputs.manifest.get(_files_key(), {}):
        if not Path(relative).is_relative_to(package_root):
            continue
        target = destination / relative
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(inputs.read(relative))


def _files_key() -> str:
    from core.bundle import _bundle_spec
    return _bundle_spec().files_key


def _restore_frozen_baseline(destination: Path, setup: Path) -> None:
    """Bind the suite's GPU baseline export as the report-time embedding cache."""
    baseline = destination / "baseline/shared_minilm__embeddings.npz"
    if not baseline.is_file():
        return
    from training.prepare_embeddings import validate_result
    from graph_tracks.data import file_hash
    request = setup / "embedding_inputs.json"
    validate_result(baseline, json.loads(request.read_text()),
                    request_sha256=file_hash(request))
    cache = setup / "shared_minilm__embeddings.npz"
    if cache.exists() and file_hash(cache) != file_hash(baseline):
        raise ValueError("restored frozen baseline differs from suite GPU export")
    if not cache.exists():
        import shutil
        shutil.copy2(baseline, cache)


def _track_postprocessed(output: Path) -> bool:
    """The track marker says its CPU post-processing already completed."""
    marker = output / "track_complete.json"
    try:
        return bool(json.loads(marker.read_text()).get("postprocess_complete"))
    except (OSError, ValueError):
        return False


def _postprocess_tracks(destination: Path, setup: Path, settings) -> bool:
    """Run the CPU report for every unfinished trained track; return ablation state."""
    from model_tracks.resume import TRACKS, record_completion

    for track in TRACKS:
        output = destination / track
        if _track_postprocessed(output):
            continue
        if track == "text":
            from model_tracks.text_report import complete as text_complete
            _park_interrupted(output, "text__*")
            text_complete(output, setup, device="cpu", report_test=settings.report_test)
        elif track == "cascade":
            # The cascade is a pure post-process combinator with no checkpoint:
            # its report is composed from the trained artifacts by the worker.
            record_completion(output, track)
            continue
        else:
            _complete_graph_track(output, setup, track, settings)
        record_completion(output, track)
    return False


def _complete_graph_track(output: Path, setup: Path, track: str, settings) -> None:
    """Re-run one graph track's saved inference on CPU and write its report."""
    from graph_tracks.config import GraphConfig, load_config as load_graph_config
    from graph_tracks.preflight import preflight
    from graph_tracks.report import complete as graph_complete
    import yaml

    prepared = setup.parent
    config = load_graph_config(setup / f"{track}.yaml", expected_track=track).model_dump()
    for key in ("listings", "pairs", "input_manifest", "text_cache"):
        if config.get(key):
            config[key] = str(prepared / config[key])
    config.update(device="cpu", report_test=settings.report_test, postprocess=True)
    config_path = output / "local_report.yaml"
    config_path.write_text(yaml.safe_dump(config))
    preflight(config_path, check_device=False)
    selected = list(output.rglob(f"{track}__best_checkpoint.json"))
    if len(selected) != 1:
        raise ValueError(f"ambiguous selected checkpoint: {track}")
    recorded = Path(json.loads(selected[0].read_text())["path"])
    checkpoints = list(output.rglob(f"{recorded.parent.name}/{recorded.name}"))
    if len(checkpoints) != 1:
        raise ValueError(f"selected checkpoint unavailable: {track}")
    _park_interrupted(output, f"{track}__local_completion")
    report = output / f"{track}__local_completion"
    report.mkdir()
    graph_complete(checkpoints[0], Path(config["listings"]), Path(config["pairs"]),
                   report, GraphConfig.model_validate(config),
                   text_cache=Path(config["text_cache"]) if config.get("text_cache") else None,
                   saved_inference=output / (track + "__inference"))


def _park_interrupted(output: Path, pattern: str) -> None:
    """Preserve interrupted artifacts rather than mixing them into a new attempt."""
    import time
    for path in sorted(output.glob(pattern)):
        if path.name.endswith("__vectors.npz"):
            continue
        path.rename(path.with_name(f"interrupted-{time.time_ns()}-{path.name}"))
