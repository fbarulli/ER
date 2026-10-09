"""The two bundle steps: generation (prepare inputs) and finalize.

``prepare_inputs`` reuses the owned preparation path
(:mod:`training.prepare_all`, whose ``suite_inputs`` stage is
:mod:`model_tracks.package`) and returns the verified inputs ``Bundle``.

``finalize`` is the only place CPU post-processing, post-training ablation and
sealing run. It consumes a verified result ``Bundle`` (materialize), extracts the
matching prepared inputs, produces the CPU reports and ablation over that
materialized tree, then seals a result-only bundle through
:meth:`core.bundle.Bundle.seal_result`. No step re-measures the incoming archive:
the boundary check is :meth:`core.bundle.Bundle.load`, the only other check is
the writer's own verify inside ``write_archive`` (which also captures the sealed
archive's whole-file byte size as it writes).

Lane change (owner ruling: the operator box is no longer a finalize surface):
``finalize`` is a remote CPU lane job. A Kaggle/Colab CPU job runs this module
from a sparse checkout (``python -m model_tracks.bundle_steps --role result
--lane kaggle --inputs <inputs.tar.zst> --result <result.tar.zst>
--output <sealed.tar.zst>``); the ``cli``/lane side owns provisioning, the
transfer and the receipts. Locally the same step is reached through
:func:`model_tracks.local_complete.complete`.

Bundle-contract names always come from ``training_cfg().bundle`` / the input
package's own suite config; the literals this module does spell are the
lane-local working-tree names that no bundle spec owns (see the constants
below), plus the artifact stems built through ``graph_tracks.artifacts.name``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.artifacts import Artifacts
from core.bundle import Bundle, BundlePipeline, BundleRole, bundle_spec
from core.results import Results
from core.tracing import SCOPE_ENTITY, flush_stage_trace, stage_trace

#: The stage name this module owns in the ONE consolidated pipeline trace.
STAGE = "finalize"

#: The module's trace writer: the shared shim's slot (``None`` until first use;
#: see :func:`core.tracing.stage_trace`), so importing this module never touches
#: the trace layout. Never reset: one finalize job emits materialization,
#: per-track postprocess, ablation and seal rows into the same stage commit.
_TRACE = None


def trace():
    """The ONE writer for the ``finalize`` stage of the current run."""
    global _TRACE
    _TRACE = stage_trace(STAGE, _TRACE)
    return _TRACE


def flush_trace():
    """Commit this process's finalize rows once; a no-op while empty."""
    return flush_stage_trace(_TRACE)

#: The GPU supervisor's baseline export directory (written by
#: :mod:`model_tracks.run`, which owns its transport into the result tree). It
#: is a lane-local working tree, not a bundle-contract member.
BASELINE_DIR = "baseline"
#: The baseline ablation request the suite ships inside that directory
#: (:mod:`model_tracks.baseline_ablation` owns the ``ablation/request.json``
#: layout, relative to the baseline directory).
BASELINE_ABLATION_REQUEST = f"{BASELINE_DIR}/ablation/request.json"
#: The finalize job's own CPU re-run config for one graph lane. It is written
#: into the working tree and never replaces the packaged ``worker.yaml``.
LOCAL_REPORT_CONFIG = "local_report.yaml"


def _setup_layout():
    """The declared prepared-setup layout (training.preparation.graph_setup)."""
    from core.common import prepared_setup_layout
    return prepared_setup_layout()


def _shared_embeddings_name() -> str:
    """The prepared setup's shared-embedding filename (declared layout)."""
    return _setup_layout().shared_embeddings


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
    config = pipeline.config or Path(bundle_spec().suite_config)
    prepare_all(run_dir=run_dir, tracks_config=config)

    context = _load_run_context(config)
    archive = run_dir / (f"{context.prep.suite_archive_name}."
                         f"{context.suite.input_archive_format}")
    _require(
        archive.resolve() == output.resolve(),
        f"prepared inputs archive {archive} does not match pipeline.output {output}")
    bundle = Bundle.load(archive, BundleRole.inputs)
    trace().add(
        "prepare_inputs", "verified",
        in_count=1, out_count=1,
        reason='generation wrote the sealed inputs archive; it is verified once at its boundary',
        detail={'archive': str(archive), 'size': bundle.path.stat().st_size,
                'members': len(bundle.members()), 'run_dir': str(run_dir),
                'config': str(config)},
        source='config/paths.yaml layout training_tracks_suite',
    )
    flush_trace()
    return bundle


def finalize(pipeline: BundlePipeline, result: Bundle, *, inputs: Bundle | None = None) -> Bundle:
    """Finalize: materialize -> prepared inputs -> postprocess -> ablation -> seal.

    The step is self-contained and lane-agnostic: it is handed the two verified
    bundles (``pipeline.inputs`` and ``result``) and a destination, and it writes
    exactly one sealed result archive. ``pipeline.work_dir`` says where the tree
    is materialized (defaults to a sibling of the sealed archive named by the run
    tag) and ``pipeline.prepared_dir`` where the prepared inputs are extracted
    (defaults to ``work_dir/<bundle.prepared_inputs_dir>``, which the result
    member predicate drops from the seal).

    ``inputs`` is the already-verified boundary handle for ``pipeline.inputs``;
    a caller that verified the archive for its own identity check passes it here
    so the same bytes are never re-verified inside one process.
    """
    from core.common import TRAIN_ROOT
    from model_tracks.config import SuiteConfig
    from model_tracks.package import package_member
    from model_tracks.resume import (
        TRACKS, record_completion, recorded_ablation_skip, validate_training_binding,
    )
    import yaml

    if result.role is not BundleRole.result:
        raise ValueError(f"finalize requires a result bundle, got {result.role.value}")
    if pipeline.role is not BundleRole.result:
        raise ValueError(f"finalize requires the result role, got {pipeline.role.value}")
    if pipeline.inputs is None:
        raise ValueError("finalize requires pipeline.inputs (the verified prepared inputs)")
    if pipeline.output is None:
        raise ValueError("finalize requires pipeline.output (the sealed result archive)")

    spec = bundle_spec()
    run_tag = result.run_tag()
    _require(bool(run_tag), "result bundle manifest carries no run tag")
    output = Path(pipeline.output)

    inputs = inputs if inputs is not None else Bundle.load(pipeline.inputs, BundleRole.inputs)
    if inputs.role is not BundleRole.inputs:
        raise ValueError(f"finalize requires an inputs bundle, got {inputs.role.value}")
    suite = result.read_json(spec.suite_manifest_file)
    settings = SuiteConfig.model_validate(
        yaml.safe_load(inputs.read(package_member("suite_package_config"))))
    _require(suite.get(spec.run_tag_key) == run_tag, "result suite manifest run tag differs")
    validate_training_binding(suite, inputs.manifest, settings, run_tag)
    trace().add(
        "finalize", "bundles_verified",
        in_count=2, out_count=2,
        reason='both bundles are verified exactly once at their Bundle boundary; no later step re-measures',
        detail={'run_tag': run_tag, 'lane': getattr(pipeline, 'lane', None),
                'device': getattr(pipeline, 'device', None),
                'result_archive': str(result.path), 'result_size': result.path.stat().st_size,
                'result_members': len(result.members()),
                'inputs_archive': str(inputs.path), 'inputs_size': inputs.path.stat().st_size,
                'inputs_members': len(inputs.members()),
                'post_training_ablation': bool(settings.post_training_ablation),
                'report_test': bool(settings.report_test),
                'output': str(output)},
        source=str(inputs.path),
    )

    work_dir = Path(pipeline.work_dir) if pipeline.work_dir is not None \
        else output.parent / run_tag
    prepared_dir = Path(pipeline.prepared_dir) if pipeline.prepared_dir is not None \
        else work_dir / spec.prepared_inputs_dir
    extract_prepared_inputs(inputs, prepared_dir,
                             package_member("suite_package_config"))
    setup = prepared_dir / settings.setup_dir

    tree = _materialize_result(result, work_dir, spec)
    destination = tree._root()
    _restore_frozen_baseline(destination, setup)

    if settings.post_training_ablation and (destination / BASELINE_ABLATION_REQUEST).is_file():
        from model_tracks.baseline_ablation import complete as complete_baseline_ablation
        complete_baseline_ablation(destination / BASELINE_DIR, setup,
                                   config=TRAIN_ROOT / settings.ablation_config)
        trace().add(
            "finalize", "baseline_ablation",
            in_count=1, out_count=1,
            reason='the untrained baseline ablation is completed from the suite GPU export',
            detail={'request': BASELINE_ABLATION_REQUEST,
                    'config': str(TRAIN_ROOT / settings.ablation_config)},
            source=BASELINE_ABLATION_REQUEST,
        )
    elif settings.post_training_ablation:
        trace().add(
            "finalize", "baseline_ablation",
            scope=SCOPE_ENTITY, key=BASELINE_DIR,
            reason='post-training ablation is enabled but the suite shipped no baseline ablation '
                   'request; the step is skipped rather than invented',
            detail={'request': BASELINE_ABLATION_REQUEST,
                    'present': (destination / BASELINE_ABLATION_REQUEST).is_file()},
            source=BASELINE_ABLATION_REQUEST,
        )

    _postprocess_tracks(tree, setup, settings)

    if settings.post_training_ablation:
        if recorded_ablation_skip(destination):
            from core.run_log import RunLogger
            RunLogger(__name__).info(
                "[finalize] suite skipped attribute ablation; no saved ablation to consume")
            trace().add(
                "finalize", "saved_ablation",
                reason='the suite recorded a deliberate ablation skip; there is no saved ablation to consume',
                detail={'destination': str(destination), 'recorded_skip': True},
                source=str(destination / bundle_spec().suite_events_file),
            )
        else:
            from model_tracks.post_training_ablation import complete_saved
            complete_saved(destination, settings)
            for track in TRACKS:
                record_completion(destination / track, track)
            trace().add(
                "finalize", "saved_ablation",
                in_count=len(TRACKS), out_count=len(TRACKS),
                reason='the saved GPU ablation exports are consumed and each track is marked complete',
                detail={'destination': str(destination), 'recorded_skip': False,
                        'tracks': list(TRACKS)},
                source=str(Results.for_root(destination, run_tag).receipt()),
            )

    location = pipeline.postprocess_location or spec.postprocess_location_bundle
    suite["postprocess_location"] = location
    (destination / spec.suite_manifest_file).write_text(json.dumps(suite, indent=2))
    sealed_members = tree.collect_result_members()
    all_members = tree.members()
    sealed = tree.seal_result(
        output, metadata={**pipeline.metadata, spec.run_tag_key: run_tag,
                          "postprocess_location": location})
    trace().add(
        "finalize", "seal",
        in_count=len(all_members), out_count=len(sealed_members),
        reason='the result role decides the sealed member set: the selected checkpoint only, '
               'never every epoch and never the extracted prepared inputs',
        detail={'output': str(output), 'size': sealed.path.stat().st_size,
                'bytes': output.stat().st_size,
                'tree_files': len(all_members), 'sealed_members': len(sealed_members),
                'postprocess_location': location,
                'suite_manifest': spec.suite_manifest_file,
                'run_tag': run_tag},
        source=str(output),
    )
    flush_trace()
    return sealed


def _materialize_result(result: Bundle, work_dir: Path, spec) -> Bundle:
    """The result tree at ``work_dir``, unpacked once and then left alone.

    A finalize attempt can be interrupted after its CPU reports landed; the
    retry must continue that work (per-track completion markers are what make it
    idempotent), so an already-materialized tree is never overwritten with the
    archive's older member bytes. The suite manifest is the marker that the tree
    is a materialized result.
    """
    archive_members = len(result.members())
    if (work_dir / spec.suite_manifest_file).is_file():
        tree = Bundle.from_directory(work_dir, BundleRole.result)
        trace().add(
            "materialize", "result_tree",
            in_count=archive_members, out_count=0,
            reason='a materialized result tree already exists, so this step unpacks NOTHING: an '
                   'interrupted attempt continues on the tree and is never overwritten with the '
                   'archive\'s older member bytes',
            detail={'work_dir': str(work_dir), 'reused_existing_tree': True,
                    'archive_members': archive_members, 'tree_files': len(tree.members()),
                    'suite_manifest': spec.suite_manifest_file},
            source=str(work_dir),
        )
        return tree
    tree = result.materialize(work_dir)
    tree_files = len(tree.members())
    trace().add(
        "materialize", "result_tree",
        in_count=archive_members, out_count=archive_members,
        reason='the verified result archive is unpacked member-for-member into the finalize work '
               'tree, which already carries the extracted prepared inputs (they are inputs to this '
               'process, never deliverables)',
        detail={'work_dir': str(work_dir), 'reused_existing_tree': False,
                'archive_members': archive_members, 'tree_files': tree_files,
                'extracted_prepared_files': max(0, tree_files - archive_members),
                'archive': str(result.path)},
        source=str(result.path),
    )
    return tree


def extract_prepared_inputs(inputs: Bundle, destination: Path, config_member: str) -> None:
    """Extract the input package's prepared tree under ``destination`` (no re-measure).

    The inputs bundle was verified once at its boundary, so members are written
    from the trusted handle. An already-extracted file is reused when it matches
    the manifest size; when it does not, it is rewritten from the trusted bytes
    (owner policy 2026-10-08: an incompatible cached intermediate is rebuilt
    silently, never a reason to fail). The size comparison is the idempotent
    retry path, not a stage re-verify.
    """
    from core.portable_archive import file_size
    package_root = Path(config_member).parent
    members = inputs.manifest.get(bundle_spec().files_key, {})
    written = reused = rebuilt = outside = 0
    for relative, expected in members.items():
        if not Path(relative).is_relative_to(package_root):
            outside += 1
            continue
        target = destination / relative
        if target.exists() and file_size(target) == expected:
            reused += 1
            continue
        rebuilt += int(target.exists())
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(inputs.read(relative))
        written += 1
    trace().add(
        "extract_prepared_inputs", "extracted",
        in_count=len(members), out_count=written + reused,
        reason='the inputs bundle was verified once at its boundary; members are written from the '
               'trusted handle, a matching file already on disk is reused, and a mismatching one is '
               'rebuilt from the trusted bytes instead of failing',
        detail={'members_in_manifest': len(members), 'written': written,
                'already_present_verified': reused, 'rebuilt_incompatible': rebuilt,
                'outside_package': outside,
                'package_root': package_root, 'destination': str(destination)},
        source=str(inputs.path),
    )


def _restore_frozen_baseline(destination: Path, setup: Path) -> None:
    """Bind the suite's GPU baseline export as the report-time embedding cache."""
    shared = _shared_embeddings_name()
    baseline = destination / BASELINE_DIR / shared
    if not baseline.is_file():
        trace().add(
            "baseline", "embedding_cache",
            in_count=0, out_count=0,
            reason='the suite shipped no GPU baseline embedding export, so report time runs '
                   'without a restored frozen cache',
            detail={'baseline': f"{BASELINE_DIR}/{shared}", 'shared_embeddings': shared,
                    'destination': str(destination)},
            source=f"{BASELINE_DIR}/{shared}",
        )
        return
    from training.prepare_embeddings import validate_result
    from graph_tracks.data import file_size
    request = setup / _setup_layout().embedding_request
    # Identity only: the request size is NOT re-derived and compared against the
    # cache's record (owner directive 2026-10-08: no freshness checks anywhere).
    validate_result(baseline, json.loads(request.read_text()))
    cache = setup / shared
    baseline_size = file_size(baseline)
    copied = not cache.exists()
    # The frozen GPU baseline export is the trusted copy: an existing report-time
    # cache that differs from it is the incompatible cached intermediate of the
    # owner policy, so it is overwritten silently rather than quarantined.
    replaced = cache.exists() and file_size(cache) != baseline_size
    if copied or replaced:
        import shutil
        shutil.copy2(baseline, cache)
    trace().add(
        "baseline", "embedding_cache",
        in_count=1, out_count=1,
        reason=('the frozen GPU baseline export was copied in as the report-time cache'
                if copied else
                'the report-time cache differed from the frozen GPU baseline export and was '
                'rebuilt from it'
                if replaced else
                'the report-time cache already holds the frozen GPU baseline export'),
        detail={'baseline': str(baseline), 'baseline_size': baseline_size,
                'cache': str(cache), 'copied': copied, 'rebuilt': replaced,
                'embedding_request': str(request)},
        source=str(baseline),
    )


def _track_postprocessed(output: Path) -> bool:
    """The track marker says its CPU post-processing already completed."""
    marker = output / bundle_spec().complete_file
    try:
        return bool(json.loads(marker.read_text()).get("postprocess_complete"))
    except (OSError, ValueError):
        return False


def _postprocess_tracks(tree: Bundle, setup: Path, settings) -> None:
    """Run the CPU report for every unfinished trained track (idempotent).

    A track whose marker already says ``postprocess_complete`` is left alone, so
    a GPU-only suite (whose cascade worker already composed its report) costs
    nothing here; an unfinished track is re-reported from its selected
    checkpoint, exactly as the suite trained it.
    """
    from model_tracks.resume import TRACKS, record_completion

    root = tree._root()
    outcomes: list[dict[str, object]] = []
    for track in TRACKS:
        output = root / track
        if _track_postprocessed(output):
            outcomes.append({'track': track, 'outcome': 'already_postprocessed',
                             'route': None,
                             'reason': 'the track marker already records postprocess_complete, so '
                                       'this finalize attempt leaves the track alone'})
            continue
        if track == "text":
            from model_tracks.text_report import complete as text_complete
            _park_interrupted(output, "text__*")
            text_complete(output, setup, device="cpu", report_test=settings.report_test)
            route = 'text_report'
        elif track == "cascade":
            _complete_cascade_track(tree, output, setup, settings)
            route = 'cascade_report'
        else:
            _complete_graph_track(tree, output, setup, track, settings)
            route = 'graph_report'
        record_completion(output, track)
        outcomes.append({'track': track, 'outcome': 'reported', 'route': route,
                         'reason': 'the track carried no postprocess_complete marker, so its '
                                   'selected checkpoint was re-reported on CPU'})
    reported = [outcome for outcome in outcomes if outcome['outcome'] == 'reported']
    trace().add(
        "postprocess", "tracks",
        in_count=len(TRACKS), out_count=len(reported),
        reason='the CPU report runs only for tracks whose marker does not already record '
               'postprocess_complete; a finished cascade is left alone',
        detail={'tracks': list(TRACKS), 'reported': len(reported),
                'already_postprocessed': len(outcomes) - len(reported),
                'report_test': bool(settings.report_test), 'root': str(root)},
        source='model_tracks.resume.TRACKS',
    )
    trace().add_entities(
        "postprocess.track", outcomes,
        key_of=lambda outcome: outcome['track'],
        reason_of=lambda outcome: outcome['outcome'],
        detail_of=lambda outcome: {'track': outcome['track'], 'route': outcome['route'],
                                   'reason': outcome['reason']},
        source='model_tracks.resume.TRACKS',
    )


def _prepared_path(value: str, *, setup: Path) -> Path:
    """Locate one lane-config input inside the extracted prepared-inputs tree.

    A lane config records an input either as the package machine's absolute
    path or as the checkout-relative fixture form; in both cases the member's
    identity is its path below the suite's ``setup_dir``, which is exactly what
    ``extract_prepared_inputs`` reproduces under the finalize tree.
    """
    from core.common import TRAIN_ROOT
    prepared = setup.parent
    prefix = setup.relative_to(prepared).as_posix().rstrip("/") + "/"
    raw = str(value).replace("\\", "/")
    if prefix in raw:
        return setup / raw.split(prefix, 1)[1]
    path = Path(value)
    if not path.is_absolute():
        path = Path(TRAIN_ROOT) / path
    try:
        return prepared / path.resolve().relative_to(Path(TRAIN_ROOT).resolve())
    except ValueError:
        return path


def _rewrite_prepared_inputs(config: dict, keys, *, setup: Path) -> None:
    """Point a lane config's prepared inputs at the extracted finalize tree."""
    for key in keys:
        if config.get(key):
            config[key] = str(_prepared_path(config[key], setup=setup))


def _complete_graph_track(tree: Bundle, output: Path, setup: Path, track: str,
                          settings) -> None:
    """Re-run one graph track's saved inference on CPU and write its report.

    The selected checkpoint is resolved through the bundle's role contract
    (``track__best_checkpoint.json`` recorded path, located under the track),
    never through a per-surface re-derivation.
    """
    from graph_tracks.config import GraphConfig, load_config as load_graph_config
    from graph_tracks.preflight import preflight
    from graph_tracks.report import complete as graph_complete
    import yaml

    config = load_graph_config(setup / f"{track}.yaml", expected_track=track).model_dump()
    _rewrite_prepared_inputs(config, ("listings", "pairs", "input_manifest", "text_cache"),
                             setup=setup)
    config.update(device="cpu", report_test=settings.report_test, postprocess=True)
    config_path = output / LOCAL_REPORT_CONFIG
    config_path.write_text(yaml.safe_dump(config))
    preflight(config_path, check_device=False)
    checkpoint = tree.checkpoint(track)
    if checkpoint is None or not checkpoint.is_file():
        raise ValueError(f"selected checkpoint unavailable: {track}")
    report = output / Artifacts.member_name("local_completion", track=track)
    _park_interrupted(output, report.name)
    report.mkdir()
    saved_inference = output / Artifacts.member_name("inference", track=track)
    from model_tracks.ablation import file_size as _file_size
    trace().add(
        "checkpoint_select", "selected",
        scope=SCOPE_ENTITY, key=track,
        in_count=1, out_count=1,
        reason='the checkpoint is resolved through the bundle role contract (the recorded marker '
               'located under the track), never through a per-surface re-derivation',
        detail={'track': track, 'checkpoint': str(checkpoint),
                'checkpoint_size': _file_size(checkpoint),
                'bytes': checkpoint.stat().st_size,
                'report': str(report), 'report_test': bool(settings.report_test),
                'device': 'cpu'},
        source=str(output / Artifacts.member_name("best_checkpoint", track=track)),
    )
    graph_complete(checkpoint, Path(config["listings"]), Path(config["pairs"]),
                   report, GraphConfig.model_validate(config),
                   text_cache=Path(config["text_cache"]) if config.get("text_cache") else None,
                   saved_inference=saved_inference)
    trace().add(
        "postprocess", "graph_track",
        scope=SCOPE_ENTITY, key=track, in_count=1, out_count=1,
        reason='the saved inference is re-run on CPU from the selected checkpoint and reported',
        detail={'track': track, 'report': str(report), 'device': 'cpu',
                'saved_inference': str(saved_inference)},
        source=str(report),
    )


def _complete_cascade_track(tree: Bundle, output: Path, setup: Path, settings) -> None:
    """Re-compose the cascade report from the materialized trained artifacts.

    The cascade trains nothing, so it has no checkpoint to reload and the
    trained-lane branch above cannot apply to it. It is a combinator over the
    text ranker's ANN export and the gnn_only decider's saved scorer, both of
    which the trained lanes already wrote into this tree, so completing an
    interrupted cascade attempt means re-running the worker's cascade lane with
    one implementation. The lane's declared ``text_index`` / ``gnn_checkpoint``
    are package-time setup placeholders, so they are cleared and the resolver
    locates the trained artifacts under their own tracks instead.
    """
    from graph_tracks.config import load_config as load_graph_config
    from graph_tracks.data import load_records
    from graph_tracks.report import report_cascade
    from graph_tracks.train import load_pairs
    from model_tracks import worker

    lane = load_graph_config(setup / "cascade.yaml", expected_track="cascade")
    listings = _prepared_path(lane.listings, setup=setup)
    pairs = _prepared_path(lane.pairs, setup=setup)
    lane = lane.model_copy(update={"listings": str(listings), "pairs": str(pairs),
                                   "text_index": None, "gnn_checkpoint": None,
                                   "report_test": bool(settings.report_test)})
    artifacts = worker._cascade_artifacts(tree._root(), lane)
    records = load_records(listings)
    pair_index = load_pairs(pairs, records)
    _park_interrupted(output, "cascade__*")
    ranked, relevant, decisions = worker._cascade_roles(records, pair_index, artifacts)
    # An unknown query count is reported as unknown, never guessed: the roles
    # object owns the count and a caller may legitimately hand back a shape that
    # does not expose it.
    queries = len(ranked.query_ids) if hasattr(ranked, "query_ids") else None
    trace().add(
        "checkpoint_select", "cascade_role",
        scope=SCOPE_ENTITY, key="cascade", in_count=1, out_count=1,
        reason='the cascade trains nothing: its roles are the trained text ranker index and the '
               'trained gnn_only scorer checkpoint located under their own tracks',
        detail={'cascade_report': str(output),
                'text_index': str(artifacts['text_index']),
                'text_vectors': str(artifacts['text_vectors']),
                'gnn_vectors': str(artifacts['gnn_vectors']),
                'gnn_checkpoint': str(artifacts['gnn_checkpoint']),
                'queries': queries, 'relevant_sets': len(relevant),
                'retrieval_ks': list(sorted(set(lane.retrieval_ks) | {1}))},
        source=str(artifacts['gnn_checkpoint']),
    )
    report_cascade(ranked, relevant, decisions, output, track="cascade",
                   ks=tuple(sorted(set(lane.retrieval_ks) | {1})))
    worker._record_cascade_report_manifest(output, lane, artifacts)
    trace().add(
        "postprocess", "cascade_track",
        # A UNIT row: one composed cascade report, whatever the query count.
        scope=SCOPE_ENTITY, key="cascade", in_count=None, out_count=1,
        reason='the cascade report and its calibrated manifest are composed from the trained lanes',
        detail={'cascade_report': str(output / Artifacts.member_name(
                        "cascade_report", track="cascade")),
                'report_manifest': str(output / Artifacts.member_name(
                        "report_manifest", track="cascade")),
                'queries': queries},
        source=str(output),
    )


def _park_interrupted(output: Path, pattern: str) -> None:
    """Preserve interrupted artifacts rather than mixing them into a new attempt."""
    import time
    parked = [path for path in sorted(output.glob(pattern))
              if not path.name.endswith("__vectors.npz")]
    for path in parked:
        path.rename(path.with_name(f"interrupted-{time.time_ns()}-{path.name}"))
    trace().add(
        "postprocess", "park_interrupted",
        in_count=len(parked), out_count=len(parked),
        reason=('a prior attempt\'s artifacts are preserved under an interrupted- prefix instead of '
                'being mixed into the new attempt' if parked else
                'no artifact of the previous attempt exists for this pattern'),
        detail={'output': str(output), 'pattern': pattern, 'parked': len(parked),
                'sample_parked': [path.name for path in parked[:5]]},
        source=str(output),
    )
    trace().add_entities(
        "postprocess.parked_artifact", parked,
        key_of=lambda path: path.name,
        reason_of=lambda path: 'interrupted_artifact_preserved',
        detail_of=lambda path: {'artifact': path.name, 'pattern': pattern},
        source=str(output),
    )


def main(argv: list[str] | None = None) -> None:
    """The lane entrypoint: one bundle step, run from a sparse checkout.

    Generation (``--role inputs``) writes the sealed inputs archive; finalize
    (``--role result``) consumes a verified result archive plus its inputs
    archive and writes the sealed result archive. This is the process a Kaggle
    CPU kernel or a Colab CPU stage invokes; it deliberately owns no
    provisioning, no transfer and no publication.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=[role.value for role in BundleRole], required=True)
    parser.add_argument("--lane", required=True, help="lane name recorded by the caller")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, help="the verified prepared-inputs archive")
    parser.add_argument("--result", type=Path, help="the verified result archive to finalize")
    parser.add_argument("--config", type=Path, help="generation suite config")
    parser.add_argument("--work-dir", type=Path, help="where finalize materializes the tree")
    parser.add_argument("--prepared-dir", type=Path, help="where finalize extracts inputs")
    parser.add_argument("--sparse-path", action="append", default=[],
                        help="repository-relative path the sparse checkout must carry")
    args = parser.parse_args(argv)
    pipeline = BundlePipeline(
        role=args.role, device=args.device, lane=args.lane, output=args.output,
        inputs=args.inputs, config=args.config, work_dir=args.work_dir,
        prepared_dir=args.prepared_dir, sparse_paths=tuple(args.sparse_path))
    if pipeline.role is BundleRole.inputs:
        print(pipeline.prepare_inputs().path)
        return
    if args.result is None:
        raise SystemExit("--result is required for a finalize job")
    print(pipeline.finalize(Bundle.load(args.result, BundleRole.result)).path)


if __name__ == "__main__":
    main()
