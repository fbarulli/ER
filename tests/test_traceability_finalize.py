"""tests/test_traceability_finalize.py — the finalize/postprocess/ablation trace rows.

One decisive test per instrumented entry point of the FINALIZE half of the
pipeline (the training and data-processing halves are covered by
tests/test_traceability_stages.py). Every test:

  * redirects the trace to a tmp file (``core.tracing.trace_path``) and pins a run
    id (``EUROMONITOR_TRACE_RUN``), so nothing touches the real results tree;
  * resets the per-module trace writers, so one test's rows never leak into the
    next test's file;
  * runs the entry point;
  * reads the file back with ``core.tracing.read_trace`` and validates EVERY row
    against ``core.schemas.TraceRow`` (the frame boundary contract);
  * asserts the accounting that entry point owes: the funnel counts on the seal,
    the per-track census, and the named ENTITY behind a skipped/quarantined step.

Covered entry points: bundle_steps.finalize / local_complete.complete (via the
real CPU completion path), run.run (the suite supervisor), worker.run (a track
worker), staged_ablation.forward + baseline_ablation.forward,
ablation.prepare/encode/report, ablation_cohort.prepare_cohort,
post_training_ablation.complete_saved (quarantine path),
staged_ablation.prepare_suite, and snapshot_completion.complete.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import core.tracing as tracing
from core.schemas import TraceRow

#: The modules whose module-level trace writers this file drives.
TRACED_MODULES = (
    "bundle_steps", "run", "worker", "local_complete", "snapshot_completion",
    "ablation", "ablation_cohort", "post_training_ablation", "staged_ablation",
    "baseline_ablation",
)


@pytest.fixture()
def trace_target(tmp_path, monkeypatch) -> Path:
    """The consolidated trace redirected into a tmp file, with a pinned run id."""
    target = tmp_path / "training_trace.csv"
    monkeypatch.setattr(tracing, "trace_path", lambda: target)
    monkeypatch.setenv("EUROMONITOR_TRACE_RUN", "run-test-finalize")
    return target


@pytest.fixture(autouse=True)
def fresh_writers(monkeypatch):
    """Give every test empty per-module writers (they are process-lifetime singletons)."""
    import importlib

    for name in TRACED_MODULES:
        module = importlib.import_module(f"model_tracks.{name}")
        monkeypatch.setattr(module, "_TRACE", None, raising=False)
    yield


def read_validated(target: Path) -> pd.DataFrame:
    """Read the trace back and prove every row satisfies the row contract."""
    frame = tracing.read_trace(target)
    assert list(frame.columns) == list(tracing.TRACE_COLUMNS)
    tracing.assert_trace_frame(frame)
    dropped = pd.to_numeric(frame["dropped_count"], errors="coerce").fillna(0)
    assert (dropped >= 0).all(), frame[dropped < 0][
        ["stage", "step", "in_count", "out_count", "dropped_count"]].to_string()
    for row in frame.to_dict("records"):
        TraceRow.model_validate(row)
    return frame


def stage_steps(frame: pd.DataFrame, stage: str) -> set[str]:
    return set(frame[frame["stage"].astype(str) == stage]["step"].astype(str))


def only(frame: pd.DataFrame, stage: str, step: str) -> pd.Series:
    hit = frame[(frame["stage"].astype(str) == stage) & (frame["step"].astype(str) == step)]
    assert len(hit) == 1, f"expected exactly one {stage}/{step} row, got {len(hit)}"
    return hit.iloc[0]


# ── 1. the CPU completion path: finalize + local_complete ──────────────────
def _local_suite(tmp_path, monkeypatch, *, cascade_complete=False,
                 post_training_ablation=False, with_runtime_source=False):
    """The hermetic downloaded-suite fixture (one training archive + one inputs archive).

    Adapted from tests/test_model_tracks_local_complete.py: the real
    ``bundle_steps.finalize`` runs, with only the lanes' report writers stubbed.
    """
    from core import common
    from graph_tracks import preflight, report
    from model_tracks import text_report, worker
    from core.portable_archive import write_archive

    monkeypatch.setattr(common, "TRAIN_ROOT", tmp_path)
    cfg = {"setup_dir": "data/model_tracks/shared",
           "text_bundle": "data/model_tracks/shared/text.pkl",
           "publish_git": False, "publish_dvc": False,
           "result_archive_format": "zip",
           "post_training_ablation": post_training_ablation}
    inline = {"data/model_tracks/suite.yaml": _yaml(cfg),
              "data/model_tracks/shared/prepared/listings.json": '{"listings": []}',
              "data/model_tracks/shared/prepared/pairs.csv": "sku_id1,sku_id2,label,split\n"}
    if with_runtime_source:
        # The frozen completion runtime the snapshot wrapper materializes.
        inline["src/model_tracks/local_complete.py"] = (
            Path(__file__).resolve().parents[1]
            / "src/model_tracks/local_complete.py").read_text()
    for track in ("gnn_only", "cascade"):
        settings = {"track": track,
                    "listings": "data/model_tracks/shared/prepared/listings.json",
                    "pairs": "data/model_tracks/shared/prepared/pairs.csv",
                    "output_dir": "results/graph_tracks"}
        if track == "cascade":
            settings["text_index"] = "results/graph_tracks/text__index"
            settings["gnn_checkpoint"] = "results/graph_tracks/gnn_only__best_checkpoint.json"
        inline[f"data/model_tracks/shared/{track}.yaml"] = _yaml(settings)
    inputs = {"text": {"bundle_size": "bundle"}}
    input_zip = write_archive(tmp_path / "input.zip", {}, inline=inline,
                              manifest_name="model_tracks_package.json",
                              metadata={"preflight": inputs})
    root = tmp_path / "remote"
    root.mkdir()
    (root / "suite_manifest.json").write_text(_json({"run_tag": "run", "inputs": inputs,
                                                    "config": cfg,
                                                    "resume_identity": {"implementation": {}}}))
    for track in ("text", "gnn_only", "cascade"):
        output = root / track
        output.mkdir()
        checkpoint = output / "checkpoint-1"
        checkpoint.mkdir()
        (checkpoint / "model.pt").write_bytes(b"trained")
        if track != "text":
            (output / f"{track}__best_checkpoint.json").write_text(
                _json({"path": "/remote/checkpoint-1/model.pt"}))
        if track == "cascade" and cascade_complete:
            (output / "cascade__cascade_report.json").write_text("{}")
            _manifest(output / "cascade__report_manifest.json", "cascade", False)
        from model_tracks.resume import record_completion
        record_completion(output, track,
                          postprocess_complete=(track == "cascade") and cascade_complete)
    training_zip = write_archive(
        tmp_path / "run.training.zip",
        {p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file()},
        manifest_name="suite_bundle_manifest.json", metadata={"run_tag": "run"})

    def text(output, setup, *, device, report_test):
        assert device == "cpu"
        (output / "text__training_report.md").write_text("local text report")
        (output / "text__index").mkdir()
        (output / "text__vectors.npz").write_bytes(b"text-vectors")
        _manifest(output / "text__completion_manifest.json", "text", report_test)

    def graph(checkpoint, listings, pairs, output, cfg, **kwargs):
        (output / "report.md").write_text("local graph report")
        inference = output / f"{cfg.track}__inference"
        inference.mkdir()
        (inference / f"{cfg.track}__vectors.npz").write_bytes(b"gnn-vectors")
        _manifest(output / f"{cfg.track}__report_manifest.json", cfg.track, cfg.report_test)

    monkeypatch.setattr(worker, "_cascade_roles", lambda *a, **k: ("RANKED", ["relevant"], "DECISIONS"))
    monkeypatch.setattr("graph_tracks.report.report_cascade", lambda *a, **k: None)
    monkeypatch.setattr("graph_tracks.data.load_records", lambda path: [])
    monkeypatch.setattr("graph_tracks.train.load_pairs", lambda path, records: {})
    monkeypatch.setattr(text_report, "complete", text)
    monkeypatch.setattr(report, "complete", graph)
    monkeypatch.setattr(preflight, "preflight", lambda *a, **k: {})
    return training_zip, input_zip


def _yaml(value: dict) -> str:
    import yaml
    return yaml.safe_dump(value)


def _json(value: dict) -> str:
    import json
    return json.dumps(value)


def _manifest(path, track, report_test):
    from graph_tracks.report_manifest import build as build_manifest, write as write_manifest
    write_manifest(path, build_manifest(
        track=track, checkpoint="checkpoint-1/model.pt",
        checkpoint_size="0" * 64, listings_size="1" * 64, pairs_size="2" * 64,
        threshold=0.5, threshold_source="dev_youden", test_reported=bool(report_test),
        model_selection="dev_pr_auc", retrieval_ks=[10]))


def test_finalize_emits_materialize_postprocess_checkpoint_and_seal_rows(
    tmp_path, monkeypatch, trace_target,
):
    """``local_complete.complete`` drives the ONE finalize step; every step lands."""
    from model_tracks.local_complete import complete

    training_zip, input_zip = _local_suite(tmp_path, monkeypatch)
    complete(training_zip, input_zip, "run")
    frame = read_validated(trace_target)

    finalize = stage_steps(frame, "finalize")
    assert {
        "finalize.bundles_verified",
        "materialize.result_tree",
        "extract_prepared_inputs.extracted",
        "baseline.embedding_cache",
        "postprocess.tracks",
        # the bounded per-track census + entity sample + budget row
        "postprocess.track.reason_census",
        "postprocess.parked_artifact.sample_budget",
        # the checked-in checkpoint selection for the gnn_only track
        "checkpoint_select.selected",
        "finalize.seal",
    } <= finalize, finalize

    # the materialization funnel: archive members in, archive members unpacked
    # out. The work tree also carries the extracted prepared inputs, so it is
    # larger than the archive and the row says so explicitly.
    materialize = only(frame, "finalize", "materialize.result_tree")
    detail = tracing.detail_json(materialize["detail"])
    assert detail["reused_existing_tree"] is False
    assert int(materialize["in_count"]) == detail["archive_members"] > 0
    assert int(materialize["out_count"]) == detail["archive_members"]
    assert detail["tree_files"] == detail["archive_members"] + detail["extracted_prepared_files"]
    assert detail["extracted_prepared_files"] > 0

    # the per-track postprocess census: the trained lanes are reported, and the
    # census is exact (its group rows sum to the track population).
    census = frame[(frame["stage"] == "finalize")
                   & (frame["step"] == "postprocess.track.reason_census")]
    assert set(census["reason"]) == {"reported"}
    assert int(census["in_count"].astype(int).sum()) == 3

    # checkpoint-select names the exact entity and its digest
    select = only(frame, "finalize", "checkpoint_select.selected")
    assert select["scope"] == tracing.SCOPE_ENTITY and select["key"] == "gnn_only"
    select_detail = tracing.detail_json(select["detail"])
    assert select_detail["checkpoint"].endswith("checkpoint-1/model.pt")
    assert int(select_detail["checkpoint_size"]) > 0

    # the seal: tree files in, selected-only members out
    seal = only(frame, "finalize", "finalize.seal")
    seal_detail = tracing.detail_json(seal["detail"])
    assert seal_detail["sealed_members"] == int(seal["out_count"]) <= int(seal["in_count"])
    assert seal_detail["size"] and seal_detail["run_tag"] == "run"

    # the extraction funnel: written + already-present == the package members
    extract = only(frame, "finalize", "extract_prepared_inputs.extracted")
    extract_detail = tracing.detail_json(extract["detail"])
    assert extract_detail["written"] + extract_detail["already_present_verified"] == int(
        extract["out_count"])
    assert int(extract["in_count"]) == extract_detail["members_in_manifest"]

    # the local_complete stage records both boundaries, the delegation and the publication
    local = stage_steps(frame, "local_complete")
    assert {
        "complete.boundaries", "complete.transport_timings", "complete.finalized",
        "complete.validated", "complete.published", "publish.validated",
    } <= local, local
    boundaries = only(frame, "local_complete", "complete.boundaries")
    boundary_detail = tracing.detail_json(boundaries["detail"])
    assert boundary_detail["training_size"] and boundary_detail["input_size"]
    published = only(frame, "local_complete", "complete.published")
    assert int(published["out_count"]) == 1


# ── 2. the suite supervisor: selection, gate, seal, publication ────────────
def test_suite_supervisor_emits_selection_gate_seal_and_publication_rows(
    tmp_path, monkeypatch, trace_target,
):
    """``run.run`` seals the GPU result archive; the supervisor's steps land."""
    import subprocess

    from model_tracks import data_gate, resume
    from model_tracks import run as suite_run
    from model_tracks.config import SuiteConfig

    cfg = SuiteConfig(setup_dir="setup", text_bundle="bundle", device="cpu",
                      publish_git=False, publish_dvc=False,
                      result_archive_format="zip")
    config = tmp_path / "suite.yaml"
    config.write_text("setup_dir: setup\ntext_bundle: bundle\n")
    monkeypatch.setattr(suite_run, "load_config", lambda _: cfg)
    monkeypatch.setattr(suite_run, "preflight", lambda *a, **k: {})
    monkeypatch.setattr(data_gate, "validate", lambda *a, **k: SimpleNamespace(
        tracks=["text", "gnn_only", "cascade"], attestation="attest"))
    monkeypatch.setattr(resume, "suite_identity", lambda *a, **k: {"implementation": {}})
    monkeypatch.setattr(resume, "completed_track", lambda *a, **k: True)
    monkeypatch.setattr(resume, "expected_postprocess", lambda track, gpu_only: True)
    monkeypatch.setattr(suite_run, "run_parallel",
                        lambda *a, **k: {"mode": "parallel", "workers": ["text", "gnn_only"]})
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    monkeypatch.delenv("ER_GPU_TRAINING_ONLY", raising=False)

    archive = suite_run.run(config, tmp_path / "suite", "run")
    assert archive == tmp_path / "suite.zip" and archive.is_file()

    frame = read_validated(trace_target)
    steps = stage_steps(frame, "suite_run")
    assert {
        "run.preflight", "run.worker_selection", "run.skipped_track.sample_budget",
        "run.workers", "run.postprocess_lane", "run.completion_gate",
        "run.collection", "run.seal", "run.publication",
    } <= steps, steps

    # the member funnel: every result-tree file in, the selected members out
    collection = only(frame, "suite_run", "run.collection")
    collection_detail = tracing.detail_json(collection["detail"])
    assert collection_detail["selected_members"] == int(collection["out_count"])
    assert collection_detail["tree_files"] == int(collection["in_count"])

    seal = only(frame, "suite_run", "run.seal")
    seal_detail = tracing.detail_json(seal["detail"])
    assert seal_detail["size"] and int(seal["out_count"]) > 0
    assert seal_detail["bytes"] == archive.stat().st_size

    publication = only(frame, "suite_run", "run.publication")
    assert tracing.detail_json(publication["detail"])["published"] is False
    assert "disabled" in publication["reason"]

    # the postprocess lane ran as its own process and is named at entity grain
    lane = only(frame, "suite_run", "run.postprocess_lane")
    assert lane["scope"] == tracing.SCOPE_ENTITY and lane["key"] == "cascade"


# ── 3. a track worker: command, training, inference, ablation skip ─────────
def test_worker_emits_command_training_inference_and_ablation_skip_rows(
    tmp_path, monkeypatch, trace_target,
):
    """``worker.run`` on a templateless GPU session records the named ablation skip."""
    from core import common
    from model_tracks import worker
    from training import prepared_bundle, validation_inference
    from model_tracks import text_export, staged_ablation, text_report

    setup = tmp_path / "setup"
    setup.mkdir()
    (setup / "setup_manifest.json").write_text("{}")
    output = tmp_path / "run/text"
    monkeypatch.setenv("EUROMONITOR_RESULTS_DIR", str(output))
    monkeypatch.setenv("ER_GPU_TRAINING_ONLY", "1")
    monkeypatch.setenv("ER_TRACK_BARRIER", str(tmp_path / "barrier"))
    monkeypatch.setattr(common, "TRAIN_ROOT", tmp_path)
    cfg = SimpleNamespace(setup_dir="setup", text_bundle="bundle", text_model="baseline",
                          epochs=1, device="cuda", report_test=False,
                          post_training_ablation=True)
    monkeypatch.setattr(worker, "load_config", lambda _: cfg)
    monkeypatch.setattr(prepared_bundle, "load_prepared_bundle",
                        lambda _: (SimpleNamespace(payload_variant="full"), {}))
    from training.prepared_bundle import PreparedBundleManifest
    header = {"payload_variant": "full", "masking_profile": "baseline",
              "model_input": {"profile": "cleaned", "include_evidence": False},
              "n_df": 1, "n_payload": 1, "n_pos": 1, "n_neg": 1, "n_train_neg": 1,
              "n_labeled_pairs_bytes": 1, "n_canonical_records_bytes": 1,
              "n_gate_results_bytes": 1, "size": "0" * 64}
    PreparedBundleManifest.model_validate(header)
    (tmp_path / "bundle.json").write_text(_json(header))
    monkeypatch.setattr(worker, "wait_for_start", lambda *a: None)
    monkeypatch.setattr(worker.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    monkeypatch.setattr(text_export, "forward", lambda *a, **k: (None, object()))
    monkeypatch.setattr(text_report, "build_index", lambda *a, **k: output / "text__index")
    monkeypatch.setattr(staged_ablation, "forward",
                        lambda *a, **k: pytest.fail("no template means no forward"))
    monkeypatch.setattr(validation_inference, "resolve_best_checkpoint",
                        lambda _: (output / "checkpoint", {}))
    from model_tracks import resume
    monkeypatch.setattr(resume, "record_completion", lambda *a, **k: None)

    worker.run(tmp_path / "suite.yaml", "text", "run-text")
    frame = read_validated(trace_target)
    steps = stage_steps(frame, "worker")
    assert {
        "command.prepared", "barrier.released", "training.completed",
        "inference_export.completed", "attribute_ablation_export.skipped",
        "completion.verified",
    } <= steps, steps

    skip = only(frame, "worker", "attribute_ablation_export.skipped")
    assert skip["scope"] == tracing.SCOPE_ENTITY and skip["key"] == "text"
    assert skip["reason"] == "bundle_shipped_no_ablation_templates"

    # The GPU-only text lane still builds the ANN index the same-suite cascade
    # consumes; only reports/scoring are deferred to the local finalize.
    index_row = only(frame, "worker", "postprocess.completed")
    index_detail = tracing.detail_json(index_row["detail"])
    assert index_detail["gpu_only"] is True
    assert index_detail["index"].endswith("text__index")

    # the completion marker the worker records for a GPU-only text lane
    completion = only(frame, "worker", "completion.verified")
    assert tracing.detail_json(completion["detail"])["postprocess_complete"] is False
    command = only(frame, "worker", "command.prepared")
    assert "training.train_prepared" in tracing.detail_json(command["detail"])["command"]


# ── 4. staged forward: template binding + checkpoint selection ─────────────
def test_staged_forward_records_template_checkpoint_and_vectors(tmp_path, monkeypatch,
                                                                trace_target):
    """``staged_ablation.forward`` binds a selected checkpoint onto the frozen template."""
    import json

    import torch

    from model_tracks import ablation, staged_ablation
    import core.common as common

    monkeypatch.setattr(common, "TRAIN_ROOT", tmp_path)
    setup = tmp_path / "setup"
    template = setup / "ablation_templates/gnn_only"
    template.mkdir(parents=True)
    checkpoint = tmp_path / "run/gnn_only/checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({"vocabulary": {}, "support_records": [], "manifest": {"track": "gnn_only"}},
               checkpoint)
    tensors = template / "prepared_inputs.npz"
    tensors.write_bytes(b"frozen local topology")
    request = {"checkpoint": "@setup/template.pt",
               "graph_binding": ablation.digest({"vocabulary": {}, "support_records": []}),
               "sources": {"@setup/template.pt": "placeholder"},
               "settings": {"retrieval_catalog": "full"},
               "prepared_inputs": {"size": ablation.file_size(tensors)}}
    (template / "request.json").write_text(json.dumps(request))
    monkeypatch.setattr(staged_ablation, "encode", lambda *a, **k: None)

    staged_ablation.forward(checkpoint.parent, setup, "gnn_only", checkpoint, device="cuda")
    frame = read_validated(trace_target)
    steps = stage_steps(frame, "staged_ablation")
    assert {"forward.template", "forward.checkpoint_select", "forward.vectors",
            "forward.completed"} <= steps, steps

    select = only(frame, "staged_ablation", "forward.checkpoint_select")
    assert select["scope"] == tracing.SCOPE_ENTITY and select["key"] == "gnn_only"
    select_detail = tracing.detail_json(select["detail"])
    assert select_detail["role"] == "selected"
    assert select_detail["bound_source"] == "@suite/gnn_only/checkpoint.pt"

    vectors = only(frame, "staged_ablation", "forward.vectors")
    vectors_detail = tracing.detail_json(vectors["detail"])
    # the stub encoder produced nothing: the row says so instead of inventing a digest
    assert vectors_detail["reused_existing_export"] is False
    assert vectors_detail["vectors_present"] is False
    assert vectors_detail["size"] is None


def test_baseline_forward_records_the_frozen_text_template(tmp_path, monkeypatch,
                                                           trace_target):
    """``baseline_ablation.forward`` reuses the frozen text template and saved vectors."""
    import json

    from model_tracks import baseline_ablation

    setup = tmp_path / "setup"
    template = setup / "ablation_templates/text"
    template.mkdir(parents=True)
    (template / "request.json").write_text(json.dumps(
        {"settings": {"retrieval_catalog": "full", "coverage": "sampled"}}))
    (setup / "shared_minilm__embeddings.npz").write_bytes(b"saved vectors")
    monkeypatch.setattr(baseline_ablation, "forward_staged",
                        lambda *a, **k: tmp_path / "bound/request.json")

    path = baseline_ablation.forward(tmp_path, setup, tmp_path / "checkpoint", device="cuda")
    assert path == tmp_path / "bound/request.json"
    frame = read_validated(trace_target)
    row = only(frame, "baseline_ablation", "forward.text_template")
    detail = tracing.detail_json(row["detail"])
    assert detail["coverage"] == "sampled"
    assert detail["saved_text"].endswith("shared_minilm__embeddings.npz")
    assert row["key"] == "text"


def test_baseline_forward_lands_where_complete_reads(tmp_path, monkeypatch,
                                                     trace_target):
    """Baseline forward writes the SAME request path ``baseline_ablation.complete`` reads.

    Regression guard: routing baseline forward through ``Results.track_dir``
    wrote ``<run>/text/ablation/request.json`` while complete read
    ``<run>/baseline/ablation/request.json``. The path derivation is NOT stubbed
    here (only the device encoder is), so the two halves must agree.
    """
    import json

    from core import common
    from core.bundle import bundle_spec
    from model_tracks import ablation, baseline_ablation, staged_ablation

    monkeypatch.setattr(common, "TRAIN_ROOT", tmp_path)
    run = tmp_path / "run"
    output = run / "baseline"
    setup = tmp_path / "setup"
    template = setup / bundle_spec().ablation_templates_dir / "text"
    template.mkdir(parents=True)
    (template / "prepared_inputs.npz").write_bytes(b"frozen local topology")
    (setup / "shared_minilm__embeddings.npz").write_bytes(b"saved vectors")

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights").write_bytes(b"frozen baseline")
    identity = ablation.checkpoint_identity(checkpoint)
    (template / bundle_spec().ablation_request_file).write_text(json.dumps({
        "track": "text",
        "checkpoint": str(checkpoint),
        "sources": {str(checkpoint): identity},
        "settings": {"retrieval_catalog": "full", "coverage": "sampled"},
        "variants": [], "cohort_size": 2,
    }))
    # The device encode is the only stub; the folder/request derivation is real.
    monkeypatch.setattr(staged_ablation, "encode", lambda *a, **k: None)

    path = baseline_ablation.forward(output, setup, checkpoint, device="cpu")

    request_name = bundle_spec().ablation_request_file
    assert path == baseline_ablation._request_path(output)
    assert path == output / "ablation" / request_name
    assert path.is_file()
    # The regression's divergent location must never be produced.
    assert not (run / "text" / "ablation" / request_name).exists()


# ── 5. the ablation consumer quarantines a mismatched calibration ──────────
def test_post_training_ablation_quarantines_a_calibration_mismatch(tmp_path, monkeypatch,
                                                                   trace_target):
    """``complete_saved`` refuses (and names) a calibration for another checkpoint."""
    from model_tracks import post_training_ablation as auto
    from model_tracks.ablation import write as write_json

    track = tmp_path / "text"
    folder = track / "ablation"
    folder.mkdir(parents=True)
    checkpoint = track / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights").write_bytes(b"selected")
    write_json(folder / "request.json", {"checkpoint": str(checkpoint)})
    (folder / "vectors.npz").write_bytes(b"exports")
    _manifest(track / "text__completion_manifest.json", "text", False)

    with pytest.raises(ValueError, match="calibration differs"):
        auto.complete_saved(tmp_path, SimpleNamespace(ablation_config="unused"))

    frame = read_validated(trace_target)
    steps = stage_steps(frame, "post_training_ablation")
    assert {"complete_saved.calibration_source", "complete_saved.checkpoint_mismatch"} <= steps, steps
    rejected = only(frame, "post_training_ablation", "complete_saved.checkpoint_mismatch")
    assert rejected["scope"] == tracing.SCOPE_ENTITY and rejected["key"] == "text"
    assert "different checkpoint" in rejected["reason"]


# ── 6. ablation prepare -> encode -> report ───────────────────────────────
class _StubEncoder:
    """The frozen text encoder, stubbed: eval-compatible and deterministic."""

    device = SimpleNamespace(type="cpu")

    def eval(self):
        return self

    def __call__(self, features):
        import torch

        return {"sentence_embedding": torch.ones(len(features["input_ids"]), 4,
                                                dtype=torch.float32)}


def _ablation_materials(tmp_path, monkeypatch):
    """A real prepare()/encode()/report() fixture for the text track.

    Only the sentence encoder is stubbed (the same shape
    tests/test_attribute_ablation.py uses); every ablation code path runs.
    """
    import sys
    import types

    import torch

    from core import encoding_inputs
    from model_tracks import ablation as ab

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(
        SentenceTransformer=lambda *a, **k: _StubEncoder()))
    monkeypatch.setattr(encoding_inputs, "tokenization_policy",
                        lambda model: {"truncation": False})
    # ``ablation`` imported the function itself, so its own binding is patched
    # too: encode() compares the worker's policy against the frozen plan.
    monkeypatch.setattr(ab, "tokenization_policy", lambda model: {"truncation": False})
    monkeypatch.setattr(encoding_inputs, "prepare_text_features",
                        lambda model, texts, **kw: {
                            "input_ids": torch.zeros(len(texts), 1, dtype=torch.long),
                            "attention_mask": torch.ones(len(texts), 1, dtype=torch.long)})
    monkeypatch.setattr(ab, "load_token_features",
                        lambda arrays, batch, device: {
                            "input_ids": torch.zeros(batch["count"], 1, dtype=torch.long),
                            "attention_mask": torch.ones(batch["count"], 1, dtype=torch.long)})
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "frozen").write_text("test")
    catalog = tmp_path / "catalog.csv"
    pairs = tmp_path / "pairs.csv"
    config = tmp_path / "config.yaml"
    pd.DataFrame([
        {"sku_id": "a", "gtin": "0001", "brand": "A", "sku_name_eng": "Water",
         "attribute": "Volume: 500 ml"},
        {"sku_id": "b", "gtin": "0002", "brand": "A", "sku_name_eng": "Water",
         "attribute": "Volume: 1000 ml"},
    ]).to_csv(catalog, index=False)
    pd.DataFrame([{"sku_id1": "a", "sku_id2": "b", "label": "0", "split": "dev"}]).to_csv(
        pairs, index=False)
    cfg = ab.settings().model_dump()
    cfg.update(attributes=["volume", "coffee type"], output_dir=str(tmp_path / "out"),
               report_path=str(tmp_path / "report.json"))
    config.write_text(_yaml(cfg))
    return ab, {"checkpoint": checkpoint, "catalog": catalog, "pairs": pairs,
                "config": config}


def test_ablation_prepare_encode_report_rows_land(tmp_path, monkeypatch, trace_target):
    """`prepare` freezes the cohort, `encode` runs the jobs, `report` censuses flips."""
    import json

    ab, paths = _ablation_materials(tmp_path, monkeypatch)
    request_path = ab.prepare(paths["catalog"], paths["pairs"], paths["checkpoint"],
                              config=paths["config"])
    vectors = request_path.parent / "vectors.npz"
    ab.encode(request_path, vectors, device="cpu")
    frozen = tmp_path / "baseline.json"
    frozen.write_text(json.dumps({
        "threshold": 0.5, "track": "text",
        "checkpoint_size": ab.checkpoint_identity(paths["checkpoint"])}))
    report_path = ab.report(request_path, vectors, 0.5, threshold_source=str(frozen),
                            config=paths["config"])
    assert report_path.is_file() and vectors.is_file()

    frame = read_validated(trace_target)
    steps = stage_steps(frame, "ablation")
    assert {
        "prepare.pairs_selected", "prepare.catalog_rows", "prepare.endpoints",
        "prepare.attributes", "prepare.variants", "prepare.variant_effect.reason_census",
        "prepare.candidates", "prepare.request_persisted",
        "encode.plan", "encode.text_vectors", "encode.candidates", "encode.jobs",
        "encode.persisted",
        "vectors.validated", "report.validated", "report.comparisons",
        "report.decision_flip.reason_census", "report.persisted",
    } <= steps, steps

    # the cohort funnel: the one dev pair in the file is the one selected pair
    pairs_row = only(frame, "ablation", "prepare.pairs_selected")
    assert int(pairs_row["in_count"]) == 1 and int(pairs_row["out_count"]) == 1

    # the no-effect interventions are prepare's ENTITY exceptions, with keys
    no_effect = frame[(frame["stage"] == "ablation")
                      & (frame["step"] == "prepare.variant_effect.reason_census")
                      & (frame["reason"] == "no_changed_input")]
    assert len(no_effect) == 1 and int(no_effect.iloc[0]["in_count"]) >= 1
    effects = frame[(frame["stage"] == "ablation") & (frame["step"] == "prepare.variant_effect")]
    assert set(effects["scope"]) == {tracing.SCOPE_ENTITY}
    assert {"volume:text", "coffee type:text"} <= set(effects["key"])

    # the flip census is exact: its buckets sum to the emitted comparison rows
    comparisons = only(frame, "ablation", "report.comparisons")
    census = frame[(frame["stage"] == "ablation")
                   & (frame["step"] == "report.decision_flip.reason_census")]
    assert set(census["reason"]) <= {"flip", "no_flip"}
    assert int(census["in_count"].astype(int).sum()) == int(comparisons["out_count"])
    request = json.loads(request_path.read_text())
    expected_rows = (len(request["variants"]) - 1) * len(request["pairs"])
    assert int(comparisons["in_count"]) == int(comparisons["out_count"]) == expected_rows


# ── 7. baseline ablation: calibration -> frozen report -> persist ──────────
def test_baseline_ablation_complete_rows(tmp_path, monkeypatch, trace_target):
    """`baseline_ablation.complete` fits the dev threshold and seals the report."""
    import json
    import shutil

    import numpy as np

    from model_tracks import baseline_ablation as ba

    ab, paths = _ablation_materials(tmp_path, monkeypatch)
    request_path = ab.prepare(paths["catalog"], paths["pairs"], paths["checkpoint"],
                              config=paths["config"])
    request = json.loads(request_path.read_text())
    request["checkpoint_role"] = "baseline"

    output = tmp_path / "run/text"
    folder = output / "ablation"
    folder.mkdir(parents=True)
    ab.write(folder / "request.json", request)
    shutil.copy2(request_path.parent / "prepared_inputs.npz", folder / "prepared_inputs.npz")
    ab.encode(folder / "request.json", folder / "vectors.npz", device="cpu")

    setup = tmp_path / "setup" / "prepared"
    setup.mkdir(parents=True)
    (setup / "listings.json").write_text("{}")
    (setup / "pairs.csv").write_text("sku_id1,sku_id2,label,split\n")
    (output / "shared_minilm__embeddings.npz").write_bytes(b"saved vectors")

    identity = ab.checkpoint_identity(paths["checkpoint"])
    monkeypatch.setattr(ba, "load_records", lambda path: [{"sku_id": "a"}, {"sku_id": "b"},
                                                         {"sku_id": "c"}])
    monkeypatch.setattr(ba, "load_text_cache",
                        lambda path, ids: (np.eye(3, 4, dtype=np.float32),
                                           {"checkpoint_size": identity}))
    monkeypatch.setattr(ba, "load_pairs", lambda path, records: {
        "dev": (np.array([[0, 1], [1, 2]]), np.array([1, 0]))})

    report_path = ba.complete(output, tmp_path / "setup")
    assert report_path.is_file()

    frame = read_validated(trace_target)
    steps = stage_steps(frame, "baseline_ablation")
    assert {"complete.load", "complete.calibration", "complete.binding",
            "complete.frozen_report", "complete.persisted"} <= steps, steps

    calibration = only(frame, "baseline_ablation", "complete.calibration")
    calibration_detail = tracing.detail_json(calibration["detail"])
    assert calibration_detail["checkpoint_size"] == identity
    assert calibration_detail["dev_pairs"] == 2
    assert calibration_detail["dev_positives"] == 1
    assert calibration_detail["dev_negatives"] == 1
    assert calibration_detail["test_used_for_selection"] is False
    assert int(calibration["in_count"]) == 2

    # the nested report rows land in the same trace (the frozen threshold row)
    assert "report.validated" in stage_steps(frame, "ablation")
    binding = only(frame, "baseline_ablation", "complete.binding")
    assert tracing.detail_json(binding["detail"])["threshold"] == calibration_detail["threshold"]


# ── 8. the exhaustive ablation cohort ─────────────────────────────────────
def test_ablation_cohort_rows_and_difficulty_skip_census(tmp_path, monkeypatch, trace_target):
    """`prepare_cohort` censuses every pair and names the unmeasured difficulty."""
    import shutil

    from model_tracks import ablation_cohort

    repo = Path(__file__).resolve().parents[1]
    setup = tmp_path / "smoke_200"
    shutil.copytree(repo / "data/prepared/smoke_200", setup)
    from training.prepared_bundle import load_prepared_bundle

    _, bundle = load_prepared_bundle(setup / "text_prepared.pkl.gz", verify_inputs=False)
    folder = ablation_cohort.prepare_cohort(setup, bundle)
    assert (folder / "pairs.csv").is_file()

    frame = read_validated(trace_target)
    steps = stage_steps(frame, "ablation_cohort")
    assert {
        "prepare_cohort.inputs", "prepare_cohort.clean_pairs", "prepare_cohort.bundle_pairs",
        "prepare_cohort.objective_pairs", "prepare_cohort.difficulty", "prepare_cohort.frame",
        "prepare_cohort.persisted",
    } <= steps, steps

    difficulty = only(frame, "ablation_cohort", "prepare_cohort.difficulty")
    detail = tracing.detail_json(difficulty["detail"])
    assert detail["measured"] + detail["unmeasured"] == int(difficulty["in_count"])
    assert int(difficulty["out_count"]) == detail["measured"]

    # the unmeasured pairs are the ENTITY exceptions, censused exactly
    budget = only(frame, "ablation_cohort",
                  "prepare_cohort.difficulty_skipped.sample_budget")
    assert int(budget["in_count"]) == detail["unmeasured"]
    census = frame[(frame["stage"] == "ablation_cohort")
                   & (frame["step"] == "prepare_cohort.difficulty_skipped.reason_census")]
    if detail["unmeasured"]:
        assert census["reason"].tolist() == ["no_frozen_encoder_input"]
        assert int(census["in_count"].astype(int).sum()) == detail["unmeasured"]

    # the frame row carries the frozen coverage census it validated
    frame_row = only(frame, "ablation_cohort", "prepare_cohort.frame")
    frame_detail = tracing.detail_json(frame_row["detail"])
    assert frame_detail["pair_rows"] == int(frame_row["out_count"])
    assert frame_detail["minted_endpoints_total"] >= frame_detail["minted_endpoints_covered"]


# ── 9. the staged ablation suite: templates frozen before training ─────────
def test_staged_prepare_suite_rows_and_staging_census(tmp_path, monkeypatch, trace_target):
    """`prepare_suite` freezes support/vocabulary and censuses the staging it drops."""
    import json
    import shutil

    from core import common
    from model_tracks import ablation as ab
    from model_tracks import staged_ablation

    repo = Path(__file__).resolve().parents[1]
    setup = tmp_path / "smoke_200"
    shutil.copytree(repo / "data/prepared/smoke_200", setup)
    monkeypatch.setattr(common, "TRAIN_ROOT", tmp_path)
    config = tmp_path / "ablation.yaml"
    cfg = ab.settings().model_dump()
    cfg.update(output_dir=str(tmp_path / "templates"), coverage="sampled",
               attributes=["volume"])
    config.write_text(_yaml(cfg))
    # a LOCAL checkpoint directory: the template builder pins it by content hash
    baseline = tmp_path / "text_baseline"
    baseline.mkdir()
    (baseline / "model.safetensors").write_bytes(b"frozen baseline")

    prepared: list[str] = []

    def fake_prepare(catalog, pairs, checkpoint, **kwargs):
        """The stub: emit a request + its tensors, exactly as prepare() would."""
        prepared.append(str(kwargs.get("track")))
        staging = tmp_path / "staging" / str(len(prepared))
        staging.mkdir(parents=True)
        (staging / "prepared_inputs.npz").write_bytes(b"tensors")
        (staging / "request.json").write_text(json.dumps({
            "sources": {}, "cohort_size": "cohort", "coverage": {"mode": "sampled"},
            "variants": [], "checkpoint": "template.pt", "text_checkpoint": None,
            "settings": {"retrieval_catalog": "full"}}))
        return staging / "request.json"

    monkeypatch.setattr(staged_ablation, "prepare", fake_prepare)
    # a pre-existing content-addressed staging dir the cleanup must drop
    leftover = setup / "ablation_templates" / "deadbeef"
    leftover.mkdir(parents=True)
    (leftover / "request.json").write_text("{}")

    templates = staged_ablation.prepare_suite(setup, baseline, config)
    assert templates == setup / "ablation_templates"
    assert not leftover.exists()

    frame = read_validated(trace_target)
    steps = stage_steps(frame, "staged_ablation")
    assert {
        "prepare_suite.cohort_gate", "prepare_suite.support_vocabulary",
        "prepare_suite.cleanup_staging", "prepare_suite.removed_staging.reason_census",
        "prepare_suite.completed",
    } <= steps, steps

    support = only(frame, "staged_ablation", "prepare_suite.support_vocabulary")
    support_detail = tracing.detail_json(support["detail"])
    assert int(support["in_count"]) == support_detail["listing_records"]
    assert int(support["out_count"]) == support_detail["train_support"] > 0
    assert support_detail["vocabulary"] > 0

    # the dropped staging dirs are censused exactly and named at entity grain
    cleanup = only(frame, "staged_ablation", "prepare_suite.cleanup_staging")
    census = frame[(frame["stage"] == "staged_ablation")
                   & (frame["step"] == "prepare_suite.removed_staging.reason_census")]
    assert int(cleanup["out_count"]) == 0
    assert int(census["in_count"].astype(int).sum()) == int(cleanup["in_count"]) == 1
    removed = frame[(frame["stage"] == "staged_ablation")
                    & (frame["step"] == "prepare_suite.removed_staging")]
    assert set(removed["key"]) == {"deadbeef"}

    done = only(frame, "staged_ablation", "prepare_suite.completed")
    assert tracing.detail_json(done["detail"])["tracks"] == ["text", "gnn_only"]
    assert prepared == ["text", "gnn_only"]


# ── 10. the frozen snapshot wrapper ───────────────────────────────────────
def test_snapshot_completion_records_inventory_and_receipt(tmp_path, monkeypatch, trace_target):
    """`snapshot_completion.complete` materializes the frozen runtime and receipts it."""
    import json

    from model_tracks import snapshot_completion
    from model_tracks.local_complete import complete

    training_zip, input_zip = _local_suite(tmp_path, monkeypatch, with_runtime_source=True)
    final = complete(training_zip, input_zip, "run")

    def fake_frozen_run(command, cwd=None, env=None, check=False):
        """Stand in for the frozen subprocess: report the already-sealed archive."""
        assert "local_complete" in command[2]
        Path(command[6]).write_text(json.dumps({"final": str(final)}))
        assert Path(cwd).is_dir()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(snapshot_completion.subprocess, "run", fake_frozen_run)
    assert snapshot_completion.complete(training_zip, input_zip, "run") == final

    frame = read_validated(trace_target)
    steps = stage_steps(frame, "snapshot_completion")
    assert {
        "complete.runtime_inventory", "complete.working_tree_drift.sample_budget",
        "complete.frozen_runtime", "complete.receipt",
    } <= steps, steps

    inventory = only(frame, "snapshot_completion", "complete.runtime_inventory")
    inventory_detail = tracing.detail_json(inventory["detail"])
    assert inventory_detail["local_complete_present"] is True
    assert int(inventory["out_count"]) == inventory_detail["inventory_files"] >= 1
    drift = only(frame, "snapshot_completion", "complete.working_tree_drift.sample_budget")
    assert inventory_detail["working_tree_mismatches"] == int(drift["in_count"])

    receipt = only(frame, "snapshot_completion", "complete.receipt")
    receipt_detail = tracing.detail_json(receipt["detail"])
    assert receipt_detail["final_archive_size"] and receipt_detail["run_tag"] == "run"
    assert receipt_detail["working_tree_mismatches"] == inventory_detail["working_tree_mismatches"]
