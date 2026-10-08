"""core.artifacts.Artifacts owns the run's emitted artifacts — public API only.

One test per public behavior: collection from a run tree and from a verified
bundle handle, the role/track lookups, config resolution, and the STRUCTURAL
identity (run tag + names + sizes + counts, never a digest).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.artifacts import (
    Artifacts,
    artifacts_config_path,
    artifacts_spec,
    results_owned_components,
)


def _run_tree(root: Path) -> None:
    """A miniature run tree with one declared member per shape."""
    spec = artifacts_spec()
    writes = {
        "text/text__vectors.npz": 100,
        "text/text__reports/text__model_evaluation_summary.csv": 10,
        "text/text__reports/text__scored_pairs.csv": 20,
        "text/text__index/hnsw.bin": 5,
        "text/_checkpoints/checkpoint-1/trainer_state.json": 7,
        "text/track_complete.json": 3,
        # Results-owned: the ablation report must never be collected here.
        "text/ablation/report.json": 9,
        "gnn_only/gnn_only__graph_model.pt": 42,
        "gnn_only/gnn_only__inference/gnn_only__index/x.bin": 6,
        # The extracted prepared inputs are a process input, never a deliverable.
        "local_inputs/leak.csv": 1,
    }
    for relative, size in writes.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * size)
    run_decl = spec.run_artifacts["suite_manifest"]
    (root / "suite_manifest.json").write_text(json.dumps({"declared": run_decl.kind}))


def test_from_run_collects_and_classifies_the_declared_tree(tmp_path):
    """A declared tree aggregates; Results-owned members and inputs are excluded."""
    _run_tree(tmp_path)
    run = Artifacts.from_run(tmp_path, "r1")

    # trees aggregate to ONE member, with the summed bytes and the file count.
    index = run.member("text/text__index")
    assert index.kind == "dir" and index.declared == "index"
    assert (index.size_bytes, index.count) == (5, 1)
    reports = run.member("text/text__reports")
    assert (reports.size_bytes, reports.count) == (30, 2)

    # per-track and per-role lookups.
    assert run.manifest_for("result") == "suite_bundle_manifest.json"
    assert run.tracks() == ("text", "gnn_only")
    assert set(run.track_members("text")) >= {"text/text__vectors.npz", "text/text__index"}
    assert run.member("gnn_only/gnn_only__graph_model.pt").role == "result"
    assert run.member("text/_checkpoints").role == "recovery"
    assert "text/text__vectors.npz" in run.role_members("result")
    assert "text/_checkpoints" not in run.role_members("result")

    # the declared boundary: neither Results' ablation nor the prepared inputs.
    assert "text/ablation/report.json" not in run.names()
    assert not any(name.startswith("local_inputs/") for name in run.names())

    # the per-track bookkeeping markers classify through the BundleSpec names.
    marker = run.member("text/track_complete.json")
    assert (marker.declared, marker.track, marker.role) == ("track_complete", "text", "result")


def test_records_structural_identity_without_any_digest(tmp_path):
    """Identity is run tag + names + sizes + counts; it carries no hash and no bytes."""
    _run_tree(tmp_path)
    identity = Artifacts.from_run(tmp_path, "r1").identity()

    assert identity["schema"] == "er-run-artifacts-v1"
    assert identity["run_tag"] == "r1" and identity["sealed"] is False
    assert identity["member_count"] == len(identity["members"])
    assert identity["total_bytes"] > 0
    assert {row["name"] for row in identity["members"]} == set(
        Artifacts.from_run(tmp_path, "r1").names())
    dumped = json.dumps(identity)
    assert "sha" not in dumped and "digest" not in dumped


def test_from_bundle_reads_the_verified_handle(tmp_path):
    """A sealed Bundle handle's member names classify through the same declaration."""
    from core.bundle import Bundle, BundleRole

    _run_tree(tmp_path)
    # An unpacked tree with no canonical manifest: the role contract is the
    # caller's, and the collection reads member names only.
    handle = Bundle.from_directory(tmp_path, BundleRole.result)
    sealed = Artifacts.from_bundle(handle)

    assert sealed.sealed is True and sealed.role == BundleRole.result.value
    assert sealed.manifest_name == "suite_bundle_manifest.json"
    assert set(sealed.track_members("text")) == {
        "text/text__vectors.npz", "text/text__index", "text/text__reports",
        "text/track_complete.json", "text/_checkpoints"}
    # a dir-backed handle is sized; an archive-backed one is not (same members).
    assert sealed.member("text/text__index").size_bytes == 5
    assert sealed.identity()["total_bytes"] > 0

    with pytest.raises(ValueError):
        Artifacts.from_bundle(handle, spec=artifacts_spec().model_copy(
            update={"roles": {"only_one": "manifest_result"}}))


def test_resolve_reads_every_address_from_config(tmp_path):
    """Resolution goes through the phone book: no path literal lives in the class."""
    _run_tree(tmp_path)
    run = Artifacts.from_run(tmp_path, "r1", spec=artifacts_spec())

    assert artifacts_config_path().name == "artifacts.yaml"
    assert run.resolve("vectors", track="text", root=tmp_path) == \
        tmp_path / "text" / "text__vectors.npz"

    archives = run.sealed_archives(run_tag="r1")
    assert set(archives) == {"inputs", "recovery", "result", "training"}
    assert archives["recovery"] == Path("r1.recovery.tar.zst")
    assert archives["result"].name == "r1.tar.zst"

    # the Results boundary is askable and names the ablation members.
    owned = results_owned_components()
    assert "ablation" in owned and "post_training_ablation.json" in owned

    with pytest.raises(KeyError):
        run.resolve("no_such_artifact")
    with pytest.raises(KeyError):
        run.track_members("no_such_track")
