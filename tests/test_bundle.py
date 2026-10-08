"""Bundle boundary + role enforcement.

The ``Bundle`` is the one sealed artifact that travels generation -> training ->
post-training. These tests pin its public contract: a single boundary integrity
check (``Bundle.load``) that fails loud on a corrupt archive, role enforcement
(result = selected checkpoint only; recovery = all epochs + optimizer; inputs =
none), and a result seal that ships the selected-only member set.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.bundle import Bundle, BundlePipeline, BundleRole, manifest_name
from core.portable_archive import verify_archive_digest, write_archive


def _spec():
    from core.bundle import _bundle_spec
    return _bundle_spec()


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def _result_tree(root: Path) -> dict[str, Path]:
    """A small result tree: two epochs, profiling, reports and a baseline."""
    files = {
        "text/_checkpoints/m/r-t_f0/checkpoint-281/model.safetensors": "weights-281",
        "text/_checkpoints/m/r-t_f0/checkpoint-281/optimizer.pt": "resume-only",
        "text/_checkpoints/m/r-t_f0/checkpoint-562/model.safetensors": "weights-562",
        "text/text__vectors.npz": "vectors",
        "baseline/shared_minilm__embeddings.npz": "baseline",
        "suite_events.jsonl": "{}\n",
        "suite_manifest.json": "{}\n",
        "text/profiles/r-t/fold0/training_trace.json": "{}",
        "resource_profile/gpu.csv": "x",
        "wandb/run-1/files/config.yaml": "x",
    }
    _write(root / "text/_checkpoints/m/r-t_f0/checkpoint-281/trainer_state.json", (
        '{"best_model_checkpoint": "checkpoint-281", "best_metric": 0.9, '
        '"global_step": 1124}'))
    return {name: _write(root / name, value) for name, value in files.items()}


def _seal(tree: Path, archive: Path, *, role: BundleRole = BundleRole.result,
          metadata: dict | None = None) -> Path:
    spec = _spec()
    manifest = spec.manifest_result if role is BundleRole.result else (
        spec.manifest_recovery if role is BundleRole.recovery else spec.manifest_inputs)
    files = {p.relative_to(tree).as_posix(): p for p in tree.rglob("*") if p.is_file()}
    return write_archive(archive, files, manifest_name=manifest,
                         metadata=metadata or ({spec.run_tag_key: "r-tag"}
                                               if role is not BundleRole.inputs else {}))


def test_load_inputs_role_carries_no_checkpoints(tmp_path: Path) -> None:
    tree = tmp_path / "inputs"
    _write(tree / "data/model_tracks/shared/listings.json", "{}")
    archive = _seal(tree, tmp_path / "inputs.tar.zst", role=BundleRole.inputs)

    bundle = Bundle.load(archive, BundleRole.inputs)
    assert bundle.role is BundleRole.inputs
    assert bundle.manifest_name == manifest_name(BundleRole.inputs)
    assert bundle.checkpoint("text") is None
    assert bundle.checkpoints("text") == []


def test_load_rejects_truncated_archive(tmp_path: Path) -> None:
    tree = tmp_path / "result"
    _result_tree(tree)
    archive = _seal(tree, tmp_path / "result.tar.zst")

    truncated = tmp_path / "truncated.tar.zst"
    payload = archive.read_bytes()
    truncated.write_bytes(payload[: len(payload) // 2])
    with pytest.raises(ValueError):
        Bundle.load(truncated, BundleRole.result)
    # The corruption guard is the boundary contract: the good archive still loads.
    assert Bundle.load(archive, BundleRole.result).role is BundleRole.result


def test_load_expected_digest_mismatch_fails(tmp_path: Path) -> None:
    tree = tmp_path / "result"
    _result_tree(tree)
    archive = _seal(tree, tmp_path / "result.tar.zst")
    _, observed = verify_archive_digest(archive, manifest_name(BundleRole.result))

    assert Bundle.load(archive, BundleRole.result, expected_digest=observed).digest == observed
    with pytest.raises(ValueError):
        Bundle.load(archive, BundleRole.result, expected_digest="0" * 64)


def test_result_role_keeps_only_selected_checkpoint(tmp_path: Path) -> None:
    tree = tmp_path / "result"
    _result_tree(tree)
    bundle = Bundle.from_directory(tree, BundleRole.result)

    assert bundle.selected_checkpoint_dirs() == frozenset(
        {"text/_checkpoints/m/r-t_f0/checkpoint-281"})
    members = bundle.collect_result_members()
    assert "text/_checkpoints/m/r-t_f0/checkpoint-281/model.safetensors" in members
    assert "text/_checkpoints/m/r-t_f0/checkpoint-562/model.safetensors" not in members
    assert "text/_checkpoints/m/r-t_f0/checkpoint-281/optimizer.pt" not in members
    assert "text/profiles/r-t/fold0/training_trace.json" not in members
    assert "resource_profile/gpu.csv" not in members
    assert "wandb/run-1/files/config.yaml" not in members
    assert "text/text__vectors.npz" in members
    assert "suite_events.jsonl" in members


def test_collect_result_members_is_result_only(tmp_path: Path) -> None:
    tree = tmp_path / "inputs"
    _write(tree / "data/model_tracks/shared/listings.json", "{}")
    with pytest.raises(ValueError):
        Bundle.from_directory(tree, BundleRole.inputs).collect_result_members()
    with pytest.raises(ValueError):
        Bundle.from_directory(tree, BundleRole.recovery).collect_result_members()


def test_recovery_role_keeps_every_epoch(tmp_path: Path) -> None:
    tree = tmp_path / "result"
    _result_tree(tree)
    bundle = Bundle.from_directory(tree, BundleRole.recovery)
    epochs = {p.name for p in bundle.checkpoints("text")}
    assert epochs == {"checkpoint-281", "checkpoint-562"}
    assert bundle.selected_checkpoint_dirs() == frozenset(
        {"text/_checkpoints/m/r-t_f0/checkpoint-281"})


def test_seal_result_round_trips_selected_only(tmp_path: Path) -> None:
    tree = tmp_path / "result"
    _result_tree(tree)
    sealed_path = tmp_path / "sealed.tar.zst"
    sealed = Bundle.from_directory(tree, BundleRole.result).seal_result(
        sealed_path, metadata={"run_tag": "r-tag"})

    assert sealed.role is BundleRole.result
    loaded = Bundle.load(sealed_path, BundleRole.result)
    members = set(loaded.members())
    assert "text/_checkpoints/m/r-t_f0/checkpoint-281/model.safetensors" in members
    assert "text/_checkpoints/m/r-t_f0/checkpoint-562/model.safetensors" not in members
    assert "text/_checkpoints/m/r-t_f0/checkpoint-281/optimizer.pt" not in members


def test_seal_result_requires_result_role(tmp_path: Path) -> None:
    tree = tmp_path / "inputs"
    _write(tree / "data/model_tracks/shared/listings.json", "{}")
    with pytest.raises(ValueError):
        Bundle.from_directory(tree, BundleRole.inputs).seal_result(tmp_path / "x.tar.zst")


def test_pipeline_steps_are_wired() -> None:
    pipeline = BundlePipeline(role=BundleRole.inputs, device="cpu", lane="kaggle")
    import inspect
    from model_tracks import bundle_steps

    assert inspect.signature(bundle_steps.prepare_inputs).parameters.keys() == {"pipeline"}
    assert set(inspect.signature(bundle_steps.finalize).parameters) == {"pipeline", "result"}
    with pytest.raises(ValueError):
        pipeline.prepare_inputs()  # no output configured
    with pytest.raises(ValueError):
        pipeline.finalize(Bundle.from_directory(
            pipeline.output or Path("/nonexistent"), BundleRole.inputs))
