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
        ".resume/attempt.json": "{}",
        "mlruns/0/meta.yaml": "x",
        "mps_pipe/control": "x",
        "mps_log/server.log": "x",
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
    # A result archive is the SELECTED checkpoint set, never the raw tree: the
    # role boundary refuses the recovery shape (several epochs per family), so
    # the fixture seals exactly what the result role ships.
    files = (Bundle.from_directory(tree, BundleRole.result).collect_result_members()
             if role is BundleRole.result else
             {p.relative_to(tree).as_posix(): p for p in tree.rglob("*") if p.is_file()})
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


def test_load_enforces_the_role_membership_contract(tmp_path: Path) -> None:
    """Role membership is a LOAD contract, not a writer-only convention.

    Each archive below is well formed (its own inventory verifies), so the only
    remaining refusal reason is the role contract: an inputs bundle carrying
    weights, a result bundle carrying every epoch, and a recovery bundle whose
    recorded epoch was stripped of its resume state. A caller-declared manifest
    is a different contract and is not role-checked.
    """
    spec = _spec()

    weights_inputs = tmp_path / "inputs_weights"
    _write(weights_inputs / "data/x.json", "{}")
    _write(weights_inputs / "text/_checkpoints/m/r_f0/checkpoint-1/model.safetensors", "w")
    with pytest.raises(ValueError, match="inputs bundle role contract"):
        Bundle.load(_seal(weights_inputs, tmp_path / "inputs.tar.zst",
                          role=BundleRole.inputs), BundleRole.inputs)

    every_epoch = tmp_path / "result_epochs"
    _result_tree(every_epoch)
    tree_files = {p.relative_to(every_epoch).as_posix(): p
                  for p in every_epoch.rglob("*") if p.is_file()}
    raw_result = write_archive(tmp_path / "result.tar.zst", tree_files,
                              manifest_name=manifest_name(BundleRole.result),
                              metadata={_spec().run_tag_key: "r-tag"})
    with pytest.raises(ValueError, match="result bundle role contract"):
        Bundle.load(raw_result, BundleRole.result)
    # The same members under the caller's own manifest name are another
    # contract (the graph track's resume-capable superset bundle): not checked.
    custom = write_archive(tmp_path / "custom.tar.zst", tree_files,
                           manifest_name="gnn_only__bundle_manifest.json",
                           metadata={_spec().run_tag_key: "r-tag"})
    assert Bundle.load(custom, BundleRole.result,
                       manifest_name="gnn_only__bundle_manifest.json").role \
        is BundleRole.result

    pruned_recovery = tmp_path / "recovery_pruned"
    checkpoint = pruned_recovery / spec.checkpoint_dir / "m/r_f0/checkpoint-1"
    _write(checkpoint / "model.safetensors", "w")
    _write(checkpoint / spec.trainer_state_file, '{"best_model_checkpoint": "checkpoint-1"}')
    recovery = _seal(pruned_recovery, tmp_path / "recovery.tar.zst",
                     role=BundleRole.recovery)
    with pytest.raises(ValueError, match="recovery bundle role contract"):
        Bundle.load(recovery, BundleRole.recovery)


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
    # Resume/diagnostic trees the delivered bundle must never carry.
    assert ".resume/attempt.json" not in members
    assert "mlruns/0/meta.yaml" not in members
    assert "mps_pipe/control" not in members
    assert "mps_log/server.log" not in members
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
    finalize_params = inspect.signature(bundle_steps.finalize).parameters
    # The contract is the REQUIRED surface: pipeline + result. Every other
    # parameter must be optional keyword-only (e.g. ``inputs``, the already
    # verified boundary handle a finalize caller may pass to avoid a second
    # verify of the same bytes in one process).
    required = {
        name for name, param in finalize_params.items()
        if param.default is inspect.Parameter.empty
    }
    assert required == {"pipeline", "result"}, required
    assert all(
        param.kind is inspect.Parameter.KEYWORD_ONLY
        for name, param in finalize_params.items()
        if name not in {"pipeline", "result"}
    )
    with pytest.raises(ValueError):
        pipeline.prepare_inputs()  # no output configured
    with pytest.raises(ValueError):
        pipeline.finalize(Bundle.from_directory(
            pipeline.output or Path("/nonexistent"), BundleRole.inputs))


def test_seal_archive_captures_the_transport_digest_and_members(tmp_path: Path) -> None:
    """Sealing yields the whole-file digest without reading the archive back.

    The digest is the transport token (the ``.sha256`` sidecar), and the member
    list is captured by the same boundary pass that verifies the archive, so a
    later stage neither re-reads nor re-parses it.
    """
    tree = tmp_path / "inputs"
    _write(tree / "data/model_tracks/shared/listings.json", "{}")
    files = {p.relative_to(tree).as_posix(): p for p in tree.rglob("*") if p.is_file()}
    archive = tmp_path / "inputs.tar.zst"

    sealed = Bundle.seal_archive(archive, files, role=BundleRole.inputs,
                                 metadata={_spec().run_tag_key: "r-tag"})
    _, observed = verify_archive_digest(archive, manifest_name(BundleRole.inputs))
    assert sealed.digest == observed
    assert sealed.role is BundleRole.inputs

    handle = Bundle.load(archive, BundleRole.inputs)
    assert handle.digest == observed
    assert handle.members() == sorted(handle.member_names)
    # The member list came from the boundary pass: it still answers after the
    # archive path is gone, which is exactly "no stage re-parses".
    moved = archive.with_name("moved.tar.zst")
    archive.rename(moved)
    assert handle.members() == sorted(handle.member_names)


def test_seal_archive_round_trips_through_materialize(tmp_path: Path) -> None:
    """A transport loads a Bundle, materializes it, and can re-seal it unchanged."""
    tree = tmp_path / "inputs"
    _write(tree / "data/model_tracks/shared/listings.json", "{}")
    _write(tree / "src/model_tracks/run.py", "code")
    files = {p.relative_to(tree).as_posix(): p for p in tree.rglob("*") if p.is_file()}
    first = Bundle.seal_archive(tmp_path / "a.tar.zst", files, role=BundleRole.inputs,
                                metadata={_spec().run_tag_key: "r-tag"})

    loaded = Bundle.load(first.path, BundleRole.inputs)
    restored = loaded.materialize(tmp_path / "tree")
    assert "src/model_tracks/run.py" in restored.members()
    assert loaded.manifest_name not in restored.members()  # manifest is container state
    second = Bundle.seal_archive(tmp_path / "b.tar.zst",
                                 {member: restored.local / member
                                  for member in restored.members()},
                                 role=BundleRole.inputs,
                                 metadata={_spec().run_tag_key: "r-tag"})
    assert Bundle.load(second.path, BundleRole.inputs).members() == loaded.members()


def test_recovery_seal_keeps_every_epoch_and_optimizer(tmp_path: Path) -> None:
    """The recovery role is 'all epochs + optimizer'; nothing is selected away."""
    tree = tmp_path / "result"
    _result_tree(tree)
    files = {p.relative_to(tree).as_posix(): p for p in tree.rglob("*") if p.is_file()}
    # The role contract is asserted positively before the write: a recovery
    # bundle that pruned the resume-only epoch state is refused, not sealed.
    from model_tracks.package import _assert_recovery_contract
    _assert_recovery_contract(tree, files)
    with pytest.raises(ValueError, match="pruned resume state"):
        _assert_recovery_contract(tree, {
            name: path for name, path in files.items() if "checkpoint-562" not in name})
    sealed = Bundle.seal_archive(tmp_path / "recovery.tar.zst", files,
                                 role=BundleRole.recovery,
                                 metadata={_spec().run_tag_key: "r-tag"})

    handle = Bundle.load(sealed.path, BundleRole.recovery)
    members = set(handle.members())
    assert "text/_checkpoints/m/r-t_f0/checkpoint-281/optimizer.pt" in members
    assert "text/_checkpoints/m/r-t_f0/checkpoint-562/model.safetensors" in members
    # Epoch enumeration needs the materialized tree (the archive is trusted, not
    # listed): the recovery contract is 'all epochs + optimizer'.
    restored = handle.materialize(tmp_path / "recovery_tree")
    assert {p.name for p in restored.checkpoints("text")} == {"checkpoint-281", "checkpoint-562"}


def test_result_role_drops_extracted_prepared_inputs(tmp_path: Path) -> None:
    """A finalize job's extracted prepared inputs are an input, never a deliverable."""
    tree = tmp_path / "result"
    _result_tree(tree)
    spec = _spec()
    _write(tree / spec.prepared_inputs_dir / "data/model_tracks/shared/suite.yaml", "yaml")
    bundle = Bundle.from_directory(tree, BundleRole.result)

    members = bundle.collect_result_members()
    assert not any(spec.prepared_inputs_dir in Path(member).parts for member in members)
    assert "text/text__vectors.npz" in members


def test_supervisor_success_path_needs_no_events_sidecar(tmp_path: Path, monkeypatch) -> None:
    """Single-archive handoff: a sealed run keeps no second `.events.jsonl` copy.

    On success the suite event stream was flushed into the result archive before
    sealing, so the lane has nothing else to download; only a run that never
    sealed retains the sidecar beside its output.
    """
    from types import SimpleNamespace
    from model_tracks import run as suite_run

    spec = _spec()
    monkeypatch.setattr(suite_run, 'load_config', lambda _: SimpleNamespace(
        profiling=False, result_archive_format='zip', dvc_enabled=False,
        post_training_ablation=False))
    monkeypatch.setattr(suite_run, '_run', lambda *a, **k: tmp_path / 'suite.zip')
    archive = suite_run.run(tmp_path / 'suite.yaml', tmp_path / 'suite', 'suite')

    assert archive == tmp_path / 'suite.zip'
    assert (tmp_path / 'suite' / spec.suite_events_file).is_file()
    assert not (tmp_path / ('suite' + spec.events_sidecar_suffix)).exists()


def test_bundle_steps_lane_entrypoint_refuses_finalize_without_a_result(tmp_path: Path) -> None:
    """The lane entrypoint is the remote CPU job surface (no operator box)."""
    from model_tracks import bundle_steps

    with pytest.raises(SystemExit):
        bundle_steps.main(['--role', 'result', '--lane', 'kaggle',
                           '--output', str(tmp_path / 'sealed.tar.zst')])


def test_bundle_steps_entrypoint_hands_the_lane_fields_to_the_step(tmp_path: Path,
                                                                  monkeypatch) -> None:
    """`python -m model_tracks.bundle_steps` builds the pipeline from its flags."""
    from types import SimpleNamespace
    from model_tracks import bundle_steps

    captured = {}
    monkeypatch.setattr(bundle_steps, 'Bundle',
                        SimpleNamespace(load=lambda path, role, **kw: 'result-bundle'))
    monkeypatch.setattr(bundle_steps, 'finalize',
                        lambda pipeline, result: captured.update(pipeline=pipeline,
                                                                 result=result)
                        or SimpleNamespace(path=tmp_path / 'sealed.tar.zst'))
    bundle_steps.main(['--role', 'result', '--lane', 'kaggle', '--device', 'cpu',
                       '--inputs', str(tmp_path / 'inputs.tar.zst'),
                       '--result', str(tmp_path / 'result.tar.zst'),
                       '--output', str(tmp_path / 'sealed.tar.zst'),
                       '--sparse-path', 'src/model_tracks'])

    pipeline = captured['pipeline']
    assert captured['result'] == 'result-bundle'
    assert pipeline.role is BundleRole.result
    assert pipeline.lane == 'kaggle' and pipeline.device == 'cpu'
    assert pipeline.inputs == tmp_path / 'inputs.tar.zst'
    assert pipeline.output == tmp_path / 'sealed.tar.zst'
    assert pipeline.sparse_paths == ('src/model_tracks',)


def _graph_marker_tree(root: Path, recorded: str, members: tuple[str, ...]) -> None:
    """One graph track tree: the ``best_checkpoint`` marker plus member copies."""
    _write(root / "gnn_only/gnn_only__best_checkpoint.json",
           '{"path": "' + recorded + '"}')
    for member in members:
        _write(root / "gnn_only" / member, "weights")


def test_checkpoint_resolves_a_marker_by_its_parent_qualified_name(tmp_path: Path) -> None:
    """The name PAIR identifies the member, even when bare names are duplicated.

    A real marker records a remote absolute path, and the same model name can sit
    under several parents (the 1008 suite ships four copies). The documented rule
    is parent-qualified first, so a unique parent-qualified match still resolves
    to exactly one member rather than tripping the ambiguity guard.
    """
    tree = tmp_path / "result"
    selected_name = ("run-gnn_only/_checkpoints/gnn_only/run_f0/checkpoint-2/"
                     "gnn_only__graph_model.pt")
    _graph_marker_tree(
        tree, "/content/run/results/gnn_only/" + selected_name,
        (selected_name,
         "run-gnn_only/_checkpoints/gnn_only/run_f0/checkpoint-1/gnn_only__graph_model.pt",
         "staging/checkpoint-3/gnn_only__graph_model.pt",
         "gnn_only__graph_model.pt"))

    selected = Bundle.from_directory(tree, BundleRole.result).checkpoint("gnn_only")
    assert selected == tree / "gnn_only" / selected_name


def test_checkpoint_refuses_a_marker_that_matches_two_members(tmp_path: Path) -> None:
    """Two members answering the recorded name are refused, never name-ranked."""
    tree = tmp_path / "result"
    _graph_marker_tree(tree, "/remote/checkpoint-2/gnn_only__graph_model.pt",
                       ("a/checkpoint-2/gnn_only__graph_model.pt",
                        "b/checkpoint-2/gnn_only__graph_model.pt"))

    with pytest.raises(ValueError, match="ambiguous selected checkpoint: gnn_only"):
        Bundle.from_directory(tree, BundleRole.result).checkpoint("gnn_only")


def test_checkpoint_refuses_an_ambiguous_bare_name_fallback(tmp_path: Path) -> None:
    """The bare-name transport fallback is refused when it is not unique either."""
    ambiguous = tmp_path / "ambiguous"
    _graph_marker_tree(ambiguous, "gnn_only__graph_model.pt",
                       ("a/deep/gnn_only__graph_model.pt",
                        "b/gnn_only__graph_model.pt"))
    with pytest.raises(ValueError, match="ambiguous selected checkpoint: gnn_only"):
        Bundle.from_directory(ambiguous, BundleRole.result).checkpoint("gnn_only")

    # The fallback itself is unchanged: one bare-name match still resolves.
    single = tmp_path / "single"
    _graph_marker_tree(single, "gnn_only__graph_model.pt",
                       ("deep/gnn_only__graph_model.pt",))
    assert Bundle.from_directory(single, BundleRole.result).checkpoint("gnn_only") == (
        single / "gnn_only/deep/gnn_only__graph_model.pt")
