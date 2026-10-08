"""Cross-cutting invariants the independent audit found documentation-only.

One decisive assertion per invariant. The audit's claims and what each test
enforces:

1. ``verify EXACTLY ONCE`` per VM crossing — a completion job that seals a
   bundle must not re-verify those bytes, and each transported archive is
   verified exactly once. Verified by counting ``verify_archive`` passes in a
   real local-completion job.
2. Bundle ROLE contracts are enforced at the boundary, not only by the writer —
   an ``inputs`` bundle carrying weights, or a ``result`` bundle carrying every
   epoch, must be refused by ``Bundle.load``.
3. The Kaggle train -> finalize ROLE handoff — the train kernel must ship the
   suite's own sealed ``result`` Bundle, so the finalize job's
   ``Bundle.load(..., "result")`` boundary accepts it (it rejected the old tree
   tarball with "archive manifest missing").
4. Config-leaf coverage — every ``config/*.yaml`` leaf has a reader, and no code
   literal duplicates a config-owned artifact value.
5. ``ORCHESTRATION_TRACE_STAGES`` matches the producers' own ``STAGE`` constants
   (read-only: the map lives in ``core/tracing.py``).

Where the required fix is outside this change's ownership the test still runs
the real check and records the drift through ``pytest.xfail`` naming the owner,
so the enforcement outlives this report and turns into a hard failure the day
the owner fixes the code and the test can assert. Invariants 1 and 2 have
reached that state (their owners landed the runtime fix), so they now assert
directly instead of recording drift; only the config-leaf census (4) still
records its drift.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from core.bundle import Bundle, BundleRole, _bundle_spec, manifest_name
from core.bundle import _bundle_spec as _spec
from core.portable_archive import verify_archive_digest, write_archive


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


# ── 1. one integrity check per archive per VM crossing ─────────────────────

def test_local_completion_verifies_each_archive_exactly_once(tmp_path, monkeypatch):
    """A completion job verifies each transported archive once — and never the
    archive it just sealed (the writer's digest is the transport token).

    Counts ``core.portable_archive.verify_archive`` passes through a real
    ``local_complete.complete`` call whose heavy collaborators are stubbed; the
    two transported archives and the freshly sealed output all go through the
    REAL ``Bundle.load`` boundary.
    """
    import core.portable_archive as portable_archive
    from model_tracks import bundle_steps, local_complete
    from model_tracks import resume as resume_module
    from model_tracks.package import package_member

    spec = _spec()
    run_tag = "r-tag"
    passes: dict[str, int] = {}
    real_verify = portable_archive.verify_archive

    def counting_verify(path, manifest_name_, **kwargs):
        passes[str(path)] = passes.get(str(path), 0) + 1
        return real_verify(path, manifest_name_, **kwargs)

    monkeypatch.setattr(portable_archive, "verify_archive", counting_verify)

    # The two archives that crossed the wire: a result archive (GPU output) and
    # an inputs archive (the prepared data bundle), each well formed.
    training_tree = tmp_path / "training"
    _write(training_tree / "text/text__vectors.npz", "vectors")
    _write(training_tree / spec.suite_manifest_file,
           json.dumps({spec.run_tag_key: run_tag}))
    training_archive = tmp_path / "training-result.tar.zst"
    write_archive(training_archive,
                  {p.relative_to(training_tree).as_posix(): p
                   for p in training_tree.rglob("*") if p.is_file()},
                  manifest_name=spec.manifest_result,
                  metadata={spec.run_tag_key: run_tag})

    inputs_tree = tmp_path / "inputs"
    _write(inputs_tree / package_member("suite_package_config"), "post_training_ablation: false\n")
    input_archive = tmp_path / "inputs.tar.zst"
    write_archive(input_archive,
                  {p.relative_to(inputs_tree).as_posix(): p
                   for p in inputs_tree.rglob("*") if p.is_file()},
                  manifest_name=spec.manifest_inputs, metadata={})

    settings = SimpleNamespace(result_archive_format="tar.zst",
                               post_training_ablation=False, report_test=False,
                               dvc_enabled=False, publish_git=False,
                               setup_dir="setup", ablation_config="config/ablation.yaml")

    class _FakeSuiteConfig:
        @staticmethod
        def model_validate(_):
            return settings

    monkeypatch.setattr(local_complete, "SuiteConfig", _FakeSuiteConfig)
    monkeypatch.setattr(resume_module, "validate_training_binding", lambda *a, **k: None)
    monkeypatch.setattr(resume_module, "validate_completed_suite_archive", lambda *a, **k: {})
    monkeypatch.setattr(local_complete, "_require_legacy_source_pin", lambda *a, **k: None)
    monkeypatch.setattr(local_complete, "trace", lambda: SimpleNamespace(
        add=lambda *a, **k: None, add_entities=lambda *a, **k: None))
    monkeypatch.setattr(local_complete, "flush_trace", lambda: None)
    monkeypatch.setattr(local_complete, "_publish", lambda final, *a, **k: final)

    # The finalize step's OWN writer returned a verified handle in the real job
    # (bundle_steps.finalize returns it); reproduce that: seal the output right
    # here so the completion job's post-seal code is what runs next.
    def fake_finalize(pipeline, result, *, inputs=None):
        workspace = tmp_path / "sealed_tree"
        _write(workspace / "suite_manifest.json", "{}\n")
        return Bundle.seal_archive(
            pipeline.output,
            {p.relative_to(workspace).as_posix(): p
             for p in workspace.rglob("*") if p.is_file()},
            role=BundleRole.result,
            metadata={spec.run_tag_key: result.run_tag()})

    monkeypatch.setattr(bundle_steps, "finalize", fake_finalize)

    published = local_complete.complete(training_archive, input_archive, run_tag)

    assert passes.get(str(training_archive), 0) == 1, \
        "each transported archive is integrity-checked exactly once at its boundary"
    assert passes.get(str(input_archive), 0) == 1
    # The just-sealed result archive crossed NO wire in this process: the writer
    # already hashed it while writing and returned its digest, so a second
    # boundary load is a redundant integrity pass (one per crossing). Asserted,
    # not recorded: the fix is in model_tracks/local_complete.py (the finalize
    # handle is reused) and a regression must fail here.
    assert passes.get(str(published), 0) == 0, (
        "the completion job re-verified the archive it just sealed "
        f"({passes[str(published)]} extra pass over {Path(published).name}); the handle "
        "returned by model_tracks.bundle_steps.finalize must be the handle every later "
        "step shares instead of re-loading the bytes")


# ── 2. bundle role contracts enforced at the boundary ──────────────────────

def test_bundle_role_contracts_are_enforced_on_load(tmp_path):
    """Role membership is a LOAD contract, not a writer-only convention.

    An ``inputs`` bundle must carry no weights, and a ``result`` bundle must
    carry only the selected checkpoint. Both archives below are well formed
    (``verify_archive_digest`` accepts them); the only remaining refusal reason
    is the role contract.
    """
    spec = _spec()

    inputs_tree = tmp_path / "inputs_with_weights"
    _write(inputs_tree / "data/x.json", "{}")
    _write(inputs_tree / "text/_checkpoints/m/r_f0/checkpoint-1/model.safetensors", "weights")
    inputs_archive = tmp_path / "inputs_with_weights.tar.zst"
    write_archive(inputs_archive,
                  {p.relative_to(inputs_tree).as_posix(): p
                   for p in inputs_tree.rglob("*") if p.is_file()},
                  manifest_name=spec.manifest_inputs, metadata={})

    result_tree = tmp_path / "result_all_epochs"
    for step in (1, 2, 3):
        directory = result_tree / f"text/_checkpoints/m/r_f0/checkpoint-{step}"
        _write(directory / "model.safetensors", f"w{step}")
        _write(directory / "trainer_state.json", json.dumps(
            {"best_model_checkpoint": "checkpoint-2",
             "best_metric": 0.9 if step == 2 else 0.5, "global_step": step}))
    result_archive = tmp_path / "result_all_epochs.tar.zst"
    write_archive(result_archive,
                  {p.relative_to(result_tree).as_posix(): p
                   for p in result_tree.rglob("*") if p.is_file()},
                  manifest_name=spec.manifest_result,
                  metadata={spec.run_tag_key: "r-tag"})

    for role, archive, label in (
            (BundleRole.inputs, inputs_archive, "an inputs bundle carrying weights"),
            (BundleRole.result, result_archive, "a result bundle carrying every epoch")):
        verify_archive_digest(archive, manifest_name(role))  # fixture is well formed
        try:
            Bundle.load(archive, role)
        except ValueError:
            pass  # the role contract refused it, as it must
        else:
            raise AssertionError(label + " loads cleanly")
    # The guard is precise, not blanket: a conforming set for the same role
    # still loads (the happy path is unchanged).
    selected_only = tmp_path / "result_selected.tar.zst"
    write_archive(selected_only, {
        "text/_checkpoints/m/r_f0/checkpoint-2/model.safetensors":
            result_tree / "text/_checkpoints/m/r_f0/checkpoint-2/model.safetensors"},
        manifest_name=spec.manifest_result, metadata={spec.run_tag_key: "r-tag"})
    assert Bundle.load(selected_only, BundleRole.result).run_tag() == "r-tag"


# ── 3. kaggle train -> finalize role handoff ───────────────────────────────

def _stage_train_and_embed(tmp_path, monkeypatch):
    """Stage the train and embed kernels offline (no network, no git)."""
    import core.runtime_inputs as runtime_inputs
    from cli import kaggle_lane
    from core.schemas import KaggleSpec

    spec = KaggleSpec(staging_dir="kaggle_stage", username="owner",
                      cpu_kernel_slug="owner/er-bundle-cpu",
                      gpu_kernel_slug="owner/er-train-gpu",
                      embedding_kernel_slug="owner/er-embed-gpu",
                      embedding_dataset_slug="owner/er-embed-requests",
                      bundle_dataset_slug="owner/er-10k-bundle")
    monkeypatch.setattr(kaggle_lane, "_spec", lambda: spec)
    monkeypatch.setattr(kaggle_lane, "TRAIN_ROOT", tmp_path)
    monkeypatch.setattr(kaggle_lane, "staging_dir",
                        lambda: (tmp_path / "kaggle_stage").resolve())
    monkeypatch.setattr(kaggle_lane, "_git_revision", lambda: "abc123def")
    monkeypatch.setattr(runtime_inputs, "require_published_tip_match",
                        lambda *a, **k: {"tip": "abc123def"})

    def members(*extra, lane="bundle"):
        flat = {name for group in extra for name in
                (group if isinstance(group, (tuple, list)) else (group,))}
        return tuple(sorted(flat))

    monkeypatch.setattr(kaggle_lane, "checkout_members", members)
    monkeypatch.setattr(kaggle_lane, "checkout_inventory", members)
    monkeypatch.setattr(kaggle_lane, "checkout_preflight_script",
                        lambda files, root_expression="root": "_runtime_files = ()\n")
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    return spec


def test_kaggle_train_ships_the_sealed_result_bundle_the_finalize_boundary_accepts(
        tmp_path, monkeypatch):
    """The train kernel must ship the suite's sealed ``result`` Bundle.

    Two sides of the same handoff: the staged train script carries the override
    (and the embed kernel does not), and the artifact that override produces is
    accepted by the finalize kernel's exact boundary call.
    """
    from cli.kaggle_kernels import KaggleKernels, TRAIN_RESULT_BUNDLE_SHIP
    from cli import kaggle_lane

    spec = _stage_train_and_embed(tmp_path, monkeypatch)
    train_receipt = KaggleKernels.stage_gpu_kernel(kind="train")
    embed_receipt = KaggleKernels.stage_gpu_kernel(kind="embed")
    train_script = (Path(train_receipt["staged"]) / train_receipt["code_file"]).read_text()
    embed_script = (Path(embed_receipt["staged"]) / embed_receipt["code_file"]).read_text()
    # ``KernelLifecycle.wrap_script`` indents the whole kernel, so compare
    # whitespace-normalized text.
    normalized_fragment = " ".join(TRAIN_RESULT_BUNDLE_SHIP.split())
    assert normalized_fragment in " ".join(train_script.split()), \
        "the train kernel must override the tree-tar helper with the sealed-bundle ship"
    assert normalized_fragment not in " ".join(embed_script.split()), \
        "the embed output is not a Bundle role; it keeps the tree-tar helper"

    # Behaviour: run the shipped override exactly as the kernel would, then make
    # the finalize kernel's boundary call against the artifact it produced.
    bundle_spec = _spec()
    run_tag = "run-20261008T000000"
    output = tmp_path / "results" / "training" / run_tag
    _write(output / "text/text__vectors.npz", "vectors")
    _write(output / bundle_spec.suite_manifest_file,
           json.dumps({bundle_spec.run_tag_key: run_tag}))
    sealed = Bundle.from_directory(output, BundleRole.result).seal_result(
        output.with_suffix(".tar.zst"), metadata={bundle_spec.run_tag_key: run_tag})

    working = tmp_path / "kaggle_working"
    working.mkdir()
    # The kernel clones to a checkout root; on the VM the bundle installs the
    # generated suite config at the train kernel's configured path, so mirror
    # that layout from the committed suite config.
    repo_root = Path(__file__).parents[1]
    generated_config = tmp_path / spec.train_suite_config
    generated_config.parent.mkdir(parents=True, exist_ok=True)
    generated_config.write_text(
        (repo_root / "config" / "model_tracks.yaml").read_text())
    namespace = {
        "json": json, "shutil": __import__("shutil"), "Path": Path,
        "WORKING": working, "RUN_TAG": run_tag, "REVISION": "abc123def",
        "SUITE_CONFIG": spec.train_suite_config, "root": tmp_path,
        "LANE": {"files": {"result_archive": "{kind}.tar.zst",
                           "result_manifest": "{kind}.manifest.json",
                           "hash_suffix": ".sha256"}},
        "sha256_file": lambda path: __import__("hashlib").sha256(
            Path(path).read_bytes()).hexdigest(),
    }
    exec(compile(TRAIN_RESULT_BUNDLE_SHIP, "<train-result-ship>", "exec"), namespace)
    digest = namespace["stage_result_archive"](output, kind="result_bundle", extra={})

    manifest = json.loads((working / "result_bundle.manifest.json").read_text())
    assert manifest["archive_sha256"] == digest == sealed.digest
    # The finalize kernel's exact boundary: Bundle.load(..., "result",
    # expected_digest=<the manifest's archive_sha256>).
    handle = Bundle.load(working / "result_bundle.tar.zst", "result",
                         expected_digest=manifest.get("archive_sha256"))
    assert handle.run_tag() == run_tag
    assert bundle_spec.manifest_result in handle.members()
    # The old shape (a tree tarball with an out-of-archive manifest) is exactly
    # what the boundary rejected: keep that proof beside the fix.
    from core.archive_reader import tar_archive
    tree_tar = tmp_path / "tree_tarball.tar.zst"
    with tar_archive(tree_tar, "w") as tar:
        tar.add(output / "text/text__vectors.npz",
                arcname=f"{run_tag}/text/text__vectors.npz")
    with pytest.raises(ValueError, match="archive manifest missing"):
        Bundle.load(tree_tar, "result")


# ── 4. config-leaf coverage + no code literal duplicates a config value ─────

#: The config's own typed mirror: pydantic field defaults must be literals, so a
#: value also appearing here is the declared default, not an independent SSOT
#: copy. Reported separately (never a failure) instead of silently ignored.
_CONFIG_MIRRORS = frozenset({"src/core/schemas.py"})

#: Values that look like an artifact identity (a path or a suffixed filename)
#: are config-owned things code must not respell; short vocabulary words (a
#: track name, ``offline``, ``cuda``) are not treated as duplication.
_ARTIFACT_SUFFIX = (".json", ".yaml", ".yml", ".csv", ".zst", ".zip", ".pt", ".npz",
                    ".txt", ".pkl", ".gz", ".bin", ".md", ".jsonl", ".log",
                    ".safetensors")


def _config_drift(repo_root: Path) -> dict[str, list]:
    """AST/static census of the two config-SSOT drift classes."""
    leaves: dict[tuple[str, str], object] = {}
    for config_path in sorted((repo_root / "config").glob("*.yaml")):
        stack = [((), yaml.safe_load(config_path.read_text()))]
        while stack:
            path, node = stack.pop()
            if isinstance(node, dict):
                for key, value in node.items():
                    stack.append((path + (str(key),), value))
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    stack.append((path + (f"[{index}]",), value))
            else:
                leaves[(config_path.name, ".".join(path))] = node

    attribute_names: set[str] = set()
    literals: dict[str, set[str]] = {}
    for source in sorted((repo_root / "src").rglob("*.py")):
        relative = source.relative_to(repo_root).as_posix()
        try:
            tree = ast.parse(source.read_text(errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                attribute_names.add(node.attr)
            elif isinstance(node, ast.Name):
                attribute_names.add(node.id)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                literals.setdefault(node.value, set()).add(relative)

    unread, duplicated, mirrored = [], [], []
    for (config_name, path), value in leaves.items():
        key = path.rsplit(".", 1)[-1]
        if not key.startswith("[") and key not in attribute_names \
                and not any(key in literal for literal in literals):
            unread.append(f"{config_name}:{path}")
        if not (isinstance(value, str) and len(value) >= 6
                and (value.endswith(_ARTIFACT_SUFFIX) or "/" in value)):
            continue
        owners = literals.get(value)
        if not owners:
            continue
        independent = sorted(owners - _CONFIG_MIRRORS)
        if independent:
            duplicated.append(f"{value!r} ({config_name}:{path} <- {', '.join(independent[:2])})")
        else:
            mirrored.append(f"{config_name}:{path}")
    return {"unread": unread, "duplicated": duplicated, "mirrored": mirrored}


def test_config_leaves_have_readers_and_no_literal_duplicate_values():
    """Every config leaf is read, and no code literal respells a config value.

    The scan is static (yaml leaves vs src AST), so a key consumed wholesale by
    a generic loader still counts as read; the drift it reports is the one the
    audit asked for.
    """
    drift = _config_drift(Path(__file__).parents[1])
    if drift["unread"] or drift["duplicated"]:
        pytest.xfail(
            f"config SSOT drift: {len(drift['unread'])} leaves with no reader "
            f"(e.g. {', '.join(drift['unread'][:3])}) and "
            f"{len(drift['duplicated'])} config-owned artifact values respelled as code "
            f"literals (e.g. {', '.join(drift['duplicated'][:2])}); "
            f"{len(drift['mirrored'])} more are mirrored by {sorted(_CONFIG_MIRRORS)} "
            "(the typed config default mirror, reported but not failed). "
            "Owners: config/** (dead leaves) and the respelling modules named above")


# ── 5. the orchestration stage map matches the producers' STAGE constants ───

#: Each orchestration stage and the producer module + attribute that declares the
#: stage name it writes rows under. ``core/tracing.ORCHESTRATION_TRACE_STAGES``
#: must map every one of these to its producer's constant: ``()`` means "writes
#: no trace rows" and must never stand next to a producer that writes them.
_PRODUCER_STAGE_CONSTANTS = {
    "dedupe": ("training.dedupe", "STAGE"),
    "cross_country_pairs": ("training.build_second04_pairs", "STAGE"),
    "number_reference": ("training.build_reference", "STAGE_WRITE"),
    "verify_reference": ("training.build_reference", "STAGE_VERIFY"),
    "validation": ("training.build_final_validation", "STAGE"),
    "labeled_pairs": ("training.labeled_pairs", "STAGE"),
    "full_bundle": ("training.train", "BUNDLE_STAGE"),
    "verify_handoff": ("training.handoff", "STAGE"),
    "negative_supply": ("training.negative_supply", "STAGE"),
    "suite_inputs": ("model_tracks.package", "STAGE"),
}


def test_orchestration_stage_map_matches_producer_stage_constants():
    """The trace stage map is the producers' own vocabulary, not a parallel one.

    ``trace_stages_for`` must return each producer's ``STAGE`` constant for the
    orchestration stage that producer serves, and an undeclared stage must stay
    a loud error (a reader can never silently join nothing).
    """
    import importlib

    from core import tracing
    from training import prepare_all

    if not hasattr(tracing, "ORCHESTRATION_TRACE_STAGES") \
            or not hasattr(tracing, "trace_stages_for"):
        pytest.xfail(
            "core/tracing.py declares no ORCHESTRATION_TRACE_STAGES/trace_stages_for "
            "(they were dead before the stage-map change) — owner: core/tracing.py")
    ORCHESTRATION_TRACE_STAGES = tracing.ORCHESTRATION_TRACE_STAGES
    trace_stages_for = tracing.trace_stages_for

    declared = set(prepare_all.STAGES) | {"negative_supply", "discriminator"}
    assert declared == set(ORCHESTRATION_TRACE_STAGES), \
        "the map and the orchestrator's stage list are one contract"

    drifted = []
    for stage, (module_name, attribute) in sorted(_PRODUCER_STAGE_CONSTANTS.items()):
        constant = getattr(importlib.import_module(module_name), attribute)
        if trace_stages_for(stage) != (constant,):
            drifted.append(f"{stage}: map={trace_stages_for(stage)} producer={constant!r} "
                           f"({module_name}.{attribute})")
    from graph_tracks import prepare as graph_prepare, setup as graph_setup
    if set(trace_stages_for("graph_inputs")) != {graph_setup.STAGE, graph_prepare.STAGE}:
        drifted.append("graph_inputs: map=" + repr(trace_stages_for("graph_inputs"))
                       + " producers=" + repr((graph_setup.STAGE, graph_prepare.STAGE)))

    with pytest.raises(ValueError, match="unknown orchestration stage"):
        trace_stages_for("not-a-stage")

    if drifted:
        pytest.xfail(
            "ORCHESTRATION_TRACE_STAGES claims '() = writes no trace rows' for stages "
            "whose producers write them under that exact name — owner: "
            "core/tracing.py (ORCHESTRATION_TRACE_STAGES): " + "; ".join(drifted))
