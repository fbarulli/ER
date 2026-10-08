"""ColabSpec owns the Colab lane contract, and there are ZERO freshness checks.

Owner directive 2026-10-08 (strengthened). Everything the Colab lanes need is
declared once in ``core.schemas.ColabSpec`` (backed by ``config/training.yaml``
``colab:``) and read through its accessors:

* the sparse-checkout path set (dirs with a trailing slash + root files),
* the ONE data bundle feeding BOTH lanes (``--gpu CPU`` and ``--gpu T4``),
* per-lane transcripts + dirs and per-lane sessions, plus the launcher-lock key,
* the W&B wiring (project/mode projected from ``tracking.wandb`` + env names),
* the ``freshness_checks`` ruling, which can only be False.

The tests fall into three groups: the class resolves, the consumers read the
class instead of re-spelling a literal, and no freshness/staleness comparison
can reappear anywhere in ``src/``.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from core.common import TRAIN_ROOT, training_cfg
from core.schemas import ColabSpec, canonical_suite_matrix

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def spec() -> ColabSpec:
    return training_cfg().colab


def _python_sources() -> list[Path]:
    return sorted(path for path in SRC.rglob("*.py") if "__pycache__" not in path.parts)


# ── the sparse-checkout contract has ONE home ───────────────────────────────


def test_sparse_checkout_comes_from_the_class():
    from cli import colab_runtime

    declared = spec()
    assert colab_runtime.RUNTIME_DIRECTORY_PATHS == declared.checkout_directory_paths()
    assert colab_runtime.RUNTIME_REQUIRED_ROOT_FILES == declared.checkout_root_files()
    assert colab_runtime.runtime_checkout_paths() == declared.checkout_patterns()
    # A directory entry carries a trailing slash; a root file does not.
    assert all(name.endswith("/") for name in declared.checkout_directory_paths())
    assert all(not name.endswith("/") for name in declared.checkout_root_files())


def test_the_checkout_set_is_the_config_declaration():
    declared = spec()
    assert set(declared.checkout_paths) == set(training_cfg().colab.checkout_paths)
    # The Colab set is deliberately not the Kaggle lane's.
    assert {name.rstrip("/") for name in declared.checkout_paths} != \
        set(training_cfg().kaggle.checkout_paths)


def test_the_checkout_lists_have_one_home():
    """The declared list is never re-declared per surface.

    Individual generic entries (``"scripts/"``, ``"pyproject.toml"``) legitimately
    appear in unrelated modules, so the pin is structural: the two projected
    tuple names exist in exactly ONE module, and that module's values are the
    class's own accessors (pinned above). Anything that re-lists the checkout set
    has to introduce a second declaration name, which this fails on.
    """
    declaring = {
        name: sorted(path.relative_to(ROOT).as_posix()
                     for path in _python_sources()
                     if name in path.read_text(encoding="utf-8"))
        for name in ("RUNTIME_DIRECTORY_PATHS =", "RUNTIME_REQUIRED_ROOT_FILES =")
    }
    assert declaring == {
        "RUNTIME_DIRECTORY_PATHS =": ["src/cli/colab_runtime.py"],
        "RUNTIME_REQUIRED_ROOT_FILES =": ["src/cli/colab_runtime.py"],
    }


# ── the data bundle feeds BOTH lanes ────────────────────────────────────────


def test_one_data_bundle_serves_the_cpu_and_the_gpu_lane():
    bundle = spec().data_bundle
    cpu, gpu = spec().lane_for("CPU"), spec().lane_for("T4")
    assert cpu is spec().lanes["cpu"] and gpu is spec().lanes["gpu"]
    # Both lanes train from the same prepared setup and the same suite config.
    assert bundle.setup_dir == "data/prepared/smoke_200"
    assert bundle.suite_config.startswith(bundle.setup_dir + "/")
    # The smoke suite the shell entrypoint launches IS the declared bundle.
    assert canonical_suite_matrix().entry("smoke_200").suite_config == bundle.suite_config
    assert training_cfg().preparation.smoke_dir == bundle.setup_dir


def test_data_bundle_archive_names_are_derived_not_literal():
    bundle = spec().data_bundle
    assert bundle.archive_name == training_cfg().kaggle.files.bundle_archive
    assert bundle.run_archive_name("R1", "tar.zst") == "R1__inputs.tar.zst"
    assert bundle.run_archive_glob() == "*__inputs.*"
    assert bundle.recovery_archive_name("R1") == "R1.recovery.tar.zst"
    assert bundle.remote_archive_name("R1", "tar.zst") == "R1__all_tracks.tar.zst"
    assert bundle.remote_recovery_name("R1") == "R1__recovery.tar.zst"
    assert bundle.git_transport_name("abc") == "abc.tar.zst"


def test_transport_member_names_are_not_respelled_in_python():
    bundle = spec().data_bundle
    for member in (bundle.transport_member, bundle.recovery_member):
        offenders = [path.relative_to(ROOT).as_posix()
                     for path in _python_sources()
                     if f"'{member}'" in path.read_text(encoding="utf-8")]
        assert offenders == [], f"{member} spelled in code: {offenders}"


def test_the_smoke_entrypoint_asks_the_class_for_its_lane():
    script = (ROOT / "scripts" / "run_colab_smoke.sh").read_text(encoding="utf-8")
    assert "--print-lane-env" in script
    assert "EUROMONITOR_LANE_SUITE_CONFIG" in script
    # Neither the session pair nor the suite config is hardcoded in the shell.
    for literal in ("smoke-cpu", "smoke-gpu", "lane_cpu.log", "lane_gpu.log",
                    "data/prepared/smoke_200/suite.yaml"):
        assert literal not in script, literal


# ── logs, sessions, wandb, lock ─────────────────────────────────────────────


def test_per_lane_sessions_and_transcripts_are_declared():
    declared = spec()
    assert declared.lanes["cpu"].session == "smoke-cpu"
    assert declared.lanes["gpu"].session == "smoke-gpu"
    assert declared.lanes["cpu"].log_name == "lane_cpu.log"
    assert declared.lanes["gpu"].log_name == "lane_gpu.log"
    assert declared.transcript_name("smoke-cpu") == "lane_cpu.log"
    assert declared.transcript_name("smoke-gpu") == "lane_gpu.log"
    # The configured session keeps the default transcript; any other isolated
    # session gets its own file so concurrent lanes never truncate one.
    assert declared.transcript_name(declared.session) == declared.default_log_name
    assert declared.transcript_name("er-prep-50pct") == "lane_er-prep-50pct.log"


def test_lane_identity_is_not_respelled_in_python():
    declared = spec()
    literals = {declared.lanes["cpu"].session, declared.lanes["gpu"].session,
                declared.lanes["cpu"].log_name, declared.lanes["gpu"].log_name}
    offenders = {
        path.relative_to(ROOT).as_posix(): sorted(
            literal for literal in literals
            if f'"{literal}"' in path.read_text(encoding="utf-8")
            or f"'{literal}'" in path.read_text(encoding="utf-8"))
        for path in _python_sources()
    }
    assert {name: found for name, found in offenders.items() if found} == {}


def test_launcher_log_name_follows_the_declared_lane():
    import os
    import subprocess
    import sys

    declared = spec()
    for session, expected in ((declared.lanes["cpu"].session, "lane_cpu.log"),
                              (declared.lanes["gpu"].session, "lane_gpu.log"),
                              (declared.session, declared.default_log_name)):
        completed = subprocess.run(
            [sys.executable, "-c",
             "import cli.colab as c; print(c.LANE_LOG_NAME)"],
            cwd=ROOT, capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": str(SRC),
                 "EUROMONITOR_COLAB_SESSION": session,
                 "EUROMONITOR_LANE_LOG": ""},
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip() == expected, (session, completed.stdout)


def test_the_log_roof_is_the_declared_directory():
    declared = spec()
    import cli.colab as colab

    assert colab.lane_transcript_path() == (
        TRAIN_ROOT / declared.log_dir / declared.default_log_name).resolve()
    # The root/training per-session transcripts live under the same declared dir.
    from cli import log_capture
    assert log_capture.lane_log_at(declared.log_dir, "x.log").parent == \
        (TRAIN_ROOT / declared.log_dir).resolve()


def test_paths_yaml_legacy_bindings_agree_with_the_class():
    """The declared default transcript IS the legacy ``colab_live_log`` binding."""
    from core.common import F

    declared = spec()
    expected = (TRAIN_ROOT / declared.log_dir / declared.default_log_name).resolve()
    assert Path(F["colab_live_log"]).resolve() == expected
    assert Path(F["colab_training_log"]).resolve() == expected


def test_launcher_lock_name_comes_from_the_class():
    from cli import colab as colab_module
    from cli import colab_launch

    declared = spec()
    lock = colab_launch._colab_launch_lock_path()
    assert lock.name == declared.launcher_lock_name(colab_module.SESSION)
    assert lock.name.startswith("launcher-") and lock.name.endswith(".lock")


def test_wandb_settings_are_projected_from_the_tracking_ssot():
    declared = spec()
    tracking = training_cfg().tracking.wandb
    assert declared.wandb.project == tracking.project
    assert declared.wandb.mode == tracking.mode
    assert declared.wandb.api_key_env == "WANDB_API_KEY"
    # A standalone spec keeps its own copy; the projection happens at load.
    assert ColabSpec.model_validate(
        {**declared.model_dump(), "wandb": {**declared.wandb.model_dump(),
                                            "project": "other", "mode": "disabled"}}
    ).wandb.project == "other"


# ── ZERO freshness checks ───────────────────────────────────────────────────


def test_the_class_states_there_are_no_freshness_checks():
    declared = spec()
    assert declared.freshness_checks is False
    assert declared.verifies_freshness() is False


def test_the_class_refuses_a_freshness_check():
    """``freshness_checks`` is ``Literal[False]``: no config can turn it on."""
    with pytest.raises(ValidationError):
        ColabSpec.model_validate({**spec().model_dump(), "freshness_checks": True})


def test_the_config_declares_the_rule():
    text = (ROOT / "config" / "training.yaml").read_text(encoding="utf-8")
    assert re.search(r"^\s*freshness_checks:\s*false\s*$", text, flags=re.MULTILINE)


#: Patterns that ARE a freshness/staleness comparison. Each is checked to have
#: zero occurrences in src/: a re-derived request hash compared against a
#: recorded one, the not-yet-produced input allowance, and the retired
#: freshness-gate remnants.
_FORBIDDEN_FRESHNESS_PATTERNS = (
    "request_sha256=",
    "allow_gpu_pending",
    "GPU_PENDING",
    "ER_DATA_GATE_GPU_PENDING",
    "ER_SKIP_CONFIG_VERIFY",
    "SuiteFreshnessManifest",
    "freshness.json",
)


def test_no_freshness_pattern_can_reappear_in_src():
    offenders: dict[str, list[str]] = {}
    for path in _python_sources():
        text = path.read_text(encoding="utf-8")
        found = [pattern for pattern in _FORBIDDEN_FRESHNESS_PATTERNS if pattern in text]
        if found:
            offenders[path.relative_to(ROOT).as_posix()] = found
    assert offenders == {}, f"freshness patterns reappeared: {offenders}"


def test_no_recorded_hash_is_compared_against_a_recomputed_one():
    """The other half of the rule: a recorded digest is never the fresh test."""
    comparisons = re.compile(
        r"request_sha256['\"]?\]?\s*[!=]=|['\"]request_sha256['\"]\s*\)?\s*[!=]=")
    offenders = [path.relative_to(ROOT).as_posix()
                 for path in _python_sources()
                 if comparisons.search(path.read_text(encoding="utf-8"))]
    assert offenders == [], f"recorded-hash freshness comparisons: {offenders}"


def test_result_contracts_have_no_freshness_parameter():
    """The consumers' public contracts cannot ask for a staleness verdict."""
    import inspect

    from model_tracks import ablation, post_training_ablation
    from training import prepare_embeddings

    for function in (prepare_embeddings.validate_result,
                     prepare_embeddings.validate_prepared_provenance):
        assert "request_sha256" not in inspect.signature(function).parameters
    assert "request_sha256" not in ablation.validate_vectors.__code__.co_names
    assert "request_sha256" not in post_training_ablation.SavedAblationReport.model_fields
