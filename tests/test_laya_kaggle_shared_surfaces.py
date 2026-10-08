"""The laya lane's kaggle-facing stages, consolidated onto the kaggle owners.

The laya lane used to re-implement the kaggle lane's kernel metadata and argv
shapes. Where behavior is identical it now calls the owner:

* ``KaggleKernels.kernel_metadata`` — the four laya stage metadata documents
  (decision, holdout-eval, finetune, finetune-eval) are built by the owner;
* ``KaggleKernels.kernels_push_argv`` / ``kernels_output_argv`` — the push and
  output argv (addressed as ``python -m kaggle``);
* ``KaggleDatasets.dataset_publish_commands`` — the create/version argv;
* ``KernelTemplates.substitute`` — the ``@TOKEN@`` loop the scripts render with;
* ``KaggleKernels.push_with_session_capture`` — the clear/push/capture launch aid.

The lane deliberately diverges in exactly three places, and each is pinned here
(never silently unified): the dataset attach passes ``-r zip`` (its own
``dir_mode_args``), the CLI is addressed as ``[sys.executable, "-m", "kaggle"]``,
and the fetched result's integrity contract is the in-archive receipt (a plain
tar.gz, not a Bundle role) in ``collect_kaggle_result``.

The emitted bytes this file protects:

* the ``kernels push`` / ``kernels output`` argv tokens,
* the ``kernel-metadata.json`` document (and that the laya stage emits it
  through the owner, not a re-spelled copy),
* the ``datasets create`` / ``datasets version`` argv,
* the ``@TOKEN@`` substitution loop the staged kernel scripts are rendered with.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from cli.kaggle_datasets import KaggleDatasets
from cli.kaggle_kernel_templates import KernelTemplates
from cli.kaggle_kernels import KaggleKernels
from cli import laya_lane


def _legacy_substitute(script, values):
    """The pre-consolidation loop, kept here as the byte reference."""
    for token, replacement in values.items():
        script = script.replace(f"@{token}@", replacement)
    return script


def test_shared_substitution_reproduces_the_legacy_loop_bytes():
    template = "a=@A@ b=@B@ c=@C@ @A@\n"
    values = {"A": "1", "B": "@C@", "C": "x"}  # B's value is inserted verbatim
    assert KernelTemplates.substitute(template, values) == \
        _legacy_substitute(template, values)


def test_laya_templates_render_identically_through_the_shared_loop():
    """The real laya templates render byte-for-byte through the shared loop.

    The staged kernel scripts are these constants plus the token substitution;
    with the real value mappings the lane's substitution and the shared one
    agree exactly, so consolidating the lane onto the shared loop cannot move
    the files the lane stages (two passed bakes included).
    """
    spec = laya_lane._spec()
    from core.hosted_dataset import hosted_registry

    # the ONE thing the lane's renderer adds to the shared loop: the session
    # mount root, declared in the hosted registry (`config/hosted_datasets.yaml`)
    mount_root = str(hosted_registry().mount_root)
    values = {
        "LAYA_PACKAGE": spec.laya_package,
        "CHECKPOINT_HUB": spec.checkpoint_hub,
        "DECISION_KIND": "attribute",
        "RUN_TAG": "laya_t",
        "DECISION_CSV": "dataset.csv",
        "STATE_COLUMN": "both_columns",
        "BATCH_SIZE": str(spec.laya_decision_batch_size),
        "MIN_CONFIDENCE": repr(spec.min_router_confidence),
        "QUESTION_SCHEMA_FILE": repr("laya.question.json"),
        "REPOSITORY": "https://example/repo.git",
        "BRANCH": "main",
        "REVISION": "0" * 40,
        # the finetune kernel's session self-report helper (the ONE kernel-side
        # home, shared with the kaggle lane's kernel templates)
        "SESSION_REPORT": laya_lane._session_report_helper(),
    }
    templates = (laya_lane.LAYA_RUNTIME_PREFLIGHT,
                 laya_lane.DECISION_KERNEL_SCRIPT,
                 laya_lane.EVAL_KERNEL_SCRIPT,
                 laya_lane.FINETUNE_RUNTIME_PREFLIGHT,
                 laya_lane.FINETUNE_KERNEL_SCRIPT,
                 laya_lane.FINETUNE_EVAL_RUNTIME_PREFLIGHT,
                 laya_lane.FINETUNE_EVAL_KERNEL_SCRIPT,
                 laya_lane.NOTEBOOK_SCRIPT)
    for template in templates:
        assert laya_lane._template(template, values) == \
            KernelTemplates.substitute(template, {"MOUNT_ROOT": mount_root,
                                                  **values})
        # no mount-root token ever survives into a staged payload
        assert "@MOUNT_ROOT@" not in laya_lane._template(template, values)
    # a template carrying no token is not rewritten at all
    assert laya_lane._template(laya_lane.NOTEBOOK_SCRIPT, {}) == \
        laya_lane.NOTEBOOK_SCRIPT
    # the two-pass bake (preflight rendered FIRST, then dropped into the script)
    for preflight_name, script_name in (
            ("LAYA_RUNTIME_PREFLIGHT", "DECISION_KERNEL_SCRIPT"),
            ("FINETUNE_RUNTIME_PREFLIGHT", "FINETUNE_KERNEL_SCRIPT"),
            ("FINETUNE_EVAL_RUNTIME_PREFLIGHT", "FINETUNE_EVAL_KERNEL_SCRIPT")):
        preflight = getattr(laya_lane, preflight_name)
        script = getattr(laya_lane, script_name)
        merged = {**values,
                  "RUNTIME_PREFLIGHT": laya_lane._template(preflight, values)}
        assert laya_lane._template(script, merged) == \
            KernelTemplates.substitute(script, {"MOUNT_ROOT": mount_root,
                                                **merged})


def test_render_runtime_reproduces_the_legacy_chain_bytes():
    """The kaggle lane's rendered kernel scripts must not move."""
    from cli import kaggle_lane as lane

    spec = lane._spec()
    script = "@COMMAND_RUNNER@\n@SIZE_HELPER@\nLANE = json.loads(@LANE_JSON@)\n"
    from core.common import data_cfg, training_cfg

    payload = spec.model_dump(mode="json")
    payload["paths"] = data_cfg().paths.model_dump(mode="json")
    payload["archives"] = training_cfg().archives.model_dump(mode="json")
    legacy = (script.replace("@COMMAND_RUNNER@", KernelTemplates.REMOTE_COMMAND_RUNNER)
              .replace("@SIZE_HELPER@", KernelTemplates.SIZE_HELPER)
              .replace("@LANE_JSON@", repr(json.dumps(payload))))
    assert KernelTemplates.render_runtime(script, spec) == legacy


def test_kernels_argv_tokens_are_byte_identical_for_both_lanes():
    prefix = [sys.executable, "-m", "kaggle"]
    assert KaggleKernels.kernels_push_argv(prefix, "stage") == \
        [sys.executable, "-m", "kaggle", "kernels", "push", "-p", "stage"]
    assert KaggleKernels.kernels_output_argv(prefix, "owner/kernel", "stage") == \
        [sys.executable, "-m", "kaggle", "kernels", "output", "owner/kernel",
         "-p", "stage"]
    assert KaggleKernels.kernels_push_argv(["/usr/bin/kaggle"], "s") == \
        ["/usr/bin/kaggle", "kernels", "push", "-p", "s"]


def test_laya_push_dry_run_emits_the_same_argv():
    """The pusher's documented argv does not move (no config, no push)."""
    plan = laya_lane.push_kaggle_kernel("/some/stage", execute=False)
    assert plan["argv"] == [sys.executable, "-m", "kaggle", "kernels", "push",
                            "-p", "/some/stage"]
    assert plan["mode"] == "dry-run"


def test_laya_push_routes_through_the_shared_launch_aid(tmp_path, monkeypatch):
    """Executed laya push: the shared argv helper AND the shared
    clear/push/capture launch-aid sequence (never re-spelled in this lane)."""
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "kernel-metadata.json").write_text(
        json.dumps({"id": "owner/laya"}), encoding="utf-8")
    monkeypatch.setattr(laya_lane, "TRAIN_ROOT", tmp_path)
    monkeypatch.setattr(laya_lane, "_staged_laya_push_preflight",
                        lambda path: None)
    seen: dict = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        # The push contract is fail-loud: a zero exit code alone is not enough,
        # the CLI's success line must be present (no "error" text).
        return subprocess.CompletedProcess(argv, 0, stdout="Kernel successfully pushed", stderr="")

    monkeypatch.setattr(laya_lane.subprocess, "run", fake_run)
    calls: list = []

    def fake_session_capture(slug, push):
        calls.append(("capture", slug))
        push()
        calls.append(("captured", slug))
        return {"session_id": "sess-9"}

    monkeypatch.setattr(KaggleKernels, "push_with_session_capture",
                        staticmethod(fake_session_capture))
    plan = laya_lane.push_kaggle_kernel(stage, execute=True)
    assert plan["returncode"] == 0 and plan["pushed"] is True
    assert plan["session_id"] == "sess-9"
    assert seen["argv"] == [sys.executable, "-m", "kaggle", "kernels", "push",
                            "-p", str(stage)]
    assert calls == [("capture", "owner/laya"), ("captured", "owner/laya")]


def test_dataset_publish_argv_tokens_are_byte_identical_for_both_lanes():
    bundle = KaggleDatasets.dataset_publish_commands(
        "/usr/bin/kaggle", "payload", message="er bundle: cohort=c",
        dir_mode_args=["-r", "--dir-mode", "skip"])
    assert bundle["version"] == ["/usr/bin/kaggle", "datasets", "version",
                                 "-r", "--dir-mode", "skip", "-m",
                                 "er bundle: cohort=c", "-p", "payload"]
    laya = KaggleDatasets.dataset_publish_commands(
        "/usr/bin/kaggle", "payload", message="laya inputs laya_1",
        dir_mode_args=["-r", "zip"])
    assert laya["version"] == ["/usr/bin/kaggle", "datasets", "version",
                               "-r", "zip", "-m", "laya inputs laya_1",
                               "-p", "payload"]
    assert laya["create"] == ["/usr/bin/kaggle", "datasets", "create",
                              "-p", "payload"]


def test_kernel_metadata_reproduces_the_legacy_document_bytes():
    """The 11-key kernel-metadata.json shape (JSON key order included)."""
    legacy_gpu = {
        "id": "owner/an-er-kernel",
        "title": "An Er Kernel",
        "code_file": "laya_decision.py",
        "language": "python",
        "kernel_type": "script",
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": ["fbarulli/er-laya-payload"],
        "kernel_sources": [],
        "competition_sources": [],
        "is_private": True,
    }
    built = KaggleKernels.kernel_metadata(
        "owner/an-er-kernel", "laya_decision.py", enable_gpu=True,
        dataset_sources=["fbarulli/er-laya-payload"])
    assert json.dumps(built, indent=2) == json.dumps(legacy_gpu, indent=2)
    assert list(built) == list(legacy_gpu)

    legacy_cpu = {
        "id": "owner/an-er-kernel",
        "title": "An Er Kernel",
        "code_file": "er_stop.py",
        "language": "python",
        "kernel_type": "script",
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "is_private": True,
    }
    cpu = KaggleKernels.kernel_metadata("owner/an-er-kernel", "er_stop.py",
                                        enable_gpu=False)
    assert json.dumps(cpu, indent=2) == json.dumps(legacy_cpu, indent=2)


def test_laya_stage_emits_the_owners_metadata_document(tmp_path, monkeypatch):
    """The four laya stage metadata dicts are gone: staging a decision kernel
    must build ``kernel-metadata.json`` through ``KaggleKernels.kernel_metadata``
    (the owner), never a re-spelled 11-key copy. A recorder wraps the owner and
    the written document must equal the owner's output for the same inputs."""
    from pathlib import Path

    from core.schemas import LayaSpec

    monkeypatch.setattr(laya_lane, "_spec", lambda: LayaSpec())
    monkeypatch.setattr(laya_lane, "TRAIN_ROOT", tmp_path)
    from core.common import training_cfg as _tcfg

    base = _tcfg()
    forced = base.model_copy(update={
        "kaggle": base.kaggle.model_copy(update={"branch": "main"})})
    monkeypatch.setattr(laya_lane, "training_cfg", lambda: forced)

    schema = tmp_path / "config/laya.question.json"
    schema.parent.mkdir(parents=True, exist_ok=True)
    schema.write_text(json.dumps({"questions": {
        "attribute_alignment": {"type": "choice", "instructions": "verdict?",
                                "criteria": {"aligned": None}}}}))
    import core.common as core_common

    dataset = tmp_path / "dataset.csv"
    dataset.write_text("sku_id,sku_name_eng,attribute\n"
                       "SKU0,Name 0 500 ml,Volume: 500; Brand: A\n")
    monkeypatch.setitem(core_common.F, "dataset", dataset)
    monkeypatch.setitem(core_common.F, "final_validation", dataset)

    monkeypatch.setattr(laya_lane, "_git_revision", lambda: "0" * 40)
    import subprocess as sp

    def fake_run(args, **kwargs):
        if "rev-parse" in args:
            return sp.CompletedProcess(args, 0, stdout="0" * 40, stderr="")
        return sp.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(laya_lane.subprocess, "run", fake_run)

    calls: list = []
    real = KaggleKernels.kernel_metadata

    def recorder(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(KaggleKernels, "kernel_metadata", staticmethod(recorder))

    receipt = laya_lane.stage_decision_kernel(decision_kind="attribute")
    stage = Path(receipt["staged"])
    written = json.loads((stage / "kernel-metadata.json").read_text())
    assert calls, "staging must build the metadata through the owner"
    args, kwargs = calls[0]
    assert written == real(*args, **kwargs)
    assert list(written) == list(real(*args, **kwargs))
    assert kwargs["dataset_sources"] == ["fbarulli/er-laya-requests"]
    assert written["code_file"] == "laya_decision.py"


def test_push_with_session_capture_clears_then_captures(monkeypatch):
    """ONE home for the launch-aid sequence around a push."""
    from cli import kaggle_lane as lane

    events: list[tuple] = []
    monkeypatch.setattr(lane, "clear_kernel_session_id", lambda slug: events.append(("clear", slug)))
    monkeypatch.setattr(lane, "capture_kernel_session_id",
                        lambda slug: (events.append(("capture", slug)),
                                      {"session_id": "sess-1"})[1])

    def push():
        events.append(("push", None))

    captured = KaggleKernels.push_with_session_capture("owner/kernel", push)
    assert [name for name, _ in events] == ["clear", "push", "capture"]
    assert captured == {"session_id": "sess-1"}


def test_push_with_session_capture_survives_a_failed_capture(monkeypatch):
    from cli import kaggle_lane as lane

    monkeypatch.setattr(lane, "clear_kernel_session_id", lambda slug: None)
    logged: list[str] = []
    monkeypatch.setattr(lane, "_log_lane", logged.append)

    def boom(slug):
        raise RuntimeError("no session yet")

    monkeypatch.setattr(lane, "capture_kernel_session_id", boom)
    captured = KaggleKernels.push_with_session_capture("owner/kernel", lambda: None)
    assert captured == {"session_id": None}
    assert any("session-id capture skipped" in line for line in logged)


def test_push_with_session_capture_never_captures_after_a_failed_push(monkeypatch):
    from cli import kaggle_lane as lane

    captured: list[str] = []
    monkeypatch.setattr(lane, "clear_kernel_session_id", lambda slug: None)
    monkeypatch.setattr(lane, "capture_kernel_session_id",
                        lambda slug: captured.append(slug))

    def fail():
        raise subprocess.CalledProcessError(1, ["kaggle"])

    with pytest.raises(subprocess.CalledProcessError):
        KaggleKernels.push_with_session_capture("owner/kernel", fail)
    assert captured == []
