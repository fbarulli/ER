"""The laya lane's kaggle-facing stages, consolidated onto the kaggle owners.

The laya lane used to re-implement five functions the kaggle lane already owns.
Where behavior is identical the laya lane now calls the owner; where the lane
deliberately differs (it attaches DATASETS instead of cloning the repo, and it
addresses the CLI as ``python -m kaggle``), the SHARED part is pinned byte-for-
byte here.

The emitted bytes this file protects:

* the ``kernels push`` / ``kernels output`` argv tokens,
* the ``kernel-metadata.json`` document,
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
            KernelTemplates.substitute(template, values)
        assert laya_lane._template(template, {}) == template
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
            KernelTemplates.substitute(script, merged)


def test_render_runtime_reproduces_the_legacy_chain_bytes():
    """The kaggle lane's rendered kernel scripts must not move."""
    from cli import kaggle_lane as lane

    spec = lane._spec()
    script = "@COMMAND_RUNNER@\n@SHA256_HELPER@\nLANE = json.loads(@LANE_JSON@)\n"
    from core.common import data_cfg, training_cfg

    payload = spec.model_dump(mode="json")
    payload["paths"] = data_cfg().paths.model_dump(mode="json")
    payload["archives"] = training_cfg().archives.model_dump(mode="json")
    legacy = (script.replace("@COMMAND_RUNNER@", KernelTemplates.REMOTE_COMMAND_RUNNER)
              .replace("@SHA256_HELPER@", KernelTemplates.SHA256_HELPER)
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
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

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
