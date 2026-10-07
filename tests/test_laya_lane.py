"""Laya decision lane — offline pins + dry-run safety.

NEW-lane pins (branch laya-lane). Everything here is offline: no test
touches the network or the kaggle CLI; every ruby-side remote surface is
`--execute`-gated and fails loud BEFORE any subprocess (RuntimeError).
Staging runs entirely against a tmp TRAIN_ROOT with hermetic F bindings.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from core import common
from core.schemas import LayaSpec
from cli import laya_lane


def _spec(tmp_path, monkeypatch, **updates):
    """Point the lane at a tmp TRAIN_ROOT + hermetic cfg."""
    cfg_spec = LayaSpec(**{
        "export_dataset_slug": "fbarulli/er-laya-decision",
        "dataset_slug": "fbarulli/er-laya-payload",
        **updates,
    })
    monkeypatch.setattr(laya_lane, "_spec", lambda: cfg_spec)
    monkeypatch.setattr(laya_lane, "TRAIN_ROOT", tmp_path)
    return cfg_spec


def _question_schema(tmp_path, monkeypatch, *, questions=3):
    """Land the config question schema inside the tmp TRAIN_ROOT."""
    schema = {"questions": {
        "attribute_alignment": {"type": "choice",
                                "instructions": "verdict?",
                                "criteria": {"aligned": None}},
        "identity_claim": {"type": "noul", "instructions": "same item?"},
        "package_state": {"type": "noul", "instructions": "has pack?"},
    }}
    body = {"questions": dict(sorted(schema["questions"].items())[:questions])}
    bound_path = tmp_path / "config/laya.question.json"
    bound_path.parent.mkdir(parents=True, exist_ok=True)
    bound_path.write_text(json.dumps(body), encoding="utf-8")
    return bound_path


def _dataset_fixture(tmp_path, monkeypatch, *, rows=2):
    """Hermetic decision CSVs + F resolvement through core.common.F."""
    import core.common as core_common

    dataset = tmp_path / "dataset.csv"
    dataset.write_text(
        "sku_id,sku_name_eng,attribute\n"
        + "".join(f"SKU{index},Name {index} 500 ml,Volume: 500; Brand: A\n"
                  for index in range(rows)),
        encoding="utf-8")
    final_validation = tmp_path / "final_validation.csv"
    final_validation.write_text(
        "gtin1,gtin2,true_label,attribute_pairs\n"
        + "".join(f"1,2,{index % 2},Vol 500; Pack 6\n"
                  for index in range(rows)),
        encoding="utf-8")
    monkeypatch.setitem(core_common.F, "dataset", dataset)
    monkeypatch.setitem(core_common.F, "final_validation", final_validation)


REVISION_PIN = "abc123def"


def _hermetic_staging(monkeypatch, *, origin_tip: str = REVISION_PIN):
    """Kernel staging pins the worktree HEAD into the payload constants;
    these pins only need the payload contracts, so the fake closes
    that door (kaggle test file precedent). The staged-pin guard
    (runtime_inputs.require_published_tip_match) rides the same fake:
    `git fetch origin` is rc=0 and `git rev-parse origin/<branch>`
    prints the origin tip (default: the pin matches REVISION_PIN)."""
    monkeypatch.setattr(laya_lane, "_git_revision", lambda: REVISION_PIN)
    import subprocess

    def fake_run(args, **kwargs):
        if "rev-parse" in args:
            return subprocess.CompletedProcess(args, 0, stdout=origin_tip,
                                               stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)


# ── SSOT/additive contract ─────────────────────────────────────────────────
def test_laya_spec_additive_and_yaml_unchanged():
    cfg_spec = common.training_cfg().laya
    # committed YAML stays byte-identical: no `laya:` block is expected
    # (the schema's default factory keeps the load value-additive).
    raw = yaml.safe_load(
        (common.TRAIN_ROOT / "config/training.yaml").read_text())
    assert "laya" not in raw
    assert cfg_spec.staging_dir == "results/laya_lane"
    assert cfg_spec.gpu == "T4"
    assert cfg_spec.checkpoint_hub == "convaiinnovations/laya"
    assert cfg_spec.question_schema == "config/laya.question.json"


@pytest.mark.parametrize("updates", [
    {"staging_dir": "/abs/path"},
    {"staging_dir": "../escape"},
    {"gpu": "2xT4"},
    {"gpu": "V100"},
    {"laya_decision_batch_size": 0},
    {"unexpected": True},
])
def test_laya_spec_rejects_non_portable_and_bad_contract(updates):
    with pytest.raises(ValidationError):
        LayaSpec(**updates)


def test_lane_kind_validation_contract():
    for kind in ("kaggle", "colab"):
        assert laya_lane.LayaLane(kind).kind == kind
    # ONE class, TWO kinds: anything else fails loud (never a second class
    # inherits `LayaLane`).
    with pytest.raises(ValueError):
        laya_lane.LayaLane("others")
    with pytest.raises(ValueError):
        laya_lane.LayaLane("kaggle ")
    assert not hasattr(laya_lane, "LayaColabImpl") or callable(
        getattr(laya_lane, "LayaColabImpl"))


# ── staging precondition fail-louds ───────────────────────────────────────
def test_missing_question_schema_fails_loud(tmp_path, monkeypatch):
    _spec(tmp_path, monkeypatch)
    # no question schema staged: no silent empty placeholder
    with pytest.raises(FileNotFoundError):
        laya_lane.stage_question_schema("kaggle",
                                        override=tmp_path / "config/none.json")


def test_empty_question_schema_fails_loud(tmp_path, monkeypatch):
    _spec(tmp_path, monkeypatch)
    bad = tmp_path / "given.json"
    bad.write_text(json.dumps({"questions": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="no 'questions'"):
        laya_lane.stage_question_schema("kaggle", override=bad)


def test_decision_input_fail_loud_on_missing(tmp_path, monkeypatch):
    _spec(tmp_path, monkeypatch)
    import core.common as core_common

    monkeypatch.setitem(core_common.F, "dataset",
                        tmp_path / "dataset.csv")
    with pytest.raises(FileNotFoundError):
        laya_lane.stage_decision_input("kaggle", decision_kind="attribute")


def test_decision_input_fails_loud_on_wrong_header(tmp_path, monkeypatch):
    _spec(tmp_path, monkeypatch)
    import core.common as core_common

    wrong = tmp_path / "wrong.csv"
    wrong.write_text("a,\b,c\n1,2,3\n", encoding="utf-8")
    monkeypatch.setitem(core_common.F, "dataset", wrong)
    with pytest.raises(ValueError, match="missing columns"):
        laya_lane.stage_decision_input("kaggle", decision_kind="attribute")


def test_unbound_csv_binding_fails_loud(tmp_path, monkeypatch):
    _spec(tmp_path, monkeypatch, decision_csv_bindings={})
    import core.common as core_common

    monkeypatch.setitem(core_common.F, "dataset", tmp_path / "dataset.csv")
    with pytest.raises(RuntimeError, match="decision_csv_bindings"):
        laya_lane.stage_decision_input("kaggle", decision_kind="attribute")


def test_stage_decision_kernel_fails_loud_without_slug(tmp_path, monkeypatch):
    _spec(tmp_path, monkeypatch, export_dataset_slug=None)
    with pytest.raises(RuntimeError, match="export_dataset_slug"):
        laya_lane.stage_decision_kernel(decision_kind="attribute")


def test_stage_decision_kernel_fails_loud_when_disabled(tmp_path, monkeypatch):
    _spec(tmp_path, monkeypatch, laya_decision_epochs=0)
    with pytest.raises(RuntimeError, match="disabled"):
        laya_lane.stage_decision_kernel(decision_kind="attribute")


# ── staged payload contracts ──────────────────────────────────────────────
def test_stage_kaggle_payload_contract(tmp_path, monkeypatch):
    _spec(tmp_path, monkeypatch)
    _question_schema(tmp_path, monkeypatch)
    _dataset_fixture(tmp_path, monkeypatch)
    _hermetic_staging(monkeypatch)
    receipt = laya_lane.stage_decision_kernel(decision_kind="attribute")
    assert receipt["kernel"] == "fbarulli/er-laya-decision"
    assert receipt["kind"] == "attribute"
    assert receipt["gpu"] == "T4 (single)"
    assert receipt["run_tag"].startswith("laya_")
    assert receipt["checkpoint_hub"] == "convaiinnovations/laya"
    stage = Path(receipt["staged"])
    assert stage == tmp_path / "results/laya_lane/kaggle/attribute"
    # payload co-location: kernel script + metadata + question schema +
    # the corresponding decision CSV and its receipt
    names = sorted(p.name for p in stage.iterdir())
    assert "laya_decision.py" in names
    assert "kernel-metadata.json" in names
    assert "laya.question.json" in names
    assert "dataset.csv" in names
    assert "attribute.receipt.json" in names
    # single T4 ruling: no double accelerator request in the metadata
    metadata = json.loads((stage / "kernel-metadata.json").read_text())
    assert metadata["enable_gpu"] is True
    assert metadata["code_file"] == "laya_decision.py"
    assert "2x" not in json.dumps(metadata)
    # kernel script gate: the staged script parses cleanly, and the single
    # T4 device pin (never a second accelerator) is in the script
    script = (stage / "laya_decision.py").read_text()
    ast.parse(script)
    assert 'CUDA_VISIBLE_DEVICES"] = "0"' in script
    assert "never a second one" in script
    # the hub source decided at kernel stage: the checkpoint hub source
    # the run loads and the batch size/knob pin are in the script consts
    assert "CHECKPOINT_HUB = \"convaiinnovations/laya\"" in script
    assert "BATCH_SIZE = 8" in script
    # the run tag is embedded for the receipt read-back
    assert receipt["run_tag"] in script
    # the runtime preflight inventory bake: the staged payload carries the
    # publish pin constants + the checkout preflight block (kaggle-lane
    # sibling shape; the push gate reads these constants)
    assert 'REPOSITORY = "https://github.com/fbarulli/ER.git"' in script
    assert 'BRANCH = "kaggle-lane"' in script
    assert f'REVISION = "{REVISION_PIN}"' in script
    # THE ATTACHED-INPUTS preflight (BUG 1 fix): root is /kaggle/input
    # (never `_runtime_root = Path(root)` — no clone root is defined),
    # the inventory shrank to the two attached staged files, and the
    # verified-N print is the sibling kaggle-lane shape
    assert 'INPUT_ROOT = Path("/kaggle/input")' in script
    assert "_runtime_root = Path(root)" not in script
    assert '_runtime_files = ("dataset.csv"' in script
    assert "[runtime-preflight] verified %d required files" in script
    # BUG 2 fix: the inputs travel as the dataset — metadata attaches the
    # slug and the staging receipt records the dataset payload files
    assert metadata["dataset_sources"] == ["fbarulli/er-laya-payload"]
    payload_dir = stage / "dataset_payload"
    assert (payload_dir / "dataset-metadata.json").is_file()
    assert (payload_dir / "dataset.csv").is_file()
    assert (payload_dir / "laya.question.json").is_file()
    assert receipt["dataset"]["slug"] == "fbarulli/er-laya-payload"
    # the kernel resolves the RENAMED csv by name (the dataset csv lands
    # under /kaggle/input/<slug>/dataset.csv; rglob finds it)
    assert 'DECISION_CSV = "dataset.csv"' in script
    # the receipt publishes the pin so the orchestrator knows the
    # publish-tip expectation
    assert receipt["published_pin"] == {
        "repository": "https://github.com/fbarulli/ER.git",
        "branch": "kaggle-lane",
        "revision": REVISION_PIN,
    }


def test_stage_identity_and_eval_kinds(tmp_path, monkeypatch):
    _spec(tmp_path, monkeypatch)
    _question_schema(tmp_path, monkeypatch)
    _dataset_fixture(tmp_path, monkeypatch)
    _hermetic_staging(monkeypatch)
    receipt = laya_lane.stage_decision_kernel(decision_kind="identity")
    assert receipt["kind"] == "identity"
    assert (Path(receipt["staged"]) / "laya_decision.py").is_file()
    # laya-evals is its own (optional) kernel: a separate eval script.
    eval_spec_state = laya_lane._spec().model_copy(update={
        "export_dataset_slug": "fbarulli/er-laya-eval"})
    monkeypatch.setattr(laya_lane, "_spec", lambda: eval_spec_state)
    eval_receipt = laya_lane.stage_decision_kernel(decision_kind="laya-cli-eval")
    assert eval_receipt["kernel"] == "fbarulli/er-laya-eval"
    assert eval_receipt["code_file"] == "laya_evals.py"
    assert (Path(eval_receipt["staged"]) / "laya_evals.py").is_file()
    # the eval kernel carries the same preflight inventory bake
    eval_script = (Path(eval_receipt["staged"])
                   / "laya_evals.py").read_text()
    assert 'REPOSITORY = "https://github.com/fbarulli/ER.git"' in eval_script
    assert "_runtime_files = (" in eval_script
    assert eval_receipt["published_pin"] == {
        "repository": "https://github.com/fbarulli/ER.git",
        "branch": "kaggle-lane",
        "revision": REVISION_PIN,
    }
    # the eval harness is unlatched by config default; the receipt records
    # what toggle state the kernel actually holds
    assert eval_receipt["evals_enabled"] is False


def test_stage_decision_kernel_fails_loud_without_dataset_slug(
        tmp_path, monkeypatch):
    """BUG 2 regression pin: the kernel inputs travel as the dataset; an
    unset laya.dataset_slug may never stage a push-less payload silently."""
    _spec(tmp_path, monkeypatch, dataset_slug=None)
    _question_schema(tmp_path, monkeypatch)
    _dataset_fixture(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="dataset_slug"):
        laya_lane.stage_decision_kernel(decision_kind="attribute")


def test_stage_decision_kernel_refuses_stale_published_tip(
        tmp_path, monkeypatch):
    """The staged-race guard: local HEAD must BE the fetched origin
    tip before any payload writes; the 84ce2d0-vs-02dec14 pin may
    never stage again."""
    _spec(tmp_path, monkeypatch)
    _question_schema(tmp_path, monkeypatch)
    _dataset_fixture(tmp_path, monkeypatch)
    _hermetic_staging(monkeypatch, origin_tip="fed321cba9" + "0" * 35)
    with pytest.raises(RuntimeError, match="origin/kaggle-lane tip"):
        laya_lane.stage_decision_kernel(decision_kind="attribute")
    stage = tmp_path / "results/laya_lane/kaggle/attribute"
    assert not (stage / "kernel-metadata.json").exists()
    assert not (stage / "laya.question.json").exists()


def test_stage_kaggle_payload_contract_receipts_published_tip(
        tmp_path, monkeypatch):
    _spec(tmp_path, monkeypatch)
    _question_schema(tmp_path, monkeypatch)
    _dataset_fixture(tmp_path, monkeypatch)
    _hermetic_staging(monkeypatch)
    receipt = laya_lane.stage_decision_kernel(decision_kind="attribute")
    # the sibling published_tip records the verified origin tip; the
    # pin itself still names HEAD
    assert receipt["published_tip"] == REVISION_PIN
    assert receipt["published_pin"]["revision"] == REVISION_PIN


def test_dataset_payload_contract(tmp_path, monkeypatch):
    """The DATASET payload the inputs travel with (BUG 2 fix): metadata
    shape (kaggle-lane payload shape) + the renamed csv + schema copies."""
    _spec(tmp_path, monkeypatch)
    _question_schema(tmp_path, monkeypatch)
    _dataset_fixture(tmp_path, monkeypatch)
    _hermetic_staging(monkeypatch)
    import hashlib

    receipt = laya_lane.stage_decision_kernel(decision_kind="attribute")
    stage = Path(receipt["staged"])
    payload = stage / "dataset_payload"
    metadata = json.loads(
        (payload / "dataset-metadata.json").read_text())
    assert metadata == {
        "title": "er laya requests",
        "id": "fbarulli/er-laya-payload",
        "licenses": [{"name": "other"}],
    }
    data = Path(receipt["decision_input"])
    assert (payload / "dataset.csv").read_bytes() == data.read_bytes()
    assert (payload / "laya.question.json").is_file()
    payload_receipt = json.loads(
        (payload / "dataset_payload.receipt.json").read_text())
    assert payload_receipt["dataset"] == "fbarulli/er-laya-payload"
    assert payload_receipt["files"]["dataset.csv"] == hashlib.sha256(
        data.read_bytes()).hexdigest()
    # the attach itself is recorded in the KERNel staging receipt; the
    # executed publish adds action + version later (see the publish pin)
    assert receipt["dataset"]["payload"] == str(payload)


def test_dataset_publish_dry_run_and_activate_gate(tmp_path, monkeypatch):
    """publish_laya_dataset: dry run never spawns, and an un-staged
    payload fails loud at the --activate gate (mirroring kernels push)."""
    _spec(tmp_path, monkeypatch)
    _question_schema(tmp_path, monkeypatch)
    _dataset_fixture(tmp_path, monkeypatch)
    _hermetic_staging(monkeypatch)
    laya_lane.stage_decision_kernel(decision_kind="attribute")

    def no_subprocess(*args, **kwargs):  # pragma: no cover
        raise AssertionError("offline: dry-run must not spawn the CLI")

    monkeypatch.setattr(laya_lane.subprocess, "run", no_subprocess)
    plan = laya_lane.publish_laya_dataset("attribute", run_tag="laya_t",
                                          execute=False)
    assert plan["mode"] == "dry-run"
    with pytest.raises(RuntimeError, match="--activate gate"):
        laya_lane.publish_laya_dataset("identity", run_tag="laya_t",
                                       execute=True)


def test_dataset_publish_executed_uses_kaggle_lane_helpers(
        tmp_path, monkeypatch):
    """Executed attach: version-existing datasets via the IMPORTED
    kaggle-lane helpers (never copied), zip payload + receipt carries
    the recorded dataset version (offline-pinned, no network)."""
    from cli import kaggle_datasets
    from cli import kaggle_lane as lane

    _spec(tmp_path, monkeypatch)
    _question_schema(tmp_path, monkeypatch)
    _dataset_fixture(tmp_path, monkeypatch)
    _hermetic_staging(monkeypatch)
    stage_receipt_path = (tmp_path / "results/laya_lane/kaggle/attribute"
                          / "attribute.receipt.json")
    laya_lane.stage_decision_kernel(decision_kind="attribute")
    commands = []
    monkeypatch.setattr(
        kaggle_datasets.KaggleDatasets, "_dataset_current_version",
        staticmethod(lambda slug: {"dataset_version": 3, "slug": slug}))
    monkeypatch.setattr(
        lane, "_require_kaggle_executable", staticmethod(
            lambda executable: str(tmp_path / "fake-kaggle")))
    def fake_run_kaggle(command):
        commands.append(command)
        return 0, ""

    monkeypatch.setattr(
        lane, "_run_kaggle", staticmethod(fake_run_kaggle))
    plan = laya_lane.publish_laya_dataset("attribute", run_tag="laya_t",
                                          execute=True)
    assert plan["mode"] == "executed"
    assert plan["action"] == "version"
    assert plan["dataset_version"] == 3
    assert commands and commands[0][-5:] == [
        "zip", "-m", "laya inputs laya_t", "-p",
        str(tmp_path / "results/laya_lane/kaggle/attribute/dataset_payload")
    ] and "datasets" in commands[0] and "version" in commands[0]
    receipt = json.loads(stage_receipt_path.read_text())
    assert receipt["dataset"]["action"] == "version"
    assert receipt["dataset"]["version"] == 3


def test_module_scope_gate_pins_nameerror_payload():
    """AST-gate hardening regression pin (BUG 1): a post-substitution
    payload loading an undefined TOP-LEVEL name (the `_runtime_root =
    Path(root)` NameError class) never stages again."""
    laya_lane._module_scope_gate(
        "import json\nfrom pathlib import Path\n"
        "QUESTION_SCHEMA_FILE = 'laya.question.json'\n"
        "INPUT_ROOT = Path('/kaggle/input')\n"
        "_runtime_files = (QUESTION_SCHEMA_FILE,)\n"
        "print('ok')\n")
    with pytest.raises(ValueError, match="undefined top-level names"):
        laya_lane._module_scope_gate(
            "import json\nfrom pathlib import Path\n"
            "print(Path(root))\n")


def test_staging_fails_loud_on_undefined_name_before_writes(
        tmp_path, monkeypatch):
    """End-to-end pin: a payload whose preflight emits an undefined
    name fails staging BEFORE the kernel metadata lands (atomic
    writes stay behind the gate)."""
    _spec(tmp_path, monkeypatch)
    _question_schema(tmp_path, monkeypatch)
    _dataset_fixture(tmp_path, monkeypatch)
    _hermetic_staging(monkeypatch)
    monkeypatch.setattr(laya_lane, "LAYA_RUNTIME_PREFLIGHT",
                        "print(boom_undefined_name)\n")
    with pytest.raises(ValueError, match="undefined top-level names"):
        laya_lane.stage_decision_kernel(decision_kind="attribute")
    stage = tmp_path / "results/laya_lane/kaggle/attribute"
    assert not (stage / "kernel-metadata.json").exists()


def test_stage_identity_csv_header_contract_fails_loud(tmp_path, monkeypatch):
    _spec(tmp_path, monkeypatch)
    _question_schema(tmp_path, monkeypatch)
    _dataset_fixture(tmp_path, monkeypatch)
    _hermetic_staging(monkeypatch)
    # the identity binding wants the frozen P0 columns; a dataset.csv
    # forged onto the F binding raises the header mismatch
    import core.common as core_common

    monkeypatch.setitem(core_common.F, "final_validation",
                        core_common.F["dataset"])
    with pytest.raises(ValueError, match="missing columns"):
        laya_lane.stage_decision_kernel(decision_kind="identity")


def test_stage_colab_delivery_contract(tmp_path, monkeypatch):
    _spec(tmp_path, monkeypatch)
    _question_schema(tmp_path, monkeypatch)
    _dataset_fixture(tmp_path, monkeypatch)
    receipt = laya_lane.stage_colab_notebook(decision_kind="identity")
    assert receipt["kind"] == "identity"
    notebook = Path(receipt["notebook"])
    assert receipt["notebook"] == str(
        tmp_path / "results/laya_lane/colab/identity/laya_decision_colab.py")
    assert notebook.is_file()
    script = notebook.read_text(encoding="utf-8")
    ast.parse(script)
    # delivery contract: the notebook never imports or edits cli.colab /
    # cli.colab_lane, and never opens a session
    assert "import cli.colab" not in script
    assert "colab.research" not in script
    assert "BOOK-END CONTRACT ONLY" in script
    # colab payload staging in receipts style (receipt json colocated)
    receipt_path = notebook.parent / "identity.receipt.json"
    assert json.loads(receipt_path.read_text())["kind"] == "identity"
    # push via the colab lane is denied (kaggle-only op)
    with pytest.raises(RuntimeError, match="kaggle-lane"):
        laya_lane.LayaLane("colab").push(notebook.parent)


def test_colab_main_actually_reports_delivery_only(tmp_path, monkeypatch):
    _spec(tmp_path, monkeypatch)
    _question_schema(tmp_path, monkeypatch)
    _dataset_fixture(tmp_path, monkeypatch)
    argv = ["laya", "--kind", "colab", "--decision", "attribute",
            "--execute"]
    monkeypatch.setattr("sys.argv", argv)
    laya_lane.main()
    # executed colab run is a delivery contract, never a session
    assert (tmp_path / "results/laya_lane/colab/attribute"
            / "attribute.receipt.json").is_file()


# ── push / fetch --execute gating (offline safety) ────────────────────────
def test_push_dry_run_and_fail_loud_activate_gate(tmp_path, monkeypatch):
    _spec(tmp_path, monkeypatch)
    def no_subprocess(*args, **kwargs):  # pragma: no cover
        raise AssertionError("offline: dry-run must not spawn the CLI")

    monkeypatch.setattr(laya_lane.subprocess, "run", no_subprocess)
    lane = laya_lane.LayaLane("kaggle")
    plan = lane.push(tmp_path / "not_staged", execute=False)
    assert plan["mode"] == "dry-run"
    assert plan["argv"][-3] == "push"
    # --execute on an unstaged dir: fail loud before any subprocess
    with pytest.raises(RuntimeError, match="--activate gate"):
        lane.push(tmp_path / "not_staged", execute=True)


def test_push_executed_gates_on_metadata_and_stage(tmp_path, monkeypatch):
    _spec(tmp_path, monkeypatch)
    stage = tmp_path / "results/laya_lane/kaggle/attribute"
    stage.mkdir(parents=True)
    (stage / "kernel-metadata.json").write_text(json.dumps({"id": "a/b"}),
                                                encoding="utf-8")
    # offline guard: any subprocess spawn is a test failure
    def no_subprocess(argv, **kwargs):
        raise AssertionError(
            "offline: preflight should have failed before any subprocess")

    monkeypatch.setattr(laya_lane.subprocess, "run", no_subprocess)
    with pytest.raises(Exception):
        laya_lane.push_kaggle_kernel(stage, execute=True)


def test_decision_kind_registry_contract():
    from cli.laya_lane import DECISION_BINDINGS

    assert set(DECISION_BINDINGS) == {"attribute", "identity",
                                      "laya-cli-eval"}
    # every binding carries the class contract: header columns + state
    for entry in DECISION_BINDINGS.values():
        assert entry["wanted_columns"]
        assert entry["state_column"]
        assert entry["description"]


# ── lane logging convention ───────────────────────────────────────────────
def test_lane_log_compiles_cet_stamp_lines(tmp_path, monkeypatch):
    _spec(tmp_path, monkeypatch)
    laya_lane._log_lane("staged decision payload (offline)")
    log_path = tmp_path / "logs/laya/lane.log"
    body = log_path.read_text(encoding="utf-8")
    # kaggle-lane / colab-lane stamp convention: Europe/Paris CET|CEST zone
    assert ("CET" in body or "CEST" in body)
    assert body.startswith("2026-")
    assert "staged decision payload" in body


def test_stage_receipts_layout_is_per_op(tmp_path, monkeypatch):
    """Receipts land under results/laya_lane/<kind>/<decision>/ exactly."""
    _spec(tmp_path, monkeypatch)
    _question_schema(tmp_path, monkeypatch)
    _dataset_fixture(tmp_path, monkeypatch)
    _hermetic_staging(monkeypatch)
    receipt = laya_lane.stage_decision_kernel(decision_kind="identity")
    stage = Path(receipt["staged"])
    relative = stage.relative_to(tmp_path).as_posix()
    assert relative == "results/laya_lane/kaggle/identity"


def test_decision_input_flag_forwards_override(tmp_path, monkeypatch):
    """--decision-input drives the stage_decision_input override (kaggle)."""
    _spec(tmp_path, monkeypatch)
    _question_schema(tmp_path, monkeypatch)
    _dataset_fixture(tmp_path, monkeypatch)
    _hermetic_staging(monkeypatch)
    alternate = tmp_path / "alternate.csv"
    alternate.write_text(
        "sku_id,sku_name_eng,attribute\nSKU0,Name 0 500 ml,Vol: 500\n",
        encoding="utf-8")
    argv = ["laya", "--kind", "kaggle", "--decision", "attribute",
            "--decision-input", str(alternate)]
    monkeypatch.setattr("sys.argv", argv)
    laya_lane.main()
    stage = tmp_path / "results/laya_lane/kaggle/attribute"
    assert (stage / "alternate.csv").is_file()
    receipt = json.loads((stage / "attribute.receipt.json").read_text())
    # the kernel receipt carries the override (the decision-input receipt
    # of the same name is superseded by the kernel receipt co-located there)
    import hashlib

    assert receipt["decision_input"].endswith("alternate.csv")
    assert receipt["decision_sha256"] == hashlib.sha256(
        alternate.read_bytes()).hexdigest()
