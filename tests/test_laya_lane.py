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


def _hosted_slug(role: str) -> str:
    """The hosted-dataset slug the lane must use for ``role`` (registry SSOT)."""
    from core.hosted_dataset import hosted_registry

    return hosted_registry().by_role(role).slug


def _drop_hosted_role(monkeypatch, role: str) -> None:
    """Serve a registry that lacks ``role`` (the fail-loud pin, never a path)."""
    from core import laya_config
    from core.hosted_dataset import hosted_registry

    base = hosted_registry()
    entries = {slug: entry for slug, entry in base.entries.items()
               if entry.role != role}
    monkeypatch.setattr(laya_config, "hosted_registry",
                        lambda: base.model_copy(update={"entries": entries}))


def _spec(tmp_path, monkeypatch, **updates):
    """Point the lane at a tmp TRAIN_ROOT + hermetic cfg.

    No hosted slug is set here: the lane reads every one from the registry
    (``config/hosted_datasets.yaml``), so these tests assert the registry's own
    values (``_hosted_slug``).
    """
    cfg_spec = LayaSpec(**updates)
    monkeypatch.setattr(laya_lane, "_spec", lambda: cfg_spec)
    monkeypatch.setattr(laya_lane, "TRAIN_ROOT", tmp_path)
    # Hermetic branch: the real config may carry a local branch pin (e.g.
    # `kaggle.branch: laya` while running from a feature branch); these tests
    # pin the main-branch contract, so force it regardless of the checkout.
    from core.common import training_cfg as _tcfg

    _base = _tcfg()
    _forced = _base.model_copy(update={
        "kaggle": _base.kaggle.model_copy(update={"branch": "main"})})
    monkeypatch.setattr(laya_lane, "training_cfg", lambda: _forced)
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


def test_laya_spec_references_the_hosted_registry():
    """SSOT pin: LayaSpec declares NO hosted slug.

    The six fields this lane used to carry (`base_model_dataset`,
    `dataset_slug`, `export_dataset_slug`, `finetune_dataset_slug`,
    `finetune_ckpt_dataset`, `holdout_dataset_slug`) are gone: every slug is
    read from the registry, and ``config/hosted_datasets.yaml`` is the ONE
    place a hosted slug is spelled (the lane's kernel slugs are not hosted
    datasets and stay fields).
    """
    from core import laya_config
    from core.hosted_dataset import hosted_registry

    removed = {
        "base_model_dataset", "dataset_slug", "export_dataset_slug",
        "finetune_dataset_slug", "finetune_ckpt_dataset",
        "holdout_dataset_slug",
    }
    assert not removed & set(LayaSpec.model_fields)
    source = Path(laya_config.__file__).read_text(encoding="utf-8")
    registry = hosted_registry()
    # no hosted slug as a CODE literal (a comment may name one): the registry
    # document is the only place a slug is declared
    literals = {node.value for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.Constant)
                and isinstance(node.value, str)}
    assert not literals & set(registry.slugs())
    spec = LayaSpec()
    assert spec.hosted_slug("requests") == "fbarulli/er-laya-requests"
    assert spec.hosted_slug("base") == "fbarulli/er-laya-base"
    assert spec.mount_root == Path("/kaggle/input")


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


# ── the eval-path calibration/abstention selection (laya.eval_calibration) ──
def test_eval_calibration_defaults_preserve_the_landed_eval():
    """No block -> the landed eval exactly (temperature fit on, no abstention)."""
    from core.laya_config import EvalCalibrationSpec

    spec = LayaSpec()
    assert spec.eval_calibration.temperature is True
    assert spec.eval_calibration.abstention is False
    assert spec.eval_calibration.target_error == 0.10
    assert spec.eval_calibration.min_abstain_n == 10
    assert spec.eval_calibration.min_confidence is None
    # ONE declaration: the baked field tuple names exactly the spec surface.
    assert set(laya_lane.EVAL_CALIBRATION_FIELDS) == set(
        EvalCalibrationSpec.model_fields)
    assert laya_lane.eval_calibration_config(spec) == {
        "temperature": True, "abstention": False, "target_error": 0.10,
        "min_abstain_n": 10, "min_confidence": None}


def test_eval_calibration_flows_yaml_into_the_ssot_config():
    spec = LayaSpec(eval_calibration={
        "abstention": True, "target_error": 0.25, "min_abstain_n": 5,
        "min_confidence": 0.8})
    config = laya_lane.eval_calibration_config(spec)
    assert config == {"temperature": True, "abstention": True,
                      "target_error": 0.25, "min_abstain_n": 5,
                      "min_confidence": 0.8}
    # an unknown knob fails loud (extra='forbid'), never silently ignored
    with pytest.raises(ValidationError):
        LayaSpec(eval_calibration={"nope": 1})
    # abstention cuts on the CALIBRATED confidence scale: it needs the
    # temperature fit, so the contradictory pair is refused at config load.
    with pytest.raises(ValidationError, match="requires temperature"):
        LayaSpec(eval_calibration={"abstention": True, "temperature": False})
    with pytest.raises(ValidationError):
        LayaSpec(eval_calibration={"min_confidence": 1.5})


def test_finetune_package_and_corpus_dir_are_config_owned():
    """The pin + the corpus root are LayaSpec knobs, never code literals."""
    spec = LayaSpec()
    assert spec.finetune_package == "laya>=0.3.29"
    assert spec.finetune_corpus_dir == "data/laya"
    custom = LayaSpec(finetune_package="laya==9.9.9",
                      finetune_corpus_dir="data/laya_v2")
    assert custom.finetune_package == "laya==9.9.9"
    assert custom.finetune_corpus_dir == "data/laya_v2"
    # both stay portable names (no absolute path, no traversal)
    with pytest.raises(ValidationError):
        LayaSpec(finetune_corpus_dir="../escape")
    # the deprecated module alias now derives from the SSOT default
    from core.laya_config import LayaSpec as _LayaSpec

    assert laya_lane.FINETUNE_LAYA_PACKAGE == _LayaSpec().finetune_package


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


def test_stage_decision_kernel_fails_loud_without_the_hosted_role(
        tmp_path, monkeypatch):
    """The slug is not a knob: a registry that does not declare the decision
    roles fails the stage loudly BEFORE any payload write."""
    _spec(tmp_path, monkeypatch)
    _drop_hosted_role(monkeypatch, "decisions")
    with pytest.raises(KeyError, match="decisions"):
        laya_lane.stage_decision_kernel(decision_kind="attribute")
    stage = tmp_path / "results/laya_lane/kaggle/attribute"
    assert not (stage / "kernel-metadata.json").exists()


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
    assert receipt["kernel"] == _hosted_slug("decisions")
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
    assert 'BRANCH = "main"' in script
    assert f'REVISION = "{REVISION_PIN}"' in script
    # THE ATTACHED-INPUTS preflight (BUG 1 fix): root is /kaggle/input
    # (never `_runtime_root = Path(root)` — no clone root is defined),
    # the inventory shrank to the two attached staged files, and the
    # verified-N print is the sibling kaggle-lane shape
    assert 'INPUT_ROOT = Path("/kaggle/input")' in script
    assert "_runtime_root = Path(root)" not in script
    assert '_runtime_files = ("dataset.csv"' in script
    assert "[runtime-preflight] verified %d required files" in script
    # the inputs travel as the hosted `requests` dataset — metadata attaches
    # the registry's slug and the staging receipt records the dataset payload
    assert metadata["dataset_sources"] == [_hosted_slug("requests")]
    payload_dir = stage / "dataset_payload"
    assert (payload_dir / "dataset-metadata.json").is_file()
    assert (payload_dir / "dataset.csv").is_file()
    assert (payload_dir / "laya.question.json").is_file()
    assert receipt["dataset"]["slug"] == _hosted_slug("requests")
    # the kernel resolves the RENAMED csv by name (the dataset csv lands
    # under /kaggle/input/<slug>/dataset.csv; rglob finds it)
    assert 'DECISION_CSV = "dataset.csv"' in script
    # the receipt publishes the pin so the orchestrator knows the
    # publish-tip expectation
    assert receipt["published_pin"] == {
        "repository": "https://github.com/fbarulli/ER.git",
        "branch": "main",
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
    # laya-evals is its own (optional) kernel: a separate eval script, and it
    # publishes under the SAME hosted `decisions` dataset (kernel id == slug).
    eval_receipt = laya_lane.stage_decision_kernel(decision_kind="laya-cli-eval")
    assert eval_receipt["kernel"] == _hosted_slug("decisions")
    assert eval_receipt["code_file"] == "laya_evals.py"
    assert (Path(eval_receipt["staged"]) / "laya_evals.py").is_file()
    # the eval kernel carries the same preflight inventory bake
    eval_script = (Path(eval_receipt["staged"])
                   / "laya_evals.py").read_text()
    assert 'REPOSITORY = "https://github.com/fbarulli/ER.git"' in eval_script
    assert "_runtime_files = (" in eval_script
    assert eval_receipt["published_pin"] == {
        "repository": "https://github.com/fbarulli/ER.git",
        "branch": "main",
        "revision": REVISION_PIN,
    }
    # the eval harness is unlatched by config default; the receipt records
    # what toggle state the kernel actually holds
    assert eval_receipt["evals_enabled"] is False


def test_stage_decision_kernel_refuses_stale_published_tip(
        tmp_path, monkeypatch):
    """The staged-race guard: local HEAD must BE the fetched origin
    tip before any payload writes; the 84ce2d0-vs-02dec14 pin may
    never stage again."""
    _spec(tmp_path, monkeypatch)
    _question_schema(tmp_path, monkeypatch)
    _dataset_fixture(tmp_path, monkeypatch)
    _hermetic_staging(monkeypatch, origin_tip="fed321cba9" + "0" * 35)
    with pytest.raises(RuntimeError, match="origin/main tip"):
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
    from core.portable_archive import ByteCount

    receipt = laya_lane.stage_decision_kernel(decision_kind="attribute")
    stage = Path(receipt["staged"])
    payload = stage / "dataset_payload"
    metadata = json.loads(
        (payload / "dataset-metadata.json").read_text())
    assert metadata == {
        "title": "er laya requests",
        "id": _hosted_slug("requests"),
        "licenses": [{"name": "other"}],
    }
    data = Path(receipt["decision_input"])
    assert (payload / "dataset.csv").read_bytes() == data.read_bytes()
    assert (payload / "laya.question.json").is_file()
    payload_receipt = json.loads(
        (payload / "dataset_payload.receipt.json").read_text())
    assert payload_receipt["dataset"] == _hosted_slug("requests")
    assert payload_receipt["files"]["dataset.csv"] == ByteCount(
        data.read_bytes()).total
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
                                      "laya-cli-eval", "finetune",
                                      "finetune-eval"}
    # every binding carries the class contract: header columns + state
    for entry in DECISION_BINDINGS.values():
        assert entry["wanted_columns"]
        assert entry["state_column"]
        assert entry["description"]


def test_holdout_report_name_is_the_declared_results_leaf():
    """The holdout report name is owned by Results, never re-spelled here."""
    from core.results import Results

    assert laya_lane.HOLDOUT_EVAL_REPORT_FILE == Results.leaf("holdout_report")
    # The generated kernel carries the same leaf, not a copy of the literal.
    assert 'Results.leaf("holdout_report")' in laya_lane.HOLDOUT_EVAL_KERNEL_SCRIPT
    assert "holdout_report.json" not in laya_lane.HOLDOUT_EVAL_KERNEL_SCRIPT


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
    from core.portable_archive import ByteCount

    assert receipt["decision_input"].endswith("alternate.csv")
    assert receipt["decision_size"] == ByteCount(
        alternate.read_bytes()).total


# ── accuracy/F1 metric contract (owner order 2026-10-07) ──────────────────
# The owner order: "add accuracy + f1, change the dataset to pairs we
# already know are the same — give it all the pairs we know are the
# same". The identity decision CSV (data/laya/metrics_pairs.csv, built by
# scripts/laya_metrics_pairs.py over the GROUND-TRUTH pairs) must stage
# with the EXPECTED-metric receipt contract: expected rows + label
# distribution computed from the csv + the gold columns the harvest
# computes accuracy/F1 against, WITHOUT re-deriving.
def test_metric_expectation_contract_rides_identity_receipt(
        tmp_path, monkeypatch):
    _spec(tmp_path, monkeypatch)
    _question_schema(tmp_path, monkeypatch)
    _dataset_fixture(tmp_path, monkeypatch, rows=3)
    _hermetic_staging(monkeypatch)
    receipt = laya_lane.stage_decision_kernel(decision_kind="identity")
    assert receipt["expected_rows"] == 3
    assert receipt["expected_label_distribution"] == {"0": 2, "1": 1}
    assert receipt["metric_expectation"]["accuracy_gold"] == "label"
    assert receipt["metric_expectation"]["f1_gold"] == (
        "identity_claim-vs-true_label")


def test_metric_expectation_computed_from_the_csv_itself(
        tmp_path, monkeypatch):
    _spec(tmp_path, monkeypatch)
    _dataset_fixture(tmp_path, monkeypatch, rows=7)
    receipt = laya_lane.stage_decision_input(
        "kaggle", decision_kind="identity")
    # 7 rows -> labels 0,1,0,1,... = 4 zeros / 3 ones, read from the CSV
    assert receipt["rows"] == 7
    assert receipt["expected_rows"] == 7
    assert receipt["expected_label_distribution"] == {"0": 4, "1": 3}
    # the expectation is COMPUTED, not asserted: re-derive once to compare
    import csv

    with open(receipt["source"], newline="") as handle:
        values = [row["true_label"] for row in csv.DictReader(handle)]
    assert sorted(values) == ["0"] * 4 + ["1"] * 3


def test_metric_expectation_absent_without_true_label_column(
        tmp_path, monkeypatch):
    _spec(tmp_path, monkeypatch)
    _question_schema(tmp_path, monkeypatch)
    _dataset_fixture(tmp_path, monkeypatch)
    _hermetic_staging(monkeypatch)
    receipt = laya_lane.stage_decision_kernel(decision_kind="attribute")
    # the attribute decision csv carries no ground-truth labels: the
    # labeled decision contract keys stay ABSENT, never silently claimed
    assert "expected_label_distribution" not in receipt
    assert "expected_rows" not in receipt
    assert "metric_expectation" not in receipt




# ── the ground-truth pairs metrics builder (scripts/laya_metrics_pairs.py) ──
# Owner order 2026-10-07: the identity dataset becomes the GROUND-TRUTH
# pairs — every track_setup listing pair (owner approved: 564 confirmed-
# same + 12 confirmed-different), composed in the final_validation.csv
# `attribute_pairs` shape, deterministic row order (never a shuffle).
def _builder():
    import importlib.util

    path = (Path(__file__).resolve().parents[1]
            / "scripts/laya_metrics_pairs.py")
    spec = importlib.util.spec_from_file_location("laya_metrics_pairs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pairs_fixture(tmp_path, *, rows=(("S1", "S2", "1", "train"),
                                      ("S2", "S3", "0", "dev"))):
    catalog = tmp_path / "catalog.csv"
    catalog.write_text(
        "sku_id,gtin,attribute\n"
        + "".join(
            f"S{index},4{index:09}"
            f",Volume: 500; Pack Type: Bottle; Flavour: lemon, lime\n"
            for index in range(1, 4)),
        encoding="utf-8")
    pairs = tmp_path / "pairs.csv"
    pairs.write_text(
        "sku_id1,sku_id2,label,split\n"
        + "".join(f"{a},{b},{label},{split}\n" for a, b, label, split in rows),
        encoding="utf-8")
    return pairs, catalog


def _fv_fixture(tmp_path, *, rows=(("15", "16", "1"), ("25", "26", "0"))):
    """rows are (gtin1, gtin2, true_label)."""
    fv = tmp_path / "final_validation.csv"
    fv.write_text(
        "gtin1,gtin2,gtin1_norm,gtin2_norm,true_label,fold,fold_2,"
        "component_id,component_id_2,straddles_fold,endpoint_in_train,"
        "v1_volume,v2_volume,v1_pack,v2_pack,v1_package_type,"
        "v2_package_type,v1_sweetener,v2_sweetener,v1_flavor,v2_flavor,"
        "v1_carbonation,v2_carbonation\n"
        + "".join(
            f"{g1},{g2},{g1.zfill(14)},{g2.zfill(14)},{label},2,2,7,7,False,False,"
            f"[500.0],[500.0],[],[],['bottle'],['bottle'],[],[],"
            f"['lemon'],['lemon'],['still'],['still']\n"
            for g1, g2, label in rows),
        encoding="utf-8")
    return fv


def test_builder_side_composition_mirrors_final_validation_shape():
    builder = _builder()
    side = builder.compose_side(
        "Volume: 1500; Juice Content: 0-2%; Pack Type: Liquid Carton; "
        "Sweetener: no sugar; Flavour: apple, pear; Carbonization: still")
    # the six slice fields in the final_validation list-literal shape:
    # bare tokens, never quoted, sorted; unmeasured fields are ''.
    # Pin the side composition against the frozen header convention the
    # emitted columns carry (v1_* / v2_* over the same field names).
    assert builder.SLICE_FIELDS == (
        "volume", "pack", "package_type", "sweetener", "flavor",
        "carbonation")
    assert side["volume"] == "[1500.0]"
    assert side["package_type"] == "[liquid_carton]"
    assert side["sweetener"] == "[no_sugar]"
    assert side["flavor"] == "[apple, pear]"
    assert side["carbonation"] == "[still]"
    assert side["pack"] == ""  # no pack-count field in the catalog vocabulary
    empty = builder.compose_side("Juice Content: 0-2%")
    assert empty == {field: "" for field in builder.SLICE_FIELDS}


def test_builder_state_is_one_joined_v1_v2_string():
    builder = _builder()
    state = builder.compose_state(
        builder.compose_side("Volume: 500"),
        builder.compose_side("Volume: 500; Flavour: lime"))
    assert state == ("volume: v1=[500.0] v2=[500.0]; pack: v1= v2=; "
                     "package_type: v1= v2=; sweetener: v1= v2=; "
                     "flavor: v1= v2=[lime]; carbonation: v1= v2=")


def test_builder_emits_all_pairs_in_row_order_with_fv_columns(tmp_path):
    builder = _builder()
    pairs, catalog = _pairs_fixture(tmp_path)
    output = tmp_path / "metrics_pairs.csv"
    census = builder.build(pairs_path=pairs, catalog_path=catalog,
                           output=output)
    import csv

    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    # every pair, none invented, pairs-file row order preserved
    assert len(rows) == 2
    assert (rows[0]["gtin1"], rows[0]["gtin2"], rows[0]["true_label"]) == (
        "4000000001", "4000000002", "1")
    assert rows[1]["true_label"] == "0"
    # the SAME columns final_validation.csv carries + the state column
    assert list(rows[0]) == list(builder.FINAL_VALIDATION_COLUMNS) + [
        builder.STATE_COLUMN]
    # normalized gtins left-zero-pad to 14 (the folds.normalize_gtin rule)
    assert (rows[0]["gtin1_norm"], rows[0]["gtin2_norm"]) == (
        "00004000000001", "00004000000002")
    # values that do not exist in the pairs source stay EMPTY, never invented
    for column in ("fold", "fold_2", "component_id", "component_id_2",
                   "straddles_fold", "endpoint_in_train"):
        assert rows[0][column] == ""
    # census: rows + label distribution + 0 missing lookups
    assert census["rows"] == 2
    assert census["label_distribution"] == {"0": 1, "1": 1}
    assert census["missing_sku_lookups"] == 0
    from core.portable_archive import ByteCount

    assert census["size"] == ByteCount(
        output.read_bytes()).total


def test_builder_missing_sku_lookup_fails_loud_before_writing(tmp_path):
    builder = _builder()
    pairs, catalog = _pairs_fixture(tmp_path, rows=(("S9", "S2", "0", "dev"),))
    output = tmp_path / "metrics_pairs.csv"
    with pytest.raises(RuntimeError, match="resolve to no eligible_catalog"):
        builder.build(pairs_path=pairs, catalog_path=catalog, output=output)
    assert not output.exists()


def test_builder_fails_loud_on_bad_labels(tmp_path):
    builder = _builder()
    pairs, catalog = _pairs_fixture(
        tmp_path, rows=(("S1", "S2", "1", "train"), ("S2", "S3", "7", "dev")))
    with pytest.raises(RuntimeError, match="outside \\{0, 1\\}"):
        builder.build(pairs_path=pairs, catalog_path=catalog,
                      output=tmp_path / "metrics_pairs.csv")


def test_builder_fails_loud_on_listing_pairs_header_drift(tmp_path):
    builder = _builder()
    _, catalog = _pairs_fixture(tmp_path)
    pairs = tmp_path / "drifted.csv"
    pairs.write_text("wrong,header\nc,1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="listing_pairs header drifted"):
        builder.build(pairs_path=pairs, catalog_path=catalog,
                      output=tmp_path / "metrics_pairs.csv")


def test_builder_fails_loud_on_final_validation_header_drift(tmp_path):
    builder = _builder()
    pairs, catalog = _pairs_fixture(tmp_path)
    fv = tmp_path / "final_validation.csv"
    fv.write_text("gtin1,gtin2,true_label\n1,2,1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="header drifted"):
        builder.build(pairs_path=pairs, catalog_path=catalog,
                      final_validation_path=fv,
                      output=tmp_path / "metrics_pairs.csv")


def test_builder_shape_proof_matches_final_validation_rows(tmp_path):
    builder = _builder()
    pairs, catalog = _pairs_fixture(tmp_path)
    output = tmp_path / "metrics_pairs.csv"
    builder.build(pairs_path=pairs, catalog_path=catalog, output=output)
    fv = _fv_fixture(tmp_path)
    samples = builder.shape_proof(
        final_validation_path=fv, pairs_path=pairs,
        catalog_path=catalog, proof_samples=3)
    assert samples == []
    # a normalized-gtin match against the real population proves the
    # shape: same six slice fields, v1 first then v2, list-literal sides
    fv2 = _fv_fixture(
        tmp_path,
        rows=(("4000000001", "4000000002", "1"),))
    samples = builder.shape_proof(
        final_validation_path=fv2, pairs_path=pairs,
        catalog_path=catalog, proof_samples=3)
    assert len(samples) == 1
    sample = samples[0]
    assert sample["true_label"] == 1
    assert sample["final_validation_true_label"] == 1
    assert sample["attribute_pairs"].split(";")[0] == (
        "volume: v1=[500.0] v2=[500.0]")


def test_builder_final_validation_columns_match_the_frozen_file():
    """The mirrored header stays pinned to the actual data/final_validation
   .csv file (drift between the two is a fail-loud build error, so the
    pin guards both directions)."""
    builder = _builder()
    fv = builder.FINAL_VALIDATION_PATH
    if not fv.is_file():
        pytest.skip(f"{fv} not built")
    import csv

    with fv.open(newline="") as handle:
        header = next(csv.reader(handle))
    assert header == list(builder.FINAL_VALIDATION_COLUMNS)


# ── the fine-tune corpus builder (scripts/laya_build_dataset.py) ───────────
# Owner order 2026-10-07: "we will finetune laya with the correct dataset".
# The builder emits the JSONL the laya trainer consumes — one case per line:
#   {"state": <str>, "questions": {<schema verbatim>}, "expected": {qid: lbl}}
# STATE cases (one per eligible_catalog row) label `package_state`; PAIR
# cases label `identity_claim` over the ground-truth listing pairs PLUS a
# balanced stratified gate `hard_no` negative sample; gate `fallback` rows
# are quarantined (never corpus). Deterministic (seed 1729).
def _corpus_builder():
    import importlib.util

    path = (Path(__file__).resolve().parents[1]
            / "scripts/laya_build_dataset.py")
    spec = importlib.util.spec_from_file_location("laya_build_dataset", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _corpus_question_fixture(tmp_path):
    """The config/laya.question.json contract in miniature (three questions,
    verbatim into every case)."""
    questions = {
        "attribute_alignment": {"type": "choice", "instructions": "verdict?",
                                "criteria": {"aligned": None}},
        "identity_claim": {"type": "noul", "instructions": "same item?"},
        "package_state": {"type": "noul", "instructions": "has pack?"},
    }
    path = tmp_path / "laya.question.json"
    path.write_text(json.dumps({"schema": "test", "questions": questions}),
                    encoding="utf-8")
    return path, questions


def _corpus_sources_fixture(tmp_path):
    """Hermetic catalog / listing_pairs / gate_results:

      * S1 Volume numeric, S4 numeric Pack Size, S5 'NxM<unit>' multipack
        -> package_state true; S2 Pack Type only, S3 flavour only -> false.
      * two listing positives (train/test) + one listing negative (dev).
      * three joinable gate hard_no rows (two reasons) + one hard_no row
        with a missing gtin (dropped), one fallback (quarantined), one
        proceed (ignored).
    """
    catalog = tmp_path / "catalog.csv"
    catalog.write_text(
        "sku_id,retailer,gtin,attribute\n"
        "S1,r,4000000001,Volume: 500; Pack Type: Bottle\n"
        "S2,r,4000000002,Pack Type: Bottle\n"
        "S3,r,4000000003,Flavour: lime\n"
        "S4,r,4000000004,Pack Size: 6\n"
        "S5,r,4000000005,Pack Type: Can; 24x330ml\n",
        encoding="utf-8")
    pairs = tmp_path / "pairs.csv"
    pairs.write_text(
        "sku_id1,sku_id2,label,split\n"
        "S1,S2,1,train\n"
        "S3,S4,0,dev\n"
        "S4,S5,1,test\n",
        encoding="utf-8")
    gate = tmp_path / "gate.csv"
    gate.write_text(
        "gtin1,gtin2,canon1,canon2,gate_decision,gate_reason,similarity\n"
        "4000000001,4000000003,c1,c2,hard_no,"
        "Critical attribute mismatch: flavor,0.90\n"
        "4000000002,4000000004,c1,c2,hard_no,"
        "Pack blocker: pack size,0.80\n"
        "4000000005,4000000001,c1,c2,hard_no,"
        "Critical attribute mismatch: flavor,0.85\n"
        "9999999999,4000000001,c1,c2,hard_no,Package material mismatch,0.70\n"
        "4000000001,4000000004,c1,c2,fallback,unresolved,0.50\n"
        "4000000002,4000000003,c1,c2,proceed,ok,0.99\n",
        encoding="utf-8")
    return catalog, pairs, gate


def test_corpus_package_state_rule_is_documented_and_explicit():
    builder = _corpus_builder()
    # measured unit volume (finite numeric Volume:) -> true
    assert builder.package_state("Volume: 500") is True
    assert builder.package_state("Volume: 1500; Pack Type: Carton") is True
    assert builder.package_state("Volume: 1.5") is True
    # a Pack Type alone is a FORM, not a quantity -> false
    assert builder.package_state("Pack Type: Bottle") is False
    assert builder.package_state("Pack Type: Can") is False
    # explicit numeric pack-count key -> true
    assert builder.package_state("Pack Size: 6") is True
    assert builder.package_state("Number of items: 12") is True
    # an 'NxM<unit>' multipack token -> true
    assert builder.package_state("Pack Type: Can; 24x330ml") is True
    # no package evidence at all -> false
    assert builder.package_state("Flavour: lime") is False
    # the rule is published in the receipt (no undocumented scoring)
    assert "Pack Type" in builder.PACKAGE_STATE_RULE
    assert "Volume" in builder.PACKAGE_STATE_RULE


def test_corpus_build_emits_laya_jsonl_counts_and_receipt(tmp_path):
    builder = _corpus_builder()
    catalog, pairs, gate = _corpus_sources_fixture(tmp_path)
    question_path, questions = _corpus_question_fixture(tmp_path)
    out = tmp_path / "laya"
    receipt = builder.build(catalog_path=catalog, pairs_path=pairs,
                            gate_path=gate, question_path=question_path,
                            output_dir=out, seed=1729, hard_no_cap=1000)

    counts = receipt["counts"]
    # STATE: one case per catalog row; the explicit rule splits 3/2
    assert counts["state_cases"] == 5
    assert counts["state_package_state_true"] == 3
    assert counts["state_package_state_false"] == 2
    # the Pack-Type-only diagnostic never leaks into the gold
    assert counts["state_pack_type_only_no_quantity"] == 1
    # PAIRS: all ground-truth listing pairs, split preserved
    assert counts["listing_pairs_positive"] == 2
    assert counts["listing_pairs_negative"] == 1
    # GATE: joinable hard_no negatives, balanced against the positives;
    # the missing-gtin hard_no is dropped+counted, fallback quarantined
    assert counts["gate_hard_no_available"] == 3
    assert counts["gate_hard_no_sampled"] == 1
    assert counts["gate_fallback_quarantined"] == 1
    assert counts["dropped_missing"]["hard_no"] == 1
    assert counts["dropped_missing"]["total"] == 1
    assert counts["identity_positive_total"] == 2
    assert counts["identity_negative_total"] == 2  # balanced
    # the published rule + seed ride the receipt
    assert receipt["seed"] == 1729
    assert receipt["package_state_rule"] == builder.PACKAGE_STATE_RULE

    # every split jsonl: laya case shape, questions verbatim, valid labels
    from core.portable_archive import ByteCount

    for key in ("train", "dev", "test"):
        path = out / f"{key}.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == receipt["split_sizes"][key]
        assert receipt["size"][f"{key}.jsonl"] == ByteCount(
            path.read_bytes()).total
        for line in lines:
            record = json.loads(line)
            assert set(record) == {"state", "questions", "expected",
                                   "difficulty_slice", "gate_reason",
                                   "attribute"}
            assert isinstance(record["state"], str) and record["state"]
            assert record["questions"] == questions
            # only golds the fixture schema declares; attribute_alignment is
            # deliberately unlabelled (the membership gate skips undeclared
            # questions, and the fixture declares no new pair questions)
            assert set(record["expected"]) <= {"package_state",
                                               "identity_claim"}
            for qid, label in record["expected"].items():
                assert record["questions"][qid]["type"] == "noul"
                assert label in ("true", "false")
    # balanced identity gold end to end
    identity = [
        json.loads(line)["expected"]["identity_claim"]
        for key in ("train", "dev", "test")
        for line in (out / f"{key}.jsonl").read_text().splitlines()
        if "identity_claim" in json.loads(line)["expected"]]
    assert identity.count("true") == 2
    assert identity.count("false") == 2
    # the listing-pair split is preserved exactly
    assert receipt["split_counts"]["train"]["listing_positive"] == 1
    assert receipt["split_counts"]["dev"]["listing_negative"] == 1
    assert receipt["split_counts"]["test"]["listing_positive"] == 1
    # one state package_state gold per state line
    package = [
        json.loads(line)["expected"]["package_state"]
        for key in ("train", "dev", "test")
        for line in (out / f"{key}.jsonl").read_text().splitlines()
        if "package_state" in json.loads(line)["expected"]]
    assert package.count("true") == 3
    assert package.count("false") == 2
    assert receipt["size"]["train.jsonl"]  # present


def test_corpus_fallback_is_quarantined_never_in_corpus(tmp_path):
    builder = _corpus_builder()
    catalog, pairs, gate = _corpus_sources_fixture(tmp_path)
    question_path, _ = _corpus_question_fixture(tmp_path)
    out = tmp_path / "laya"
    builder.build(catalog_path=catalog, pairs_path=pairs, gate_path=gate,
                  question_path=question_path, output_dir=out, seed=1729)
    import csv

    unknown = out / "unknown_pairs.csv"
    with unknown.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["gate_decision"] == "fallback"
    # the quarantined row carries its join flags + composed state
    assert rows[0]["gtin1_in_catalog"] == "True"
    assert rows[0]["gtin2_in_catalog"] == "True"
    assert rows[0]["attribute_pairs"]
    # NEVER in the corpus: no fallback state string appears in any split
    body = "".join((out / f"{key}.jsonl").read_text()
                   for key in ("train", "dev", "test"))
    assert rows[0]["attribute_pairs"] not in body


def test_corpus_build_is_deterministic(tmp_path):
    builder = _corpus_builder()
    catalog, pairs, gate = _corpus_sources_fixture(tmp_path)
    question_path, _ = _corpus_question_fixture(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    receipt_one = builder.build(catalog_path=catalog, pairs_path=pairs,
                                gate_path=gate, question_path=question_path,
                                output_dir=first, seed=1729)
    receipt_two = builder.build(catalog_path=catalog, pairs_path=pairs,
                                gate_path=gate, question_path=question_path,
                                output_dir=second, seed=1729)
    # a rerun reproduces every byte (seed + deterministic split allocation)
    assert receipt_one["size"] == receipt_two["size"]
    for name in ("train.jsonl", "dev.jsonl", "test.jsonl",
                 "unknown_pairs.csv", "receipt.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_corpus_fails_loud_on_listing_pairs_header_drift(tmp_path):
    builder = _corpus_builder()
    catalog, _, gate = _corpus_sources_fixture(tmp_path)
    question_path, _ = _corpus_question_fixture(tmp_path)
    drifted = tmp_path / "drifted.csv"
    drifted.write_text("wrong,header\nc,1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="listing_pairs header drifted"):
        builder.build(catalog_path=catalog, pairs_path=drifted,
                      gate_path=gate, question_path=question_path,
                      output_dir=tmp_path / "out")


def test_corpus_fails_loud_on_missing_pair_sku(tmp_path):
    builder = _corpus_builder()
    catalog, _, gate = _corpus_sources_fixture(tmp_path)
    question_path, _ = _corpus_question_fixture(tmp_path)
    pairs = tmp_path / "pairs.csv"
    pairs.write_text("sku_id1,sku_id2,label,split\nS1,S9,0,dev\n",
                     encoding="utf-8")
    with pytest.raises(RuntimeError, match="resolve to no catalog row"):
        builder.build(catalog_path=catalog, pairs_path=pairs, gate_path=gate,
                      question_path=question_path, output_dir=tmp_path / "out")


def test_corpus_fails_loud_on_catalog_missing_columns(tmp_path):
    builder = _corpus_builder()
    _, pairs, gate = _corpus_sources_fixture(tmp_path)
    question_path, _ = _corpus_question_fixture(tmp_path)
    catalog = tmp_path / "catalog.csv"
    catalog.write_text("sku_id,attribute\nS1,Volume: 500\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing required columns"):
        builder.build(catalog_path=catalog, pairs_path=pairs, gate_path=gate,
                      question_path=question_path, output_dir=tmp_path / "out")


# ── standalone archive/hash boundaries (Bundle migration) ──────────────────
def test_fetch_plan_is_json_serializable_end_to_end(tmp_path, monkeypatch):
    """`--fetch` prints its plan with json.dumps: the traceability documents the
    plan carries must be stored in JSON form. A pydantic model left in the plan
    made the print raise `TypeError: Object of type TraceabilityReport is not
    JSON serializable`."""
    import io
    import subprocess
    import tarfile

    _spec(tmp_path, monkeypatch)
    report_path = (Path(__file__).resolve().parents[1] / "results/laya_lane"
                   "/kaggle/finetune/output/checkpoint/train_report.json")
    if not report_path.is_file():
        pytest.skip(f"real artifact not present: {report_path}")
    digest = "6" * 64
    receipt = {"corpus_size": {"train.jsonl": digest},
               "output_dir": "/kaggle/working/checkpoint"}

    def fake_run(args, **kwargs):
        # `kaggle kernels output <slug> -p <stage>`: stage the kernel handoff
        # archive offline (no CLI is ever spawned).
        destination = Path(args[args.index("-p") + 1])
        destination.mkdir(parents=True, exist_ok=True)
        with tarfile.open(destination / "laya_attribute.tar.gz", "w:gz") as tar:
            tar.add(report_path, arcname="train_report.json")
            body = json.dumps(receipt).encode()
            info = tarfile.TarInfo("laya_attribute.receipt.json")
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(laya_lane.subprocess, "run", fake_run)
    plan = laya_lane.collect_kaggle_result("attribute", "owner/slug",
                                          execute=True)
    document = plan["traceability"]["train_report.json"]
    assert isinstance(document, dict)  # JSON form, never a pydantic model
    assert document["provenance"]["digests"]["corpus_size"] == digest
    # the plan is exactly what main() prints
    assert json.loads(json.dumps(plan)) == plan


def test_fetch_emits_the_record_grain_traceability_artifact(tmp_path, monkeypatch):
    """A fetched ``<kind>.decisions.jsonl`` becomes a WRITTEN record-derived report.

    Falsified 2026-10-08: ``decision_csv_records`` / ``eval_case_records`` /
    ``records_traceability`` had no production caller at all, and `--fetch`
    never wrote a traceability artifact. The per-row identity grain now goes
    through the adapters, is validated against the shared contract and lands
    through the declared ``traceability_report`` layout.
    """
    import io
    import subprocess
    import tarfile

    _spec(tmp_path, monkeypatch)
    # the schema this box STAGED before the push (the fetched grain needs it)
    schema_dir = laya_lane.staging_dir() / "kaggle" / "question"
    schema_dir.mkdir(parents=True, exist_ok=True)
    (schema_dir / laya_lane.QUESTION_SCHEMA_FILE).write_text(
        json.dumps({"questions": {"identity_claim": {"type": "noul"}}}),
        encoding="utf-8")
    monkeypatch.setitem(common._BINDING_ROOTS, "results", tmp_path)
    monkeypatch.setattr(common, "trace_artifact", lambda *args, **kwargs: None)
    receipt = {"gpu_kind": "identity", "batch_size": 4, "split": "dev",
               "decision_csv_size": "8" * 64}

    def fake_run(args, **kwargs):
        destination = Path(args[args.index("-p") + 1])
        destination.mkdir(parents=True, exist_ok=True)
        with tarfile.open(destination / "laya_identity.tar.gz", "w:gz") as tar:
            for name, body in (
                    ("laya_identity.receipt.json",
                     json.dumps(receipt).encode()),
                    ("identity.decisions.jsonl",
                     b"".join(json.dumps(
                         {"_row": {"gtin1": str(index), "gtin2": str(index + 1)}})
                         .encode() + b"\n" for index in range(3)))):
                info = tarfile.TarInfo(name)
                info.size = len(body)
                tar.addfile(info, io.BytesIO(body))
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(laya_lane.subprocess, "run", fake_run)
    plan = laya_lane.collect_kaggle_result("identity", "owner/slug", execute=True)
    assert plan["decision_rows"] == 3
    document = plan["traceability"]["record_grain"]
    assert document["coverage"]["records_total"] == 3
    assert document["coverage"]["by_source"] == {"identity_decision_csv": 3}
    assert document["coverage"]["by_dimension"]["split"] == {"dev": 3}
    assert document["overall"] is None  # identity-only: no metrics measured
    artifact = Path(plan["traceability_artifacts"]["record_grain"])
    assert artifact.is_file()
    assert json.loads(artifact.read_text(encoding="utf-8")) == document
    assert json.loads(json.dumps(plan)) == plan


def test_base_model_archive_seals_as_an_inputs_bundle(tmp_path, monkeypatch):
    """The base-model dataset archive seals through the shared writer.

    The receipt's size token IS the sealed digest (the archive is never read
    back), and the archive loads as an `inputs` Bundle while the
    `rl_agent_config.json` layout `extract_base_model` resolves stays intact.
    """
    from core.bundle import Bundle, BundleRole

    _spec(tmp_path, monkeypatch)
    source = tmp_path / "checkpoint"
    source.mkdir()
    (source / "rl_agent_config.json").write_text("{}", encoding="utf-8")
    (source / "weights.bin").write_bytes(b"weights")

    receipt = laya_lane.package_base_model(
        source_dir=source, dataset_slug="owner/base",
        archive_name="convaiinnovations-laya.tar.zst",
        member_name="convaiinnovations-laya", output_dir=tmp_path / "stage")

    archive = tmp_path / "stage" / "convaiinnovations-laya.tar.zst"
    assert receipt["bundle_role"] == "inputs"
    assert receipt["manifest"] == laya_lane.BASE_MODEL_MANIFEST_FILE
    assert receipt["size"] == laya_lane.file_size(archive)
    handle = Bundle.load(archive, BundleRole.inputs,
                         manifest_name=laya_lane.BASE_MODEL_MANIFEST_FILE)
    assert "convaiinnovations-laya/rl_agent_config.json" in handle.members()
    assert "convaiinnovations-laya/weights.bin" in handle.members()
    # the kaggle dataset shape lands beside the archive
    assert (tmp_path / "stage" / "dataset-metadata.json").is_file()


# ── finetune session id: kernel self-report -> follower -> --session-id ─────

def _finetune_corpus(tmp_path):
    """The JSONL splits + receipt `stage_finetune_kernel` requires (hermetic)."""
    corpus = tmp_path / "data/laya"
    corpus.mkdir(parents=True, exist_ok=True)
    for name in ("train.jsonl", "dev.jsonl", "test.jsonl"):
        (corpus / name).write_text('{"state": "s"}\n', encoding="utf-8")
    (corpus / "receipt.json").write_text('{"seed": 1729}\n', encoding="utf-8")
    return corpus


def _rendered_finetune_script(tmp_path, monkeypatch) -> str:
    _spec(tmp_path, monkeypatch)
    _finetune_corpus(tmp_path)
    _hermetic_staging(monkeypatch)
    receipt = laya_lane.stage_finetune_kernel(run_tag="laya_test")
    script = (Path(receipt["staged"]) / laya_lane.FINETUNE_CODE_FILE).read_text()
    # the staged payload must carry NO leftover marker of any kind
    assert "@SESSION_REPORT@" not in script
    return script


def _rendered_session_env(script):
    """The rendered kernel's OWN `report_kernel_session` + `session_env`.

    Compiled from the staged text (not from the template constants), so the
    test exercises exactly the bytes a Kaggle session executes.
    """
    import os

    wanted = {"report_kernel_session", "session_env"}
    nodes = [node for node in ast.walk(ast.parse(script))
             if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert {node.name for node in nodes} == wanted
    namespace = {"os": os}
    exec(compile(ast.fix_missing_locations(
        ast.Module(body=nodes, type_ignores=[])), "<rendered-finetune>",
        "exec"), namespace)
    return namespace["session_env"]


def test_finetune_session_self_report_reaches_the_recorded_session_id(
        tmp_path, monkeypatch, capsys):
    """Item 3 end to end: kernel self-report -> follower -> recorded session id.

    The finetune kernel prints the container's session id at boot; the
    host-side SSE follower persists it to logs/kaggle/<kernel>.session_id; the
    `--session-id` reader (and the verified stop it feeds) reads exactly that
    file. Kaggle sets no KAGGLE_KERNEL_RUN_ID/KAGGLE_SESSION_ID, and the stream
    URL carries no id on the current SDK, so this line is the only source.
    """
    import types

    import kagglesdk.kaggle_client
    from cli import kaggle_lane
    from core.schemas import KaggleSpec

    script = _rendered_finetune_script(tmp_path, monkeypatch)
    session_env = _rendered_session_env(script)

    container = "kaggle_er-laya-finetune-123456789-webtier"
    monkeypatch.setenv("KAGGLE_CONTAINER_NAME", container)
    capsys.readouterr()
    payload = session_env()
    printed = capsys.readouterr().out
    assert payload == {"KAGGLE_CONTAINER_NAME": container,
                       "session_id": "123456789"}
    assert "[kaggle-session] session_id=123456789" in printed

    # the host side of the crossing: the follower reads that line off the SSE
    # stream and persists it (lane_logs_dir resolves under the tmp TRAIN_ROOT).
    monkeypatch.setattr(kaggle_lane, "_spec",
                        lambda: KaggleSpec(staging_dir="kaggle_stage"))
    monkeypatch.setattr(kaggle_lane, "TRAIN_ROOT", tmp_path)

    class Stream:
        def iter_lines(self):
            yield ('data: {"stream_name":"stdout","time":1,"data":'
                   + json.dumps(printed) + "}")

    fake_api = types.SimpleNamespace(
        get_kernel_session_logs_stream=lambda request: Stream())
    monkeypatch.setattr(
        kagglesdk.kaggle_client, "KaggleClient",
        lambda env: types.SimpleNamespace(kernels=types.SimpleNamespace(
            kernels_api_client=fake_api)))

    kaggle_lane.stream_kernel_logs("fbarulli/er-laya-finetune")
    session_file = (tmp_path / "logs/kaggle/er-laya-finetune.session_id")
    assert session_file.read_text().strip() == "123456789"

    # the readers the CLI surfaces use: `--session-id` and the stop target
    assert laya_lane.recorded_session_id("fbarulli/er-laya-finetune") == 123456789
    assert laya_lane.container_session_id(container) == 123456789
    # and the SDK cancel consumes exactly this id
    plan = laya_lane.stop_kaggle_kernel("fbarulli/er-laya-finetune", execute=False)
    assert plan["kernel"] == "fbarulli/er-laya-finetune"


def test_recorded_session_id_is_none_without_a_launch(tmp_path, monkeypatch):
    """No recorded id -> None (never a fabricated session), offline."""
    from cli import kaggle_lane
    from core.schemas import KaggleSpec

    monkeypatch.setattr(kaggle_lane, "_spec",
                        lambda: KaggleSpec(staging_dir="kaggle_stage"))
    monkeypatch.setattr(kaggle_lane, "TRAIN_ROOT", tmp_path)
    assert laya_lane.recorded_session_id("fbarulli/er-laya-finetune") is None
    assert laya_lane.container_session_id("kaggle_not-a-container") is None
