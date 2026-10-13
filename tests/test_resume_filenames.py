"""tests/test_resume_filenames.py — resume-artifact filenames read the BundleSpec SSOT.

The checkpoint resume-artifact filename set and the trainer-state keys were once
hardcoded in four places in ``training/training.py``. Every one of them must now
resolve through ``training_cfg().bundle`` (``core.schemas.BundleSpec``):

  * ``resume_only_filenames``    the native-HF resume state (optimizer/scheduler/
                                 scaler/RNG/training-args);
  * ``trainer_state_file``       ``trainer_state.json`` (NOT a resume-only member);
  * ``trainer_best_key``         the selected-checkpoint key inside trainer state;
  * ``colab.checkpoint_manifest_name``  the checkpoint manifest filename.

The four sites are ``_files_block``, ``_resume_block``, ``on_save`` and
``_save_checkpoint`` (the last two share ``_required_resume_filenames``). These
tests import the helpers and compare their output against the config values, and
pin that the config values themselves are byte-identical to the historical
literals so existing checkpoints still load.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.common import training_cfg
from training import training as training_mod


def _bundle():
    return training_cfg().bundle


def test_config_values_are_byte_identical_to_the_historical_filenames() -> None:
    """A rename must stay byte-identical: the SSOT spells the same names."""
    bundle = _bundle()
    assert set(bundle.resume_only_filenames) == {
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "training_args.bin",
        "scaler.pt",
    }
    assert bundle.trainer_state_file == "trainer_state.json"
    assert bundle.trainer_best_key == "best_model_checkpoint"
    assert training_cfg().colab.checkpoint_manifest_name == "checkpoint_manifest.json"


def test_resume_component_filenames_resolve_to_the_config_tuple() -> None:
    """Every resume component name resolves to (and from) the SSOT tuple."""
    bundle = _bundle()
    for name in (
        "optimizer.pt",
        "scheduler.pt",
        "scaler.pt",
        "rng_state.pth",
        "training_args.bin",
    ):
        assert name in bundle.resume_only_filenames
        assert training_mod._resume_component_filename(name) == name


def test_resume_component_filename_reports_instead_of_inventing() -> None:
    """A filename the config does not declare is reported, never hardcoded."""
    with pytest.raises(RuntimeError, match="resume component filename"):
        training_mod._resume_component_filename("not_in_config.pt")


def test_files_block_reads_the_bundle_spec(tmp_path: Path) -> None:
    """Site 1: the manifest files block resolves every name through config."""
    bundle = _bundle()
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"w")

    block = training_mod._CheckpointPublisher._files_block(
        checkpoint, optimizer=object(), scheduler=object(), scaler=object()
    )
    for key in ("optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict",
                "rng_state", "training_args"):
        assert block[key] in bundle.resume_only_filenames, (key, block[key])
    assert block["trainer_state"] == bundle.trainer_state_file
    assert block["model_state_dict"] == ["model.safetensors"]


def test_files_block_omits_absent_optimizer_scheduler_scaler(tmp_path: Path) -> None:
    """Optional resume components stay None when their owner is absent."""
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    (checkpoint / "pytorch_model.bin").write_bytes(b"w")
    block = training_mod._CheckpointPublisher._files_block(
        checkpoint, optimizer=None, scheduler=None, scaler=None
    )
    assert block["optimizer_state_dict"] is None
    assert block["scheduler_state_dict"] is None
    assert block["scaler_state_dict"] is None
    assert block["rng_state"] in _bundle().resume_only_filenames


def test_resume_block_reads_the_bundle_spec() -> None:
    """Site 2: the resume component map resolves every name through config."""
    bundle = _bundle()
    resume = training_mod._CheckpointPublisher._resume_block()["native_hf_resume"]
    assert resume["trainer_state"] == bundle.trainer_state_file
    assert resume["trainer_control"] == f"{bundle.trainer_state_file}:control"
    for key in ("training_args", "optimizer", "scheduler", "rng"):
        assert resume[key] in bundle.resume_only_filenames, (key, resume[key])


def test_required_resume_filenames_match_the_resume_contract() -> None:
    """Sites 3 and 4: the preflight set is exactly the historical five names."""
    bundle = _bundle()
    manifest = training_cfg().colab.checkpoint_manifest_name
    required = training_mod._required_resume_filenames()

    assert required == (
        training_mod._resume_component_filename("optimizer.pt"),
        training_mod._resume_component_filename("scheduler.pt"),
        training_mod._resume_component_filename("rng_state.pth"),
        manifest,
        bundle.trainer_state_file,
    )
    for name in required[:3]:
        assert name in bundle.resume_only_filenames
    # scaler.pt / training_args.bin are resume-only but NOT in the preflight set
    assert set(required) == {"optimizer.pt", "scheduler.pt", "rng_state.pth",
                             manifest, bundle.trainer_state_file}


def test_trainer_best_key_resolves_to_the_bundle_spec() -> None:
    """Site 4: the selected-checkpoint key inside trainer state is config-owned."""
    assert training_mod._trainer_best_key() == _bundle().trainer_best_key
    assert training_mod._trainer_state_filename() == _bundle().trainer_state_file
