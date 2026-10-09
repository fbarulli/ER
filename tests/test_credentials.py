"""Credential source SSOT: the owner class and its config block.

Pins the contract the API-key behavior was moved onto (owner directive):
a config-declared env file read once, process-environment precedence,
fail-loud on a missing required key, and no value ever leaking through a
repr or a log line.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from core.credentials import (
    CredentialError,
    CredentialsSpec,
    CredentialStore,
)


def _store(tmp_path: Path, body: str, **spec_updates) -> CredentialStore:
    """A store over a real temp env file resolved against ``tmp_path``."""
    (tmp_path / "secrets.env").write_text(body, encoding="utf-8")
    spec = CredentialsSpec(env_file="secrets.env", **spec_updates)
    return CredentialStore.from_config(spec, root=tmp_path)


def test_load_from_declared_env_file_resolves_against_root(tmp_path, monkeypatch):
    for name in ("KAGGLE_API_KEY", "WANDB_API_KEY", "HF_TOKEN", "KAGGLE_USERNAME"):
        monkeypatch.delenv(name, raising=False)
    store = _store(tmp_path, 'KAGGLE_API_KEY="from-file"\nWANDB_API_KEY=wandb-file\n')
    assert store.resolve("kaggle_api_key").get_secret_value() == "from-file"
    assert store.resolve("wandb_api_key").get_secret_value() == "wandb-file"
    # a key absent from both file and env is optional-None, not an error
    assert store.resolve_optional("hf_token") is None


def test_missing_env_file_is_empty_not_an_error(tmp_path, monkeypatch):
    for name in ("KAGGLE_API_KEY", "WANDB_API_KEY", "HF_TOKEN", "KAGGLE_USERNAME"):
        monkeypatch.delenv(name, raising=False)
    store = CredentialStore.from_config(CredentialsSpec(env_file="nope.env"),
                                        root=tmp_path)
    assert store.resolve_optional("wandb_api_key") is None


def test_process_environment_wins_over_the_file(tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_API_KEY", "from-env")
    store = _store(tmp_path, "KAGGLE_API_KEY=from-file\n")
    assert store.resolve("kaggle_api_key").get_secret_value() == "from-env"


def test_missing_required_key_fails_loud_with_traceback(tmp_path, monkeypatch, caplog):
    for name in ("KAGGLE_API_KEY", "KAGGLE_USERNAME", "WANDB_API_KEY", "HF_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    caplog.set_level(logging.ERROR, logger="core.credentials")
    store = _store(tmp_path, "")
    with pytest.raises(CredentialError, match="empty or unset"):
        store.resolve("kaggle_api_key")
    assert "credential" in caplog.text
    assert "Traceback" in caplog.text


def test_unknown_logical_name_is_named(tmp_path):
    store = _store(tmp_path, "KAGGLE_API_KEY=x\n")
    with pytest.raises(KeyError, match="unknown credential"):
        store.resolve("not_a_key")


def test_secret_is_redacted_and_never_logged(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("WANDB_API_KEY", "super-secret-value")
    caplog.set_level(logging.DEBUG)
    store = _store(tmp_path, "")
    secret = store.resolve("wandb_api_key")
    assert secret.get_secret_value() == "super-secret-value"
    assert "super-secret-value" not in repr(secret)
    assert "**********" in repr(secret)
    assert "super-secret-value" not in caplog.text


def test_apply_to_environment_fills_absent_and_keeps_live(tmp_path, monkeypatch):
    for name in ("WANDB_API_KEY", "HF_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("WANDB_API_KEY", "live-w")
    store = CredentialStore.from_mapping(
        CredentialsSpec(), {"WANDB_API_KEY": "file-w", "HF_TOKEN": "file-h"})
    store.apply_to_environment()
    assert os.environ["WANDB_API_KEY"] == "live-w"
    assert os.environ["HF_TOKEN"] == "file-h"


def test_stdlib_parser_handles_export_quotes_and_comments():
    text = '# comment\nexport A=1\nB="two"\nC=\'three\'\n\nno_equals\n'
    assert CredentialStore._parse_env_text(text) == {
        "A": "1", "B": "two", "C": "three"}


def test_config_block_is_optional_and_defaults_preserve_behavior():
    spec = CredentialsSpec()
    assert spec.env_file == "../.env"
    assert spec.keys.kaggle_api_key == "KAGGLE_API_KEY"
    assert spec.keys.wandb_api_key == "WANDB_API_KEY"


def test_training_config_without_the_block_still_validates():
    """Additive: the shipped training.yaml minus credentials still loads."""
    import yaml

    from core.common import TRAINING_CONFIG_PATH
    from core.schemas import TrainingConfig

    raw = yaml.safe_load(TRAINING_CONFIG_PATH.read_text(encoding="utf-8"))
    raw.pop("credentials", None)
    config = TrainingConfig.model_validate(raw)
    assert config.credentials == CredentialsSpec()


def test_write_credentials_reads_through_the_owner(tmp_path, monkeypatch):
    """The kaggle lane's credential path is the owner, not raw os.environ."""
    import json

    from cli import kaggle_lane
    from core.schemas import KaggleSpec

    monkeypatch.setattr(kaggle_lane, "_spec",
                        lambda: KaggleSpec(username="owner"))
    monkeypatch.setattr(kaggle_lane, "TRAIN_ROOT", tmp_path)
    target = tmp_path / "home" / ".kaggle" / "kaggle.json"
    monkeypatch.setattr(kaggle_lane, "CREDENTIALS_PATH", target)
    monkeypatch.setattr(kaggle_lane, "ACCESS_TOKEN_PATH",
                        tmp_path / "home" / ".kaggle" / "access_token")
    monkeypatch.setenv("KAGGLE_API_KEY", "token-owner")
    plan = kaggle_lane.write_credentials(execute=True)
    assert plan["key_present"] is True
    assert json.loads(target.read_text()) == {"username": "owner", "key": "token-owner"}
