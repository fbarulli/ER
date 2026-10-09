"""src/core/credentials.py — the credential-source SSOT and its owner.

``config/training.yaml`` ``credentials:`` (validated by :class:`CredentialsSpec`)
declares the ONE env file and the logical-name -> environment-variable map.
:class:`CredentialStore` owns reading that file (python-dotenv when present,
else a documented stdlib parser), applying process-environment precedence, and
handing out :class:`~pydantic.SecretStr` values.

Security contract: never print or log a value, never fall back to a silent
empty credential, never resolve by symlink or ad-hoc shell sourcing. A missing
REQUIRED key fails loud and records the full traceback.
"""

from __future__ import annotations

import logging
import os
import traceback
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

_log = logging.getLogger(__name__)

#: Default env-file path, relative to the repository root. The backing secrets
#: file lives one directory ABOVE the project root, so the fragment is "../.env".
DEFAULT_ENV_FILE = "../.env"


class CredentialError(RuntimeError):
    """A required credential is absent; the owner logs the full traceback."""


class CredentialKeysSpec(BaseModel):
    """Logical credential name -> environment-variable name (never a value)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kaggle_api_key: str = "KAGGLE_API_KEY"
    kaggle_username: str = "KAGGLE_USERNAME"
    wandb_api_key: str = "WANDB_API_KEY"
    hf_token: str = "HF_TOKEN"


class CredentialsSpec(BaseModel):
    """training.credentials — the credential source SSOT (additive).

    Declares only the SOURCE: the env-file path (repo-root-relative; resolved
    against the repository root, never the cwd) and the logical-name ->
    environment-variable map. It also owns the two Kaggle CLI credential paths
    (the materialized ``kaggle.json`` and the stale ``access_token`` guard), so
    the kaggle lane keeps ONE credential registry. A config without this block
    loads with these defaults, preserving current behavior.
    """

    model_config = ConfigDict(extra="forbid")

    env_file: str = DEFAULT_ENV_FILE
    kaggle_credentials_file: str = ".kaggle/kaggle.json"
    kaggle_access_token_file: str = ".kaggle/access_token"
    keys: CredentialKeysSpec = Field(default_factory=CredentialKeysSpec)

    @model_validator(mode="after")
    def _paths_are_home_relative_fragments(self) -> CredentialsSpec:
        if not self.env_file.strip():
            raise ValueError("credentials.env_file must be non-empty")
        for name in ("kaggle_credentials_file", "kaggle_access_token_file"):
            value = Path(getattr(self, name))
            if value.is_absolute() or ".." in value.parts or not value.parts:
                raise ValueError(
                    f"credentials.{name} must be a home-relative path "
                    f"fragment: {getattr(self, name)!r}"
                )
        return self


class CredentialStore:
    """Read-only repository of the config-declared secrets, read once.

    Factory construction: :meth:`from_config` for production (reads the declared
    env file) and :meth:`from_mapping` for tests/embedded callers. Precedence is
    explicit and documented: the live process environment WINS over the declared
    env file. Values leave as :class:`~pydantic.SecretStr` and are never logged
    or printed.
    """

    def __init__(self, spec: CredentialsSpec, *,
                 file_values: Mapping[str, str]) -> None:
        self._spec = spec
        self._values = dict(file_values)

    @classmethod
    def from_config(cls, spec: CredentialsSpec | None = None, *,
                    root: Path | None = None) -> CredentialStore:
        """Build from the config SSOT, reading the declared env file once.

        ``root`` is the repository root the declared ``env_file`` resolves
        against; it defaults to the discovered project root. An alternate
        checkout or a test passes its own root so the lookup follows it.
        """
        if spec is None:
            from core.common import training_cfg

            spec = training_cfg().credentials
        base = Path(root) if root is not None else cls._project_root()
        path = cls._resolve_env_file(base, spec)
        return cls(spec, file_values=cls._read_env_file(path))

    @classmethod
    def from_mapping(cls, spec: CredentialsSpec,
                     values: Mapping[str, str]) -> CredentialStore:
        """Build over an in-memory env mapping (tests, embedded callers)."""
        return cls(spec, file_values=values)

    @property
    def spec(self) -> CredentialsSpec:
        """The validated credential-source contract this store resolves."""
        return self._spec

    def resolve(self, logical_name: str) -> SecretStr:
        """One REQUIRED secret by logical name; fail loud when absent."""
        env_var = self._env_var_for(logical_name)
        try:
            return self._lookup(env_var)
        except CredentialError:
            _log.error("credential %r (%s) is missing; full traceback:\n%s",
                       logical_name, env_var, traceback.format_exc())
            raise

    def resolve_optional(self, logical_name: str) -> SecretStr | None:
        """One optional secret by logical name, or None (never fails)."""
        return self._lookup_optional(self._env_var_for(logical_name))

    def resolve_env(self, env_var: str) -> SecretStr:
        """One REQUIRED secret by raw environment-variable name."""
        try:
            return self._lookup(env_var)
        except CredentialError:
            _log.error("credential %s is missing; full traceback:\n%s",
                       env_var, traceback.format_exc())
            raise

    def resolve_env_optional(self, env_var: str) -> SecretStr | None:
        """One optional secret by raw environment-variable name."""
        return self._lookup_optional(env_var)

    def apply_to_environment(self) -> None:
        """Export the declared file values for child processes.

        A live process value is never overwritten (it wins). Applied silently:
        nothing is logged.
        """
        for env_var in self._spec.keys.model_dump().values():
            if os.environ.get(env_var, "").strip():
                continue
            value = self._values.get(env_var, "")
            if value:
                os.environ[env_var] = value

    # ── internals ──────────────────────────────────────────────────────────
    def _env_var_for(self, logical_name: str) -> str:
        mapping = self._spec.keys.model_dump()
        if logical_name not in mapping:
            raise KeyError(
                f"unknown credential {logical_name!r}; declared logical names: "
                f"{sorted(mapping)}")
        return mapping[logical_name]

    def _lookup(self, env_var: str) -> SecretStr:
        value = self._lookup_optional(env_var)
        if value is None:
            raise CredentialError(
                f"credential {env_var!r} is empty or unset; declare it in the "
                "config credentials env file or the process environment "
                "(never commit it)")
        return value

    def _lookup_optional(self, env_var: str) -> SecretStr | None:
        live = os.environ.get(env_var, "").strip()
        if live:
            return SecretStr(live)
        file_value = self._values.get(env_var, "").strip()
        if file_value:
            return SecretStr(file_value)
        return None

    @staticmethod
    def _project_root() -> Path:
        from core.common import TRAIN_ROOT

        return TRAIN_ROOT

    @staticmethod
    def _resolve_env_file(root: Path, spec: CredentialsSpec) -> Path:
        fragment = Path(spec.env_file)
        return fragment if fragment.is_absolute() else (root / fragment).resolve()

    @staticmethod
    def _read_env_file(path: Path) -> dict[str, str]:
        if not path.is_file():
            return {}
        try:
            from dotenv import dotenv_values

            return {key: value for key, value in dotenv_values(path).items()
                    if value is not None}
        except ImportError:
            _log.warning("python-dotenv absent; parsing %s with the stdlib parser",
                         path)
            return CredentialStore._parse_env_text(path.read_text(encoding="utf-8"))

    @staticmethod
    def _parse_env_text(text: str) -> dict[str, str]:
        """Minimal stdlib KEY=VALUE parser used when python-dotenv is absent.

        Skips blanks/comments, tolerates an ``export `` prefix and strips one
        pair of surrounding quotes. Not a full dotenv implementation:
        shortcut: no interpolation/escaping, upgrade if the file needs them.
        """
        values: dict[str, str] = {}
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("export "):
                stripped = stripped[len("export "):].lstrip()
            key, separator, value = stripped.partition("=")
            if not separator:
                continue
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if key:
                values[key] = value
        return values
