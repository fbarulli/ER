"""src/training/hpo_study.py — the HPO study-storage owner.

``config/training.yaml`` ``credentials:`` (validated by
:class:`core.credentials.CredentialsSpec`) declares the remote-URL env-var
name, the generation-id env-var name and the TRAIN_ROOT-relative local SQLite
study file. :class:`StudyOwner` is the Factory/Repository that resolves the one
Optuna storage a session uses, through the credential owner (never raw
``os.environ`` at a call site).

Selection rule (SSOT), in order:

1. a remote PostgreSQL URL is configured (process env wins, then the declared
   env file) → :class:`StudyBackend.REMOTE`;
2. otherwise a local SQLite study file is configured → :class:`StudyBackend.LOCAL`
   (shared by a single VM's parallel worker processes);
3. otherwise fail LOUD with the full traceback.

Security contract: the resolved URL may carry a password, so it leaves as a
:class:`~pydantic.SecretStr` and every repr/log uses :meth:`redacted_url`.
"""

from __future__ import annotations

import logging
import traceback
from enum import StrEnum
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, SecretStr

from core.credentials import CredentialKeysSpec, CredentialStore, StudySpec

_log = logging.getLogger(__name__)

#: URL query keys whose value is a secret and must be masked in a redaction.
_SENSITIVE_QUERY_KEYS = frozenset(
    {"password", "pass", "secret", "token", "sslpassword"})


class StudyBackend(StrEnum):
    """Which source the resolved study storage came from."""

    REMOTE = "remote"
    LOCAL = "local"


class StudyResolutionError(RuntimeError):
    """No study storage could be resolved; the owner logs the full traceback."""


def redact_url(url: str) -> str:
    """Mask the userinfo password and any secret query value of a URL.

    The scheme/host/database stay readable so a log line names the endpoint
    without ever printing the credential.
    """
    parts = urlsplit(url)
    netloc = parts.netloc
    if "@" in netloc:
        userinfo, _, hostport = netloc.rpartition("@")
        user = userinfo.split(":", 1)[0]
        netloc = f"{user}:***@{hostport}" if user else f"***@{hostport}"
    query = parts.query
    if query:
        query = urlencode([
            (key, "***" if key.lower() in _SENSITIVE_QUERY_KEYS else value)
            for key, value in parse_qsl(query, keep_blank_values=True)
        ])
    return urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))


class ResolvedStudy(BaseModel):
    """The one Optuna storage a session opens (URL never logged unredacted)."""

    model_config = ConfigDict(frozen=True)

    backend: StudyBackend
    url: SecretStr
    study_name: str

    @property
    def is_remote(self) -> bool:
        return self.backend is StudyBackend.REMOTE

    @property
    def redacted_url(self) -> str:
        return redact_url(self.url.get_secret_value())

    def storage_url(self) -> str:
        """The real URL for a consumer (RDBStorage / a staged kernel)."""
        return self.url.get_secret_value()

    def __repr__(self) -> str:
        return (f"ResolvedStudy(backend={self.backend.value!r}, "
                f"url={self.redacted_url!r}, study_name={self.study_name!r})")

    __str__ = __repr__


class StudyOwner:
    """Resolve the HPO study storage from the config SSOT + credential owner.

    Factory construction: :meth:`from_config` for production (reads the
    declared env file once) and explicit injection for tests/embedded callers.
    """

    def __init__(self, *, spec: StudySpec, keys: CredentialKeysSpec,
                 store: CredentialStore, root: Path) -> None:
        self._spec = spec
        self._keys = keys
        self._store = store
        self._root = Path(root)

    @classmethod
    def from_config(cls, *, credentials=None, credential_store=None,
                    root: Path | None = None) -> StudyOwner:
        """Build from the config SSOT, reading the declared env file once."""
        from core.common import TRAIN_ROOT, training_cfg

        creds = credentials if credentials is not None else training_cfg().credentials
        base = Path(root) if root is not None else TRAIN_ROOT
        store = (credential_store if credential_store is not None
                 else CredentialStore.from_config(creds, root=base))
        return cls(spec=creds.study, keys=creds.keys, store=store, root=base)

    @property
    def url_env(self) -> str:
        """The config-declared environment-variable name for the remote URL."""
        return self._keys.optuna_storage_url

    @property
    def generation_env(self) -> str:
        """The config-declared environment-variable name for the generation id."""
        return self._keys.hpo_generation_id

    @property
    def local_file(self) -> str | None:
        """The config-declared, TRAIN_ROOT-relative local study fragment."""
        return self._spec.local_file

    def generation_id(self, override: str | None = None) -> str:
        """The generation id: an explicit override wins, else the source env."""
        if override is not None and str(override).strip():
            return str(override).strip()
        value = self._store.resolve_env_optional(self.generation_env)
        return value.get_secret_value().strip() if value is not None else ""

    def remote_url(self) -> str | None:
        """The configured remote PostgreSQL URL, or None (never logged)."""
        value = self._store.resolve_env_optional(self.url_env)
        if value is None:
            return None
        return value.get_secret_value().strip() or None

    def local_url(self) -> str | None:
        """The config-declared SQLite URL (TRAIN_ROOT-relative), or None."""
        fragment = (self._spec.local_file or "").strip()
        if not fragment:
            return None
        return "sqlite:///" + str(self._root / Path(fragment))

    def resolve(self, *, generation_id: str, model_key: str) -> ResolvedStudy:
        """Resolve the ONE storage for a study; fail LOUD when none is usable."""
        from training.hpo_control_plane import generation_study_name

        name = generation_study_name(generation_id=generation_id,
                                     model_key=model_key)
        remote = self.remote_url()
        if remote:
            return ResolvedStudy(backend=StudyBackend.REMOTE,
                                 url=SecretStr(remote), study_name=name)
        local = self.local_url()
        if local:
            return ResolvedStudy(backend=StudyBackend.LOCAL,
                                 url=SecretStr(local), study_name=name)
        try:
            raise StudyResolutionError(
                f"no HPO study storage for {name!r}: {self.url_env} is unset "
                "and no credentials.study.local_file is configured; set a "
                "remote PostgreSQL URL or declare the local SQLite study path")
        except StudyResolutionError:
            _log.error("HPO study storage unresolved; full traceback:\n%s",
                       traceback.format_exc())
            raise
