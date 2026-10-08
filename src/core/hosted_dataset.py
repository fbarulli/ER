"""src/core/hosted_dataset.py — the HOSTED-dataset registry (sibling to ``Dataset``).

WHY THIS MODULE EXISTS
----------------------
The Kaggle/Colab dataset slugs of this project used to live scattered as
per-lane literals: ``core.laya_config.LayaSpec`` (``base_model_dataset``,
``dataset_slug``, ``export_dataset_slug``, ``finetune_dataset_slug``,
``finetune_ckpt_dataset``, ``holdout_dataset_slug``) and ``core.schemas``
``KaggleSpec`` (``bundle_dataset_slug``, ``embedding_dataset_slug``). Nothing
owned the answers to "what is this dataset FOR (role), which way does it
travel (direction), which kernel attaches it, and where does it mount/stage?".

:class:`HostedRegistry` is that owner. It is a SIBLING of the local
:class:`core.dataset.Dataset` (which stays the local project-data owner), not a
mixin into it: one class declares the hosted surface, and consumers ASK it
instead of re-spelling a slug.

WHAT IT BAKES IN
----------------
  entry(slug)             look one hosted dataset up by its ``owner/slug``
  by_role(role)           the dataset that plays a role (base/corpus/.../embeddings)
  by_direction(direction) every dataset travelling input (mount-in) or output (publish)
  by_kernel(kind)         every dataset a kernel kind attaches
  kernel_inputs(kind)     the ones that kernel MOUNTS
  kernel_outputs(kind)    the ones that kernel PUBLISHES
  mount_path(slug)        ``/kaggle/input/<handle>`` on a session
  local_path(slug)        the staged local directory under ``staging_root``
  mount_member(..)/local_member(..)   one declared member file inside either
  existing_on_kaggle()    entries hosted on Kaggle today
  pending_on_kaggle()     entries declared but NOT yet hosted (see ``on_kaggle``)

Entry helpers (:class:`HostedDataset`) carry identity (``owner``/``handle``),
direction (``is_input``/``is_output``), attach (``accepts_kernel``) and
membership (``has_member``) reasoning, so no consumer re-derives them.

DIVISION OF LABOR
-----------------
The registry never seals and never transports: sealing is ``core.bundle``
(``Bundle``/``BundlePipeline``/``BundleRole``). Where a hosted dataset travels
as a sealed bundle, the entry names the ``BundleRole`` it travels under
(``bundle_role``) — ``BundleRole`` is reused, never re-declared, and no second
role enum exists here: the hosted ``role`` vocabulary (base/corpus/requests/
decisions/holdout/ckpt/bundle/embeddings) is a DATA-role vocabulary declared in
``config/hosted_datasets.yaml``, a different axis from the transport role.

NAKED DATA
----------
Addresses and roles only. No hashes, no versions, no existence gates anywhere
in this module; ``on_kaggle`` is a hosting FACT, never a validity verdict.

DECLARATION
-----------
Everything lives in ``config/hosted_datasets.yaml``: the mount root and the
mount/local templates, the direction/role/kernel-kind vocabularies, and each
entry's slug, name, role, directions, members, attach set, mount handle, staged
local name and ``on_kaggle`` fact. This module spells no slug, no path, no
role and no kernel kind; it reads the document through the ONE read+validate
home ``core.common.load_validated_yaml``.
"""
from __future__ import annotations

import traceback
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from core.bundle import BundleRole
from core.run_log import RunLogger

log = RunLogger(__name__)

#: The declared registry document lives beside the other config documents.
CONFIG_NAME = "hosted_datasets.yaml"

#: The one separator between a slug's owner and its handle.
_SLUG_SEPARATOR = "/"


class HostedDataset(BaseModel):
    """One hosted dataset, as declared: identity, role, direction, attach, address.

    Frozen value type. Every field is read from ``config/hosted_datasets.yaml``
    (validated by :class:`HostedRegistrySpec`); the model itself only refuses a
    malformed value and bakes in the lookups a consumer would otherwise
    re-derive.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: ``owner/handle`` — the Kaggle dataset slug (e.g. an ``owner/er-*`` name).
    slug: str = Field(min_length=3)
    #: Human name (what the dataset is), never used for addressing.
    name: str = Field(min_length=1)
    #: The hosted DATA role (base/corpus/requests/decisions/holdout/ckpt/
    #: bundle/embeddings) — validated against the declared role vocabulary.
    role: str = Field(min_length=1)
    #: Ordered: ``input`` = mount-in, ``output`` = publish. ``output, input``
    #: reads publish-then-mount (the fine-tune checkpoint).
    directions: tuple[str, ...] = Field(min_length=1)
    #: Kernel kinds that attach this dataset (mount and/or publish).
    attach: tuple[str, ...] = Field(min_length=1)
    #: The files the dataset carries (which splits/artifacts it is a set of).
    members: tuple[str, ...] = Field(min_length=1)
    #: The directory Kaggle mounts the dataset under (== the slug handle).
    mount: str = Field(min_length=1)
    #: The staged local directory name under the registry's staging root.
    local: str = Field(min_length=1)
    #: Whether the dataset exists on Kaggle today (a hosting fact, not a gate).
    on_kaggle: bool = True
    #: The ``BundleRole`` this dataset travels under when it IS a sealed bundle
    #: (reuse, never a second role enum); ``None`` for non-bundle datasets.
    bundle_role: BundleRole | None = None

    # ----------------------------------------------------------- validators
    @field_validator("slug")
    @classmethod
    def _slug_is_owner_and_handle(cls, value: str) -> str:
        """A hosted slug is exactly ``owner/handle``, both non-empty."""
        parts = value.split(_SLUG_SEPARATOR)
        if len(parts) != 2 or not all(part.strip() for part in parts):
            raise ValueError(
                f"hosted dataset slug must be 'owner/handle': {value!r}")
        return value

    @field_validator("mount", "local")
    @classmethod
    def _is_one_directory_name(cls, value: str) -> str:
        """A mount/staging directory name is ONE name, never a path fragment."""
        if (value in {".", ".."} or _SLUG_SEPARATOR in value
                or "\\" in value or not value.strip()):
            raise ValueError(
                f"hosted dataset directory name must be a single name: {value!r}")
        return value

    @field_validator("directions", "attach", "members")
    @classmethod
    def _names_are_non_empty_and_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not name.strip() for name in value) or len(set(value)) != len(value):
            raise ValueError(
                f"hosted dataset names must be non-empty and unique: {value}")
        return value

    @model_validator(mode="after")
    def _mount_is_the_slug_handle(self) -> "HostedDataset":
        """Kaggle mounts a dataset under its handle; a mismatch is a typo, not config."""
        if self.mount != self.handle:
            raise ValueError(
                f"hosted dataset {self.slug!r} declares mount {self.mount!r}, but "
                f"Kaggle mounts it under its handle {self.handle!r}")
        return self

    # ------------------------------------------------------------- identity
    @property
    def owner(self) -> str:
        """The publishing account half of the slug."""
        return self.slug.split(_SLUG_SEPARATOR)[0]

    @property
    def handle(self) -> str:
        """The dataset half of the slug — the directory Kaggle mounts it under."""
        return self.slug.split(_SLUG_SEPARATOR)[1]

    # ------------------------------------------------------------ direction
    @property
    def is_input(self) -> bool:
        """``True`` when the dataset is mounted in by some kernel."""
        return "input" in self.directions

    @property
    def is_output(self) -> bool:
        """``True`` when the dataset is published by some kernel."""
        return "output" in self.directions

    # --------------------------------------------------------------- attach
    def accepts_kernel(self, kind: str) -> bool:
        """Whether a kernel ``kind`` attaches this dataset."""
        return kind in self.attach

    def has_member(self, name: str) -> bool:
        """Whether ``name`` is one of the dataset's declared members."""
        return name in self.members


class HostedRegistrySpec(BaseModel):
    """The declared registry document (``config/hosted_datasets.yaml``).

    Owns the vocabularies and the invariants: every entry must speak the
    declared direction/role/kernel-kind vocabularies, slugs/roles/mounts must
    be unique (a role identifies EXACTLY ONE hosted dataset), and every
    template must render.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    #: Absolute mount root on a session (e.g. ``/kaggle/input``).
    mount_root: str = Field(min_length=1)
    #: ``str.format`` template rendering ``mount_root`` + an entry's mount name.
    mount_template: str = Field(min_length=1)
    #: Repo-relative staging root for materialized hosted datasets.
    staging_root: str = Field(min_length=1)
    #: ``str.format`` template rendering the staging root + an entry's local name.
    local_template: str = Field(min_length=1)
    directions: tuple[str, ...] = Field(min_length=1)
    roles: tuple[str, ...] = Field(min_length=1)
    kernel_kinds: tuple[str, ...] = Field(min_length=1)
    datasets: tuple[HostedDataset, ...] = Field(min_length=1)

    # ----------------------------------------------------------- validators
    @field_validator("mount_root")
    @classmethod
    def _mount_root_is_absolute(cls, value: str) -> str:
        if not value.startswith(_SLUG_SEPARATOR):
            raise ValueError(
                f"hosted registry mount_root must be absolute: {value!r}")
        return value

    @field_validator("staging_root")
    @classmethod
    def _staging_root_is_repo_relative(cls, value: str) -> str:
        candidate = Path(value)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(
                "hosted registry staging_root must be a portable repo-relative "
                f"path: {value!r}")
        return value

    @model_validator(mode="after")
    def _declaration_is_coherent(self) -> "HostedRegistrySpec":
        self._vocabularies_are_unique()
        self._entries_speak_the_declared_vocabularies()
        self._slugs_are_unique()
        self._roles_identify_one_entry()
        self._mounts_are_unique()
        self._templates_render()
        return self

    def _vocabularies_are_unique(self) -> None:
        for label, values in (("directions", self.directions),
                              ("roles", self.roles),
                              ("kernel_kinds", self.kernel_kinds)):
            if len(set(values)) != len(values):
                raise ValueError(
                    f"hosted registry {label} must be unique: {values}")

    def _entries_speak_the_declared_vocabularies(self) -> None:
        """Every entry's role/directions/attach must come from the declarations."""
        for entry in self.datasets:
            self._role_is_declared(entry)
            self._directions_are_declared(entry)
            self._attach_kinds_are_declared(entry)

    def _role_is_declared(self, entry: HostedDataset) -> None:
        if entry.role not in self.roles:
            raise ValueError(
                f"hosted dataset {entry.slug!r} declares role {entry.role!r}, "
                f"which is not in the declared roles {list(self.roles)}")

    def _directions_are_declared(self, entry: HostedDataset) -> None:
        undeclared = [d for d in entry.directions if d not in self.directions]
        if undeclared:
            raise ValueError(
                f"hosted dataset {entry.slug!r} declares direction(s) {undeclared}, "
                f"which are not in the declared directions {list(self.directions)}")

    def _attach_kinds_are_declared(self, entry: HostedDataset) -> None:
        undeclared = [k for k in entry.attach if k not in self.kernel_kinds]
        if undeclared:
            raise ValueError(
                f"hosted dataset {entry.slug!r} attaches to kernel kind(s) "
                f"{undeclared}, which are not in the declared kernel_kinds "
                f"{list(self.kernel_kinds)}")

    def _slugs_are_unique(self) -> None:
        slugs = [entry.slug for entry in self.datasets]
        duplicates = sorted({slug for slug in slugs if slugs.count(slug) > 1})
        if duplicates:
            raise ValueError(f"hosted registry declares duplicate slug(s): {duplicates}")

    def _roles_identify_one_entry(self) -> None:
        roles = [entry.role for entry in self.datasets]
        duplicates = sorted({role for role in roles if roles.count(role) > 1})
        if duplicates:
            raise ValueError(
                f"hosted registry role(s) {duplicates} are declared by more than "
                "one dataset; a role must identify exactly one hosted dataset")

    def _mounts_are_unique(self) -> None:
        mounts = [entry.mount for entry in self.datasets]
        duplicates = sorted({mount for mount in mounts if mounts.count(mount) > 1})
        if duplicates:
            raise ValueError(f"hosted registry declares duplicate mount(s): {duplicates}")

    def _templates_render(self) -> None:
        """Both templates must render for every entry (a bad template fails loud)."""
        for entry in self.datasets:
            self.mount_template.format(root=self.mount_root, mount=entry.mount)
            self.local_template.format(root=self.staging_root, local=entry.local)


def hosted_registry_config_path() -> Path:
    """The declared registry document (``config/hosted_datasets.yaml``)."""
    from core.common import CONFIG_DIR

    return CONFIG_DIR / CONFIG_NAME


def hosted_registry_spec() -> HostedRegistrySpec:
    """Read + validate the registry declaration through the ONE read+validate home."""
    from core.common import load_validated_yaml

    return load_validated_yaml(
        hosted_registry_config_path(), HostedRegistrySpec,
        label="Hosted dataset declaration",
    )


class HostedRegistry(BaseModel):
    """The hosted-dataset registry: lookups + addresses, resolved from the SSOT.

    Frozen. Built by :meth:`from_config` (or the cached :func:`hosted_registry`
    accessor). It never touches the network, never seals and never checks
    freshness: it answers WHERE a hosted dataset lives and WHAT it is for.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    #: Absolute mount root on a session (``/kaggle/input``).
    mount_root: Path
    #: Resolved, repo-relative staging root for materialized hosted datasets.
    staging_root: Path
    #: Declared direction/role/kernel-kind vocabularies (verbatim from yaml).
    directions: tuple[str, ...]
    roles: tuple[str, ...]
    kernel_kinds: tuple[str, ...]
    #: ``str.format`` templates, verbatim from yaml (never re-spelled in code).
    mount_template: str
    local_template: str
    #: The declared entries, keyed by slug.
    entries: dict[str, HostedDataset]

    # ------------------------------------------------------------ construction
    @classmethod
    def from_config(cls, spec: HostedRegistrySpec | None = None, *,
                    root: Path | None = None) -> "HostedRegistry":
        """Resolve the declared entries and roots against the SSOT.

        ``spec``/``root`` are injectable so a test can pin resolution without
        touching the real config; production callers use the defaults
        (``core.common.TRAIN_ROOT``).
        """
        from core.common import TRAIN_ROOT

        try:
            spec = hosted_registry_spec() if spec is None else spec
        except Exception:
            log.error(
                "failed to read the hosted-dataset declaration:\n"
                + traceback.format_exc())
            raise
        root = TRAIN_ROOT if root is None else root
        return cls(
            name=spec.name,
            mount_root=Path(spec.mount_root),
            staging_root=(Path(root) / spec.staging_root).resolve(),
            directions=spec.directions,
            roles=spec.roles,
            kernel_kinds=spec.kernel_kinds,
            mount_template=spec.mount_template,
            local_template=spec.local_template,
            entries={entry.slug: entry for entry in spec.datasets},
        )

    # ----------------------------------------------------------------- lookup
    def slugs(self) -> tuple[str, ...]:
        """Every declared hosted dataset slug, in declaration order."""
        return tuple(self.entries)

    def entry(self, slug: str) -> HostedDataset:
        """The declared entry for one ``owner/handle`` slug."""
        if slug not in self.entries:
            raise KeyError(
                f"unknown hosted dataset {slug!r}; declared slugs: "
                f"{list(self.entries)}")
        return self.entries[slug]

    def by_role(self, role: str) -> HostedDataset:
        """The ONE dataset playing a declared role (``corpus``, ``ckpt``, ...)."""
        matches = [entry for entry in self.entries.values() if entry.role == role]
        if not matches:
            raise KeyError(
                f"no hosted dataset declares role {role!r}; declared roles: "
                f"{list(self.roles)}")
        return matches[0]

    def by_direction(self, direction: str) -> tuple[HostedDataset, ...]:
        """Every dataset travelling a declared direction (``input``/``output``)."""
        if direction not in self.directions:
            raise KeyError(
                f"unknown hosted direction {direction!r}; declared directions: "
                f"{list(self.directions)}")
        return tuple(entry for entry in self.entries.values()
                     if direction in entry.directions)

    def by_kernel(self, kind: str) -> tuple[HostedDataset, ...]:
        """Every dataset a kernel ``kind`` attaches (mounted and/or published)."""
        if kind not in self.kernel_kinds:
            raise KeyError(
                f"unknown kernel kind {kind!r}; declared kernel_kinds: "
                f"{list(self.kernel_kinds)}")
        return tuple(entry for entry in self.entries.values()
                     if entry.accepts_kernel(kind))

    def kernel_inputs(self, kind: str) -> tuple[HostedDataset, ...]:
        """The datasets a kernel ``kind`` MOUNTS (direction ``input``)."""
        return tuple(entry for entry in self.by_kernel(kind) if entry.is_input)

    def kernel_outputs(self, kind: str) -> tuple[HostedDataset, ...]:
        """The datasets a kernel ``kind`` PUBLISHES (direction ``output``)."""
        return tuple(entry for entry in self.by_kernel(kind) if entry.is_output)

    def existing_on_kaggle(self) -> tuple[HostedDataset, ...]:
        """The datasets hosted on Kaggle today."""
        return tuple(entry for entry in self.entries.values() if entry.on_kaggle)

    def pending_on_kaggle(self) -> tuple[HostedDataset, ...]:
        """The declared datasets not hosted on Kaggle yet (declared, not a gate)."""
        return tuple(entry for entry in self.entries.values() if not entry.on_kaggle)

    # ------------------------------------------------------------- addresses
    def mount_path(self, slug: str) -> Path:
        """Where a session mounts the dataset: ``/kaggle/input/<handle>``."""
        entry = self.entry(slug)
        return Path(self.mount_template.format(root=self.mount_root, mount=entry.mount))

    def local_path(self, slug: str) -> Path:
        """The staged local directory for the dataset (under ``staging_root``)."""
        entry = self.entry(slug)
        return Path(self.local_template.format(
            root=self.staging_root, local=entry.local))

    def mount_member(self, slug: str, member: str) -> Path:
        """One declared member file inside the dataset's session mount."""
        return self.mount_path(slug) / self._declared_member(slug, member)

    def local_member(self, slug: str, member: str) -> Path:
        """One declared member file inside the dataset's staged local directory."""
        return self.local_path(slug) / self._declared_member(slug, member)

    def paths(self, slug: str) -> dict[str, Path]:
        """Both addresses of one dataset: the session mount and the staged local dir."""
        return {"mount": self.mount_path(slug), "local": self.local_path(slug)}

    # --------------------------------------------------------------- private
    def _declared_member(self, slug: str, member: str) -> str:
        """Refuse a member the dataset does not declare (typo guard, one place)."""
        entry = self.entry(slug)
        if not entry.has_member(member):
            raise KeyError(
                f"hosted dataset {slug!r} declares no member {member!r}; "
                f"declared members: {list(entry.members)}")
        return member


@lru_cache(maxsize=1)
def hosted_registry() -> HostedRegistry:
    """The process-wide hosted-dataset registry, resolved from the SSOT once."""
    return HostedRegistry.from_config()
