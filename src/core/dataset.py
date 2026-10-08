"""src/core/dataset.py — the project-level Dataset: ONE owner of the data surface.

WHY THIS MODULE EXISTS
----------------------
The project's data surface used to be fragmented: file names lived in
``config/paths.yaml`` (``files:``), generated-artifact templates in the
``layouts:`` block, the raw-export read semantics in
``DataConfig.dataset_csv_read``, the prepared-input shape in
``training.prepared_bundle.CanonicalLayout`` / ``PreparedBundleManifest``,
and every consumer re-derived its own answer by indexing ``core.common.F``
inline. There was no object that could be asked "what is this project's
dataset, and what are its members?".

:class:`Dataset` is that object. It OWNS, in ONE place:

  * the source export — its path plus the shared read spec;
  * the file set — the line-item artifacts the dataset is derived into;
  * the canonical/prepared layout — the declared tree roots that hold the
    derived, bundled outputs (``prepared/*``, ``track_setup/*``);
  * the derived artifacts (canonical records, gate results, deduped rows,
    labeled pairs, the validation split, the number-token reference,
    the sku->representative map);
  * the identity — a content digest over the declared line-item members.

It BAKES IN the behavior consumers need, so a consumer asks the class instead
of re-spelling a literal:

  member(name)        resolve one declared member to an absolute Path
  paths()             every declared member (source + line items + trees)
  load(name, **read)  read a line-item frame through the declared read spec
  load_source()       the raw export
  load_deduped()      the deduplicated rows
  load_canonical_records()   the validated canonical-record artifact
  identity()          sha256 over the declared line-item members' bytes
  as_bundle(path, role)      transport hand-off to the sealed Bundle

DIVISION OF LABOR WITH ``core.bundle.Bundle``
---------------------------------------------
``Bundle`` (``src/core/bundle.py``) stays the sealed TRANSPORT form of a
dataset: an archive whose integrity is verified exactly once at the boundary.
``Dataset`` owns the data itself and never seals; :meth:`Dataset.as_bundle`
delegates the boundary check to ``Bundle.load``. Sealing remains
``BundlePipeline``'s job (its code is the only bundling code in the tree).

NO VALIDITY GATE HERE
---------------------
The class performs no age/validity verdict on its data. The ONLY integrity
surface it offers is the on-demand content digest in :meth:`Dataset.identity`.
A missing member is recorded as absence in the digest, never raised as a
verdict — the data stays naked; nothing here decides whether it is "good
enough" to use.

DECLARATION
-----------
The member list is declared in ``config/dataset.yaml`` (a spec, validated by
:class:`DatasetSpec` through the ONE read+validate home
``core.common.load_validated_yaml``). Each entry names an existing
``paths.yaml`` binding and is resolved through ``core.common.F`` /
``core.common.artifact``, so the class can never drift from the phone book:
resolution is the same accessor call the rest of the tree makes.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.schemas import DatasetCsvReadSpec

#: The declared member spec lives beside the other config documents.
CONFIG_NAME = "dataset.yaml"

#: The name of the source-export member (`Dataset.source`).
SOURCE = "source"


class DatasetMemberSpec(BaseModel):
    """One declared dataset member: which SSOT binding resolves it.

    ``via`` selects the phone book the key is looked up in — ``files`` for a
    line-item artifact, ``layouts`` for a generated-artifact/tree template —
    and ``key`` is the entry name. A key the phone book does not declare is
    refused by :meth:`Dataset.from_config`, so a typo can never silently
    resolve to nothing.
    """

    model_config = ConfigDict(extra="forbid")

    via: Literal["files", "layouts"]
    key: str = Field(min_length=1)


class DatasetSpec(BaseModel):
    """The dataset's member declaration (``config/dataset.yaml``).

    Shape-only validation lives here; binding existence is checked by
    :meth:`Dataset.from_config` against the loaded ``DataConfig`` (so the spec
    model stays usable with a synthetic config in tests).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    source: DatasetMemberSpec
    members: dict[str, DatasetMemberSpec] = Field(min_length=1)
    layout: dict[str, DatasetMemberSpec] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _names_are_disjoint(self) -> "DatasetSpec":
        """No member may shadow the source name or a tree root (`[]` = fine)."""
        if SOURCE in self.members or SOURCE in self.layout:
            raise ValueError(f"dataset member {SOURCE!r} is reserved for the source export")
        overlap = sorted(set(self.members) & set(self.layout))
        if overlap:
            raise ValueError(f"dataset member(s) {overlap} are declared as both line item and tree")
        return self


def dataset_config_path() -> Path:
    """The declared member-spec document (``config/dataset.yaml``)."""
    from core.common import TRAIN_ROOT

    return TRAIN_ROOT / "config" / CONFIG_NAME


def dataset_spec() -> DatasetSpec:
    """Read + validate the member declaration through the ONE read+validate home."""
    from core.common import load_validated_yaml

    return load_validated_yaml(
        dataset_config_path(), DatasetSpec, label="Dataset member declaration"
    )


def _binding_path(binding: DatasetMemberSpec, data: Any) -> Path:
    """Resolve one declared binding through the phone book the tree already uses.

    ``files`` bindings resolve through ``core.common.F`` and ``layouts``
    bindings through ``core.common.artifact`` — the SAME accessors every other
    module calls, so a ``Dataset`` member and the current inline accessor return
    the identical absolute Path. An undeclared key fails loud with the file it
    was expected in.
    """
    from core import common

    if binding.via == "files":
        if binding.key not in type(data.files).model_fields:
            raise ValueError(
                f"dataset binding files.{binding.key} is not declared in "
                f"config/paths.yaml files:"
            )
        return common.F[binding.key]
    if binding.key not in data.layouts:
        raise ValueError(
            f"dataset binding layouts.{binding.key} is not declared in "
            f"config/paths.yaml layouts:"
        )
    return common.artifact(binding.key)


class Dataset(BaseModel):
    """The project dataset: owner of its members, its layout, and its identity.

    Built from the validated config SSOT by :meth:`from_config` (or the cached
    :func:`dataset` accessor). Frozen, so a consumer cannot mutate the resolved
    surface in place.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    read: DatasetCsvReadSpec
    source: Path
    #: Line-item artifacts: the digest surface, loaded through ``read``.
    members: dict[str, Path]
    #: Declared tree roots (prepared/*, track_setup/*) — addressed, not digested.
    layout: dict[str, Path]

    # ------------------------------------------------------------ construction
    @classmethod
    def from_config(
        cls, spec: DatasetSpec | None = None, *, data: Any = None
    ) -> "Dataset":
        """Resolve the declared members against the validated config SSOT.

        ``spec``/``data`` are injectable so a test can pin resolution without
        touching the real config; production callers use the defaults.
        """
        from core.common import data_cfg

        spec = dataset_spec() if spec is None else spec
        data = data_cfg() if data is None else data
        return cls(
            name=spec.name,
            read=data.dataset_csv_read,
            source=_binding_path(spec.source, data),
            members={name: _binding_path(binding, data)
                     for name, binding in spec.members.items()},
            layout={name: _binding_path(binding, data)
                    for name, binding in spec.layout.items()},
        )

    # ------------------------------------------------------------------ paths
    def member(self, name: str) -> Path:
        """Resolve one declared member (the source, a line item, or a tree root)."""
        if name == SOURCE:
            return self.source
        if name in self.members:
            return self.members[name]
        if name in self.layout:
            return self.layout[name]
        raise KeyError(
            f"unknown dataset member {name!r}; declared members: {sorted(self.paths())}"
        )

    def paths(self) -> dict[str, Path]:
        """Every declared member: the source, the line items, and the tree roots."""
        return {SOURCE: self.source, **self.members, **self.layout}

    # ------------------------------------------------------------------ frames
    def load(self, name: str, **read: Any):
        """Read one line-item frame through the declared shared read spec.

        ``read`` overrides individual parsing keys; the declared
        ``dataset_csv_read`` semantics apply for everything not overridden, so a
        consumer steers a read without re-spelling the SSOT defaults.
        """
        import pandas as pd

        if name in self.layout:
            raise ValueError(
                f"{name!r} is a declared tree root, not a loadable frame"
            )
        spec = self.read
        kwargs: dict[str, Any] = {
            "dtype": spec.dtype,
            "keep_default_na": spec.keep_default_na,
            "na_filter": spec.na_filter,
        }
        kwargs.update(read)
        return pd.read_csv(self.member(name), **kwargs)

    def load_source(self):
        """The ONE inspected raw export, read under the shared read spec."""
        return self.load(SOURCE)

    def load_deduped(self):
        """The deduplicated rows the pipeline consumes."""
        return self.load("dataset_deduped")

    def load_canonical_records(self):
        """The canonical-record artifact through its validated reader.

        The artifact has its own frame contract (``gtin`` as str, upgrades and
        the ``check_canonical_records_frame`` assertion), so the class does not
        re-implement that read: it delegates to the ONE validated reader,
        ``core.common.canonical_records_frame``.
        """
        from core.common import canonical_records_frame

        return canonical_records_frame()

    # ---------------------------------------------------------------- identity
    def identity(self) -> str:
        """The dataset's content digest over its declared line-item members.

        Deterministic and portable: the digest is a function of ``name`` plus
        the members' bytes ALONE (never the checkout root), so the same bytes at
        two paths share one identity. An absent member contributes its name and
        an absence marker rather than raising — this is a content digest, not a
        completeness verdict.

        ``layout`` tree roots are deliberately excluded: they are addressing,
        not line-item data, and walking them would fold a whole archive into
        every identity call.
        """
        from core.manifest import sha256_file

        digest = hashlib.sha256()
        digest.update(self.name.encode("utf-8"))
        for name in (SOURCE, *sorted(self.members)):
            digest.update(b"\x00")
            digest.update(name.encode("utf-8"))
            digest.update(b"\x00")
            try:
                digest.update(sha256_file(self.member(name)).encode("ascii"))
            except FileNotFoundError:
                digest.update(b"-")
        return digest.hexdigest()

    # --------------------------------------------------------------- transport
    def as_bundle(self, path: Any, role: Any = "inputs", *,
                  expected_digest: str | None = None,
                  manifest_name: str | None = None):
        """Hand the dataset off to its sealed transport form, ``Bundle``.

        The class does not seal (that is ``BundlePipeline``'s single
        responsibility); it only names the seam: the dataset's transport form is
        a ``Bundle``, and its integrity is verified exactly once, at that
        boundary.
        """
        from core.bundle import Bundle

        return Bundle.load(
            Path(path), role,
            expected_digest=expected_digest, manifest_name=manifest_name,
        )


@lru_cache(maxsize=1)
def dataset() -> Dataset:
    """The process-wide project dataset, resolved from the config SSOT once."""
    return Dataset.from_config()
