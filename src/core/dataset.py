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
  * the official SETS and their SPLITS — a set is a named data surface with a
    declared split map; a set that has none says so (empty, never invented);
  * the identity — a structural census over the declared line-item members
    (their names plus byte sizes; never a content digest).

It BAKES IN the behavior consumers need, so a consumer asks the class instead
of re-spelling a literal:

  member(name)        resolve one declared member to an absolute Path
  paths()             every declared member (source + line items + trees)
  sets()              every declared set (name -> resolved Path)
  set_path(name)      resolve one declared set to an absolute Path
  splits(name)        a set's declared splits; a set with no splits says so
  load(name, **read)  read a line-item frame through the declared read spec
  load_source()       the raw export
  load_deduped()      the deduplicated rows
  load_canonical_records()   the validated canonical-record artifact
  identity()          structural census (names + byte sizes) of the declared members
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
surface it offers is the on-demand structural census in :meth:`Dataset.identity`.
A missing member is recorded as absent in the census, never raised as a
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

The same document declares the SETS with their SPLITS, the FLEX
``validation_size`` (a fraction in (0, 1) or a row count — never a hardcoded
count in code), and the ``binned_sets`` list (names only) of sets the owner
has removed from the official surface.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from core.bundle import BundleRole
from core.schemas import DatasetCsvReadSpec

#: The declared member spec lives beside the other config documents.
CONFIG_NAME = "dataset.yaml"

#: The name of the source-export member (`Dataset.source`).
SOURCE = "source"


def _member_size(path: Path) -> int | None:
    """The member's byte size, or ``None`` when the member is absent.

    Structural identity only: the census never reads (or hashes) content.
    """
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return None


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


class DatasetSetSpec(BaseModel):
    """One declared SET: the binding that resolves it plus its named splits.

    ``binding`` resolves the set's own data surface; ``splits`` maps each split
    name to the binding that resolves that split. An EMPTY ``splits`` declares
    that the set has NO splits — the class reports that rather than inventing
    one. Binding existence is checked by :meth:`Dataset.from_config`.
    """

    model_config = ConfigDict(extra="forbid")

    binding: DatasetMemberSpec
    splits: dict[str, DatasetMemberSpec] = Field(default_factory=dict)


class DatasetSpec(BaseModel):
    """The dataset's declaration (``config/dataset.yaml``).

    Shape-only validation lives here; binding existence is checked by
    :meth:`Dataset.from_config` against the loaded ``DataConfig`` (so the spec
    model stays usable with a synthetic config in tests).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    source: DatasetMemberSpec
    members: dict[str, DatasetMemberSpec] = Field(min_length=1)
    layout: dict[str, DatasetMemberSpec] = Field(default_factory=dict)
    #: The declared sets; each carries its own split map (empty ⇒ no splits).
    sets: dict[str, DatasetSetSpec] = Field(default_factory=dict)
    #: The FLEX validation size: a fraction in (0, 1) or a row count >= 1.
    validation_size: float | int | None = None
    #: The declared bin list (names only): sets removed from the official surface.
    binned_sets: list[str] = Field(default_factory=list)

    @field_validator("validation_size")
    @classmethod
    def _validation_size_is_flex(
        cls,
        value: float | int | None,  # noqa: PYI041 (int must stay int: a row count)
    ) -> float | int | None:
        """A validation size is a fraction in (0, 1) or a row count >= 1 — never a fixed literal."""
        if value is None:
            return value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
            return value
        if isinstance(value, float) and 0.0 < value < 1.0:
            return value
        raise ValueError(
            "validation_size must be a fraction in (0, 1) or a row count >= 1, "
            f"got {value!r}"
        )

    @model_validator(mode="after")
    def _names_are_disjoint(self) -> DatasetSpec:
        """No member may shadow the source name or a tree root (`[]` = fine)."""
        if SOURCE in self.members or SOURCE in self.layout:
            raise ValueError(f"dataset member {SOURCE!r} is reserved for the source export")
        overlap = sorted(set(self.members) & set(self.layout))
        if overlap:
            raise ValueError(f"dataset member(s) {overlap} are declared as both line item and tree")
        set_overlap = sorted(set(self.sets) & ({SOURCE} | set(self.members) | set(self.layout)))
        if set_overlap:
            raise ValueError(f"dataset set(s) {set_overlap} shadow a source/member/tree name")
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


class DatasetSplits(BaseModel):
    """One declared set's resolved splits.

    ``members`` maps each declared split name to its resolved Path. An EMPTY
    ``members`` is the class SAYING SO: the set declares no splits, and none is
    invented for it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The set these splits belong to.
    set: str
    #: Split name -> resolved Path (empty ⇒ the set declares no splits).
    members: dict[str, Path] = Field(default_factory=dict)

    @property
    def declares_no_splits(self) -> bool:
        """``True`` when the set's declaration carries no splits."""
        return not self.members


class Dataset(BaseModel):
    """The project dataset: owner of its members, its sets/splits, its layout,
    and its identity.

    Built from the validated config SSOT by :meth:`from_config` (or the cached
    :func:`dataset` accessor). Frozen, so a consumer cannot mutate the resolved
    surface in place.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    read: DatasetCsvReadSpec
    source: Path
    #: Line-item artifacts: the census surface, loaded through ``read``.
    members: dict[str, Path]
    #: Declared tree roots (prepared/*, track_setup/*) — addressed, not sized.
    layout: dict[str, Path]
    #: Declared sets: name -> the binding that resolves the set's data surface.
    set_paths: dict[str, Path] = Field(default_factory=dict)
    #: Declared set splits: set name -> split name -> resolved Path ({} ⇒ none).
    set_splits: dict[str, dict[str, Path]] = Field(default_factory=dict)
    #: The FLEX validation size (fraction in (0, 1) or row count >= 1).
    validation_size: float | int | None = None
    #: The declared bin list (names only).
    binned_sets: tuple[str, ...] = ()

    # ------------------------------------------------------------ construction
    @classmethod
    def from_config(
        cls, spec: DatasetSpec | None = None, *, data: Any = None
    ) -> Dataset:
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
            set_paths={name: _binding_path(entry.binding, data)
                       for name, entry in spec.sets.items()},
            set_splits={name: {split: _binding_path(binding, data)
                               for split, binding in entry.splits.items()}
                        for name, entry in spec.sets.items()},
            validation_size=spec.validation_size,
            binned_sets=tuple(spec.binned_sets),
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

    # ------------------------------------------------------------------- sets
    def sets(self) -> dict[str, Path]:
        """Every declared set: its name and the binding that resolves it."""
        return dict(self.set_paths)

    def set_path(self, name: str) -> Path:
        """Resolve one declared set to its absolute Path (fail loud if undeclared)."""
        self._require_declared_set(name)
        return self.set_paths[name]

    def splits(self, name: str) -> DatasetSplits:
        """The declared splits of one set.

        A set with NO splits says so: the returned ``members`` map is empty and
        :attr:`DatasetSplits.declares_no_splits` is ``True`` — the class never
        invents a split.
        """
        self._require_declared_set(name)
        return DatasetSplits(set=name, members=dict(self.set_splits.get(name, {})))

    def _require_declared_set(self, name: str) -> None:
        """Refuse an undeclared set name, naming the declared ones."""
        if name not in self.set_paths:
            raise KeyError(
                f"unknown dataset set {name!r}; declared sets: {sorted(self.set_paths)}"
            )

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
    def identity(self) -> dict[str, Any]:
        """The dataset's structural census over its declared line-item members.

        Deterministic and portable: the census records ``name`` plus the
        members' byte sizes ALONE (never the checkout root), so the same bytes
        at two paths share one census. An absent member is recorded as ``None``
        rather than raising — this is a census, not a completeness verdict.

        ``layout`` tree roots are deliberately excluded: they are addressing,
        not line-item data, and walking them would fold a whole archive into
        every census call.
        """
        return {
            "dataset": self.name,
            "members": {
                name: _member_size(self.member(name))
                for name in (SOURCE, *sorted(self.members))
            },
        }

    # --------------------------------------------------------------- transport
    def as_bundle(self, path: Any, role: BundleRole | str = BundleRole.inputs, *,
                  manifest_name: str | None = None):
        """Hand the dataset off to its sealed transport form, ``Bundle``.

        The class does not seal (that is ``BundlePipeline``'s single
        responsibility); it only names the seam: the dataset's transport form is
        a ``Bundle``, and its integrity is verified exactly once, at that
        boundary (by member names and byte sizes).
        """
        from core.bundle import Bundle

        return Bundle.load(
            Path(path), role,
            manifest_name=manifest_name,
        )


@lru_cache(maxsize=1)
def dataset() -> Dataset:
    """The process-wide project dataset, resolved from the config SSOT once."""
    return Dataset.from_config()
