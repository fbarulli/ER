"""src/core/artifacts.py — the run's ARTIFACT OWNER.

WHY THIS MODULE EXISTS
----------------------
Every run of the model-track suite emits a large, previously UNDECLARED set of
outputs: the per-track artifacts the training class writes (``text__vectors.npz``,
``gnn_only__graph_model.pt``, ``<track>__reports/``, ``<track>__completion_manifest.json``,
the ``_checkpoints/**/checkpoint-N`` trees, ...), the run-root members the
supervisor writes (``suite_manifest.json``, ``suite_events.jsonl``,
``resource_profile/``, ``models_manifest.json``), and the SEALED archives the
Bundle class writes (one per ``BundleRole``).  Their names were spelled inline at
each producer (``text_report.py``, ``graph_tracks/report.py``,
``model_tracks/bundle_steps.py``) and again at each consumer, so nothing could be
asked "what did this run produce, and which member belongs to which role/track?".
The model-track lane (``text_export``, ``text_report``, ``worker``,
``bundle_steps``, ``publish``, ``run``, ``local_complete``,
``post_training_ablation``) now asks this class for the filename or the address
of every member it writes or reads.  ``graph_tracks/**`` and ``training/**``
still spell theirs through their own helpers (out of this owner's scope).

:class:`Artifacts` is that object.  It OWNS:

  * the inventory — every declared run artifact, resolved from
    ``config/artifacts.yaml`` through the SAME phone book the tree already uses
    (``BundleSpec`` fields, ``paths.yaml`` layouts/files, the validated training
    config), so no call site re-spells a literal;
  * the COLLECTION — :meth:`Artifacts.from_run` walks a materialized run tree;
    :meth:`Artifacts.from_bundle` reads a verified ``Bundle`` handle. Both feed
    one classifier, so the two can never disagree about a member's identity;
  * the LOOKUPS — per role (``inputs``/``recovery``/``result``; the vocabulary
    is :class:`core.bundle.BundleRole`'s values, declared once in
    ``artifacts.yaml`` ``roles:``) and per track (``text``/``gnn_only``/
    ``cascade``), plus the SINGLE-MEMBER queries a producer/consumer uses
    instead of a path literal: :meth:`Artifacts.resolve` (a declared key to its
    Path) and :meth:`Artifacts.member_name` (the declared filename, for a
    caller holding a track root it places itself). Both are classmethods: the
    declaration, not a collected run, decides the answer;
  * the STRUCTURAL IDENTITY — :meth:`Artifacts.identity` (run tag + member names
    + byte sizes + counts).  There is NO content digest anywhere in this module:
    integrity is the sealed Bundle's boundary check, never a second verdict.

DIVISION OF LABOR
-----------------
``Bundle`` stays the sealed transport (its own member contract, its own
writer).  ``Training``/the trainers stay the producers.  ``Artifacts`` is the
read-side owner: it names and collects what they emitted, and it hands the
sealed-archive addresses back through :meth:`Artifacts.sealed_archives`.  It
never imports the Bundle implementation (only its role enum, the vocabulary
SSOT): :meth:`Artifacts.from_bundle` consumes the verified HANDLE by contract
(``role``, ``run_tag()``, ``members()``, ``manifest_name``, optional ``local``),
so a real ``Bundle`` and a test double are both acceptable inputs and the two
layers stay independent.

THE Results BOUNDARY (the plug between training and post-training)
------------------------------------------------------------------
The results that live BETWEEN training and post-training analysis (the ablation
request/report/vectors, the baseline ablation, the holdout report) are owned by
``core.results.Results``, NOT by this class.  ``config/artifacts.yaml`` declares
their tree members under ``results_owned``, and collection EXCLUDES them, so the
split is declared once instead of re-derived by each caller.  The plug is:
``Artifacts.from_run(run_root, run_tag).track_members(track)`` plus
``Artifacts.identity()`` give Results the trained artifacts and the structural
run identity; Results owns every analysis artifact built on top of them.

NO PATH LITERALS
----------------
The declared inventory document is the only name-like value this module holds
(``artifacts.yaml``, resolved under ``core.common.CONFIG_DIR``).  Everything else
is resolved from config: BundleSpec fields (``{checkpoint_dir}``,
``{trainer_state_file}``, ...), paths.yaml layouts, or a dotted lookup into the
validated training config.
"""
from __future__ import annotations

import traceback
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.run_log import RunLogger

_LOG = RunLogger(__name__)

#: The declared inventory document, resolved under ``core.common.CONFIG_DIR``.
CONFIG_NAME = "artifacts.yaml"
#: The structural-identity schema tag (names, sizes and counts; never a digest).
IDENTITY_SCHEMA = "er-run-artifacts-v1"

#: One artifact's shape on disk. ``dir`` entries are trees and aggregate.
KIND = Literal["file", "dir", "archive"]
#: Where a declaration's address comes from (the phone book it resolves through).
VIA = Literal["bundle", "layouts", "files", "config"]


# ───────────────────────────────────────────────────────────────────────────
# the declared spec (config/artifacts.yaml)
# ───────────────────────────────────────────────────────────────────────────
class ArtifactDecl(BaseModel):
    """One declared artifact: its shape, role, and how its address resolves.

    Exactly one resolution is declared: either a ``name`` template (relative to
    the run root, or to a track's root for ``track_artifacts``) or a ``via``
    phone book plus a ``key`` in it. A declaration that resolves through neither
    is refused at config load, never silently collected as nothing.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: KIND
    #: The BundleRole the member travels in (``None`` = not role-scoped).
    role: str | None = None
    #: Empty = the declaration applies to every declared track.
    tracks: tuple[str, ...] = ()
    via: VIA | None = None
    key: str | None = None
    name: str | None = None
    fields: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _has_one_resolution(self) -> "ArtifactDecl":
        if self.via is None and not self.name:
            raise ValueError("an artifact declaration needs a `name` or a `via` + `key`")
        if self.via is not None and not self.key:
            raise ValueError("a `via` resolution requires a `key`")
        if self.via is None and self.fields:
            raise ValueError("`fields` belongs to a `via` resolution, not to a bare `name`")
        return self


class ArtifactSpec(BaseModel):
    """The run-artifact declaration (``config/artifacts.yaml``).

    Shape-only validation lives here; binding existence is checked at resolve
    time (so the spec model stays usable with a synthetic declaration in tests).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    #: The artifact-bearing tracks, in suite order (model_tracks.resume.TRACKS).
    tracks: tuple[str, ...] = Field(min_length=1)
    #: role value -> the BundleSpec field that owns its manifest member name.
    roles: dict[str, str] = Field(min_length=1)
    track_artifacts: dict[str, ArtifactDecl] = Field(min_length=1)
    run_artifacts: dict[str, ArtifactDecl] = Field(default_factory=dict)
    sealed_archives: dict[str, ArtifactDecl] = Field(default_factory=dict)
    #: Tree members owned by ``core.results.Results`` (excluded from collection).
    results_owned: dict[str, ArtifactDecl] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _keys_are_unique(self) -> "ArtifactSpec":
        groups = (self.track_artifacts, self.run_artifacts,
                  self.sealed_archives, self.results_owned)
        keys = [key for group in groups for key in group]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        if duplicates:
            raise ValueError(
                f"artifact key(s) {duplicates} are declared in more than one group")
        return self

    @model_validator(mode="after")
    def _tracks_and_role_fields_are_declared(self) -> "ArtifactSpec":
        from core.schemas import BundleSpec

        known = set(self.tracks)
        for key, decl in self.track_artifacts.items():
            unknown = sorted(set(decl.tracks) - known)
            if unknown:
                raise ValueError(
                    f"track_artifacts.{key} names undeclared track(s) {unknown}")
        unknown_fields = sorted(set(self.roles.values()) - set(BundleSpec.model_fields))
        if unknown_fields:
            raise ValueError(
                f"roles names unknown BundleSpec field(s) {unknown_fields}")
        return self

    @model_validator(mode="after")
    def _roles_are_the_bundle_roles(self) -> "ArtifactSpec":
        """The role vocabulary IS ``core.bundle.BundleRole`` — never a second enum.

        The ``roles:`` mapping's keys restate the enum's values once, because each
        role's manifest member name has to resolve to the BundleSpec field that
        owns it; the mirror is checked here so a renamed/added role cannot leave a
        stale vocabulary behind in this declaration.
        """
        from core.bundle import BundleRole

        unknown = sorted(set(self.roles) - {role.value for role in BundleRole})
        if unknown:
            raise ValueError(
                f"roles declares {unknown}, which core.bundle.BundleRole does not define")
        return self

    @model_validator(mode="after")
    def _role_scoped_entries_are_declared(self) -> "ArtifactSpec":
        """Every declared role on an entry is one of the spec's own role keys."""
        for group_name in ("track_artifacts", "run_artifacts", "sealed_archives"):
            group = getattr(self, group_name)
            for key, decl in group.items():
                if decl.role is not None and decl.role not in self.roles:
                    raise ValueError(
                        f"{group_name}.{key} declares role {decl.role!r}, which the "
                        f"declared roles {sorted(self.roles)} do not include")
        return self


def artifacts_config_path() -> Path:
    """The declared inventory document (``config/artifacts.yaml``)."""
    from core.common import CONFIG_DIR

    return CONFIG_DIR / CONFIG_NAME


@lru_cache(maxsize=1)
def artifacts_spec() -> ArtifactSpec:
    """Read + validate the declaration through the ONE read+validate home."""
    from core.common import load_validated_yaml

    return load_validated_yaml(
        artifacts_config_path(), ArtifactSpec, label="Run-artifact declaration")


# ───────────────────────────────────────────────────────────────────────────
# config resolution (no literals: every value comes from the phone book)
# ───────────────────────────────────────────────────────────────────────────
def _bundle_field(key: str) -> Any:
    """The value of one ``config/training.yaml bundle:`` field (BundleSpec)."""
    from core.common import training_cfg

    spec = training_cfg().bundle
    if key not in type(spec).model_fields:
        raise ValueError(
            f"artifact declaration names bundle.{key}, which config/training.yaml "
            f"bundle: does not declare")
    return getattr(spec, key)


def _config_field(dotted: str) -> Any:
    """A dotted lookup into the validated training config (e.g. a name template)."""
    from core.common import training_cfg

    node: Any = training_cfg()
    for part in dotted.split("."):
        if not hasattr(node, part):
            raise ValueError(
                f"artifact declaration names training config {dotted!r}, "
                f"which has no {part!r}")
        node = getattr(node, part)
    return node


def _placeholders() -> dict[str, str]:
    """Every string name a declaration template may interpolate.

    The BundleSpec and ColabSpec fields ARE the declared names for the checkpoint
    layout, the track markers and the checkpoint manifest, so a template asks for
    them by field name instead of spelling the value again (e.g.
    ``"{track}__{checkpoint_manifest_name}"``).
    """
    from core.common import training_cfg

    cfg = training_cfg()
    values: dict[str, str] = {}
    for source in (cfg.bundle, cfg.colab):
        for field in type(source).model_fields:
            value = getattr(source, field)
            if isinstance(value, str):
                values[field] = value
    return values


def _render(template: str, *, track: str | None = None, **fields: object) -> str:
    """Fill a declaration template from config values ONLY (a typo fails loud)."""
    values = dict(_placeholders())
    values.update({key: str(value) for key, value in fields.items()})
    if track is not None:
        values["track"] = track
    try:
        return template.format(**values)
    except KeyError as exc:
        _LOG.error("artifact template %r needs undeclared value %s:\n%s",
                   template, exc, traceback.format_exc())
        raise ValueError(
            f"artifact template {template!r} needs undeclared value {exc}") from exc


def _declared_name(decl: ArtifactDecl, *, track: str | None = None,
                   **fields: object) -> str | None:
    """The literal member name a declaration resolves to (``None`` when address-only)."""
    if decl.name is not None:
        return _render(decl.name, track=track, **fields) if "{" in decl.name else decl.name
    if decl.via == "bundle" and decl.key is not None:
        return str(_bundle_field(decl.key))
    return None


def _role_manifest(spec: ArtifactSpec, role: str) -> str:
    """The canonical manifest member name for one role (config SSOT)."""
    value = str(role)
    if value not in spec.roles:
        raise ValueError(
            f"artifact declaration has no role {value!r}; declared: {sorted(spec.roles)}")
    return str(_bundle_field(spec.roles[value]))


def _checked_role(spec: ArtifactSpec, role: Any) -> str:
    """One role value proven to belong to the declared vocabulary.

    ``BundleRole`` is the SSOT for the vocabulary; the declared ``roles:`` keys
    mirror it. A live handle's role arrives here, so drift between the two fails
    loud at the boundary rather than selecting an empty member set silently.
    """
    value = getattr(role, "value", role)
    if not isinstance(value, str) or value not in spec.roles:
        raise ValueError(
            f"role {role!r} is not one of the declared roles {sorted(spec.roles)}")
    return value


def _declared_names(spec: ArtifactSpec) -> list[str]:
    """Every declared artifact key (the spec's whole surface)."""
    return sorted({*spec.track_artifacts, *spec.run_artifacts,
                   *spec.sealed_archives, *spec.results_owned})


def _decl_for(spec: ArtifactSpec, key: str) -> ArtifactDecl:
    """The declaration for one key, or fail loud with the declared surface."""
    for group in (spec.track_artifacts, spec.run_artifacts,
                  spec.sealed_archives, spec.results_owned):
        decl = group.get(key)
        if decl is not None:
            return decl
    raise KeyError(f"unknown artifact {key!r}; declared: {_declared_names(spec)}")


def results_owned_components(spec: ArtifactSpec | None = None) -> frozenset[str]:
    """The path components ``core.results.Results`` owns (never collected here).

    Public because the boundary is a contract, not an internal detail: a Results
    owner can ask which components this class deliberately leaves alone.
    """
    spec = artifacts_spec() if spec is None else spec
    components: set[str] = set()
    for decl in spec.results_owned.values():
        name = _declared_name(decl)
        if name:
            components.add(name)
    return frozenset(components)


def _prepared_inputs_dir() -> str:
    """The extracted prepared-inputs dir: an INPUT to the process, never a deliverable."""
    return str(_bundle_field("prepared_inputs_dir"))


def _stat_size(path: Path) -> int | None:
    """One file's byte size; an unreadable member is logged in full, never hidden."""
    try:
        return path.stat().st_size
    except OSError:
        _LOG.error("artifact size unavailable for %s:\n%s", path, traceback.format_exc())
        return None


# ───────────────────────────────────────────────────────────────────────────
# the collected member
# ───────────────────────────────────────────────────────────────────────────
class ArtifactMember(BaseModel):
    """One collected artifact: its run-relative name plus its structural facts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Run-relative posix name (a tree aggregates to its own directory name).
    name: str = Field(min_length=1)
    kind: KIND
    #: The ``track_artifacts``/``run_artifacts`` key that classified it (or None).
    declared: str | None = None
    #: The BundleRole the member travels in.
    role: str | None = None
    track: str | None = None
    #: Byte size: a file's size, a tree's total, or None when unknown (an archive
    #: member is sized only through a materialized tree).
    size_bytes: int | None = Field(default=None, ge=0)
    #: 1 for a file/archive; the number of files under a tree.
    count: int = Field(default=1, ge=1)

    def structural(self) -> dict[str, Any]:
        """This member's structural identity (no digest, by construction)."""
        return {
            "name": self.name,
            "kind": self.kind,
            "declared": self.declared,
            "role": self.role,
            "track": self.track,
            "size_bytes": self.size_bytes,
            "count": self.count,
        }


# ───────────────────────────────────────────────────────────────────────────
# the owner
# ───────────────────────────────────────────────────────────────────────────
class Artifacts(BaseModel):
    """The artifacts ONE run emitted: collected, classified, and identified.

    Built only through :meth:`from_run` (a materialized run tree) or
    :meth:`from_bundle` (a verified ``Bundle`` handle), so the collection always
    comes from a real producer surface instead of a hand-built list.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    run_tag: str
    #: The BundleRole whose member contract this collection represents.
    role: str
    #: The role's canonical manifest member name (``manifest_for`` re-resolves it).
    manifest_name: str
    #: True when the collection came from a sealed Bundle boundary.
    sealed: bool
    #: The run root, when the collection came from a materialized tree.
    root: Path | None = None
    #: The declaration this collection was classified against.
    spec: ArtifactSpec
    members: dict[str, ArtifactMember] = Field(default_factory=dict)

    # ------------------------------------------------------------- construction
    @classmethod
    def from_run(cls, root: Path | str, run_tag: str, *, role: str = "result",
                 spec: ArtifactSpec | None = None) -> "Artifacts":
        """Collect the artifacts of one materialized run tree.

        The tree is walked once; a declared ``dir`` member aggregates its files
        into ONE member (its total bytes and file count), a declared file is one
        member, and every other file is collected unclassified rather than
        dropped. The extracted prepared inputs and the Results-owned members are
        excluded by declaration, not by a caller's guess.
        """
        spec = artifacts_spec() if spec is None else spec
        role = _checked_role(spec, role)
        root = Path(root)
        return cls._collect(spec, cls._run_entries(root), role=role, run_tag=run_tag,
                            sealed=False, root=root)

    @classmethod
    def from_bundle(cls, bundle: Any, *, spec: ArtifactSpec | None = None) -> "Artifacts":
        """Collect the member inventory of one verified ``Bundle`` handle.

        ``bundle`` is the verified HANDLE, not the class: it must expose ``role``
        (a value whose ``.value`` is a declared role), ``run_tag()``,
        ``members()``, ``manifest_name`` and optionally ``local`` (a materialized
        tree, the only way a member can be sized). The handle is trusted (its
        integrity was checked exactly once at the boundary), so this reads member
        NAMES and, for a materialized tree, byte sizes; it never re-opens or
        re-verifies the archive. A result bundle's member set is already
        selected-only, which is how "all-epoch vs selected" is expressed on this
        side.
        """
        spec = artifacts_spec() if spec is None else spec
        role = _checked_role(spec, bundle.role)
        return cls._collect(spec, cls._bundle_entries(bundle), role=role,
                            run_tag=bundle.run_tag(), sealed=True, root=None,
                            manifest_name=bundle.manifest_name)

    # -------------------------------------------------------------- collection
    @classmethod
    def _collect(cls, spec: ArtifactSpec, entries: Iterator[tuple[str, int | None]], *,
                 role: str, run_tag: str, sealed: bool, root: Path | None,
                 manifest_name: str | None = None) -> "Artifacts":
        """Classify + aggregate raw ``(name, size)`` entries into members (one pass)."""
        owned = results_owned_components(spec)
        prepared = _prepared_inputs_dir()
        members: dict[str, ArtifactMember] = {}
        for relative, size in entries:
            parts = PurePosixPath(relative).parts
            if owned.intersection(parts) or prepared in parts:
                continue
            prefix, key, decl, track = _match(spec, relative)
            if decl is not None and decl.kind == "dir":
                members[prefix] = _aggregate(members.get(prefix), prefix, key, decl,
                                             track=track, role=role, size=size)
                continue
            declared_role = decl.role if decl is not None and decl.role else role
            members[prefix] = ArtifactMember(
                name=prefix, kind=decl.kind if decl is not None else "file",
                declared=key, role=declared_role, track=track, size_bytes=size, count=1)
        return cls(run_tag=run_tag, role=role,
                   manifest_name=manifest_name or _role_manifest(spec, role),
                   sealed=sealed, root=root, spec=spec, members=members)

    @staticmethod
    def _run_entries(root: Path) -> Iterator[tuple[str, int | None]]:
        """Every file under a run tree as ``(run-relative name, bytes)``."""
        for path in sorted(root.rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            yield path.relative_to(root).as_posix(), _stat_size(path)

    @staticmethod
    def _bundle_entries(bundle: Any) -> Iterator[tuple[str, int | None]]:
        """Every member of a verified bundle as ``(member name, bytes|None)``."""
        local = getattr(bundle, "local", None)
        for name in bundle.members():
            yield name, _stat_size(Path(local) / name) if local is not None else None

    # --------------------------------------------------------------- lookups
    def member(self, name: str) -> ArtifactMember:
        """One collected member by its run-relative name."""
        try:
            return self.members[name]
        except KeyError:
            raise KeyError(
                f"unknown artifact member {name!r}; collected: {self.names()}") from None

    def names(self) -> list[str]:
        """Every collected member name, sorted."""
        return sorted(self.members)

    def declared_names(self) -> list[str]:
        """Every declared artifact key (the spec's whole surface)."""
        return _declared_names(self.spec)

    def declared_tracks(self) -> tuple[str, ...]:
        """The tracks the declaration covers, in suite order."""
        return tuple(self.spec.tracks)

    def tracks(self) -> tuple[str, ...]:
        """The declared tracks actually present in this collection, in suite order."""
        present = {member.track for member in self.members.values() if member.track}
        return tuple(track for track in self.spec.tracks if track in present)

    def track_members(self, track: str) -> dict[str, ArtifactMember]:
        """Every member belonging to one track (``{}`` when the track emitted none)."""
        if track not in self.spec.tracks:
            raise KeyError(
                f"unknown track {track!r}; declared tracks: {list(self.spec.tracks)}")
        return {name: member for name, member in self.members.items()
                if member.track == track}

    def role_members(self, role: str) -> dict[str, ArtifactMember]:
        """The members one role's seal carries (the declared role vocabulary)."""
        value = _checked_role(self.spec, role)
        return {name: member for name, member in self.members.items()
                if member.role == value}

    def roles(self) -> tuple[str, ...]:
        """The declared role vocabulary (``BundleRole``'s values)."""
        return tuple(self.spec.roles)

    def manifest_for(self, role: str) -> str:
        """The canonical manifest member name for one role."""
        return _role_manifest(self.spec, role)

    def sealed_archives(self, *, run_tag: str | None = None,
                        fmt: str | None = None) -> dict[str, Path]:
        """The sealed archive address per role, resolved through the phone book."""
        from core.common import training_cfg

        run_tag = self.run_tag if run_tag is None else run_tag
        fmt = training_cfg().archives.format if fmt is None else fmt
        return {role: self.resolve(role, run_tag=run_tag, fmt=fmt)
                for role in self.spec.sealed_archives}

    # ------------------------------------------------------------- resolution
    @classmethod
    def resolve(cls, key: str, *, track: str | None = None,
                root: Path | str | None = None, spec: ArtifactSpec | None = None,
                **fields: object) -> Path:
        """Resolve one declared artifact key to a path (config SSOT, no literals).

        A classmethod because resolution is a pure function of the declaration,
        not of a collected run: a producer/consumer that names exactly ONE
        member asks here instead of spelling the literal, and needs no tree walk.
        ``root`` anchors a name-based declaration (the run root, or the parent of
        a track root); a phone-book declaration (``layouts``/``files``) resolves
        absolutely and ignores it. ``fields`` fill the declaration's placeholders
        (``run_tag``, ``fmt``, and any BundleSpec field name).
        """
        spec = artifacts_spec() if spec is None else spec
        return _resolve_decl(_decl_for(spec, key), key=key, track=track, root=root,
                             fields=fields, under_track=key in spec.track_artifacts)

    @classmethod
    def member_name(cls, key: str, *, track: str | None = None,
                    spec: ArtifactSpec | None = None) -> str:
        """The declared member FILENAME of one artifact key (config SSOT).

        A caller that holds a track root — or any tree this class does not
        address — asks for the declared filename here and places it itself. An
        address-only declaration (``via: layouts``/``files``, or a ``config``
        template needing fields) has no fixed literal filename and fails loud
        instead of returning a guess.
        """
        spec = artifacts_spec() if spec is None else spec
        name = _declared_name(_decl_for(spec, key), track=track)
        if name is None:
            raise ValueError(
                f"artifact {key!r} is address-only; it declares no member filename")
        return name

    # --------------------------------------------------------------- identity
    def identity(self) -> dict[str, Any]:
        """The STRUCTURAL identity of this collection — never a content digest.

        Run tag + member names + byte sizes + counts. Sizes may be ``None`` for an
        archive-backed handle (a member is sized only through a materialized
        tree); the counts and names still identify the collection, and no byte is
        ever read to build this.
        """
        members = [self.members[name] for name in self.names()]
        sizes = [member.size_bytes for member in members if member.size_bytes is not None]
        return {
            "schema": IDENTITY_SCHEMA,
            "run_tag": self.run_tag,
            "role": self.role,
            "manifest_name": self.manifest_name,
            "sealed": self.sealed,
            "member_count": len(members),
            "sized_count": len(sizes),
            "total_bytes": sum(sizes),
            "tracks": {track: len(self.track_members(track)) for track in self.tracks()},
            "members": [member.structural() for member in members],
        }

    def summary(self) -> dict[str, Any]:
        """The identity without the per-member rows (a one-line-per-run census)."""
        identity = self.identity()
        identity.pop("members")
        return identity


# ───────────────────────────────────────────────────────────────────────────
# classification + aggregation (module-private helpers)
# ───────────────────────────────────────────────────────────────────────────
def _match(spec: ArtifactSpec, relative: str
           ) -> tuple[str, str | None, ArtifactDecl | None, str | None]:
    """Classify one run-relative path: ``(member name, declared key, decl, track)``.

    The longest declared prefix wins, so a declared tree (``<track>__inference``)
    owns every file beneath it and a track's ``<track>__index`` is matched at the
    track root. A path no declaration matches returns ``(relative, None, None, track)``
    — collected, never silently dropped.
    """
    parts = PurePosixPath(relative).parts
    track = parts[0] if parts and parts[0] in spec.tracks else None
    best: tuple[int, str, str, ArtifactDecl, str | None] | None = None
    if track is not None:
        for key, decl in spec.track_artifacts.items():
            if decl.tracks and track not in decl.tracks:
                continue
            name = _declared_name(decl, track=track)
            if name is None:
                continue
            best = _consider(best, decl, _name_parts(name), parts, relative,
                             key, track, offset=1)
    else:
        for key, decl in spec.run_artifacts.items():
            name = _declared_name(decl)
            if name is None:
                continue
            best = _consider(best, decl, _name_parts(name), parts, relative,
                             key, None, offset=0)
    if best is None:
        return relative, None, None, track
    return best[1], best[2], best[3], best[4]


def _name_parts(name: str) -> tuple[str, ...]:
    return PurePosixPath(name).parts


def _consider(best, decl, name_parts, parts, relative, key, track, *, offset):
    """Keep the deepest declared match for one path (longest prefix wins)."""
    if decl.kind == "dir":
        if parts[offset:offset + len(name_parts)] != name_parts:
            return best
        end = offset + len(name_parts)
        if len(parts) < end:
            return best
        prefix, depth = PurePosixPath(*parts[:end]).as_posix(), end
    else:
        if parts[offset:] != name_parts:
            return best
        prefix, depth = relative, len(parts)
    if best is None or depth > best[0]:
        return (depth, prefix, key, decl, track)
    return best


def _aggregate(existing: ArtifactMember | None, prefix: str, key: str,
               decl: ArtifactDecl, *, track: str | None, role: str,
               size: int | None) -> ArtifactMember:
    """Fold one file into its declared tree member (byte total + file count)."""
    if existing is None:
        return ArtifactMember(name=prefix, kind="dir", declared=key,
                              role=decl.role or role, track=track,
                              size_bytes=size, count=1)
    total = None if existing.size_bytes is None or size is None \
        else existing.size_bytes + size
    return existing.model_copy(update={"size_bytes": total,
                                       "count": existing.count + 1})


def _resolve_decl(decl: ArtifactDecl, *, key: str, track: str | None,
                  root: Path | str | None, fields: dict[str, object],
                  under_track: bool) -> Path:
    """One declaration to a Path: phone-book-absolute or name-anchored."""
    if decl.via == "layouts":
        from core.common import artifact

        return artifact(decl.key, dict(fields))
    if decl.via == "files":
        from core.common import F

        if decl.key not in F:
            raise ValueError(
                f"artifact {key!r} names files.{decl.key}, which config/paths.yaml "
                f"files: does not declare")
        return F[decl.key]
    if decl.via == "config":
        value = _config_field(decl.key)
        if not isinstance(value, str):
            raise ValueError(
                f"artifact {key!r} names non-string training config value {decl.key!r}")
        rendered = _render(value, **fields)
        return Path(root) / rendered if root is not None else Path(rendered)
    name = _declared_name(decl, track=track, **fields)
    if name is None:
        raise ValueError(f"artifact {key!r} declares no resolvable name")
    if root is None:
        return Path(track, name) if under_track and track is not None else Path(name)
    base = Path(root, track) if under_track and track is not None else Path(root)
    return base / name
