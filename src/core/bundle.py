"""Bundle + BundlePipeline — the artifact and the process.

``Bundle`` is the one sealed artifact that travels generation -> training ->
post-training across Colab and Kaggle, GPU and CPU. It is a pydantic value type
over an on-disk archive plus the contract its role implies; integrity is checked
EXACTLY ONCE at the boundary (``Bundle.load``), then the object is trusted, so no
stage re-hashes or re-parses members.

``BundlePipeline`` is the only code allowed to perform bundling: generation
(prepare inputs) and finalize (select checkpoint + post-process + ablation). It
runs as a lane job via Kaggle or Colab from a sparse checkout; every consumer
loads a ``Bundle`` and never re-implements a bundling step.

Every name-like value comes from ``training_cfg().bundle`` (config SSOT); this
module spells no literal.
"""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from core.archive_reader import open_archive
from core.portable_archive import source_inventory, verify_archive_digest

#: The epoch-checkpoint directory name prefix (``checkpoint-281``); the role
#: contract and the layout walks below share this one literal.
_CHECKPOINT_PREFIX = "checkpoint-"


def _bundle_spec():
    """The bundle contract from config (lazy import: core.common owns config)."""
    from core.common import training_cfg
    return training_cfg().bundle


class BundleRole(str, Enum):
    """What a bundle is for; the role pins the manifest and allowed behavior."""

    inputs = "inputs"      # prepared inputs that drive training (generation)
    recovery = "recovery"  # resume state: every epoch's weights + optimizer
    result = "result"      # trained result: selected weights + reports + ablation


def manifest_name(role: BundleRole | str) -> str:
    """The manifest member name for a role (config SSOT)."""
    spec = _bundle_spec()
    return {
        BundleRole.inputs: spec.manifest_inputs,
        BundleRole.recovery: spec.manifest_recovery,
        BundleRole.result: spec.manifest_result,
    }[BundleRole(role)]


def _epoch_dirs(names) -> list[str]:
    """The ``<track>/_checkpoints/.../checkpoint-N`` dirs a member inventory holds.

    A checkpoint dir is named by the transport, so it is derived from member
    NAMES alone: no member is read to inventory the epochs a bundle carries.
    """
    spec = _bundle_spec()
    found: set[str] = set()
    for name in names:
        parts = PurePosixPath(name).parts
        for index, part in enumerate(parts):
            # A *directory* component (never the last, file-bearing part) named
            # ``checkpoint-N`` under the configured checkpoint root.
            if (index < len(parts) - 1 and part.startswith(_CHECKPOINT_PREFIX)
                    and spec.checkpoint_dir in parts[:index]):
                found.add(PurePosixPath(*parts[:index + 1]).as_posix())
    return sorted(found)


def _role_member_violations(role: BundleRole, names) -> list[str]:
    """Members that break ``role``'s membership contract (``[]`` = the set conforms).

    Role membership is a LOAD contract, not a writer-only convention, so the
    boundary refuses a role-violating set instead of trusting the caller. The
    contract is decidable from the sealed inventory alone -- no member bytes are
    read -- so the boundary's single integrity pass stays the single pass:

    * ``inputs``  -- prepared inputs drive training; they carry no weights.
    * ``result``  -- the SELECTED checkpoint per training run: several epochs
      sharing one checkpoint family is the recovery shape, never a result.
    * ``recovery``-- every epoch that recorded its state ships that state too
      (a recorded ``trainer_state.json`` without resume state was pruned).

    A caller that names a custom ``manifest_name`` has declared a different
    contract (the graph track's resume-capable superset bundle) and is not
    role-checked; a role archive always travels under its own canonical
    manifest, which is what every consumer looks for.
    """
    spec = _bundle_spec()
    names = tuple(names)
    if role is BundleRole.inputs:
        return [name for name in names
                if spec.checkpoint_dir in PurePosixPath(name).parts]

    if role is BundleRole.result:
        families: dict[str, list[str]] = {}
        for directory in _epoch_dirs(names):
            families.setdefault(PurePosixPath(directory).parent.as_posix(),
                                []).append(directory)
        return [f"{family} carries {len(dirs)} epochs (selected checkpoint only): "
                + ", ".join(Path(directory).name for directory in dirs)
                for family, dirs in sorted(families.items()) if len(dirs) > 1]

    violations = []
    for directory in _epoch_dirs(names):
        if f"{directory}/{spec.trainer_state_file}" not in names:
            continue
        resume_state = [name for name in names
                        if name.startswith(directory + "/")
                        and PurePosixPath(name).name in spec.resume_only_filenames]
        if not resume_state:
            violations.append(
                f"{directory} records {spec.trainer_state_file} without resume state")
    return violations


def _refuse_role_violation(role: BundleRole, names, *, where: str) -> None:
    """Refuse a role-violating member set; a no-op for a conforming one."""
    violations = _role_member_violations(role, names)
    if violations:
        raise ValueError(
            f"{where} violates the {role.value} bundle role contract: "
            + "; ".join(violations[:4]))


class _DirectoryReader:
    """Reader over an already-unpacked bundle tree (same shape as open_archive)."""

    def __init__(self, root: Path):
        self.root = root

    def namelist(self) -> list[str]:
        return sorted(p.relative_to(self.root).as_posix()
                      for p in self.root.rglob("*") if p.is_file())

    def read(self, name: str) -> bytes:
        return (self.root / name).read_bytes()

    def open(self, name: str):
        return (self.root / name).open("rb")

    def extract(self, name: str, destination: Path | str) -> str:
        target = Path(destination) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((self.root / name).read_bytes())
        return str(target)


class Bundle(BaseModel):
    """A verified handle over one sealed archive (+ its role contract).

    Constructed only through :meth:`load` (or :meth:`from_directory` for a tree
    that is already unpacked), so the boundary integrity check cannot be skipped.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    role: BundleRole
    path: Path
    manifest_name: str
    manifest: dict[str, Any] = Field(default_factory=dict)
    digest: str | None = None
    local: Path | None = None  # materialized working tree, set by materialize()
    #: Member names captured by the boundary verify; a path-backed handle then
    #: answers members()/has() without re-parsing (or re-inflating) the archive.
    member_names: tuple[str, ...] = ()

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, path: Path | str, role: BundleRole | str, *,
             expected_digest: str | None = None,
             manifest_name: str | None = None) -> "Bundle":
        """Verify ``path`` once at the boundary, then return a trusted handle.

        ``expected_digest`` is the transport's recorded sha256 (Colab ``.sha256``,
        Kaggle receipt ``archive_sha256``, git transport inventory). A mismatch is
        the corruption guard: fail loud, keep the partial, never install.

        The boundary also enforces the role's membership contract (see
        :func:`_role_member_violations`), so a role-violating set never loads as
        a trusted handle. Only a canonical-manifest archive is role-checked: a
        caller naming its own manifest has declared another contract.
        """
        role = BundleRole(role)
        path = Path(path)
        name = manifest_name or globals()["manifest_name"](role)
        names: list[str] = []
        manifest, observed = verify_archive_digest(path, name, names=names)
        if expected_digest is not None and observed != expected_digest:
            raise ValueError(
                f"bundle failed the boundary integrity check: {path}\n"
                f"  observed sha256={observed}\n  expected sha256={expected_digest}")
        spec = _bundle_spec()
        if role is BundleRole.result and spec.run_tag_key not in manifest:
            raise ValueError(f"bundle manifest {name} carries no {spec.run_tag_key}: {path}")
        if name == globals()["manifest_name"](role):
            _refuse_role_violation(role, names, where=f"bundle {path}")
        return cls(role=role, path=path, manifest_name=name,
                   manifest=manifest, digest=observed,
                   member_names=tuple(sorted(names)))

    @classmethod
    def from_directory(cls, directory: Path | str, role: BundleRole | str, *,
                       manifest_name: str | None = None) -> "Bundle":
        """Wrap an already-unpacked tree (no archive to verify)."""
        role = BundleRole(role)
        directory = Path(directory)
        name = manifest_name or globals()["manifest_name"](role)
        manifest_path = directory / name
        manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                    if manifest_path.is_file() else {})
        return cls(role=role, path=directory, manifest_name=name,
                   manifest=manifest, digest=None, local=directory)

    # --------------------------------------------------------------- members
    @contextmanager
    def reader(self):
        """One read session over this bundle (a verified handle; no re-verify).

        A caller that needs several members opens this once instead of calling
        :meth:`read` per member, so an archive is parsed (and, for a zstd tar,
        inflated) exactly once per stage.
        """
        if self.local is not None:
            yield _DirectoryReader(self.local)
            return
        with open_archive(self.path) as archive:
            yield archive

    def members(self) -> list[str]:
        if self.local is not None:
            return sorted(p.relative_to(self.local).as_posix()
                          for p in self.local.rglob("*") if p.is_file())
        if self.member_names:
            return list(self.member_names)
        with open_archive(self.path) as archive:
            return list(archive.namelist())

    def has(self, member: str) -> bool:
        return member in set(self.members())

    def read(self, member: str) -> bytes:
        with self.reader() as source:
            return source.read(member)

    def read_json(self, member: str) -> Any:
        return json.loads(self.read(member))

    def materialize(self, destination: Path | str) -> "Bundle":
        """Unpack every member into ``destination`` once; return the dir-backed view.

        The bundle's own manifest is container metadata, not a tree member, so it
        is never written into the materialized tree.
        """
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        if self.local is None:
            with open_archive(self.path) as archive:
                for member in archive.namelist():
                    if member == self.manifest_name:
                        continue
                    archive.extract(member, destination)
        else:
            for member in self.members():
                target = destination / member
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((self.local / member).read_bytes())
        return self.model_copy(update={"local": destination})

    # ------------------------------------------------------------- contract
    def run_tag(self) -> str:
        return str(self.manifest.get(_bundle_spec().run_tag_key, ""))

    def _root(self) -> Path:
        return self.local if self.local is not None else self.path

    def track_inventory(self, track: str) -> dict[str, Any]:
        return self.read_json(f"{track}/" + _bundle_spec().inventory_file)

    def track_complete(self, track: str) -> bool:
        spec = _bundle_spec()
        try:
            marker = self.read_json(f"{track}/" + spec.complete_file)
        except (KeyError, FileNotFoundError):
            return False
        return marker.get("status") == spec.complete_status

    def track_dir(self, track: str) -> Path:
        return self._root() / track

    # ----------------------------------------------------------- checkpoints
    @staticmethod
    def _recorded_members(track_root: Path, recorded: Path) -> list[Path]:
        """Locate a marker-recorded member by NAME under the track's tree.

        The one documented transport rule: a marker records the path on the
        machine that trained, so the identity that survives transport is the
        name pair — the parent-qualified search first, a bare-name search when
        the recorded parent resolves to nothing (a bare recorded name has no
        parent to qualify with). Returns every match, in name order, so a caller
        can tell a unique resolution from an ambiguous one.
        """
        parent = recorded.parent.name
        candidates = (
            sorted(track_root.rglob(f"{parent}/{recorded.name}")) if parent else []
        )
        if not candidates:
            candidates = sorted(track_root.rglob(recorded.name))
        return candidates

    def checkpoint(self, track: str) -> Path | None:
        """The SELECTED checkpoint dir for ``track`` (resolved once, deterministically).

        Text records the trainer-selected best in ``trainer_state.json`` (ranked by
        metric then step, mirroring resolve_best_checkpoint); a graph track records
        it in its ``best_checkpoint`` marker. Result bundles carry only this one;
        recovery bundles carry all epoch checkpoints. Inputs carry no weights.

        A marker whose recorded name resolves to MORE than one member is refused:
        the rule above identifies a member by name, so several matches are a stale
        state whose choice must be made by the caller, never silently by name order.
        """
        if self.role is BundleRole.inputs:
            return None
        spec = _bundle_spec()
        root = self._root()
        track_root = self._track_root(root, track)
        best: tuple[tuple[float, int], Path] | None = None
        for state_path in sorted(track_root.rglob(_CHECKPOINT_PREFIX + "*/" + spec.trainer_state_file)):
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            recorded = state.get(spec.trainer_best_key)
            if not recorded:
                continue
            candidate = state_path.parent.parent / Path(str(recorded)).name
            if not candidate.is_dir():
                continue
            rank = (float(state.get(spec.trainer_metric_key, float("-inf"))),
                    int(state.get(spec.trainer_step_key, 0)))
            if best is None or rank > best[0]:
                best = (rank, candidate)
        if best is not None:
            return best[1]
        for marker in sorted(track_root.rglob(spec.best_checkpoint_glob)):
            try:
                recorded = Path(str(json.loads(marker.read_text(encoding="utf-8"))
                                    .get(spec.best_checkpoint_path_key, "")))
            except (OSError, ValueError):
                continue
            if not recorded.name:
                continue
            # The recorded path may be a remote machine's absolute path (the
            # marker is written where the track trained), so the member is
            # located by name under this track's tree; see _recorded_members for
            # the parent-qualified/bare-name rule, which is shared with the
            # publication guard so the two can never disagree.
            candidates = self._recorded_members(track_root, recorded)
            if len(candidates) > 1:
                matches = ", ".join(member.relative_to(root).as_posix()
                                    for member in candidates)
                raise ValueError(
                    f"ambiguous selected checkpoint: {track} "
                    f"({recorded.name!r} matches {len(candidates)} members: {matches})")
            if candidates:
                return candidates[0]
        return None

    def checkpoints(self, track: str) -> list[Path]:
        """Every materialized checkpoint dir for ``track`` (recovery role).

        Role enforcement: inputs bundles carry no weights, so this is always
        empty there; recovery carries every epoch; result carries the single
        selected directory the seal kept. Enumerates a materialized (dir-backed)
        tree; listing an archive's epochs is not a Bundle operation.
        """
        if self.role is BundleRole.inputs:
            return []
        spec = _bundle_spec()
        checkpoint_dir = self._track_root(self._root(), track) / spec.checkpoint_dir
        if not checkpoint_dir.is_dir():
            return []
        return sorted(p for p in checkpoint_dir.rglob(_CHECKPOINT_PREFIX + "*") if p.is_dir())

    def selected_checkpoint_dirs(self) -> frozenset[str]:
        """Posix dirs (relative to this bundle's root) of selected checkpoints.

        The result-member contract: text records the trainer-selected best in
        every ``trainer_state.json`` (ranked metric then step, exactly like
        :func:`training.validation_inference.resolve_best_checkpoint`); each
        graph track records its selected checkpoint in ``*__best_checkpoint.json``.
        Every other ``checkpoint-N`` tree is resume-only.
        """
        if self.role is BundleRole.inputs:
            return frozenset()
        spec = _bundle_spec()
        root = self._root()
        selected: set[str] = set()
        best: tuple[tuple[float, int], Path] | None = None
        for state_path in sorted(root.rglob(_CHECKPOINT_PREFIX + "*/" + spec.trainer_state_file)):
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            recorded = state.get(spec.trainer_best_key)
            if not recorded:
                continue
            directory = state_path.parent.parent / Path(str(recorded)).name
            if not directory.is_dir():
                continue
            rank = (float(state.get(spec.trainer_metric_key, float("-inf"))),
                    int(state.get(spec.trainer_step_key, 0)))
            if best is None or rank > best[0]:
                best = (rank, directory)
        if best is not None:
            selected.add(best[1].relative_to(root).as_posix())
        for marker in sorted(root.rglob(spec.best_checkpoint_glob)):
            try:
                recorded = Path(str(json.loads(marker.read_text(encoding="utf-8"))
                                    .get(spec.best_checkpoint_path_key, "")))
            except (OSError, ValueError):
                continue
            if not recorded.name:
                continue
            for found in root.rglob(f"{recorded.parent.name}/{recorded.name}"):
                selected.add(found.parent.relative_to(root).as_posix())
                break
        return frozenset(selected)

    @staticmethod
    def is_result_member(relative: str, *,
                         selected_checkpoints: frozenset[str] = frozenset()) -> bool:
        """Whether a root-relative path belongs in a RESULT bundle.

        The single member predicate: checkpoint trees ship only when the
        checkpoint is selected, resume-only state and profiling/caches are
        dropped, and the archive walk and the per-track inventory can never
        disagree. Names come from :class:`BundleSpec`.
        """
        spec = _bundle_spec()
        parts = Path(relative).parts
        if not parts or any(part in spec.result_excluded_dirs for part in parts):
            return False
        if spec.prepared_inputs_dir in parts:
            # Prepared inputs a finalize job extracted to run its CPU work: they
            # are an input to the process, never a deliverable of the bundle.
            return False
        if parts[-1] in spec.resume_only_filenames:
            return False
        if parts[-1] in {".env", "config.local"}:
            return False
        if any(part.endswith(".publication") or part.endswith("__payload")
               for part in parts):
            return False
        if "_artifact_publications" in parts and not relative.endswith(".json"):
            return False
        if ".dvc" in parts and "cache" in parts:
            return False
        if spec.checkpoint_dir in parts:
            if not selected_checkpoints:
                return False
            member = PurePosixPath(relative)
            return any(member == PurePosixPath(selected)
                       or PurePosixPath(selected) in member.parents
                       for selected in selected_checkpoints)
        return True

    def collect_result_members(self, *,
                               selected_checkpoints: frozenset[str] | None = None
                               ) -> dict[str, Path]:
        """The member set this bundle would seal as a RESULT (selected-only).

        Role enforcement: only a ``result`` bundle has a result-member set; the
        selection is resolved from this bundle's own tree, so no stage re-parses
        the archive and no caller re-derives the predicate.
        """
        if self.role is not BundleRole.result:
            raise ValueError(
                f"collect_result_members requires the result role, got {self.role.value}")
        root = self._root()
        if selected_checkpoints is None:
            selected_checkpoints = self.selected_checkpoint_dirs()
        return {p.relative_to(root).as_posix(): p for p in root.rglob("*")
                if p.is_file() and not p.is_symlink()
                and self.is_result_member(p.relative_to(root).as_posix(),
                                          selected_checkpoints=selected_checkpoints)}

    @classmethod
    def seal_archive(cls, output: Path | str, files: dict[str, Path], *,
                     role: BundleRole | str, metadata: dict[str, Any] | None = None,
                     inline: dict[str, str] | None = None, profile: bool = False,
                     manifest_name: str | None = None) -> "Bundle":
        """Write ``files`` as one sealed archive for ``role`` — the only writer.

        Every bundling step seals through here: the writer hashes each source
        exactly once while writing and verifies the written bytes, and the
        sealed archive's whole-file SHA256 is captured during that same write,
        so the returned handle's ``digest`` is the transport token with no
        re-read and no second integrity pass.

        The returned handle is the writer's own: its manifest mirrors the
        archive's (caller metadata plus the member inventory the writer froze),
        and its member list is the set that was written — so a caller that needs
        the completion contract's inventory or member names reuses this handle
        instead of loading the bytes back. A role-violating member set is
        refused before the bytes land.
        """
        from core.portable_archive import write_archive
        role = BundleRole(role)
        spec = _bundle_spec()
        name = manifest_name or globals()["manifest_name"](role)
        output = Path(output)
        inline = dict(inline or {})
        # A materialized tree can carry the source bundle's own manifest; it is
        # container metadata and must never collide with the sealed manifest.
        files = {target: source for target, source in files.items() if target != name}
        if not files:
            raise ValueError("refusing to seal an empty bundle")
        payload = dict(metadata or {})
        if role is BundleRole.result and spec.run_tag_key not in payload:
            # A result bundle must be identifiable by run tag, or the boundary
            # check cannot accept it; fail at the writer rather than on load.
            raise ValueError("sealing a result bundle requires a run tag")
        if name == globals()["manifest_name"](role):
            _refuse_role_violation(role, (*files, *inline),
                                   where=f"the bundle to seal at {output}")
        # The inventory the sealed manifest carries, from the same memoized
        # source-digest pass the writer uses: the handle below can then answer
        # the completion contract's member inventory without re-reading bytes.
        inventory = source_inventory(files, inline)
        hasher = hashlib.sha256()
        write_archive(output, files, manifest_name=name, metadata=payload,
                      inline=inline, profile=profile, digest=hasher)
        return cls(role=role, path=output, manifest_name=name,
                   manifest={**payload, spec.files_key: inventory},
                   digest=hasher.hexdigest(),
                   member_names=tuple(sorted({*files, *inline, name})))

    def seal_result(self, output: Path | str, *,
                    metadata: dict[str, Any] | None = None,
                    profile: bool = False) -> "Bundle":
        """Write this bundle's result-only members as one sealed RESULT archive.

        Role enforcement decides the member set (selected checkpoint only); the
        seal itself is :meth:`seal_archive`, so the archive is written and
        verified exactly once.
        """
        if self.role is not BundleRole.result:
            raise ValueError(f"seal_result requires the result role, got {self.role.value}")
        spec = _bundle_spec()
        files = self.collect_result_members()
        if not files:
            raise ValueError("refusing to seal an empty result bundle")
        return self.seal_archive(
            output, files, role=BundleRole.result, profile=profile,
            metadata={spec.run_tag_key: self.run_tag(), **(metadata or {})})

    @classmethod
    def trusted(cls, path: Path | str, role: BundleRole | str, manifest: dict[str, Any], *,
                manifest_name: str | None = None) -> "Bundle":
        """Wrap a boundary that was just verified by its writer (no re-check)."""
        role = BundleRole(role)
        return cls(role=role, path=Path(path),
                   manifest_name=manifest_name or globals()["manifest_name"](role),
                   manifest=manifest, digest=None)

    @staticmethod
    def _track_root(root: Path, track: str) -> Path:
        """Locate the track's artifact subtree under either the suite root or a
        track dir (the layout differs only by where the caller start from)."""
        direct = root / track
        return direct if direct.is_dir() else root

    # -------------------------------------------------------------- ablation
    def ablation_templates(self, track: str) -> bool:
        """True when the bundle ships the prepared ablation samples for ``track``."""
        spec = _bundle_spec()
        target = f"{spec.ablation_templates_dir}/{track}/{spec.ablation_request_file}"
        return any(member == target or member.endswith("/" + target)
                   for member in self.members())

    def ablation_skipped(self) -> bool:
        """True when a track's event log records a deliberate ablation-export skip."""
        spec = _bundle_spec()
        for member in self.members():
            if Path(member).name not in (spec.worker_events_file, spec.suite_events_file):
                continue
            for line in self.read(member).decode(errors="replace").splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if (event.get("phase") in spec.ablation_skip_phases
                        and event.get("status") == spec.ablation_skip_status):
                    return True
        return False


class BundlePipeline(BaseModel):
    """The bundling process — the only place generation/finalize/ablation run.

    A pipeline is parameterized entirely by config (payload type, lane, device,
    sparse-checkout paths, role) and executed as a lane job via Kaggle or Colab.
    Consumers never construct or run bundling steps themselves.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    role: BundleRole
    device: str
    lane: str
    #: Repository-relative paths the sparse checkout must carry for this job.
    sparse_paths: tuple[str, ...] = ()
    #: Where the sealed bundle is written/found.
    output: Path | None = None
    #: The three-track suite config the pipeline prepares from (config SSOT).
    config: Path | None = None
    #: Optional explicit preparation run directory (defaults beside the output).
    run_dir: Path | None = None
    #: The verified prepared-inputs archive a finalize job consumes.
    inputs: Path | None = None
    #: Where a finalize job materializes the result tree (defaults to
    #: ``output.parent / run_tag``), and where it extracts the prepared inputs
    #: (defaults to ``work_dir / bundle.prepared_inputs_dir``).
    work_dir: Path | None = None
    prepared_dir: Path | None = None
    #: The provenance string a finalize job records (defaults to the bundle
    #: spec's non-local location; the local lane passes its own).
    postprocess_location: str | None = None
    #: Extra manifest entries the sealing step records (transport identity such
    #: as the source archive digests); the step's own keys always win.
    metadata: dict[str, Any] = Field(default_factory=dict)

    def prepare_inputs(self) -> "Bundle":
        """Generation: prepare the inputs bundle (CPU)."""
        from model_tracks.bundle_steps import prepare_inputs
        return prepare_inputs(self)

    def finalize(self, result: Bundle) -> "Bundle":
        """Finalize: select checkpoint, post-process, ablate, seal the result bundle."""
        from model_tracks.bundle_steps import finalize
        return finalize(self, result)

