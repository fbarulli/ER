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

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from core.archive_reader import open_archive
from core.portable_archive import is_result_archive_member, verify_archive_digest


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

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, path: Path | str, role: BundleRole | str, *,
             expected_digest: str | None = None,
             manifest_name: str | None = None) -> "Bundle":
        """Verify ``path`` once at the boundary, then return a trusted handle.

        ``expected_digest`` is the transport's recorded sha256 (Colab ``.sha256``,
        Kaggle receipt ``archive_sha256``, git transport inventory). A mismatch is
        the corruption guard: fail loud, keep the partial, never install.
        """
        role = BundleRole(role)
        path = Path(path)
        name = manifest_name or manifest_name(role)
        manifest, observed = verify_archive_digest(path, name)
        if expected_digest is not None and observed != expected_digest:
            raise ValueError(
                f"bundle failed the boundary integrity check: {path}\n"
                f"  observed sha256={observed}\n  expected sha256={expected_digest}")
        spec = _bundle_spec()
        if role is not BundleRole.recovery and spec.run_tag_key not in manifest:
            raise ValueError(f"bundle manifest {name} carries no {spec.run_tag_key}: {path}")
        return cls(role=role, path=path, manifest_name=name,
                   manifest=manifest, digest=observed)

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
    def members(self) -> list[str]:
        if self.local is not None:
            return sorted(p.relative_to(self.local).as_posix()
                          for p in self.local.rglob("*") if p.is_file())
        with open_archive(self.path) as archive:
            return list(archive.namelist())

    def has(self, member: str) -> bool:
        return member in set(self.members())

    def read(self, member: str) -> bytes:
        if self.local is not None:
            return (self.local / member).read_bytes()
        with open_archive(self.path) as archive:
            return archive.read(member)

    def read_json(self, member: str) -> Any:
        return json.loads(self.read(member))

    def materialize(self, destination: Path | str) -> "Bundle":
        """Unpack every member into ``destination`` once; return the dir-backed view."""
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        if self.local is None:
            with open_archive(self.path) as archive:
                for member in archive.namelist():
                    archive.extract(member, destination)
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
    def checkpoint(self, track: str) -> Path | None:
        """The SELECTED checkpoint dir for ``track`` (resolved once, deterministically).

        Text records the trainer-selected best in ``trainer_state.json`` (ranked by
        metric then step, mirroring resolve_best_checkpoint); a graph track records
        it in its ``best_checkpoint`` marker. Result bundles carry only this one;
        recovery bundles carry all epoch checkpoints.
        """
        spec = _bundle_spec()
        root = self._root()
        track_root = self._track_root(root, track)
        best: tuple[tuple[float, int], Path] | None = None
        for state_path in sorted(track_root.rglob("checkpoint-*/" + spec.trainer_state_file)):
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
            candidate = track_root / recorded
            if recorded.name and candidate.is_file():
                return candidate
        return None

    def checkpoints(self, track: str) -> list[Path]:
        """Every materialized checkpoint dir for ``track`` (recovery role)."""
        spec = _bundle_spec()
        checkpoint_dir = self._track_root(self._root(), track) / spec.checkpoint_dir
        if not checkpoint_dir.is_dir():
            return []
        return sorted(p for p in checkpoint_dir.rglob("checkpoint-*") if p.is_dir())

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

    def prepare_inputs(self) -> "Bundle":
        """Generation: prepare the inputs bundle (CPU)."""
        from model_tracks.bundle_steps import prepare_inputs
        return prepare_inputs(self)

    def finalize(self, result: Bundle) -> "Bundle":
        """Finalize: select checkpoint, post-process, ablate, seal the result bundle."""
        from model_tracks.bundle_steps import finalize
        return finalize(self, result)


def collect_result_members(root: Path, *, selected_checkpoints: frozenset[str] = frozenset()
                           ) -> dict[str, Path]:
    """The member set a RESULT bundle may carry (selected-only, no profiling)."""
    return {p.relative_to(root).as_posix(): p for p in root.rglob("*")
            if p.is_file() and not p.is_symlink()
            and is_result_archive_member(p.relative_to(root).as_posix(),
                                         selected_checkpoints=selected_checkpoints)}
