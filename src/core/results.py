"""src/core/results.py — the project Results: ONE owner of the result surface.

WHY THIS MODULE EXISTS
----------------------
Between training and post-training analysis there is a set of documents that
belongs to neither lane: the post-training ablation REQUEST the training outputs
land for the GPU lane, the frozen prepared tensors it names, the encoded
VECTORS, the frozen ablation REPORT, the sealed BASELINE THRESHOLD binding, the
run RECEIPT, and the Laya HOLDOUT verification report. Every consumer re-spelled those names: the request
comes from the bundle contract, vectors/report from ``paths.yaml`` layouts, and
``baseline_threshold.json``, ``prepared_inputs.npz``,
``post_training_ablation.json`` and ``holdout_report.json`` are inline literals
in ``model_tracks.baseline_ablation`` / ``post_training_ablation`` /
``staged_ablation`` / ``cli.laya_lane``.

:class:`Results` is the object that OWNS that surface for one suite run:

  path(name, track)     resolve one declared result role via the SSOT accessor
  paths(tracks)         every addressed result of the run
  request/report/...    the concrete per-track lookups (no literals at call sites)
  produce_request(...)  bind a frozen template onto the training-selected checkpoint
  write_request(...)    land that request where the GPU lane reads it
  saved(track)          the five-member saved-ablation export the consumer re-verifies
  frozen_threshold() / baseline() / load_report()   consume the landed results
  receipt()             the run-level post-training receipt
  holdout_report()      the Laya component-disjoint holdout verification report
  thresholds()          the operating-point ladder (references the ONE SSOT)
  identity()            STRUCTURE only: run tag + member names + byte sizes + counts

DECLARATION
-----------
``config/results.yaml`` declares every role plus the ONE already-declared
address it resolves through (``paths.yaml`` ``layouts:``, the bundle contract,
or this file's own names). A name that already has a home is REFERENCED, never
re-spelled. Resolution goes through the SAME accessors the rest of the tree
calls (``core.common.artifact``, ``core.bundle.bundle_spec``), so a ``Results``
member and the current inline call site return the identical absolute Path.

DIVISION OF LABOR WITH ``Artifacts``
------------------------------------
``Artifacts`` collects the RAW outputs of training and of the Bundle: the
checkpoints, the per-track artifacts, the metric CSVs, traces/events and sealed
archives. ``Results`` owns the DERIVED documents that sit BETWEEN the training
lane and its post-training analysis: the request the GPU lane consumes, and the
report/vectors/threshold/holdout documents analysis reads back. A raw
trainer output (a checkpoint, the trained encoder vectors, the per-track report
manifest) is an artifact owned by ``Artifacts``; a document the post-training
lane consumes or produces is a result.

STRUCTURE, NOT A VERDICT
------------------------
Nothing here derives an id from content, versions anything or judges
freshness. :meth:`Results.identity` is a structural census (run tag, member
names, byte sizes, counts) so a reader can tell two runs apart; a missing
member is absence in that census, never a raise and never a gate.
"""
from __future__ import annotations

import copy
import json
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.run_log import RunLogger

_LOG = RunLogger(__name__)

#: The declared result-spec document lives beside the other config documents.
CONFIG_NAME = "results.yaml"

#: Scope tags a declared role may carry.
_TRACK = "track"
_RUN = "run"
_PROJECT = "project"


class ResultRoleSpec(BaseModel):
    """One declared RESULT role: the ONE address it resolves through.

    ``via`` selects the phone book the role is looked up in:

      ``layouts``   ``core.common.artifact(key, {run_tag, track})``
      ``bundle``    ``core.bundle.bundle_spec().<key>``, resolved under the
                    track's ablation dir
      ``names``     this spec's own ``names[key]``, resolved under the scope dir

    A key the phone book does not declare is refused by
    :meth:`Results.from_config`, so a typo can never silently resolve to
    nothing.
    """

    model_config = ConfigDict(extra="forbid")

    via: Literal["layouts", "bundle", "names"]
    scope: Literal["track", "run", "project"] = _TRACK
    key: str = Field(min_length=1)

    @model_validator(mode="after")
    def _key_matches_via(self) -> "ResultRoleSpec":
        """A bundle role is track-scoped; nothing else constrains the pair."""
        if self.via == "bundle" and self.scope != _TRACK:
            raise ValueError("a bundle role is track-scoped")
        return self


class ThresholdSpec(BaseModel):
    """The operating-point ladder: a reference to the ONE SSOT accessor."""

    model_config = ConfigDict(extra="forbid")

    via: Literal["common"] = "common"
    key: str = Field(min_length=1)


class ResultsSpec(BaseModel):
    """The result surface's declaration (``config/results.yaml``).

    Shape-only validation lives here; the phone books (layouts, bundle, the
    accessor behind the threshold ladder) are checked by
    :meth:`Results.from_config`, so the spec model stays usable with a
    synthetic config in tests.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    run_layout: str = Field(min_length=1)
    ablation_dir: str = Field(min_length=1)
    tracks: tuple[str, ...] = Field(min_length=1)
    threshold_ladder: ThresholdSpec
    holdout_dir: str = Field(min_length=1)
    roles: dict[str, ResultRoleSpec] = Field(min_length=1)
    names: dict[str, str] = Field(min_length=1)
    sets: dict[str, tuple[str, ...]] = Field(min_length=1)

    @model_validator(mode="after")
    def _references_resolve(self) -> "ResultsSpec":
        """Every name key and set member names a real role."""
        for role_name, role in self.roles.items():
            if role.via == "names" and role.key not in self.names:
                raise ValueError(
                    f"results role {role_name!r} names undeclared key {role.key!r}"
                )
        for set_name, members in self.sets.items():
            unknown = sorted(set(members) - set(self.roles))
            if unknown:
                raise ValueError(
                    f"results set {set_name!r} names undeclared role(s) {unknown}"
                )
        if len(set(self.tracks)) != len(self.tracks):
            raise ValueError("results tracks must be unique")
        return self


def results_config_path() -> Path:
    """The declared result-spec document (``config/results.yaml``)."""
    from core.common import TRAIN_ROOT

    return TRAIN_ROOT / "config" / CONFIG_NAME


def results_spec() -> ResultsSpec:
    """Read + validate the role declaration through the ONE read+validate home."""
    from core.common import load_validated_yaml

    return load_validated_yaml(
        results_config_path(), ResultsSpec, label="Results role declaration"
    )


class Results(BaseModel):
    """The results of one suite run: owner of its roles and its structure.

    Built from the validated config SSOT by :meth:`from_config` (or the cached
    :func:`results` accessor). Frozen, so a consumer cannot mutate the resolved
    surface in place.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    run_tag: str
    #: The run root — results/model_tracks/{run_tag}.
    root: Path
    ablation_dir: str
    tracks: tuple[str, ...]
    threshold_ladder: ThresholdSpec
    holdout_dir: str
    roles: dict[str, ResultRoleSpec]
    names: dict[str, str]
    sets: dict[str, tuple[str, ...]]

    # ------------------------------------------------------------ construction
    @classmethod
    def from_config(cls, run_tag: str, *, spec: ResultsSpec | None = None) -> "Results":
        """Resolve the declared roles against the validated config SSOT.

        ``spec`` is injectable so a test can pin resolution without touching the
        real config; production callers use the default.
        """
        from core.common import artifact

        spec = results_spec() if spec is None else spec
        if not run_tag:
            raise ValueError("Results requires a non-empty run tag")
        return cls(
            name=spec.name,
            run_tag=run_tag,
            root=artifact(spec.run_layout, {"run_tag": run_tag}),
            ablation_dir=spec.ablation_dir,
            tracks=tuple(spec.tracks),
            threshold_ladder=spec.threshold_ladder,
            holdout_dir=spec.holdout_dir,
            roles=dict(spec.roles),
            names=dict(spec.names),
            sets={key: tuple(value) for key, value in spec.sets.items()},
        )

    # ------------------------------------------------------------------ paths
    def _role(self, name: str) -> ResultRoleSpec:
        """The declared role, or fail loud with the declared set."""
        role = self.roles.get(name)
        if role is None:
            raise KeyError(
                f"unknown results role {name!r}; declared roles: {sorted(self.roles)}"
            )
        return role

    def _require_track(self, track: str) -> str:
        """One declared track, or fail loud with the declared set."""
        if track not in self.tracks:
            raise ValueError(
                f"unknown results track {track!r}; declared tracks: {list(self.tracks)}"
            )
        return track

    def _project_dir(self) -> Path:
        """The RESULTS root plus the declared holdout landing dir."""
        from core.common import RESULTS

        return RESULTS / self.holdout_dir

    def path(self, name: str, track: str | None = None) -> Path:
        """Resolve one declared result role to an absolute Path.

        A ``track``-scoped role requires the track; a ``run``/``project`` role
        refuses one (a literal track argument would otherwise be silently
        ignored).
        """
        role = self._role(name)
        if role.scope == _TRACK:
            if track is None:
                raise ValueError(f"results role {name!r} is track-scoped; pass a track")
            return self._resolve(name, role, self._track_dir(track), track=track)
        if track is not None:
            raise ValueError(
                f"results role {name!r} is {role.scope}-scoped; it takes no track"
            )
        if role.scope == _RUN:
            return self._resolve(name, role, self.root)
        return self._resolve(name, role, self._project_dir())

    def _track_dir(self, track: str) -> Path:
        """The per-track folder holding that track's ablation results."""
        return self.root / self._require_track(track) / self.ablation_dir

    def _resolve(
        self, name: str, role: ResultRoleSpec, folder: Path, track: str | None = None
    ) -> Path:
        """One role rendered through the SSOT accessor its ``via`` names."""
        if role.via == "names":
            return folder / self.names[role.key]
        if role.via == "bundle":
            # The bundle contract is config/training.yaml ``bundle:`` read
            # through the ONE validated accessor, exactly as
            # ``core.bundle.bundle_spec()`` does; Results names no transport.
            from core.common import training_cfg

            return folder / getattr(training_cfg().bundle, role.key)
        if role.via == "layouts":
            from core.common import artifact

            return artifact(role.key, {"run_tag": self.run_tag, "track": track})
        raise ValueError(f"results role {name!r} declares unknown via {role.via!r}")

    def paths(self, tracks: tuple[str, ...] | None = None) -> dict[str, Path]:
        """Every addressed result of the run, keyed ``<track>.<role>``/``<role>``.
        """
        selected = self.tracks if tracks is None else tuple(tracks)
        resolved: dict[str, Path] = {}
        for name, role in self.roles.items():
            if role.scope == _TRACK:
                for track in selected:
                    resolved[f"{track}.{name}"] = self.path(name, track)
            else:
                resolved[name] = self.path(name)
        return resolved

    # ----------------------------------------------------------- lookups
    def request(self, track: str) -> Path:
        """The post-training request the GPU lane reads for one track."""
        return self.path("request", track)

    def prepared_inputs(self, track: str) -> Path:
        """The frozen tensors the request names for one track."""
        return self.path("prepared_inputs", track)

    def vectors(self, track: str) -> Path:
        """The encoded vectors/scores export for one track."""
        return self.path("vectors", track)

    def report(self, track: str) -> Path:
        """The frozen ablation report for one track."""
        return self.path("report", track)

    def baseline_threshold(self, track: str) -> Path:
        """The sealed baseline calibration binding for one track."""
        return self.path("baseline_threshold", track)

    def receipt(self) -> Path:
        """The run-level receipt the post-training consumer writes."""
        return self.path("receipt")

    def holdout_report(self) -> Path:
        """The Laya component-disjoint holdout verification report."""
        return self.path("holdout_report")

    def saved(self, track: str) -> dict[str, Path]:
        """The declared saved-ablation export the consumer re-verifies, by role.

        The member set is declared once (``sets.saved_ablation``); the class
        never re-lists it.
        """
        self._require_track(track)
        return {name: self.path(name, track) for name in self.sets["saved_ablation"]}

    # ------------------------------------------------------- request production
    def suite_address(self, path: str | Path) -> str:
        """The request-relative ``@suite/<rel>`` address of a path under the root.

        The ablation request's portable resolver maps ``@suite`` to the run
        root, so a training output is addressed exactly as the GPU lane
        re-locates it. A path outside the root fails loud here rather than
        producing an unresolvable address.
        """
        relative = Path(path).resolve().relative_to(self.root.resolve())
        return "@suite/" + relative.as_posix()

    def produce_request(
        self,
        track: str,
        *,
        template: dict[str, Any],
        checkpoint: str | Path,
        checkpoint_identity: str,
    ) -> dict[str, Any]:
        """Bind a frozen template request onto the training-selected checkpoint.

        The template is the frozen prepared request; the training output is the
        selected checkpoint plus the identity its owner recorded. The identity
        is data handed in, never derived here. The returned document is what
        :meth:`write_request` lands.
        """
        self._require_track(track)
        document = copy.deepcopy(dict(template))
        selected = self.suite_address(checkpoint)
        sources = dict(document.get("sources") or {})
        superseded = document.get("checkpoint")
        if superseded and superseded in sources:
            sources.pop(superseded)
        sources[selected] = checkpoint_identity
        document["sources"] = sources
        document["checkpoint"] = selected
        document["checkpoint_role"] = "selected"
        return document

    def write_request(self, track: str, document: dict[str, Any]) -> Path:
        """Land the produced request where the GPU lane reads it.

        The bytes are the shared ablation writer's canonical form (sorted keys,
        two-space indent, trailing newline), so seeding the request through this
        class and through ``model_tracks.ablation.write`` is byte-identical.
        """
        path = self.request(track)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(document, sort_keys=True, ensure_ascii=False, indent=2,
                       allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return path

    # ---------------------------------------------------- landed-result readers
    @staticmethod
    def _read_json(path: Path) -> Any:
        """Read one landed JSON document, recording the FULL traceback on failure."""
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _LOG.error(traceback.format_exc())
            raise

    def load_request(self, track: str) -> Any:
        """The landed post-training request for one track."""
        return self._read_json(self.request(track))

    def load_report(self, track: str) -> Any:
        """The landed frozen ablation report for one track."""
        return self._read_json(self.report(track))

    def baseline(self, track: str) -> Any:
        """The landed baseline calibration document for one track."""
        return self._read_json(self.baseline_threshold(track))

    def frozen_threshold(self, track: str) -> float:
        """The frozen threshold the sealed baseline calibration pins.

        Read from the landed binding (never refit here): a missing or
        non-finite value fails loud with the binding path named.
        """
        document = self.baseline(track)
        if not isinstance(document, dict) or "threshold" not in document:
            raise ValueError(
                f"baseline calibration {self.baseline_threshold(track)} carries no threshold"
            )
        value = float(document["threshold"])
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(
                f"baseline calibration {self.baseline_threshold(track)} has a non-finite threshold"
            )
        return value

    def landed(self, track: str) -> set[str]:
        """The declared track results PRESENT on disk (a census, never a gate)."""
        self._require_track(track)
        return {
            name
            for name, role in self.roles.items()
            if role.scope == _TRACK and self.path(name, track).is_file()
        }

    # ---------------------------------------------------------------- SSOT refs
    def thresholds(self) -> tuple[float, ...]:
        """The operating-point ladder every report lane sweeps (ONE SSOT).

        Delegates to the accessor ``results.yaml`` names (``core.common``), so
        the ladder stays declared once in ``config/training.yaml``.
        """
        from core import common

        accessor = getattr(common, self.threshold_ladder.key, None)
        if not callable(accessor):
            raise ValueError(
                f"results threshold ladder names unknown common accessor {self.threshold_ladder.key!r}"
            )
        return tuple(float(value) for value in accessor())

    # ---------------------------------------------------------------- identity
    def identity(self) -> dict[str, Any]:
        """The run's STRUCTURAL identity: run tag + names + byte sizes + counts.

        Derived from structure alone: a changed byte size shows up here, but no
        id is ever computed from content. A member that is absent is recorded as
        an absence, never raised -- this is a census, not a completeness verdict.
        """
        members = []
        for name, path in sorted(self.paths().items()):
            present = path.is_file()
            members.append(
                {"name": name, "present": present,
                 "size": path.stat().st_size if present else 0}
            )
        return {
            "name": self.name,
            "run_tag": self.run_tag,
            "root": self.root.name,
            "members": members,
            "counts": {
                "tracks": len(self.tracks),
                "results": len(members),
                "present": sum(1 for member in members if member["present"]),
            },
        }


@lru_cache(maxsize=None)
def results(run_tag: str) -> Results:
    """The process-wide Results for one run tag, resolved from the SSOT once."""
    return Results.from_config(run_tag)
