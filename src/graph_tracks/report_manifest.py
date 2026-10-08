"""Single builder for the per-track report manifest.

Every model-track postprocess writes one of these. Before this module the
graph-track lane assembled the key set inline in ``graph_tracks/report.py``
while ``model_tracks/text_report.py`` hand-wrote a completely different,
much smaller ``*__completion_manifest.json``. The text manifest omitted
``test_used_for_selection``, ``metrics_scope``, ``trained_endpoints_scored``,
``unlabeled_pairs_are_negatives``, ``identity_conflict_policy_applied``,
``retrieval_protocol``, ``test_reported``, ``model_selection``,
``graph_context``, ``checkpoint_sha256``, ``threshold`` and
``threshold_source`` -- so the dashboard and any downstream reader had to
special-case the text track, and the text track silently shipped with no
statement of whether test labels were used for selection.

Both lanes now call :func:`build`, so the honesty contract cannot drift again.
:data:`REQUIRED_KEYS` is asserted by the tests that cover each lane.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

MANIFEST_SCHEMA = "er-track-report-manifest-v1"

#: Every manifest must carry all of these. Enforced by
#: tests/test_track_report_manifest.py for each track.
REQUIRED_KEYS = (
    "schema",
    "track",
    "checkpoint",
    "checkpoint_sha256",
    "listings_sha256",
    "pairs_sha256",
    "threshold",
    "threshold_source",
    "test_used_for_selection",
    "test_reported",
    "model_selection",
    "graph_context",
    "trained_endpoints_scored",
    "retrieval_protocol",
    "retrieval_ks",
    "unlabeled_pairs_are_negatives",
    "identity_conflict_policy_applied",
    "metrics_scope",
    "performance",
    "confidence_intervals",
)

#: The two per-lane traceability tables, present on every manifest.
TRACEABILITY_KEYS = ("slices", "attributes")


def _default_traceability(track: str, slices: list, attributes: list) -> dict:
    """Name the provenance of each traceability table, or why it is empty.

    ``slices`` is the generalization-slice table and ``attributes`` the
    attribute-separation table. A trained lane without a live population still
    writes its attribute rows next to the report as
    ``<track>__attribute_separation_summary.csv``; the cascade is a combinator
    over trained artifacts and has neither population, so it records an
    explicit not-applicable instead of an indistinguishable empty list.
    """
    slice_note = ("computed from the scored dev/test pair population" if slices
                  else f"{track} scored no generalization-slice population")
    if attributes:
        attribute_note = ("computed from the scored pair population and the "
                          "attribute registry")
    elif track == "cascade":
        attribute_note = ("not applicable: the cascade carries no listing-attribute "
                          "scoring population")
    else:
        attribute_note = (f"not embedded here; written to "
                          f"{track}__attribute_separation_summary.csv")
    return {"slices": slice_note, "attributes": attribute_note}


class TrackReportManifest(BaseModel):
    """Schema for every lane's report, including fixed holdout guarantees."""
    model_config = ConfigDict(extra='allow', allow_inf_nan=False)
    schema_id: Literal['er-track-report-manifest-v1'] = Field(alias='schema')
    track: Literal['text', 'gnn_only', 'cascade']
    checkpoint: str | None
    checkpoint_sha256: str = Field(min_length=1)
    listings_sha256: str = Field(min_length=1)
    pairs_sha256: str = Field(min_length=1)
    threshold: float
    threshold_source: Literal['dev_youden']
    test_used_for_selection: Literal[False]
    test_reported: bool
    model_selection: Literal['dev_pr_auc']
    graph_context: str = Field(min_length=1)
    trained_endpoints_scored: Literal[False]
    retrieval_protocol: str = Field(min_length=1)
    retrieval_ks: list[StrictInt] = Field(min_length=1)
    unlabeled_pairs_are_negatives: Literal[False]
    identity_conflict_policy_applied: Literal[False]
    metrics_scope: Literal['model-only']
    performance: dict
    confidence_intervals: dict

    @model_validator(mode='after')
    def check_ladder(self):
        if any(k < 1 for k in self.retrieval_ks) or len(set(self.retrieval_ks)) != len(self.retrieval_ks):
            raise ValueError('retrieval_ks must contain unique positive integers')
        return self


def build(
    *,
    track: str,
    checkpoint: str | Path | None,
    checkpoint_sha256: str,
    listings_sha256: str,
    pairs_sha256: str,
    threshold: float,
    threshold_source: str,
    test_reported: bool,
    model_selection: str,
    retrieval_ks,
    performance: dict | None = None,
    confidence_intervals: dict | None = None,
    # Per-lane extras that do not belong in the shared honesty contract.
    vectors_metadata: dict | None = None,
    summary: list | None = None,
    retrieval: list | None = None,
    slices: list | None = None,
    attributes: list | None = None,
    roles: dict | None = None,
    traceability: dict | None = None,
    report_test: bool | None = None,
    extra: dict | None = None,
) -> dict:
    """Assemble the shared manifest contract plus any lane extras.

    The honesty fields are set from constants here rather than passed in by the
    caller, because every lane agrees on their values: the tracks all fit the
    threshold on dev, score model-only (no identity-conflict policy, no trained
    endpoints in the catalog), and treat unlabeled candidates as unknown rather
    than negative.  ``identity_conflict_policy_applied`` and
    ``trained_endpoints_scored`` stay explicit ``False`` so a reader can tell
    the difference between "not applied" and "unknown".

    The two traceability keys ``slices`` and ``attributes`` are emitted on
    every manifest, with ``traceability`` naming where each table came from or
    why it is empty. The cascade combinator has no listing catalog to slice or
    attribute-score, so its tables are an explicit not-applicable rather than a
    silent omission, and the key set never drifts between the three tracks.
    """
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "track": track,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "checkpoint_sha256": checkpoint_sha256,
        "listings_sha256": listings_sha256,
        "pairs_sha256": pairs_sha256,
        "threshold": float(threshold),
        "threshold_source": threshold_source,
        "test_used_for_selection": False,
        "test_reported": bool(test_reported),
        "model_selection": model_selection,
        "graph_context": "none (text encoder)" if track == "text" else "training-listings-only",
        "trained_endpoints_scored": False,
        "retrieval_protocol": (
            "within-split catalog, self excluded, direct known positives only"
        ),
        "retrieval_ks": list(retrieval_ks),
        "unlabeled_pairs_are_negatives": False,
        "identity_conflict_policy_applied": False,
        "metrics_scope": "model-only",
        "performance": performance or {},
        "confidence_intervals": confidence_intervals or {},
    }
    if report_test is not None:
        manifest["report_test"] = bool(report_test)
    if vectors_metadata is not None:
        manifest["vectors_metadata"] = vectors_metadata
    if summary is not None:
        manifest["summary"] = summary
    if retrieval is not None:
        manifest["retrieval"] = retrieval
    # Emitted unconditionally so the traceability shape is identical across the
    # trained lanes and the cascade; an absent table is an empty list with a
    # machine-readable reason, never a missing key.
    manifest["slices"] = list(slices) if slices is not None else []
    manifest["attributes"] = list(attributes) if attributes is not None else []
    manifest["traceability"] = (
        dict(traceability) if traceability is not None
        else _default_traceability(track, manifest["slices"], manifest["attributes"])
    )
    if roles is not None:
        manifest["roles"] = roles
    if extra:
        overlap = set(manifest).intersection(extra)
        if overlap:
            raise ValueError(f"report extras overwrite contract fields: {sorted(overlap)}")
        manifest.update(extra)
    missing = [key for key in REQUIRED_KEYS if key not in manifest]
    if missing:
        raise ValueError(f"report manifest missing required keys: {missing}")
    return TrackReportManifest.model_validate(manifest).model_dump(by_alias=True)


def _fold_cascade_extras(path: Path, manifest: dict) -> dict:
    """Attach the cascade report's traceability extras to its per-track manifest.

    The cascade is a combinator: its ``report_cascade`` output carries the
    ranker (candidate-recall) and decider (PR-AUC / precision@recall / ECE)
    metrics plus the cascade's explicit traceability statement, but the suite
    completion contract only names the shared manifest. The manifest writer is
    the one place every lane's contract is assembled, so the cascade's roles,
    ``slices``/``attributes`` tables and their ``traceability`` note are folded
    in here rather than hand-rolled by a caller. A caller-supplied value wins;
    this is a no-op for every other track.
    """
    if manifest.get("track") != "cascade":
        return manifest
    from graph_tracks.artifacts import name
    report_path = path.parent / name("cascade", "cascade_report.json")
    if not report_path.is_file():
        return manifest
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return manifest
    folded = dict(manifest)
    if not folded.get("roles") and isinstance(payload.get("roles"), dict):
        folded["roles"] = payload["roles"]
    for key in TRACEABILITY_KEYS:
        if not folded.get(key) and isinstance(payload.get(key), list):
            folded[key] = payload[key]
    if isinstance(payload.get("traceability"), dict):
        folded["traceability"] = payload["traceability"]
    return folded


def write(path: Path, manifest: dict) -> Path:
    """Write a manifest, refusing to emit one with an incomplete contract."""
    missing = [key for key in REQUIRED_KEYS if key not in manifest]
    if missing:
        raise ValueError(
            f"refusing to write {path.name}: manifest missing {missing}"
        )
    manifest = _fold_cascade_extras(Path(path), manifest)
    validated = TrackReportManifest.model_validate(manifest)
    candidate = path.with_suffix(path.suffix + ".partial")
    candidate.write_text(validated.model_dump_json(indent=2, by_alias=True) + "\n")
    candidate.replace(path)
    return path


__all__ = ["MANIFEST_SCHEMA", "REQUIRED_KEYS", "TRACEABILITY_KEYS", "build", "write"]