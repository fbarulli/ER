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
    if slices is not None:
        manifest["slices"] = slices
    if extra:
        overlap = set(manifest).intersection(extra)
        if overlap:
            raise ValueError(f"report extras overwrite contract fields: {sorted(overlap)}")
        manifest.update(extra)
    missing = [key for key in REQUIRED_KEYS if key not in manifest]
    if missing:
        raise ValueError(f"report manifest missing required keys: {missing}")
    return TrackReportManifest.model_validate(manifest).model_dump(by_alias=True)


def write(path: Path, manifest: dict) -> Path:
    """Write a manifest, refusing to emit one with an incomplete contract."""
    missing = [key for key in REQUIRED_KEYS if key not in manifest]
    if missing:
        raise ValueError(
            f"refusing to write {path.name}: manifest missing {missing}"
        )
    validated = TrackReportManifest.model_validate(manifest)
    candidate = path.with_suffix(path.suffix + ".partial")
    candidate.write_text(validated.model_dump_json(indent=2, by_alias=True) + "\n")
    candidate.replace(path)
    return path


__all__ = ["MANIFEST_SCHEMA", "REQUIRED_KEYS", "build", "write"]