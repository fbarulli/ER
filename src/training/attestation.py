"""Consumer-side attestation for check-free training.

The handoff boundary (``training.handoff``) validates every training input
once per preparation run and writes ``run_dir/handoff.json``.  This module
turns that report into a portable attestation the trainer can verify with a
single structural size read instead of re-running the full validation stack:
after the boundary, training is just training.
"""
from __future__ import annotations

from core.portable_archive import ByteCount
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from core.timing import emit_timing

SCHEMA: Final = 'er-training-attestation-v1'


class TrainingAttestation(BaseModel):
    """Portable proof that one handoff boundary validated this run's inputs."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    attestation_schema: Literal["er-training-attestation-v1"] = Field(alias="schema")
    status: Literal["pass"]
    run_dir: str
    finished_at: str
    bundle_path: str
    bundle_size: int = Field(ge=0)
    provenance_size: int | None = Field(default=None, ge=0)
    provenance_verified: str | None = None
    plan_identity: dict[str, Any] | None = None
    checks: dict[str, Any]
    attested_at: str


def record_provenance_verification(attestation: TrainingAttestation) -> str:
    """Telemetry gate for the proactive provenance evidence (D1).

    Never fails the run; the measured status is carried on the attestation
    object and in one ``[timing]`` line so runs can conclude the item from
    data.
    """
    from core.common import training_cfg

    status = "missing_manifest"
    try:
        sibling = Path(attestation.run_dir) / training_cfg().preparation.manifest_file
        if sibling.is_file():
            payload = json.loads(sibling.read_text(encoding="utf-8"))
            provenance = payload.get("provenance") if isinstance(payload, dict) else None
            if not isinstance(provenance, dict):
                status = "unusable_manifest_provenance"
            elif attestation.provenance_size is None:
                status = "missing_size"
            else:
                canonical = json.dumps(
                    provenance, sort_keys=True, separators=(",", ":")
                )
                size = ByteCount(canonical.encode()).total
                status = "verified" if size == attestation.provenance_size else "mismatch"
    except Exception:
        status = "unreadable_manifest"
    attestation.provenance_verified = status
    emit_timing(
        f"[timing] training.attestation provenance_verified={status} "
        f"run_dir={attestation.run_dir}"
    )
    return status


def _read_json(path: Path, *, what: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{what} missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise ValueError(f"{what} is not valid JSON: {path}: {error}") from error


def _validated_attestation(payload: dict[str, Any], path: Path) -> TrainingAttestation:
    try:
        return TrainingAttestation.model_validate(payload)
    except ValidationError as error:
        raise ValueError(f"training attestation rejected: {path}: {error}") from error


def load_attestation(path: Path) -> TrainingAttestation:
    """Load and schema-validate a persisted ``TrainingAttestation`` JSON."""
    path = Path(path)
    started = time.monotonic()
    payload = _read_json(path, what="training attestation")
    attestation = _validated_attestation(payload, path)
    emit_timing(f"[timing] training.attestation loaded path={path.name} "
                f"bundle_size={attestation.bundle_size} "
                f"seconds={time.monotonic() - started:.3f}")
    return attestation


def verify_attestation(attestation: TrainingAttestation, *, bundle_path: Path) -> None:
    """Check that an attestation passed and names a bundle that is present.

    Nothing about the bundle bytes is re-proven: a bundle is immutable, so its
    identity IS the bundle (owner directive: data is never checked). The
    boundary's structural contracts (loss / sampling) are checked by
    :func:`verify_plan_identity`.
    """
    if attestation.status != "pass":
        raise ValueError(f"training attestation does not pass: status={attestation.status!r}")
    path = Path(bundle_path)
    if not path.is_file():
        raise ValueError(f"attested prepared bundle missing: {path}")


def verify_plan_identity(attestation: TrainingAttestation, *, loss: str,
                         train_frac: float, sample: bool) -> None:
    """Cheap contract check: the invocation matches the attested frozen plan.

    The boundary attested the plan for ONE objective; a trainer invoking a
    different loss/train_frac/sample would silently train outside the
    attested batch contract.  This is a dict compare, not a re-validation.
    An attestation without a plan identity proves nothing about the batch
    contract, so it fails closed here instead of passing silently.
    """
    identity = attestation.plan_identity
    if not identity:
        raise ValueError(
            "training attestation carries no plan identity; the invocation "
            "cannot be checked against the attested frozen plan "
            f"(run_dir={attestation.run_dir!r})"
        )
    requested = {"loss": loss, "train_frac": float(train_frac), "sample": bool(sample)}
    mismatches = {
        name: {"attested": identity.get(name), "invoked": value}
        for name, value in requested.items()
        if identity.get(name) != value
    }
    if mismatches:
        raise ValueError(
            f"training invocation differs from the attested frozen plan: {mismatches}"
        )


def write_attestation(attestation: TrainingAttestation, path: Path) -> None:
    """Persist an attestation with the stable JSON key set (alias round-trip)."""
    from core.manifest import atomic_write_json
    atomic_write_json(attestation.model_dump(mode="json", by_alias=True), Path(path))


def _handoff_report(payload: dict[str, Any], handoff_path: Path) -> Any:
    from training.handoff import HandoffReport

    try:
        return HandoffReport.model_validate(payload)
    except ValidationError as error:
        raise ValueError(f"handoff report rejected: {handoff_path}: {error}") from error


def _header_size(header: dict[str, Any], *, handoff_path: Path) -> int:
    size = header.get("size")
    if not isinstance(size, int) or size < 0:
        raise ValueError(f"handoff report bundle_header lacks a bundle size: {handoff_path}")
    return size


def _provenance_size(run_dir: Path) -> int | None:
    """The canonical provenance block's byte length, if the sibling exists."""
    from core.common import training_cfg

    path = Path(run_dir) / training_cfg().preparation.manifest_file
    if not path.is_file():
        return None
    payload = _read_json(path, what="preparation manifest")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        return None
    canonical = json.dumps(provenance, sort_keys=True, separators=(",", ":"))
    return ByteCount(canonical.encode()).total


def _handoff_attestation(payload: dict[str, Any], handoff_path: Path,
                         *, bundle_path: Path) -> TrainingAttestation:
    finished_at = payload.pop("finished_at", None)
    report = _handoff_report(payload, handoff_path)
    if report.status != "pass":
        raise ValueError(f"handoff report does not pass: status={report.status!r}: {handoff_path}")
    if not isinstance(finished_at, str) or not finished_at:
        raise ValueError(f"handoff report lacks finished_at: {handoff_path}")
    attestation_block = report.loss_batch_correctness
    return TrainingAttestation(
        attestation_schema=SCHEMA,
        status="pass",
        run_dir=str(handoff_path.parent),
        finished_at=finished_at,
        bundle_path=str(Path(bundle_path)),
        bundle_size=_header_size(report.bundle_header, handoff_path=handoff_path),
        provenance_size=_provenance_size(handoff_path.parent),
        plan_identity=(dict(attestation_block.plan_identity) if attestation_block else None),
        checks=report.checks,
        attested_at=datetime.now(timezone.utc).isoformat(),
    )


def attestation_from_handoff(handoff_path: Path, *, bundle_path: Path) -> TrainingAttestation:
    """Build an attestation from a preparation run's ``handoff.json`` report.

    The provenance digest is taken from the run manifest sibling
    (``run_dir/manifest.json``) when it exists; without it the attestation
    carries ``provenance_size=None``.
    """
    handoff_path = Path(handoff_path)
    started = time.monotonic()
    payload = _read_json(handoff_path, what="handoff report")
    attestation = _handoff_attestation(payload, handoff_path, bundle_path=bundle_path)
    record_provenance_verification(attestation)
    emit_timing(f"[timing] training.attestation built_from_handoff path={handoff_path.name} "
                f"bundle_size={attestation.bundle_size} "
                f"seconds={time.monotonic() - started:.3f}")
    return attestation


def read_attestation(path: Path, *, bundle_path: Path) -> TrainingAttestation:
    """Load an attestation, accepting either a persisted one or handoff.json."""
    path = Path(path)
    started = time.monotonic()
    payload = _read_json(path, what="training attestation")
    if payload.get("schema") == SCHEMA:
        attestation = _validated_attestation(payload, path)
    elif "bundle_header" in payload:
        attestation = _handoff_attestation(payload, path, bundle_path=bundle_path)
    else:
        raise ValueError(
            f"not a training attestation or handoff report: {path} "
            f"(schema={payload.get('schema')!r})"
        )
    record_provenance_verification(attestation)
    emit_timing(f"[timing] training.attestation read path={path.name} "
                f"kind={'attestation' if payload.get('schema') == SCHEMA else 'handoff'} "
                f"bundle_size={attestation.bundle_size} "
                f"seconds={time.monotonic() - started:.3f}")
    return attestation
