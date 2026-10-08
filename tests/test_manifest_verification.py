"""The silent-drop guardrail's production re-verification (verify_manifest).

``core.manifest.verify_manifest`` re-checks a published stage manifest against
the files on disk; the handoff boundary (``training.handoff``) now runs it over
every manifest the preparation already published. These tests pin the two
halves of that contract:

  * a valid published manifest passes;
  * a tampered one (stale output bytes, or an edited recorded hash) fails loud;
  * the handoff boundary's helper verifies what exists and skips what a later
    lane has not published yet.
"""
import hashlib
import json
from pathlib import Path

import pytest

from core.manifest import atomic_write_json, verify_manifest
from core.schemas import ManifestFile, StageManifest


def _write_manifest(manifest_dir, *, stage="dedupe", output_bytes=b"hello"):
    """Write one complete single-output StageManifest plus its output file."""
    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    output = manifest_dir.parent / f"{stage}_output.csv"
    output.write_bytes(output_bytes)
    manifest = StageManifest(
        schema_version="1",
        stage=stage,
        started="2026-01-01T00:00:00+00:00",
        finished="2026-01-01T00:00:01+00:00",
        status="complete",
        inputs=[],
        outputs=[ManifestFile(
            path=str(output), sha256=hashlib.sha256(output_bytes).hexdigest(),
            rows=None, cols=None, expected=True)],
        row_accounting={},
        environment={},
        expected_outputs=[output.name],
    )
    atomic_write_json(manifest.model_dump(mode="json"), manifest_dir / f"{stage}.json")
    return manifest_dir, output


def test_verify_manifest_passes_on_valid_manifest(tmp_path):
    manifest_dir, _ = _write_manifest(tmp_path / "manifests")
    verify_manifest("dedupe", manifest_dir=manifest_dir)  # must not raise


def test_verify_manifest_rejects_tampered_output(tmp_path):
    manifest_dir, output = _write_manifest(tmp_path / "manifests")
    output.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        verify_manifest("dedupe", manifest_dir=manifest_dir)


def test_verify_manifest_rejects_edited_recorded_hash(tmp_path):
    manifest_dir, _ = _write_manifest(tmp_path / "manifests")
    path = manifest_dir / "dedupe.json"
    document = json.loads(path.read_text())
    document["outputs"][0]["sha256"] = "f" * 64
    path.write_text(json.dumps(document))
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        verify_manifest("dedupe", manifest_dir=manifest_dir)


def test_handoff_manifest_verify_passes_on_valid_and_reports_stages(tmp_path):
    from training.handoff import _manifest_directory, _verify_published_manifests
    manifest_dir = _manifest_directory(tmp_path)
    _write_manifest(manifest_dir, stage="dedupe")
    summary = _verify_published_manifests(manifest_dir)
    assert summary == {"verified_stages": ["dedupe"]}


def test_handoff_manifest_verify_rejects_tampered(tmp_path):
    from training.handoff import _manifest_directory, _verify_published_manifests
    manifest_dir = _manifest_directory(tmp_path)
    _, output = _write_manifest(manifest_dir, stage="dedupe")
    output.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        _verify_published_manifests(manifest_dir)


def test_handoff_manifest_verify_skips_absent_stages(tmp_path):
    from training.handoff import _manifest_directory, _verify_published_manifests
    manifest_dir = _manifest_directory(tmp_path)
    # nothing published yet: a later-lane absence is not a failure
    assert _verify_published_manifests(manifest_dir) == {"verified_stages": []}
