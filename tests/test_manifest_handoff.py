"""The handoff boundary enumerates published stage manifests.

``training.handoff`` records WHICH stage manifests a run published; it never
re-verifies the recorded output bytes (owner directive 2026-10-09: data is never
checked anywhere). These tests pin that enumeration contract:

  * a published stage is listed;
  * a tampered output is NOT re-checked (a record, never a refusal);
  * a stage that has not published yet is skipped, not failed.
"""
from core.portable_archive import ByteCount
from pathlib import Path

from core.manifest import atomic_write_json
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
            path=str(output), size=ByteCount(output_bytes).total,
            rows=None, cols=None, expected=True)],
        row_accounting={},
        environment={},
        expected_outputs=[output.name],
    )
    atomic_write_json(manifest.model_dump(mode="json"), manifest_dir / f"{stage}.json")
    return manifest_dir, output


def test_handoff_records_published_manifest_stages(tmp_path):
    from training.handoff import _manifest_directory, _published_manifest_stages
    manifest_dir = _manifest_directory(tmp_path)
    _write_manifest(manifest_dir, stage="dedupe")
    summary = _published_manifest_stages(manifest_dir)
    assert summary == {"published_stages": ["dedupe"]}


def test_handoff_does_not_recheck_tampered_output(tmp_path):
    """Data is never checked: the boundary enumerates published stages, no byte re-check."""
    from training.handoff import _manifest_directory, _published_manifest_stages
    manifest_dir = _manifest_directory(tmp_path)
    _, output = _write_manifest(manifest_dir, stage="dedupe")
    output.write_bytes(b"tampered")
    assert _published_manifest_stages(manifest_dir) == {"published_stages": ["dedupe"]}


def test_handoff_skips_absent_stages(tmp_path):
    from training.handoff import _manifest_directory, _published_manifest_stages
    manifest_dir = _manifest_directory(tmp_path)
    # nothing published yet: a later-lane absence is not a failure
    assert _published_manifest_stages(manifest_dir) == {"published_stages": []}
