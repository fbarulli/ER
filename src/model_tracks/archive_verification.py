"""Post-download verification of a sealed model-tracks suite archive.

The sealing-time guard (``model_tracks.resume.validate_completed_suite_archive``)
runs on the machine that WROTE the archive. Once the archive has crossed the
wire to a local checkout, the bytes are the only evidence left, so this module
re-verifies them and records the outcome as a ``.verification.json`` sidecar:

* the ``.sha256`` sidecar, when present (transfer integrity), and
* the full sealing-time contract (byte integrity of every member, run tag,
  per-track completion markers, exactly one calibrated report manifest per
  track, saved-ablation binding) plus a per-track summary of what was
  actually calibrated (threshold, checkpoint identity, test reporting).

``verification_result`` never raises on contract failure: it returns a
JSON-safe result with ``status`` of ``verified``, ``failed`` or
``unreadable`` so the dashboard panel and the CLI can both show what broke
without crashing.  ``write_verification`` persists that result atomically.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from core.archive_reader import archive_sidecar
from model_tracks.config import SuiteConfig
from model_tracks.post_training_ablation import ABLATION_TRACKS, SavedAblationReport
from model_tracks.resume import (
    TRACKS, TrainingInputBinding, _events_skip_ablation, validate_completed_suite_archive,
)

VERIFICATION_SCHEMA = 'er-suite-verification-v1'


def _spec():
    """The bundle contract from config (single source for member names)."""
    from core.bundle import bundle_spec
    return bundle_spec()


def report_member(track: str) -> str:
    """The calibrated report manifest a completed track must carry exactly one of.

    The derivation lives with the manifest contract it names
    (``graph_tracks.report_manifest``); the import stays lazy so this module
    does not pull the graph lane in just to spell a filename.
    """
    from graph_tracks.report_manifest import report_member as manifest_member
    return manifest_member(track)


def archive_run_tag(archive: Path) -> str:
    for ending in ('.tar.zst', '.zip'):
        if Path(archive).name.endswith(ending):
            return Path(archive).name[:-len(ending)]
    return Path(archive).stem


def _not_interrupted(relative: str) -> bool:
    parts = Path(relative).parts
    return not any(part.startswith('interrupted-') or '.interrupted-' in part for part in parts)


def verification_result(archive: Path, run_tag: str | None = None,
                        settings: SuiteConfig | None = None) -> dict:
    """Verify a sealed suite archive; returns the JSON-safe outcome.

    ``settings`` pins the suite configuration the archive must have run
    under (the sealing-time binding check).  The Colab lane embeds the
    remote machine's configuration (``/content/...`` setup dirs), so local
    callers pass ``None`` there; the local lane passes the suite config so
    a re-packaged archive cannot masquerade as a different one.
    """
    archive = Path(archive)
    run_tag = run_tag or archive_run_tag(archive)
    spec = _spec()
    result: dict = {
        'schema': VERIFICATION_SCHEMA,
        'verified_at': datetime.now(timezone.utc).isoformat(),
        'run_tag': run_tag,
        'archive': archive.name,
        'zip_sha256': {},
        'tracks': {},
    }
    sidecar = archive_sidecar(archive, spec.sha256_sidecar_suffix)
    if sidecar.is_file() and not sidecar.is_symlink():
        expected = sidecar.read_text(encoding='utf-8').strip().splitlines()[0]
        result['zip_sha256'] = {'sidecar': sidecar.name, 'expected': expected,
                                'actual': None, 'match': None}
    else:
        result['zip_sha256'] = {'sidecar': None, 'match': None,
                                'note': 'no .sha256 sidecar to check against'}
    try:
        from core.bundle import Bundle, BundleRole
        # Exactly one boundary check; its streaming pass also yields the
        # whole-archive digest the sidecar is compared against.
        handle = Bundle.load(archive, BundleRole.result)
        if result['zip_sha256'].get('expected') is not None:
            result['zip_sha256']['actual'] = handle.digest
            result['zip_sha256']['match'] = (
                result['zip_sha256']['expected'] == handle.digest)
        validate_completed_suite_archive(archive, run_tag, settings=settings, bundle=handle)
        with handle.reader() as source:
            binding = TrainingInputBinding.model_validate_json(
                source.read(spec.suite_manifest_file))
            ablation_enabled = binding.settings.post_training_ablation
            # A GPU suite that shipped no ablation templates records a deliberate
            # skip; the cascade never ships an ablation at all.
            ablation_skipped = any(
                _events_skip_ablation(source.read(name).decode(errors='replace'))
                for name in source.namelist()
                if name.endswith((spec.suite_events_file, spec.worker_events_file)))
            ablation_required = ablation_enabled and not ablation_skipped
            for track in TRACKS:
                suffix = report_member(track)
                reports = [relative for relative in source.namelist()
                           if relative.startswith(f'{track}/')
                           and relative.split('/')[-1] == suffix
                           and _not_interrupted(relative)]
                if len(reports) != 1:
                    raise ValueError(
                        'completed archive lacks one calibrated track report: ' + track)
                report = json.loads(source.read(reports[0]))
                entry: dict = {
                    'report': reports[0],
                    'threshold': report['threshold'],
                    'checkpoint_sha256': report['checkpoint_sha256'],
                    'test_reported': report['test_reported'],
                }
                if ablation_required and track in ABLATION_TRACKS:
                    saved = SavedAblationReport.model_validate_json(
                        source.read(f'{track}/ablation/report.json'))
                    entry['ablation'] = {
                        'threshold': saved.threshold,
                        'threshold_binding': saved.threshold_binding.model_dump(),
                    }
                result['tracks'][track] = entry
    except FileNotFoundError:
        result['status'] = 'unreadable'
        result['error'] = f'archive not found: {archive.name}'
        return result
    except Exception as exc:  # the panel must show the failure, not crash
        result['status'] = 'failed'
        result['error'] = f'{type(exc).__name__}: {exc}'
        return result
    result['status'] = 'verified'
    return result


def write_verification(archive: Path, result: dict) -> Path:
    """Persist the outcome atomically as the ``.verification.json`` sidecar."""
    path = archive_sidecar(archive, '.verification.json')
    candidate = path.with_name(path.name + '.partial')
    candidate.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    candidate.replace(path)
    return path


def load_verification(archive: Path) -> dict | None:
    path = archive_sidecar(archive, '.verification.json')
    if not path.is_file() or path.is_symlink():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


__all__ = ['VERIFICATION_SCHEMA', 'archive_run_tag', 'load_verification',
           'report_member', 'verification_result', 'write_verification']
