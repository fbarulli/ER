"""Guard: no reference to the retired experiment-tracking service survives.

The live layers are W&B plus local DVC, and the datasets stay naked: there is no
dataset registry, no dataset versioning service and no retired third-party
tracker. A leftover reference -- an import, a tracking-URI environment variable,
or the tracker's run directory spelled into a ship/exclusion rule -- is a stale
extra layer that a copy-paste silently reintroduces, so this guard scans the
surfaces this project owns and fails on any occurrence.

The forbidden token is assembled here from its two halves. This file is the only
one allowed to spell it, which is what lets the guard scan ``tests/`` too
instead of punching a hole for itself.

Scope: every surface this project owns as code -- ``src``, ``config``,
``dashboard``, ``docs``, ``notebooks``, ``requirements``, ``scripts``,
``submission`` and ``tests``, plus the root-level code/config files
(``*.py``, ``*.yaml``, ``*.toml``, ``*.txt``, ``*.cfg``, ``*.ini``, .gitignore).
Generated run state and evidence records (``artifacts/``, ``jev/``, ``logs/``,
``results/``, ``training_results/``, ``training_profile/``, ``wandb/``,
``.dvc/``, ``data/``, ``colab_cli_state/``) are data, not code: a captured log
or a frozen before-image is a RECORD of what ran, so it is deliberately out of
scope and must never be rewritten to look tidier than it was.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCANNED_DIRECTORIES = (
    "src", "config", "dashboard", "docs", "notebooks", "requirements",
    "scripts", "submission", "tests",
)
#: Root-level code and config files (the backend/CLI entry scripts, augment.yaml,
#: pyproject.toml, requirements.txt, .gitignore, ...).
SCANNED_ROOT_PATTERNS = ("*.py", "*.yaml", "*.yml", "*.toml", "*.txt", "*.cfg", "*.ini")
TEXT_SUFFIXES = frozenset({
    ".py", ".yaml", ".yml", ".md", ".rst", ".sh", ".toml", ".txt", ".json",
    ".cfg", ".ini",
})

# Assembled at runtime so this guard's own source is a clean scan target.
FORBIDDEN = re.compile("|".join(("ml" + "flow", "ml" + "runs")), re.IGNORECASE)


def _scanned_paths() -> list[Path]:
    """Every owned code/config file the guard inspects (this guard's own source too)."""
    candidates = [REPO_ROOT / name for name in SCANNED_DIRECTORIES]
    for pattern in SCANNED_ROOT_PATTERNS:
        candidates.extend(REPO_ROOT.glob(pattern))
    for directory in SCANNED_DIRECTORIES:
        candidates.extend((REPO_ROOT / directory).rglob("*"))
    return sorted({
        path
        for path in candidates
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix in TEXT_SUFFIXES
    })


def test_no_retired_tracker_reference_survives() -> None:
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{number}: {line.strip()}"
        for path in _scanned_paths()
        for number, line in enumerate(
            path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1
        )
        if FORBIDDEN.search(line)
    ]
    assert not offenders, "stale tracking-layer reference(s):\n" + "\n".join(offenders)


def test_publish_ships_artifacts_the_worker_does_not_own(tmp_path: Path) -> None:
    """The Hub publish predicate owns ONE exclusion set: the worker's own files.

    No directory name is a ship rule any more, so a re-added tracker output
    directory fails both here and in the text scan above.
    """
    from training.artifact_store import UPLOAD_EXCLUDED_FILES, _is_uploadable

    assert set(UPLOAD_EXCLUDED_FILES) == {
        "canonical_records.csv", "gate_results.csv", "training.status"
    }

    owned = tmp_path / "runs" / "0"
    owned.mkdir(parents=True)
    for name in UPLOAD_EXCLUDED_FILES:
        (owned / name).write_text("owned", encoding="utf-8")
        assert not _is_uploadable(owned / name)

    nested = tmp_path / "runs" / ("ml" + "runs") / "0"
    nested.mkdir(parents=True)
    artifact = nested / "vectors.npz"
    artifact.write_text("data", encoding="utf-8")
    assert _is_uploadable(artifact)
    assert not _is_uploadable(nested)
