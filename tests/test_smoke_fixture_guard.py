"""The smoke fixture guard keeps tracked suite fixtures byte-identical.

Regression class of 9b342be: the packaging publish swap rewrites tracked
fixtures IN PLACE under the setup dir (ablation_settings.yaml and
embedding_inputs.json were serialized with deleted disposable-slot paths
because the wrapper's hand-listed snapshot missed them). The guard derives its
snapshot scope from git itself — every tracked file under the setup dir — so
no fixture a writer rewrites can escape restore: the dirty pass below touches
ALL tracked fixtures (driven from ``git ls-files``, not a spelled list) and
the assertion is the checkout-level contract: git-clean, byte-identical.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
GUARD = REPO_ROOT / "scripts" / "smoke_fixture_guard.sh"
SETUP_DIR = "data/prepared/smoke_suite"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(['git', '-C', str(repo), *args], capture_output=True,
                          text=True, check=True).stdout


def _seed_repo(tmp_path: Path) -> Path:
    """A scratch checkout with the tracked suite fixtures committed."""
    repo = tmp_path / 'repo'
    fixtures = {
        f'{SETUP_DIR}/ablation_settings.yaml': 'output_dir: canonical\n',
        f'{SETUP_DIR}/embedding_inputs.json': '{"ids": []}\n',
        f'{SETUP_DIR}/suite.yaml': 'setup_dir: .\n',
        f'{SETUP_DIR}/ablation_templates/text/request.json': '{}\n',
        f'{SETUP_DIR}/text_export_request.json': '{}\n',
    }
    for rel, content in fixtures.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    identity = {'GIT_AUTHOR_NAME': 'guard', 'GIT_AUTHOR_EMAIL': 'guard@test',
                'GIT_COMMITTER_NAME': 'guard', 'GIT_COMMITTER_EMAIL': 'guard@test'}
    for args in (('init', '-q'), ('add', '-A'), ('commit', '-qm', 'pin fixtures')):
        subprocess.run(['git', '-C', str(repo), *args], capture_output=True,
                       text=True, check=True, env={**os.environ, **identity})
    return repo


def test_guard_restores_every_tracked_fixture_the_run_rewrites(tmp_path):
    repo = _seed_repo(tmp_path)
    backup = tmp_path / 'snapshot'
    # One bash session runs the wrapper's seam: snapshot, the run rewrites
    # every tracked fixture in place (the publish swap's effect) and drops an
    # untracked staging dir beside the templates, then the EXIT trap restores.
    script = f'''
set -euo pipefail
. "{GUARD}"
smoke_guard_snapshot "{SETUP_DIR}" "{backup}"
for f in $(git ls-files -- "{SETUP_DIR}"); do
  printf '.worktrees/package-smoke_200-DEAD/rewritten\\n' >> "$f"
done
mkdir -p "{SETUP_DIR}/ablation_templates/text_selected"
printf 'staged\\n' > "{SETUP_DIR}/ablation_templates/text_selected/request.json"
trap smoke_guard_restore EXIT
'''
    subprocess.run(['bash', '-c', script], cwd=repo, check=True)
    assert _git(repo, 'status', '--porcelain') == '', \
        'a tracked fixture the guard missed stays dirty (or a run-leftover ' \
        'untracked dir stays): the derived snapshot must cover the whole dir'
