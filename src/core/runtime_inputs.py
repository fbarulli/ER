"""Frozen evidence shared by preparation, GPU workers and their transport."""
from pathlib import Path


def evidence_members() -> tuple[str, ...]:
    from core.common import LAYOUTS

    registry = LAYOUTS['semantic_family_registry']
    if registry.root != 'repo' or registry.fields:
        raise ValueError('semantic family registry must be a static repository input')
    return ('artifacts/evidence/attribute_universe_census.json', registry.template)


def evidence_files(root: Path) -> dict[str, Path]:
    files = {member: root / member for member in evidence_members()}
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError('Required runtime evidence missing: ' + ', '.join(missing))
    return files


def checkout_members(extra=(), *, lane="bundle") -> tuple[str, ...]:
    """One source/config/model/reference selection for both remote platforms."""
    from core.common import load_config, training_cfg
    # The checkout-path shape contract (absolute / traversal / pattern charset)
    # has ONE home: ``cli.colab_runtime.is_checkout_relative_path``. This staging
    # surface asks it instead of re-spelling the rule (the old inline copy here
    # missed the ``{}`` charset the shared predicate refuses), so a path can no
    # longer pass locally yet be refused by the VM's sparse checkout.
    from cli.colab_runtime import is_checkout_relative_path

    if lane not in {'bundle', 'training'}:
        raise ValueError(f'Unknown runtime checkout lane: {lane}')
    # Bundle starts from the raw export and regenerates derived inputs.
    # Training consumes the committed prepared data (or its verified overlay).
    lane_members = ('dataset.csv',) if lane == 'bundle' else ('data', 'dataset.csv')
    members = ('src', 'scripts', 'config', 'requirements', 'artifacts/wheels',
               'artifacts/evidence', load_config()['paths']['models_dir'],
               training_cfg().preparation.smoke_dir,
               'pyproject.toml', 'requirements.txt', 'colab_backend.py',
               *evidence_members(), *lane_members, *extra)
    result = []
    for member in members:
        path = Path(member)
        if not is_checkout_relative_path(str(member)):
            raise ValueError('runtime checkout path must be repository-relative')
        if path.as_posix() not in result:
            result.append(path.as_posix())
    return tuple(result)


def checkout_inventory(extra=(), *, lane="bundle") -> tuple[str, ...]:
    """Freeze the tracked file inventory; never require caches or local outputs."""
    import subprocess
    from core.common import TRAIN_ROOT

    members = checkout_members(extra, lane=lane)
    paths = subprocess.check_output(
        ['git', 'ls-files', '-z', '--', *members], cwd=TRAIN_ROOT,
        text=True).split('\0')
    files = tuple(sorted(set(filter(None, paths)) | set(evidence_members())))
    missing = [name for name in files if not (TRAIN_ROOT / name).is_file()]
    # A missing/untracked requested input must not disappear from ls-files.
    missing.extend(name for name in members if name != 'artifacts/wheels'
                   and not any(file == name or file.startswith(name + '/') for file in files))
    if missing:
        raise FileNotFoundError('Runtime preflight missing/untracked inputs: ' + ', '.join(sorted(set(missing))))
    return files


def checkout_preflight_script(files, root_expression: str = 'root') -> str:
    """Stdlib-only check, usable before dependency installation on either VM."""
    return f'''
from pathlib import Path
_runtime_root = Path({root_expression})
_runtime_files = {tuple(files)!r}
_runtime_missing = [name for name in _runtime_files if not (_runtime_root / name).is_file()]
if _runtime_missing:
    raise FileNotFoundError("Runtime preflight missing files: " + ", ".join(_runtime_missing))
print("[runtime-preflight] verified %d required files" % len(_runtime_files), flush=True)
'''


def remote_revision_preflight(repository: str, branch: str, files, *, revision=None) -> str:
    """Inspect the published Git tree locally before contacting a compute service."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory(prefix='er-runtime-preflight-') as folder:
        def git(*args):
            return subprocess.check_output(
                ['git', '-C', folder, *args], text=True, stderr=subprocess.PIPE,
                timeout=120).strip()
        git('init', '--bare', '--quiet')
        # Only commit/tree metadata is needed; do not download model/data blobs.
        git('fetch', '--quiet', '--depth=1', '--filter=blob:none', '--no-tags',
            repository, 'refs/heads/' + branch)
        published = git('rev-parse', 'FETCH_HEAD')
        if revision is not None and revision != published:
            raise ValueError(
                f'Runtime preflight revision mismatch: {repository} branch={branch} '
                f'publishes {published}, but the launch pins {revision}. '
                'Push the intended revision or regenerate the launch for the configured branch.')
        entries = git('ls-tree', '-r', '-z', published).split('\0')
        available = {entry.split('\t', 1)[1] for entry in entries
                     if '\t' in entry and entry.startswith('100')}
        missing = sorted(set(files) - available)
        if missing:
            raise FileNotFoundError(
                f'Runtime preflight: published branch={branch} revision={published} '
                'is missing required files: ' + ', '.join(missing) +
                '. Push the runtime changes to this branch or correct the configured branch.')
    print(f'[runtime-preflight] published branch={branch} revision={published} '
          f'verified={len(files)} files', flush=True)
    return published


def published_tip(repository: str, branch: str) -> str:
    """Resolve <branch> tip ON THE ORIGIN after fetching it (the SSOT pin).

    `repository` comes from the lane config the caller already resolved
    (the fetch rides git's own `origin` remote, so it is validated once
    there); a fetch or rev-parse failure raises loud — a stale or
    unreachable tip may never back a staged payload.
    """
    import subprocess

    from core.common import TRAIN_ROOT

    fetch = subprocess.run(
        ['git', 'fetch', 'origin', branch, '-q'], cwd=TRAIN_ROOT,
        capture_output=True, text=True)
    if fetch.returncode != 0:
        raise RuntimeError(
            f'git fetch origin {branch} failed in {TRAIN_ROOT}: '
            f'{fetch.stderr.strip()}')
    parse = subprocess.run(
        ['git', 'rev-parse', f'origin/{branch}'], cwd=TRAIN_ROOT,
        capture_output=True, text=True)
    if parse.returncode != 0:
        raise RuntimeError(
            f'git rev-parse origin/{branch} failed in {TRAIN_ROOT}: '
            f'{parse.stdout.strip()}{parse.stderr.strip()}')
    return parse.stdout.strip()


def require_published_tip_match(head: str, repository: str, branch: str) -> str:
    """The owner order 'origin/<branch> == HEAD' baked into every staging
    surface: a staging surface that sets a REVISION pin calls this BEFORE it
    writes any payload; a pin that is not the published branch tip is a
    staged race (the 84ce2d0 receipt vs 02dec14 HEAD class) and must never
    stage. Returns the published tip so receipts can record it."""
    tip = published_tip(repository, branch)
    if head != tip:
        raise RuntimeError(
            f'staged revision {head[:12]} != origin/{branch} tip {tip[:12]} '
            '— pull or push so the pin is the published state, then re-stage')
    return tip


def staged_kernel_preflight(stage_dir: Path) -> None:
    """Check the actual staged script, before any Kaggle CLI call."""
    import ast
    import json

    metadata = json.loads((stage_dir / 'kernel-metadata.json').read_text())
    script = (stage_dir / metadata['code_file']).read_text()
    values = {}
    for node in ast.walk(ast.parse(script)):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    'REPOSITORY', 'BRANCH', 'REVISION', '_runtime_files'
                }:
                    values[target.id] = ast.literal_eval(node.value)
    required = {'REPOSITORY', 'BRANCH', 'REVISION', '_runtime_files'}
    if required - values.keys():
        raise ValueError('Staged kernel lacks runtime preflight inventory; regenerate it')
    remote_revision_preflight(values['REPOSITORY'], values['BRANCH'],
                              values['_runtime_files'], revision=values['REVISION'])
