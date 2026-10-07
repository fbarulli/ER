"""Consolidate run logs into one ordered file.

The operator otherwise chases per-stage / per-lane / per-run logs across
``logs/``, ``results/``, ``training_results/`` and the scratch roots. This
walks the given roots, orders every log by mtime, and writes a single
concatenated document with a header per source so one file tells the whole
story.

  PYTHONPATH=src .venv/bin/python scripts/consolidate_logs.py
  # only the last 3h, include scratch smoke logs, cap each file:
  PYTHONPATH=src .venv/bin/python scripts/consolidate_logs.py \
      --roots logs results training_results /tmp/opencode \
      --since-minutes 180 --max-bytes-per-file 200000

Defaults are behavior-neutral (read-only discovery; one output file).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ROOTS = ('logs', 'results', 'training_results', 'ablation_profile', 'training_profile')
DEFAULT_EXTENSIONS = ('.log', '.jsonl')


def discover(roots: tuple[str, ...], extensions: tuple[str, ...]) -> list[Path]:
    """Every log file under the roots (a file root is taken verbatim)."""
    found: list[Path] = []
    for root in roots:
        path = Path(root)
        if not path.exists():
            continue
        if path.is_file():
            if path.suffix in extensions:
                found.append(path)
            continue
        for candidate in path.rglob('*'):
            if candidate.is_file() and candidate.suffix in extensions:
                found.append(candidate)
    return found


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', nargs='*', default=list(DEFAULT_ROOTS))
    parser.add_argument('--extensions', nargs='*', default=list(DEFAULT_EXTENSIONS))
    parser.add_argument('--output', type=Path, default=Path('logs/consolidated.log'))
    parser.add_argument('--since-minutes', type=float, default=None,
                        help='only files modified within the last N minutes')
    parser.add_argument('--max-bytes-per-file', type=int, default=None,
                        help='truncate each source after N bytes')
    args = parser.parse_args()

    cutoff = None
    if args.since_minutes is not None:
        cutoff = datetime.now(timezone.utc).timestamp() - args.since_minutes * 60

    entries = []
    for path in discover(tuple(args.roots), tuple(args.extensions)):
        mtime = _mtime(path)
        if cutoff is not None and mtime < cutoff:
            continue
        entries.append((mtime, path))
    entries.sort(key=lambda item: item[0])

    output: Path = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open('w', encoding='utf-8', errors='replace') as handle:
        handle.write(f"# consolidated logs — {datetime.now(timezone.utc).isoformat()} — "
                     f"{len(entries)} file(s)\n")
        handle.write(f"# roots: {', '.join(str(root) for root in args.roots)}\n")
        for mtime, path in entries:
            stamp = datetime.fromtimestamp(mtime, timezone.utc).isoformat() if mtime else 'unknown'
            handle.write('\n' + '=' * 100 + '\n')
            handle.write(f'===== {path}  (mtime={stamp})\n')
            handle.write('=' * 100 + '\n')
            try:
                data = path.read_bytes()
            except OSError as exc:
                handle.write(f'[unreadable: {exc}]\n')
                continue
            if args.max_bytes_per_file is not None and len(data) > args.max_bytes_per_file:
                data = data[:args.max_bytes_per_file] + b'\n[... truncated ...]\n'
            handle.write(data.decode('utf-8', errors='replace'))
            if not data.endswith(b'\n'):
                handle.write('\n')
            written += 1
    print(f'{output}  files={written}  bytes={output.stat().st_size}')


if __name__ == '__main__':
    main()
