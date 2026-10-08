"""cProfile driver for bench_text.py (writes a .prof next to this file)."""
from __future__ import annotations

import cProfile
import contextlib
import io
import pstats
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE / 'bench_text.py'


def main() -> int:
    top = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    only = sys.argv[2] if len(sys.argv) > 2 else None
    argv = ['bench_text', '--label', 'prof', '--reps', '2']
    if only:
        argv += ['--only', only]
    saved = sys.argv
    sys.argv = argv
    profiler = cProfile.Profile()
    try:
        profiler.enable()
        with contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(str(BENCH), run_name='__main__')
    except SystemExit:
        pass
    finally:
        profiler.disable()
        sys.argv = saved
    out = HERE / 'bench.prof'
    profiler.dump_stats(str(out))
    stats = pstats.Stats(profiler)
    stats.sort_stats('tottime').print_stats(top)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
