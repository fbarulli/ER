"""Contention-robust A/B driver for bench_text.py.

The host has 4 CPUs and the sibling reg2 lanes run profiling rounds, pytest and
cargo builds on it, so a single before/after pair measured back-to-back is not
trustworthy (observed 33 % spread between best and median inside one run).
This driver ALTERNATES the two source trees round-robin and keeps the minimum
per bench per tree: the minimum is the least contended sample, and alternating
makes both trees see a comparable contention mix.

    python ab_bench.py --a /tmp/opc/reg2-A-pristine/src --b /tmp/opc/ER-reg2-A/src \
        --rounds 3 --reps 2 --tag r16
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE / 'bench_text.py'


def run_tree(src: str, reps: int, only: str | None, out: Path) -> dict:
    env = dict(os.environ)
    env['ER_BENCH_SRC'] = src
    argv = [sys.executable, str(BENCH), '--label', 'ab', '--reps', str(reps),
            '--json', str(out)]
    if only:
        argv += ['--only', only]
    subprocess.run(argv, env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return json.loads(out.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', required=True, help='baseline src tree')
    parser.add_argument('--b', required=True, help='optimized src tree')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--reps', type=int, default=2)
    parser.add_argument('--only', default=None)
    parser.add_argument('--tag', default='ab')
    args = parser.parse_args()

    best: dict[str, dict[str, float]] = {'a': {}, 'b': {}}
    digests: dict[str, set] = {'a': set(), 'b': set()}
    totals: dict[str, list] = {'a': [], 'b': []}
    with tempfile.TemporaryDirectory() as tmp:
        for round_no in range(args.rounds):
            for key, src in (('a', args.a), ('b', args.b)):
                result = run_tree(src, args.reps, args.only,
                                  Path(tmp) / f'{key}{round_no}.json')
                digests[key].add(result['digest'])
                totals[key].append(result['total_best'])
                for name, stats in result['benches'].items():
                    previous = best[key].get(name)
                    if previous is None or stats['best'] < previous:
                        best[key][name] = stats['best']
                print(f'  round {round_no} tree {key}: total_best={result["total_best"]:.4f}',
                      flush=True)

    names = [name for name in best['a']]
    print(f'\n{"bench":20s} {"A (baseline)":>13} {"B (optimized)":>13} {"delta":>9}')
    total_a = total_b = 0.0
    for name in names:
        a = best['a'][name]
        b = best['b'].get(name, float('nan'))
        total_a += a
        total_b += b
        print(f'{name:20s} {a:13.6f} {b:13.6f} {(b - a) / a * 100:8.1f}%')
    print(f'{"TOTAL":20s} {total_a:13.6f} {total_b:13.6f} '
          f'{(total_b - total_a) / total_a * 100:8.1f}%')
    print(f'\ndigest A {sorted(digests["a"])}')
    print(f'digest B {sorted(digests["b"])}')
    print(f"IDENTICAL={digests['a'] == digests['b'] and len(digests['a']) == 1}")
    payload = {'tag': args.tag, 'a': args.a, 'b': args.b, 'rounds': args.rounds,
               'reps': args.reps, 'best_a': best['a'], 'best_b': best['b'],
               'totals_a': totals['a'], 'totals_b': totals['b'],
               'digests_a': sorted(digests['a']), 'digests_b': sorted(digests['b'])}
    (HERE / f'{args.tag}.json').write_text(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
