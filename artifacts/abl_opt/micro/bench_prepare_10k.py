"""End-to-end A/B of the 10k prepare stage (the stage row_identity lives in).

`ablation.prepare` on the real 10k cohort exercises the identity extractor
(11,441 row_identity calls) and the pack/volume scan family (100,948
_PackEvidenceReader.read calls); this script times it without the profiler and
digests the produced request, so a rewrite can be shown to change no prepared
input while being faster.

  python artifacts/abl_opt/micro/bench_prepare_10k.py --label before
Run it once with the change reverted (git stash) and once with it applied.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('EUROMONITOR_PROJECT_ROOT', str(ROOT))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import ablation_timing_10k as harness10k  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', default='')
    parser.add_argument('--save', default=None, help='keep the prepared request here')
    args = parser.parse_args()
    harness10k.ensure_inputs()
    from model_tracks import ablation
    load_before = os.getloadavg()
    started = time.perf_counter()
    request_path = ablation.prepare(harness10k.CATALOG, harness10k.PAIRS,
                                    harness10k.CHECKPOINT, config=harness10k.CONFIG)
    seconds = time.perf_counter() - started
    result = {'label': args.label, 'wall_seconds': round(seconds, 3),
              'loadavg_start': [round(v, 2) for v in load_before],
              'loadavg_end': [round(v, 2) for v in os.getloadavg()],
              'request': str(request_path),
              'fingerprint_request': harness10k.fingerprint_prepare(request_path)}
    if args.save:
        target = Path(args.save)
        target.write_text(json.dumps({'label': args.label, 'wall_seconds': result['wall_seconds'],
                                      'fingerprint_request': result['fingerprint_request'],
                                      'prepared_inputs_size': json.loads(request_path.read_text())
                                      ['prepared_inputs']['size']}, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
