"""Script-side persist of the /gate rendered evidence sample.

Regenerates dashboard/evidence/datagen/gate_decision_sample.json — the same
deterministic 5-per-bucket snapshot the gate page renders — by calling the
dashboard's own builders (SSOT: data/gate_results.csv + dataset.csv through
core.common.load_dataset). The route no longer writes on GET.

Usage: python dashboard/write_gate_sample.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import app as dashboard  # noqa: E402  (sets BROADWAY_* env + PATH for core/*)


def main() -> None:
    g = dashboard._gate_results_frame(dashboard._gate_results_path.stat().st_mtime_ns)
    raw, listings = dashboard._gate_raw(dashboard.DATA_PATH.stat().st_mtime_ns)
    snapshot = dashboard._gate_snapshot(g, raw, listings)
    target = dashboard._gate_dir / 'gate_decision_sample.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(snapshot, indent=1))
    counts = {k: len(v) for k, v in snapshot.items()}
    print(f'wrote {target} · buckets {counts}')


if __name__ == '__main__':
    main()
