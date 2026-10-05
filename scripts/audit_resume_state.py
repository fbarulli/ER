#!/usr/bin/env python
"""D2 resume audit: count resume-able checkpoints across run dirs.

Read-only scan (telemetry for the resume-semantics decision). Reports:
  - ``resume_pointer_v1``    .resume/<name>.dvc pointer files (v1 manifestless),
                             split by ``outs: []`` stranded vs outs-listed
  - ``local_trainer_state``  local untouched ``checkpoint-*/trainer_state.json``
  - ``resume_meta_v2``       future resume manifests (contract version bump),
                             always 0 until D2 introduces them
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RESUME_DIR = ".resume"
RESUME_META_V2 = "resume_meta.json"


def _count_dvc_pointers(resume_dir: Path) -> dict[str, int]:
    stranded = 0
    outs_listed = 0
    total = 0
    for pointer in sorted(resume_dir.glob("*.dvc")):
        total += 1
        try:
            payload = pointer.read_text(encoding="utf-8")
            import yaml

            no_outs = (yaml.safe_load(payload) or {}).get("outs", []) == []
        except Exception:
            no_outs = False
        if no_outs:
            stranded += 1
        else:
            outs_listed += 1
    return {"v1_pointer_total": total, "v1_stranded": stranded, "v1_outs_listed": outs_listed}


def scan(root: Path) -> dict[int, int] | dict[str, int | dict[str, int]]:
    root = Path(root)
    report = {
        "root": str(root),
        "resume_pointer_v1": _count_dvc_pointers(root / RESUME_DIR)
        if (root / RESUME_DIR).is_dir()
        else {"v1_pointer_total": 0, "v1_stranded": 0, "v1_outs_listed": 0},
        "local_trainer_state": [str(p) for p in sorted(root.rglob("trainer_state.json"))],
        "resume_meta_v2": [str(p) for p in sorted(root.rglob(RESUME_META_V2))],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default="results", type=Path)
    args = parser.parse_args()
    print(json.dumps(scan(args.root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
