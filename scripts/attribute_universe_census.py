#!/usr/bin/env python3
"""Attribute universe census: measure the FULL 37-key raw-attribute registry
on the live dataset, write the census bound as
``layouts.attribute_universe_census`` (artifacts/evidence/, sorted keys,
including the datagen budget), and verify the census against the pinned
measured baseline (src/core/attribute_universe.MEASURED_BASELINE).

Read-only over inputs. Fail-loud: a live census that drifts from the
baseline outside the +/-1% tolerance exits non-zero BEFORE the artifact is
written — a silently-stale census JSON is exactly the failure mode the
drift gate exists to prevent.

Live-data check only: unit tests pin census SEMANTICS on synthetic frames
(tests/test_attribute_universe.py), never the 71,623-row corpus.
"""

from __future__ import annotations

import json

from core.attribute_universe import AttributeUniverse, MEASURED_BASELINE
from core.common import artifact, ensure_parent, load_dataset
from core.common import trace_artifact


def main(argv: list[str] | None = None) -> int:
    frame = load_dataset()
    universe = AttributeUniverse(frame)
    census = universe.census()
    budget = universe.datagen_budget(census=census)

    print(f"[census] rows={census['rows']:,} valid_gtin_rows={census['valid_gtin_rows']:,} "
          f"same-gtin pairs={census['same_gtin_pairs_total']:,}")
    print(f"{'key':44}{'rows':>8}{'sets':>7}{'pairs':>8}{'confl':>7}{'rate':>7}")
    for key, stats in sorted(census["keys"].items()):
        print(f"{key:44}{stats['rows_populated']:>8,}{stats['distinct_value_sets']:>7}"
              f"{stats['same_gtin_pairs_both_populated']:>8}{stats['conflict_pairs']:>7}"
              f"{stats['conflict_rate']:>7.4f}")

    universe.verify_census(census)
    print(f"[verify] {len(MEASURED_BASELINE)} pinned keys reproduce within +/-1%")

    census_path = artifact("attribute_universe_census")
    payload = {
        "census": census,
        "datagen_budget": budget,
        "baseline": {
            key: dict(sorted(pinned.items()))
            for key, pinned in sorted(MEASURED_BASELINE.items())
        },
    }
    ensure_parent(census_path)
    census_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    trace_artifact("attribute_universe_census", census_path, producer=__name__)
    print(f"wrote {census_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
