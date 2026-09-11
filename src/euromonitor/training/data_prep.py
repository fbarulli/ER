"""data_prep.py — run the OFFICIAL data-prep pipeline (euromonitor.pipeline.run_within_brand_pipeline).

Loads the raw export (raw column names, dtype=str), runs the within-brand
pipeline (extraction → canonical → gating → similarity), writes
canonical_records.csv + gate_results.csv into the config results dir.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from euromonitor.pipeline import run_within_brand_pipeline
from euromonitor.core.common import load_raw_export


def main() -> None:
    df = load_raw_export()
    pairs, canon = run_within_brand_pipeline(df)
    print(f"pairs: {len(pairs):,} | canonical records: {len(canon):,}")
    # AUDIT 2026-09-09: digit tokens resolved by the regex fallback (not in
    # the reference CSV) — the one degradation the numbers lane allows;
    # printed so it can never be silent.
    import euromonitor.pipeline as _dp

    print(
        f"[numbers] {_dp._UNSEEN_TOKEN_TOTAL:,} digit-token resolutions "
        f"via regex fallback (not in reference CSV)"
    )


if __name__ == "__main__":
    main()
