"""01_data_prep.py — run the OFFICIAL data-prep pipeline (data_pipe.run_within_brand_pipeline).

Loads the raw export (raw column names, dtype=str), runs the within-brand
pipeline (extraction → canonical → gating → similarity), writes
canonical_records.csv + gate_results.csv into the config results dir.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from data_pipe import run_within_brand_pipeline
from lib.common import load_raw_export


def main() -> None:
    df = load_raw_export()
    pairs, canon = run_within_brand_pipeline(df)
    print(f"pairs: {len(pairs):,} | canonical records: {len(canon):,}")


if __name__ == "__main__":
    main()
