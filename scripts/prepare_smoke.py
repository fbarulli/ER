#!/usr/bin/env python3
"""The ONE production entry over ``model_tracks.smoke_inputs.prepare_smoke``.

``prepare_smoke`` is the canonical sampler: it projects a frozen parent prepared
setup into a smaller CPU lifecycle smoke that retains the parent's split
assignments and materialized augmentation lineage, and writes the child
``suite.yaml``. It had no production caller (only tests), so this thin entry is
the single point an operator runs before launching a sampled suite:

    uv run python scripts/prepare_smoke.py --setup data/track_setup \
        --output data/prepared/smoke_1000 --sample 1000

The printed path is the child ``suite.yaml``; launch it with
``er-colab --what tracks --tracks-config <that path>``. All behavior lives in
``prepare_smoke``; this module is only its argument boundary.
"""
import argparse
from pathlib import Path


def main() -> None:
    from core.run_log import RunLogger
    from model_tracks.smoke_inputs import prepare_smoke

    RunLogger.configure_console()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setup", type=Path, required=True,
                    help="the frozen parent prepared setup dir (e.g. data/track_setup)")
    ap.add_argument("--output", type=Path, required=True,
                    help="the smoke child dir to write (must not already exist)")
    ap.add_argument("--sample", type=int, default=100,
                    help="listing count for the smoke (default 100)")
    ap.add_argument("--suite-config", type=Path, default=None,
                    help="parent suite config (default: config/model_tracks.yaml)")
    args = ap.parse_args()
    suite = prepare_smoke(args.setup, args.output, sample=args.sample,
                          suite_config=args.suite_config)
    print(suite)


if __name__ == "__main__":
    main()
