"""Standalone Colab bundle lane: run the CSV-to-inputs lifecycle on a VM CPU.

Supersedes the `--what bundle` path in cli.colab (whose remote pin-patch
regex broke through nested f-string escaping — a character class that
matched a literal backslash instead of digits, silently no-op-ing the pin
rewrite). This module:

- reuses cli.colab's proven provisioning helpers (check_colab_cli,
  ensure_session, prepare_remote_layout, run_colab_exec_stream, upload and
  download transports) — the connection/provisioning path that always worked;
- patches the VM audit pins with a PLAIN LINE FILTER (no regex, no nested
  escaping to get wrong): any cohort export drives the lane while the drift
  gate still catches mid-run export changes;
- captures prepare_all's stderr on the VM and prints its tail on failure, so
  a VM failure is one log read away from diagnosed;
- never tears the VM down: the session stays open (use `colab stop` or the
  stop lane to end it explicitly).

Usage:
  PYTHONPATH=src python -m cli.colab_bundle --dataset-csv dataset_50pct.csv
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

_REMOTE_SCRIPT = '''
import glob, hashlib, os, subprocess, sys, tarfile
import pandas as pd
csv_path = __REMOTE_ROOT__ + "/dataset.csv"
_rows = len(pd.read_csv(csv_path, dtype=str))
_sha = hashlib.sha256(open(csv_path, "rb").read()).hexdigest()
_cfg = __REMOTE_ROOT__ + "/config/training.yaml"
_patched = []
for _line in open(_cfg, encoding="utf-8").readlines():
    if _line.startswith("  source_export_expected_rows:"):
        _patched.append("  source_export_expected_rows: %d\\n" % _rows)
    elif _line.startswith("  source_export_expected_sha256:"):
        _patched.append('  source_export_expected_sha256: "%s"\\n' % _sha)
    else:
        _patched.append(_line)
open(_cfg, "w", encoding="utf-8").write("".join(_patched))
print("[bundle] VM audit pins -> rows=%d sha=%s..." % (_rows, _sha[:12]), flush=True)
_err = open(__REMOTE_ROOT__ + "/prepare_all_stderr.log", "w")
_proc = subprocess.run(
    [sys.executable, "-m", "training.prepare_all"],
    cwd=__REMOTE_ROOT__,
    stderr=_err,
)
_err.close()
if _proc.returncode != 0:
    _tail = open(__REMOTE_ROOT__ + "/prepare_all_stderr.log").read()[-4000:]
    print("[bundle] prepare_all FAILED rc=%d; stderr tail:" % _proc.returncode, flush=True)
    print(_tail, flush=True)
    raise RuntimeError("prepare_all failed on the VM (rc=%d)" % _proc.returncode)
_run_dir = sorted(glob.glob(__REMOTE_ROOT__ + "/results/training_prep/*"))[-1]
_delivery = __REMOTE_ROOT__ + "/bundle_delivery.tar.gz"
with tarfile.open(_delivery, "w:gz") as _tar:
    _tar.add(_run_dir, arcname="training_prep/" + os.path.basename(_run_dir))
    for _rel in ("data/canonical_records.csv", "data/gate_results.csv",
                 "data/dataset_deduped.csv", "data/labeled_pairs.csv",
                 "data/final_validation.csv", "data/number_tokens_reference.csv",
                 "data/sku_to_rep.csv"):
        if os.path.exists(__REMOTE_ROOT__ + "/" + _rel):
            _tar.add(__REMOTE_ROOT__ + "/" + _rel, arcname=_rel)
    if os.path.isdir(__REMOTE_ROOT__ + "/data/track_setup"):
        _tar.add(__REMOTE_ROOT__ + "/data/track_setup", arcname="data/track_setup")
    if os.path.isdir(__REMOTE_ROOT__ + "/data/prepared/full"):
        _tar.add(__REMOTE_ROOT__ + "/data/prepared/full", arcname="data/prepared/full")
print("[bundle] delivery archive ready", flush=True)
'''


def main() -> None:
    from cli.colab import (
        SESSION,
        _BOOTSTRAP,
        _download_file_with_visibility,
        _upload_with_retries,
        check_colab_cli,
        ensure_session,
        prepare_remote_layout,
        run_colab_exec_stream,
    )
    from core.common import TRAINING_RESULTS

    parser = argparse.ArgumentParser(
        description="Run the CSV-to-inputs bundle lifecycle on a Colab VM CPU."
    )
    parser.add_argument("--dataset-csv", type=Path, default=None)
    arguments = parser.parse_args()

    check_colab_cli()
    ensure_session()
    prepare_remote_layout(minimal_runtime=False)

    from core.common import TRAIN_ROOT

    source = arguments.dataset_csv or (TRAIN_ROOT / "dataset.csv")
    if not source.is_file():
        raise FileNotFoundError(f"bundle raw export not found: {source}")
    run_id = datetime.now(timezone.utc).strftime("bundle_%m%dT%H%M%S%fZ")
    delivery_dir = TRAINING_RESULTS / ("colab_bundle_" + run_id)
    delivery_dir.mkdir(parents=True, exist_ok=True)

    print(f"[bundle] uploading raw export {source} ...", flush=True)
    _upload_with_retries(source, f"{REMOTE_ROOT}/dataset.csv", timeout=3600)

    script = (_BOOTSTRAP + _REMOTE_SCRIPT).replace("__REMOTE_ROOT__", REMOTE_ROOT)
    run_colab_exec_stream(SESSION, script, timeout=4 * 3600, log_name="bundle")

    _download_file_with_visibility(
        remote=f"{REMOTE_ROOT}/bundle_delivery.tar.gz",
        local=delivery_dir / "bundle_delivery.tar.gz",
        worker=None,
        index=1,
        total=1,
        run_id=delivery_dir.name,
    )
    print(f"[bundle] delivered -> {delivery_dir / 'bundle_delivery.tar.gz'}", flush=True)


if __name__ == "__main__":
    main()
