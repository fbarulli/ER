"""Repair feature lineage in existing bundles without regenerating their pairs.

Usage: PYTHONPATH=src python scripts/repair_augmented_features.py BUNDLE [...]
Checks the original digest and retains the recorded build configuration.
"""

import argparse
import gzip
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np

from training.masking import extend_augmented_features
from training.prepared_bundle import load_prepared_bundle


def repair(path: Path) -> None:
    header = path.with_suffix(path.suffix + ".json")
    manifest = json.loads(header.read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError(f"original bundle digest mismatch: {path}")
    with gzip.open(path, "rb") as stream:
        data = pickle.load(stream)
    audit = data["mask_audit"] + data["hard_negative_mask_audit"]
    if not audit:
        load_prepared_bundle(path)
        print(f"{path}: no augmented rows")
        return
    first = min(int(row["copy_payload_idx"]) for row in audit)
    old = np.asarray(data["structured_features"])
    fixed = extend_augmented_features(old[:first], data["payload"], audit)
    changed = int(np.count_nonzero(np.any(old != fixed, axis=1)))
    if not changed:
        load_prepared_bundle(path)
        print(f"{path}: already consistent")
        return
    data["structured_features"] = fixed
    temporary = path.with_name(path.name + ".repairing")
    temporary_header = temporary.with_suffix(temporary.suffix + ".json")
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as stream:
                pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
        manifest["sha256"] = hashlib.sha256(temporary.read_bytes()).hexdigest()
        temporary_header.write_text(json.dumps(manifest, indent=2) + "\n")
        load_prepared_bundle(temporary)
        os.replace(temporary, path)
        os.replace(temporary_header, header)
    finally:
        temporary.unlink(missing_ok=True)
        temporary_header.unlink(missing_ok=True)
    print(f"{path}: repaired {changed} feature rows; payload and pairs preserved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundles", type=Path, nargs="+")
    for bundle in parser.parse_args().bundles:
        repair(bundle)
