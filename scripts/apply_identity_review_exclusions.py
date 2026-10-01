#!/usr/bin/env python3
"""Remove reviewed identities from frozen label/split artifacts, retaining audit evidence.

The source export and deduped listing catalog are preserved. Future generations
are protected by runtime guards. This repairs existing artifacts in place only
with --apply; removed rows and before/after hashes are retained separately.
"""
import argparse
import hashlib
import json
import logging
from pathlib import Path

import pandas as pd
from core.common import F, TRAIN_ROOT
from core.identity_policy import POLICY_PATH, review_mask

logger = logging.getLogger(__name__)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(apply=False):
    names = ['canonical_records', 'gate_results', 'labeled_pairs', 'final_validation',
             'validation_fold_map']
    records = []
    for name in names:
        path = F.get(name)
        if path is None or not path.exists():
            continue
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
        columns = [c for c in ('gtin', 'gtin1', 'gtin2', 'barcode') if c in frame]
        if not columns:
            continue
        mask = pd.Series(False, index=frame.index)
        for column in columns:
            mask |= review_mask(frame[column])
        entry = {'artifact': name, 'path': str(path.relative_to(TRAIN_ROOT)),
                 'before_rows': len(frame), 'excluded_rows': int(mask.sum()),
                 'after_rows': int((~mask).sum()), 'before_sha256': sha(path),
                 'removed_rows': frame.loc[mask].to_dict('records')}
        if apply and mask.any():
            temporary = path.with_suffix(path.suffix + '.review-tmp')
            frame.loc[~mask].to_csv(temporary, index=False)
            temporary.replace(path)
        entry['after_sha256'] = sha(path)
        records.append(entry)
        logger.info('%s: %s excluded / %s original', name, mask.sum(), len(frame))
    return {'applied': apply, 'policy_sha256': sha(POLICY_PATH), 'artifacts': records,
            'source_export_modified': False, 'deduped_catalog_modified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--output', type=Path, default=TRAIN_ROOT / 'dashboard/evidence/identity/identity_exclusion_application.json')
    args = parser.parse_args()
    result = run(args.apply)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
