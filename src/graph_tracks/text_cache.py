"""Frozen local MiniLM cache using the existing model-input composition SSOT."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from graph_tracks.data import file_hash


def checkpoint_hash(path: Path) -> str:
    if not path.is_dir():
        raise ValueError('checkpoint must be a local directory; remote revisions are not pinned here')
    files = sorted(p for p in path.rglob('*') if p.is_file())
    if not files:
        raise ValueError('empty checkpoint')
    digest = hashlib.sha256()
    for file in files:
        digest.update(str(file.relative_to(path)).encode())
        digest.update(bytes.fromhex(file_hash(file)))
    return digest.hexdigest()


def create_cache(catalog: Path, checkpoint: Path, output: Path, *, batch_size=64, device='cpu'):
    from core.model_input import build_sku_text, model_input_composition, model_input_info
    from core.product_identity import row_identity
    from sentence_transformers import SentenceTransformer
    if output.exists():
        raise FileExistsError(output)
    frame = pd.read_csv(catalog, dtype=str, keep_default_na=False, low_memory=False)
    if 'product_id' not in frame or frame.product_id.duplicated().any() or (frame.product_id == '').any():
        raise ValueError('catalog requires unique nonempty product_id')
    fingerprint = checkpoint_hash(checkpoint)
    texts = [build_sku_text(row, model_input_info(row_identity(row).as_mapping()))
             for _, row in frame.iterrows()]
    model = SentenceTransformer(str(checkpoint), device=device, local_files_only=True)
    model.eval()
    vectors = model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=True)
    from core.identity_policy import POLICY_PATH
    from core.common import TRAIN_ROOT
    metadata = {'checkpoint_sha256': fingerprint,
                'identity_policy_sha256': file_hash(POLICY_PATH),
                'identity_dimensions_sha256': file_hash(TRAIN_ROOT / 'config' / 'identity_dimensions.yaml'),
                'composition': model_input_composition().model_dump(mode='json'),
                'catalog_sha256': file_hash(catalog),
                'text_sha256': hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as handle:
        np.savez_compressed(handle, ids=frame.product_id.to_numpy(dtype=str),
                            embeddings=vectors, metadata=json.dumps(metadata, sort_keys=True))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('catalog', 'checkpoint', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    create_cache(args.catalog, args.checkpoint, args.output, batch_size=args.batch_size, device=args.device)

if __name__ == '__main__':
    main()
