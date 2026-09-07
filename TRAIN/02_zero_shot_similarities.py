"""Owner's zero-shot embedding-similarity script (bundle-adapted paths).

Encodes canonical strings for every non-hard_no candidate pair with each of
the 3 models and stores per-model cosine similarity in embedding_similarities.csv.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# model dirs resolve via config (models_dir / models_dir_sibling); hub id is
# the offline-last-resort fallback
from lib.common import RESULTS, F, _path, load_config

_cfg = load_config()
_model_dirs = [
    _path(_cfg["paths"]["models_dir"]),
    _path(_cfg["paths"]["models_dir_sibling"]),
]


def _m(sub):
    for d in _model_dirs:
        if (d / sub).exists():
            return str((d / sub).resolve())
    return (
        "microsoft/deberta-v3-base"
        if "deberta" in sub
        else f"sentence-transformers/{sub}"
    )


MODELS = {k: _m(sub) for k, sub in _cfg["models"].items()}
# CPU note: deberta-v3's relative attention is ~2000x slower than MiniLM on
# this torch-CPU build (measured 3.9 s/text vs 2 ms/text) — run deberta on
# the GPU lane; a CPU sweep leaves its column absent (04 warns, skips it).
SIM_COLUMNS = dict(_cfg["sim_columns"])

df_canon = pd.read_csv(RESULTS / F["canonical_records"])
df_gate = pd.read_csv(RESULTS / F["gate_results"], dtype={"gtin1": str, "gtin2": str})
assert "gtin" in df_canon.columns and "canonical" in df_canon.columns
gtin_to_canon = dict(
    zip(df_canon["gtin"].astype(str), df_canon["canonical"].astype(str))
)

# score EVERY gate pair (candidates AND hard_no): the evaluation set
# (labeled_pairs) draws its negatives from hard_no rows — skipping them
# would leave 04 with positives only (silent class drop)
candidates = df_gate.copy()
print(f"Total gate pairs to score: {len(candidates)}")

unique_gtins = sorted(set(candidates["gtin1"]).union(set(candidates["gtin2"])))
texts = [gtin_to_canon[g] for g in unique_gtins]
print(f"Unique GTINs to encode: {len(unique_gtins)}")

results = candidates[["gtin1", "gtin2", "gate_decision", "gate_reason"]].copy()
out = RESULTS / F["embedding_similarities"]

# resume: models already scored in a previous (crashed) run are skipped
have: list[str] = []
if out.exists():
    done = pd.read_csv(out, dtype={"gtin1": str, "gtin2": str})
    have = [c for c in done.columns if c.startswith("sim_") and done[c].notna().all()]
    if have:
        print(f"resuming — already scored: {have}", flush=True)

for model_key, model_path in MODELS.items():
    col = SIM_COLUMNS[model_key]
    if col in have:
        print(f"--- Model: {model_key} already scored, skip ---", flush=True)
        continue
    print(f"\n--- Model: {model_key} ---", flush=True)
    model = SentenceTransformer(model_path, device=DEVICE)
    model.max_seq_length = 128  # matches the trainer; keeps deberta CPU-fast
    embeddings = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    # vectorized: row-indexed gather, one einsum (no iterrows)
    gtin_idx = {g: i for i, g in enumerate(unique_gtins)}
    ai = results["gtin1"].map(gtin_idx).to_numpy()
    bi = results["gtin2"].map(gtin_idx).to_numpy()
    results[col] = np.einsum("ij,ij->i", embeddings[ai], embeddings[bi])
    del model, embeddings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # incremental write: a crash never loses the completed models
    keep = ["gtin1", "gtin2", "gate_decision", "gate_reason"] + [
        c for c in results.columns if c.startswith("sim_")
    ]
    results[keep].to_csv(out, index=False)
    print(f"    wrote {col} ({len(results):,} rows)", flush=True)

print(f"\nSaved {out} ({len(results):,} rows)")
