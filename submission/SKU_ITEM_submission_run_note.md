# SKU-to-item prediction run

- Input: the full deduplicated dataset (`61,529` SKU rows); no sample limit.
- Model: previous mixed-run checkpoint `1589`.
- Training profile: `masking_only`; ANN mining and attribute-conflict mining were disabled.
- Inference device: CPU.
- Inference threshold: `0.62`.
- Required output contract: exactly `SKU_ID,ITEM_ID`.
- Prediction method: top-1 cosine assignment against the canonical item map; scores below the threshold receive the configured unmatched prefix.
