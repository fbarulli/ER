# ER matching recommendation: ANN, graph, and Semantic IDs

## Recommendation

Use ANN to retrieve plausible candidates, score candidate pairs using text and structured evidence, then resolve matches with a constrained graph. Keep the residual-quantized Semantic IDs (SIDs) as an experimental retrieval or corroboration signal. Do not use SID agreement as product identity or let it override trusted identifiers and attribute conflicts.

```mermaid
flowchart LR
    A[Title, attributes, description] --> B[Structured evidence with source and confidence]
    B --> C[ANN plus exact and lexical candidate retrieval]
    C --> D[Pairwise match scoring]
    D --> E[Constrained graph resolution]
    E --> F[Canonical product ID or unresolved]
```

## Define the identity first

For exact product matching, the product identity should distinguish flavor, unit volume, and pack count. A product family relationship (for example, the same drink in different sizes) is useful, but should be represented separately from exact product identity. This prevents a graph from merging related variants just because their descriptions and embeddings are similar.

## Proposed matching flow

1. **Extract evidence.** Parse title and attributes into typed values. Use the description to corroborate an unclear title, recording the source and confidence for each value. Treat missing values as unknown. Retain contradictory declarations with a consistency flag.
2. **Retrieve broadly.** Use ANN for semantic candidates and supplement it with exact identifiers and lexical retrieval. Tune candidate depth for recall; measure whether the true canonical appears in the candidate set before evaluating downstream assignment.
3. **Score each candidate pair.** Combine text similarity with structured agreement and conflict signals. Train and evaluate on hard distinctions such as same brand and flavor but different volume, same drink in a different pack size, and regular versus sugar-free.
4. **Resolve with constraints.** Build a graph from scored SKU-to-canonical and SKU-to-SKU edges. Preserve trusted GTIN locks. Before joining components, check conflicts across the full components, not only the edge endpoints. Leave ambiguous records unresolved instead of allowing them to bridge incompatible products.
5. **Assign the canonical product ID.** Assign an ID only after resolution. Keep product-family links separate.

A retrieve-then-rerank setup is a common approach: a fast first-stage encoder finds candidates, and a more precise pair scorer orders or evaluates them. See the [Sentence Transformers retrieve-and-rerank guide](https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html).

## Existing experiment results

The saved [SID graph evaluation](/home/opc/ONE/ER/artifacts/sid/sid_graph_eval.json) reports results for a fixture of 360 SKUs and 13,250 canonicals:

| Evaluation arm | Assignment accuracy |
| --- | ---: |
| Top-1 cosine assignment | 60.0% |
| Cosine graph | 82.8% |
| Graph with SID-admitted edges | 63.1% |

The cosine graph is promising. The current graph-plus-SID arm is worse, so SID edges should not be enabled as a production boost based on this experiment. The graph evaluator uses different cosine thresholds for top-1 and graph edges, so this is not a controlled comparison. The graph arms also fail to preserve every GTIN-locked assignment: only 81.7% of the `both_equal` stratum is correct for graph-plus-SID. Fix and test lock preservation before trusting the graph result.

A separate [SID hybrid evaluation](/home/opc/ONE/ER/artifacts/sid/sid_hybrid_eval.json) uses 324 test pairs. Its hybrid arm has a small F1 gain over the bi-encoder (0.936 versus 0.929), while recall@1 falls from 0.580 to 0.395. This supports evaluating SID as an auxiliary signal, not assuming it improves candidate retrieval or identity decisions. The pair test and graph assignment test measure different tasks and should not be compared as if they were one benchmark.

## Evaluation plan

Compare pointwise assignment and constrained-graph resolution on the same held-out records with the same candidate pool and calibrated thresholds. Keep training, calibration, and test identities disjoint. Report:

- candidate recall@K, before pair scoring;
- pairwise precision, recall, and calibration;
- exact assignment accuracy and unmatched rate;
- false merges and false splits, overall and by GTIN evidence stratum;
- GTIN-lock violations and attribute-conflict violations;
- cluster-size and transitive-bridge diagnostics.

Run ablations for cosine-only, cosine plus structured evidence, and those same arms with SID signals. SID should stay only if it adds value on the held-out assignment and cluster metrics without increasing false merges or violating locks.

## Current Semantic ID interpretation

In this repository, Semantic IDs are residual-quantized codes derived from embeddings. They are useful as experimental codes for retrieval or pairwise corroboration, but code agreement is not independent product evidence. The stable canonical product ID should come from the resolved entity assignment, not from an embedding code.
