# Graph quality and training inputs

This is an experiment plan for graph quality, not a training-speed plan.
The observations below describe the `laya` branch. Proposed features are not
implemented and do not yet have measured accuracy gains.

## Current graph

- `src/graph_tracks/data.py`: a listing has numeric features and membership
  edges to typed attribute-value nodes. Relations come from
  `core.sku_identity.graph_schema()`, the schema source of truth.
- `src/graph_tracks/model.py`: `AttributeGNN.initial()` combines pooled
  attribute embeddings and numeric features. `context()` pools training
  listings into attribute states. `encode()` sends those states back to
  listings and averages the relation messages.
- Attribute types already have separate transformations. Within a type,
  neighbors receive equal weight; the final relation messages also receive
  equal weight. There are no confidence-weighted edges or direct
  listing-to-listing neighborhood relations.
- `PairScorer` applies a nonnegative learned coefficient and a bias to one
  graph cosine. It cannot distinguish pairs with the same cosine using
  explicit volume, pack, flavor, or other agreement/contradiction evidence.
- `src/graph_tracks/train.py`: full-batch pair supervision; one optimizer
  update per epoch. The loss combines binary classification and a cosine
  attraction/repulsion objective. This is not a node-classification task.
- Dev/query listings read training-only attribute context. They do not
  contribute messages to that context or exchange messages with one another.

## Quality experiments, in priority order

### 1. More informative supervision

Use positive pairs that connect the same product across retailers and naming
styles. Include difficult labeled negatives: same brand/flavor but different
volume, pack count, sweetening, or package form. Mix these with representative
easy negatives; an exclusively difficult training distribution can distort
probability calibration.

An unlabeled pair is not automatically a negative. Mine candidates using
training-only information and attach negative labels only where supported.
Keep label provenance and confidence in the experiment artifact. Weighting
labels by confidence would require an extension to the current loss.

Keep product-identity components disjoint across train/dev/test when measuring
generalization to new products. Retailer holdouts answer a different question:
whether learned identity evidence transfers to a new source.

### 2. Better attribute nodes and relation weights

Investigate high-degree attribute hubs: a generic package type can join many
unrelated products. Compare the existing means with learned relation gates,
degree-aware weights, and typed composite nodes such as brand plus flavor.
Composite nodes trade specificity for coverage, so report unseen-value and
sparse-listing performance separately.

Keep missing evidence distinct from explicit contradiction. The current graph
already prevents unknown attribute index zero from becoming a shared context
hub; preserve that semantic distinction in new approaches.

Evaluate per-relation ablations before adding depth: remove or reweight one
relation and measure the change in held-out identity performance. Learned
attention weights alone are not evidence that a relation improves predictions.

### 3. Edge features and selective neighbors

For listing-to-attribute edges, useful proposed features include extraction
confidence, evidence source, explicit versus inferred evidence, and attribute
frequency. These must actually be available at inference time.

For a separate listing-to-listing candidate relation, useful proposed features
include per-field agreement, explicit contradiction, volume/pack differences,
retrieval similarity, and neighbor rank. Candidate similarity is a feature,
not a same-product label.

Compare relation-specific neighbor budgets, mutual nearest neighbors, and
compatibility-aware selection. A same-brand neighbor is not necessarily the
same product. Preserve enough diverse evidence to avoid producing isolated
nodes or excluding difficult true matches.

Start with a weighted version of the existing aggregation. Edge-aware
attention is a later experiment: PyG's `GATv2Conv` supports edge features via
`edge_dim`, but adopting it does not by itself establish a quality gain.
[GATv2Conv documentation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html)

### 4. A more expressive pair decision

Compare the current calibrated cosine with a small symmetric pair scorer
using cosine, absolute embedding differences, elementwise products, and
explicit agreement/contradiction features. Swapping the pair endpoints should
leave the prediction unchanged.

Keep the retrieval embedding and final pair decision separately evaluated.
A stronger pair scorer can improve decisions without improving nearest-neighbor
retrieval. This proposal changes the current monotonic single-cosine contract
and requires coordinated checkpoint, inference, and cascade changes.

### 5. Cluster formation

Evaluate the final clusters, not only individual edges. A single mistaken
bridge can merge two otherwise correct components under connected components.
Compare thresholded connectivity with merge policies that consider conflicting
evidence across the proposed cluster. A missing pair score is not a known
negative, and missing evidence is not a cannot-link constraint.

Only use supported identity labels as must-link/cannot-link supervision.
Thresholds and merge policies are selected on dev, then frozen for test.

## Metrics to prioritize

| Level | Metric | What it answers |
| --- | --- | --- |
| Candidate neighborhood | Known-positive recall at the chosen neighbor budget | Did true matches survive neighbor selection? |
| Node embedding | Recall@K on the actual retrieval population | Are same-product listings close enough to retrieve? |
| Pair decision | Recall at the configured precision target | How many matches can be recovered at acceptable false-merge risk? |
| Pair decision | Precision at the configured recall target | How many false matches accompany the required coverage? |
| Pair ranking | Average precision (called `pr_auc` in this code) | Is ranking improving across thresholds? |
| Calibration | Brier score, log loss, reliability curves | Can a score threshold be interpreted consistently? |
| Final clusters | Pairwise precision/recall after clustering | How many implied same-cluster pairs are correct/missed? |
| Final clusters | B-cubed precision/recall/F1 | How pure and complete is the predicted cluster around each listing? |
| Final clusters | Overmerge and fragmentation rates | Are distinct products merged, or single products split? |
| Diagnostics | Cluster-size distribution, singleton share, largest component | Are bridges creating oversized components, or nodes becoming isolated? |

Define B-cubed per listing as intersection size between its predicted and true
clusters divided by predicted cluster size (precision) or true cluster size
(recall), then average over listings. Report performance by true cluster size
as well so a large singleton population does not conceal difficult cases.

ARI and homogeneity/completeness are useful additional summaries when complete
reference identity clusters exist. Homogeneity captures mixing different
identities; completeness captures splitting one identity. Do not optimize
silhouette or visual cluster separation as a substitute for identity truth.
[Clustering evaluation documentation](https://scikit-learn.org/1.8/modules/clustering.html#clustering-performance-evaluation)

With incomplete labels, report known-positive retrieval recall and metrics on
the labeled evaluation population; do not silently treat all other pairs as
negative or claim complete ground-truth cluster quality.

Slice metrics by retailer pair, brand frequency, attribute completeness,
unseen attributes, node degree, product family, and each contradiction type.
Use product-component bootstrap confidence intervals for pair comparisons;
pair rows sharing identities are not independent observations.

The current trainer selects checkpoints by `dev_pr_auc`. Its report chooses a
threshold with Youden's J, which is not the same as meeting the configured
precision target. For a precision-constrained deployment, compare checkpoints
and choose the dev threshold using that operating objective. A target that
cannot be achieved with meaningful coverage should be reported as unattained.

## What to feed the existing trainer

The inputs are listing records plus labeled pairs. No node-class labels or
precomputed cluster IDs are required for the current loss.

1. Listing records: stable `sku_id`, split, categorical attribute sets, and
   numeric value sets. Examples of attributes are brand, flavor, sweetener,
   package type, and material; numeric descriptors currently include volume
   and pack. The actual fields come from `graph_schema()`.
2. Pair rows: `sku_id1`, `sku_id2`, `label`, `split`, and optionally the existing
   training `example_id`. A positive means same product under the chosen
   identity definition; a negative means different product. Shared attributes
   alone do not supply this label.
3. The existing prepared graph representation: vocabulary, split-local node
   ordering, membership edges, pooling metadata, and local pair indices.
   The current CUDA path consumes the prepared `graph_plan.json` and
   `graph_inputs.npz` representation.

These are immutable inputs. A revised feature set, neighborhood, or supervision
population produces a new artifact; this plan adds no data-checking gates.

### Tensor batch currently consumed

| Value | Shape/type | Meaning |
| --- | --- | --- |
| `support.numeric` | float32 `[N_train, 3 * len(NUMERIC)]` | For each numeric field: log1p minimum, log1p maximum, presence |
| `support.edges[relation]` | two int64 vectors `[E_relation]` | Listing-local index and relation-specific attribute-value index |
| `vocabulary[relation]` | attribute-value vocabulary | Index zero represents unknown; known values start at one |
| `train_pairs` | int64 `[P_train, 2]` | Two local training-listing indices per supervised pair |
| `train_labels` | float32 `[P_train]` | Same-product target, zero or one |
| `dev_batch` | same node/edge structure, dev listings | Query features; reads the updated training-only context |
| `dev_pairs`, dev labels | `[P_dev, 2]`, `[P_dev]` | Dev-local pair indices and evaluation targets |

For example, pairs `[[0, 1], [0, 2]]` with labels `[1, 0]` mean that listings
0 and 1 are the same product and listings 0 and 2 are different. This does not
insert either pair as a message-passing edge. Membership edges and supervised
pairs have separate purposes and separate index spaces.

At each epoch, the model builds training attribute context, encodes the support
listings, scores all training pairs, and makes one update. Dev evaluation then
rebuilds context using updated model weights and encodes dev listings against
that context. Reusing detached context across training updates would change
the objective, because context depends on trainable parameters.

The active `gnn_only` config rejects a text embedding cache. Adding semantic
node features would therefore be a deliberate new model/config contract,
not an extra tensor that the existing trainer already accepts.

### If we later introduce sampled graph batches

Each batch would start with supervised seed pairs and include:

- The unique endpoints of those pairs and their local node features.
- The typed attribute nodes needed by those endpoints, plus sampled
  training-support neighbors that produce their attribute context.
- Typed membership edges, with any proposed edge features aligned to edges.
- A mapping from the original endpoints to batch-local indices, pair labels,
  and optional label weights if that loss extension is adopted.

Sample enough of the listing-to-attribute-to-listing computation to support
both endpoints. A random batch of pair rows with no supporting neighborhood
is not an equivalent batch for this model. Sampling support changes the
attribute means and therefore the learning objective; evaluate it as a graph
quality experiment, not just a batching implementation detail.

Keep dev/test labels out of graph construction and neighbor mining. To retain
the current inductive behavior, query nodes read training support without
contributing to it. For training seed pairs, including their own membership
edges follows the existing architecture; inserting supervised positive pair
edges would be a separate change with potential label leakage.

## Experiment sequence

Hold the evaluation populations and identity definition fixed. Compare the
current graph against: improved supervised-pair composition, relation gates,
edge confidence features, selective listing neighbors, and an expressive
pair scorer, one change at a time. Then combine only improvements supported
by dev results and evaluate the selected approach once on held-out test.

Use both pair-level and final-cluster metrics for selection. If candidate
recall deteriorates, fix neighbor selection before tuning the scorer; if
ranking improves but clusters overmerge, inspect calibration and merge policy.
