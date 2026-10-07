"""Local exact ranks and existing HNSW comparisons on a fixed candidate catalog."""
import tempfile
from pathlib import Path
import numpy as np
from training.hnsw_index import PersistentHnswIndex


class RetrievalComparison:
    """Exact ranks and the HNSW comparison over one candidate catalog.

    SCORING KERNEL (measured, artifacts/abl_opt/micro/bench_ranks_sweep.json).
    One matrix product per BLOCK of query sources is 6-59x faster than one GEMV
    per source, because a per-source GEMV re-reads the whole catalog for every
    source (358 sources x 10,000 x 384 float32 = 5.5 GB of traffic to produce
    358 x 10,000 scores) while the block product streams the catalog once. It is
    NOT used here: a multi-row product does not accumulate in the same order as
    the GEMV it replaces, so its scores differ in the last bits (max |delta|
    1.4e-07 on unit vectors, 18,352 of 20,000 scores off by one ulp at 20,000
    candidates). A rank counts strict comparisons against one value, so that
    delta flips a rank whenever another candidate sits within it of that value —
    measured: 1 rank of 358 sources on the synthetic 20k grid, while the 500
    (smoke) and 10,000 (real cohort) fixtures came out identical. Ranks are
    report content, and this lane's contract is byte-identical output, so the
    exact per-source loop is kept and the block path was reverted (r19). A
    margin-guarded hybrid was tried first: it cannot certify exactness in
    practice, because with ~10,000 candidates the nearest other score is
    typically ~4e-05 away, far below the ~2.3e-05 bound on the summation
    difference, so nearly every source fell back to the GEMV and the hybrid ran
    ​0.70x (slower than not batching at all).

    The remaining rewrites are bit-exact: `catalog @ queries[source]` produces
    identical scores to the original `queries[source] @ catalog.T` (asserted),
    and the tie contest counts `scores[:target_index] == value` instead of
    building `(scores == value) & (candidate_order < target_index)`.
    """

    def __init__(self, ids, vectors, request, request_path, cfg):
        self.ids, self.vectors, self.request, self.cfg = ids, vectors, request, cfg
        if vectors.ndim != 2 or len(ids) != len(vectors) or len(set(ids)) != len(ids):
            raise ValueError('candidate catalog alignment mismatch')
        if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors,axis=-1),1,atol=1e-4):
            raise ValueError('candidate vectors must be finite and normalized')
        lookup = {key:n for n,key in enumerate(ids)}
        self.endpoints = [lookup[key] for key in request['ids']]
        query_lookup = {key:n for n,key in enumerate(request['ids'])}
        self.pairs = [(query_lookup[p['sku_id1']],query_lookup[p['sku_id2']]) for p in request['pairs']]
        self.tmp = tempfile.TemporaryDirectory(dir=request_path.parent)
        from model_tracks.ablation import resolve
        marker = resolve(request['checkpoint']) if request.get('checkpoint') else request_path
        self.index = PersistentHnswIndex(Path(self.tmp.name)/'index',M=cfg.hnsw_m,
            ef_construction=cfg.hnsw_ef_construction,ef_search=cfg.hnsw_ef_search)
        self.index.build(vectors,ids,checkpoint=marker,model_name=request['track'])

    def ranks(self, queries):
        # Only compare queries against candidates; never construct catalog².
        pair_ranks = [[None, None] for _ in self.pairs]
        targets = {}
        for n,(a,b) in enumerate(self.pairs):
            for side,source,target in ((0,a,b),(1,b,a)):
                targets.setdefault(source, []).append((n, side, target))
        endpoints = self.endpoints
        catalog = self.vectors
        for source, requested in targets.items():
            # Exact per-source scores: see the class docstring for why the
            # faster block product cannot be used behind a rank.
            scores = catalog @ queries[source]
            scores[endpoints[source]] = -np.inf
            for n, side, target in requested:
                target_index = endpoints[target]
                value = scores[target_index]
                # Ties are won by the lower catalog index: slicing to the
                # target is that same contest with one temporary instead of two
                # (count((scores == value) & (candidate_order < target_index))).
                pair_ranks[n][side] = int(1+np.count_nonzero(scores > value)+np.count_nonzero(
                    scores[:target_index] == value))
        return pair_ranks

    def ann_hits(self, queries):
        labels,_ = self.index.query(queries,top_k=max(self.cfg.retrieval_ks)+1)
        neighbors = [[int(n) for n in row if n != self.endpoints[i]] for i,row in enumerate(labels)]
        return [{str(k):[self.endpoints[b] in neighbors[a][:k],self.endpoints[a] in neighbors[b][:k]]
                 for k in self.cfg.retrieval_ks} for a,b in self.pairs]

    def close(self):
        self.tmp.cleanup()
