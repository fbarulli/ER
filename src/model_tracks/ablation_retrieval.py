"""Local exact ranks and existing HNSW comparisons on a fixed candidate catalog."""
import tempfile
from pathlib import Path
import numpy as np
from training.hnsw_index import PersistentHnswIndex


class RetrievalComparison:
    # One BLAS call scores a whole BLOCK of query sources at once. A per-source
    # GEMV re-reads the entire catalog for every source (358 sources x 10,000 x
    # 384 float32 = 5.5 GB of traffic to produce 358 x 10,000 scores on the 10k
    # cohort), while the block GEMM streams the catalog once and writes every
    # source's scores. Measured (artifacts/abl_opt/micro/bench_ranks.py, 4 CPU
    # cores, one source per row):
    #   candidates  sources   per-source   block GEMM
    #          500       33      0.98 ms      2.94 ms   <- small catalogs: GEMV
    #         1000       33      2.47 ms      4.99 ms
    #         2000       64     42.39 ms     11.71 ms   <- block GEMM wins
    #        10000      358   1339.40 ms     44.00 ms
    # Small matrices pay a fixed multi-threaded bring-up in the GEMM path that
    # outweighs streaming a catalog that already fits in cache, so dispatch on
    # size instead of always batching.
    SCORE_BLOCK = 256
    BATCH_MIN_CANDIDATES = 1500
    BATCH_MIN_SOURCES = 8

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

    def _rank(self, pair_ranks, sources, targets, queries, endpoints, block_scores=None):
        """Rank the requested side(s) of every source in one block.

        Called once per block: with `block_scores` absent the block's scores are
        one GEMV per source (small catalogs), otherwise they arrive from a
        single matrix product for the whole block (large catalogs).
        """
        catalog = self.vectors
        for row, source in enumerate(sources):
            scores = (catalog @ queries[source]) if block_scores is None else block_scores[row]
            scores[endpoints[source]] = -np.inf
            for n, side, target in targets[source]:
                target_index = endpoints[target]
                value = scores[target_index]
                # Ties are won by the lower catalog index: slicing to the
                # target is that same contest with one temporary instead of two.
                pair_ranks[n][side] = int(1+np.count_nonzero(scores > value)+np.count_nonzero(
                    scores[:target_index] == value))

    def ranks(self, queries):
        # Only compare queries against candidates; never construct catalog².
        pair_ranks = [[None, None] for _ in self.pairs]
        targets = {}
        for n,(a,b) in enumerate(self.pairs):
            for side,source,target in ((0,a,b),(1,b,a)):
                targets.setdefault(source, []).append((n, side, target))
        endpoints = self.endpoints
        catalog = self.vectors
        sources = list(targets)
        if len(catalog) >= self.BATCH_MIN_CANDIDATES and len(sources) >= self.BATCH_MIN_SOURCES:
            for start in range(0, len(sources), self.SCORE_BLOCK):
                block = sources[start:start+self.SCORE_BLOCK]
                self._rank(pair_ranks, block, targets, queries, endpoints,
                           queries[block] @ catalog.T)
        else:
            self._rank(pair_ranks, sources, targets, queries, endpoints)
        return pair_ranks

    def ann_hits(self, queries):
        labels,_ = self.index.query(queries,top_k=max(self.cfg.retrieval_ks)+1)
        neighbors = [[int(n) for n in row if n != self.endpoints[i]] for i,row in enumerate(labels)]
        return [{str(k):[self.endpoints[b] in neighbors[a][:k],self.endpoints[a] in neighbors[b][:k]]
                 for k in self.cfg.retrieval_ks} for a,b in self.pairs]

    def close(self):
        self.tmp.cleanup()
