"""Local exact ranks and existing HNSW comparisons on a fixed candidate catalog."""
import tempfile
from pathlib import Path
import numpy as np
from training.hnsw_index import PersistentHnswIndex


class RetrievalComparison:
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
        candidate_order = np.arange(len(self.vectors))
        for source, requested in targets.items():
            scores = queries[source] @ self.vectors.T
            scores[self.endpoints[source]] = -np.inf
            for n, side, target in requested:
                target_index = self.endpoints[target]
                value = scores[target_index]
                rank = 1+np.count_nonzero(scores > value)+np.count_nonzero(
                    (scores == value)&(candidate_order < target_index))
                pair_ranks[n][side] = int(rank)
        return pair_ranks

    def ann_hits(self, queries):
        labels,_ = self.index.query(queries,top_k=max(self.cfg.retrieval_ks)+1)
        neighbors = [[int(n) for n in row if n != self.endpoints[i]] for i,row in enumerate(labels)]
        return [{str(k):[self.endpoints[b] in neighbors[a][:k],self.endpoints[a] in neighbors[b][:k]]
                 for k in self.cfg.retrieval_ks} for a,b in self.pairs]

    def close(self):
        self.tmp.cleanup()
