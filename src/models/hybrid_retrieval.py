import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict


def _minmax(x):
    x = np.asarray(x, dtype=float)
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 1e-12 else np.zeros_like(x)


class _CachingEmbedder:
    """Wraps an embedding model; embed_query results are cached by text (the evaluation metrics call
    embed_query for the same few hundred items thousands of times)."""

    def __init__(self, base):
        self._base, self._q = base, {}

    def embed_query(self, text):
        if text not in self._q:
            self._q[text] = (self._base.embed_query(text) if hasattr(self._base, "embed_query")
                             else self._base.embed_documents([text])[0])
        return self._q[text]

    def embed_documents(self, texts):
        return self._base.embed_documents(texts)

    def __getattr__(self, name):
        return getattr(self._base, name)


class RetrievalEngine:
    def __init__(self, chroma_db, embedding_model, embedding_lookup=None):
        self.chroma_db = chroma_db
        self.embedding_model = _CachingEmbedder(embedding_model)
        self._cache = {}                      # text -> embedding (shared by all stages/variants)

    def embed_texts(self, texts):
        missing = list({t for t in texts if t not in self._cache})
        if missing:
            for t, v in zip(missing, self.embedding_model.embed_documents(missing)):
                self._cache[t] = np.asarray(v, dtype=np.float32)
        return np.vstack([self._cache[t] for t in texts])

    def hybrid_retrieve(self, user_recs: List[Dict], query: str, cf_weight: float = 0.5,
                        normalize: bool = True) -> List[Dict]:
        """hybrid = w * CF + (1-w) * semantic. With normalize=True both components are
        min-max scaled within the user's candidate pool, so w is the true relative weight
        (raw LightFM scores are unbounded; cosine similarity is not)."""
        if not user_recs:
            return []
        q = self.embed_texts([query])
        items = self.embed_texts([r["augmented_text"] for r in user_recs])
        sem = cosine_similarity(q, items)[0]
        cf = np.array([float(r.get("cf_score", r.get("score", 0.0))) for r in user_recs])
        cf_n, sem_n = (_minmax(cf), _minmax(sem)) if normalize else (cf, sem)
        hybrid = cf_weight * cf_n + (1 - cf_weight) * sem_n
        out = [{**r,
                "semantic_score": float(s), "cf_score": float(c),
                "cf_score_norm": float(cn), "semantic_score_norm": float(sn),
                "hybrid_score": float(h)}
               for r, s, c, cn, sn, h in zip(user_recs, sem, cf, cf_n, sem_n, hybrid)]
        return sorted(out, key=lambda x: x["hybrid_score"], reverse=True)

    def mmr_diversify(self, items: List[Dict], query: str, lambda_param: float = 0.7,
                      top_k: int = 50, relevance_key="hybrid_score") -> List[Dict]:
        """MMR: argmax  lambda * relevance(i) - (1-lambda) * max_{j in S} sim(i, j).
        relevance_key='hybrid_score' (default) keeps the CF signal in the relevance term;
        relevance_key=None reproduces the earlier behaviour (query similarity only)."""
        if not items:
            return []
        E = self.embed_texts([i["augmented_text"] for i in items])
        if relevance_key and relevance_key in items[0]:
            rel = _minmax([i[relevance_key] for i in items])
        else:
            rel = cosine_similarity(self.embed_texts([query]), E)[0]
        ds = cosine_similarity(E)

        selected, scores, remaining = [], {}, list(range(len(items)))
        while remaining and len(selected) < min(top_k, len(items)):
            best, best_s = None, -np.inf
            for idx in remaining:
                if not selected:
                    s = lambda_param * rel[idx]
                else:
                    s = lambda_param * rel[idx] - (1 - lambda_param) * max(ds[idx][j] for j in selected)
                if s > best_s:
                    best, best_s = idx, s
            selected.append(best)
            scores[best] = float(best_s)
            remaining.remove(best)
        return [{**items[i], "mmr_score": scores[i], "diversity_score": scores[i]} for i in selected]