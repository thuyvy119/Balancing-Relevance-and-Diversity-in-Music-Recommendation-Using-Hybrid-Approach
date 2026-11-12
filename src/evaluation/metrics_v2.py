import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def _rank_metrics(rel, n_ideal=None):
    # AP, reciprocal rank and binary-gain NDCG for a 0/1 relevance list
    '''n_ideal: number of relevant items the ideal list could contain (e.g. relevant items in the candidate
    pool, capped at k) 
    Without it, IDCG is built from the list's own relevant items, so NDCG only measures
    ordering inside the list and ignores how many relevant items were retrieved'''
    rel = list(rel)
    n_rel = sum(rel)
    n_ideal = n_rel if n_ideal is None else max(n_ideal, n_rel)
    ap, hits = 0.0, 0
    for i, r in enumerate(rel):
        if r:
            hits += 1
            ap += hits / (i + 1)
    ap = ap / n_rel if n_rel else 0.0
    rr = next((1.0 / (i + 1) for i, r in enumerate(rel) if r), 0.0)
    dcg = sum(r / np.log2(i + 2) for i, r in enumerate(rel))
    idcg = sum(1 / np.log2(i + 2) for i in range(n_ideal))
    return ap, rr, (dcg / idcg if idcg else 0.0)


def ground_truth_metrics(recs, gt, k=10):
    # Accuracy against the user's held-out positive items (independent of any embedding model)
    hits = [1 if r["iid"] in gt else 0 for r in recs[:k]]
    n_hit = sum(hits)
    dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(gt), k)))
    return {
        f"GT_Precision@{k}": n_hit / k,
        f"GT_Recall@{k}": n_hit / len(gt),
        f"GT_HitRate@{k}": float(n_hit > 0),
        f"GT_NDCG@{k}": dcg / idcg if idcg else 0.0,
    }


def pool_recall(pool, gt):
    # Share of the user's ground-truth items inside the CF candidate pool (ceiling for all variants)
    return len({r["iid"] for r in pool} & gt) / len(gt)


def semantic_metrics(recs, query, retriever, thresholds=(0.25, 0.35, 0.45), k=10):
    # Query-relevance by embedding similarity at several thresholds (threshold sensitivity)
    # MAP / MRR / NDCG use the first threshold (0.25, the value used in the original paper)
    recs = recs[:k]
    texts = [f"{r.get('title', '')} {r.get('categories', '')} {r.get('description', '')}" for r in recs]
    sims = cosine_similarity(retriever.embed_texts([query]), retriever.embed_texts(texts))[0]
    out = {"Semantic_MeanSim": float(sims.mean())}
    for t in thresholds:
        out[f"Semantic_P@{k}_t{t}"] = float((sims >= t).mean())
    ap, rr, ndcg = _rank_metrics((sims >= thresholds[0]).astype(int))
    out.update({f"Semantic_MAP@{k}": ap, f"Semantic_MRR@{k}": rr, f"Semantic_NDCG@{k}": ndcg})
    return out


class EvalEmbedder:
    # Embedding model used ONLY for evaluation, different from the retrieval model
    # so relevance is not judged by the model that did the ranking

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        from langchain_huggingface import HuggingFaceEmbeddings
        self.model = HuggingFaceEmbeddings(model_name=model_name)
        self._cache = {}

    def embed_texts(self, texts):
        missing = list({t for t in texts if t not in self._cache})
        if missing:
            for t, v in zip(missing, self.model.embed_documents(missing)):
                self._cache[t] = np.asarray(v, dtype=np.float32)
        return np.vstack([self._cache[t] for t in texts])


def genre_metrics(recs, keywords, k=10, pool=None):
    # Embedding-free relevance: an item is relevant if one of its Amazon category labels
    # contains a keyword that defines the query's genre (e.g. 'progressive')
    kws = [w.lower() for w in keywords]
    is_rel = lambda r: int(any(w in str(r.get("categories", "")).lower() for w in kws))
    rel = [is_rel(r) for r in recs[:k]]
    # pool-normalised NDCG: ideal list = the relevant items available in the user's candidate pool
    n_ideal = min(sum(is_rel(p) for p in pool), k) if pool else None
    ap, rr, ndcg = _rank_metrics(rel, n_ideal=n_ideal)
    return {f"Genre_P@{k}": sum(rel) / max(len(rel), 1), f"Genre_MAP@{k}": ap,
            f"Genre_MRR@{k}": rr, f"Genre_NDCG@{k}": ndcg}