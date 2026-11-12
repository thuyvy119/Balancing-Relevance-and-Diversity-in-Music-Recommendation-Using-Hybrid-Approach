import re

_AUG_CACHE = {}          # iid -> augmented text 
_POOL_CACHE = {}         # id(df) -> {user_id: recommendations}

POOL_VARIANTS = ["cf_only", "semantic_only", "hybrid", "hybrid_mmr", "full"]
_SWEEP = re.compile(r"hybrid_a[\d.]+|hybrid_mmr_l[\d.]+|hybrid_mmr_qsim")


def is_pool_variant(v):
    return v in POOL_VARIANTS or bool(_SWEEP.fullmatch(v))


def _augment(chroma_db, rec):
    iid = rec["iid"]
    if iid in _AUG_CACHE:
        return _AUG_CACHE[iid]
    base = rec.get("combined_text", "")
    try:
        chunks = chroma_db.similarity_search(query=base, k=3, filter={"iid": iid})
        text = base + "\n" + " ".join(c.page_content for c in chunks)
    except Exception as e:
        print(f"Augmentation error for item {iid}: {e}")
        text = base
    _AUG_CACHE[iid] = text
    return text


def get_user_pool(pipeline_objects, user_id):
    key = id(pipeline_objects.df)
    if key not in _POOL_CACHE:
        _POOL_CACHE[key] = dict(zip(pipeline_objects.df["user_id"], pipeline_objects.df["recommendations"]))
    return _POOL_CACHE[key].get(user_id, [])


def recommend_all_variants(pipeline_objects, user_id, query, variants=POOL_VARIANTS,
                           k=10, cf_weight=0.5, mmr_lambda=0.7, mmr_top_k=20):
    pool = get_user_pool(pipeline_objects, user_id)
    out, meta = {}, {}
    if not pool:
        return out, meta
    retr = pipeline_objects.retriever

    if "cf_only" in variants:
        out["cf_only"] = sorted(pool, key=lambda r: -float(r.get("score", 0)))[:k]

    augmented = [{**r, "augmented_text": _augment(pipeline_objects.chroma_db, r)} for r in pool]

    if "semantic_only" in variants:
        out["semantic_only"] = retr.hybrid_retrieve(augmented, query, cf_weight=0.0)[:k]

    hybrid = retr.hybrid_retrieve(augmented, query, cf_weight=cf_weight)
    if "hybrid" in variants:
        out["hybrid"] = hybrid[:k]

    if "hybrid_mmr" in variants or "full" in variants:
        diversified = retr.mmr_diversify(hybrid, query, lambda_param=mmr_lambda, top_k=mmr_top_k)
        out["hybrid_mmr"] = diversified[:k]
        if "full" in variants:
            out["full"] = pipeline_objects.llm_reranker.rerank(query=query, candidates=diversified)[:k]
            meta["llm_fallback"] = float(pipeline_objects.llm_reranker.last_fallback)
    for v in variants:
        m = re.fullmatch(r"hybrid_a([\d.]+)", v)
        if m:
            out[v] = retr.hybrid_retrieve(augmented, query, cf_weight=float(m.group(1)))[:k]
        m = re.fullmatch(r"hybrid_mmr_l([\d.]+)", v)
        if m:
            out[v] = retr.mmr_diversify(hybrid, query, lambda_param=float(m.group(1)), top_k=mmr_top_k)[:k]
        if v == "hybrid_mmr_qsim":
            out[v] = retr.mmr_diversify(hybrid, query, lambda_param=mmr_lambda, top_k=mmr_top_k,
                                        relevance_key=None)[:k]
    return out, meta


def recommend_for_user(pipeline_objects, user_id, query, debug=False, use_llm=True, variant=None):
    variant = variant or ("full" if use_llm else "hybrid_mmr")
    out, _ = recommend_all_variants(pipeline_objects, user_id, query, variants=[variant])
    recs = out.get(variant, [])
    if debug:
        for i, it in enumerate(recs, 1):
            print(f"{i}. {it.get('title','No Title')} | iid={it.get('iid')} | "
                  f"hybrid={it.get('hybrid_score',0):.4f} | sem={it.get('semantic_score',0):.4f}")
    return recs