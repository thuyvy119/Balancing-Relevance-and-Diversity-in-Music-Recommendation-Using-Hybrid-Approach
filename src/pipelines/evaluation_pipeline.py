import os
import hashlib
import pickle
import pandas as pd
from src.services.pipeline_loader import load_pipeline
from src.evaluation.evaluator import (evaluate_variants, summarize, paired_tests, bootstrap_ci,
                                      pool_queries, aggregate_coverage, ALL_VARIANTS)
from src.evaluation.metrics_v2 import EvalEmbedder
from src.utils.io import save_recommendations

SHOW = ["GT_Recall@10", "GT_NDCG@10", "PoolRecall@50", "Genre_P@10", "Genre_NDCG@10",
        "Semantic_P@10_t0.25", "Semantic_P@10_t0.35", "Indep_P@10_t0.35", "Indep_MeanSim",
        "IntraListSimilarity", "Entropy", "Serendipity", "llm_fallback"]


def evaluation_pipeline(train_data, test_data, query, sample_size=None, variants=None, seed=42,
                        user_split=None, cf_weight=0.5, mmr_lambda=0.7, llm_sample_size=None,
                        eval_embedding_model=None, light=False, out_dir="results/ablation"):
    """query: str | list[str] | dict{query: [genre keywords]}  (keywords give an embedding-free relevance judge).
    All variants are evaluated on the SAME users. 'full' (LLM) can be limited to the first llm_sample_size users."""
    os.makedirs(out_dir, exist_ok=True)
    kw = query if isinstance(query, dict) else {}
    queries = [query] if isinstance(query, str) else list(query)
    variants = variants or ALL_VARIANTS

    print("\nLoading pipeline...")
    po = load_pipeline()
    eval_emb = EvalEmbedder(eval_embedding_model) if eval_embedding_model else None

    frames, rec_logs, catalog_size = [], [], None
    part_dir = f"{out_dir}/partial"            # one checkpoint per query: a crash loses at most one query
    os.makedirs(part_dir, exist_ok=True)
    for q in queries:
        cfg = repr((q, tuple(variants), sample_size, llm_sample_size, seed, user_split,
                    cf_weight, mmr_lambda, light, eval_embedding_model, kw.get(q)))
        ckpt = f"{part_dir}/{hashlib.md5(cfg.encode()).hexdigest()[:12]}.pkl"
        if os.path.exists(ckpt):
            print(f"Resuming from checkpoint for query: {q}")
            with open(ckpt, "rb") as fh:
                res = pickle.load(fh)
        else:
            res = evaluate_variants(po, train_data, test_data, q, variants=variants, sample_size=sample_size,
                                    seed=seed, user_split=user_split, cf_weight=cf_weight, mmr_lambda=mmr_lambda,
                                    genre_keywords=kw.get(q), eval_embedder=eval_emb,
                                    llm_sample_size=llm_sample_size, light=light)
            with open(ckpt, "wb") as fh:
                pickle.dump(res, fh)
        frames.append(res["per_user"])
        rec_logs += res["recommendations"]
        catalog_size = res["catalog_size"]

    per_user = pool_queries(pd.concat(frames, ignore_index=True))
    summary = summarize(per_user)
    ci = bootstrap_ci(per_user)
    ref = "full" if "full" in variants else variants[-1]
    tests = paired_tests(per_user, reference=ref)
    coverage = aggregate_coverage(rec_logs, catalog_size)

    per_user.to_csv(f"{out_dir}/per_user_metrics.csv", index=False)
    summary.to_csv(f"{out_dir}/summary_by_variant.csv", index=False)
    ci.to_csv(f"{out_dir}/bootstrap_ci.csv", index=False)
    tests.to_csv(f"{out_dir}/paired_tests.csv", index=False)
    coverage.to_csv(f"{out_dir}/aggregate_coverage.csv", index=False)
    for v in variants:
        sub = [r for r in rec_logs if r["variant"] == v]
        if sub:
            save_recommendations(sub, v, save_prefix=v, save_dir=f"{out_dir}/recommendations")

    print(f"\nLLM calls: {getattr(po.llm_reranker, 'n_calls', 0)}, unusable outputs (fell back to MMR order): "
          f"{getattr(po.llm_reranker, 'n_fallback', 0)}")
    show = ["query", "variant"] + [f"{c}_mean" for c in SHOW if f"{c}_mean" in summary.columns] + ["n_users"]
    print("\nSUMMARY (mean over users; SD and 95% CI are in the CSV files)")
    print(summary[show].round(4).to_string(index=False))
    print(f"\nCOVERAGE ACROSS USERS (catalog size = {catalog_size})")
    print(coverage.round(4).to_string(index=False))
    return {"per_user": per_user, "summary": summary, "ci": ci, "tests": tests, "coverage": coverage}