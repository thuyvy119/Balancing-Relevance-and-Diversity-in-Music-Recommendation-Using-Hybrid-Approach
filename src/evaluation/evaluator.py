import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import wilcoxon

from src.services.recommender_service import recommend_all_variants, get_user_pool, is_pool_variant, POOL_VARIANTS
from src.evaluation.metrics_v2 import ground_truth_metrics, pool_recall, semantic_metrics, genre_metrics
from src.evaluation.diversity_metrics import calculate_diversity_metrics
from src.utils.user_utils import build_ground_truth, get_qualified_users

ALL_VARIANTS = ["random", "popularity"] + POOL_VARIANTS
ALL_VARIANTS_NO_LLM = [v for v in ALL_VARIANTS if v != "full"]
ALPHA_SWEEP = [f"hybrid_a{a}" for a in (0.25, 0.75)]            # 0 = semantic_only, 1 = cf_only, 0.5 = hybrid
LAMBDA_SWEEP = [f"hybrid_mmr_l{l}" for l in (0.4, 0.5, 0.6, 0.8, 0.9)] + ["hybrid_mmr_qsim"]

KEY_METRICS = ["GT_Recall@10", "GT_NDCG@10", "Genre_P@10", "Genre_NDCG@10",
               "Semantic_P@10_t0.25", "Semantic_P@10_t0.35", "Semantic_NDCG@10",
               "Indep_P@10_t0.25", "Indep_P@10_t0.35", "Indep_MeanSim",
               "IntraListSimilarity", "Entropy", "Serendipity"]
TEST_METRICS = KEY_METRICS


def _baseline_helpers(train_data, k, rng):
    tr = train_data[train_data["label"] > 0].drop_duplicates(["uid", "iid"])
    cols = [c for c in ["iid", "title", "categories", "description"] if c in tr.columns]
    catalog = tr.drop_duplicates("iid")[cols].to_dict("records")
    by_iid = {r["iid"]: r for r in catalog}
    pop_list = [by_iid[i] for i in tr["iid"].value_counts().index[:2000]]
    seen = tr.groupby("uid")["iid"].apply(set).to_dict()

    def popularity(uid):
        s = seen.get(uid, set())
        return [r for r in pop_list if r["iid"] not in s][:k]

    def random_items(uid):
        s = seen.get(uid, set())
        out = []
        while len(out) < k:
            r = rng.choice(catalog)
            if r["iid"] not in s and r not in out:
                out.append(r)
        return out

    return popularity, random_items


def evaluate_variants(pipeline_objects, train_data, test_data, query, variants=ALL_VARIANTS,
                      sample_size=None, seed=42, k=10, thresholds=(0.25, 0.35, 0.45),
                      user_split=None, cf_weight=0.5, mmr_lambda=0.7,
                      genre_keywords=None, eval_embedder=None, llm_sample_size=None, light=False):
    """genre_keywords: list of category keywords defining relevance for this query (embedding-free judge).
    eval_embedder: independent embedding model for evaluation (see metrics_v2.EvalEmbedder).
    llm_sample_size: run the (slow) 'full' variant only for the first N sampled users.
    light: only ground-truth metrics (fast; use for large-user accuracy runs)."""
    rng = random.Random(seed)
    gt = build_ground_truth(train_data, test_data, warm_only=True)
    users = get_qualified_users(gt, set(pipeline_objects.df["user_id"]))
    print(f"Qualified users (>=1 warm test positive and a CF pool): {len(users)}")
    if user_split in ("tune", "eval"):      # disjoint halves: tune alpha/lambda on one, report on the other
        shuffled = users[:]
        random.Random(0).shuffle(shuffled)
        half = len(shuffled) // 2
        users = sorted(shuffled[:half] if user_split == "tune" else shuffled[half:])
        print(f"user_split={user_split}: {len(users)} users")
    random.Random(seed).shuffle(users)       # fixed order -> smaller samples are prefixes of larger ones
    if sample_size:
        users = users[:sample_size]

    popularity, random_items = _baseline_helpers(train_data, k, rng)
    pool_vars = [v for v in variants if is_pool_variant(v)]
    retr = pipeline_objects.retriever
    rows, recs_log = [], []

    for n, u in enumerate(tqdm(users, desc=f"query: {query[:30]}")):
        uv = pool_vars
        if llm_sample_size is not None and n >= llm_sample_size:
            uv = [v for v in pool_vars if v != "full"]
        by_variant, meta = {}, {}
        if uv:
            by_variant, meta = recommend_all_variants(pipeline_objects, u, query, uv, k=k,
                                                      cf_weight=cf_weight, mmr_lambda=mmr_lambda)
        if "popularity" in variants:
            by_variant["popularity"] = popularity(u)
        if "random" in variants:
            by_variant["random"] = random_items(u)

        pr = pool_recall(get_user_pool(pipeline_objects, u), gt[u])
        for v, recs in by_variant.items():
            if not recs:
                continue
            row = {"query": query, "variant": v, "user_id": u, "n_recs": len(recs),
                   "PoolRecall@50": pr,
                   "llm_fallback": meta.get("llm_fallback", np.nan) if v == "full" else np.nan}
            row.update(ground_truth_metrics(recs, gt[u], k))
            if light:
                rows.append(row)
                recs_log.append({"user_id": u, "variant": v, "query": query, "recommendations": recs})
                continue
            row.update(semantic_metrics(recs, query, retr, thresholds, k))
            if eval_embedder is not None:
                ind = semantic_metrics(recs, query, eval_embedder, thresholds, k)
                row.update({key.replace("Semantic_", "Indep_"): val for key, val in ind.items()})
            if genre_keywords:
                row.update(genre_metrics(recs, genre_keywords, k, pool=get_user_pool(pipeline_objects, u)))
            row.update(calculate_diversity_metrics(recs, pipeline_objects))
            rows.append(row)
            recs_log.append({"user_id": u, "variant": v, "query": query, "recommendations": recs})

    catalog_size = int(train_data.loc[train_data["label"] > 0, "iid"].nunique())
    return {"per_user": pd.DataFrame(rows), "recommendations": recs_log, "catalog_size": catalog_size}


def aggregate_coverage(recs_log, catalog_size, max_pairs_users=200, seed=0):
    """Across users: distinct items recommended (coverage) and mean pairwise Jaccard overlap of users'
    lists (1 = everybody gets the same list = no personalization)."""
    rng = random.Random(seed)
    df = pd.DataFrame([{"query": r["query"], "variant": r["variant"],
                        "ids": [x["iid"] for x in r["recommendations"]]} for r in recs_log])
    rows = []
    for (q, v), g in df.groupby(["query", "variant"]):
        sets = [set(x) for x in g["ids"]]
        uniq = set().union(*sets)
        sub = rng.sample(sets, min(max_pairs_users, len(sets)))
        jac = [len(a & b) / len(a | b) for i, a in enumerate(sub) for b in sub[i + 1:] if a | b]
        rows.append({"query": q, "variant": v, "n_users": len(sets), "unique_items": len(uniq),
                     "catalog_coverage": len(uniq) / catalog_size,
                     "mean_pairwise_jaccard": float(np.mean(jac)) if jac else np.nan})
    return pd.DataFrame(rows)


def pool_queries(per_user):
    """Add rows with query='ALL': each user's metrics averaged over queries (same users in all queries)."""
    if per_user["query"].nunique() < 2:
        return per_user
    num = list(per_user.select_dtypes("number").columns)
    pooled = per_user.groupby(["variant", "user_id"])[num].mean().reset_index()
    pooled["query"] = "ALL"
    return pd.concat([per_user, pooled], ignore_index=True)


def summarize(per_user):
    num = [c for c in per_user.select_dtypes("number").columns if c != "n_recs"]
    g = per_user.groupby(["query", "variant"])[num]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std"))
    out["n_users"] = per_user.groupby(["query", "variant"]).size()
    return out.reset_index()


def bootstrap_ci(per_user, metrics=KEY_METRICS, n_boot=1000, seed=0):
    """95% bootstrap CI of the mean over users."""
    rng = np.random.default_rng(seed)
    rows = []
    for (q, v), g in per_user.groupby(["query", "variant"]):
        for m in metrics:
            if m not in g.columns:
                continue
            x = g[m].dropna().values
            if len(x) < 2:
                continue
            means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
            rows.append({"query": q, "variant": v, "metric": m, "n": len(x), "mean": x.mean(),
                         "ci_low": np.percentile(means, 2.5), "ci_high": np.percentile(means, 97.5)})
    return pd.DataFrame(rows)


def paired_tests(per_user, reference="full", metrics=TEST_METRICS):
    """Paired Wilcoxon signed-rank test of `reference` vs every other variant (same users), Holm-corrected."""
    rows = []
    for q, dq in per_user.groupby("query"):
        for m in metrics:
            if m not in dq.columns:
                continue
            piv = dq.pivot_table(index="user_id", columns="variant", values=m)
            if reference not in piv.columns:
                continue
            for v in piv.columns:
                if v == reference:
                    continue
                pair = piv[[reference, v]].dropna()
                diff = pair[reference] - pair[v]
                p = np.nan
                if len(pair) >= 10 and not (diff == 0).all():
                    p = wilcoxon(pair[reference], pair[v]).pvalue
                rows.append({"query": q, "metric": m, "reference": reference, "vs": v, "n": len(pair),
                             "ref_mean": pair[reference].mean(), "vs_mean": pair[v].mean(),
                             "mean_diff": diff.mean(), "p_value": p})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    valid = t["p_value"].notna()
    p = t.loc[valid, "p_value"].values
    order = np.argsort(p)
    holm = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[idx]))
        holm[idx] = running
    t.loc[valid, "p_holm"] = holm
    return t