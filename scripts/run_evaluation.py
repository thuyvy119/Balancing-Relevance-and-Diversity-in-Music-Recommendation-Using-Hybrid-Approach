import os, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
from src.pipelines.evaluation_pipeline import evaluation_pipeline
from src.evaluation.evaluator import ALL_VARIANTS, ALL_VARIANTS_NO_LLM, ALPHA_SWEEP, LAMBDA_SWEEP

QUERIES = {
    "classic rock albums with great guitar solos":          ["classic rock", "guitar"],
    "progressive rock concept albums":                      ["progressive"],
    "heavy metal and hard rock for a workout":              ["metal", "hard rock"],
    "soulful singer-songwriter music for a quiet evening":  ["singer-songwriter", "adult contemporary", "adult alternative"],
    "modern country music":                                 ["country"],
    "contemporary R&B and soul":                            ["r&b", "soul"],
}
EVAL_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"   # independent of the retrieval model (mpnet)
CF_WEIGHT = 0.5       # fixed 
MMR_LAMBDA = 0.6      

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    train = pd.read_csv("data/train_set.csv")
    test = pd.read_csv("data/test_set.csv")        
    first = dict(list(QUERIES.items())[:1])

    common = dict(cf_weight=CF_WEIGHT, eval_embedding_model=EVAL_EMBEDDER)
    if mode == "smoke":
        evaluation_pipeline(train, test, first, sample_size=20, variants=ALL_VARIANTS,
                            mmr_lambda=MMR_LAMBDA, out_dir="results/smoke", **common)
    elif mode == "tune":
        evaluation_pipeline(train, test, QUERIES, sample_size=200,
                            variants=ALL_VARIANTS_NO_LLM + ALPHA_SWEEP + LAMBDA_SWEEP,
                            user_split="tune", out_dir="results/tune", **common)
    elif mode == "eval":
        # 500 users for everything except the slow LLM variant, which runs on the first 100 of them
        evaluation_pipeline(train, test, QUERIES, sample_size=500, llm_sample_size=100,
                            variants=ALL_VARIANTS, user_split="eval", mmr_lambda=MMR_LAMBDA,
                            out_dir="results/final", **common)
    elif mode == "gt":
        # accuracy against held-out items needs many users (hits are rare): all of half B, no LLM, GT metrics only
        evaluation_pipeline(train, test, QUERIES, sample_size=None, variants=ALL_VARIANTS_NO_LLM,
                            user_split="eval", mmr_lambda=MMR_LAMBDA, light=True,
                            out_dir="results/gt", cf_weight=CF_WEIGHT)
    else:
        raise SystemExit("mode must be smoke | tune | eval | gt")