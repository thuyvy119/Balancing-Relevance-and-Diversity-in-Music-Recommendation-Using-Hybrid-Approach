import os, sys, itertools
import numpy as np, pandas as pd
from multiprocessing import Pool, cpu_count
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k
from lightfm_convergence import split_per_user, build, K_P, K_R

QUICK = "--quick" in sys.argv
LRS = [0.005, 0.01]
ALPHAS = [1e-6, 1e-5, 1e-4]
DIMS = [10, 32, 64]
SEEDS = [0, 1, 2]
CHECKPOINTS = [5, 10, 20, 30, 50]
OUT = "results/lightfm_sweep.csv"

if QUICK:  # timing test with the slowest config
    LRS, ALPHAS, DIMS, SEEDS, CHECKPOINTS = [0.01], [1e-6], [64], [0], [5, 10]
    OUT = "results/lightfm_sweep_quick.csv"

_TR = _VA = None

def _init(tr, va):
    global _TR, _VA
    _TR, _VA = tr, va

def run(job):
    lr, alpha, dim, seed = job
    model = LightFM(loss='warp', learning_rate=lr, no_components=dim,
                    item_alpha=alpha, user_alpha=alpha, random_state=seed)
    rows, done = [], 0
    for ep in CHECKPOINTS:
        model.fit_partial(_TR, epochs=ep - done, num_threads=1)
        done = ep
        rows.append(dict(
            lr=lr, alpha=alpha, dim=dim, seed=seed, epoch=ep,
            val_r50=recall_at_k(model, _VA, train_interactions=_TR, k=K_R).mean(),
            val_p10=precision_at_k(model, _VA, train_interactions=_TR, k=K_P).mean(),
        ))
    return rows

if __name__ == "__main__":
    data = pd.read_csv("data/train_set.csv")
    data = data[data['label'] > 0]
    data = data.drop_duplicates(subset=['uid', 'iid']).reset_index(drop=True)
    train, val = split_per_user(data)
    tr, va = build(train, val)

    jobs = list(itertools.product(LRS, ALPHAS, DIMS, SEEDS))
    if os.path.exists(OUT):  # resume
        d = pd.read_csv(OUT)
        done = set(zip(d.lr, d.alpha, d.dim, d.seed))
        jobs = [j for j in jobs if j not in done]
    print(f"{len(jobs)} runs to do")

    os.makedirs("results", exist_ok=True)
    n_workers = max(1, min(cpu_count() - 1, 6))
    with Pool(n_workers, initializer=_init, initargs=(tr, va)) as pool:
        for k, rows in enumerate(pool.imap_unordered(run, jobs), 1):
            pd.DataFrame(rows).to_csv(OUT, mode='a', index=False,
                                      header=not os.path.exists(OUT))
            print(f"finished {k}/{len(jobs)}", flush=True)