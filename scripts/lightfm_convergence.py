import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score

K = 10
CHECKPOINTS = [1, 2, 5, 10, 20, 30, 40, 50, 75, 100]
LRS = [0.005, 0.01, 0.05, 0.1]
SEEDS = [0, 1, 2]
SPLIT_SEED = 42
K_P, K_R = 10, 50

def split_per_user(df, val_frac=0.2, seed=SPLIT_SEED, min_interactions=2):
    df = df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    val_idx = []
    for _, g in df.groupby('uid'):
        if len(g) < min_interactions:
            continue
        n_val = max(1, int(round(len(g) * val_frac)))
        val_idx.extend(rng.choice(g.index.values, n_val, replace=False))
    val = df.loc[val_idx]
    train = df.drop(index=val_idx)

    # safety net: no val pair may also exist in train
    train_pairs = set(zip(train['uid'], train['iid']))
    val = val[[(u, i) not in train_pairs for u, i in zip(val['uid'], val['iid'])]]

    # warm items only: CF cannot score items unseen in training
    n0 = len(val)
    val = val[val['iid'].isin(set(train['iid']))]
    print(f"Val pairs {n0} -> {len(val)} after warm filter; val users: {val['uid'].nunique()}")
    return train, val

def popularity_recall(tr, va, k):
    pop = np.asarray(tr.sum(axis=0)).ravel()
    order = np.argsort(-pop)
    tr_csr, va_csr = tr.tocsr(), va.tocsr()
    vals = []
    for u in range(va_csr.shape[0]):
        v = set(va_csr[u].indices)
        if not v:
            continue
        seen = set(tr_csr[u].indices)
        top = [i for i in order[:k + len(seen)] if i not in seen][:k]
        vals.append(len(v & set(top)) / len(v))
    return float(np.mean(vals))

def build(train, val):
    full = pd.concat([train, val])
    ds = Dataset()
    ds.fit(users=full['uid'].unique(), items=full['iid'].unique())  # shared ID mapping
    tr, _ = ds.build_interactions(zip(train['uid'], train['iid']))
    va, _ = ds.build_interactions(zip(val['uid'], val['iid']))
    return tr, va

def run_curve(tr, va, lr, seed, no_components=10):
    model = LightFM(loss='warp', learning_rate=lr, no_components=no_components,
                    item_alpha=1e-6, user_alpha=1e-6, random_state=seed)
    rows, done = [], 0
    for ep in CHECKPOINTS:
        model.fit_partial(tr, epochs=ep - done, num_threads=1)  
        done = ep
        rows.append(dict(
            lr=lr, seed=seed, epoch=ep,
            val_p10=precision_at_k(model, va, train_interactions=tr, k=K_P).mean(),
            val_r50=recall_at_k(model, va, train_interactions=tr, k=K_R).mean(),
            val_auc=auc_score(model, va, train_interactions=tr).mean(),
        ))
    return rows

if __name__ == "__main__":
    data = pd.read_csv("data/train_set.csv")
    data = data[data['label'] > 0]
    data = data.drop_duplicates(subset=['uid', 'iid']).reset_index(drop=True)
    print(data['uid'].map(data['uid'].value_counts()).describe())  # sanity check

    train, val = split_per_user(data)
    tr, va = build(train, val)
    pop_r50 = popularity_recall(tr, va, K_R)
    print("Popularity Recall@50:", pop_r50)

    res = pd.DataFrame([r for lr in LRS for s in SEEDS for r in run_curve(tr, va, lr, s)])
    res.to_csv("results/lightfm_convergence.csv", index=False)

    agg = res.groupby(['lr', 'epoch']).agg(['mean', 'std']).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, m, title in [(axes[0], 'val_p', f'Validation Precision@{K}'),
                         (axes[1], 'val_r', f'Validation Recall@{K}')]:
        for lr in LRS:
            a = agg[agg['lr'] == lr]
            ax.plot(a['epoch'], a[(m, 'mean')], marker='o', label=f'lr={lr}')
            ax.fill_between(a['epoch'], a[(m, 'mean')] - a[(m, 'std')],
                            a[(m, 'mean')] + a[(m, 'std')], alpha=0.2)
        ax.axvline(50, ls='--', c='gray')
        ax.set_xlabel('Epochs'); ax.set_title(title); ax.legend()
    plt.tight_layout()
    plt.savefig("results/lightfm_learning_curve.png", dpi=300)  # 300 dpi (also helps 2.4)