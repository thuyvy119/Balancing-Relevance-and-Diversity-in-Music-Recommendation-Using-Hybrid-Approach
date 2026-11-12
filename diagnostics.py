import numpy as np, pandas as pd
from scripts.lightfm_convergence import split_per_user, build 

data = pd.read_csv("data/train_set.csv")
print(data['label'].value_counts())              
print(data.groupby('uid').size().describe())     
print(data['uid'].nunique(), data['iid'].nunique())

data = data.drop_duplicates(['uid', 'iid']).reset_index(drop=True)
train, val = split_per_user(data)
tr, va = build(train, val)

pop = np.asarray(tr.sum(axis=0)).ravel()
va_items = va.tocoo().col
print("val items with 0 train interactions:", (pop[va_items] == 0).mean())

tr_csr, va_csr = tr.tocsr(), va.tocsr()
order = np.argsort(-pop)
hits = []
for u in range(va_csr.shape[0]):
    v = set(va_csr[u].indices)
    if not v:
        continue
    seen = set(tr_csr[u].indices)
    top = [i for i in order[:10 + len(seen)] if i not in seen][:10]
    hits.append(len(v & set(top)) / 10)
print("popularity P@10:", np.mean(hits))