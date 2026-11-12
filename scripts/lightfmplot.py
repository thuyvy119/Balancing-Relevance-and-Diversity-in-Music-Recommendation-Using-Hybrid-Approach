import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
res = pd.read_csv("results/lightfm_convergence.csv")   # skip if res is still in memory
pop_r50 =  0.02993722839208112
agg = res.groupby(['lr', 'epoch'])[['val_p10', 'val_r50', 'val_auc']].agg(['mean', 'std']).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
panels = [(axes[0], 'val_r50', 'Validation Recall@50'),
          (axes[1], 'val_p10', 'Validation Precision@10')]

for ax, m, title in panels:
    for lr in sorted(res['lr'].unique()):
        a = agg[agg['lr'] == lr]
        mean = a[(m, 'mean')].values
        std = a[(m, 'std')].fillna(0).values
        ax.plot(a['epoch'], mean, marker='o', label=f'lr={lr}')
        ax.fill_between(a['epoch'], np.maximum(mean - std, 0), mean + std, alpha=0.2)
    ax.axvline(50, ls='--', c='gray')
    ax.set_xlabel('Epochs')
    ax.set_title(title)

axes[0].axhline(pop_r50, ls=':', c='k', label='popularity')
axes[0].legend()
plt.tight_layout()
plt.savefig("results/lightfm_learning_curve.png", dpi=300)

summary = res.groupby(['lr', 'epoch'])[['val_r50', 'val_p10', 'val_auc']].mean().round(4)
print(summary.to_string())

best = summary['val_r50'].max()
print("\nBest mean Recall@50:", best)
for lr, g in summary['val_r50'].groupby(level=0):
    ok = g[g >= 0.99 * g.max()]
    print(f"lr={lr}: peak={g.max():.4f} at epoch {g.idxmax()[1]}, "
          f"smallest epoch within 1% of its peak = {ok.index[0][1]}")