import pandas as pd
res = pd.read_csv("results/lightfm_sweep.csv")
agg = (res.groupby(['lr', 'alpha', 'dim', 'epoch'])[['val_r50', 'val_p10']]
          .agg(['mean', 'std']).reset_index())
agg.columns = ['lr', 'alpha', 'dim', 'epoch', 'r50', 'r50_sd', 'p10', 'p10_sd']

best = agg['r50'].max()
print("Best:", agg.loc[agg['r50'].idxmax()].to_dict(), "| popularity: 0.0299")
elig = agg[agg['r50'] >= 0.98 * best].sort_values(['dim', 'epoch', 'r50'],
                                                   ascending=[True, True, False])
print(elig.head(15).to_string())          # first row = choice under the rule