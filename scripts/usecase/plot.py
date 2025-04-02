import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

index = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
columns = ['R', 'GR', 'GR_rec', 'GR_dec']

df_mean = pd.DataFrame(index=index, columns=columns)
df_std = pd.DataFrame(index=index, columns=columns)

opt_score = 0.9190

for i in reversed(index):
    
    scores = list()
    scores_training = list()
    scores_retrained = list()

    for seed in range(1, 10 + 1):
        with open(f"results/gen-minigrid/use{seed}-repsize{i}/score.txt") as f:
            scores.append(float(f.read()))
        with open(f"results/gen-minigrid/use{seed}-repsize{i}/score_training.txt") as f:
            scores_training.append(float(f.read()))
        with open(f"results/gen-minigrid/use{seed}-repsize{i}-retrained/score.txt") as f:
            scores_retrained.append(float(f.read()))

    regret = 1. - np.array(scores_training) / opt_score
    regret_g = 1. - np.array(scores) / opt_score
    regret_grec = 1. - np.array(scores_retrained) / opt_score
    regret_grec = np.minimum(regret_g, regret_grec)
    regret_gdec = regret_g - regret_grec

    df_mean.loc[i,'R'] = regret.mean()
    df_mean.loc[i,'GR'] = regret_g.mean()
    df_mean.loc[i,'GR_rec'] = regret_grec.mean()
    df_mean.loc[i,'GR_dec'] = regret_gdec.mean()

    df_std.loc[i,'R'] = regret.std()
    df_std.loc[i,'GR'] = regret_g.std()
    df_std.loc[i,'GR_rec'] = regret_grec.std()
    df_std.loc[i,'GR_dec'] = regret_gdec.std()


sns.set_theme()

plt.figure(figsize=(4,3))

ax = sns.lineplot(x=np.log2(index), y=df_mean['R'].values, marker='v', label='$R$')
lower = np.array(df_mean['R'].values - df_std['R'].values, dtype=float)
upper = np.array(df_mean['R'].values + df_std['R'].values, dtype=float)
ax.fill_between(np.log2(index), lower, upper, alpha=0.1)

ax = sns.lineplot(x=np.log2(index), y=df_mean['GR'].values, marker='^', label='$\\widehat{GR}$')
lower = np.array(df_mean['GR'].values - df_std['GR'].values, dtype=float)
upper = np.array(df_mean['GR'].values + df_std['GR'].values, dtype=float)
ax.fill_between(np.log2(index), lower, upper, alpha=0.1)

plt.legend()
plt.xlabel("Number of Latent Channels")
plt.xticks([0, 2, 4, 6], ["$2^0$", "$2^2$", "$2^4$", "$2^6$"])
plt.ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig(f"reports/usecase-regrets.pdf")


plt.figure(figsize=(4,3))

ax = sns.lineplot(x=np.log2(index), y=df_mean['GR_rec'].values, marker='v', label='$\\widehat{GR}^{rec}$')
lower = np.array(df_mean['GR_rec'].values - df_std['GR_rec'].values, dtype=float)
upper = np.array(df_mean['GR_rec'].values + df_std['GR_rec'].values, dtype=float)
ax.fill_between(np.log2(index), lower, upper, alpha=0.1)

ax = sns.lineplot(x=np.log2(index), y=df_mean['GR_dec'].values, marker='^', label='$\\widehat{GR}^{dec}$')
lower = np.array(df_mean['GR_dec'].values - df_std['GR_dec'].values, dtype=float)
upper = np.array(df_mean['GR_dec'].values + df_std['GR_dec'].values, dtype=float)
ax.fill_between(np.log2(index), lower, upper, alpha=0.1)

plt.legend()
plt.xlabel("Number of Latent Channels")
plt.xticks([0, 2, 4, 6], ["$2^0$", "$2^2$", "$2^4$", "$2^6$"])
plt.ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig(f"reports/usecase-decompregrets.pdf")


