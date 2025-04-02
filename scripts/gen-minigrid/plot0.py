import pandas as pd
import numpy as np

df = pd.DataFrame(
    np.zeros((5, 4)),
    index = ['id', 'hidecols', 'hidedoor', 'hideboth', 'onehot'],
    columns = ['R', 'GR', 'GR_rec', 'GR_dec']).astype(str)

opt_score = 0.9190

for ind in df.index:

    score_training, score, score_retrained = list(), list(), list()
    for seed in [1, 2, 3, 4, 5]:

        with open(f"results/gen-minigrid/{ind}{seed}/score_training.txt") as f:
            score_training.append(float(f.read()))
        with open(f"results/gen-minigrid/{ind}{seed}/score.txt") as f:
            score.append(float(f.read()))
        with open(f"results/gen-minigrid/{ind}{seed}-retrained/score.txt") as f:
            score_retrained.append(float(f.read()))


    regret = 1. - np.array(score_training) / opt_score
    regret_g = 1. - np.array(score) / opt_score
    regret_grec = 1. - np.array(score_retrained) / opt_score
    regret_grec = np.minimum(regret_g, regret_grec)
    regret_gdec = regret_g - regret_grec

    df.loc[ind, 'R'] = f"{regret.mean():.3f} ({regret.std():.3f})"
    df.loc[ind, 'GR'] = f"{regret_g.mean():.3f} ({regret_g.std():.3f})"
    df.loc[ind, 'GR_rec'] = f"{regret_grec.mean():.3f} ({regret_grec.std():.3f})"
    df.loc[ind, 'GR_dec'] = f"{regret_gdec.mean():.3f} ({regret_gdec.std():.3f})"

print(df)
