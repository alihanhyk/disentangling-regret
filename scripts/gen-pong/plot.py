import pandas as pd
import numpy as np

df = pd.DataFrame(
    np.zeros((4, 8)),
    index = ['id', 'dists', 'justfield', 'justcount'],
    columns = pd.MultiIndex.from_product([['test_stochastic', 'test_observational'], ['R', 'GR', 'GR_rec', 'GR_dec']])).astype(str)

max_score = 1.0
min_score = -1.0

for ind in df.index:

    score_training = list()
    score, score_retrained = list(), list()
    for seed in [1, 2, 3, 4, 5]:

        with open(f"results/gen-pong/{ind}{seed}/score_training.txt", 'r') as f:
            score_training.append(float(f.read()))
        with open(f"results/gen-pong/{ind}{seed}/score_test_stochastic.txt", 'r') as f:
            score.append(float(f.read()))
        
        with open(f"results/gen-pong/{ind}{seed}-retrained-stochastic/score_test_stochastic.txt", 'r') as f:
            score_retrained.append(float(f.read()))

    regret = 1. - (np.array(score_training) - min_score) / (max_score - min_score)
    regret_g = 1. - (np.array(score) - min_score) / (max_score - min_score)
    regret_grec = 1. - (np.array(score_retrained) - min_score) / (max_score - min_score)
    regret_grec = np.minimum(regret_g, regret_grec)
    regret_gdec = regret_g - regret_grec

    df.loc[ind, ('test_stochastic', 'R')] = f"{regret.mean():.3f} ({regret.std():.3f})"
    df.loc[ind, ('test_stochastic', 'GR')] = f"{regret_g.mean():.3f} ({regret_g.std():.3f})"
    df.loc[ind, ('test_stochastic', 'GR_rec')] = f"{regret_grec.mean():.3f} ({regret_grec.std():.3f})"
    df.loc[ind, ('test_stochastic', 'GR_dec')] = f"{regret_gdec.mean():.3f} ({regret_gdec.std():.3f})"

    score, score_retrained = list(), list()
    for seed in [1, 2, 3, 4, 5]:
        with open(f"results/gen-pong/{ind}{seed}/score_test_observational.txt", 'r') as f:
            score.append(float(f.read()))
        with open(f"results/gen-pong/{ind}{seed}-retrained-observational/score_test_observational.txt", 'r') as f:
            score_retrained.append(float(f.read()))

    regret_g = 1. - (np.array(score) - min_score) / (max_score - min_score)
    regret_grec = 1. - (np.array(score_retrained) - min_score) / (max_score - min_score)
    regret_grec = np.minimum(regret_g, regret_grec)
    regret_gdec = regret_g - regret_grec

    df.loc[ind, ('test_observational', 'R')] = f"{regret.mean():.3f} ({regret.std():.3f})"
    df.loc[ind, ('test_observational', 'GR')] = f"{regret_g.mean():.3f} ({regret_g.std():.3f})"
    df.loc[ind, ('test_observational', 'GR_rec')] = f"{regret_grec.mean():.3f} ({regret_grec.std():.3f})"
    df.loc[ind, ('test_observational', 'GR_dec')] = f"{regret_gdec.mean():.3f} ({regret_gdec.std():.3f})"

print(df)
