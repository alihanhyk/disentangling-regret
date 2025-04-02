import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

###

parser = argparse.ArgumentParser()
parser.add_argument('--no-action', action='store_true')
parser.add_argument('--no-obs', action='store_true')
args = parser.parse_args()

max_score = .9684
# this is
#   the maximum score achievable in a single episode (with a lucky maze layout)
#   not the optimal score averages over different episodes!

###

sns.set_theme()

if not args.no_action:

    xs = np.array([.1, .2, .3, .4, .5])
    regret = np.zeros((5, 5))
    regret_rec = np.zeros((5, 5))

    for i, seed in enumerate([1, 2, 3, 4, 5]):
        for j, tag in enumerate(['10', '20', '30', '40', '50']):

            with open(f"results/regret-minigrid/dec{tag}_{seed}/score.txt", 'r') as f:
                score = float(f.read())
            with open(f"results/regret-minigrid/dec{tag}R_{seed}/score.txt", 'r') as f:
                score_rec = float(f.read())
                score_rec = max(score, score_rec)

            regret[i,j] = 1. - score / max_score
            regret_rec[i,j] = 1. - score_rec / max_score
        
    plt.figure(figsize=(4,3))

    ax = sns.lineplot(x=xs, y=regret.mean(0), marker='o', label="$R$")
    ax.fill_between(xs, regret.mean(0)-regret.std(0), regret.mean(0)+regret.std(0), alpha=0.2)

    ax = sns.lineplot(x=xs, y=regret_rec.mean(0), marker='v', label="$R^{\\text{rec}}$")
    ax.fill_between(xs, regret_rec.mean(0)-regret_rec.std(0), regret_rec.mean(0)+regret_rec.std(0), alpha=0.2)

    regret_dec = regret - regret_rec
    ax = sns.lineplot(x=xs, y=regret_dec.mean(0), marker='^', label="$R^{\\text{dec}}$")
    ax.fill_between(xs, regret_dec.mean(0)-regret_dec.std(0), regret_dec.mean(0)+regret_dec.std(0), alpha=0.2)

    plt.legend()
    plt.xlabel("Random Action Probability")
    plt.ylim(-0.025, 0.425)

    plt.tight_layout()
    plt.savefig(f"reports/regret/minigrid-action.pdf")

if not args.no_obs:

    xs = np.array([.1, .2, .3, .4, .5])
    regret = np.zeros((5, 5))
    regret_rec = np.zeros((5, 5))

    for i, seed in enumerate([1, 2, 3, 4, 5]):
        for j, tag in enumerate(['10', '20', '30', '40', '50']):
        
            with open(f"results/regret-minigrid/rec{tag}_{seed}/score.txt", 'r') as f:
                score = float(f.read())
            with open(f"results/regret-minigrid/rec{tag}R_{seed}/score.txt", 'r') as f:
                score_rec = float(f.read())
                score_rec = max(score, score_rec)

            regret[i,j] = 1. - score / max_score
            regret_rec[i,j] = 1. - score_rec / max_score

    plt.figure(figsize=(4,3))

    ax = sns.lineplot(x=xs, y=regret.mean(0), marker='o', label="$R$")
    ax.fill_between(xs, regret.mean(0)-regret.std(0), regret.mean(0)+regret.std(0), alpha=0.2)

    ax = sns.lineplot(x=xs, y=regret_rec.mean(0), marker='v', label="$R^{\\text{rec}}$")
    ax.fill_between(xs, regret_rec.mean(0)-regret_rec.std(0), regret_rec.mean(0)+regret_rec.std(0), alpha=0.2)

    regret_dec = regret - regret_rec
    ax = sns.lineplot(x=xs, y=regret_dec.mean(0), marker='^', label="$R^{\\text{dec}}$")
    ax.fill_between(xs, regret_dec.mean(0)-regret_dec.std(0), regret_dec.mean(0)+regret_dec.std(0), alpha=0.2)

    plt.legend()
    plt.xlabel("Observation Masking Probability")
    plt.ylim(-0.025, 0.425)

    plt.tight_layout()
    plt.savefig(f"reports/regret/minigrid-obs.pdf")
