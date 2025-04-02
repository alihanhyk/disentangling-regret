# Disentangling Recognition and Decision Regrets in Image-Based Reinforcement Learning
Code author: Alihan Hüyük ([ahuyuk@seas.harvard.edu](mailto:ahuyuk@seas.harvard.edu))

This repository contains implementations of RL agents and environments designed to study recognition and decision regrets. For more details, please refer to our paper: "[Disentangling Recognition and Decision Regrets in Image-Based Reinforcement Learning](https://arxiv.org/abs/2409.13108)."

`ColoredKeys` is a grid-based maze environemnt in which an agent must navigate to a goal cell. Reaching the goal requires picking up a key and unlocking a door. However, keys and doors are cosmetically colored, either randomly (making color an irrelevant feature) or based on the maze configuration (making it a spurious feature). This makes generalization difficult, inducing either high recognition regret or high decision regret. `ColoredKeysFilter` allows different features (like colors or door locations) to be filtered out of observations so that their impact on generalization performance can be measured.

Similarly, `environments.mypong` implements different wrappers of the Atari game Pong that introduce irrelevant or spurious features. These features can also be filtered out of observations to enable controlled experiments.

### Installation
1. Clone the repository
2. Create a new virtual environment with Python 3.11
3. Run the following commands:
```bash
pip install -r requirements.txt
pip install -e .
```

### Running Experiments
Here is an example where we measure the recognition and decision generalization regrets of an agent trained in `MultiCoupled` but deployed in `Single` (two versions of `ColoredKeys` that only differ in terms of how they are colored):
```bash
python scripts/gen-minigrid/main.py --env MultiCoupled --exp-name example                        # training
python scripts/gen-minigrid/eval.py --env MultiCoupled --exp-name example --output training      # measuring regular regret
python scripts/gen-minigrid/eval.py --env Single --exp-name example                              # measuring generalization regret
python scripts/gen-minigrid/main.py --env Single --exp-name example-retrained --retrain example  # re-training of the decision policy
python scripts/gen-minigrid/eval.py --env Single --exp-name example-retrained                    # measuring recognition generalization regret
```

Results in our paper can be reproduced by running the following commands:
```bash
bash jobs/run-regret.sh
python scripts/regret/plot-minigrid.py  # Figure 3a
python scripts/regret/plot-pong.py      # Figure 3b
```
```bash
bash jobs/run-gen-minigrid.sh
python scripts/gen-minigrid/plot0.py                      # for regret metrics in Table 1
python scripts/gen-minigrid/plot1.py --exp-name id        # for similarity matrices in Table 1
python scripts/gen-minigrid/plot1.py --exp-name hidecols
python scripts/gen-minigrid/plot1.py --exp-name hidedoor
python scripts/gen-minigrid/plot1.py --exp-name hideboth
python scripts/gen-minigrid/plot1.py --exp-name onehot
```
```bash
bash jobs/run-gen-pong.sh
python scripts/gen-pong/plot.py  # Table 2
```
```bash
bash jobs/run-usecase.sh
python scripts/usecase/plot.py  # Figure 5
```

### Citing
If you use this software, please cite our paper:
```bibtex
@article{huyuk2024disentangling,
  title={Disentangling Recognition and Decision Regrets in Image-Based Reinforcement Learning},
  author={H{\"u}y{\"u}k, Alihan and Koblitz, Arndt Ryo and Mohajeri, Atefeh and Andrews, Matthew},
  journal={arXiv preprint arXiv:2409.13108},
  year={2024}
}
```
