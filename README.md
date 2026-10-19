# RL-Traffic-Sim

## Toy DQN training

Run from the project root, using the Python environment with Gymnasium and Stable-Baselines3 installed:

```powershell
python rl_agent/train.py
```

The defaults validated with queue reward are 150,000 timesteps, exploration fraction `0.3`,
final exploration probability `0.05`, learning rate `5e-4`, target update interval `1000`,
and training reward multiplier `0.01`. Traffic dynamics and the 16-value clipped observation
are unchanged. The reward functions still return their original values; scaling happens
only in the training wrapper, after the episode monitor.

Each run gets a new directory under Git-ignored `logs/`, containing:

- `train.monitor.csv`: individual episode rewards in their original units.
- `validation.json`: evaluations without exploration every 10,000 steps, plus the final weights.
- `best_model.zip`: checkpoint selected using validation traffic seeds 100–109.
- `final_model.zip`: final weights, saved separately even if they perform worse.
- `results.json`: best and final checkpoint results plus the selected reward's fixed-cycle baseline on separate test traffic seeds 1000–1029.
- `config.json`: settings needed to reproduce the run.

Use the best checkpoint for evaluation. The final checkpoint is not automatically the best.
These results validate the toy pipeline; train fresh and evaluate again for SUMO.

Parameters can be changed explicitly:

```powershell
python rl_agent/train.py --seed 43
python rl_agent/train.py --reward-scale 1 --target-update-interval 10000 --learning-rate 0.0001
python rl_agent/train.py --reward wait
python rl_agent/train.py --reward wait --reward-scale 0.001
python rl_agent/train.py --reward combined
```

The second command reproduces the original hyperparameter configuration with the new logging
and evaluation workflow. Queue reward uses training scale `0.01`; selecting wait reward
automatically uses the starting scale `0.00025`. Combined reward uses training scale `1.0`
because its formula already normalizes waiting: `-sum(wait) / 500 - applied_switch`.
An ignored switch request during
yellow has no switch penalty. These combined weights are starting values, not tuned results.
An explicit `--reward-scale` overrides the default. The fixed-cycle baseline in `results.json`
uses the same selected reward and test traffic seeds. Wait and combined reward training settings
have not been tuned by the queue-reward experiments.
`rl_agent/train_dqn.py` uses the same entry point and defaults.

Run the training integration checks with:

```powershell
python -m unittest rl_agent.test_training -v
```
