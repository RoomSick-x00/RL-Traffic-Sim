# RL-Traffic-Sim

## SUMO integration and reward comparison

The Member 2 pipeline now accepts `--env sumo --scenario normal` (or `peak`) and
uses a 150-step cap by default. Training and every evaluation use the selected
factory, cap and reward. Live SUMO verification still needs the SUMO machine.

Start there with `python -m rl_agent.train --env sumo --scenario normal --smoke`.
Follow [SUMO_HANDOFF.md](SUMO_HANDOFF.md) for the one-seed timing pilot, the matched
three-reward/three-seed study, resume commands, metrics and the Member 3 ZIP handoff.
The comparison runner requires matching smoke results before a full SUMO study.

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

## Measure SUMO episode lengths (Member 1)

On the machine with SUMO installed, ensure `sumo` is on PATH and Python can import
`traci`. From the project root, run:

```powershell
python test/measure_episode_lengths.py
```

This runs normal and peak traffic with seeds 42, 43, and 44, using the network's
automatic fixed signal cycle (10 seconds green, 3 seconds yellow per direction).
It closes SUMO between episodes, prints completion lengths, and saves a new
`logs/episode_lengths_*/results.json`. Send that JSON file back for choosing the
training episode cap; logs remain Git-ignored.

The default 10,000-step limit is only a measurement safeguard. Runs reaching it
without finishing are marked truncated and must not be treated as completed runs.
Step counts exclude the one simulation step inside reset; each measured step is
one simulated second with the supplied configurations. To change the run settings:

```powershell
python test/measure_episode_lengths.py --scenarios normal peak --seeds 10 20 30 --max-steps 20000
```

The loop can be checked without SUMO using `python test/test_episode_lengths.py`.

The Gymnasium wrapper defaults to a 150-step cap (`max_steps=150`), based on the
reported 50–75-step SUMO fixed-cycle episodes. It counts calls to `step()` after
each reset. Raw SUMO `done` sets `terminated`; reaching the cap sets `truncated`.
Both flags can be true if traffic finishes exactly at the cap. Toy training and
evaluation explicitly retain `max_steps=500`. The wrapper forwards raw `done`
unchanged as `terminated`; with the toy environment's current 500-step `done`,
both flags become true at step 500. This simple completion mapping is intended
for SUMO's demand-finished signal.
