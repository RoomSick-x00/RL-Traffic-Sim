# Member 2: SUMO reward comparison and handoff

## Checkpoint status

- 1–5: toy design, wrapper, and initial DQN validation completed previously.
- 6: integration code prepared; **open until a live SUMO smoke run succeeds**.
- 7: **open** until queue, wait, and combined rewards have three completed training runs each.
- 8: optional PPO deferred; finish the DQN reward comparison first.
- 9: packaging automated; **open** until Member 3 receives the real trained artifacts.

An earlier Git commit called “Checkpoint 7” was a code snapshot, not evidence that
the three real reward experiments had been completed.

## Run on the machine with SUMO

Use the project root as the working directory. Python needs NumPy, Gymnasium,
Stable-Baselines3/PyTorch, and the TraCI client compatible with the installed SUMO.
The `sumo` executable must be on PATH. Run `sumo --version` and
`python -c "import traci, gymnasium, stable_baselines3"` to check dependencies.
Toy commands do not import TraCI or require SUMO.

First run the complete pipeline smoke test:

```powershell
python -m rl_agent.train --env sumo --scenario normal --reward queue --smoke
```

This checks the Gymnasium interface, runs 1,000 DQN timesteps, evaluates during
training every 500 steps, saves/reloads the best model, and evaluates the final
model and fixed-cycle baseline. It uses validation traffic seeds 100/101 and test
traffic seeds 1000/1001. It also exercises connection isolation while the training
simulation is still open. All files go into a new printed `logs/dqn_*/` directory.
A poor learned reward after 1,000 steps is expected; this run tests integration.
Check `results.json` for finished baseline episodes, no teleports, reasonable
states, and working episode resets. Investigate exceptions or capped baseline
episodes before proceeding. Save that results file as Checkpoint 6 evidence.

After smoke passes, time one full queue run:

```powershell
python -m rl_agent.train --env sumo --scenario normal --reward queue --seed 42
```

Inspect `training_seconds_including_validation`, `training_steps_per_second`,
and `total_seconds` in results. They distinguish training/periodic-validation
time from the complete run including final evaluation. Do not estimate real
runtime from toy FPS. Toy hyperparameters are starting settings for SUMO.

Then run the matched experiment (replace the example smoke path):

```powershell
python -m rl_agent.compare_rewards --env sumo --scenario normal --smoke-results logs/dqn_YOUR_SMOKE_RUN/results.json
```

This runs three rewards × seeds 42, 43, 44 = **nine fresh DQN runs**, each with
150,000 timesteps, cap 150, learning rate 0.0005, target update interval 1,000,
exploration fraction 0.3 and final epsilon 0.05. The separate timing pilot is not
silently reused. Every run uses validation seeds 100–109 and test seeds 1000–1029.
The best checkpoint is selected by its own unscaled reward on validation seeds;
the test seeds do not select the checkpoint. Hyperparameters and traffic must
remain fixed across variants once the study starts. If tuning is necessary,
use training/validation evidence and rerun a matched study.

For a peak-demand study, repeat the smoke and comparison commands with
`--scenario peak`. Both scenarios together require 18 full runs. Traffic files
remain Member 1's inputs; the current normal scenario has 13 trips and peak has
20. These are small finite scenarios, so do not claim generalization to sustained
city traffic. More demand scenarios belong with Member 1.

An interrupted comparison can be resumed with the same options plus
`--resume logs/reward_comparison_YOUR_RUN`. Successful runs are skipped, failed
runs get a fresh directory, and existing model files are not overwritten. The
runner rejects mismatched settings or changed source/scenario files. Re-run the
smoke test after code changes. The comparison command requires matching SUMO
smoke evidence before launching the full study.

## Control and environment contract

- Raw state: 4 vehicle counts, 4 queues, 4 lane waiting aggregates, phase index.
- Model observation: clipped vehicle/16, queue/10, wait/400, plus one-hot phase.
- Actions: 0 keeps the current green; 1 requests the next yellow. Yellow stays
  automatic for 3 seconds and ignores switch requests. This is keep/switch, not
  arbitrary selection of a signal phase.
- Agent mode refreshes green's remaining duration to prevent automatic expiry.
  The baseline and episode-length script leave the original 10/3/10/3 program
  running. Neither the route demands nor the network's signal program is edited.
- Each raw environment has its own labeled TraCI connection and routes all
  operations through that connection. Closing evaluation cannot close training.
- The raw interface returns `(state, done)`. For SUMO, `done` means no expected
  vehicles remain. The wrapper maps it to `terminated`; the 150-step cap sets
  `truncated`. Both may be true if completion occurs exactly at the cap.
- The initial simulation step inside reset is excluded from wrapper step counts.
  SUMO diagnostics include that initial step. The measured baseline lengths were
  normal 50/53/52 and peak 64/64/75; 150 is a provisional cap, not a proven optimum.
- The toy still has its own 500-step `done`; with the simple SUMO mapping this
  produces both flags at 500. Do not use new toy training results as final evidence.

The bridge follows SUMO's documented [independent connection interface](https://sumo.dlr.de/docs/TraCI/Interfacing_TraCI_from_Python.html)
and [green-duration control](https://sumo.dlr.de/docs/Simulation/Traffic_Lights.html).

## Reward formulas and fair comparison

| Variant | Reported reward | Optimizer multiplier |
|---|---|---|
| queue | `-sum(queue)` | 0.01 |
| wait | `-sum(wait)` | 0.00025 |
| combined | `-sum(wait)/500 - applied_switch` | 1.0 |

Normalization bounds only clip the observation, not the raw reward inputs.
The optimizer multipliers are explicit experimental settings, not extra traffic
rewards. Monitor and evaluation retain the formulas above. Comparing these
variants therefore compares the declared formula plus its declared numerical
scaling. Do not silently tune scales separately after seeing held-out results.
One combined switch costs as much as a per-step aggregate wait of 500 units.
The native baseline is also charged for actual green-to-yellow switches, even
though its input action is always 0.

Do not rank different formulas by raw reward totals. Each evaluation records:

- Per-episode step count, return, termination and truncation flags.
- Mean total queue and mean total lane waiting aggregate across measured steps.
- Sums of these signals over steps and physical green-to-yellow switch counts.
- SUMO arrivals, departures, remaining demand, teleports, simulated time and version.

`mean_wait` is the average sum of lane waiting aggregates, **not** average vehicle
travel time or average trip waiting time. `wait_sum_over_steps` repeatedly samples
that aggregate and is not an independent total delay measure. Member 3 should
use the episode-level records and completion/remaining-demand counts together;
short capped episodes must not appear superior just because fewer steps were
observed. Teleports must be disclosed. Different traffic seeds randomize SUMO
dynamics; these fixed route files do not create new demand distributions.

## Files for Member 3

Each run contains `best_model.zip`, `final_model.zip`, `config.json`,
`train.monitor.csv`, `validation.json`, `results.json`, and a `source/` snapshot.
Configuration records reward scale, environment/scenario, step cap, seed splits,
hyperparameters, dependency versions and SHA-256 hashes of source/scenario files.
SUMO version is recorded in evaluation episode diagnostics.

The comparison directory contains `manifest.json` and `comparison.json` with
per-training-seed values and mean/sample-standard-deviation across training seeds.
After all runs complete, the runner creates a sibling `*_handoff.zip` containing
the study. Share this ZIP directly with Member 3; logs and models produced under
`logs/` remain Git-ignored. This automation does not send files to anybody.

For repeat evaluation from the project root:

```python
from stable_baselines3 import DQN
from rl_agent.env_factory import EnvironmentFactory
from rl_agent.rewards import queue_reward
from rl_agent.training import evaluate_policy

model = DQN.load("logs/YOUR_RUN/best_model.zip", device="cpu")
metrics = evaluate_policy(
    model, seeds=range(1000, 1030), reward_fn=queue_reward,
    env_factory=EnvironmentFactory("sumo", "normal"), max_steps=150,
)
```

Match the run's reward, scenario, cap, dependency versions and source snapshot.
Changing them defines a new evaluation, not reproduction of the saved result.

## Local verification without SUMO

```powershell
python -m unittest rl_agent.test_training rl_agent.test_sumo_bridge rl_agent.test_pipeline rl_agent.test_comparison -v
python test/test_episode_lengths.py
python -m rl_agent.train --env toy --smoke
```

These checks validate wiring and mocked connection behavior. They do not clear
Checkpoint 6 or substitute for real SUMO runs.
