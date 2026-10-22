# Progress since the Checkpoints 1–5 summary

**Updated:** 23 October 2026  
**Owner:** Member 2 — RL agent design, training, and reward comparison  
**Previous document:** [checkpoint_1_5_summary.md](checkpoint_1_5_summary.md)  
**Code reviewed through:** `f0625d6` — visual SUMO replay

This records work completed after the previous summary, including decisions that
supersede its open items. Results come from the SUMO outputs shared in this chat;
implementation details were checked against the repository. Wait and combined
runs may have been started elsewhere, but their completed results have not yet
been supplied. They remain unconfirmed here.

## Current checkpoint status

| Checkpoint | Status | Evidence or remaining work |
|---|---|---|
| 1–5: design and toy pipeline | Previously completed | See the earlier summary; toy results are pipeline evidence. |
| 6: real SUMO integration | Completed for normal traffic | Live smoke run and full queue-reward training/evaluation succeeded on the SUMO machine. |
| 7: all reward variants trained | In progress: 3 of 9 normal-scenario runs confirmed | Queue seeds 42, 43, 44 complete; wait and combined seeds 42, 43, 44 still need result verification. |
| 8: optional PPO | Deferred | Finish the DQN reward study first. |
| 9: handoff to Member 3 | Pending | Packaging support exists; complete study artifacts still need collection and delivery. |

The old commit titled “Checkpoint 7” was a code snapshot, not completion of all
real SUMO experiments. The older `SUMO_HANDOFF.md` and README statements that live
verification is still pending describe the earlier handoff stage; the shared
normal-scenario runs now provide that verification.

## 1. Finalized the reward implementations

All three functions are available in `rl_agent/rewards.py` and compatible with
the wrapper's two-argument reward call.

| Variant | Raw reward used for reporting/evaluation | Training multiplier |
|---|---|---|
| Queue | `-sum(raw_state[4:8])` | `0.01` |
| Wait | `-sum(raw_state[8:12])` | `0.00025` |
| Combined | `-sum(raw_state[8:12]) / 500 - applied_switch` | `1.0` |

- The wrapper captures the phase before stepping and checks the resulting phase
  to detect an applied green-to-yellow switch. An ignored switch request during
  yellow does not receive a switch penalty.
- Queue and wait rewards accept and ignore the extra switch argument.
- Combined reward uses normalized waiting with `alpha = beta = 1.0`. One switch
  costs the same as an aggregate waiting value of 500 for one reward step.
- Reward calculation uses raw state values. Observation clipping does not cap
  the underlying waiting values or reward penalties.
- Training scaling happens after Monitor records the raw reward. The earlier
  summary's phrase “divided by 0.01” was incorrect: queue reward is **multiplied
  by 0.01**, equivalent to dividing by 100.

## 2. Aligned the real environment and Gymnasium wrapper

- Changed `TrafficEnvironment.step(action)` to return `(state, done)`; reward
  calculation belongs in the wrapper.
- Added seed forwarding to the real environment's reset and SUMO startup.
- Added wrapper lifecycle tracking: a new reset closes the previous episode,
  then starts a fresh one. `close()` clears the started flag.
- Added a validated `max_steps` constructor setting and resettable step counter.
  Removed the unused `done_is_truncation` option and hardcoded cap check.
- `terminated = done` represents exhausted SUMO demand.
  `truncated = step_count >= max_steps` represents the wrapper's safety cap.
  Both can be true when completion coincides with the cap.
- Both raw environments expose `get_state()` without advancing the simulation.
  The wrapper uses it to capture the old phase.
- Step `info` now includes the raw state and applied-switch indicator for metrics.
- The state/action contract remains 13 raw values, 16 processed values, and two
  actions: keep or request a switch. Normalization remains vehicle/16, queue/10,
  wait/400 with upper clipping and one-hot phase encoding.

The current simple completion mapping is intended for SUMO. The toy environment
still returns `done` at its own 500-step limit, so the current wrapper can report
both flags there; this differs from the toy-only mapping in the old summary.

## 3. Measured episode lengths and chose the SUMO cap

Added `test/measure_episode_lengths.py` and its checks. The script records
episode lengths in a new ignored logs folder, using SUMO's native fixed cycle.

| Scenario | Seed 42 | Seed 43 | Seed 44 |
|---|---:|---:|---:|
| Normal | 50 | 53 | 52 |
| Peak | 64 | 64 | 75 |

All six reported measurements completed naturally. Selected **150 steps** as a
provisional SUMO cap with margin above these baseline lengths. Toy experiments
retain 500. Counts exclude the initial simulation step inside reset; a capped
SUMO episode therefore reaches simulation time 151 seconds with these configs.
This cap is a safeguard, not a proven optimum for every policy or future demand.

## 4. Made training and evaluation swappable

- Added `rl_agent/env_factory.py` to construct toy or real environments. SUMO
  dependencies are imported only when SUMO is selected.
- Threaded `env_factory` and `max_steps` through training, validation, final/best
  evaluation, and baseline evaluation, removing the hardcoded toy instances.
- Added command-line selection of environment, scenario, reward, and seed, plus
  a short `--smoke` integration run.
- Preserved validation-based best-checkpoint selection and separate final weights.
  SUMO agents train fresh; toy-trained weights are not the final research models.
- Preserved matched settings: 150,000 timesteps, learning rate `0.0005`, target
  update interval 1,000, exploration fraction `0.3`, final epsilon `0.05`, and CPU.
- Validation uses traffic seeds 100–109; held-out testing uses 1000–1029. Training
  seeds are 42, 43, and 44. Test results do not select the best checkpoint.

## 5. Corrected SUMO control and connection handling

- Each raw environment uses a unique labeled TraCI connection. Training and
  evaluation can coexist without evaluation closing or controlling training.
- Agent mode holds a green phase when action 0 is selected, preventing the
  network's automatic green expiry from overriding the agent's keep decision.
- Action 1 requests yellow from green. Yellow transitions remain automatic.
- The fixed-cycle baseline and measurement script use the original native
  10/3/10/3-second signal program. Baseline combined reward also charges actual
  green-to-yellow transitions.
- Added arrival, departure, teleport, remaining-demand, simulation-time, and SUMO
  version diagnostics. Failed resets close their connection.

## 6. Added reproducible comparison and handoff support

- Added `rl_agent/compare_rewards.py` for three rewards across three training seeds.
- Added a SUMO smoke-evidence gate and checks for compatible study settings/source.
- Added a run manifest, resume support, per-seed comparisons, and aggregate means
  and sample standard deviations across training seeds.
- Added source/scenario snapshots, SHA-256 hashes, dependency versions, and timing
  records to run artifacts.
- Each training run saves `best_model.zip`, `final_model.zip`, `config.json`,
  `train.monitor.csv`, `validation.json`, `results.json`, and a source snapshot.
- The comparison runner creates a handoff ZIP after its study completes. It does
  not automatically collect separately launched runs or send files to Member 3.
- Logs and models under `logs/` remain Git-ignored. They must be shared separately.
- Added `SUMO_HANDOFF.md` with commands, metric definitions, and reproduction notes.

## 7. Verified the pipeline and inspected real training results

Local verification previously passed 23 checks across training, the mocked SUMO
bridge, pipeline, comparison runner, and episode-length measurement, plus a toy
smoke run. Live SUMO execution took place on the teammate's SUMO machine.

The initial 1,000-step SUMO smoke run saved/evaluated models successfully, but its
policy switched once and left seven vehicles at the cap in the supplied episodes
(evaluation reward around -970). That short run established integration, not a
useful trained policy. Full 150,000-step queue runs subsequently improved strongly.

### Confirmed normal-scenario queue results

Each row evaluates the validation-selected best model on 30 held-out traffic seeds.
Larger reward (closer to zero) is better for this formula.

| Training seed | Best test reward | Final test reward | Fixed-cycle reward | Best-model completion |
|---|---:|---:|---:|---:|
| 42 | approximately -40.07 | approximately -40.43 | -89.00 | 30/30 |
| 43 | -46.60 | -55.83 | -89.00 | 30/30 |
| 44 | -46.77 | -51.63 | -89.00 | 30/30 |

Seed 42 is recorded from the results shared earlier in the conversation. Seeds
43 and 44 were also rechecked in the supplied `results.json` and `results1.json`.
All best-model test episodes completed with no teleports in the reviewed results.
Seed 43's final model capped one test episode, reinforcing why best and final
weights are saved separately.

Across the three best models, mean queue reward is approximately **-44.48**, versus
**-89.00** for the baseline: about **50% lower accumulated queue penalty** on this
normal scenario. This is not a claim of 50% lower vehicle travel time. The 90
evaluations reuse the same 30 traffic seeds across three models, rather than
representing 90 distinct demand scenarios.

The normal demand contains 13 trips and peak contains 20. Current results support
the tested small scenario; peak training and broader traffic generalization have
not been demonstrated. Peak episode-length measurements alone do not establish
peak learned-policy performance.

## 8. Added visual model replay

Added `rl_agent/replay.py` and README instructions, committed as `f0625d6`.

```powershell
python -m rl_agent.replay --model logs/YOUR_RUN/best_model.zip --seed 2000 --delay 300
```

The command loads a saved model and adjacent `config.json`, selects the saved
scenario/reward/cap, launches `sumo-gui`, and chooses deterministic actions without
training. Delay controls animation speed. It prints episode diagnostics and waits
for Enter in an interactive terminal before closing; `--no-wait` skips the pause.

Mock checks passed for completion, truncation, saved settings, inference-only
behavior, and cleanup. CLI help was checked. Actual GUI playback still needs to
be confirmed on the SUMO machine.

## Remaining work

1. Collect and verify the six wait/combined runs (seeds 42, 43, 44). Commands were
   provided for overnight execution; completion has not yet been reported here.
2. Compare all nine best checkpoints using common metrics: queue, lane waiting
   aggregate, switches, completion, remaining vehicles, and teleports. Raw reward
   totals from different formulas are not directly comparable.
3. Report means and variation across training seeds, retaining the per-episode
   records. `mean_wait` is a time-average of summed lane waiting aggregates, not
   average per-vehicle trip waiting time.
4. Assemble the nine run folders, models, logs, configs, source snapshots, and
   comparison table; send them to Member 3. Individually launched runs need to be
   collected explicitly rather than rerun through the full matrix by accident.
5. Confirm visual replay on the SUMO machine.
6. Decide whether peak-demand experiments and optional PPO fit the remaining
   project scope/time. Neither is counted as completed in this update.

Checkpoint 7 closes when all required reward runs and their logs are verified.
Checkpoint 9 closes when Member 3 actually receives the evaluation-ready artifacts.
