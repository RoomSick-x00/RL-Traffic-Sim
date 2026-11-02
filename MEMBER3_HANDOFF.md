# Member 3 handoff: normal SUMO reward comparison

## What to collect

| Artifact | Contents | Current location/status |
|---|---|---|
| Original nine DQN runs | Queue, pure wait, original combined beta=1; training seeds 42/43/44 | `dqn_all.zip` supplied earlier; older `logs/member3_handoff_20261027_235257.zip` also covers these runs |
| Tuned combined runs | `combined_beta_0p02_seed42`, `combined_beta_0p02_seed43`, `combined_beta_0p02_seed44` | Ask Member 1 for **all three full folders**; their JSON results alone are insufficient to load the models |
| Tuned held-out results | Three JSON files for the saved best models on traffic seeds 1000–1029 | `test_evaluations_dc77ef1e.zip` supplied to Member 2 |
| Junction validation diagnostics | Four controllers plus fixed cycle, traffic seeds 100–109 | `validation_diagnostics_6b9301d9.zip` supplied to Member 2 |
| GUI replay trace | Queue model, traffic seed 2000 | `junction_replay_2000_f521a7c1.json` supplied to Member 2 |
| Comparison and provenance | Tables, formulas, run settings, caveats | `reward_comparison_normal.md`, `checkpoint_1_9_summary.md`, and this file in Git |

The named ZIP/JSON experiment files are outside Git because model and log
outputs are ignored. Do not expect `git pull` to download them. The Member 3
delivery is still pending until the tuned model folders and these results are
actually transferred.

## Run folder contract

Each training folder should contain `best_model.zip`, `final_model.zip`,
`config.json`, `validation.json`, `results.json` (for original runs),
`train.monitor.csv`, and `source/`. The tuned beta=0.02 runs intentionally used
`--validation-only`; their `results.json` contains `best_validation` and
`final_validation`, while the separate `test_evaluations_dc77ef1e.zip` supplies
`best_test`. Use the validation-selected `best_model.zip` for comparisons.

All models use 13 raw state values, 16 normalized observations, and two actions:
0 keeps the phase, 1 requests a switch through yellow. The frozen normal study
uses 150,000 training steps, 150-step episode cap, training seeds 42/43/44,
validation traffic seeds 100–109, and test traffic seeds 1000–1029. Shared DQN
settings: learning rate 5e-4, target update interval 1,000, exploration
fraction 0.3, and final epsilon 0.05. The run's `config.json` records its
reward formula, optimizer multiplier, software versions, and source hashes.

## How to evaluate and report

Read `reward_comparison_normal.md` for the verified held-out table. Treat the
three training-seed means as three model replications; the same 30 traffic
seeds are reused for each model. Compare mean queue, mean lane waiting
aggregate, physical switch count, completion, truncation, remaining vehicles,
and teleports. Different reward functions have different numerical scales, so
do not rank them by raw reward totals. `mean_wait` is a time average of summed
lane aggregates, not average trip or per-vehicle waiting time.

The tuned combined beta=0.02 models averaged 2.221 mean lane waiting and 5.94
switches per held-out episode; pure wait averaged 1.980 and 8.47. This is a
wait-versus-switch tradeoff, not proof of universal superiority. The 13-trip
normal scenario does not establish peak or sustained-traffic performance.
The optional internal-junction metric measures stopped vehicle-seconds inside
the junction on validation episodes; it is a distinct quantity from SUMO's
lane waiting aggregate. A permissive left turn may yield inside the junction.

On a SUMO machine with the complete run folders, the test-only evaluator is:

```cmd
python -m rl_agent.evaluate_saved --split test --run-dirs logs/combined_beta_0p02_seed42 logs/combined_beta_0p02_seed43 logs/combined_beta_0p02_seed44
```

This loads saved weights and writes new `logs/test_evaluations_*/` JSON files.
It does not train or select a different checkpoint. The existing held-out
results already satisfy this step; use the command only for reproduction.
