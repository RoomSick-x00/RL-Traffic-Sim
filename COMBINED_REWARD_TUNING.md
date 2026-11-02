# Combined reward: validation-only beta experiment

## Follow-up order agreed on 28 October

### 1. Fill the shared validation table before any new training

Every row below uses traffic seeds 100–109. Learned rows use the checkpoint
selected by that run's validation reward; all are training seed 42.

| Controller | Mean lane waiting | Switches per episode |
|---|---:|---:|
| Fixed cycle | 9.359 | 4.0 |
| Pure wait | 2.077 | 6.4 |
| Combined beta=0.02 | 1.843 | 7.1 |
| Combined beta=0.05 | 4.048 | 1.6 |
| Combined beta=0.10 | 4.030 | 1.6 |

All five controllers completed all ten episodes. This table exposes the
tradeoff; it does not establish a universally best beta.

### 2. Match training seeds before trying more betas

Pure-wait seeds 43 and 44 already completed the same 150,000-step budget. Their
original validation-selected checkpoints are reusable; no duplicate training
is needed. Archived configs have matching dynamics and training settings; the
follow-up's code differences add beta configuration and validation-only output.

| Pure-wait training seed | Existing folder | Best step | Validation mean wait | Switches |
|---|---|---:|---:|---:|
| 42 | `dqn_iyz2gfcn` | 60,000 | 2.077 | 6.4 |
| 43 | `dqn_jx51d8jd` | 60,000 | 2.047 | 8.1 |
| 44 | `dqn_rvxs2gzr` | 40,000 | 1.908 | 11.5 |

After applying the updated code, only these two new training runs are needed now:

```cmd
python -m rl_agent.train --env sumo --scenario normal --reward combined --seed 43 --switch-penalty 0.02 --validation-only --run-dir logs/combined_beta_0p02_seed43
python -m rl_agent.train --env sumo --scenario normal --reward combined --seed 44 --switch-penalty 0.02 --validation-only --run-dir logs/combined_beta_0p02_seed44
```

Both default to 150,000 timesteps and cap 150. Send the resulting validation and
result files before selecting another candidate. Only after reviewing these
replications, try beta=0.03 and 0.04; do not expand the search beyond those two.

### 3. Add a defined proxy for potentially wasted switches

Evaluation now logs `empty_target_switches`: an actual green-to-yellow switch
when the target axis has zero incoming-lane vehicles while the source axis still
has a nonzero queue, measured immediately before the step. Vehicle counts, rather
than queue counts, avoid calling a target empty when moving vehicles are present.
The same rule applies to DQN and the native fixed cycle. Ignored yellow requests
are not counted. A switch with no queued vehicles on the source axis is not flagged.

Each episode records `switch_events` with step number, source/target phases,
source queue and target vehicle counts, plus the proxy flag. Aggregate outputs
include `mean_empty_target_switches` and `empty_target_switch_fraction` (total
flagged switches / total switches, null if there are no switches).

This is an operational proxy, not a causal count of wasted actions. It does not
see arrivals during yellow, future demand, or the alternative outcome had the
agent kept green. A zero count does not prove switching was efficient. Interpret
it alongside waiting, queues, completion and yellow fraction; do not add it to
the reward during this experiment.

Old logs do not contain pre-switch snapshots, so this metric cannot be reliably
reconstructed from their episode totals. Re-evaluate saved best models on
validation seeds, without retraining or touching test seeds:

```cmd
python -m rl_agent.evaluate_saved --run-dirs logs/dqn_iyz2gfcn logs/dqn_jx51d8jd logs/dqn_rvxs2gzr logs/combined_beta_0p02_seed42 logs/combined_beta_0p05_seed42 logs/combined_beta_0p1_seed42
```

Use the actual paths if the folders have different names. Each run also gets its
matching fixed-cycle evaluation. New files go into a unique
`logs/validation_diagnostics_*` directory, preserving the original logs. They
include common-metric differences from the original selected validation result,
model/evaluation-source hashes, current package versions and changed source names.
The command rejects changes to recorded environment/network/route files and has
no test-evaluation option. New training evaluations automatically include the
same diagnostic. Share these new JSON files along with the two replication runs.

### 4. Freeze before final evaluation

Freeze the chosen beta, training budget/settings, checkpoint-selection rule,
metric definitions and traffic scenario using validation evidence. The original
test seeds 1000–1029 were already inspected and motivated this follow-up; they
cannot now be described as an untouched test set. For a fresh final assessment,
reserve a new, disjoint traffic-seed set before running final evaluations, apply
it equally to all selected variants and the baseline, and do not tune afterward.
This still evaluates the same finite route demand, not a new demand distribution.

No new SUMO training or live validation evaluation was launched locally.

## Received results — 28 October 2026

All three seed-42 trials reached 150,000 timesteps. Their embedded configs,
versions, source hashes, validation seeds, and training settings match except
beta. Each result matches its paired validation history and best-checkpoint
selection. Episode rewards agree with `-wait_sum_over_steps/500 - beta*switches`.
The configs record validation-only execution and no test seeds, and the result
files contain no held-out test sections. Separate config files were not needed
for this analysis because each results file includes the config.

| Result / validation files | Beta | Best timestep | Mean queue | Mean lane waiting | Switches per episode |
|---|---:|---:|---:|---:|---:|
| `re_mp1.json` / `vl_mp1.json` | 0.02 | 90,000 | 0.893 | 1.843 | 7.1 |
| `re_mp2.json` / `vl_mp2.json` | 0.05 | 130,000 | 1.153 | 4.048 | 1.6 |
| `re_mp3.json` / `vl_mp3.json` | 0.10 | 120,000 | 1.146 | 4.030 | 1.6 |
| Shared fixed cycle | — | — | 1.625 | 9.359 | 4.0 |

These are the validation-selected best models on traffic seeds 100–109, not
held-out results or averages across multiple training seeds. All best and final
models and baselines completed all ten evaluation episodes with 13 arrivals,
no remaining demand, no truncations, and no teleports. Earlier checkpoints were
not uniformly successful: beta=0.1 completed only 4/10 episodes at both 10,000 and
30,000 timesteps. Selection therefore matters.

For a comparison on the same validation seeds, the original beta=1 seed-42 best
checkpoint had mean queue 1.249, mean lane waiting 5.434 and 1.0 switches. The
original pure-wait seed-42 best checkpoint had mean queue 0.992, mean lane waiting
2.077 and 6.4 switches. Those figures were read from the original archived
validation histories, not mixed with their held-out test averages.

Interpretation:

- Beta=0.02 reduces waiting by about 54.5% relative to beta=0.05, at the cost of
  increasing switching from 1.6 to 7.1. It switches 6–8 times in each episode.
- Beta=0.05 and 0.1 are very similar on these common metrics. Both switch once
  in seven episodes and three times in three episodes. The tiny difference in
  waiting does not establish a robust preference between them.
- Beta=0.02 is the candidate to replicate if reducing waiting is the priority.
  It has not demonstrated the intended reduction in switching compared with pure
  wait reward: the original pure-wait seed-42 model averaged 6.4 switches on the
  same validation seeds, versus 7.1 here.
- These results support sensitivity to the switch penalty. They do not prove the
  original one-switch model ignored its input state, nor establish an optimal beta.
- Keep the original nine-run study. Any beta=0.02 replication on seeds 43 and 44
  belongs to this follow-up; review validation stability before a final evaluation.

The three new model files themselves have not been supplied or loaded yet.
No held-out test evaluation was launched during this review.

## Experiment commands and protocol

This is a follow-up to the completed nine-run normal-scenario comparison. Keep
those original results and models. The reward remains `-sum(wait)/500 - beta *
applied_switch`, with alpha=1. Only beta changes; optimizer scale remains 1.0.

Use the updated code on the SUMO machine, from the project root. Run these one
at a time, initially using training seed 42 for all three candidates:

```cmd
python -m rl_agent.train --env sumo --scenario normal --reward combined --seed 42 --total-timesteps 150000 --max-steps 150 --switch-penalty 0.1 --validation-only --run-dir logs/combined_beta_0p1_seed42
python -m rl_agent.train --env sumo --scenario normal --reward combined --seed 42 --total-timesteps 150000 --max-steps 150 --switch-penalty 0.05 --validation-only --run-dir logs/combined_beta_0p05_seed42
python -m rl_agent.train --env sumo --scenario normal --reward combined --seed 42 --total-timesteps 150000 --max-steps 150 --switch-penalty 0.02 --validation-only --run-dir logs/combined_beta_0p02_seed42
```

An existing run directory is rejected rather than overwritten; use a new name
for any rerun. Without `--switch-penalty`, the original combined beta=1 remains
the default. Other rewards reject this option. Beta must be finite and nonnegative.

Each directory saves the models, monitor log, source snapshot, `config.json`,
`validation.json`, and `results.json`. The config records beta in
`reward_parameters.switch_penalty`, `validation_only: true`, and an empty
`test_seeds` list. Validation seeds remain 100–109. There is no evaluation on
held-out test seeds in these tuning runs.

Results contain `best_validation`, `final_validation`, and
`fixed_cycle_validation` instead of the usual test sections. Within each run,
the best checkpoint is still selected using that run's own validation reward.
For choosing between betas, compare common validation metrics (waiting, queue,
switches, completion, remaining vehicles, and teleports), not raw reward totals:
changing beta changes the reward objective. Report this as exploratory tuning;
one training seed is not evidence of a robust improvement. Examine the tradeoff
before locking a beta, then replicate with seeds 43 and 44. Keep held-out testing
for the finalized choice, and disclose that the original test results motivated
this follow-up investigation.

Send the three `config.json`, `results.json`, and `validation.json` sets for
review after the runs finish. Saved-model replay reconstructs beta from the
config; legacy configs without reward parameters still use beta=1.

Local verification covers reward persistence, legacy behavior, baseline switch
charging, CLI forwarding, and a short real DQN run on a mock environment proving
the validation-only path does not evaluate test seeds. Actual SUMO tuning runs
must take place on the SUMO machine. The commands train new models, not replay
the old ones.
