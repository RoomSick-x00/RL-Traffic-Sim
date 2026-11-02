# Normal SUMO scenario: completed DQN reward comparison

Verified 27 October 2026. This updates the experiment status in
[checkpoint_6_7_progress.md](checkpoint_6_7_progress.md).

## Completion and verification

All nine required normal-scenario DQN runs now have supplied results: queue,
wait, and combined rewards, each trained with seeds 42, 43, and 44 for 150,000
timesteps. Checkpoint 7's training/results requirement is complete. The subsequent
`dqn_all.zip` delivery contains all nine complete run folders. All 18 best/final
model files loaded successfully on CPU, with finite parameters, matching spaces
and timesteps, and valid predictions on four synthetic phase observations.
The monitor logs are present and the source snapshots match their recorded hashes.
Live SUMO reproduction remains pending; loading and synthetic predictions do not
establish that recorded episode results reproduce.

The six new result files match their validation files by the maximum validation
reward and its first occurrence (the selected best step). Each validation file
contains the 15 scheduled evaluations plus a final evaluation at 150,000 steps.
That repeated final timestep is expected from the current evaluation code.

Across all nine runs, recorded settings, dependency versions, and source/scenario
hashes match except the intended training seed, reward function, and reward scale.
All use normal traffic, cap 150, learning rate 0.0005, target update interval
1,000, exploration fraction 0.3, final epsilon 0.05, validation seeds 100–109,
and held-out test seeds 1000–1029. Source hashes establish what the configs record;
the complete original run folders remain the reproduction artifacts.

The six new files' per-episode returns agree with the configured wait/combined
formulas, and their reported mean returns agree with the episode records.
All nine best models completed 30/30 test episodes without truncation, remaining
vehicles, or teleports. The six newly supplied best-model evaluations each record
13 departures and 13 arrivals per episode.

## File mapping

| Reward | Training seed | Result file | Validation file | Selected timestep |
|---|---:|---|---|---:|
| Wait | 42 | `re1.json` | `val1.json` | 60,000 |
| Wait | 43 | `res2 .json` | `val2.json` | 60,000 |
| Wait | 44 | `re3.json` | `val3.json` | 40,000 |
| Combined | 42 | `res4.json` | `val4.json` | 100,000 |
| Combined | 43 | `res5.json` | `val5.json` | 70,000 |
| Combined | 44 | `res6.json` | `val6.json` | 50,000 |

These six pairs were supplied in the user's Desktop/results folder. Queue seed
42 was recovered from its earlier attached results JSON; queue seeds 43 and 44
were rechecked in Downloads/results.json and Downloads/results1.json.

## Comparison using shared metrics

Each model is the checkpoint selected on validation reward, then evaluated on
the same 30 test traffic seeds. Entries below are means of the three training-seed
means, with sample standard deviation across those three means. Lower is better
for queue and waiting; fewer switches describe a behavioral tradeoff, not an
unconditional measure of quality.

| Controller/reward | Mean total queue per step | Mean total lane waiting per step | Switches per episode | Completion |
|---|---:|---:|---:|---:|
| Fixed cycle | 1.709 | 10.110 | 4.00 | 100% |
| Queue | 1.043 ± 0.095 | 3.727 ± 1.378 | 5.69 ± 1.74 | 100% |
| Wait | 0.974 ± 0.095 | 1.980 ± 0.365 | 8.47 ± 2.66 | 100% |
| Original combined (beta=1) | 1.212 ± 0.000 | 4.974 ± 0.000 | 1.00 ± 0.00 | 100% |
| Tuned combined (beta=0.02) | 0.962 ± 0.060 | 2.221 ± 0.165 | 5.94 ± 0.68 | 100% |

The fixed-cycle row is the shared baseline evaluated on the same 30 traffic seeds;
it is not three independent trained models. `Mean total lane waiting` is a
time-average of the sum of lane waiting aggregates, not average trip waiting time.
The tuned combined row is a subsequent three-seed experiment selected using
validation seeds 100–109. Its held-out evaluation used the same test seeds
1000–1029 and saved best checkpoints. All 90 tuned-model test episodes had zero
truncations, teleports, and remaining vehicles. These results are in
`test_evaluations_dc77ef1e.zip`; its fixed-cycle metrics match the baseline above.

For completeness, accumulated episode signals averaged across test episodes and
training seeds are:

| Controller/reward | Queue sum over steps | Lane-waiting sum over steps |
|---|---:|---:|
| Fixed cycle | 89.000 | 526.833 |
| Queue | 44.478 | 158.667 |
| Wait | 43.278 | 88.522 |
| Original combined (beta=1) | 51.033 | 210.033 |
| Tuned combined (beta=0.02) | 41.300 | 95.211 |

The waiting sum repeatedly samples a lane aggregate; it must not be relabeled
total vehicle delay. Compare all variants through these common metrics rather
than their differently scaled raw reward totals.

## Interpretation

- Wait reward has the lowest observed mean waiting and mean queue in this study,
  with the most switching. Its small queue advantage over queue reward is an
  observed difference, not an established statistically significant advantage.
- Original combined beta=1 reduces switching to exactly one per episode in all evaluated
  best models. It accepts higher queues and waiting, while still clearing demand
  and improving both measures relative to the fixed cycle.
- Tuned combined beta=0.02 gives much less waiting than the original combined
  beta=1, with more switching. Compared with pure wait, it switches less on
  average (5.94 versus 8.47) while mean lane waiting is slightly higher
  (2.221 versus 1.980). Its tiny observed queue advantage over pure wait is
  not evidence of a reliable advantage with only three training seeds.
- Queue reward falls between wait and combined on waiting and switching.
- All three combined best models have identical held-out episode results. Their
  training seeds, validation histories, and selected timesteps differ. The logs
  are consistent with the models learning the same evaluated behavior; they do
  not prove the saved network weights are identical.
- Wait seed 42's final model scores better on the held-out test set than its
  validation-selected best model. Retain the validation-selected checkpoint for
  the comparison; choosing retrospectively using test results would bias it.

These are three training seeds evaluated on the same 30 traffic seeds, not 270
independent demand scenarios. The normal route file has only 13 trips. A one-switch
policy can clear this finite demand; sustained or peak traffic behavior has not
been established by these results. The study compares each declared reward with
its declared optimizer multiplier (queue 0.01, wait 0.00025, combined 1.0).

## Remaining handoff work

1. Run live SUMO evaluation reproduction using the received models and their saved
   environment, code, and dependency settings. Local model loading checks have
   passed; the local NumPy version differs from the recorded training version.
   Confirm GUI replay separately if it has not already been demonstrated.
2. The prepared handoff ZIP under `logs/member3_handoff_20261027_235257.zip`
   contains the original nine runs. Add the three tuned combined run folders
   and their held-out evaluation JSONs to the final Member 3 handoff. Logs and
   model artifacts remain Git-ignored.
3. Provide this comparison and the artifacts to Member 3 for plots, report, and
   further evaluation. Checkpoint 9 remains pending actual delivery.
4. Optional PPO and additional peak-demand training remain separate scope decisions.

The tuned beta=0.02 runs were trained separately after the original nine-run
study. No model was retrained using held-out test results.
