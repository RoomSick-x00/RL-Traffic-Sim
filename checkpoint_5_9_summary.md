# Checkpoints 5–9: DQN training and SUMO results

This is the current Member 2 record. The earlier
`checkpoint_1_5_summary.md` and `checkpoint_6_7_progress.md` retain the
development history; their older status and interface statements are superseded
by this file and `checkpoint_1_9_summary.md`.

| Checkpoint | Status | Evidence |
|---|---|---|
| 5 — first toy DQN | Complete | Gymnasium wrapper, reward logging, validation-selected checkpoints, and toy training on three seeds verified the pipeline. Toy weights are not SUMO results. |
| 6 — real SUMO swap | Complete for normal scenario | Live smoke run, normal-scenario training, GUI replay, and junction diagnostics succeeded on Member 1's SUMO machine. The real environment uses the same 13-value raw state and 2-action wrapper. |
| 7 — reward variants | Complete for normal scenario | Queue, pure wait, and original combined each trained at 150,000 steps on seeds 42/43/44. Tuned combined beta=0.02 was also trained on all three seeds. Validation selected the best checkpoint within each run. |
| 8 — PPO | Optional, not run | DQN supplies the required reward-function comparison. |
| 9 — Member 3 handoff | Packaged; pending delivery | The three tuned run folders were received and audited. A single archive containing all 12 runs, evaluations, and reports is ready at `logs/member3_normal_dqn_handoff_20261102.zip`; Member 3 receipt remains unconfirmed. |

## Fixed experimental setup

Normal SUMO demand has 13 trips. Each DQN run uses 150,000 training steps,
training seed 42/43/44, validation traffic seeds 100–109, and held-out test
traffic seeds 1000–1029. The SUMO wrapper caps an episode at 150 steps. Shared
optimizer settings are learning rate 5e-4, target update interval 1,000,
exploration fraction 0.3, and final epsilon 0.05. The model uses a 16-value
processed observation and keep/switch actions. Queue, wait, and combined
rewards have separate declared optimizer multipliers; compare traffic metrics,
not returns from different reward formulas.

The original combined switch weight beta=1 produced a one-switch policy.
The beta=0.02 value was chosen using validation evidence. Its three saved best
models were then evaluated on the fixed 30 held-out traffic seeds, without
another training run. The test evaluation produced 90/90 completed episodes,
zero truncations, teleports, and remaining vehicles.

| Controller | Mean total queue/step | Mean lane waiting aggregate/step | Switches/episode | Completion |
|---|---:|---:|---:|---:|
| Fixed cycle | 1.709 | 10.110 | 4.00 | 100% |
| Queue | 1.043 ± 0.095 | 3.727 ± 1.378 | 5.69 ± 1.74 | 100% |
| Pure wait | 0.974 ± 0.095 | 1.980 ± 0.365 | 8.47 ± 2.66 | 100% |
| Original combined beta=1 | 1.212 ± 0.000 | 4.974 ± 0.000 | 1.00 ± 0.00 | 100% |
| Tuned combined beta=0.02 | 0.962 ± 0.060 | 2.221 ± 0.165 | 5.94 ± 0.68 | 100% |

Each learned entry is the mean and sample standard deviation across three
training-seed means; each model saw the same 30 held-out traffic seeds. This is
three trained models per variant, not 90 independent demand patterns. Pure
wait has the lowest observed waiting. Tuned combined switches less than pure
wait while accepting slightly more waiting. The tiny queue difference between
those two is not enough to establish a reliable advantage. The fixed demand is
small, so these numbers do not establish performance under sustained traffic.

GUI replay and the junction trace confirmed permissive `g` left turns can
yield inside the intersection. Incoming-only waiting misses this delay. The
validation-only junction diagnostic recorded 0–4.9 mean internal stopped
vehicle-seconds per episode across the compared controllers. Member 1 confirmed
that this priority rule is intended; no network edit or retraining followed.

## Evidence and next handoff

- `reward_comparison_normal.md` has the detailed original and tuned comparison.
- `COMBINED_REWARD_TUNING.md` records validation-based beta selection and seed matching.
- `JUNCTION_DIAGNOSTIC_CHECKS.md` defines the inside-junction measurement.
- `FINAL_HELDOUT_EVALUATION.md` records the test-only command.
- The supplied `test_evaluations_dc77ef1e.zip` holds the three tuned held-out JSON results. The original nine models are in `dqn_all.zip` and the earlier handoff archive. These archives are external artifacts, not Git files.
- `MEMBER3_HANDOFF.md` identifies the final transfer archive and the limits to report.

Checkpoint 9 closes only when Member 3 receives the transfer archive. Peak-demand
training and PPO remain optional future experiments.
