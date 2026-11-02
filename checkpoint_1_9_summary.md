# Member 2 checkpoint 1–9 summary

This is the current checkpoint status for the normal-scenario DQN reward
comparison. Historical details are in `checkpoint_1_5_summary.md` and
`checkpoint_5_9_summary.md`.

| # | Checkpoint | Status | What was done |
|---|---|---|---|
| 1 | State/action design | Complete | Member 1's SUMO environment exposes four vehicle counts, four queue counts, four lane waiting aggregates, and a phase index. The wrapper converts 13 raw values to 16 clipped/one-hot values. Actions are keep or request switch, with automatic yellow. |
| 2 | Reward design | Complete | Queue `-sum(queue)`, wait `-sum(wait)`, and combined `-sum(wait)/500 - beta × applied_switch` were defined and implemented. Beta=1 was the original combined setting; beta=0.02 was selected later on validation evidence. |
| 3 | Toy environment | Complete | A seeded four-direction queue simulator exercised the action, state, reward, and reset pipeline. Its learned weights are not part of the SUMO comparison. |
| 4 | Gymnasium wrapper | Complete | `Discrete(2)`, `Box(16)` observations, seeded reset, reward callback, and five-value step output are implemented. A 150-step cap applies to SUMO runs. |
| 5 | Toy DQN | Complete | Stable-Baselines3 DQN trained with validation-based model selection and separate held-out evaluation. Reward scaling and target-network settings were checked before moving to SUMO. |
| 6 | Real SUMO swap | Complete for normal traffic | The factory selects Member 1's TraCI environment. Live smoke, full training/evaluation, GUI replay, and junction diagnostics ran on the SUMO machine. |
| 7 | Reward variants trained | Complete for normal traffic | Queue, wait, original combined, and validation-tuned combined beta=0.02 each have three 150,000-step training runs and saved validation-selected models. The tuned combined models also have held-out results. |
| 8 | PPO | Optional, not run | The required comparison uses DQN. |
| 9 | Handoff to Member 3 | Pending delivery | The report and test JSONs are ready. Collect the three tuned model folders from Member 1, then transfer all models, logs, and results to Member 3. |

The same normal scenario and 30 held-out traffic seeds 1000–1029 were used
for the reported comparison. On those seeds, pure wait averaged 1.980 lane
waiting aggregate units per step and 8.47 switches per episode. Tuned combined
beta=0.02 averaged 2.221 and 5.94 respectively. All evaluated best models
completed every held-out episode in the recorded comparison. See
`reward_comparison_normal.md` for all variants, variability, and limits.

`mean_wait` is a time average of the sum of current lane waiting aggregates;
it is not mean trip delay. The normal demand file contains 13 trips, so the
results do not establish performance under heavier or sustained traffic.
Member 1 confirmed that a permissive lowercase `g` left turn may yield inside
the junction. The added internal stopped-time metric records that effect
separately; the network was not changed.

For reproducible artifacts, commands, and the remaining transfer, use
`MEMBER3_HANDOFF.md`. PPO and peak-demand experiments remain optional.
