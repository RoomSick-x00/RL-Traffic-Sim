# RL Traffic Signal Simulation

Member 2's reinforcement-learning pipeline compares DQN traffic-signal
controllers in Member 1's single-junction SUMO environment. The normal-scenario
reward study is complete; the Member 3 artifact transfer is pending. See
[`checkpoint_1_9_summary.md`](checkpoint_1_9_summary.md) for checkpoint status,
[`reward_comparison_normal.md`](reward_comparison_normal.md) for results, and
[`MEMBER3_HANDOFF.md`](MEMBER3_HANDOFF.md) for the files Member 3 needs.
The complete 12-run transfer archive is prepared under Git-ignored
`logs/member3_normal_dqn_handoff_20261102.zip` and must be sent directly.

## Environment and agent

The raw SUMO state has 13 values: N/S/E/W vehicle counts, queues, lane waiting
aggregates, then the signal phase. `GymnasiumWrapper` divides counts by 16,
queues by 10, and waiting by 400 with clipping to [0, 1], then one-hot encodes
the four phases. Its 16-value observation uses `Box(0, 1)` and its two actions
are keep (0) or request switch (1). The SUMO environment handles the automatic
yellow phases. The wrapper caps normal episodes at 150 steps; completion of
the demand scenario is termination.

Three reward formulas were compared: queue `-sum(queue)`, wait `-sum(wait)`,
and combined `-sum(wait)/500 - beta × applied_switch`. The original combined
setting used beta=1. A later validation-only tuning pass selected beta=0.02.
Their training multipliers are recorded in each run's `config.json`; raw
returns from different formulas are not comparable across variants.

The normal scenario contains 13 trips. Three training seeds (42–44) and 30
held-out traffic seeds (1000–1029) were used. The tuned combined controller
averaged 5.94 switches and 2.221 summed lane waiting aggregate units per step;
pure wait averaged 8.47 and 1.980. All recorded best-model held-out episodes
completed. These are finite-demand results, not evidence for sustained traffic.

## Run and reproduce

Use Python with NumPy, Gymnasium, Stable-Baselines3/PyTorch, and SUMO's TraCI
client. The `sumo` executable must be on PATH for the real environment.
Commands below run from the repository root.

```cmd
python -m rl_agent.train --env sumo --scenario normal --reward queue --smoke
python -m rl_agent.replay --model logs/YOUR_RUN/best_model.zip --seed 2000 --delay 300
python -m rl_agent.evaluate_saved --split test --run-dirs logs/combined_beta_0p02_seed42 logs/combined_beta_0p02_seed43 logs/combined_beta_0p02_seed44
```

The smoke command checks the live training pipeline; the replay command opens
SUMO GUI and uses saved weights; the final command evaluates saved best models
on the fixed test seeds without training. For a fresh full experiment, use
[`SUMO_HANDOFF.md`](SUMO_HANDOFF.md). For junction traces, use
[`JUNCTION_DIAGNOSTIC_CHECKS.md`](JUNCTION_DIAGNOSTIC_CHECKS.md).

The toy simulator in `rl_agent/toy_env.py` is for pipeline checks. Its model
weights are not used in the SUMO results. Run local code checks with:

```cmd
python -m unittest rl_agent.test_training rl_agent.test_pipeline rl_agent.test_comparison rl_agent.test_junction_diagnostics rl_agent.test_reward_tuning -v
```

Training creates per-run folders under Git-ignored `logs/`. A complete model
handoff needs the folders themselves, including `best_model.zip`, `config.json`,
and `source/`; a Git pull or JSON summary alone does not contain trained weights.
