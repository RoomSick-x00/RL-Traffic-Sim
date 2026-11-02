# Final held-out evaluation: combined beta=0.02

> Completed on Member 1's SUMO machine. The three output JSONs were supplied in
> `test_evaluations_dc77ef1e.zip` and summarized in
> [reward_comparison_normal.md](reward_comparison_normal.md). The command below
> is retained for reproduction.

The combined beta=0.02 configuration is frozen after validation on traffic
seeds 100–109 and training seeds 42, 43, and 44. The three best models were
chosen using validation reward. Do not retrain or tune beta using the test
results.

On Member 1's SUMO machine, pull the latest code and run from the project root
in Command Prompt:

```cmd
git pull origin main
python -m rl_agent.evaluate_saved --split test --run-dirs logs/combined_beta_0p02_seed42 logs/combined_beta_0p02_seed43 logs/combined_beta_0p02_seed44
```

The command loads each run's `best_model.zip`, its saved reward configuration,
and the fixed held-out traffic seeds 1000–1029. It evaluates the fixed-cycle
controller for comparison, and saves three JSON files in a newly printed
`logs/test_evaluations_<id>/` folder. It does not train any model.

Send the entire output folder and retain all three original run folders,
including their `best_model.zip`, `config.json`, `results.json`, and
`validation.json` files. Compare test metrics on the same traffic seeds with
the already completed queue and pure-wait runs. Rewards from different reward
functions have different units, so compare traffic metrics such as queue,
lane waiting, switches, completion, remaining vehicles, and teleports.
