# Junction waiting checks — no retraining

Use the updated project code on the SUMO machine. Keep the existing environment,
network, routes, model files and rewards. Diagnostics only read simulation state.

## 1. Replay the observed case and save its trace

From the project root in CMD:

```cmd
python -m rl_agent.replay --model logs/dqn_s782alz3/best_model.zip --seed 2000 --delay 1000 --junction-diagnostics --no-wait
```

This opens the queue seed-42 model in SUMO GUI, then automatically writes
`logs/junction_replay_2000_<unique-id>.json`. No output redirection or manual
terminal copying is required. The file contains the episode summary and:

- Per-step vehicle IDs, lane IDs, speeds and stopped flags for the incoming roads
  and this junction's internal lanes.
- Each observed vehicle's route edges, cached on its first observation (routes
  are fixed in this experiment).
- Signal state, phase before/after the step, action, and simulation time.
- Controlled connections indexed by signal index, to interpret `G`/`g` and route
  movements together. The signal string alone does not establish all priority
  relations; use the unchanged network's junction request rules too.

Member 1 should match the vehicle that waits inside the intersection to its ID
and route in this trace, then confirm the intended yielding behavior. The JSON
records what happened; it does not decide whether the network design is intended.

## 2. Compare the four saved controllers plus fixed cycle

Run this command once, after step 1 finishes:

```cmd
python -m rl_agent.evaluate_saved --junction-diagnostics --run-dirs logs/dqn_s782alz3 logs/dqn_iyz2gfcn logs/dqn_o_tzn83t logs/combined_beta_0p02_seed42
```

These are queue, pure wait, original combined beta=1, and combined beta=0.02,
all with training seed 42. Each uses its saved validation seeds (100–109 in these
runs). Fixed cycle is also evaluated for each run using that run's reward formula;
its common traffic metrics should agree across runs. This does not train models
or evaluate test seeds. If the folders are nested elsewhere, adjust the paths.

New output is saved to a printed `logs/validation_diagnostics_<unique-id>/`
directory. Each result JSON includes model and baseline summaries and per-episode
traces. Original run files are not overwritten. Zip the entire new directory
along with the replay JSON and send them for review.

## Measurements and interpretation

The three aggregate fields under `best_validation.junction_diagnostics` and
`fixed_cycle_validation.junction_diagnostics` are:

- `mean_incoming_stopped_vehicle_seconds`
- `mean_internal_stopped_vehicle_seconds`
- `mean_total_stopped_vehicle_seconds`

Each is a mean of episode totals. Within an episode, the measurement is the sum
of each lane's last-step halting count multiplied by SUMO's step duration.
SUMO considers speed below 0.1 m/s halting. This uses step duration in seconds,
so e.g. two halted vehicles for 0.5 seconds contribute one vehicle-second.
It does not repeatedly sum accumulated vehicle waiting counters.

Internal lanes are identified by this single junction's `:<traffic_light_id>_`
prefix. Every such lane, including the continuation after an internal waiting
point, is included once. Incoming lanes are the four named in the raw environment.
The logs exclude the initial step inside reset, outgoing roads and vehicles not
yet inserted. Thus these are sampled stopped-time measurements over the chosen
regions, not complete trip delay or network-wide travel time. A vehicle joining
or leaving the region within a step introduces the usual sampling limitation.

See the [SUMO lane retrieval documentation](https://sumo.dlr.de/docs/TraCI/Lane_Value_Retrieval.html)
for the halting-count definition.

Compare internal stopped time and the combined incoming+internal total alongside
completion, remaining vehicles, teleports, episode length, switches and the
existing waiting/queue metrics. Check `difference_from_original_validation`;
material differences in the old metrics need investigation before interpreting
the added measurements. Internal stopped time is not directly subtractable from
the old `wait_sum_over_steps`: those quantities have different definitions.

The existing empty-target-switch proxy is included too. Its `switch_events` can
be inspected alongside the new trace to see whether target traffic arrives during
the yellow interval. No reward, observation or signal change is part of this check.

Live SUMO execution is required on Member 1's machine. Local mock tests check
lane scope, duration units, per-episode aggregation, automatic trace saving,
cleanup and preservation of existing evaluation metrics.
