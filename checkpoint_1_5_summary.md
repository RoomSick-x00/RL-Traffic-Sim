# RL Traffic Signal Project — Checkpoints 1–5 Summary
**Member 2 (RL Agent Design & Training) — handoff notes for Member 3**

---

## Checkpoint 1: State & Action Space

**Confirmed interface from Member 1's environment (`traffic_env.py`):**
- `state = env.reset()`
- `next_state, reward, done = env.step(action)`
- SUMO step length: **1 second** (`traci.simulationStep()` called once per step)
- Yellow phases last 3 seconds in SUMO, and vehicles close to the junction may legally pass through yellow (confirmed via SUMO docs + `test.net.xml`)
- Episode ends when `traci.simulation.getMinExpectedNumber() <= 0` (demand scenario exhausted)

**Raw state — 13 values:**
```
[N_veh, S_veh, E_veh, W_veh, N_queue, S_queue, E_queue, W_queue, N_wait, S_wait, E_wait, W_wait, phase]
```
- `queue` = `traci.lane.getLastStepHaltingNumber()` (vehicles with speed < 0.1 m/s)
- `wait` = `traci.lane.getWaitingTime()` — a **lane-level aggregate across currently-present vehicles**, not per-vehicle and not cumulative over the whole episode
- `phase`: 0 = green NS, 1 = yellow NS, 2 = green EW, 3 = yellow EW; cycle is 0→1→2→3→0

**Action space:** `Discrete(2)` — 0 = keep current phase, 1 = request phase change (switch always passes through the yellow phase first; agent does not pick phases directly).

**Preprocessing (done outside the raw env, in the Gymnasium wrapper) — raw 13 → processed 16-dim:**
- Vehicle count ÷ 16 (normalization bound; see below), clipped to 1.0
- Queue length ÷ 10 (derived from 92.8 m lane length and ~9 vehicles observed max queue under a stress test with demand 25→40), clipped to 1.0
- Waiting time ÷ 400 (derived from ~9 vehicles queued × staggered per-car wait, with margin for partial-clear-then-refill scenarios), clipped to 1.0
- Phase: **one-hot encoded** (4 values), not scalar-divided — phase is categorical, not ordinal, so dividing by 3 would falsely imply phase 3 is "more" than phase 0

**Reward:** explicitly decoupled from the environment. The raw env never computes reward — reward is computed in training code so multiple reward formulations can be tested against an identical environment.

---

## Checkpoint 2: Reward Function Variants

Three variants designed, each with a known behavioral weakness:

1. **Queue-based:** `reward = -Σ(queue_length)`
   Fast reaction to congestion, but blind to *duration* — can let a low-traffic lane starve indefinitely if a high-traffic lane keeps refilling, since it only reacts to momentary count.

2. **Wait-time-based:** `reward = -Σ(wait_time)`
   Naturally resists starvation, since an ignored car's contribution to the sum grows every step it's ignored (unlike queue-count, which doesn't reflect duration). Weakness: has no notion of *switching cost* — a phase switch always passes through a yellow transition (dead time, no vehicles clear), so chasing the single oldest delay can trigger avoidable extra switches instead of clearing a larger, already-green queue first.

3. **Combined:** `reward = -(α × Σwait_time) - (β × switch_penalty)`, where `switch_penalty = 1` when `action == 1`, else `0`.
   Balances congestion against switching cost. **Important:** α and β must be kept in comparable scale — if β dominates (e.g., β ≫ α), the agent learns to never switch at all, since the bounded wait-time penalty (capped ~400) can never outweigh a disproportionately large switch cost, leading to one direction getting permanent green.

A separate `max(waiting_time)` "starvation-specific" term was considered and rejected — wait-time-sum already captures starvation on its own (a stuck car's contribution compounds every step it's ignored), so no separate rule is needed.

---

## Checkpoint 3: Toy/Mock Environment

Built `rl_agent/toy_env.py` — `ToyTrafficEnv`, a plain Python class (no Gymnasium inheritance), matching Member 1's interface shape (`reset()` → state, `step(action)` → `(state, done)`) so it can be swapped for the real SUMO env later with minimal changes.

**Internal model (all per-direction, N/S/E/W, stored as dicts):**
- `queues`: starts at 0; green phase → 60% chance to clear one car (`max(0, queue-1)`); red/yellow → 40% chance one car arrives (`queue+1`)
- `waiting`: resets to 0 when that direction's axis is green; otherwise accumulates `+= current queue value` each step (an approximation of the real aggregate wait-time signal — captures both "how many cars" and "how long")
- `phase`: starts at 0; `_update_phase()` enforces the real cycle — yellow phases (1, 3) always auto-advance to the next green regardless of action; green phases (0, 2) only move to yellow when `action == 1`
- `step_count`: increments each step; episode truncates (not terminates — nothing in the toy env represents genuine task completion) at 500 steps

**`get_state()`** builds the 13-value vector by reusing queue values for both the "vehicle count" and "queue" slots (the toy env doesn't model a separate vehicle-count quantity — a deliberate, acknowledged simplification).

**Known bug fixed during testing:** `random.choices()` returns a list (e.g. `[1]`, not `1`), which silently failed all `action == 1` comparisons in `_update_phase`, causing the phase to never switch. Fixed by indexing `[0]` into the result.

---

## Checkpoint 4: Gymnasium Wrapper

Built `rl_agent/gym_wrapper.py` — `GymnasiumWrapper(gym.Env)`, which wraps *any* raw env matching the toy env's interface (so the same wrapper will later wrap Member 1's real env too).

- `action_space = Discrete(2)`
- `observation_space = Box(low=0.0, high=1.0, shape=(16,), dtype=np.float32)` — bounds enforced by clipping during normalization (values that exceed expected physical bounds, e.g. during a runaway/bugged state, are capped at 1.0 rather than given extra headroom)
- `_process_state(raw_state)`: splits the 13 raw values, normalizes/clips each group, one-hot encodes phase, returns the 16-dim array
- `reset()` → calls raw env's `reset()`, processes state, returns `(state, info)` per Gymnasium's 2-value spec
- `step(action)` → calls raw env's `step(action)`, processes state, maps the toy env's `done` to **`truncated`** (an arbitrary step-count cutoff, not a true task-completion signal, so `terminated` is always `False`), returns the Gymnasium 5-tuple `(state, reward, terminated, truncated, info)`
- `reward_fn` is passed into the wrapper's constructor, so swapping between the Checkpoint 2 reward variants requires no change to the wrapper or environment code — only which function is passed in from the training script

**Determinism fix:** `check_env()` (Gymnasium's built-in validator) initially failed on step-determinism, because `ToyTrafficEnv` used Python's global, unseeded `random.random()`. Fixed by giving the toy env its own seedable `random.Random()` instance, seeded via `reset(seed=...)`, forwarded through the wrapper. This matters for fair comparison between reward variants/training runs — same seed should reproduce the same traffic conditions.

---

## Checkpoint 5: First DQN Agent (Toy Env)

**Reward functions implemented in `rl_agent/rewards.py`:** `queue_reward(raw_state)` and `wait_reward(raw_state)`, both operating on the raw 13-value state (the combined reward with switch penalty is not yet implemented — it needs the action, not just the state, so it requires a small extra change to the wrapper and is deferred to a later checkpoint).

**Initial training issue:** reward was hardcoded to `0.0` in the wrapper (a leftover placeholder) — agent had nothing to learn from (`ep_rew_mean` stuck at exactly 0 across all 50k steps). Fixed by wiring `reward_fn` through.

**Debugging the training curve, in order:**
1. With `queue_reward` wired in and SB3 DQN defaults, `ep_rew_mean` improved briefly then plateaued around -3.6e4 to -3.7e4. Root cause: SB3's default `exploration_fraction=0.1` meant exploration decayed to its floor (0.05) within ~8 episodes (5,000 of 50,000 steps) — far too little time to discover a good policy before mostly locking into exploitation.
2. Raised `exploration_fraction` to 0.3 — reward improved further (best ~-2.58e4) but then **degraded** over continued training even as exploration stayed at its floor, confirmed via averaging the first 20 vs. last 20 logged episodes of a longer (150k-step) run (~23% worse on average by the end) — a real trend, not noise.
3. Final diagnosis and fix (validated with a controlled 2×2 ablation — reward scaling × target-update-interval, isolated before combined, per training seed 42, then replicated on seeds 43/44):
   - **Reward scaling**: training reward divided by 0.01 before reaching DQN (applied *after* `Monitor` logs the true, unscaled reward — so all logged/evaluated numbers stay in real units; only what the learning algorithm sees internally is scaled). Reward magnitudes here (tens of thousands) are unusually large for DQN's default settings, which can destabilize training.
   - **`target_update_interval`**: lowered from SB3's default (10,000) to 1,000, for more frequent target-network stabilization.
   - **`learning_rate`**: raised to `5e-4` (tested against `1e-5` on seed 43; `5e-4` validated best).
   - Proper **train/validation/test split** added: the model is evaluated every 10,000 steps (no exploration) on 10 fixed validation seeds; the best-scoring checkpoint (`best_model.zip`) and the final checkpoint (`final_model.zip`) are both saved, since the final weights can be worse than an earlier checkpoint. The winning configuration was then independently tested on 30 held-out test seeds.

**Final validated results (best checkpoint, 30 test episodes, closer to 0 is better):**

| Training seed | Avg. test reward |
|---|---|
| 42 | −5,674.9 |
| 43 | −5,696.7 |
| 44 | −6,835.0 |
| **Overall avg.** | **−6,069** |

Compared against:
- **Random actions:** −37,638
- **Fixed-cycle baseline:** −6,643

**The trained DQN agent beats the fixed-cycle baseline by roughly 9%**, and dramatically outperforms random actions — confirming real learning occurred. Some variance across seeds (seed 44 noticeably worse) is expected, given the environment's built-in stochastic arrivals/clearing.

**Final training configuration locked in:**
```python
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=5e-4,
    target_update_interval=1000,
    exploration_fraction=0.3,
    exploration_final_eps=0.05,
    seed=seed,
    device="cpu",
)
model.learn(total_timesteps=150000)
```
with training-time reward scaled by 0.01 (applied after logging, not before).

---

## Project structure so far

```
RL-Traffic-Sim/
├── test/                  # Member 1's folder — SUMO configs, traffic_env.py, scratch scripts
├── rl_agent/
│   ├── toy_env.py         # ToyTrafficEnv (Checkpoint 3)
│   ├── gym_wrapper.py     # GymnasiumWrapper (Checkpoint 4)
│   ├── rewards.py         # queue_reward, wait_reward (Checkpoint 2/5)
│   ├── train.py           # launcher
│   ├── training.py        # training implementation, experiment/eval machinery
│   └── test_training.py   # tests for reward scaling, logging order, repeatable evaluation
├── logs/                  # per-run experiment outputs (config, monitor csv, checkpoints) — gitignored
├── .gitignore
└── README.md
```

## Open items / not yet finalized
- Normalization bounds (16 / 10 / 400) are evidence-based approximations, not final — may be revisited once more SUMO demand scenarios are tested
- Max episode step cap for the **real** SUMO env is not yet finalized (will be set after measuring real scenario duration)
- Combined reward variant (wait-time + switch penalty) is designed (Checkpoint 2) but not yet implemented in code — needs the wrapper to pass `action` into the reward function, not just `raw_state`
- `wait_reward` has not yet been run through the tuned training setup — only `queue_reward` has full validated results so far
- Toy env's results are for pipeline validation only and will be discarded once Checkpoint 6 swaps in Member 1's real SUMO environment
