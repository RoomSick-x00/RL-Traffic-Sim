"""Train DQN with interchangeable traffic environments and unscaled evaluation rewards."""

import argparse
import json
from pathlib import Path
import random
import tempfile
import hashlib
from importlib.metadata import PackageNotFoundError, version
import shutil
import sys
from time import perf_counter

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

if __package__:
    from .gym_wrapper import GymnasiumWrapper
    from .rewards import queue_reward, wait_reward, combined_reward
    from .toy_env import ToyTrafficEnv
    from .env_factory import EnvironmentFactory
else:
    from gym_wrapper import GymnasiumWrapper
    from rewards import queue_reward, wait_reward, combined_reward
    from toy_env import ToyTrafficEnv
    from env_factory import EnvironmentFactory


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VALIDATION_SEEDS = tuple(range(100, 110))
TEST_SEEDS = tuple(range(1000, 1030))
DEFAULT_REWARD_SCALE = 0.01
DEFAULT_WAIT_REWARD_SCALE = 0.00025
DEFAULT_TARGET_UPDATE_INTERVAL = 1000
DEFAULT_LEARNING_RATE = 5e-4
REWARDS = {"queue": queue_reward, "wait": wait_reward, "combined": combined_reward}


def save_sources(run_dir):
    """Keep the exact code and scenarios alongside models, including uncommitted edits."""
    files = [path for path in (PROJECT_ROOT / "rl_agent").glob("*.py")
             if path.name != "tempCodeRunnerFile.py"]
    files += [PROJECT_ROOT / "test" / name for name in ("__init__.py", "traffic_env.py")]
    files += list((PROJECT_ROOT / "test").glob("*.xml"))
    files += list((PROJECT_ROOT / "test").glob("*.sumocfg"))
    hashes = {}
    for source in files:
        relative = source.relative_to(PROJECT_ROOT)
        destination = run_dir / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[relative.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    for name in ("README.md", "SUMO_HANDOFF.md"):
        shutil.copy2(PROJECT_ROOT / name, run_dir / "source" / name)
    return hashes


def package_versions():
    versions = {"python": sys.version}
    for name in ("numpy", "torch", "gymnasium", "stable-baselines3", "traci"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def resolve_reward_scale(reward_fn, reward_scale):
    if reward_scale is not None:
        return reward_scale
    if reward_fn is queue_reward:
        return DEFAULT_REWARD_SCALE
    if reward_fn is wait_reward:
        return DEFAULT_WAIT_REWARD_SCALE
    if reward_fn is combined_reward:
        return 1.0  # Waiting is already normalized inside combined_reward.
    raise ValueError("Pass --reward-scale for a custom reward function")


class TrainingRewardScale(gym.RewardWrapper):
    """Scale the optimizer's reward without changing raw reward functions."""

    def __init__(self, env, scale):
        super().__init__(env)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("reward scale must be positive and finite")
        self.scale = scale

    def reward(self, reward):
        return float(reward * self.scale)


def evaluate_policy(model=None, seeds=TEST_SEEDS, baseline=None, reward_fn=queue_reward,
                    env_factory=ToyTrafficEnv, max_steps=500):
    """Use fresh traffic seeds and report rewards in their original units."""
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("evaluation needs at least one seed")
    raw_env = env_factory()
    native_cycle = baseline == "fixed_cycle" and hasattr(raw_env, "use_fixed_cycle")
    if native_cycle:
        raw_env.use_fixed_cycle()
    env = GymnasiumWrapper(raw_env, reward_fn, max_steps=max_steps)
    returns = []
    episodes = []
    phase_counts = [0, 0, 0, 0]
    total_steps = 0
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            action_rng = random.Random(seed + 10000)
            episode_return = 0.0
            queue_total = wait_total = switches = 0
            step = 0
            while True:
                if model is not None:
                    action = int(model.predict(observation, deterministic=True)[0])
                elif baseline == "random":
                    action = action_rng.choice([0, 1])
                elif baseline == "fixed_cycle":
                    # SUMO uses its native 10/3/10/3 cycle; toy uses 10/1/10/1.
                    action = 0 if native_cycle else int(step % 22 in (10, 21))
                else:
                    raise ValueError("Provide a model or a recognized baseline")
                old_phase = raw_env.get_state()[12]
                observation, reward, terminated, truncated, info = env.step(action)
                state = info["raw_state"]
                switched = int(old_phase in (0, 2) and state[12] in (1, 3))
                if native_cycle:
                    # Charge the same physical switching cost to the automatic baseline.
                    reward = reward_fn(state, switched)
                episode_return += reward
                phase_counts[state[12]] += 1
                queue_total += sum(state[4:8])
                wait_total += sum(state[8:12])
                # Count all green-to-yellow transitions, including the native baseline.
                switches += switched
                total_steps += 1
                step += 1
                if terminated or truncated:
                    break
            returns.append(episode_return)
            episode = {
                "seed": seed, "reward": episode_return, "steps": step,
                "terminated": bool(terminated), "truncated": bool(truncated),
                "queue_sum_over_steps": queue_total, "wait_sum_over_steps": wait_total,
                "mean_queue": queue_total / step, "mean_wait": wait_total / step,
                "switches": switches,
            }
            if hasattr(raw_env, "get_diagnostics"):
                episode.update(raw_env.get_diagnostics())
            episodes.append(episode)
    finally:
        env.close()
    return {
        "mean_reward": float(np.mean(returns)),
        "std_reward": float(np.std(returns)),
        "episode_returns": returns,
        "episode_seeds": list(seeds),
        "yellow_fraction": (phase_counts[1] + phase_counts[3]) / total_steps,
        "phase_fractions": [count / total_steps for count in phase_counts],
        "episodes": episodes,
        "completion_rate": float(np.mean([row["terminated"] for row in episodes])),
        "truncation_rate": float(np.mean([row["truncated"] for row in episodes])),
        "mean_queue": float(np.mean([row["mean_queue"] for row in episodes])),
        "mean_wait": float(np.mean([row["mean_wait"] for row in episodes])),
        "mean_switches": float(np.mean([row["switches"] for row in episodes])),
    }


class EvaluateAndSave(BaseCallback):
    def __init__(self, run_dir, frequency=10000, reward_fn=queue_reward,
                 env_factory=ToyTrafficEnv, max_steps=500, seeds=VALIDATION_SEEDS):
        super().__init__()
        self.run_dir = Path(run_dir)
        self.frequency = frequency
        self.reward_fn = reward_fn
        self.env_factory = env_factory
        self.max_steps = max_steps
        self.seeds = tuple(seeds)
        self.history = []
        self.best_mean_reward = -float("inf")
        self.best_step = None

    def _on_step(self):
        if self.num_timesteps % self.frequency == 0:
            self.evaluate()
        return True

    def evaluate(self):
        metrics = evaluate_policy(self.model, self.seeds, reward_fn=self.reward_fn,
                                  env_factory=self.env_factory, max_steps=self.max_steps)
        metrics["timesteps"] = self.model.num_timesteps
        self.history.append(metrics)
        if metrics["mean_reward"] > self.best_mean_reward:
            self.best_mean_reward = metrics["mean_reward"]
            self.best_step = metrics["timesteps"]
            self.model.save(self.run_dir / "best_model.zip")
        (self.run_dir / "validation.json").write_text(json.dumps(self.history, indent=2))
        print(
            f"Step {metrics['timesteps']}: evaluation reward={metrics['mean_reward']:.1f}, "
            f"yellow={metrics['yellow_fraction']:.1%}, best={self.best_mean_reward:.1f}",
            flush=True,
        )


def train(total_timesteps=150000, reward_scale=None,
          target_update_interval=DEFAULT_TARGET_UPDATE_INTERVAL,
          seed=42, run_dir=None, learning_rate=DEFAULT_LEARNING_RATE, reward_fn=queue_reward,
          env_factory=ToyTrafficEnv, max_steps=500, eval_frequency=10000,
          validation_seeds=VALIDATION_SEEDS, test_seeds=TEST_SEEDS, run_kind="training"):
    """Write each run to its own directory; never overwrite an existing run."""
    reward_scale = resolve_reward_scale(reward_fn, reward_scale)
    if total_timesteps <= 0 or eval_frequency <= 0:
        raise ValueError("timesteps and evaluation frequency must be positive")
    validation_seeds, test_seeds = tuple(validation_seeds), tuple(test_seeds)
    if not validation_seeds or not test_seeds:
        raise ValueError("validation and test seeds cannot be empty")
    if set(validation_seeds) & set(test_seeds):
        raise ValueError("validation and test traffic seeds must be disjoint")
    # Fail on missing runtime dependencies before allocating a run directory.
    raw_wrapper = GymnasiumWrapper(env_factory(), reward_fn, max_steps=max_steps)
    logs = PROJECT_ROOT / "logs"
    logs.mkdir(exist_ok=True)
    if run_dir is None:
        run_dir = Path(tempfile.mkdtemp(prefix="dqn_", dir=logs))
    else:
        run_dir = Path(run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "total_timesteps": total_timesteps,
        "reward_scale": reward_scale,
        "target_update_interval": target_update_interval,
        "seed": seed,
        "learning_rate": learning_rate,
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.05,
        "reward_function": reward_fn.__name__,
        "environment": (env_factory.describe() if hasattr(env_factory, "describe")
                        else {"factory": f"{getattr(env_factory, '__module__', type(env_factory).__module__)}."
                                         f"{getattr(env_factory, '__qualname__', type(env_factory).__qualname__)}"}),
        "max_steps": max_steps,
        "validation_seeds": list(validation_seeds), "test_seeds": list(test_seeds),
        "eval_frequency": eval_frequency,
        "run_kind": run_kind,
        "versions": package_versions(),
        "source_sha256": save_sources(run_dir),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    torch.set_num_threads(1)
    # Monitor sees original rewards; DQN receives scaled rewards.
    env = TrainingRewardScale(Monitor(raw_wrapper, str(run_dir / "train.monitor.csv")), reward_scale)
    callback = EvaluateAndSave(run_dir, frequency=eval_frequency, reward_fn=reward_fn,
                               env_factory=env_factory, max_steps=max_steps, seeds=validation_seeds)
    eval_options = dict(reward_fn=reward_fn, env_factory=env_factory,
                        max_steps=max_steps, seeds=test_seeds)
    started = perf_counter()
    try:
        model = DQN(
            "MlpPolicy", env, verbose=0, device="cpu", seed=seed,
            exploration_fraction=0.3, exploration_final_eps=0.05,
            learning_rate=learning_rate, target_update_interval=target_update_interval,
        )
        print(f"Training in {run_dir}", flush=True)
        model.learn(total_timesteps=total_timesteps, callback=callback)
        learn_seconds = perf_counter() - started
        model.save(run_dir / "final_model.zip")
        # Evaluate the final weights too: the last gradient update occurs after _on_step.
        callback.evaluate()
        best_model = DQN.load(run_dir / "best_model.zip", device="cpu")
        results = {
            "config": config,
            "best_validation_mean_reward": callback.best_mean_reward,
            "best_step": callback.best_step,
            "best_test": evaluate_policy(best_model, **eval_options),
            "final_test": evaluate_policy(model, **eval_options),
            "fixed_cycle_test": evaluate_policy(baseline="fixed_cycle", **eval_options),
            "training_seconds_including_validation": learn_seconds,
            "training_steps_per_second": model.num_timesteps / learn_seconds,
            "actual_timesteps": model.num_timesteps,
            "total_seconds": perf_counter() - started,
        }
        (run_dir / "results.json").write_text(json.dumps(results, indent=2))
        print(f"Completed: best held-out reward={results['best_test']['mean_reward']:.1f}; "
              f"final={results['final_test']['mean_reward']:.1f}; "
              f"fixed cycle={results['fixed_cycle_test']['mean_reward']:.1f}", flush=True)
        return run_dir
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-timesteps", type=int, default=150000)
    parser.add_argument("--reward-scale", type=float, default=None)
    parser.add_argument("--target-update-interval", type=int, default=DEFAULT_TARGET_UPDATE_INTERVAL)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reward", choices=("queue", "wait", "combined"), default="queue")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--env", choices=("toy", "sumo"), default="toy")
    parser.add_argument("--scenario", choices=("normal", "peak"), default="normal")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--eval-frequency", type=int, default=10000)
    parser.add_argument("--smoke", action="store_true",
                        help="check Gymnasium, then run 1000 training steps and two evaluation seeds")
    args = parser.parse_args()
    options = vars(args)
    backend, scenario = options.pop("env"), options.pop("scenario")
    options["env_factory"] = EnvironmentFactory(backend, scenario)
    if options["max_steps"] is None:
        options["max_steps"] = 150 if backend == "sumo" else 500
    options["reward_fn"] = REWARDS[options.pop("reward")]
    if options.pop("smoke"):
        from gymnasium.utils.env_checker import check_env
        checked_env = GymnasiumWrapper(options["env_factory"](), options["reward_fn"],
                                       max_steps=options["max_steps"])
        try:
            check_env(checked_env, skip_render_check=True)
        finally:
            checked_env.close()
        options.update(total_timesteps=1000, eval_frequency=500, run_kind="smoke",
                       validation_seeds=(100, 101), test_seeds=(1000, 1001))
    train(**options)
