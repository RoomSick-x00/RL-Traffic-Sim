"""Train and compare toy DQN policies using unscaled evaluation rewards."""

import argparse
import json
from pathlib import Path
import random
import tempfile

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
else:
    from gym_wrapper import GymnasiumWrapper
    from rewards import queue_reward, wait_reward, combined_reward
    from toy_env import ToyTrafficEnv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VALIDATION_SEEDS = tuple(range(100, 110))
TEST_SEEDS = tuple(range(1000, 1030))
DEFAULT_REWARD_SCALE = 0.01
DEFAULT_WAIT_REWARD_SCALE = 0.00025
DEFAULT_TARGET_UPDATE_INTERVAL = 1000
DEFAULT_LEARNING_RATE = 5e-4


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


def evaluate_policy(model=None, seeds=TEST_SEEDS, baseline=None, reward_fn=queue_reward):
    """Use fresh traffic seeds and report rewards in their original units."""
    env = GymnasiumWrapper(ToyTrafficEnv(), reward_fn)
    returns = []
    phase_counts = [0, 0, 0, 0]
    total_steps = 0
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            action_rng = random.Random(seed + 10000)
            episode_return = 0.0
            step = 0
            while True:
                if model is not None:
                    action = int(model.predict(observation, deterministic=True)[0])
                elif baseline == "random":
                    action = action_rng.choice([0, 1])
                elif baseline == "fixed_cycle":
                    # Ten green steps per pair, with one yellow step between them.
                    action = int(step % 22 in (10, 21))
                else:
                    raise ValueError("Provide a model or a recognized baseline")
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_return += reward
                phase_counts[env.raw_env.phase] += 1
                total_steps += 1
                step += 1
                if terminated or truncated:
                    break
            returns.append(episode_return)
    finally:
        env.close()
    return {
        "mean_reward": float(np.mean(returns)),
        "std_reward": float(np.std(returns)),
        "episode_returns": returns,
        "episode_seeds": list(seeds),
        "yellow_fraction": (phase_counts[1] + phase_counts[3]) / total_steps,
        "phase_fractions": [count / total_steps for count in phase_counts],
    }


class EvaluateAndSave(BaseCallback):
    def __init__(self, run_dir, frequency=10000, reward_fn=queue_reward):
        super().__init__()
        self.run_dir = Path(run_dir)
        self.frequency = frequency
        self.reward_fn = reward_fn
        self.history = []
        self.best_mean_reward = -float("inf")
        self.best_step = None

    def _on_step(self):
        if self.num_timesteps % self.frequency == 0:
            self.evaluate()
        return True

    def evaluate(self):
        metrics = evaluate_policy(self.model, VALIDATION_SEEDS, reward_fn=self.reward_fn)
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
          seed=42, run_dir=None, learning_rate=DEFAULT_LEARNING_RATE, reward_fn=queue_reward):
    """Write each run to its own directory; never overwrite an existing run."""
    reward_scale = resolve_reward_scale(reward_fn, reward_scale)
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
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    torch.set_num_threads(1)
    raw_wrapper = GymnasiumWrapper(ToyTrafficEnv(), reward_fn)
    # Monitor sees original rewards; DQN receives scaled rewards.
    env = TrainingRewardScale(Monitor(raw_wrapper, str(run_dir / "train.monitor.csv")), reward_scale)
    callback = EvaluateAndSave(run_dir, reward_fn=reward_fn)
    try:
        model = DQN(
            "MlpPolicy", env, verbose=0, device="cpu", seed=seed,
            exploration_fraction=0.3, exploration_final_eps=0.05,
            learning_rate=learning_rate, target_update_interval=target_update_interval,
        )
        print(f"Training in {run_dir}", flush=True)
        model.learn(total_timesteps=total_timesteps, callback=callback)
        model.save(run_dir / "final_model.zip")
        # Evaluate the final weights too: the last gradient update occurs after _on_step.
        callback.evaluate()
        best_model = DQN.load(run_dir / "best_model.zip", device="cpu")
        results = {
            "config": config,
            "best_validation_mean_reward": callback.best_mean_reward,
            "best_step": callback.best_step,
            "best_test": evaluate_policy(best_model, reward_fn=reward_fn),
            "final_test": evaluate_policy(model, reward_fn=reward_fn),
            "fixed_cycle_test": evaluate_policy(baseline="fixed_cycle", reward_fn=reward_fn),
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
    args = parser.parse_args()
    options = vars(args)
    options["reward_fn"] = {
        "queue": queue_reward,
        "wait": wait_reward,
        "combined": combined_reward,
    }[options.pop("reward")]
    train(**options)
