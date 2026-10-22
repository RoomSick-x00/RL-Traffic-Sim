"""Integration checks for factories, evaluation and short DQN runs."""

import json
from pathlib import Path
import tempfile
import unittest

from .rewards import combined_reward
from .training import evaluate_policy, train


class ReadOnlyStateEnv:
    """No .phase attribute: same public state access as SUMO."""
    instances = []

    def __init__(self):
        self.instances.append(self)
        self.seeds = []
        self.episode_steps = []
        self.open = False

    def reset(self, seed=None):
        if self.open:
            raise RuntimeError("old episode was not closed")
        self.open = True
        self.seeds.append(seed)
        self.steps = 0
        return self.get_state()

    def get_state(self):
        return [1] * 4 + [2] * 4 + [3] * 4 + [0]

    def step(self, action):
        self.steps += 1
        return self.get_state(), False

    def close(self):
        if self.open:
            self.episode_steps.append(self.steps)
        self.open = False


class NativeCycleEnv(ReadOnlyStateEnv):
    def use_fixed_cycle(self):
        self.native = True

    def get_state(self):
        return [0] * 12 + [0 if self.steps == 0 else 1]

    def step(self, action):
        assert self.native and action == 0
        self.steps += 1
        return self.get_state(), True


class PipelineTests(unittest.TestCase):
    def setUp(self):
        ReadOnlyStateEnv.instances = []

    def test_evaluation_uses_factory_cap_and_shared_metrics(self):
        result = evaluate_policy(seeds=(123,), baseline="random",
                                 env_factory=ReadOnlyStateEnv, max_steps=3)
        self.assertEqual(result["episode_returns"], [-24.0])
        self.assertEqual(result["episodes"][0]["steps"], 3)
        self.assertEqual(result["mean_queue"], 8)
        self.assertEqual(result["mean_wait"], 12)
        self.assertEqual(result["truncation_rate"], 1)
        self.assertEqual(result["completion_rate"], 0)
        self.assertFalse(ReadOnlyStateEnv.instances[0].open)

    def test_native_baseline_pays_combined_switch_cost(self):
        result = evaluate_policy(seeds=(123,), baseline="fixed_cycle",
                                 env_factory=NativeCycleEnv, max_steps=150,
                                 reward_fn=combined_reward)
        self.assertEqual(result["episode_returns"], [-1.0])
        self.assertEqual(result["mean_switches"], 1)

    def test_short_training_forwards_factory_to_every_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            run = train(total_timesteps=8, run_dir=Path(directory) / "run",
                        env_factory=ReadOnlyStateEnv, max_steps=3,
                        eval_frequency=4, validation_seeds=(100,), test_seeds=(1000,))
            result = json.loads((run / "results.json").read_text())
            history = json.loads((run / "validation.json").read_text())
            self.assertEqual([row["timesteps"] for row in history], [4, 8, 8])
            for key in ("best_test", "final_test", "fixed_cycle_test"):
                self.assertEqual(result[key]["episode_returns"], [-24.0])
                self.assertEqual(result[key]["episode_seeds"], [1000])
            self.assertEqual(result["config"]["max_steps"], 3)
            self.assertTrue((run / "best_model.zip").is_file())
            self.assertTrue((run / "source" / "rl_agent" / "training.py").is_file())
        self.assertGreaterEqual(len(ReadOnlyStateEnv.instances), 7)
        self.assertTrue(all(not env.open for env in ReadOnlyStateEnv.instances))
        self.assertTrue(all(count <= 3 for env in ReadOnlyStateEnv.instances
                            for count in env.episode_steps))


if __name__ == "__main__":
    unittest.main()
