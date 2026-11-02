"""Check beta persistence and separation of tuning from held-out testing."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import training
from .rewards import combined_reward, make_combined_reward, reward_from_config
from .test_pipeline import NativeCycleEnv, ReadOnlyStateEnv


class RewardTuningTests(unittest.TestCase):
    def test_reward_reconstruction_preserves_beta_and_legacy_defaults(self):
        state = [0] * 8 + [195, 0, 0, 0] + [0]
        for beta in (0.1, 0.05, 0.02):
            reward = make_combined_reward(beta)
            restored = reward_from_config({"reward_function": reward.__name__,
                                           "reward_parameters": reward.reward_parameters})
            self.assertAlmostEqual(restored(state, 1), -0.39 - beta)
            self.assertAlmostEqual(restored(state, 0), -0.39)
            self.assertEqual(training.resolve_reward_scale(reward, None), 1.0)
        legacy = reward_from_config({"reward_function": "combined_reward"})
        self.assertAlmostEqual(legacy(state, 1), -1.39)
        for beta in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                make_combined_reward(beta)

    def test_native_baseline_uses_selected_beta(self):
        metrics = training.evaluate_policy(
            seeds=(100,), baseline="fixed_cycle", env_factory=NativeCycleEnv,
            reward_fn=make_combined_reward(0.05))
        self.assertEqual(metrics["episode_returns"], [-0.05])

    def test_validation_only_run_never_uses_test_seeds(self):
        ReadOnlyStateEnv.instances = []
        original_evaluate = training.evaluate_policy
        with tempfile.TemporaryDirectory() as directory, patch.object(
                training, "evaluate_policy", wraps=original_evaluate) as evaluate:
            run = training.train(
                total_timesteps=8, eval_frequency=4, max_steps=3,
                run_dir=Path(directory) / "run", env_factory=ReadOnlyStateEnv,
                reward_fn=combined_reward, switch_penalty=0.05,
                validation_only=True, validation_seeds=(100,), test_seeds=(1000,))
            results = json.loads((run / "results.json").read_text())
            config = results["config"]
            self.assertEqual(config["reward_parameters"], {"switch_penalty": 0.05})
            self.assertEqual(config["test_seeds"], [])
            self.assertTrue(config["validation_only"])
            self.assertEqual(config["reward_scale"], 1.0)
            self.assertNotIn("best_test", results)
            for name in ("best_validation", "final_validation", "fixed_cycle_validation"):
                self.assertEqual(results[name]["episode_seeds"], [100])
            for call in evaluate.call_args_list:
                self.assertEqual(call.kwargs["reward_fn"].reward_parameters,
                                 {"switch_penalty": 0.05})
            # Training may reset with seed=None; no environment sees held-out seed 1000.
            self.assertTrue(all(1000 not in env.seeds for env in ReadOnlyStateEnv.instances))
            self.assertTrue(all(not env.open for env in ReadOnlyStateEnv.instances))
            with self.assertRaises(FileExistsError):
                training.train(total_timesteps=8, run_dir=run,
                               env_factory=ReadOnlyStateEnv)

    def test_cli_forwards_tuning_options(self):
        argv = ["train", "--env", "sumo", "--reward", "combined",
                "--switch-penalty", "0.02", "--validation-only"]
        with patch("sys.argv", argv), patch.object(training, "train") as train:
            training.main()
        options = train.call_args.kwargs
        self.assertEqual(options["switch_penalty"], 0.02)
        self.assertTrue(options["validation_only"])
        self.assertEqual(options["max_steps"], 150)
        self.assertIs(options["reward_fn"], combined_reward)


if __name__ == "__main__":
    unittest.main()
