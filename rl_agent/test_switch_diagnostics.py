"""Validate switch classification and saved-model validation-only evaluation."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from . import evaluate_saved
from .training import evaluate_policy


class SwitchEnv:
    def use_fixed_cycle(self):
        self.native = True

    def reset(self, seed=None):
        self.state = [0] * 13
        if seed == 1:  # NS queued, target EW empty.
            self.state[4] = 2
        elif seed == 2:  # EW queued; a moving vehicle on NS means target is not empty.
            self.state[12], self.state[6], self.state[0] = 2, 3, 1
        elif seed == 4:  # A yellow request is ignored.
            self.state[12] = 1
        elif seed == 5:  # Reverse direction, EW queued and target NS empty.
            self.state[12], self.state[7] = 2, 3
        return self.get_state()

    def get_state(self):
        return list(self.state)

    def step(self, action):
        self.state[12] += 1
        # Post-step demand changes must not alter the pre-decision classification.
        self.state[:12] = [0] * 12
        return self.get_state(), True

    def close(self):
        pass


class SwitchDiagnosticTests(unittest.TestCase):
    def test_switch_proxy_uses_pre_step_vehicles_and_queues_in_both_axes(self):
        model = Mock()
        model.predict.return_value = (1, None)
        for options in ({"model": model}, {"baseline": "fixed_cycle"}):
            m = evaluate_policy(seeds=(1, 2, 3, 4, 5), env_factory=SwitchEnv, **options)
            self.assertEqual([e["empty_target_switches"] for e in m["episodes"]], [1, 0, 0, 0, 1])
            self.assertEqual([e["switches"] for e in m["episodes"]], [1, 1, 1, 0, 1])
            self.assertEqual(m["mean_empty_target_switches"], 0.4)
            self.assertEqual(m["empty_target_switch_fraction"], 0.5)
            self.assertEqual(m["episodes"][1]["switch_events"][0]["target_vehicles_before"], 1)
            self.assertEqual(m["episodes"][0]["switch_events"][0]["source_queue_before"], 2)
        m = evaluate_policy(model=model, seeds=(4,), env_factory=SwitchEnv)
        self.assertIsNone(m["empty_target_switch_fraction"])

    def test_saved_evaluation_reconstructs_beta_and_uses_only_validation_seeds(self):
        metrics = {"mean_reward": -0.5, "mean_wait": 2.0, "mean_queue": 1.0,
                   "mean_switches": 3.0, "completion_rate": 1.0, "truncation_rate": 0.0,
                   "mean_empty_target_switches": 0.0}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            (run / "config.json").write_text(json.dumps({
                "reward_function": "combined_reward", "reward_parameters": {"switch_penalty": 0.02},
                "validation_seeds": [100, 101], "test_seeds": [1000],
                "environment": {"backend": "sumo", "scenario": "normal"}, "max_steps": 150,
            }))
            (run / "best_model.zip").write_bytes(b"mock model")
            (run / "validation.json").write_text(json.dumps([{**metrics, "timesteps": 90000}]))
            with patch.object(evaluate_saved.DQN, "load") as load, patch.object(
                    evaluate_saved, "evaluate_policy", return_value=metrics) as evaluate:
                path = evaluate_saved.evaluate_run(run, root / "output", junction_diagnostics=True)
            result = json.loads(path.read_text())
            self.assertEqual(result["evaluation_split"], "validation")
            self.assertEqual(evaluate.call_count, 2)
            for call in evaluate.call_args_list:
                self.assertEqual(call.kwargs["seeds"], [100, 101])
                self.assertEqual(call.kwargs["max_steps"], 150)
                self.assertTrue(call.kwargs["junction_diagnostics"])
                self.assertEqual(call.kwargs["reward_fn"].reward_parameters, {"switch_penalty": 0.02})
            self.assertEqual(set(result["difference_from_original_validation"].values()), {0.0})
            self.assertNotIn("best_test", result)
            load.return_value.learn.assert_not_called()

    def test_changed_environment_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({
                "validation_seeds": [100], "test_seeds": [],
                "source_sha256": {"test/traffic_env.py": "incorrect hash"},
            }))
            with patch.object(evaluate_saved.DQN, "load") as load:
                with self.assertRaisesRegex(ValueError, "Environment differs"):
                    evaluate_saved.evaluate_run(root, root / "output")
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
