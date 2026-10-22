"""Loop checks that do not require SUMO."""

import unittest
from unittest.mock import Mock

from measure_episode_lengths import measure_episode


class EpisodeLengthTests(unittest.TestCase):
    def test_completion_counts_steps_and_closes(self):
        env = Mock()
        env.step.side_effect = [([], False), ([], True)]
        result = measure_episode(env, seed=42, max_steps=5)
        self.assertEqual(result, {
            "seed": 42, "steps": 2, "terminated": True, "truncated": False,
        })
        env.reset.assert_called_once_with(seed=42)
        self.assertEqual(env.step.call_count, 2)
        env.step.assert_called_with(0)
        env.close.assert_called_once()

    def test_safety_limit_is_not_completion(self):
        env = Mock()
        env.step.return_value = ([], False)
        result = measure_episode(env, seed=43, max_steps=3)
        self.assertEqual(result["steps"], 3)
        self.assertTrue(result["truncated"])
        self.assertFalse(result["terminated"])
        env.close.assert_called_once()

    def test_completion_on_limit_is_still_completion(self):
        env = Mock()
        env.step.return_value = ([], True)
        result = measure_episode(env, seed=42, max_steps=1)
        self.assertTrue(result["terminated"])
        self.assertFalse(result["truncated"])

    def test_step_failure_closes_connection(self):
        env = Mock()
        env.step.side_effect = RuntimeError("simulation failed")
        with self.assertRaises(RuntimeError):
            measure_episode(env, seed=42, max_steps=5)
        env.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
