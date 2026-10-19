import random
import unittest

import numpy as np
from stable_baselines3.common.monitor import Monitor

from rl_agent.gym_wrapper import GymnasiumWrapper
from rl_agent.rewards import combined_reward, queue_reward, wait_reward
from rl_agent.toy_env import ToyTrafficEnv
from rl_agent.training import TrainingRewardScale, evaluate_policy, resolve_reward_scale


class TrainingTests(unittest.TestCase):
    def test_scaling_preserves_observations_and_raw_episode_logs(self):
        raw_env = GymnasiumWrapper(ToyTrafficEnv(), queue_reward)
        training_env = TrainingRewardScale(
            Monitor(GymnasiumWrapper(ToyTrafficEnv(), queue_reward)), 0.01
        )
        raw_obs, _ = raw_env.reset(seed=42)
        scaled_obs, _ = training_env.reset(seed=42)
        np.testing.assert_array_equal(raw_obs, scaled_obs)
        raw_total = 0.0
        for step in range(500):
            action = int(step % 22 in (10, 21))
            obs, reward, terminated, truncated, _ = raw_env.step(action)
            scaled_obs, scaled_reward, scaled_term, scaled_trunc, info = training_env.step(action)
            np.testing.assert_array_equal(obs, scaled_obs)
            self.assertAlmostEqual(scaled_reward, reward * 0.01)
            self.assertEqual((terminated, truncated), (scaled_term, scaled_trunc))
            raw_total += reward
        self.assertTrue(truncated)
        self.assertEqual(info["episode"]["r"], raw_total)
        self.assertEqual(info["episode"]["l"], 500)
        training_env.close()
        raw_env.close()

    def test_evaluation_is_repeatable_and_does_not_consume_global_randomness(self):
        before = random.getstate()
        first = evaluate_policy(seeds=(1000, 1001), baseline="random")
        second = evaluate_policy(seeds=(1000, 1001), baseline="random")
        self.assertEqual(first, second)
        self.assertEqual(before, random.getstate())
        self.assertEqual(len(first["episode_returns"]), 2)

    def test_invalid_scaling_is_rejected(self):
        for scale in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                TrainingRewardScale(GymnasiumWrapper(ToyTrafficEnv(), queue_reward), scale)

    def test_scale_defaults_follow_reward_choice_and_allow_override(self):
        self.assertEqual(resolve_reward_scale(queue_reward, None), 0.01)
        self.assertEqual(resolve_reward_scale(wait_reward, None), 0.00025)
        self.assertEqual(resolve_reward_scale(combined_reward, None), 1.0)
        self.assertEqual(resolve_reward_scale(wait_reward, 0.001), 0.001)

    def test_combined_reward_penalizes_wait_and_actual_switches(self):
        raw_state = [0] * 8 + [2, 3, 4, 5] + [0]
        self.assertAlmostEqual(combined_reward(raw_state, 0), -0.028)
        self.assertAlmostEqual(combined_reward(raw_state, 1), -1.028)
        self.assertEqual(wait_reward(raw_state, 1), -14.0)
        self.assertEqual(queue_reward(raw_state, 1), 0.0)

        env = GymnasiumWrapper(ToyTrafficEnv(), combined_reward)
        env.reset(seed=42)
        _, green_switch_reward, _, _, _ = env.step(1)
        self.assertEqual(env.raw_env.phase, 1)
        self.assertAlmostEqual(green_switch_reward, -sum(env.raw_env.get_state()[8:12]) / 500 - 1)

        _, yellow_request_reward, _, _, _ = env.step(1)
        self.assertEqual(env.raw_env.phase, 2)
        self.assertAlmostEqual(yellow_request_reward, -sum(env.raw_env.get_state()[8:12]) / 500)
        env.close()

    def test_fixed_cycle_uses_wait_reward_when_selected(self):
        baseline = evaluate_policy(seeds=(1000,), baseline="fixed_cycle", reward_fn=wait_reward)
        raw_env = ToyTrafficEnv()
        raw_env.reset(seed=1000)
        expected = 0.0
        for step in range(500):
            state, _ = raw_env.step(int(step % 22 in (10, 21)))
            expected += wait_reward(state)
        self.assertEqual(baseline["episode_returns"], [expected])


if __name__ == "__main__":
    unittest.main()
