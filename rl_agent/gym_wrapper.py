import gymnasium as gym
import numpy as np


class GymnasiumWrapper(gym.Env):
    def __init__(self, raw_env, reward_fn):
        super().__init__()
        self.raw_env = raw_env
        self.reward_fn = reward_fn
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(16,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        raw_state = self.raw_env.reset(seed=seed)
        processed_state = self._process_state(raw_state)
        return processed_state, {}

    def step(self, action):
        raw_state, done = self.raw_env.step(action)
        processed_state = self._process_state(raw_state)
        reward = self.reward_fn(raw_state)
        terminated = False
        truncated = done  # The toy environment ends at its step limit.
        info = {}
        return processed_state, reward, terminated, truncated, info

    def _process_state(self, raw_state):
        veh = raw_state[0:4]
        veh_norm = [min(1.0, v / 16) for v in veh]

        queue = raw_state[4:8]
        queue_norm = [min(1.0, q / 10) for q in queue]

        wait = raw_state[8:12]
        wait_norm = [min(1.0, w / 400) for w in wait]

        phase = raw_state[12]
        phase_one_hot = [0, 0, 0, 0]
        phase_one_hot[phase] = 1

        final_state = veh_norm + queue_norm + wait_norm + phase_one_hot
        return np.array(final_state, dtype=np.float32)
    
if __name__ == "__main__":
    from toy_env import ToyTrafficEnv
    from rewards import queue_reward
    from gymnasium.utils.env_checker import check_env

    raw_env = ToyTrafficEnv()
    wrapped_env = GymnasiumWrapper(raw_env, reward_fn=queue_reward)

    check_env(wrapped_env)
    print("check_env passed!")

    state, info = wrapped_env.reset()
    print("Initial state:", state, state.shape)

    for i in range(5):
        action = wrapped_env.action_space.sample()
        state, reward, terminated, truncated, info = wrapped_env.step(action)
        print(f"Step {i}: state={state}, reward={reward}, terminated={terminated}, truncated={truncated}")
