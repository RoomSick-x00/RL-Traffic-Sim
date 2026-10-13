from stable_baselines3 import DQN
from toy_env import ToyTrafficEnv
from gym_wrapper import GymnasiumWrapper
from rewards import queue_reward

raw_env = ToyTrafficEnv()
env = GymnasiumWrapper(raw_env, reward_fn=queue_reward)

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)
model.save("dqn_toy_v1")
