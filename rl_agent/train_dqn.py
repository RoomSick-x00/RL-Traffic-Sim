from pathlib import Path

from stable_baselines3 import DQN

from gym_wrapper import GymnasiumWrapper
from rewards import queue_reward
from toy_env import ToyTrafficEnv


def main():
    raw_env = ToyTrafficEnv()
    env = GymnasiumWrapper(raw_env, reward_fn=queue_reward)

    model = DQN("MlpPolicy", env, verbose=1, seed=42, device="cpu")
    model.learn(total_timesteps=50_000)

    model_path = Path(__file__).resolve().parent.parent / "models" / "dqn_toy.zip"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    env.close()
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
