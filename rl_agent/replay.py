"""Watch a saved DQN control one episode in SUMO's GUI."""

import argparse
import json
from pathlib import Path
import shutil
import sys
from uuid import uuid4

from stable_baselines3 import DQN

from .env_factory import EnvironmentFactory
from .gym_wrapper import GymnasiumWrapper
from .rewards import reward_from_config
from .junction_diagnostics import JunctionDiagnostics


def replay(model_path, seed=2000, delay=300, wait=True, junction_diagnostics=False):
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    config_path = model_path.parent / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Keep the run's config.json beside the model: {config_path}")
    config = json.loads(config_path.read_text())
    if config["environment"]["backend"] != "sumo":
        raise ValueError("Choose a model trained with --env sumo.")
    if delay < 0:
        raise ValueError("delay must be zero or greater (milliseconds)")
    gui = shutil.which("sumo-gui")
    if gui is None:
        raise RuntimeError("sumo-gui is not on PATH. Run this on the SUMO machine.")

    reward_fn = reward_from_config(config)
    model = DQN.load(str(model_path), device="cpu")
    raw_env = EnvironmentFactory(**config["environment"])()
    raw_env.sumo_cmd[0] = gui
    raw_env.sumo_cmd.extend(["--start", "--delay", str(delay)])
    env = GymnasiumWrapper(raw_env, reward_fn, max_steps=config["max_steps"])
    total_reward = 0.0
    steps = switches = 0
    try:
        state, _ = env.reset(seed=seed)
        junction = JunctionDiagnostics(raw_env) if junction_diagnostics else None
        while True:
            old_phase = raw_env.get_state()[12]
            action, _ = model.predict(state, deterministic=True)
            state, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            steps += 1
            switches += info["applied_switch"]
            if junction is not None:
                junction.sample(steps, int(action), old_phase, info["raw_state"][12])
            if terminated or truncated:
                break
        summary = {
            "seed": seed, "steps": steps, "reward": total_reward,
            "switches": switches, "terminated": bool(terminated),
            "truncated": bool(truncated), **raw_env.get_diagnostics(),
        }
        print(json.dumps(summary, indent=2))
        if junction is not None:
            logs = Path(__file__).resolve().parent.parent / "logs"
            logs.mkdir(exist_ok=True)
            path = logs / f"junction_replay_{seed}_{uuid4().hex[:8]}.json"
            path.write_text(json.dumps({"model": str(model_path.resolve()), "config": config,
                                        "summary": summary, "junction_diagnostics": junction.result()}, indent=2))
            print(f"Saved junction replay diagnostic: {path}", flush=True)
        if wait and sys.stdin.isatty():
            input("Episode finished. Press Enter here to close SUMO... ")
        return summary
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path,
                        help="Saved best_model.zip (with config.json beside it)")
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--delay", type=int, default=300,
                        help="Milliseconds between simulation steps (default: 300)")
    parser.add_argument("--no-wait", action="store_true",
                        help="Close the GUI immediately when the episode ends")
    parser.add_argument("--junction-diagnostics", action="store_true",
                        help="Save vehicle/lane/signal traces and stopped-time measurements automatically")
    args = parser.parse_args()
    try:
        replay(args.model, seed=args.seed, delay=args.delay, wait=not args.no_wait,
               junction_diagnostics=args.junction_diagnostics)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")
    except (KeyboardInterrupt, EOFError):
        pass


if __name__ == "__main__":
    main()
