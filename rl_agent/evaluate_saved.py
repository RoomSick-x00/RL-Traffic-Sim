"""Re-evaluate saved best models on validation seeds without training or test access."""

import argparse
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from stable_baselines3 import DQN

from .env_factory import EnvironmentFactory
from .rewards import reward_from_config
from .training import PROJECT_ROOT, evaluate_policy, package_versions


def evaluate_run(run_dir, output_dir, junction_diagnostics=False):
    run_dir, output_dir = Path(run_dir).resolve(), Path(output_dir)
    config = json.loads((run_dir / "config.json").read_text())
    model_path = run_dir / "best_model.zip"
    seeds = config["validation_seeds"]
    if junction_diagnostics and config["environment"]["backend"] != "sumo":
        raise ValueError("Junction diagnostics require a SUMO run")
    if not seeds or set(seeds) & set(config.get("test_seeds", [])):
        raise ValueError("Saved validation seeds must be nonempty and disjoint from test seeds")
    # Changes to evaluation diagnostics are allowed; changed dynamics are not.
    changed_sources = []
    for name, digest in config.get("source_sha256", {}).items():
        source = PROJECT_ROOT / name
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            changed_sources.append(name)
            if (name in ("rl_agent/env_factory.py", "rl_agent/gym_wrapper.py", "rl_agent/toy_env.py",
                         "test/traffic_env.py", "test/__init__.py")
                    or name.endswith((".xml", ".sumocfg"))):
                raise ValueError(f"Environment differs from saved run: {name}")
    model = DQN.load(model_path, device="cpu")
    options = dict(seeds=seeds, reward_fn=reward_from_config(config),
                   env_factory=EnvironmentFactory(**config["environment"]), max_steps=config["max_steps"],
                   junction_diagnostics=junction_diagnostics)
    metrics = evaluate_policy(model=model, **options)
    baseline = evaluate_policy(baseline="fixed_cycle", **options)
    result = {
        "run_directory": str(run_dir), "config": config,
        "evaluation_split": "validation", "evaluation_seeds": list(seeds),
        "junction_diagnostics_enabled": junction_diagnostics,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "evaluation_versions": package_versions(),
        "changed_sources_since_training": changed_sources,
        "evaluation_source_sha256": {
            name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
            for name in ("rl_agent/evaluate_saved.py", "rl_agent/training.py", "rl_agent/rewards.py",
                         "rl_agent/junction_diagnostics.py")},
        "best_validation": metrics, "fixed_cycle_validation": baseline,
    }
    # Check common metrics against the original validation-selected checkpoint.
    history = json.loads((run_dir / "validation.json").read_text())
    expected = max(history, key=lambda row: row["mean_reward"])
    result["difference_from_original_validation"] = {
        name: metrics[name] - expected[name]
        for name in ("mean_reward", "mean_wait", "mean_queue", "mean_switches",
                     "completion_rate", "truncation_rate")}
    result["original_validation_best_step"] = expected["timesteps"]
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{run_dir.name}_{uuid4().hex[:8]}.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"Saved validation diagnostics: {path}", flush=True)
    print(f"wait={metrics['mean_wait']:.3f}, switches={metrics['mean_switches']:.2f}, "
          f"empty-target switches={metrics['mean_empty_target_switches']:.2f}", flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--junction-diagnostics", action="store_true",
                        help="Log incoming/internal stopped time, vehicle traces and signals (SUMO only)")
    args = parser.parse_args()
    output = PROJECT_ROOT / "logs" / f"validation_diagnostics_{uuid4().hex[:8]}"
    for run in args.run_dirs:
        evaluate_run(run, output, junction_diagnostics=args.junction_diagnostics)


if __name__ == "__main__":
    main()
