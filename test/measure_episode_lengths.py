"""Measure SUMO episode lengths using the network's existing fixed signal cycle."""

import argparse
import json
from pathlib import Path
import statistics
import tempfile


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return value


def measure_episode(env, seed, max_steps):
    """Count step() calls after reset; always close a successfully started episode."""
    env.reset(seed=seed)
    try:
        for steps in range(1, max_steps + 1):
            # Action 0 leaves SUMO's automatic 10/3/10/3-second program running.
            _, done = env.step(0)
            if done:
                break
        return {
            "seed": seed,
            "steps": steps,
            "terminated": bool(done),
            "truncated": not bool(done),
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+", choices=("normal", "peak"),
                        default=["normal", "peak"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--max-steps", type=positive_int, default=10000,
                        help="measurement safety limit, not the final training cap")
    args = parser.parse_args()

    # Keep --help and the loop tests usable on machines without SUMO/TraCI.
    from traffic_env import TrafficEnvironment

    logs = Path(__file__).resolve().parent.parent / "logs"
    logs.mkdir(exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="episode_lengths_", dir=logs))
    output = run_dir / "results.json"
    results = {
        "controller": "SUMO network fixed cycle: 10/3/10/3 seconds; action=0",
        "max_steps": args.max_steps,
        "scenarios": args.scenarios,
        "seeds": args.seeds,
        "counting": "steps counts env.step calls; excludes the one step inside reset",
        "episodes": [],
    }
    print(f"Results: {output}", flush=True)
    for scenario in args.scenarios:
        for seed in args.seeds:
            print(f"Running {scenario}, seed {seed}...", flush=True)
            episode = measure_episode(TrafficEnvironment(scenario), seed, args.max_steps)
            episode["scenario"] = scenario
            results["episodes"].append(episode)
            # Save after each episode so earlier measurements survive a later failure.
            output.write_text(json.dumps(results, indent=2), encoding="utf-8")
            status = "finished" if episode["terminated"] else "TRUNCATED at safety limit"
            print(f"{scenario}, seed {seed}: {episode['steps']} steps, {status}", flush=True)

    for scenario in dict.fromkeys(args.scenarios):
        runs = [run for run in results["episodes"] if run["scenario"] == scenario]
        finished = [run["steps"] for run in runs if run["terminated"]]
        truncated = sum(run["truncated"] for run in runs)
        if finished:
            print(f"{scenario}: completed min={min(finished)}, "
                  f"median={statistics.median(finished)}, max={max(finished)} steps; "
                  f"truncated={truncated}/{len(runs)}")
        else:
            print(f"{scenario}: no completed episodes; truncated={truncated}/{len(runs)}")
    print("Share results.json. Truncated lengths are lower bounds, not completion times.")


if __name__ == "__main__":
    main()
