"""Run matched DQN reward experiments and package their results for Member 3."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import tempfile
from uuid import uuid4

from .env_factory import EnvironmentFactory
from .training import PROJECT_ROOT, REWARDS, package_versions, resolve_reward_scale, train


def validate_smoke(path, factory, max_steps):
    results = json.loads(Path(path).read_text(encoding="utf-8"))
    config = results["config"]
    if (config.get("run_kind") != "smoke" or config.get("environment") != factory.describe()
            or config.get("max_steps") != max_steps):
        raise ValueError("Smoke results must match this environment, scenario, and step cap.")
    for name in ("best_test", "final_test", "fixed_cycle_test"):
        if not results.get(name, {}).get("episodes"):
            raise ValueError("Smoke results are missing completed pipeline evaluations.")
    baseline = results["fixed_cycle_test"]["episodes"]
    if any(not row["terminated"] or row["truncated"] or row.get("teleports", 0)
           for row in baseline):
        raise ValueError("Investigate unfinished/teleporting fixed-cycle smoke runs before training.")
    if not config.get("source_sha256"):
        raise ValueError("Smoke results must include the source manifest.")
    for relative, digest in config["source_sha256"].items():
        source = PROJECT_ROOT / relative
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source changed since the smoke test: {relative}. Run smoke again.")


def summarize(manifest, root):
    rows = []
    metrics = ("mean_queue", "mean_wait", "mean_switches", "completion_rate", "truncation_rate")
    for entry in manifest["runs"].values():
        if entry["status"] != "complete":
            continue
        result = json.loads((root / entry["directory"] / "results.json").read_text())
        rows.append({"reward": entry["reward"], "seed": entry["seed"],
                     "directory": entry["directory"],
                     **{key: result["best_test"][key] for key in metrics}})
    groups = {}
    for reward in manifest["settings"]["rewards"]:
        matching = [row for row in rows if row["reward"] == reward]
        groups[reward] = {}
        for metric in metrics:
            values = [row[metric] for row in matching]
            groups[reward][metric] = {
                "mean": statistics.mean(values) if values else None,
                "std_across_training_seeds": statistics.stdev(values) if len(values) > 1 else None,
            }
    return {"settings": manifest["settings"], "runs": rows, "by_reward": groups,
            "note": "Compare common traffic metrics, not reward totals across different formulas. "
                    "Check completion/truncation and per-episode diagnostics before ranking policies."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=("toy", "sumo"), default="sumo")
    parser.add_argument("--scenario", choices=("normal", "peak"), default="normal")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--rewards", nargs="+", choices=tuple(REWARDS), default=list(REWARDS))
    parser.add_argument("--total-timesteps", type=int, default=150000)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--smoke-results", type=Path,
                        help="required for SUMO; results.json from a successful --smoke run")
    parser.add_argument("--resume", type=Path, help="resume an interrupted comparison directory")
    args = parser.parse_args()
    if args.total_timesteps <= 0 or any(seed < 0 for seed in args.seeds):
        parser.error("timesteps must be positive and seeds nonnegative")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.rewards)) != len(args.rewards):
        parser.error("seeds and rewards must not contain duplicates")
    cap = args.max_steps if args.max_steps is not None else (150 if args.env == "sumo" else 500)
    if cap <= 0:
        parser.error("max-steps must be positive")
    factory = EnvironmentFactory(args.env, args.scenario)
    if args.env == "sumo":
        if args.smoke_results is None:
            parser.error("run the SUMO --smoke command first, then supply --smoke-results")
        validate_smoke(args.smoke_results, factory, cap)
    # Check dependencies before allocating the comparison directory.
    factory().close()
    settings = {"environment": factory.describe(), "seeds": args.seeds,
                "rewards": args.rewards, "total_timesteps": args.total_timesteps,
                "max_steps": cap,
                "versions": package_versions(),
                "reward_scales": {name: resolve_reward_scale(REWARDS[name], None)
                                  for name in args.rewards}}
    if args.resume:
        root = args.resume.resolve()
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["settings"] != settings:
            raise ValueError("Resume settings must match the original comparison.")
        # Prevent mixing changed code/scenarios with already completed runs.
        for entry in manifest["runs"].values():
            if entry["status"] == "complete":
                config = json.loads((root / entry["directory"] / "config.json").read_text())
                for name, digest in config["source_sha256"].items():
                    if hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() != digest:
                        raise ValueError(f"Cannot resume after source changes: {name}")
    else:
        logs = PROJECT_ROOT / "logs"
        logs.mkdir(exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="reward_comparison_", dir=logs))
        manifest = {"settings": settings, "runs": {}}
        if args.smoke_results:
            shutil.copy2(args.smoke_results, root / "smoke_results.json")
    print(f"Comparison: {root}", flush=True)
    for reward in args.rewards:
        for seed in args.seeds:
            key = f"{reward}_{seed}"
            if manifest["runs"].get(key, {}).get("status") == "complete":
                if not (root / manifest["runs"][key]["directory"] / "results.json").is_file():
                    raise ValueError(f"Missing results for {key}")
                continue
            entry = {"reward": reward, "seed": seed, "status": "running",
                     "directory": f"{key}_{uuid4().hex[:8]}"}
            manifest["runs"][key] = entry
            (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
            try:
                train(total_timesteps=args.total_timesteps, seed=seed, reward_fn=REWARDS[reward],
                      env_factory=factory, max_steps=cap, run_dir=root / entry["directory"])
                entry["status"] = "complete"
            except BaseException as exc:
                entry["status"] = "failed"
                entry["error"] = str(exc)
                raise
            finally:
                (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
                (root / "comparison.json").write_text(json.dumps(summarize(manifest, root), indent=2))
    archive = shutil.make_archive(str(root) + "_handoff", "zip", root_dir=root)
    print(f"Completed comparison. Share with Member 3: {archive}")


if __name__ == "__main__":
    main()
