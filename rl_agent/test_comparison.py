"""Check experiment matching, resume behavior, and the SUMO smoke gate."""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import compare_rewards
from .env_factory import EnvironmentFactory


class ComparisonTests(unittest.TestCase):
    def test_matrix_and_resume_keep_reward_runs_matched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source.py").write_text("# fixture")
            digest = hashlib.sha256((root / "source.py").read_bytes()).hexdigest()
            calls = []

            def fake_train(**kwargs):
                calls.append(kwargs)
                run = kwargs["run_dir"]
                run.mkdir()
                (run / "config.json").write_text(json.dumps({"source_sha256": {"source.py": digest}}))
                metrics = dict(mean_queue=1, mean_wait=2, mean_switches=3,
                               completion_rate=1, truncation_rate=0)
                (run / "results.json").write_text(json.dumps({"best_test": metrics}))

            argv = ["compare_rewards", "--env", "toy", "--total-timesteps", "8"]
            with patch.object(compare_rewards, "PROJECT_ROOT", root), \
                    patch.object(compare_rewards, "train", side_effect=fake_train), \
                    patch("sys.argv", argv), redirect_stdout(io.StringIO()):
                compare_rewards.main()
                study = next((root / "logs").glob("reward_comparison_*/manifest.json")).parent
                with patch("sys.argv", argv + ["--resume", str(study)]):
                    compare_rewards.main()
            self.assertEqual(len(calls), 9)
            self.assertEqual({call["seed"] for call in calls}, {42, 43, 44})
            self.assertTrue(all(call["max_steps"] == 500 for call in calls))
            self.assertTrue(all(call["total_timesteps"] == 8 for call in calls))
            summary = json.loads((study / "comparison.json").read_text())
            self.assertEqual(len(summary["runs"]), 9)
            self.assertEqual(set(summary["by_reward"]), {"queue", "wait", "combined"})
            self.assertTrue(Path(str(study) + "_handoff.zip").is_file())

    def test_smoke_gate_checks_scenario_completion_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.py"
            source.write_text("# fixture")
            factory = EnvironmentFactory("sumo", "normal")
            config = {"environment": factory.describe(), "max_steps": 150, "run_kind": "smoke",
                      "source_sha256": {"source.py": hashlib.sha256(source.read_bytes()).hexdigest()}}
            metrics = {"episodes": [{"terminated": True, "truncated": False, "teleports": 0}]}
            results = {"config": config, "best_test": metrics, "final_test": metrics,
                       "fixed_cycle_test": metrics}
            path = root / "results.json"
            path.write_text(json.dumps(results))
            with patch.object(compare_rewards, "PROJECT_ROOT", root):
                compare_rewards.validate_smoke(path, factory, 150)
                with self.assertRaises(ValueError):
                    compare_rewards.validate_smoke(path, EnvironmentFactory("sumo", "peak"), 150)
                results["fixed_cycle_test"]["episodes"][0]["truncated"] = True
                path.write_text(json.dumps(results))
                with self.assertRaises(ValueError):
                    compare_rewards.validate_smoke(path, factory, 150)
                results["fixed_cycle_test"]["episodes"][0]["truncated"] = False
                path.write_text(json.dumps(results))
                source.write_text("# changed")
                with self.assertRaises(ValueError):
                    compare_rewards.validate_smoke(path, factory, 150)


if __name__ == "__main__":
    unittest.main()
