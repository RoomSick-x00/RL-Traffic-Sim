"""Read-only junction logging, units, lane scope, and CLI wiring."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from .junction_diagnostics import JunctionDiagnostics
from .training import evaluate_policy
from . import evaluate_saved, replay


class TinySumo:
    def __init__(self):
        self.traffic_light_id = "1"
        self.lanes = {"north": "N", "south": "S", "east": "E", "west": "W"}
        self.sumo_cmd = ["sumo", "-c", "network.sumocfg"]
        self.connection = Mock()
        c = self.connection
        c.lane.getIDList.return_value = ["N", "S", "E", "W", ":1_5_0", ":11_5_0", "out"]
        c.lane.getLastStepHaltingNumber.side_effect = lambda lane: {"N": 2, ":1_5_0": 1}.get(lane, 0)
        c.simulation.getDeltaT.return_value = 0.5
        c.simulation.getTime.side_effect = lambda: 1 + self.steps * 0.5
        c.trafficlight.getControlledLinks.return_value = [(('E', 'out', ':1_5_0'),)]
        c.trafficlight.getRedYellowGreenState.return_value = "g"
        c.vehicle.getIDList.return_value = ["incoming", "inside", "outgoing", "other_junction"]
        c.vehicle.getLaneID.side_effect = {"incoming": "N", "inside": ":1_5_0",
                                          "outgoing": "out", "other_junction": ":11_5_0"}.get
        c.vehicle.getSpeed.side_effect = {"incoming": 0.0, "inside": 0.0}.get
        c.vehicle.getRoute.return_value = ["E", "out"]

    def reset(self, seed=None):
        self.steps = 0
        self.closed = False
        return self.get_state()

    def get_state(self):
        return [2, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, min(self.steps, 2)]

    def step(self, action):
        self.steps += 1
        return self.get_state(), self.steps >= 2

    def get_diagnostics(self):
        return {"remaining": 0}

    def close(self):
        self.closed = True


class JunctionDiagnosticTests(unittest.TestCase):
    def test_saved_model_test_split_uses_frozen_seeds_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            (run / "best_model.zip").write_bytes(b"mock")
            (run / "config.json").write_text(json.dumps({
                "environment": {"backend": "sumo", "scenario": "normal"},
                "reward_function": "combined_reward",
                "reward_parameters": {"switch_penalty": 0.02},
                "max_steps": 150, "validation_seeds": [100], "test_seeds": [],
            }))
            (run / "validation.json").write_text(json.dumps([
                {"timesteps": 60000, "mean_reward": -0.2}]))
            model = Mock()
            model.predict.return_value = (0, None)
            with patch.object(evaluate_saved.DQN, "load", return_value=model), \
                    patch.object(evaluate_saved, "EnvironmentFactory",
                                 return_value=TinySumo), \
                    patch.object(evaluate_saved, "TEST_SEEDS", (1000, 1001)):
                saved = evaluate_saved.evaluate_run(run, root / "out", split="test")
            result = json.loads(saved.read_text())
            self.assertEqual(result["evaluation_split"], "test")
            self.assertEqual(result["evaluation_seeds"], [1000, 1001])
            self.assertEqual(result["best_test"]["episode_seeds"], [1000, 1001])
            self.assertEqual(result["fixed_cycle_test"]["episode_seeds"], [1000, 1001])
            self.assertEqual(result["original_validation_best_step"], 60000)
            self.assertNotIn("difference_from_original_validation", result)
            model.learn.assert_not_called()

    def test_units_scope_routes_and_zero_time(self):
        raw = TinySumo()
        raw.reset()
        d = JunctionDiagnostics(raw)
        self.assertIsNone(d.result()["internal_stopped_fraction"])
        for step in (1, 2):
            raw.steps = step
            d.sample(step, 1, 0, 1)
        r = d.result()
        self.assertEqual(r["incoming_stopped_vehicle_seconds"], 2)
        self.assertEqual(r["internal_stopped_vehicle_seconds"], 1)
        self.assertEqual(r["total_stopped_vehicle_seconds"], 3)
        self.assertEqual(r["measured_seconds"], 1)
        self.assertAlmostEqual(r["internal_stopped_fraction"], 1/3)
        self.assertEqual(r["internal_lanes"], [":1_5_0"])
        self.assertEqual(set(r["vehicle_routes"]), {"incoming", "inside"})
        self.assertEqual(raw.connection.vehicle.getRoute.call_count, 2)
        self.assertEqual(len(r["trace"]), 2)
        raw.connection.simulation.simulationStep.assert_not_called()
        raw.connection.trafficlight.setPhase.assert_not_called()
        raw.connection.lane.getWaitingTime.assert_not_called()

    def test_evaluation_preserves_original_metrics_and_adds_each_episode(self):
        model = Mock()
        model.predict.return_value = (1, None)
        original = evaluate_policy(model=model, seeds=(100, 101), env_factory=TinySumo)
        measured = evaluate_policy(model=model, seeds=(100, 101), env_factory=TinySumo,
                                   junction_diagnostics=True)
        diagnostic = measured.pop("junction_diagnostics")
        self.assertEqual(diagnostic["mean_internal_stopped_vehicle_seconds"], 1)
        for episode in measured["episodes"]:
            self.assertEqual(episode.pop("junction_diagnostics")["incoming_stopped_vehicle_seconds"], 2)
        self.assertEqual(measured, original)

    def test_replay_saves_trace_and_closes_without_learning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "best_model.zip"
            model_path.write_bytes(b"mock")
            (root / "config.json").write_text(json.dumps({
                "environment": {"backend": "sumo", "scenario": "normal"},
                "reward_function": "queue_reward", "max_steps": 150}))
            model = Mock()
            model.predict.return_value = (1, None)
            raw = TinySumo()
            with patch.object(replay, "__file__", str(root / "rl_agent" / "replay.py")), \
                    patch.object(replay.shutil, "which", return_value="sumo-gui"), \
                    patch.object(replay.DQN, "load", return_value=model), \
                    patch.object(replay, "EnvironmentFactory", return_value=lambda: raw):
                summary = replay.replay(model_path, wait=False, junction_diagnostics=True)
            files = list((root / "logs").glob("junction_replay_*.json"))
            self.assertEqual(len(files), 1)
            saved = json.loads(files[0].read_text())
            self.assertEqual(saved["summary"], summary)
            self.assertEqual(len(saved["junction_diagnostics"]["trace"]), 2)
            self.assertTrue(raw.closed)
            model.learn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
