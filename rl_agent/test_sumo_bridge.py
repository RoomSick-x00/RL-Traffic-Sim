"""Check TraCI connection ownership and action behavior without installing SUMO."""

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch


class SumoBridgeTests(unittest.TestCase):
    def setUp(self):
        self.traci = Mock()
        path = Path(__file__).resolve().parent.parent / "test" / "traffic_env.py"
        spec = importlib.util.spec_from_file_location("traffic_env_under_test", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"traci": self.traci}):
            spec.loader.exec_module(module)
        self.Env = module.TrafficEnvironment

    def connection(self):
        connection = Mock()
        connection.getVersion.return_value = (22, "SUMO mock")
        connection.simulation.getDeltaT.return_value = 1.0
        connection.simulation.getArrivedNumber.return_value = 0
        connection.simulation.getDepartedNumber.return_value = 1
        connection.simulation.getStartingTeleportNumber.return_value = 0
        connection.simulation.getMinExpectedNumber.return_value = 2
        connection.lane.getLastStepVehicleNumber.return_value = 1
        connection.lane.getLastStepHaltingNumber.return_value = 0
        connection.lane.getWaitingTime.return_value = 0
        connection.trafficlight.getPhase.return_value = 0
        return connection

    def test_two_environments_use_their_own_connections_and_close_only_their_own(self):
        first, second = self.connection(), self.connection()
        self.traci.getConnection.side_effect = [first, second, self.connection()]
        training, evaluation = self.Env(agent_controlled=True), self.Env(agent_controlled=True)
        training.reset(seed=42)
        evaluation.reset(seed=100)
        self.assertNotEqual(training.label, evaluation.label)
        self.assertEqual(self.traci.start.call_args_list[0].kwargs,
                         {"label": training.label, "doSwitch": False})
        self.assertEqual(self.traci.start.call_args_list[0].args[0][-2:], ["--seed", "42"])
        evaluation.close()
        second.close.assert_called_once()
        first.close.assert_not_called()
        _, done = training.step(0)
        self.assertFalse(done)
        self.assertEqual(first.simulationStep.call_count, 2)
        self.assertEqual(second.simulationStep.call_count, 1)
        training.reset(seed=43)
        first.close.assert_called_once()
        evaluation.close()
        second.close.assert_called_once()
        training.close()
        self.traci.simulationStep.assert_not_called()
        self.traci.close.assert_not_called()

    def test_keep_holds_only_green_and_switch_does_not_skip_yellow(self):
        connection = self.connection()
        self.traci.getConnection.return_value = connection
        env = self.Env(agent_controlled=True)
        env.reset()
        env.apply_action(0)
        connection.trafficlight.setPhaseDuration.assert_called_once_with("1", 2.0)
        env.apply_action(1)
        connection.trafficlight.setPhase.assert_called_once_with("1", 1)
        connection.trafficlight.getPhase.return_value = 1
        connection.trafficlight.reset_mock()
        env.apply_action(0)
        env.apply_action(1)
        connection.trafficlight.setPhase.assert_not_called()
        connection.trafficlight.setPhaseDuration.assert_not_called()
        env.close()

    def test_native_baseline_does_not_hold_green(self):
        connection = self.connection()
        self.traci.getConnection.return_value = connection
        env = self.Env(agent_controlled=True)
        env.use_fixed_cycle()
        env.reset()
        env.step(0)
        connection.trafficlight.setPhaseDuration.assert_not_called()
        env.close()

    def test_get_state_is_read_only_and_reset_failure_closes(self):
        connection = self.connection()
        self.traci.getConnection.return_value = connection
        env = self.Env()
        env.reset()
        before = connection.simulationStep.call_count
        self.assertEqual(env.get_state(), [1] * 4 + [0] * 8 + [0])
        self.assertEqual(connection.simulationStep.call_count, before)
        connection.simulationStep.side_effect = RuntimeError("failed")
        with self.assertRaises(RuntimeError):
            env.reset()
        self.assertIsNone(env.connection)
        self.assertEqual(connection.close.call_count, 2)


if __name__ == "__main__":
    unittest.main()
