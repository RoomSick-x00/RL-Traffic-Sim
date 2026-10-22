from pathlib import Path
from uuid import uuid4

import traci


class TrafficEnvironment:

    def __init__(self, scenario="normal", agent_controlled=False):

        if scenario not in ("normal", "peak"):
            raise ValueError("scenario must be normal or peak")
        self.agent_controlled = agent_controlled
        self.label = f"traffic_{uuid4().hex}"
        self.connection = None
        self.arrived = self.departed = self.teleports = 0

        # ----------------------------------------------
        # SUMO CONFIGURATION
        # ----------------------------------------------

        if scenario == "peak":
            config_file = "neeligadi_peak.sumocfg"
        else:
            config_file = "neeligadi.sumocfg"

        self.sumo_cmd = [
            "sumo",
            "-c",
            str(Path(__file__).resolve().parent / config_file)
        ]

        # ----------------------------------------------
        # TRAFFIC LIGHT
        # ----------------------------------------------

        self.traffic_light_id = "1"

        # ----------------------------------------------
        # INCOMING LANES
        # ----------------------------------------------

        self.lanes = {
            "north": "-E4_0",
            "south": "-E5_0",
            "east": "-E6_0",
            "west": "E3_0"
        }

        # ----------------------------------------------
        # SIMULATION STEP
        # ----------------------------------------------

        self.step_count = 0

    # ==================================================
    # RESET ENVIRONMENT
    # ==================================================

    def reset(self, seed=None):

        self.close()
        # Start SUMO
        sumo_cmd = self.sumo_cmd.copy()
        if seed is not None:
            sumo_cmd.extend(["--seed", str(seed)])
        traci.start(sumo_cmd, label=self.label, doSwitch=False)
        self.connection = traci.getConnection(self.label)
        try:
            self.step_count = 0
            self.arrived = self.departed = self.teleports = 0
            self.sumo_version = self.connection.getVersion()[1]
            self.connection.simulationStep()
            self._record_counts()
            return self.get_state()
        except BaseException:
            self.close()
            raise

    # ==================================================
    # GET STATE
    # ==================================================

    def get_state(self):

        state = []

        # ----------------------------------------------
        # 1. VEHICLE COUNTS
        # ----------------------------------------------

        for direction in ["north", "south", "east", "west"]:

            lane = self.lanes[direction]

            vehicle_count = (
                self.connection.lane.getLastStepVehicleNumber(lane)
            )

            state.append(vehicle_count)

        # ----------------------------------------------
        # 2. QUEUE LENGTH
        # ----------------------------------------------

        for direction in ["north", "south", "east", "west"]:

            lane = self.lanes[direction]

            queue = (
                self.connection.lane.getLastStepHaltingNumber(lane)
            )

            state.append(queue)

        # ----------------------------------------------
        # 3. WAITING TIME
        # ----------------------------------------------

        for direction in ["north", "south", "east", "west"]:

            lane = self.lanes[direction]

            waiting_time = (
                self.connection.lane.getWaitingTime(lane)
            )

            state.append(waiting_time)

        # ----------------------------------------------
        # 4. TRAFFIC LIGHT PHASE
        # ----------------------------------------------

        phase = self.connection.trafficlight.getPhase(
            self.traffic_light_id
        )

        state.append(phase)

        return state

    # ==================================================
    # APPLY ACTION
    # ==================================================

    def apply_action(self, action):

        if action not in (0, 1):
            raise ValueError("action must be 0 (keep) or 1 (switch)")
        current_phase = self.connection.trafficlight.getPhase(
            self.traffic_light_id
        )

        # ----------------------------------------------
        # ACTION 0
        # Keep current phase
        # ----------------------------------------------

        if action == 0:
            if self.agent_controlled and current_phase in (0, 2):
                # Refresh green beyond the next simulation step. Yellow stays automatic.
                duration = self.connection.simulation.getDeltaT() + 1.0
                self.connection.trafficlight.setPhaseDuration(self.traffic_light_id, duration)
            return

        # ----------------------------------------------
        # ACTION 1
        # Request phase change
        # ----------------------------------------------

        if action == 1:

            # ------------------------------------------
            # Phase 0 = North-South GREEN
            # Phase 1 = Yellow
            # ------------------------------------------

            if current_phase == 0:

                self.connection.trafficlight.setPhase(
                    self.traffic_light_id,
                    1
                )

            # ------------------------------------------
            # Phase 2 = East-West GREEN
            # Phase 3 = Yellow
            # ------------------------------------------

            elif current_phase == 2:

                self.connection.trafficlight.setPhase(
                    self.traffic_light_id,
                    3
                )

            # ------------------------------------------
            # Already yellow
            # Don't interrupt it
            # ------------------------------------------

            elif current_phase in [1, 3]:

                return

    # ==================================================
    # STEP
    # ==================================================

    def step(self, action):

        # Apply action
        self.apply_action(action)

        # Advance SUMO
        self.connection.simulationStep()
        self._record_counts()

        self.step_count += 1

        # Get next state
        next_state = self.get_state()

        # ----------------------------------------------
        # CHECK IF SIMULATION IS FINISHED
        # ----------------------------------------------

        done = (
            self.connection.simulation.getMinExpectedNumber() <= 0
        )

        return next_state, done

    # ==================================================
    # CLOSE
    # ==================================================

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def use_fixed_cycle(self):
        """Call before reset to leave the network's native signal schedule in charge."""
        self.agent_controlled = False

    def _record_counts(self):
        simulation = self.connection.simulation
        self.arrived += simulation.getArrivedNumber()
        self.departed += simulation.getDepartedNumber()
        self.teleports += simulation.getStartingTeleportNumber()

    def get_diagnostics(self):
        return {
            "arrived": self.arrived,
            "departed": self.departed,
            "teleports": self.teleports,
            "remaining": self.connection.simulation.getMinExpectedNumber(),
            "simulation_seconds": self.connection.simulation.getTime(),
            "sumo_version": self.sumo_version,
        }
