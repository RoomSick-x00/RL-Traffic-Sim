import traci


class TrafficEnvironment:

    def __init__(self, scenario="normal"):

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
            config_file
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

    def reset(self):

        # Start SUMO
        traci.start(self.sumo_cmd)

        self.step_count = 0

        # Advance simulation
        traci.simulationStep()

        # Return initial state
        return self.get_state()

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
                traci.lane.getLastStepVehicleNumber(lane)
            )

            state.append(vehicle_count)

        # ----------------------------------------------
        # 2. QUEUE LENGTH
        # ----------------------------------------------

        for direction in ["north", "south", "east", "west"]:

            lane = self.lanes[direction]

            queue = (
                traci.lane.getLastStepHaltingNumber(lane)
            )

            state.append(queue)

        # ----------------------------------------------
        # 3. WAITING TIME
        # ----------------------------------------------

        for direction in ["north", "south", "east", "west"]:

            lane = self.lanes[direction]

            waiting_time = (
                traci.lane.getWaitingTime(lane)
            )

            state.append(waiting_time)

        # ----------------------------------------------
        # 4. TRAFFIC LIGHT PHASE
        # ----------------------------------------------

        phase = traci.trafficlight.getPhase(
            self.traffic_light_id
        )

        state.append(phase)

        return state

    # ==================================================
    # APPLY ACTION
    # ==================================================

    def apply_action(self, action):

        current_phase = traci.trafficlight.getPhase(
            self.traffic_light_id
        )

        # ----------------------------------------------
        # ACTION 0
        # Keep current phase
        # ----------------------------------------------

        if action == 0:
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

                traci.trafficlight.setPhase(
                    self.traffic_light_id,
                    1
                )

            # ------------------------------------------
            # Phase 2 = East-West GREEN
            # Phase 3 = Yellow
            # ------------------------------------------

            elif current_phase == 2:

                traci.trafficlight.setPhase(
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
        traci.simulationStep()

        self.step_count += 1

        # Get next state
        next_state = self.get_state()

        # ----------------------------------------------
        # REWARD
        # ----------------------------------------------

        total_queue = (
            next_state[4]
            + next_state[5]
            + next_state[6]
            + next_state[7]
        )

        # Negative queue length
        reward = -total_queue

        # ----------------------------------------------
        # CHECK IF SIMULATION IS FINISHED
        # ----------------------------------------------

        done = (
            traci.simulation.getMinExpectedNumber() <= 0
        )

        return next_state, reward, done

    # ==================================================
    # CLOSE
    # ==================================================

    def close(self):

        traci.close(wait=False)

        print("Environment closed.")