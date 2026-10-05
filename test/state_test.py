import traci

# -----------------------------
# SUMO configuration
# -----------------------------
SUMO_CMD = [
    "sumo",
    "-c",
    "neeligadi.sumocfg"
]

# Traffic light ID
TRAFFIC_LIGHT_ID = "1"

# Lane IDs from your SUMO network
LANES = [
    "-E3_0",
    "-E4_0",
    "-E5_0",
    "-E6_0",
    "E3_0",
    "E4_0",
    "E5_0",
    "E6_0"
]


# -----------------------------
# Get current traffic state
# -----------------------------
def get_state():

    state = {}

    # Current traffic-light phase
    state["phase"] = traci.trafficlight.getPhase(
        TRAFFIC_LIGHT_ID
    )

    # Traffic information for every lane
    for lane in LANES:

        state[lane] = {
            "vehicles": traci.lane.getLastStepVehicleNumber(lane),

            "queue": traci.lane.getLastStepHaltingNumber(lane),

            "waiting_time": traci.lane.getWaitingTime(lane)
        }

    return state


# -----------------------------
# Start SUMO
# -----------------------------
traci.start(SUMO_CMD)

print("Connected to SUMO!")


# -----------------------------
# Run simulation
# -----------------------------
for step in range(20):

    traci.simulationStep()


# -----------------------------
# Get current state
# -----------------------------
state = get_state()


# -----------------------------
# Display state
# -----------------------------
print("\n========== CURRENT STATE ==========")

print(
    "Traffic Light Phase:",
    state["phase"]
)

print("-----------------------------------")

for lane in LANES:

    print(
        lane,
        "| Vehicles:", state[lane]["vehicles"],
        "| Queue:", state[lane]["queue"],
        "| Waiting Time:", state[lane]["waiting_time"]
    )


# -----------------------------
# Close TraCI
# -----------------------------
traci.close(wait=False)

print("-----------------------------------")
print("TraCI closed.")