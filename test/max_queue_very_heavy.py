import traci

SUMO_CMD = [
    "sumo",
    "-c",
    "neeligadi_very_heavy.sumocfg"
]

LANES = {
    "North": "-E4_0",
    "South": "-E5_0",
    "East": "-E6_0",
    "West": "E3_0"
}

traci.start(SUMO_CMD)

max_queue = {
    "North": 0,
    "South": 0,
    "East": 0,
    "West": 0
}

step = 0

while traci.simulation.getMinExpectedNumber() > 0:

    traci.simulationStep()
    step += 1

    for direction, lane in LANES.items():

        queue = traci.lane.getLastStepHaltingNumber(lane)

        if queue > max_queue[direction]:
            max_queue[direction] = queue

traci.close(wait=False)

print("\n========== VERY HEAVY SCENARIO ==========")

for direction in LANES:
    print(
        direction,
        "maximum queue =",
        max_queue[direction],
        "vehicles"
    )

print("Total simulation steps:", step)