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

max_vehicles = {
    "North": 0,
    "South": 0,
    "East": 0,
    "West": 0
}

max_waiting = {
    "North": 0,
    "South": 0,
    "East": 0,
    "West": 0
}

step_of_max_vehicles = {}
step_of_max_waiting = {}

step = 0

while traci.simulation.getMinExpectedNumber() > 0:

    traci.simulationStep()
    step += 1

    for direction, lane in LANES.items():

        vehicles = traci.lane.getLastStepVehicleNumber(lane)
        waiting = traci.lane.getWaitingTime(lane)

        if vehicles > max_vehicles[direction]:
            max_vehicles[direction] = vehicles
            step_of_max_vehicles[direction] = step

        if waiting > max_waiting[direction]:
            max_waiting[direction] = waiting
            step_of_max_waiting[direction] = step

traci.close(wait=False)

print("\n========== STATE RANGE TEST ==========")

print("\nMaximum vehicles on each lane:")

for direction in LANES:

    print(
        direction,
        "| Maximum vehicles =",
        max_vehicles[direction],
        "| Step =",
        step_of_max_vehicles.get(direction, "-")
    )

print("\nMaximum waiting time on each lane:")

for direction in LANES:

    print(
        direction,
        "| Maximum waiting time =",
        max_waiting[direction],
        "seconds",
        "| Step =",
        step_of_max_waiting.get(direction, "-")
    )