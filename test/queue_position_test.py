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

max_total_queue = 0
max_step = 0
max_data = {}

step = 0

while traci.simulation.getMinExpectedNumber() > 0:

    traci.simulationStep()
    step += 1

    current_data = {}
    total_queue = 0

    for direction, lane in LANES.items():

        vehicles = traci.lane.getLastStepVehicleIDs(lane)

        queued = []

        for vehicle_id in vehicles:

            speed = traci.vehicle.getSpeed(vehicle_id)

            if speed < 0.1:
                position = traci.vehicle.getLanePosition(vehicle_id)

                queued.append(
                    (vehicle_id, position)
                )

        current_data[direction] = queued
        total_queue += len(queued)

    if total_queue > max_total_queue:

        max_total_queue = total_queue
        max_step = step
        max_data = current_data

traci.close(wait=False)

print("\n========== MAXIMUM QUEUE POSITION TEST ==========")

print("Maximum total queue:", max_total_queue)
print("At simulation step:", max_step)

for direction in LANES:

    print("\n", direction)

    queued = max_data[direction]

    print("Queue:", len(queued), "vehicles")

    for vehicle_id, position in queued:

        print(
            "  Vehicle:",
            vehicle_id,
            "| Position:",
            round(position, 2),
            "m"
        )