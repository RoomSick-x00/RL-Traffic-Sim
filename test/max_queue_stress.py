import traci

SUMO_CMD = [
    "sumo",
    "-c",
    "neeligadi_stress.sumocfg"
]

LANE = "-E4_0"

traci.start(SUMO_CMD)

max_queue = 0
max_step = 0
max_vehicles = []

step = 0

while traci.simulation.getMinExpectedNumber() > 0:

    traci.simulationStep()
    step += 1

    vehicles = traci.lane.getLastStepVehicleIDs(LANE)

    queued = []

    for vehicle_id in vehicles:

        speed = traci.vehicle.getSpeed(vehicle_id)

        if speed < 0.1:
            position = traci.vehicle.getLanePosition(vehicle_id)

            queued.append(
                (vehicle_id, position)
            )

    if len(queued) > max_queue:

        max_queue = len(queued)
        max_step = step
        max_vehicles = queued.copy()

traci.close(wait=False)

print("\n========== SINGLE-LANE STRESS TEST ==========")

print("Maximum queue:", max_queue, "vehicles")
print("At simulation step:", max_step)

print("\nQueued vehicles and positions:")

for vehicle_id, position in max_vehicles:

    print(
        "Vehicle:",
        vehicle_id,
        "| Position:",
        round(position, 2),
        "m"
    )