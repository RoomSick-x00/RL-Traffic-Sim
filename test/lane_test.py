import traci

sumo_cmd = [
    "sumo",
    "-c",
    "neeligadi.sumocfg"
]

traci.start(sumo_cmd)

print("Connected to SUMO!")

# Run the simulation for a few steps so vehicles enter
for step in range(20):
    traci.simulationStep()

print("\nLane information:\n")

for lane_id in traci.lane.getIDList():

    # Ignore internal junction lanes
    if lane_id.startswith(":"):
        continue

    vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
    vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)
    halting_count = traci.lane.getLastStepHaltingNumber(lane_id)

    print(
        f"Lane: {lane_id} | "
        f"Vehicles: {vehicle_count} | "
        f"Stopped/Queue: {halting_count} | "
        f"Vehicle IDs: {vehicle_ids}"
    )

traci.close(wait=False)