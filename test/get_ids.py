import traci

sumo_cmd = [
    "sumo",
    "-c",
    "neeligadi.sumocfg"
]

traci.start(sumo_cmd)

print("Connected to SUMO!")

print("\nTraffic Light IDs:")
print(traci.trafficlight.getIDList())

# Advance one step so vehicles are loaded
traci.simulationStep()

print("\nVehicle IDs:")
print(traci.vehicle.getIDList())

traci.close(wait=False)

print("\nTraCI closed.")