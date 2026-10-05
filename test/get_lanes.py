import traci

sumo_cmd = [
    "sumo",
    "-c",
    "neeligadi.sumocfg"
]

traci.start(sumo_cmd)

print("Connected to SUMO!")

print("\nLane IDs:")
print(traci.lane.getIDList())

traci.close(wait=False)