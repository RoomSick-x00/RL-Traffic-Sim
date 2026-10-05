import traci

SUMO_CMD = [
    "sumo",
    "-c",
    "neeligadi.sumocfg"
]

traci.start(SUMO_CMD)

print("Connected to SUMO!\n")

for lane_id in traci.lane.getIDList():

    # Ignore internal junction lanes
    if lane_id.startswith(":"):
        continue

    links = traci.lane.getLinks(lane_id)

    print("Lane:", lane_id)
    print("  Length:", traci.lane.getLength(lane_id))
    print("  Links:", links)
    print()

traci.close(wait=False)