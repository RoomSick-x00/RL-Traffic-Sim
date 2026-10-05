import traci

SUMO_CMD = [
    "sumo",
    "-c",
    "neeligadi.sumocfg"
]

traci.start(SUMO_CMD)

print("Connected to SUMO!\n")

INCOMING_LANES = [
    "-E4_0",
    "-E5_0",
    "-E6_0",
    "E3_0"
]

for lane_id in INCOMING_LANES:

    shape = traci.lane.getShape(lane_id)

    print("Lane:", lane_id)
    print("Start:", shape[0])
    print("End:", shape[-1])
    print()

traci.close(wait=False)