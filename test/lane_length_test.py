import traci

SUMO_CMD = [
    "sumo",
    "-c",
    "neeligadi_peak.sumocfg"
]

LANES = {
    "North": "-E4_0",
    "South": "-E5_0",
    "East": "-E6_0",
    "West": "E3_0"
}

traci.start(SUMO_CMD)

print("\n========== APPROACH LANE LENGTHS ==========")

for direction, lane in LANES.items():

    length = traci.lane.getLength(lane)

    print(
        direction,
        "| Lane:", lane,
        "| Length:", length, "meters"
    )

traci.close(wait=False)