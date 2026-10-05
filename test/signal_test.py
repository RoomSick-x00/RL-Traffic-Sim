import traci

sumo_cmd = [
    "sumo",
    "-c",
    "neeligadi.sumocfg"
]

traci.start(sumo_cmd)

print("Connected to SUMO!")

traffic_light_id = "1"

print("Traffic Light ID:", traffic_light_id)

for step in range(10):
    traci.simulationStep()

    phase = traci.trafficlight.getPhase(traffic_light_id)
    state = traci.trafficlight.getRedYellowGreenState(traffic_light_id)

    print(
        "Step:", step,
        "| Phase:", phase,
        "| Signal:", state
    )

traci.close(wait=False)