import traci

sumo_cmd = [
    "sumo-gui",
    "-c",
    "neeligadi.sumocfg"
]

traci.start(sumo_cmd)

traffic_light_id = "1"

print("Connected to SUMO!")

for step in range(60):

    # Change the signal at step 20
    if step == 20:
        print("Changing traffic signal...")
        traci.trafficlight.setPhase(traffic_light_id, 2)

    traci.simulationStep()

    phase = traci.trafficlight.getPhase(traffic_light_id)
    state = traci.trafficlight.getRedYellowGreenState(traffic_light_id)

    print(
        "Step:", step,
        "| Phase:", phase,
        "| Signal:", state
    )

traci.close(wait=False)