import traci

sumo_cmd = [
    "sumo-gui",
    "-c",
    "neeligadi.sumocfg"
]

traci.start(sumo_cmd)

print("Connected to SUMO!")

for step in range(100):
    traci.simulationStep()
    print("Simulation step:", step)

traci.close()

print("Simulation finished.")