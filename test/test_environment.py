from traffic_env import TrafficEnvironment

env = TrafficEnvironment("peak")

print("Starting environment...")

state = env.reset()

print("\nInitial State:")
print(state)

print("\nRunning environment...")

for step in range(20):

    if step == 5:
        action = 1
    else:
        action = 0

    next_state, reward, done = env.step(action)

    print(
        "Step:", step,
        "| Action:", action,
        "| Reward:", reward,
        "| Phase:", next_state[12],
        "| State:", next_state
    )

    if done:
        break

env.close()