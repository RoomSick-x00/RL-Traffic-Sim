from traffic_env import TrafficEnvironment


env = TrafficEnvironment("peak")


# ==========================================
# RUN MULTIPLE EPISODES
# ==========================================

for episode in range(3):

    print("\n===================================")
    print("Episode:", episode + 1)
    print("===================================")

    state = env.reset()

    print("Initial State:")
    print(state)

    for step in range(20):

        # Example action
        action = 0

        next_state, reward, done = env.step(action)

        print(
            "Step:", step,
            "| Action:", action,
            "| Reward:", reward,
            "| Phase:", next_state[12]
        )

        if done:
            break

    # Close SUMO after this episode
    env.close()


print("\nAll episodes completed.")