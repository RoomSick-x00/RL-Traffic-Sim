def queue_reward(raw_state):
    queue = raw_state[4:8]
    return -float(sum(queue))


def wait_reward(raw_state):
    wait = raw_state[8:12]
    return -float(sum(wait))
