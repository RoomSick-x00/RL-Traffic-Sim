def queue_reward(raw_state, action=0):
    queue = raw_state[4:8]
    return -float(sum(queue))


def wait_reward(raw_state, action=0):
    wait = raw_state[8:12]
    return -float(sum(wait))


def combined_reward(raw_state, applied_switch):
    wait = raw_state[8:12]
    wait_penalty = sum(wait) / 500
    switch_penalty = applied_switch

    alpha = 1.0
    beta = 1.0  # Waiting is normalized to a comparable scale.
    return -(alpha * wait_penalty) - (beta * switch_penalty)
