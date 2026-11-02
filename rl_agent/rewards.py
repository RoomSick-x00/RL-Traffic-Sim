from functools import wraps
import math


def queue_reward(raw_state, action=0):
    queue = raw_state[4:8]
    return -float(sum(queue))


def wait_reward(raw_state, action=0):
    wait = raw_state[8:12]
    return -float(sum(wait))


def combined_reward(raw_state, applied_switch, *, switch_penalty=1.0):
    wait = raw_state[8:12]
    wait_penalty = sum(wait) / 500
    alpha = 1.0
    return -(alpha * wait_penalty) - (switch_penalty * applied_switch)


def make_combined_reward(switch_penalty=1.0):
    """Configure beta while keeping alpha=1 and the waiting divisor=500."""
    if not math.isfinite(switch_penalty) or switch_penalty < 0:
        raise ValueError("switch penalty must be finite and nonnegative")

    @wraps(combined_reward)
    def reward(raw_state, applied_switch):
        return combined_reward(raw_state, applied_switch, switch_penalty=switch_penalty)

    reward.reward_parameters = {"switch_penalty": float(switch_penalty)}
    return reward


def reward_from_config(config):
    """Reconstruct a saved reward, including legacy combined runs with beta=1."""
    if config["reward_function"] == "combined_reward":
        return make_combined_reward(**config.get("reward_parameters", {}))
    return {"queue_reward": queue_reward, "wait_reward": wait_reward}[config["reward_function"]]
