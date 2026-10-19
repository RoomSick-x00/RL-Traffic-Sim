import random

class ToyTrafficEnv:
    def __init__(self):
        self.rng = random.Random()

    def reset(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)

        self.queues = {"N":0, "S":0, "E":0, "W":0}  
        self.waiting = {"N":0, "S":0, "E":0, "W":0}
        self.phase = 0
        self.step_count = 0

        return self.get_state()

    def step(self, action):
        self._update_phase(action)
        self._update_queues()
        self._update_waiting()
        self.step_count += 1

        state = self.get_state()
        done = self.step_count >= 500
        return state, done

    def close(self):
        pass  # The toy environment has no external process to shut down.

    def get_state(self):
        directions = ("N", "S", "E", "W")
        queue_list = [self.queues[direction] for direction in directions]
        waiting_list = [self.waiting[direction] for direction in directions]
        
        # Use queues as vehicle counts in the toy model to match SUMO's 13-value state.
        return queue_list + queue_list + waiting_list + [self.phase]

    def _update_phase(self, action):
        if self.phase == 1:
            self.phase = 2
        elif self.phase == 3:
            self.phase = 0
        elif action == 1 and self.phase == 0:
            self.phase = 1
        elif action == 1 and self.phase == 2:
            self.phase = 3

    def _update_queues(self):
        if self.phase == 0:
            if self.rng.random() < 0.6:
                self.queues["N"] = max(0, self.queues["N"] - 1)
        else:
            if self.rng.random() < 0.4:
                self.queues["N"] += 1

        if self.phase == 0:
            if self.rng.random() < 0.6:
                self.queues["S"] = max(0, self.queues["S"] - 1)
        else:
            if self.rng.random() < 0.4:
                self.queues["S"] += 1

        if self.phase == 2:
            if self.rng.random() < 0.6:
                self.queues["E"] = max(0, self.queues["E"] - 1)
        else:
            if self.rng.random() < 0.4:
                self.queues["E"] += 1

        if self.phase == 2:
            if self.rng.random() < 0.6:
                self.queues["W"] = max(0, self.queues["W"] - 1)
        else:
            if self.rng.random() < 0.4:
                self.queues["W"] += 1

    def _update_waiting(self):
        green_phases = {"N": 0, "S": 0, "E": 2, "W": 2}

        for direction, green_phase in green_phases.items():
            if self.phase == green_phase:
                self.waiting[direction] = 0
            else:
                self.waiting[direction] += self.queues[direction]
                
                
if __name__ == "__main__":
    env = ToyTrafficEnv()
    state = env.reset()
    print("Initial state:", state)

    for i in range(20):
        action = random.choices([0, 1], weights=[65, 35], k=1)[0]  # just testing with "keep= 0" "switch = 1" for now
        state, done = env.step(action)
        print(f"Step {i}: state={state}, done={done}")
        if done:
            break
