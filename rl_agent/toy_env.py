class ToyTrafficEnv:
    def reset(self):
        self.queues = [0, 0, 0, 0]  # North, South, East, West
        self.waiting = [0, 0, 0, 0]
        self.phase = 0

        # The toy environment uses queue length as its vehicle-count estimate.
        state = (
            self.queues.copy()
            + self.queues.copy()
            + self.waiting.copy()
            + [self.phase]
        )
        return state

    def _update_phase(self, action):
        if self.phase == 1:
            self.phase = 2
        elif self.phase == 3:
            self.phase = 0
        elif action == 1 and self.phase == 0:
            self.phase = 1
        elif action == 1 and self.phase == 2:
            self.phase = 3