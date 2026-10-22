"""Select an environment without importing SUMO dependencies for toy runs."""

from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
import sys

if __package__:
    from .toy_env import ToyTrafficEnv
else:
    from toy_env import ToyTrafficEnv


@dataclass(frozen=True)
class EnvironmentFactory:
    backend: str = "toy"
    scenario: str = "normal"

    def __post_init__(self):
        if self.backend not in ("toy", "sumo"):
            raise ValueError("backend must be toy or sumo")
        if self.scenario not in ("normal", "peak"):
            raise ValueError("scenario must be normal or peak")

    def __call__(self):
        if self.backend == "toy":
            return ToyTrafficEnv()
        if shutil.which("sumo") is None:
            raise RuntimeError("SUMO is not on PATH. Run this on the SUMO machine.")
        # Supports both python -m rl_agent.train and python rl_agent/train.py.
        root = str(Path(__file__).resolve().parent.parent)
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            from test.traffic_env import TrafficEnvironment
        except ModuleNotFoundError as exc:
            if exc.name == "traci":
                raise RuntimeError("This Python environment cannot import traci.") from exc
            raise
        return TrafficEnvironment(self.scenario, agent_controlled=True)

    def describe(self):
        return asdict(self)
