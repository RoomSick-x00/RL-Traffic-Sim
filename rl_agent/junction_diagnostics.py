"""Read-only SUMO measurements; never change observations, rewards, or signals."""

import math


class JunctionDiagnostics:
    def __init__(self, raw_env):
        self.connection = raw_env.connection
        self.light = raw_env.traffic_light_id
        all_lanes = set(self.connection.lane.getIDList())
        self.incoming = set(raw_env.lanes.values())
        self.internal = {lane for lane in all_lanes if lane.startswith(f":{self.light}_")}
        if not self.incoming <= all_lanes or not self.internal:
            raise ValueError("Cannot identify incoming/internal lanes for this SUMO junction")
        self.controlled_links = self.connection.trafficlight.getControlledLinks(self.light)
        self.incoming_seconds = self.internal_seconds = 0.0
        self.measured_seconds = 0.0
        self.trace = []
        self.routes = {}

    def sample(self, step, action, old_phase, phase):
        connection = self.connection
        dt = float(connection.simulation.getDeltaT())
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("SUMO step duration must be positive and finite")
        incoming_halted = sum(connection.lane.getLastStepHaltingNumber(lane)
                              for lane in sorted(self.incoming))
        internal_halted = sum(connection.lane.getLastStepHaltingNumber(lane)
                              for lane in sorted(self.internal))
        self.incoming_seconds += incoming_halted * dt
        self.internal_seconds += internal_halted * dt
        self.measured_seconds += dt
        vehicles = []
        for vehicle in sorted(connection.vehicle.getIDList()):
            lane = connection.vehicle.getLaneID(vehicle)
            if lane not in self.incoming and lane not in self.internal:
                continue
            if vehicle not in self.routes:
                self.routes[vehicle] = list(connection.vehicle.getRoute(vehicle))
            speed = float(connection.vehicle.getSpeed(vehicle))
            vehicles.append({"id": vehicle, "lane": lane,
                             "region": "internal" if lane in self.internal else "incoming",
                             "speed_m_s": speed, "stopped": speed < 0.1})
        self.trace.append({
            "step": step, "simulation_seconds": connection.simulation.getTime(),
            "step_seconds": dt, "action": int(action), "old_phase": old_phase, "phase": phase,
            "signal_state": connection.trafficlight.getRedYellowGreenState(self.light),
            "incoming_halted": incoming_halted, "internal_halted": internal_halted,
            "vehicles": vehicles,
        })

    def result(self):
        total = self.incoming_seconds + self.internal_seconds
        return {
            "definition": "Sum of per-step lane halting counts times SUMO step duration; "
                          "speed <0.1 m/s defines halting. Excludes the simulation step in reset, "
                          "outgoing roads, and vehicles not yet inserted. This is sampled stopped "
                          "vehicle-seconds in these regions, not total trip delay.",
            "junction_id": self.light,
            "incoming_lanes": sorted(self.incoming), "internal_lanes": sorted(self.internal),
            "controlled_links_by_signal_index": self.controlled_links,
            "vehicle_routes": self.routes,
            "measured_seconds": self.measured_seconds,
            "incoming_stopped_vehicle_seconds": self.incoming_seconds,
            "internal_stopped_vehicle_seconds": self.internal_seconds,
            "total_stopped_vehicle_seconds": total,
            "internal_stopped_fraction": self.internal_seconds / total if total else None,
            "trace": self.trace,
        }
