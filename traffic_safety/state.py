from dataclasses import dataclass
from enum import Enum


class PedestrianState(str, Enum):
    OUTSIDE = "outside"
    WAITING = "waiting"
    ENTERING = "entering"
    CROSSING = "crossing"
    LEAVING = "leaving"


class VehicleState(str, Enum):
    OUTSIDE = "outside"
    APPROACHING = "approaching"
    SLOWING = "slowing"
    STOPPED = "stopped"
    PASSING = "passing"


@dataclass
class ObjectState:
    track_id: int
    state: str


class StateEstimator:
    def __init__(
        self,
        stopped_speed_threshold: float = 2.0,
        slowing_threshold: float = -2.0,
    ):
        self.stopped_speed_threshold = stopped_speed_threshold
        self.slowing_threshold = slowing_threshold

        self.previous_zones = {}
        self.previous_speeds = {}
        self.previous_states = {}

    def update(self, obj):
        if obj.category == "pedestrian":
            return self._update_pedestrian(obj)

        if obj.category == "vehicle":
            return self._update_vehicle(obj)

        return "outside"

    def _update_pedestrian(self, obj):
        previous_zone = self.previous_zones.get(
            obj.track_id
        )

        current_zone = obj.zone

        if current_zone is None:
            state = PedestrianState.OUTSIDE

        elif current_zone in {
            "pedestrian_waiting_left",
            "pedestrian_waiting_right",
        }:
            if previous_zone == "crosswalk":
                state = PedestrianState.LEAVING
            else:
                state = PedestrianState.WAITING

        elif current_zone == "crosswalk":
            if previous_zone in {
                "pedestrian_waiting_left",
                "pedestrian_waiting_right",
            }:
                state = PedestrianState.ENTERING
            else:
                state = PedestrianState.CROSSING

        else:
            state = PedestrianState.OUTSIDE

        self.previous_zones[
            obj.track_id
        ] = current_zone

        self.previous_states[
            obj.track_id
        ] = state

        return state

    def _update_vehicle(self, obj):
        previous_speed = self.previous_speeds.get(
            obj.track_id
        )

        current_speed = obj.speed_kmh

        current_zone = obj.zone

        if current_zone != "vehicle_approach":
            state = VehicleState.OUTSIDE

        elif current_speed is None:
            state = VehicleState.APPROACHING

        elif current_speed <= self.stopped_speed_threshold:
            state = VehicleState.STOPPED

        elif previous_speed is not None:
            speed_delta = (
                current_speed - previous_speed
            )

            if speed_delta <= self.slowing_threshold:
                state = VehicleState.SLOWING
            else:
                state = VehicleState.APPROACHING

        else:
            state = VehicleState.APPROACHING

        self.previous_speeds[
            obj.track_id
        ] = current_speed

        self.previous_states[
            obj.track_id
        ] = state

        return state