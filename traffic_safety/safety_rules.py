from dataclasses import dataclass
from enum import Enum


class RiskLevel(str, Enum):
    SAFE = "safe"
    WARNING = "warning"
    DANGER = "danger"


@dataclass
class RiskEvent:
    pedestrian_id: int
    vehicle_id: int

    level: RiskLevel
    rule: str

    ttc_s: float | None
    distance_m: float | None


class SafetyRuleEngine:
    def __init__(
        self,
        warning_ttc_s: float = 4.0,
        danger_ttc_s: float = 2.5,
        fallback_distance_m: float = 10.0,
    ):
        self.warning_ttc_s = warning_ttc_s
        self.danger_ttc_s = danger_ttc_s
        self.fallback_distance_m = fallback_distance_m

    def evaluate(
        self,
        objects,
    ) -> list[RiskEvent]:

        pedestrians = [
            obj
            for obj in objects
            if obj.category == "pedestrian"
        ]

        vehicles = [
            obj
            for obj in objects
            if obj.category == "vehicle"
        ]

        events = []

        for pedestrian in pedestrians:
            for vehicle in vehicles:

                event = self._evaluate_pair(
                    pedestrian,
                    vehicle,
                )

                if event is not None:
                    events.append(event)

        return events

    def _evaluate_pair(
        self,
        pedestrian,
        vehicle,
    ) -> RiskEvent | None:

        # ---------------------------------------------
        # Ignore vehicles which are not relevant
        # ---------------------------------------------

        if vehicle.state in {
            "outside",
            "stopped",
        }:
            return None

        pedestrian_state = pedestrian.state

        # ---------------------------------------------
        # Rule 1
        # DANGER:
        # pedestrian is entering/crossing
        # and vehicle TTC is very small
        # ---------------------------------------------

        if (
            pedestrian_state in {
                "entering",
                "crossing",
            }
            and vehicle.ttc_crosswalk_s is not None
            and vehicle.ttc_crosswalk_s
            <= self.danger_ttc_s
        ):
            return RiskEvent(
                pedestrian_id=pedestrian.track_id,
                vehicle_id=vehicle.track_id,
                level=RiskLevel.DANGER,
                rule="pedestrian_crossing_low_ttc",
                ttc_s=vehicle.ttc_crosswalk_s,
                distance_m=vehicle.distance_to_crosswalk_m,
            )

        # ---------------------------------------------
        # Rule 2
        # WARNING:
        # pedestrian is entering/crossing
        # and vehicle is approaching with moderate TTC
        # ---------------------------------------------

        if (
            pedestrian_state in {
                "entering",
                "crossing",
            }
            and vehicle.ttc_crosswalk_s is not None
            and vehicle.ttc_crosswalk_s
            <= self.warning_ttc_s
        ):
            return RiskEvent(
                pedestrian_id=pedestrian.track_id,
                vehicle_id=vehicle.track_id,
                level=RiskLevel.WARNING,
                rule="pedestrian_crossing_vehicle_approaching",
                ttc_s=vehicle.ttc_crosswalk_s,
                distance_m=vehicle.distance_to_crosswalk_m,
            )

        # ---------------------------------------------
        # Rule 3
        # WARNING:
        # pedestrian waits near crossing
        # and vehicle is already very close
        # ---------------------------------------------

        if (
            pedestrian_state == "waiting"
            and vehicle.distance_to_crosswalk_m is not None
            and vehicle.distance_to_crosswalk_m
            <= self.fallback_distance_m
        ):
            return RiskEvent(
                pedestrian_id=pedestrian.track_id,
                vehicle_id=vehicle.track_id,
                level=RiskLevel.WARNING,
                rule="pedestrian_waiting_vehicle_close",
                ttc_s=vehicle.ttc_crosswalk_s,
                distance_m=vehicle.distance_to_crosswalk_m,
            )

        return None