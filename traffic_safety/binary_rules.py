"""Binary OK/DANGER rule engine (Phase I, variant V1).

A frame is DANGER when at least one pedestrian-vehicle pair satisfies:

R1 VEHICLE_ON_CROSSWALK_WITH_PEDESTRIAN
    pedestrian.zone == "crosswalk"
    and vehicle.zone == "crosswalk"
    and vehicle.speed_kmh > moving_speed_kmh

R2 VEHICLE_ARRIVING_WHILE_PEDESTRIAN_CROSSING
    pedestrian.zone == "crosswalk"
    and vehicle.zone == "vehicle_approach"
    and 0 <= vehicle TTA <= arrival_time_s

TTA (time to arrival at the crosswalk reference line) is read from
``ProcessedObject.ttc_crosswalk_s``. The pipeline sets it to None when the
vehicle is not approaching (stopped or moving away), so such vehicles never
trigger R2.

Missing, NaN and infinite speed/TTA values never trigger a rule.
Objects of other categories (e.g. bicycles) are ignored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


OK = 0
DANGER = 1

CROSSWALK_ZONE = "crosswalk"
VEHICLE_APPROACH_ZONE = "vehicle_approach"


class BinaryRule(str, Enum):
    VEHICLE_ON_CROSSWALK_WITH_PEDESTRIAN = (
        "vehicle_on_crosswalk_with_pedestrian"
    )
    VEHICLE_ARRIVING_WHILE_PEDESTRIAN_CROSSING = (
        "vehicle_arriving_while_pedestrian_crossing"
    )


@dataclass(frozen=True)
class BinaryRiskEvent:
    pedestrian_id: int
    vehicle_id: int
    rule: BinaryRule
    tta_s: float | None
    vehicle_speed_kmh: float | None


def _finite_or_none(value) -> float | None:
    """Return value as float, or None if missing / non-numeric / NaN / inf."""
    if value is None or isinstance(value, bool):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


def _validate_threshold(name: str, value, allow_zero: bool) -> float:
    number = _finite_or_none(value)

    if number is None:
        raise ValueError(f"{name} must be a finite number, got {value!r}")

    if number < 0 or (number == 0 and not allow_zero):
        bound = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be {bound}, got {value!r}")

    return number


class BinaryRuleEngine:
    def __init__(
        self,
        arrival_time_s: float = 4.0,
        moving_speed_kmh: float = 2.0,
    ):
        self.arrival_time_s = _validate_threshold(
            "arrival_time_s",
            arrival_time_s,
            allow_zero=False,
        )
        self.moving_speed_kmh = _validate_threshold(
            "moving_speed_kmh",
            moving_speed_kmh,
            allow_zero=True,
        )

    def evaluate(self, objects) -> list[BinaryRiskEvent]:
        """Return one event per pedestrian-vehicle pair satisfying R1 or R2."""
        pedestrians = [
            obj
            for obj in objects
            if obj.category == "pedestrian"
            and obj.zone == CROSSWALK_ZONE
        ]

        if not pedestrians:
            return []

        vehicles = [
            obj
            for obj in objects
            if obj.category == "vehicle"
        ]

        events = []

        for vehicle in vehicles:
            rule = self._vehicle_rule(vehicle)

            if rule is None:
                continue

            for pedestrian in pedestrians:
                events.append(
                    BinaryRiskEvent(
                        pedestrian_id=pedestrian.track_id,
                        vehicle_id=vehicle.track_id,
                        rule=rule,
                        tta_s=_finite_or_none(vehicle.ttc_crosswalk_s),
                        vehicle_speed_kmh=_finite_or_none(vehicle.speed_kmh),
                    )
                )

        return events

    def classify(self, objects) -> int:
        """Frame decision: DANGER (1) or OK (0)."""
        return self.frame_label(self.evaluate(objects))

    @staticmethod
    def frame_label(events) -> int:
        return DANGER if events else OK

    def _vehicle_rule(self, vehicle) -> BinaryRule | None:
        """
        Both rules require a pedestrian on the crosswalk; given that, whether
        a pair is dangerous depends only on the vehicle. The zones are
        mutually exclusive, so at most one rule applies to a vehicle.
        """
        if vehicle.zone == CROSSWALK_ZONE:
            speed = _finite_or_none(vehicle.speed_kmh)

            if speed is not None and speed > self.moving_speed_kmh:
                return BinaryRule.VEHICLE_ON_CROSSWALK_WITH_PEDESTRIAN

            return None

        if vehicle.zone == VEHICLE_APPROACH_ZONE:
            tta = _finite_or_none(vehicle.ttc_crosswalk_s)

            if tta is not None and 0 <= tta <= self.arrival_time_s:
                return BinaryRule.VEHICLE_ARRIVING_WHILE_PEDESTRIAN_CROSSING

            return None

        return None
