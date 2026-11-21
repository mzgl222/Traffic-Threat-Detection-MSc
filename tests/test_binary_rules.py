import math
import unittest
from types import SimpleNamespace

from traffic_safety.binary_rules import (
    DANGER,
    OK,
    BinaryRule,
    BinaryRuleEngine,
)


R1 = BinaryRule.VEHICLE_ON_CROSSWALK_WITH_PEDESTRIAN
R2 = BinaryRule.VEHICLE_ARRIVING_WHILE_PEDESTRIAN_CROSSING


def pedestrian(track_id=1, zone="crosswalk"):
    return SimpleNamespace(
        track_id=track_id,
        category="pedestrian",
        zone=zone,
        speed_kmh=None,
        ttc_crosswalk_s=None,
    )


def vehicle(track_id=100, zone="crosswalk", speed_kmh=None, tta_s=None):
    return SimpleNamespace(
        track_id=track_id,
        category="vehicle",
        zone=zone,
        speed_kmh=speed_kmh,
        ttc_crosswalk_s=tta_s,
    )


class BinaryRuleEngineTest(unittest.TestCase):
    def setUp(self):
        self.engine = BinaryRuleEngine(arrival_time_s=4.0, moving_speed_kmh=2.0)

    def assert_ok(self, objects):
        self.assertEqual(self.engine.evaluate(objects), [])
        self.assertEqual(self.engine.classify(objects), OK)

    def assert_danger(self, objects, rule):
        events = self.engine.evaluate(objects)
        self.assertEqual(self.engine.classify(objects), DANGER)
        self.assertEqual({event.rule for event in events}, {rule})
        return events

    # --- empty / one-sided frames -------------------------------------

    def test_no_objects(self):
        self.assert_ok([])

    def test_no_pedestrians(self):
        self.assert_ok([vehicle(zone="crosswalk", speed_kmh=30.0)])

    def test_no_vehicles(self):
        self.assert_ok([pedestrian(1), pedestrian(2)])

    # --- R1: vehicle on crosswalk -------------------------------------

    def test_stopped_vehicle_on_crosswalk_is_ok(self):
        self.assert_ok([pedestrian(), vehicle(zone="crosswalk", speed_kmh=0.5)])

    def test_moving_vehicle_on_crosswalk_is_danger(self):
        events = self.assert_danger(
            [pedestrian(1), vehicle(100, zone="crosswalk", speed_kmh=10.0)],
            R1,
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].pedestrian_id, 1)
        self.assertEqual(events[0].vehicle_id, 100)
        self.assertEqual(events[0].vehicle_speed_kmh, 10.0)

    def test_speed_exactly_at_threshold_is_ok(self):
        # R1 uses a strict inequality: speed > moving_speed_kmh.
        self.assert_ok([pedestrian(), vehicle(zone="crosswalk", speed_kmh=2.0)])

    def test_speed_just_above_threshold_is_danger(self):
        self.assert_danger(
            [pedestrian(), vehicle(zone="crosswalk", speed_kmh=2.01)],
            R1,
        )

    def test_vehicle_on_crosswalk_ignores_tta(self):
        # TTA is meaningless once the vehicle is on the crosswalk.
        self.assert_ok(
            [pedestrian(), vehicle(zone="crosswalk", speed_kmh=1.0, tta_s=1.0)]
        )

    def test_moving_vehicle_on_crosswalk_with_waiting_pedestrian_is_ok(self):
        self.assert_ok(
            [
                pedestrian(zone="pedestrian_waiting_left"),
                vehicle(zone="crosswalk", speed_kmh=20.0),
            ]
        )

    def test_pedestrian_outside_zones_is_ok(self):
        self.assert_ok(
            [pedestrian(zone=None), vehicle(zone="crosswalk", speed_kmh=20.0)]
        )

    # --- R2: vehicle approaching --------------------------------------

    def test_approaching_vehicle_with_low_tta_is_danger(self):
        events = self.assert_danger(
            [pedestrian(), vehicle(zone="vehicle_approach", speed_kmh=15.0, tta_s=2.0)],
            R2,
        )
        self.assertEqual(events[0].tta_s, 2.0)

    def test_receding_vehicle_is_ok(self):
        # The pipeline sets TTA to None when the distance is not decreasing.
        self.assert_ok(
            [pedestrian(), vehicle(zone="vehicle_approach", speed_kmh=20.0, tta_s=None)]
        )

    def test_negative_tta_is_ok(self):
        self.assert_ok([pedestrian(), vehicle(zone="vehicle_approach", tta_s=-1.0)])

    def test_tta_below_threshold_is_danger(self):
        self.assert_danger([pedestrian(), vehicle(zone="vehicle_approach", tta_s=3.99)], R2)

    def test_tta_equal_to_threshold_is_danger(self):
        self.assert_danger([pedestrian(), vehicle(zone="vehicle_approach", tta_s=4.0)], R2)

    def test_tta_above_threshold_is_ok(self):
        self.assert_ok([pedestrian(), vehicle(zone="vehicle_approach", tta_s=4.01)])

    def test_tta_zero_is_danger(self):
        self.assert_danger([pedestrian(), vehicle(zone="vehicle_approach", tta_s=0.0)], R2)

    def test_approaching_vehicle_with_waiting_pedestrian_is_ok(self):
        # Proximity / arrival alone, without a pedestrian on the crosswalk.
        self.assert_ok(
            [
                pedestrian(zone="pedestrian_waiting_right"),
                vehicle(zone="vehicle_approach", tta_s=1.0),
            ]
        )

    def test_tta_outside_approach_zone_is_ignored(self):
        self.assert_ok([pedestrian(), vehicle(zone=None, speed_kmh=30.0, tta_s=1.0)])

    # --- missing / invalid numeric values -----------------------------

    def test_missing_speed_on_crosswalk_is_ok(self):
        self.assert_ok([pedestrian(), vehicle(zone="crosswalk", speed_kmh=None)])

    def test_missing_tta_in_approach_is_ok(self):
        self.assert_ok([pedestrian(), vehicle(zone="vehicle_approach", tta_s=None)])

    def test_non_finite_speed_is_ok(self):
        for value in (math.nan, math.inf, -math.inf, "abc"):
            with self.subTest(value=value):
                self.assert_ok([pedestrian(), vehicle(zone="crosswalk", speed_kmh=value)])

    def test_non_finite_tta_is_ok(self):
        for value in (math.nan, math.inf, -math.inf, "abc"):
            with self.subTest(value=value):
                self.assert_ok([pedestrian(), vehicle(zone="vehicle_approach", tta_s=value)])

    # --- categories and multiple pairs --------------------------------

    def test_bicycle_is_ignored(self):
        bicycle = SimpleNamespace(
            track_id=5,
            category="bicycle",
            zone="crosswalk",
            speed_kmh=15.0,
            ttc_crosswalk_s=1.0,
        )
        self.assert_ok([pedestrian(), bicycle])

    def test_multiple_pedestrians_and_vehicles(self):
        objects = [
            pedestrian(1, zone="crosswalk"),
            pedestrian(2, zone="crosswalk"),
            pedestrian(3, zone="pedestrian_waiting_left"),
            vehicle(100, zone="crosswalk", speed_kmh=12.0),        # R1
            vehicle(101, zone="vehicle_approach", tta_s=3.0),      # R2
            vehicle(102, zone="crosswalk", speed_kmh=0.0),         # stopped
            vehicle(103, zone="vehicle_approach", tta_s=None),     # receding
        ]

        events = self.engine.evaluate(objects)
        pairs = {(e.pedestrian_id, e.vehicle_id): e.rule for e in events}

        self.assertEqual(
            pairs,
            {
                (1, 100): R1,
                (2, 100): R1,
                (1, 101): R2,
                (2, 101): R2,
            },
        )
        self.assertEqual(self.engine.classify(objects), DANGER)

    def test_multiple_objects_without_conflict_is_ok(self):
        self.assert_ok(
            [
                pedestrian(1, zone="crosswalk"),
                pedestrian(2, zone=None),
                vehicle(100, zone="crosswalk", speed_kmh=1.0),
                vehicle(101, zone="vehicle_approach", tta_s=6.0),
                vehicle(102, zone=None, speed_kmh=40.0),
            ]
        )

    def test_frame_label(self):
        self.assertEqual(BinaryRuleEngine.frame_label([]), OK)
        events = self.engine.evaluate(
            [pedestrian(), vehicle(zone="crosswalk", speed_kmh=10.0)]
        )
        self.assertEqual(BinaryRuleEngine.frame_label(events), DANGER)

    # --- thresholds ---------------------------------------------------

    def test_thresholds_are_configurable(self):
        engine = BinaryRuleEngine(arrival_time_s=2.0, moving_speed_kmh=5.0)
        self.assertEqual(
            engine.classify([pedestrian(), vehicle(zone="vehicle_approach", tta_s=3.0)]),
            OK,
        )
        self.assertEqual(
            engine.classify([pedestrian(), vehicle(zone="crosswalk", speed_kmh=4.0)]),
            OK,
        )

    def test_zero_moving_speed_threshold_is_allowed(self):
        engine = BinaryRuleEngine(moving_speed_kmh=0.0)
        self.assertEqual(
            engine.classify([pedestrian(), vehicle(zone="crosswalk", speed_kmh=0.1)]),
            DANGER,
        )

    def test_invalid_parameters(self):
        invalid = [
            {"arrival_time_s": 0.0},
            {"arrival_time_s": -1.0},
            {"arrival_time_s": math.nan},
            {"arrival_time_s": math.inf},
            {"arrival_time_s": None},
            {"arrival_time_s": "abc"},
            {"arrival_time_s": True},
            {"moving_speed_kmh": -0.1},
            {"moving_speed_kmh": math.nan},
            {"moving_speed_kmh": math.inf},
            {"moving_speed_kmh": None},
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    BinaryRuleEngine(**kwargs)


if __name__ == "__main__":
    unittest.main()
