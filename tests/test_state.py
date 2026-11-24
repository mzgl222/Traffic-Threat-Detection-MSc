import unittest
from types import SimpleNamespace

from traffic_safety.state import (
    PedestrianState,
    StateEstimator,
    VehicleState,
)


def pedestrian(track_id, zone):
    return SimpleNamespace(track_id=track_id, category="pedestrian", zone=zone, speed_kmh=None)


def vehicle(track_id, zone, speed_kmh=None):
    return SimpleNamespace(track_id=track_id, category="vehicle", zone=zone, speed_kmh=speed_kmh)


def make_estimator():
    # Same parameters as run.py.
    return StateEstimator(
        stopped_speed_threshold=2.0,
        slowing_threshold=-2.0,
        state_confirm_frames=3,
        stopped_confirm_frames=5,
        speed_history_length=6,
    )


PEDESTRIAN_STATES = set(PedestrianState)
VEHICLE_STATES = set(VehicleState)


class CategorySeparationTest(unittest.TestCase):
    def test_vehicle_does_not_inherit_pedestrian_state(self):
        estimator = make_estimator()
        for _ in range(10):
            estimator.update(pedestrian(7, "crosswalk"))

        # The tracker keeps id 7, but the detector now says "vehicle".
        states = [estimator.update(vehicle(7, "crosswalk", speed_kmh=10.0)) for _ in range(5)]

        self.assertEqual(states[0], VehicleState.PASSING)
        self.assertTrue(all(state in VEHICLE_STATES for state in states))

    def test_pedestrian_does_not_inherit_vehicle_state(self):
        estimator = make_estimator()
        for _ in range(10):
            estimator.update(vehicle(3, "crosswalk", speed_kmh=10.0))

        states = [estimator.update(pedestrian(3, "pedestrian_waiting_left")) for _ in range(5)]

        self.assertEqual(states[0], PedestrianState.WAITING)
        self.assertTrue(all(state in PEDESTRIAN_STATES for state in states))

    def test_state_history_is_keyed_by_category(self):
        estimator = make_estimator()
        estimator.update(pedestrian(5, "pedestrian_waiting_left"))
        estimator.update(vehicle(5, "crosswalk", speed_kmh=10.0))

        self.assertEqual(
            estimator.previous_states[("pedestrian", 5)],
            PedestrianState.WAITING,
        )
        self.assertEqual(
            estimator.previous_states[("vehicle", 5)],
            VehicleState.PASSING,
        )

    def test_same_id_different_categories_in_one_frame(self):
        estimator = make_estimator()
        for _ in range(6):
            ped_state = estimator.update(pedestrian(9, "pedestrian_waiting_right"))
            veh_state = estimator.update(vehicle(9, "vehicle_approach", speed_kmh=0.5))

        self.assertEqual(ped_state, PedestrianState.WAITING)
        self.assertEqual(veh_state, VehicleState.STOPPED)

    def test_switch_back_restores_own_history(self):
        estimator = make_estimator()
        for _ in range(5):
            estimator.update(pedestrian(4, "crosswalk"))
        for _ in range(5):
            estimator.update(vehicle(4, None, speed_kmh=20.0))

        # Back to pedestrian: its own stabilised state is still CROSSING.
        self.assertEqual(
            estimator.update(pedestrian(4, "crosswalk")),
            PedestrianState.CROSSING,
        )

    def test_vehicle_speed_history_not_mixed(self):
        estimator = make_estimator()
        for speed in (30.0, 30.0, 30.0):
            estimator.update(vehicle(2, "vehicle_approach", speed_kmh=speed))
        # A pedestrian with the same id must not touch vehicle speed history.
        estimator.update(pedestrian(2, "crosswalk"))
        self.assertEqual(list(estimator.speed_history[("vehicle", 2)]), [30.0, 30.0, 30.0])
        self.assertNotIn(("pedestrian", 2), estimator.speed_history)

    def test_other_categories_return_none_and_store_nothing(self):
        estimator = make_estimator()
        bike = SimpleNamespace(track_id=1, category="bicycle", zone="crosswalk", speed_kmh=10.0)
        self.assertIsNone(estimator.update(bike))
        self.assertEqual(estimator.previous_states, {})


class TransitionsUnchangedTest(unittest.TestCase):
    """S1 must not change the transition logic for a single-category track."""

    def test_pedestrian_waiting_to_crossing(self):
        estimator = make_estimator()
        zones = ["pedestrian_waiting_left"] * 5 + ["crosswalk"] * 5
        states = [estimator.update(pedestrian(1, zone)) for zone in zones]

        self.assertEqual(states[:5], [PedestrianState.WAITING] * 5)
        # After B1 CROSSING is confirmed on the 3rd crosswalk frame
        # (before B1 an unconfirmable ENTERING candidate delayed it by 1 frame).
        self.assertEqual(states[5:7], [PedestrianState.WAITING] * 2)
        self.assertEqual(states[7:], [PedestrianState.CROSSING] * 3)

    def test_short_flicker_does_not_change_state(self):
        estimator = make_estimator()
        zones = ["crosswalk"] * 5 + ["pedestrian_waiting_left"] * 2 + ["crosswalk"] * 3
        states = [estimator.update(pedestrian(1, zone)) for zone in zones]
        self.assertTrue(all(state == PedestrianState.CROSSING for state in states))

    def test_vehicle_stopped_needs_five_frames(self):
        estimator = make_estimator()
        states = [estimator.update(vehicle(1, "vehicle_approach", speed_kmh=20.0))]
        states += [estimator.update(vehicle(1, "vehicle_approach", speed_kmh=0.5)) for _ in range(5)]

        self.assertEqual(states[0], VehicleState.APPROACHING)
        self.assertEqual(states[1:5], [VehicleState.APPROACHING] * 4)
        self.assertEqual(states[5], VehicleState.STOPPED)

    def test_vehicle_passing_on_crosswalk(self):
        estimator = make_estimator()
        zones = ["vehicle_approach"] * 3 + ["crosswalk"] * 4
        states = [estimator.update(vehicle(1, zone, speed_kmh=15.0)) for zone in zones]
        self.assertEqual(states[-1], VehicleState.PASSING)


class RemoveTrackTest(unittest.TestCase):
    def test_remove_all_categories(self):
        estimator = make_estimator()
        estimator.update(pedestrian(8, "crosswalk"))
        estimator.update(vehicle(8, "crosswalk", speed_kmh=5.0))
        estimator.update(vehicle(9, "crosswalk", speed_kmh=5.0))

        estimator.remove_track(8)

        self.assertNotIn(("pedestrian", 8), estimator.previous_states)
        self.assertNotIn(("vehicle", 8), estimator.previous_states)
        self.assertIn(("vehicle", 9), estimator.previous_states)

    def test_remove_single_category(self):
        estimator = make_estimator()
        estimator.update(pedestrian(8, "crosswalk"))
        estimator.update(vehicle(8, "crosswalk", speed_kmh=5.0))

        estimator.remove_track(8, category="vehicle")

        self.assertIn(("pedestrian", 8), estimator.previous_states)
        self.assertNotIn(("vehicle", 8), estimator.previous_states)


class PedestrianSimplifiedStatesTest(unittest.TestCase):
    """B1: pedestrian states are OUTSIDE / WAITING / CROSSING only."""

    def run_zones(self, zones, track_id=1):
        estimator = make_estimator()
        return [estimator.update(pedestrian(track_id, zone)) for zone in zones]

    def test_enum_has_three_states(self):
        self.assertEqual(
            {state.value for state in PedestrianState},
            {"outside", "waiting", "crossing"},
        )

    def test_full_cycle_waiting_crossing_waiting(self):
        zones = (
            ["pedestrian_waiting_left"] * 4
            + ["crosswalk"] * 6
            + ["pedestrian_waiting_right"] * 5
        )
        states = self.run_zones(zones)
        W, C = PedestrianState.WAITING, PedestrianState.CROSSING
        self.assertEqual(states, [W] * 4 + [W, W] + [C] * 4 + [C, C] + [W] * 3)

    def test_first_observation_is_accepted_immediately(self):
        self.assertEqual(self.run_zones(["crosswalk"])[0], PedestrianState.CROSSING)
        self.assertEqual(self.run_zones([None])[0], PedestrianState.OUTSIDE)
        self.assertEqual(
            self.run_zones(["pedestrian_waiting_right"])[0],
            PedestrianState.WAITING,
        )

    def test_unknown_zone_is_outside(self):
        # e.g. a pedestrian walking on the road inside vehicle_approach
        states = self.run_zones(["vehicle_approach"] * 3)
        self.assertEqual(states, [PedestrianState.OUTSIDE] * 3)

    def test_two_frame_flicker_is_ignored(self):
        zones = ["crosswalk"] * 4 + [None] * 2 + ["crosswalk"] * 2
        states = self.run_zones(zones)
        self.assertTrue(all(state == PedestrianState.CROSSING for state in states))

    def test_three_frame_change_is_confirmed(self):
        zones = ["crosswalk"] * 4 + [None] * 3
        states = self.run_zones(zones)
        self.assertEqual(states[-1], PedestrianState.OUTSIDE)
        self.assertEqual(states[-2], PedestrianState.CROSSING)

    def test_alternating_candidates_reset_confirmation(self):
        # waiting -> crosswalk, None, crosswalk, None ... never 3 in a row
        zones = ["pedestrian_waiting_left"] * 3 + ["crosswalk", None] * 4
        states = self.run_zones(zones)
        self.assertTrue(all(state == PedestrianState.WAITING for state in states))

    def test_states_are_plain_strings_for_legacy_consumers(self):
        # SafetyRuleEngine and build_frame_dataset compare states as strings.
        state = self.run_zones(["crosswalk"])[0]
        self.assertEqual(state, "crossing")
        self.assertIn(state, {"entering", "crossing"})


if __name__ == "__main__":
    unittest.main()
