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

    def test_pedestrian_zone_history_not_shared_with_vehicle(self):
        estimator = make_estimator()
        estimator.update(pedestrian(5, "pedestrian_waiting_left"))
        estimator.update(vehicle(5, "crosswalk", speed_kmh=10.0))

        # Pedestrian history continues from its own previous zone.
        self.assertEqual(
            estimator.previous_zones[("pedestrian", 5)],
            "pedestrian_waiting_left",
        )
        self.assertNotIn(("vehicle", 5), estimator.previous_zones)

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
        # Known issue (S2): the first crosswalk frame yields an ENTERING
        # candidate that can never be confirmed; it resets the counter, so
        # CROSSING is confirmed only on the 4th crosswalk frame.
        self.assertEqual(states[5:8], [PedestrianState.WAITING] * 3)
        self.assertEqual(states[8:], [PedestrianState.CROSSING] * 2)
        self.assertNotIn(PedestrianState.ENTERING, states)

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


if __name__ == "__main__":
    unittest.main()
