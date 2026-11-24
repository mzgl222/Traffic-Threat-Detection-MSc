import math
import random
import unittest
from types import SimpleNamespace

import pandas as pd

from traffic_safety.motion_features import (
    FEATURE_COLUMNS,
    MotionFeatureTracker,
    compute_motion_features,
)

W = "pedestrian_waiting_left"
W2 = "pedestrian_waiting_right"
C = "crosswalk"


def ped(zone, track_id=1, category="pedestrian"):
    return SimpleNamespace(category=category, track_id=track_id, zone=zone)


def run(zones, fps=30.0, timestamps=None, track_id=1, category="pedestrian"):
    tracker = MotionFeatureTracker()
    if timestamps is None:
        timestamps = [i / fps for i in range(len(zones))]
    return [
        tracker.update(ped(zone, track_id, category), ts)
        for zone, ts in zip(zones, timestamps)
    ]


class NormalCrossingTest(unittest.TestCase):
    def test_entry_and_exit(self):
        zones = [W] * 5 + [C] * 10 + [W2] * 5
        out = run(zones, fps=10.0)

        # Initial waiting stay: beginning not observed -> unknown.
        self.assertIsNone(out[0].time_in_current_zone_s)
        self.assertIsNone(out[0].came_from_waiting)
        self.assertIsNone(out[4].time_since_crosswalk_entry_s)

        # Crosswalk observations at i=5,6 are pending; confirmed at i=7.
        self.assertEqual(out[6].stable_zone, W)
        self.assertEqual(out[7].stable_zone, C)
        # Stay dated back to the first crosswalk observation (t=0.5 s).
        self.assertAlmostEqual(out[7].time_since_crosswalk_entry_s, 0.2)
        self.assertAlmostEqual(out[14].time_since_crosswalk_entry_s, 0.9)
        self.assertAlmostEqual(out[14].time_in_current_zone_s, 0.9)
        self.assertTrue(out[7].came_from_waiting)
        self.assertFalse(out[7].came_from_crosswalk)

        # Exit to the other waiting zone, confirmed at i=17.
        self.assertEqual(out[17].stable_zone, W2)
        self.assertIsNone(out[17].time_since_crosswalk_entry_s)
        self.assertAlmostEqual(out[17].time_in_current_zone_s, 0.2)
        self.assertTrue(out[17].came_from_crosswalk)
        self.assertFalse(out[17].came_from_waiting)

    def test_pending_change_keeps_counting_old_zone(self):
        zones = [W] * 3 + [C] * 6 + [W2] * 2   # exit not yet confirmed
        out = run(zones, fps=10.0)
        # i=9,10 are pending W2 observations: still on the crosswalk.
        self.assertEqual(out[10].stable_zone, C)
        self.assertAlmostEqual(out[10].time_since_crosswalk_entry_s, 0.7)


class UnknownHistoryTest(unittest.TestCase):
    def test_pedestrian_appears_on_crosswalk(self):
        out = run([C] * 10)
        for features in out:
            self.assertEqual(features.stable_zone, C)
            self.assertIsNone(features.time_since_crosswalk_entry_s)
            self.assertIsNone(features.time_in_current_zone_s)
            self.assertIsNone(features.came_from_waiting)
            self.assertIsNone(features.came_from_crosswalk)

    def test_coming_from_outside_is_known_false(self):
        out = run([None] * 4 + [C] * 4, fps=10.0)
        self.assertEqual(out[-1].stable_zone, C)
        self.assertFalse(out[-1].came_from_waiting)
        self.assertFalse(out[-1].came_from_crosswalk)
        self.assertAlmostEqual(out[-1].time_since_crosswalk_entry_s, 0.3)

    def test_unknown_values_are_none_not_zero(self):
        features = run([W])[0]
        for column in FEATURE_COLUMNS:
            self.assertIsNone(getattr(features, column))


class FlickerAndGapsTest(unittest.TestCase):
    def test_short_flicker_is_ignored(self):
        zones = [W] * 3 + [C] * 5 + [W] * 2 + [C] * 5
        out = run(zones, fps=10.0)
        # Entry confirmed at i=5 (t=0.3 start); flicker at i=8,9 ignored.
        self.assertEqual(out[9].stable_zone, C)
        self.assertAlmostEqual(out[14].time_since_crosswalk_entry_s, 1.1)
        self.assertTrue(out[14].came_from_waiting)

    def test_flicker_to_outside_is_ignored(self):
        zones = [C] * 3 + [None] * 2 + [C] * 3
        out = run(zones)
        self.assertTrue(all(features.stable_zone == C for features in out))

    def test_gap_between_polygons_keeps_origin(self):
        # waiting -> outside (>= 3 observations) -> crosswalk
        zones = [W] * 3 + [None] * 4 + [C] * 4
        out = run(zones, fps=10.0)
        self.assertEqual(out[5].stable_zone, None)
        self.assertTrue(out[5].came_from_waiting)
        self.assertEqual(out[-1].stable_zone, C)
        self.assertTrue(out[-1].came_from_waiting)
        self.assertAlmostEqual(out[-1].time_since_crosswalk_entry_s, 0.3)

    def test_missed_detections_use_timestamps(self):
        # Track not observed between t=0.3 and t=1.5 (detection gap).
        zones = [W, W, W, C, C, C, C]
        timestamps = [0.0, 0.1, 0.2, 0.3, 1.5, 1.6, 2.0]
        out = run(zones, timestamps=timestamps)
        self.assertEqual(out[5].stable_zone, C)
        self.assertAlmostEqual(out[5].time_since_crosswalk_entry_s, 1.3)
        self.assertAlmostEqual(out[6].time_since_crosswalk_entry_s, 1.7)

    def test_alternating_zones_never_confirm(self):
        zones = [W] * 3 + [C, None] * 5
        out = run(zones)
        self.assertTrue(all(features.stable_zone == W for features in out))


class TimestampTest(unittest.TestCase):
    def test_same_seconds_at_different_fps(self):
        for fps in (10.0, 25.0, 30.0, 60.0):
            with self.subTest(fps=fps):
                n_wait, n_cross = int(fps), int(2 * fps)
                out = run([W] * n_wait + [C] * n_cross, fps=fps)
                # Last observation is (n_cross - 1) frames after entry.
                self.assertAlmostEqual(
                    out[-1].time_since_crosswalk_entry_s,
                    (n_cross - 1) / fps,
                )

    def test_irregular_timestamps(self):
        zones = [W, W, W, C, C, C, C]
        timestamps = [0.0, 0.04, 0.07, 0.11, 0.18, 0.2, 0.33]
        out = run(zones, timestamps=timestamps)
        self.assertAlmostEqual(out[5].time_since_crosswalk_entry_s, 0.2 - 0.11)
        self.assertAlmostEqual(out[6].time_since_crosswalk_entry_s, 0.33 - 0.11)

    def test_non_increasing_timestamp_raises(self):
        tracker = MotionFeatureTracker()
        tracker.update(ped(W), 1.0)
        with self.assertRaises(ValueError):
            tracker.update(ped(W), 1.0)
        with self.assertRaises(ValueError):
            tracker.update(ped(W), 0.5)

    def test_tracks_are_independent_in_time(self):
        tracker = MotionFeatureTracker()
        tracker.update(ped(W, track_id=1), 1.0)
        # Another track may have the same or earlier timestamp.
        tracker.update(ped(W, track_id=2), 1.0)
        tracker.update(ped(W, track_id=3), 0.5)


class CategoryTest(unittest.TestCase):
    def test_category_change_starts_new_history(self):
        tracker = MotionFeatureTracker()
        for i, zone in enumerate([W] * 3 + [C] * 5):
            tracker.update(ped(zone, track_id=7), i / 10)
        crossing = tracker.update(ped(C, track_id=7), 0.8)
        self.assertIsNotNone(crossing.time_since_crosswalk_entry_s)

        vehicle = tracker.update(ped(C, track_id=7, category="vehicle"), 0.9)
        self.assertIsNone(vehicle.time_since_crosswalk_entry_s)
        self.assertIsNone(vehicle.came_from_waiting)

        # The pedestrian history with the same id is untouched.
        back = tracker.update(ped(C, track_id=7), 1.0)
        self.assertAlmostEqual(back.time_since_crosswalk_entry_s, 0.7)

    def test_different_track_ids_are_not_merged(self):
        tracker = MotionFeatureTracker()
        for i in range(5):
            tracker.update(ped(W, track_id=1), i / 10)
        new = tracker.update(ped(C, track_id=2), 0.5)
        self.assertIsNone(new.came_from_waiting)
        self.assertIsNone(new.time_since_crosswalk_entry_s)


class MissingDataTest(unittest.TestCase):
    def test_missing_zone_variants_mean_outside(self):
        for value in (None, float("nan"), pd.NA, "", "  "):
            with self.subTest(value=repr(value)):
                self.assertIsNone(run([value])[0].stable_zone)

    def test_invalid_inputs_raise(self):
        tracker = MotionFeatureTracker()
        for timestamp in (None, float("nan"), math.inf, "x"):
            with self.subTest(timestamp=timestamp):
                with self.assertRaises(ValueError):
                    tracker.update(ped(W), timestamp)
        with self.assertRaises(ValueError):
            tracker.update(SimpleNamespace(category=None, track_id=1, zone=W), 0.0)
        with self.assertRaises(ValueError):
            tracker.update(SimpleNamespace(category="pedestrian", track_id=None, zone=W), 0.0)

    def test_invalid_confirm_observations(self):
        for value in (0, -1, 1.5, True, "3"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    MotionFeatureTracker(confirm_observations=value)


def random_tracks(seed: int, n_frames: int = 250, n_tracks: int = 4) -> pd.DataFrame:
    rng = random.Random(seed)
    zone_choices = [W, W2, C, None, "vehicle_approach"]
    rows = []
    current = {track: rng.choice(zone_choices) for track in range(n_tracks)}
    for frame in range(n_frames):
        timestamp = frame / 30 + rng.uniform(0, 0.005)
        for track in range(n_tracks):
            if rng.random() < 0.15:          # missed detection
                continue
            if rng.random() < 0.05:          # real zone change
                current[track] = rng.choice(zone_choices)
            zone = current[track]
            if rng.random() < 0.1:           # flicker
                zone = rng.choice(zone_choices)
            category = "vehicle" if (track == 3 and frame > 120) else "pedestrian"
            rows.append(
                {
                    "frame_idx": frame,
                    "timestamp_s": timestamp,
                    "track_id": track,
                    "category": category,
                    "zone": zone,
                }
            )
    return pd.DataFrame(rows)


class PrefixTest(unittest.TestCase):
    """No leakage from the future: features at frame t must not change
    when all frames after t are removed."""

    def test_prefix_invariance(self):
        for seed in range(5):
            tracks = random_tracks(seed)
            full = compute_motion_features(tracks)
            for cut in (1, 37, 90, 121, 180, 249):
                with self.subTest(seed=seed, cut=cut):
                    prefix = tracks[tracks.frame_idx <= cut]
                    partial = compute_motion_features(prefix)
                    pd.testing.assert_frame_equal(partial, full.loc[prefix.index])

    def test_random_data_produces_all_feature_kinds(self):
        full = compute_motion_features(random_tracks(0))
        self.assertTrue(full.time_since_crosswalk_entry_s.notna().any())
        self.assertTrue(full.time_since_crosswalk_entry_s.isna().any())
        self.assertTrue(full.came_from_waiting.fillna(False).any())
        self.assertTrue((full.time_in_current_zone_s.dropna() >= 0).all())


class DataFrameHelperTest(unittest.TestCase):
    def test_output_alignment_and_dtypes(self):
        tracks = pd.DataFrame(
            {
                "frame_idx": [0, 0, 1, 1, 2, 2, 3],
                "timestamp_s": [0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 0.3],
                "track_id": [1, 2, 1, 2, 1, 2, 1],
                "category": ["pedestrian"] * 7,
                "zone": [W, C, C, C, C, C, C],
            },
            index=[10, 11, 12, 13, 14, 15, 16],
        )
        shuffled = tracks.sample(frac=1, random_state=1)
        out = compute_motion_features(shuffled)

        self.assertEqual(list(out.index), list(shuffled.index))
        self.assertEqual(out["time_since_crosswalk_entry_s"].dtype, "float64")
        self.assertEqual(str(out["came_from_waiting"].dtype), "boolean")
        # Track 1 enters at frame 1 (t=0.1), confirmed at frame 3.
        self.assertAlmostEqual(out.loc[16, "time_since_crosswalk_entry_s"], 0.2)
        self.assertTrue(out.loc[16, "came_from_waiting"])
        # Track 2 starts on the crosswalk: unknown.
        self.assertTrue(math.isnan(out.loc[15, "time_since_crosswalk_entry_s"]))
        self.assertTrue(pd.isna(out.loc[15, "came_from_waiting"]))
        # Input not modified.
        self.assertNotIn("stable_zone", shuffled.columns)

    def test_missing_columns_raise(self):
        with self.assertRaises(ValueError):
            compute_motion_features(pd.DataFrame({"frame_idx": [0]}))

    def test_duplicate_observation_raises(self):
        tracks = pd.DataFrame(
            {
                "frame_idx": [0, 0],
                "timestamp_s": [0.0, 0.0],
                "track_id": [1, 1],
                "category": ["pedestrian", "pedestrian"],
                "zone": [W, W],
            }
        )
        with self.assertRaises(ValueError):
            compute_motion_features(tracks)


if __name__ == "__main__":
    unittest.main()
