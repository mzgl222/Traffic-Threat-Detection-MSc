import math
import random
import unittest

import pandas as pd

from traffic_safety.motion_aggregation import (
    MIN_ENTRY_COLUMN,
    UNKNOWN_ENTRY_COLUMN,
    aggregate_crosswalk_entry,
)
from traffic_safety.motion_features import compute_motion_features


def obs(frame_idx, category, stable_zone, entry):
    return {
        "frame_idx": frame_idx,
        "category": category,
        "motion_stable_zone": stable_zone,
        "time_since_crosswalk_entry_s": entry,
    }


def run(rows, frames):
    table = pd.DataFrame(rows, columns=["frame_idx", "category", "motion_stable_zone", "time_since_crosswalk_entry_s"])
    return aggregate_crosswalk_entry(table, frames).set_index("frame_idx")


class AggregationCasesTest(unittest.TestCase):
    def test_no_pedestrian_on_crosswalk(self):
        result = run([obs(0, "pedestrian", "pedestrian_waiting_left", None)], [0])
        self.assertTrue(math.isnan(result.loc[0, MIN_ENTRY_COLUMN]))
        self.assertEqual(result.loc[0, UNKNOWN_ENTRY_COLUMN], 0)

    def test_frame_without_observations(self):
        result = run([obs(0, "pedestrian", "crosswalk", 1.0)], [0, 1])
        self.assertTrue(math.isnan(result.loc[1, MIN_ENTRY_COLUMN]))
        self.assertEqual(result.loc[1, UNKNOWN_ENTRY_COLUMN], 0)

    def test_only_unknown_entries(self):
        rows = [obs(0, "pedestrian", "crosswalk", None), obs(0, "pedestrian", "crosswalk", float("nan"))]
        result = run(rows, [0])
        self.assertTrue(math.isnan(result.loc[0, MIN_ENTRY_COLUMN]))
        self.assertEqual(result.loc[0, UNKNOWN_ENTRY_COLUMN], 2)

    def test_minimum_of_known_and_count_of_unknown(self):
        rows = [
            obs(5, "pedestrian", "crosswalk", 4.0),
            obs(5, "pedestrian", "crosswalk", 1.5),
            obs(5, "pedestrian", "crosswalk", None),
        ]
        result = run(rows, [5])
        self.assertEqual(result.loc[5, MIN_ENTRY_COLUMN], 1.5)
        self.assertEqual(result.loc[5, UNKNOWN_ENTRY_COLUMN], 1)

    def test_zero_entry_time_is_kept(self):
        result = run([obs(0, "pedestrian", "crosswalk", 0.0)], [0])
        self.assertEqual(result.loc[0, MIN_ENTRY_COLUMN], 0.0)
        self.assertEqual(result.loc[0, UNKNOWN_ENTRY_COLUMN], 0)

    def test_vehicles_and_bicycles_are_ignored(self):
        rows = [
            obs(0, "vehicle", "crosswalk", 0.3),
            obs(0, "vehicle", "crosswalk", None),
            obs(0, "bicycle", "crosswalk", None),
            obs(0, "pedestrian", "crosswalk", 2.0),
        ]
        result = run(rows, [0])
        self.assertEqual(result.loc[0, MIN_ENTRY_COLUMN], 2.0)
        self.assertEqual(result.loc[0, UNKNOWN_ENTRY_COLUMN], 0)

    def test_raw_zone_is_not_used_only_stable_zone(self):
        # Pedestrian whose stable zone is still waiting (crosswalk not confirmed).
        result = run([obs(0, "pedestrian", "pedestrian_waiting_left", None)], [0])
        self.assertEqual(result.loc[0, UNKNOWN_ENTRY_COLUMN], 0)

    def test_missing_stable_zone_is_not_crosswalk(self):
        result = run([obs(0, "pedestrian", None, None)], [0])
        self.assertEqual(result.loc[0, UNKNOWN_ENTRY_COLUMN], 0)

    def test_output_dtypes_and_order(self):
        result = aggregate_crosswalk_entry(
            pd.DataFrame([obs(2, "pedestrian", "crosswalk", 1.0)]), [3, 2, 1]
        )
        self.assertEqual(list(result["frame_idx"]), [3, 2, 1])
        self.assertEqual(str(result[MIN_ENTRY_COLUMN].dtype), "float64")
        self.assertEqual(str(result[UNKNOWN_ENTRY_COLUMN].dtype), "int64")


class AggregationValidationTest(unittest.TestCase):
    def test_missing_column(self):
        with self.assertRaises(ValueError):
            aggregate_crosswalk_entry(pd.DataFrame({"frame_idx": [0]}), [0])

    def test_observation_frame_not_listed(self):
        with self.assertRaises(ValueError):
            run([obs(7, "pedestrian", "crosswalk", 1.0)], [0])

    def test_duplicate_frames(self):
        with self.assertRaises(ValueError):
            run([], [0, 0])

    def test_known_entry_outside_crosswalk_is_rejected(self):
        with self.assertRaises(ValueError):
            run([obs(0, "pedestrian", "pedestrian_waiting_left", 1.0)], [0])


class AggregationCausalityTest(unittest.TestCase):
    """Aggregates for frames <= k must not depend on frames > k."""

    def test_prefix_invariance_end_to_end(self):
        zones = [None, "crosswalk", "pedestrian_waiting_left", "vehicle_approach"]
        for seed in range(15):
            rng = random.Random(seed)
            rows = []
            for frame in range(60):
                for track_id in range(4):
                    if rng.random() < 0.8:
                        rows.append({
                            "frame_idx": frame,
                            "timestamp_s": frame / 30,
                            "track_id": track_id,
                            "category": "pedestrian" if track_id < 3 else "vehicle",
                            "zone": rng.choice(zones) if rng.random() < 0.3 else ("crosswalk" if track_id % 2 else None),
                        })
            tracks = pd.DataFrame(rows)

            def aggregate(part, frames):
                features = compute_motion_features(part)
                table = part.assign(
                    motion_stable_zone=features["stable_zone"],
                    time_since_crosswalk_entry_s=features["time_since_crosswalk_entry_s"],
                )
                return aggregate_crosswalk_entry(table, frames).set_index("frame_idx")

            full = aggregate(tracks, range(60))
            for cut in (10, 30, 45):
                prefix = aggregate(tracks[tracks.frame_idx <= cut], range(cut + 1))
                pd.testing.assert_frame_equal(prefix, full.loc[: cut], check_names=False)


if __name__ == "__main__":
    unittest.main()
