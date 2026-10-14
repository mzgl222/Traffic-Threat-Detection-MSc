import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from traffic_safety.interaction_writer import InteractionCsvWriter
from traffic_safety.safety_rules import RiskEvent, RiskLevel


def make_object(
    *,
    track_id: int,
    category: str,
    state: str,
    zone: str,
    world_x: float = 1.0,
    world_y: float = 2.0,
    confidence: float = 0.9,
    speed_kmh=None,
    distance_to_crosswalk_m=None,
    ttc_crosswalk_s=None,
):
    return SimpleNamespace(
        track_id=track_id,
        category=category,
        state=state,
        zone=zone,
        world_x=world_x,
        world_y=world_y,
        confidence=confidence,
        speed_kmh=speed_kmh,
        distance_to_crosswalk_m=distance_to_crosswalk_m,
        ttc_crosswalk_s=ttc_crosswalk_s,
    )


class InteractionCsvWriterTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_path = Path(self.temp_dir.name) / "interactions.csv"

        self.writer = InteractionCsvWriter(
            output_path=str(self.output_path),
            video_id="video_001",
            scene_id="scene_001",
            source_event_id="event_001",
            camera_id="cam_01",
        )

    def tearDown(self):
        if not self.writer.file.closed:
            self.writer.close()
        self.temp_dir.cleanup()

    def read_interactions(self):
        self.writer.file.flush()
        with self.output_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as file:
            return list(csv.DictReader(file))

    def read_frames(self):
        self.writer.frame_file.flush()
        with self.writer.frame_metadata_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as file:
            return list(csv.DictReader(file))

    def test_interactions_include_non_alert_pairs(self):
        pedestrian = make_object(
            track_id=10,
            category="pedestrian",
            state="waiting",
            zone="pedestrian_waiting_left",
        )
        vehicle = make_object(
            track_id=20,
            category="vehicle",
            state="approaching",
            zone="vehicle_approach",
            speed_kmh=25.0,
            distance_to_crosswalk_m=20.0,
            ttc_crosswalk_s=6.0,
        )

        self.writer.write_frame(
            frame_idx=100,
            timestamp_s=3.333,
            objects=[pedestrian, vehicle],
            risk_events=[],
        )

        rows = self.read_interactions()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["rule_risk_level"], "ok")
        self.assertEqual(rows[0]["rule_trigger"], "")

    def test_rule_result_does_not_select_pairs(self):
        pedestrian = make_object(
            track_id=10,
            category="pedestrian",
            state="crossing",
            zone="crosswalk",
        )
        vehicle = make_object(
            track_id=20,
            category="vehicle",
            state="approaching",
            zone="vehicle_approach",
            distance_to_crosswalk_m=8.0,
            ttc_crosswalk_s=2.0,
        )
        event = RiskEvent(
            pedestrian_id=10,
            vehicle_id=20,
            level=RiskLevel.DANGER,
            rule="pedestrian_crossing_low_ttc",
            ttc_s=2.0,
            distance_m=8.0,
        )

        self.writer.write_frame(
            frame_idx=1,
            timestamp_s=0.1,
            objects=[pedestrian, vehicle],
            risk_events=[event],
        )

        rows = self.read_interactions()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["rule_risk_level"], "danger")
        self.assertEqual(
            rows[0]["rule_trigger"],
            "pedestrian_crossing_low_ttc",
        )

    def test_missing_ttc_is_not_written_as_zero(self):
        pedestrian = make_object(
            track_id=1,
            category="pedestrian",
            state="waiting",
            zone="pedestrian_waiting_left",
        )
        vehicle = make_object(
            track_id=2,
            category="vehicle",
            state="approaching",
            zone="vehicle_approach",
            ttc_crosswalk_s=None,
        )

        self.writer.write_frame(
            frame_idx=0,
            timestamp_s=0.0,
            objects=[pedestrian, vehicle],
            risk_events=[],
        )

        row = self.read_interactions()[0]

        self.assertEqual(row["ttc_crosswalk_s"], "")
        self.assertEqual(row["feature_valid"], "1")

    def test_frame_without_pair_is_still_recorded(self):
        pedestrian = make_object(
            track_id=1,
            category="pedestrian",
            state="waiting",
            zone="pedestrian_waiting_left",
        )

        self.writer.write_frame(
            frame_idx=50,
            timestamp_s=1.5,
            objects=[pedestrian],
            risk_events=[],
        )

        self.assertEqual(len(self.read_interactions()), 0)
        frame_rows = self.read_frames()
        self.assertEqual(len(frame_rows), 1)
        self.assertEqual(frame_rows[0]["num_interactions"], "0")
        self.assertEqual(frame_rows[0]["pipeline_ok"], "1")

    def test_interaction_id_is_stable_for_same_track_pair(self):
        first = InteractionCsvWriter.make_interaction_id(4, 9)
        second = InteractionCsvWriter.make_interaction_id(4, 9)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
