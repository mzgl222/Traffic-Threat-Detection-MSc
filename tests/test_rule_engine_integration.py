import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from traffic_safety.binary_rules import DANGER, OK, BinaryRuleEngine
from traffic_safety.binary_timeline_writer import BinaryTimelineCsvWriter
from traffic_safety.interaction_writer import InteractionCsvWriter
from traffic_safety.output_paths import (
    binary_timeline_path,
    ensure_no_existing_outputs,
    output_files,
    redirect_outputs,
)
from traffic_safety.rule_engines import (
    binary_events_as_rule_results,
    create_rule_engine,
    resolve_engine_name,
)
from traffic_safety.safety_rules import SafetyRuleEngine


def make_config():
    return {
        "video": {"output": "outputs/scene/video_output.mp4"},
        "data_output": {
            "csv": "outputs/scene/csv/scenario_tracks.csv",
            "risk_csv": "outputs/scene/csv/scenario_risk_events.csv",
            "risk_timeline_csv": "outputs/scene/csv/scenario_risk_timeline.csv",
            "interactions_csv": "outputs/scene/csv/scenario_interactions.csv",
            "interaction_frames_csv": "outputs/scene/csv/scenario_interaction_frames.csv",
            "plots": "outputs/scene/plots",
        },
        "risk": {"warning_ttc_s": 4.0, "danger_ttc_s": 2.5, "fallback_distance_m": 10.0},
    }


def obj(track_id, category, zone, speed_kmh=None, tta_s=None):
    return SimpleNamespace(
        track_id=track_id,
        category=category,
        zone=zone,
        speed_kmh=speed_kmh,
        ttc_crosswalk_s=tta_s,
        world_x=1.0,
        world_y=2.0,
        confidence=0.9,
        state="",
        distance_to_crosswalk_m=None,
    )


class EngineSelectionTest(unittest.TestCase):
    def test_default_is_legacy(self):
        self.assertEqual(resolve_engine_name(None, {}), "legacy")

    def test_config_value_is_used(self):
        self.assertEqual(resolve_engine_name(None, {"rules": {"engine": "binary"}}), "binary")

    def test_cli_overrides_config(self):
        config = {"rules": {"engine": "binary"}}
        self.assertEqual(resolve_engine_name("legacy", config), "legacy")

    def test_name_is_normalised(self):
        self.assertEqual(resolve_engine_name(" Binary ", {}), "binary")

    def test_unknown_name_raises(self):
        with self.assertRaises(ValueError):
            resolve_engine_name("fuzzy", {})
        with self.assertRaises(ValueError):
            resolve_engine_name(None, {"rules": {"engine": "fuzzy"}})

    def test_create_legacy_uses_risk_section(self):
        engine = create_rule_engine("legacy", make_config())
        self.assertIsInstance(engine, SafetyRuleEngine)
        self.assertEqual(engine.warning_ttc_s, 4.0)
        self.assertEqual(engine.danger_ttc_s, 2.5)
        self.assertEqual(engine.fallback_distance_m, 10.0)

    def test_create_binary_defaults(self):
        engine = create_rule_engine("binary", make_config())
        self.assertIsInstance(engine, BinaryRuleEngine)
        self.assertEqual(engine.arrival_time_s, 4.0)
        self.assertEqual(engine.moving_speed_kmh, 2.0)

    def test_create_binary_from_config(self):
        config = make_config()
        config["binary_rules"] = {"arrival_time_s": 3.0, "moving_speed_kmh": 1.5}
        engine = create_rule_engine("binary", config)
        self.assertEqual(engine.arrival_time_s, 3.0)
        self.assertEqual(engine.moving_speed_kmh, 1.5)

    def test_create_binary_rejects_unknown_keys(self):
        config = make_config()
        config["binary_rules"] = {"arrival_time": 3.0}
        with self.assertRaises(ValueError):
            create_rule_engine("binary", config)

    def test_binary_does_not_need_risk_section(self):
        config = make_config()
        del config["risk"]
        self.assertIsInstance(create_rule_engine("binary", config), BinaryRuleEngine)


class BinaryTimelineWriterTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "nested" / "binary.csv"
        self.engine = BinaryRuleEngine()

    def tearDown(self):
        self.temp_dir.cleanup()

    def read_rows(self):
        with self.path.open(encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))

    def test_rows_and_flags(self):
        frames = [
            [],
            [obj(1, "pedestrian", "crosswalk"), obj(10, "vehicle", "crosswalk", speed_kmh=8.0)],
            [
                obj(1, "pedestrian", "crosswalk"),
                obj(2, "pedestrian", "crosswalk"),
                obj(11, "vehicle", "vehicle_approach", tta_s=2.0),
            ],
        ]
        writer = BinaryTimelineCsvWriter(str(self.path))
        for idx, objects in enumerate(frames):
            events = self.engine.evaluate(objects)
            writer.write_frame(idx, idx / 30, self.engine.frame_label(events), events)
        writer.close()

        rows = self.read_rows()
        self.assertEqual(list(rows[0].keys()), BinaryTimelineCsvWriter.FIELDNAMES)
        self.assertEqual(len(rows), 3)

        self.assertEqual(rows[0]["risk_binary"], "0")
        self.assertEqual(rows[0]["risk_label"], "ok")
        self.assertEqual(rows[0]["rules"], "")

        self.assertEqual(rows[1]["risk_binary"], "1")
        self.assertEqual(rows[1]["risk_label"], "danger")
        self.assertEqual((rows[1]["r1"], rows[1]["r2"]), ("1", "0"))
        self.assertEqual(rows[1]["num_r1_pairs"], "1")
        self.assertEqual(rows[1]["rules"], "vehicle_on_crosswalk_with_pedestrian")

        self.assertEqual((rows[2]["r1"], rows[2]["r2"]), ("0", "1"))
        self.assertEqual(rows[2]["num_r2_pairs"], "2")
        self.assertEqual(rows[2]["timestamp_s"], str(round(2 / 30, 4)))

    def test_invalid_label_raises(self):
        writer = BinaryTimelineCsvWriter(str(self.path))
        try:
            with self.assertRaises(ValueError):
                writer.write_frame(0, 0.0, 2, [])
        finally:
            writer.close()


class InteractionAdapterTest(unittest.TestCase):
    def test_binary_events_fill_interaction_rule_columns(self):
        engine = BinaryRuleEngine()
        objects = [
            obj(1, "pedestrian", "crosswalk"),
            obj(2, "pedestrian", "pedestrian_waiting_left"),
            obj(10, "vehicle", "crosswalk", speed_kmh=8.0),
        ]
        events = engine.evaluate(objects)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "interactions.csv"
            writer = InteractionCsvWriter(
                output_path=str(path),
                video_id="v",
                scene_id="s",
                source_event_id="e",
                camera_id="c",
            )
            writer.write_frame(0, 0.0, objects, binary_events_as_rule_results(events))
            writer.close()
            with path.open(encoding="utf-8", newline="") as file:
                rows = {row["pedestrian_track_id"]: row for row in csv.DictReader(file)}

        self.assertEqual(rows["1"]["rule_risk_level"], "danger")
        self.assertEqual(rows["1"]["rule_trigger"], "vehicle_on_crosswalk_with_pedestrian")
        self.assertEqual(rows["2"]["rule_risk_level"], "ok")
        self.assertEqual(rows["2"]["rule_trigger"], "")


class OutputPathsTest(unittest.TestCase):
    def test_default_binary_timeline_path(self):
        self.assertEqual(
            binary_timeline_path(make_config()),
            str(Path("outputs/scene/csv/scenario_binary_timeline.csv")),
        )

    def test_explicit_binary_timeline_path(self):
        config = make_config()
        config["data_output"]["binary_timeline_csv"] = "x/y.csv"
        self.assertEqual(binary_timeline_path(config), "x/y.csv")

    def test_redirect_keeps_structure_and_does_not_mutate(self):
        config = make_config()
        redirected = redirect_outputs(config, "runs/binary_test")

        self.assertEqual(redirected["video"]["output"], str(Path("runs/binary_test/video_output.mp4")))
        self.assertEqual(
            redirected["data_output"]["csv"],
            str(Path("runs/binary_test/csv/scenario_tracks.csv")),
        )
        self.assertEqual(
            redirected["data_output"]["binary_timeline_csv"],
            str(Path("runs/binary_test/csv/scenario_binary_timeline.csv")),
        )
        self.assertEqual(redirected["data_output"]["plots"], str(Path("runs/binary_test/plots")))
        # Original untouched.
        self.assertEqual(config["video"]["output"], "outputs/scene/video_output.mp4")
        self.assertNotIn("binary_timeline_csv", config["data_output"])

    def test_redirect_path_outside_scene_keeps_name(self):
        config = make_config()
        config["data_output"]["frame_labels_csv"] = "elsewhere/labels.csv"
        redirected = redirect_outputs(config, "runs/x")
        self.assertEqual(redirected["data_output"]["frame_labels_csv"], str(Path("runs/x/labels.csv")))

    def test_output_files_depend_on_engine(self):
        config = make_config()
        legacy = {p.name for p in output_files(config, "legacy")}
        binary = {p.name for p in output_files(config, "binary")}
        self.assertIn("scenario_risk_timeline.csv", legacy)
        self.assertNotIn("scenario_binary_timeline.csv", legacy)
        self.assertIn("scenario_binary_timeline.csv", binary)
        self.assertNotIn("scenario_risk_timeline.csv", binary)
        self.assertIn("scenario_tracks.csv", binary)

    def test_guard_refuses_existing_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            existing = Path(temp_dir) / "a.csv"
            existing.write_text("x", encoding="utf-8")
            missing = Path(temp_dir) / "b.csv"
            ensure_no_existing_outputs([missing])  # no error
            with self.assertRaises(FileExistsError):
                ensure_no_existing_outputs([missing, existing])
            self.assertEqual(existing.read_text(encoding="utf-8"), "x")


if __name__ == "__main__":
    unittest.main()
