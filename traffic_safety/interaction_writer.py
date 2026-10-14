import csv
import math
from pathlib import Path


class InteractionCsvWriter:
    """
    Export every visible pedestrian-vehicle pair for every processed frame.

    Pair selection is independent of SafetyRuleEngine:
    every pedestrian in the frame is paired with every vehicle in the frame.

    risk_events are used only to attach the rule-based baseline result
    (rule_risk_level / rule_trigger) to an already selected pair.
    """

    FIELDNAMES = [
        "video_id",
        "scene_id",
        "source_event_id",
        "camera_id",
        "frame_idx",
        "timestamp_s",
        "interaction_id",
        "pedestrian_track_id",
        "vehicle_track_id",
        "pedestrian_state",
        "vehicle_state",
        "pedestrian_zone",
        "vehicle_zone",
        "pedestrian_world_x_m",
        "pedestrian_world_y_m",
        "vehicle_world_x_m",
        "vehicle_world_y_m",
        "vehicle_speed_kmh",
        "vehicle_distance_to_crosswalk_m",
        "ttc_crosswalk_s",
        "pedestrian_confidence",
        "vehicle_confidence",
        "rule_risk_level",
        "rule_trigger",
        "feature_valid",
    ]

    FRAME_FIELDNAMES = [
        "video_id",
        "frame_idx",
        "timestamp_s",
        "num_pedestrians",
        "num_vehicles",
        "num_interactions",
        "pipeline_ok",
    ]

    RISK_RANK = {
        "ok": 0,
        "safe": 0,
        "warning": 1,
        "danger": 2,
    }

    def __init__(
        self,
        output_path: str,
        video_id: str,
        scene_id: str,
        source_event_id: str,
        camera_id: str,
        frame_metadata_path: str | None = None,
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.video_id = self._require_identifier("video_id", video_id)
        self.scene_id = self._require_identifier("scene_id", scene_id)
        self.source_event_id = self._require_identifier(
            "source_event_id",
            source_event_id,
        )
        self.camera_id = self._require_identifier("camera_id", camera_id)

        self.file = self.output_path.open(
            "w",
            newline="",
            encoding="utf-8",
        )
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=self.FIELDNAMES,
        )
        self.writer.writeheader()

        if frame_metadata_path is None:
            self.frame_metadata_path = self.output_path.with_name(
                f"{self.output_path.stem}_frames.csv"
            )
        else:
            self.frame_metadata_path = Path(frame_metadata_path)

        self.frame_metadata_path.parent.mkdir(parents=True, exist_ok=True)

        self.frame_file = self.frame_metadata_path.open(
            "w",
            newline="",
            encoding="utf-8",
        )
        self.frame_writer = csv.DictWriter(
            self.frame_file,
            fieldnames=self.FRAME_FIELDNAMES,
        )
        self.frame_writer.writeheader()

    @staticmethod
    def _require_identifier(name: str, value: str) -> str:
        value = str(value).strip()

        if not value:
            raise ValueError(
                f"{name} must be a non-empty stable identifier."
            )

        return value

    @staticmethod
    def make_interaction_id(
        pedestrian_track_id: int,
        vehicle_track_id: int,
    ) -> str:
        """
        Stable within one video while both automatic track IDs remain stable.
        Track fragmentation is deliberately not repaired automatically.
        """
        return (
            f"p{int(pedestrian_track_id):06d}"
            f"_v{int(vehicle_track_id):06d}"
        )

    @staticmethod
    def _optional_number(value, digits: int = 3):
        if value is None:
            return ""

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return ""

        if not math.isfinite(numeric):
            return ""

        return round(numeric, digits)

    @staticmethod
    def _feature_valid(pedestrian, vehicle) -> int:
        """
        Basic row-level extraction validity.

        Missing speed or TTC is not considered an extraction failure:
        both may be legitimately undefined for a valid observation.
        """
        required = [
            pedestrian.world_x,
            pedestrian.world_y,
            vehicle.world_x,
            vehicle.world_y,
        ]

        for value in required:
            if value is None:
                return 0

            try:
                if not math.isfinite(float(value)):
                    return 0
            except (TypeError, ValueError):
                return 0

        return 1

    def _rule_results_by_pair(
        self,
        risk_events,
    ) -> dict[tuple[int, int], tuple[str, str]]:
        results = {}

        for event in risk_events:
            pair = (
                int(event.pedestrian_id),
                int(event.vehicle_id),
            )

            level = str(event.level.value)
            rule = str(event.rule)

            previous = results.get(pair)

            if previous is None:
                results[pair] = (level, rule)
                continue

            previous_level, previous_rule = previous

            if (
                self.RISK_RANK.get(level, -1)
                > self.RISK_RANK.get(previous_level, -1)
            ):
                results[pair] = (level, rule)
            elif level == previous_level:
                rules = sorted(
                    set(
                        filter(
                            None,
                            previous_rule.split(";") + [rule],
                        )
                    )
                )
                results[pair] = (
                    previous_level,
                    ";".join(rules),
                )

        return results

    def write_frame(
        self,
        frame_idx: int,
        timestamp_s: float,
        objects,
        risk_events,
        pipeline_ok: bool = True,
    ) -> None:
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

        rule_results = self._rule_results_by_pair(risk_events)

        num_interactions = len(pedestrians) * len(vehicles)

        self.frame_writer.writerow(
            {
                "video_id": self.video_id,
                "frame_idx": frame_idx,
                "timestamp_s": round(timestamp_s, 4),
                "num_pedestrians": len(pedestrians),
                "num_vehicles": len(vehicles),
                "num_interactions": num_interactions,
                "pipeline_ok": int(bool(pipeline_ok)),
            }
        )

        for pedestrian in pedestrians:
            for vehicle in vehicles:
                pair = (
                    pedestrian.track_id,
                    vehicle.track_id,
                )

                rule_risk_level, rule_trigger = rule_results.get(
                    pair,
                    ("ok", ""),
                )

                self.writer.writerow(
                    {
                        "video_id": self.video_id,
                        "scene_id": self.scene_id,
                        "source_event_id": self.source_event_id,
                        "camera_id": self.camera_id,
                        "frame_idx": frame_idx,
                        "timestamp_s": round(timestamp_s, 4),
                        "interaction_id": self.make_interaction_id(
                            pedestrian.track_id,
                            vehicle.track_id,
                        ),
                        "pedestrian_track_id": pedestrian.track_id,
                        "vehicle_track_id": vehicle.track_id,
                        "pedestrian_state": (
                            pedestrian.state
                            if pedestrian.state is not None
                            else ""
                        ),
                        "vehicle_state": (
                            vehicle.state
                            if vehicle.state is not None
                            else ""
                        ),
                        "pedestrian_zone": (
                            pedestrian.zone
                            if pedestrian.zone is not None
                            else ""
                        ),
                        "vehicle_zone": (
                            vehicle.zone
                            if vehicle.zone is not None
                            else ""
                        ),
                        "pedestrian_world_x_m": self._optional_number(
                            pedestrian.world_x
                        ),
                        "pedestrian_world_y_m": self._optional_number(
                            pedestrian.world_y
                        ),
                        "vehicle_world_x_m": self._optional_number(
                            vehicle.world_x
                        ),
                        "vehicle_world_y_m": self._optional_number(
                            vehicle.world_y
                        ),
                        "vehicle_speed_kmh": self._optional_number(
                            vehicle.speed_kmh,
                            digits=2,
                        ),
                        "vehicle_distance_to_crosswalk_m": (
                            self._optional_number(
                                vehicle.distance_to_crosswalk_m
                            )
                        ),
                        "ttc_crosswalk_s": self._optional_number(
                            vehicle.ttc_crosswalk_s
                        ),
                        "pedestrian_confidence": self._optional_number(
                            pedestrian.confidence,
                            digits=4,
                        ),
                        "vehicle_confidence": self._optional_number(
                            vehicle.confidence,
                            digits=4,
                        ),
                        "rule_risk_level": rule_risk_level,
                        "rule_trigger": rule_trigger,
                        "feature_valid": self._feature_valid(
                            pedestrian,
                            vehicle,
                        ),
                    }
                )

    def close(self) -> None:
        self.file.close()
        self.frame_file.close()
