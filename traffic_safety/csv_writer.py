import csv
from pathlib import Path


class CsvResultWriter:
    FIELDNAMES = [
        "frame_idx",
        "timestamp_s",
        "track_id",
        "class_id",
        "class_name",
        "category",
        "confidence",

        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",

        "image_x",
        "image_y",

        "world_x_m",
        "world_y_m",

        "zone",
        "state",

        "speed_kmh",
        "distance_to_crosswalk_m",
        "ttc_crosswalk_s",
    ]

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

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

    def write_frame(
        self,
        frame_idx: int,
        timestamp: float,
        objects,
    ) -> None:
        for obj in objects:
            x1, y1, x2, y2 = obj.bbox

            self.writer.writerow(
                {
                    "frame_idx": frame_idx,
                    "timestamp_s": round(timestamp, 4),

                    "track_id": obj.track_id,
                    "class_id": obj.class_id,
                    "class_name": obj.class_name,
                    "category": obj.category,
                    "confidence": round(
                        obj.confidence,
                        4,
                    ),

                    "bbox_x1": round(x1, 2),
                    "bbox_y1": round(y1, 2),
                    "bbox_x2": round(x2, 2),
                    "bbox_y2": round(y2, 2),

                    "image_x": round(
                        obj.image_x,
                        2,
                    ),
                    "image_y": round(
                        obj.image_y,
                        2,
                    ),

                    "world_x_m": round(
                        obj.world_x,
                        3,
                    ),
                    "world_y_m": round(
                        obj.world_y,
                        3,
                    ),

                    "zone": (
                        obj.zone
                        if obj.zone is not None
                        else ""
                    ),

                    "state": (
                        obj.state
                        if obj.state is not None
                        else ""
                    ),

                    "speed_kmh": (
                        round(
                            obj.speed_kmh,
                            2,
                        )
                        if obj.speed_kmh is not None
                        else ""
                    ),

                    "distance_to_crosswalk_m": (
                        round(
                            obj.distance_to_crosswalk_m,
                            3,
                        )
                        if obj.distance_to_crosswalk_m is not None
                        else ""
                    ),

                    "ttc_crosswalk_s": (
                        round(
                            obj.ttc_crosswalk_s,
                            3,
                        )
                        if obj.ttc_crosswalk_s is not None
                        else ""
                    ),
                }
            )

    def close(self):
        self.file.close()