import csv
from pathlib import Path


class RiskEventCsvWriter:
    FIELDNAMES = [
        "frame_idx",
        "timestamp_s",
        "pedestrian_id",
        "vehicle_id",
        "risk_level",
        "rule",
        "ttc_s",
        "distance_m",
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
        events,
    ) -> None:
        for event in events:
            self.writer.writerow(
                {
                    "frame_idx": frame_idx,
                    "timestamp_s": round(
                        timestamp,
                        4,
                    ),
                    "pedestrian_id": (
                        event.pedestrian_id
                    ),
                    "vehicle_id": (
                        event.vehicle_id
                    ),
                    "risk_level": (
                        event.level.value
                    ),
                    "rule": event.rule,
                    "ttc_s": (
                        round(
                            event.ttc_s,
                            3,
                        )
                        if event.ttc_s is not None
                        else ""
                    ),
                    "distance_m": (
                        round(
                            event.distance_m,
                            3,
                        )
                        if event.distance_m is not None
                        else ""
                    ),
                }
            )

    def close(self) -> None:
        self.file.close()