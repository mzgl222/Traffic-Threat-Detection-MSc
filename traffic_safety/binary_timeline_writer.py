import csv
from pathlib import Path

from traffic_safety.binary_rules import DANGER, OK, BinaryRule


class BinaryTimelineCsvWriter:
    """One row per processed frame with the binary OK/DANGER decision."""

    FIELDNAMES = [
        "frame_idx",
        "timestamp_s",
        "risk_binary",
        "risk_label",
        "r1",
        "r2",
        "num_r1_pairs",
        "num_r2_pairs",
        "rules",
    ]

    LABELS = {
        OK: "ok",
        DANGER: "danger",
    }

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

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
        risk_binary: int,
        events,
    ) -> None:
        if risk_binary not in self.LABELS:
            raise ValueError(f"risk_binary must be 0 or 1, got {risk_binary!r}")

        num_r1 = sum(
            event.rule == BinaryRule.VEHICLE_ON_CROSSWALK_WITH_PEDESTRIAN
            for event in events
        )
        num_r2 = sum(
            event.rule == BinaryRule.VEHICLE_ARRIVING_WHILE_PEDESTRIAN_CROSSING
            for event in events
        )

        self.writer.writerow(
            {
                "frame_idx": frame_idx,
                "timestamp_s": round(timestamp, 4),
                "risk_binary": risk_binary,
                "risk_label": self.LABELS[risk_binary],
                "r1": int(num_r1 > 0),
                "r2": int(num_r2 > 0),
                "num_r1_pairs": num_r1,
                "num_r2_pairs": num_r2,
                "rules": ";".join(sorted({event.rule.value for event in events})),
            }
        )

    def close(self) -> None:
        self.file.close()
