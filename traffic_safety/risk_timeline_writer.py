import csv
from pathlib import Path


class RiskTimelineCsvWriter:
    FIELDNAMES = [
        "frame_idx",
        "timestamp_s",
        "risk_level",
        "risk_numeric",
        "num_risk_events",
        "num_warning_events",
        "num_danger_events",
        "rules",
    ]

    RISK_TO_NUMERIC = {
        "ok": 0,
        "warning": 1,
        "danger": 2,
    }

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
        risk_level: str,
        risk_events,
    ) -> None:
        warning_events = [
            event for event in risk_events
            if event.level.value == "warning"
        ]

        danger_events = [
            event for event in risk_events
            if event.level.value == "danger"
        ]

        rules = sorted(
            {event.rule for event in risk_events}
        )

        self.writer.writerow(
            {
                "frame_idx": frame_idx,
                "timestamp_s": round(timestamp, 4),
                "risk_level": risk_level,
                "risk_numeric": self.RISK_TO_NUMERIC[risk_level],
                "num_risk_events": len(risk_events),
                "num_warning_events": len(warning_events),
                "num_danger_events": len(danger_events),
                "rules": ";".join(rules),
            }
        )

    def close(self) -> None:
        self.file.close()