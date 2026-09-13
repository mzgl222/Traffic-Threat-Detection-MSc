from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Zone:
    name: str
    polygon: np.ndarray

    def contains(self, x: float, y: float) -> bool:
        result = cv2.pointPolygonTest(
            self.polygon,
            (float(x), float(y)),
            False,
        )

        return result >= 0


class ZoneManager:
    def __init__(self, zones_config: dict):
        self.zones = []

        for name, points in zones_config.items():
            polygon = np.array(
                points,
                dtype=np.int32,
            )

            self.zones.append(
                Zone(
                    name=name,
                    polygon=polygon,
                )
            )

    def get_zone(
        self,
        x: float,
        y: float,
    ) -> str | None:
        for zone in self.zones:
            if zone.contains(x, y):
                return zone.name

        return None

    def get_all(self) -> list[Zone]:
        return self.zones