import math
from collections import defaultdict, deque

import numpy as np


def distance_to_line(
    x: float,
    y: float,
    line_start: tuple[float, float],
    line_end: tuple[float, float],
) -> float:
    """
    Calculate perpendicular distance from a point
    to an infinite line defined by two points.
    """

    x1, y1 = line_start
    x2, y2 = line_end

    numerator = abs(
        (y2 - y1) * x
        - (x2 - x1) * y
        + x2 * y1
        - y2 * x1
    )

    denominator = math.hypot(
        y2 - y1,
        x2 - x1,
    )

    if denominator == 0:
        raise ValueError(
            "Crosswalk reference line points "
            "must be different."
        )

    return numerator / denominator


class RiskMetricsEstimator:
    def __init__(
        self,
        history_window_s: float = 0.7,
        min_samples: int = 5,
        min_closing_speed_m_s: float = 0.5,
    ):
        self.history_window_s = history_window_s
        self.min_samples = min_samples
        self.min_closing_speed_m_s = (
            min_closing_speed_m_s
        )

        # track_id -> [(timestamp, distance), ...]
        self.distance_history = defaultdict(
            lambda: deque(maxlen=60)
        )

    def update_ttc(
        self,
        track_id: int,
        timestamp: float,
        distance_to_crosswalk_m: float,
    ) -> float | None:

        history = self.distance_history[
            track_id
        ]

        history.append(
            (
                timestamp,
                distance_to_crosswalk_m,
            )
        )

        # ---------------------------------------------
        # Keep only recent observations
        # ---------------------------------------------

        while (
            history
            and timestamp - history[0][0]
            > self.history_window_s
        ):
            history.popleft()

        # ---------------------------------------------
        # Not enough information yet
        # ---------------------------------------------

        if len(history) < self.min_samples:
            return None

        # ---------------------------------------------
        # Linear regression:
        #
        # distance = slope * time + intercept
        #
        # negative slope -> approaching crosswalk
        # positive slope -> moving away
        # ---------------------------------------------

        times = np.array(
            [
                item[0]
                for item in history
            ],
            dtype=float,
        )

        distances = np.array(
            [
                item[1]
                for item in history
            ],
            dtype=float,
        )

        # Shift time close to zero for numerical stability
        times = times - times[0]

        slope, _ = np.polyfit(
            times,
            distances,
            1,
        )

        # distance decreases when slope is negative
        closing_speed_m_s = -slope

        # ---------------------------------------------
        # Vehicle is not meaningfully approaching
        # ---------------------------------------------

        if (
            closing_speed_m_s
            < self.min_closing_speed_m_s
        ):
            return None

        current_distance = distances[-1]

        # ---------------------------------------------
        # TTC
        # ---------------------------------------------

        ttc = (
            current_distance
            / closing_speed_m_s
        )

        if ttc < 0:
            return None

        return float(ttc)

    def remove_track(
        self,
        track_id: int,
    ) -> None:
        self.distance_history.pop(
            track_id,
            None,
        )