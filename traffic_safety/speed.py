import math
from collections import defaultdict, deque
from statistics import median


class SpeedEstimator:
    def __init__(
        self,
        history_length: int = 15,
        speed_history_length: int = 21,
        max_speed_kmh: float = 180.0,
    ):
        self.history_length = history_length
        self.speed_history_length = speed_history_length
        self.max_speed_kmh = max_speed_kmh

        # historia pozycji:
        # track_id -> [(timestamp, world_x, world_y), ...]
        self.position_history = defaultdict(
            lambda: deque(maxlen=self.history_length)
        )

        # historia wyliczonych prędkości:
        # track_id -> [speed1, speed2, ...]
        self.speed_history = defaultdict(
            lambda: deque(
                maxlen=self.speed_history_length
            )
        )

    def update(
        self,
        track_id: int,
        timestamp: float,
        world_x: float,
        world_y: float,
    ) -> float | None:

        history = self.position_history[
            track_id
        ]

        history.append(
            (
                timestamp,
                world_x,
                world_y,
            )
        )

        # potrzebujemy co najmniej 2 punktów
        if len(history) < 2:
            return None

        previous_time, previous_x, previous_y = (
            history[-2]
        )

        current_time, current_x, current_y = (
            history[-1]
        )

        dt = (
            current_time
            - previous_time
        )

        if dt <= 0:
            return None

        distance_m = math.hypot(
            current_x - previous_x,
            current_y - previous_y,
        )

        speed_m_s = (
            distance_m / dt
        )

        speed_kmh = (
            speed_m_s * 3.6
        )

        # odrzucamy ewidentnie błędne skoki
        if (
            speed_kmh < 0
            or speed_kmh > self.max_speed_kmh
        ):
            return self._get_smoothed_speed(
                track_id
            )

        self.speed_history[
            track_id
        ].append(speed_kmh)

        return self._get_smoothed_speed(
            track_id
        )

    def _get_smoothed_speed(
        self,
        track_id: int,
    ) -> float | None:

        history = self.speed_history[
            track_id
        ]

        if len(history) < 3:
            return None

        return float(
            median(history)
        )

    def remove_track(
        self,
        track_id: int,
    ) -> None:

        self.position_history.pop(
            track_id,
            None,
        )

        self.speed_history.pop(
            track_id,
            None,
        )