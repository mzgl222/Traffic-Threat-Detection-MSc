import math
from collections import defaultdict, deque


class SpeedEstimator:
    def __init__(self, history_length: int = 10):
        self.history_length = history_length

        self.track_history = defaultdict(
            lambda: deque(maxlen=self.history_length)
        )

    def update(
        self,
        track_id: int,
        timestamp: float,
        world_x: float,
        world_y: float
    ) -> float | None:
        history = self.track_history[track_id]

        history.append(
            (timestamp, world_x, world_y)
        )

        if len(history) < 2:
            return None

        old_time, old_x, old_y = history[0]
        new_time, new_x, new_y = history[-1]

        dt = new_time - old_time

        if dt <= 0:
            return None

        distance_m = math.hypot(
            new_x - old_x,
            new_y - old_y
        )

        speed_m_s = distance_m / dt
        speed_km_h = speed_m_s * 3.6

        return speed_km_h