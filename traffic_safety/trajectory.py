from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class TrajectoryPoint:
    timestamp: float

    image_x: float
    image_y: float

    world_x: float
    world_y: float


class TrajectoryStore:
    def __init__(self, history_length: int = 60):
        self.history_length = history_length

        self._history = defaultdict(
            lambda: deque(maxlen=self.history_length)
        )

    def update(
        self,
        track_id: int,
        timestamp: float,
        image_x: float,
        image_y: float,
        world_x: float,
        world_y: float,
    ) -> None:
        point = TrajectoryPoint(
            timestamp=timestamp,
            image_x=image_x,
            image_y=image_y,
            world_x=world_x,
            world_y=world_y,
        )

        self._history[track_id].append(point)

    def get(
        self,
        track_id: int,
    ) -> list[TrajectoryPoint]:
        return list(
            self._history.get(track_id, [])
        )

    def clear(
        self,
        track_id: int,
    ) -> None:
        self._history.pop(track_id, None)

    def clear_all(self) -> None:
        self._history.clear()