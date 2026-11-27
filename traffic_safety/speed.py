import math
from collections import defaultdict, deque
from statistics import median

MEDIAN_STEP = "median_step"
REGRESSION = "regression"
METHODS = (MEDIAN_STEP, REGRESSION)


class SpeedEstimator:
    """Vehicle speed from world-coordinate positions.

    method="median_step" (default, original behaviour)
        Speed between the two latest observations, smoothed with the median of
        the last `speed_history_length` step speeds. Step lengths are always
        positive, so detection jitter of a stationary object gives a positive
        speed (bias).

    method="regression" (Phase II, P1)
        Least-squares slope of world_x(t) and world_y(t), fitted separately on
        the observations of the last `window_s` seconds (real timestamps,
        causal: current and earlier observations only). Speed = length of the
        fitted velocity vector, in km/h. Returns None when the window holds
        fewer than `min_samples` observations or spans less than
        `min_coverage * window_s` (start of a track, after a gap longer than
        the window). No further smoothing or clipping.
        Delay: about window_s / 2 (0.25 s for 0.5 s). Speed is underestimated
        while accelerating / turning sharply and overestimated while braking.
    """

    def __init__(
        self,
        history_length: int = 15,
        speed_history_length: int = 21,
        max_speed_kmh: float = 180.0,
        method: str = MEDIAN_STEP,
        window_s: float = 0.5,
        min_samples: int = 5,
        min_coverage: float = 0.6,
    ):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        if not (isinstance(window_s, (int, float)) and math.isfinite(window_s) and window_s > 0):
            raise ValueError("window_s must be a positive finite number")
        if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 2:
            raise ValueError("min_samples must be an integer >= 2")
        if not (0 < min_coverage <= 1):
            raise ValueError("min_coverage must be in (0, 1]")

        self.history_length = history_length
        self.speed_history_length = speed_history_length
        self.max_speed_kmh = max_speed_kmh
        self.method = method
        self.window_s = float(window_s)
        self.min_samples = min_samples
        self.min_coverage = float(min_coverage)

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

        # method="regression": time window of observations per track
        self.window_history = defaultdict(deque)

    def update(
        self,
        track_id: int,
        timestamp: float,
        world_x: float,
        world_y: float,
    ) -> float | None:
        if self.method == REGRESSION:
            return self._update_regression(track_id, timestamp, world_x, world_y)

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

    # ---------------------------------------------------------
    # method="regression"
    # ---------------------------------------------------------

    def _update_regression(self, track_id, timestamp, world_x, world_y) -> float | None:
        try:
            t, x, y = float(timestamp), float(world_x), float(world_y)
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(t) and math.isfinite(x) and math.isfinite(y)):
            return None  # invalid observation is ignored, history unchanged

        window = self.window_history[track_id]
        if window and t <= window[-1][0]:
            return None  # non-increasing timestamp: ignored, history unchanged

        window.append((t, x, y))
        # keep only observations within the causal window [t - window_s, t]
        while t - window[0][0] > self.window_s + 1e-9:
            window.popleft()

        n = len(window)
        if n < self.min_samples:
            return None
        span = t - window[0][0]
        if span < self.min_coverage * self.window_s - 1e-9:
            return None

        mean_t = sum(p[0] for p in window) / n
        mean_x = sum(p[1] for p in window) / n
        mean_y = sum(p[2] for p in window) / n
        stt = sum((p[0] - mean_t) ** 2 for p in window)
        if stt <= 0:
            return None
        vx = sum((p[0] - mean_t) * (p[1] - mean_x) for p in window) / stt
        vy = sum((p[0] - mean_t) * (p[2] - mean_y) for p in window) / stt
        return math.hypot(vx, vy) * 3.6

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

        self.window_history.pop(
            track_id,
            None,
        )


def resolve_speed_settings(tracking_config: dict, method: str | None = None,
                           window_s: float | None = None) -> dict:
    """Speed-estimation settings: CLI value > config `tracking` value > default.

    Defaults keep the original behaviour (median_step, window 0.5 s used only by
    the regression method).
    """
    tracking_config = tracking_config or {}
    method = method or tracking_config.get("speed_method", MEDIAN_STEP)
    window = window_s if window_s is not None else tracking_config.get("speed_window_s", 0.5)
    return {"method": str(method).strip().lower(), "window_s": float(window)}


def speed_estimator_from_config(tracking_config: dict, method: str | None = None,
                                window_s: float | None = None) -> "SpeedEstimator":
    """Single construction point used by run.py and by the offline replay."""
    settings = resolve_speed_settings(tracking_config, method, window_s)
    return SpeedEstimator(
        history_length=tracking_config["history_length"],
        speed_history_length=tracking_config["speed_history_length"],
        method=settings["method"],
        window_s=settings["window_s"],
    )
