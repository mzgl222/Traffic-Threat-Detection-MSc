"""Causal temporal motion features based on the zone history of each object.

Features (per observation of one object):

time_in_current_zone_s
    Seconds since the object entered its current *stabilized* zone
    (None = "outside all zones" also counts as a zone). Missing (None) when
    the beginning of the stay was not observed, i.e. the object was already in
    this zone at its first observation.

time_since_crosswalk_entry_s
    Same as time_in_current_zone_s, but only while the stabilized zone is
    "crosswalk"; missing otherwise or when the entry was not observed.

came_from_waiting / came_from_crosswalk
    Whether the most recent stabilized zone *before* the current stay
    (ignoring periods outside all zones) was a pedestrian waiting zone /
    the crosswalk. Missing when there is no earlier stabilized stay for this
    object (history unknown). False when only "outside" was observed before.

Design rules
- Objects are identified by (category, track_id), as in StateEstimator;
  different track IDs are never merged.
- Strictly causal: the value for an observation uses only that observation
  and earlier ones. Calling update() never changes values already returned.
- Zone flicker is filtered with the same rule as StateEstimator: a new zone
  is accepted after `confirm_observations` consecutive observations (3).
  Once accepted, the stay is dated back to the first of these observations
  (its real beginning). Features of earlier observations are not revised.
- Time comes from timestamps (seconds), not from a fixed FPS. Gaps in a
  track (missed detections) are simply time between observations.
- Unknown values are None (never 0).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from traffic_safety.state import PEDESTRIAN_WAITING_ZONES

CROSSWALK_ZONE = "crosswalk"

FEATURE_COLUMNS = (
    "time_in_current_zone_s",
    "time_since_crosswalk_entry_s",
    "came_from_waiting",
    "came_from_crosswalk",
)


@dataclass(frozen=True)
class MotionFeatures:
    stable_zone: str | None
    time_in_current_zone_s: float | None
    time_since_crosswalk_entry_s: float | None
    came_from_waiting: bool | None
    came_from_crosswalk: bool | None


_UNSET = object()


@dataclass
class _TrackHistory:
    last_timestamp: float
    stable_zone: str | None
    stable_since: float | None          # None = beginning not observed
    origin_zone: object = _UNSET        # _UNSET = no earlier stay observed
    candidate_zone: object = _UNSET
    candidate_since: float | None = None
    candidate_count: int = 0
    previous_non_none_zone: str | None = None
    had_previous_stay: bool = False


def _normalize_zone(zone) -> str | None:
    if zone is None:
        return None
    try:
        if zone != zone:  # NaN
            return None
    except TypeError:  # pandas.NA
        return None
    text = str(zone).strip()
    return text or None


class MotionFeatureTracker:
    def __init__(self, confirm_observations: int = 3):
        if (
            isinstance(confirm_observations, bool)
            or not isinstance(confirm_observations, int)
            or confirm_observations < 1
        ):
            raise ValueError("confirm_observations must be an integer >= 1")
        self.confirm_observations = confirm_observations
        self._tracks: dict[tuple[str, int], _TrackHistory] = {}

    @staticmethod
    def _key(obj) -> tuple[str, int]:
        category = getattr(obj, "category", None)
        track_id = getattr(obj, "track_id", None)
        if category is None or str(category).strip() == "":
            raise ValueError("object category is missing")
        if track_id is None or (isinstance(track_id, float) and math.isnan(track_id)):
            raise ValueError("object track_id is missing")
        return (str(category), int(track_id))

    @staticmethod
    def _validate_timestamp(timestamp_s) -> float:
        try:
            value = float(timestamp_s)
        except (TypeError, ValueError):
            raise ValueError(f"timestamp must be a number, got {timestamp_s!r}") from None
        if not math.isfinite(value):
            raise ValueError(f"timestamp must be finite, got {timestamp_s!r}")
        return value

    def update(self, obj, timestamp_s) -> MotionFeatures:
        """Register one observation (obj.category, obj.track_id, obj.zone) at timestamp_s."""
        key = self._key(obj)
        timestamp = self._validate_timestamp(timestamp_s)
        zone = _normalize_zone(getattr(obj, "zone", None))

        history = self._tracks.get(key)

        if history is None:
            history = _TrackHistory(
                last_timestamp=timestamp,
                stable_zone=zone,
                stable_since=None,
            )
            self._tracks[key] = history
            return self._features(history, timestamp)

        if timestamp <= history.last_timestamp:
            raise ValueError(
                f"timestamps must strictly increase for {key}: "
                f"{timestamp} after {history.last_timestamp}"
            )
        history.last_timestamp = timestamp

        if zone == history.stable_zone:
            # Back in the stable zone: discard any pending (flickering) candidate.
            history.candidate_zone = _UNSET
            history.candidate_count = 0
            history.candidate_since = None
        else:
            if history.candidate_zone is not _UNSET and zone == history.candidate_zone:
                history.candidate_count += 1
            else:
                history.candidate_zone = zone
                history.candidate_count = 1
                history.candidate_since = timestamp

            if history.candidate_count >= self.confirm_observations:
                self._confirm_candidate(history)

        return self._features(history, timestamp)

    @staticmethod
    def _confirm_candidate(history: _TrackHistory) -> None:
        # Remember the last non-"outside" zone before the new stay; periods
        # outside all zones do not erase it (gaps between polygons).
        if history.stable_zone is not None:
            history.previous_non_none_zone = history.stable_zone
        history.origin_zone = history.previous_non_none_zone
        history.had_previous_stay = True

        history.stable_zone = history.candidate_zone
        history.stable_since = history.candidate_since

        history.candidate_zone = _UNSET
        history.candidate_count = 0
        history.candidate_since = None

    @staticmethod
    def _features(history: _TrackHistory, timestamp: float) -> MotionFeatures:
        if history.stable_since is None:
            time_in_zone = None
        else:
            time_in_zone = timestamp - history.stable_since

        time_since_entry = (
            time_in_zone if history.stable_zone == CROSSWALK_ZONE else None
        )

        if not history.had_previous_stay:
            came_from_waiting = None
            came_from_crosswalk = None
        else:
            origin = history.origin_zone
            came_from_waiting = origin in PEDESTRIAN_WAITING_ZONES
            came_from_crosswalk = origin == CROSSWALK_ZONE

        return MotionFeatures(
            stable_zone=history.stable_zone,
            time_in_current_zone_s=time_in_zone,
            time_since_crosswalk_entry_s=time_since_entry,
            came_from_waiting=came_from_waiting,
            came_from_crosswalk=came_from_crosswalk,
        )


def compute_motion_features(tracks, confirm_observations: int = 3):
    """Compute features for a tracks table (scenario_tracks.csv schema).

    Required columns: frame_idx, timestamp_s, track_id, category, zone.
    Rows are processed in frame order (stable within a frame), so each value
    depends only on the current and earlier frames. Returns a DataFrame with
    FEATURE_COLUMNS (plus stable_zone) aligned with the input index; missing
    values are NaN / <NA>. The input is not modified.
    """
    import pandas as pd

    required = {"frame_idx", "timestamp_s", "track_id", "category", "zone"}
    absent = sorted(required - set(tracks.columns))
    if absent:
        raise ValueError(f"tracks table is missing columns: {absent}")

    ordered = tracks.sort_values("frame_idx", kind="stable")
    tracker = MotionFeatureTracker(confirm_observations=confirm_observations)

    records = {}
    for row in ordered.itertuples():
        features = tracker.update(
            _Observation(row.category, row.track_id, row.zone),
            row.timestamp_s,
        )
        records[row.Index] = features

    result = pd.DataFrame(
        [
            {
                "stable_zone": records[index].stable_zone,
                "time_in_current_zone_s": records[index].time_in_current_zone_s,
                "time_since_crosswalk_entry_s": records[index].time_since_crosswalk_entry_s,
                "came_from_waiting": records[index].came_from_waiting,
                "came_from_crosswalk": records[index].came_from_crosswalk,
            }
            for index in tracks.index
        ],
        index=tracks.index,
    )
    for column in ("time_in_current_zone_s", "time_since_crosswalk_entry_s"):
        result[column] = pd.to_numeric(result[column], errors="coerce").astype("float64")
    for column in ("came_from_waiting", "came_from_crosswalk"):
        result[column] = result[column].astype("boolean")
    return result


@dataclass(frozen=True)
class _Observation:
    category: str
    track_id: int
    zone: object = field(default=None)
