"""Frame-level aggregation of per-object motion features (Phase II, B4).

Input: per-observation features from traffic_safety/motion_features.py
(e.g. outputs/features_v2/<scene>/scenario_tracks_features_v2.csv).

Frame features
--------------
min_time_since_crosswalk_entry_s
    Minimum *known* time_since_crosswalk_entry_s among pedestrians whose
    stabilized zone in this frame is "crosswalk". NaN when no such pedestrian
    has a known entry time (none on the crosswalk, or all entries unobserved).

num_crosswalk_pedestrians_unknown_entry
    Number of pedestrians on the crosswalk (stabilized zone) whose entry time
    is unknown. 0 (not missing) when nobody is on the crosswalk.

Together they keep three cases apart without any extra column:
    nobody on crosswalk          -> min NaN,  unknown == 0
    only unknown entries         -> min NaN,  unknown  > 0
    at least one known entry     -> min set,  unknown >= 0

"On the crosswalk" uses the stabilized zone (motion_stable_zone), i.e. the
same population for which time_since_crosswalk_entry_s is defined. It may
differ slightly from num_pedestrians_crossing (raw zone OR legacy state).

Causality: each frame uses only observations from that frame, whose motion
features are themselves causal, so no information from later frames is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from traffic_safety.motion_features import CROSSWALK_ZONE

MIN_ENTRY_COLUMN = "min_time_since_crosswalk_entry_s"
UNKNOWN_ENTRY_COLUMN = "num_crosswalk_pedestrians_unknown_entry"
FRAME_FEATURE_COLUMNS = (MIN_ENTRY_COLUMN, UNKNOWN_ENTRY_COLUMN)

REQUIRED_COLUMNS = (
    "frame_idx",
    "category",
    "motion_stable_zone",
    "time_since_crosswalk_entry_s",
)


def aggregate_crosswalk_entry(observations: pd.DataFrame, frame_indices) -> pd.DataFrame:
    """Aggregate observations of ONE video to one row per frame_idx.

    frame_indices: all frames of the video (frames without observations get
    NaN / 0). Raises ValueError on missing columns, duplicated frame indices,
    observations from frames not listed, or a known entry time outside the
    crosswalk (inconsistent input).
    """
    absent = sorted(set(REQUIRED_COLUMNS) - set(observations.columns))
    if absent:
        raise ValueError(f"observations are missing columns: {absent}")

    frames = pd.Index(pd.to_numeric(pd.Series(list(frame_indices)), errors="raise").astype("int64"))
    if frames.has_duplicates:
        raise ValueError("frame_indices contain duplicates")

    obs_frames = pd.to_numeric(observations["frame_idx"], errors="raise").astype("int64")
    if not obs_frames.isin(frames).all():
        raise ValueError("observations contain frame indices not in frame_indices")

    category = observations["category"].astype("string").str.strip().str.lower()
    zone = observations["motion_stable_zone"].astype("string").str.strip()
    entry = pd.to_numeric(observations["time_since_crosswalk_entry_s"], errors="coerce")
    entry = entry.where(np.isfinite(entry))

    on_crosswalk = (category == "pedestrian").fillna(False) & (zone == CROSSWALK_ZONE).fillna(False)
    if (entry.notna() & ~on_crosswalk & (category == "pedestrian").fillna(False)).any():
        raise ValueError("known crosswalk entry time outside the crosswalk")

    selected = pd.DataFrame(
        {
            "frame_idx": obs_frames[on_crosswalk].to_numpy(),
            "entry": entry[on_crosswalk].to_numpy(dtype="float64"),
        }
    )
    grouped = selected.groupby("frame_idx")["entry"]
    min_entry = grouped.min()  # NaN when all are NaN
    unknown = grouped.apply(lambda values: int(values.isna().sum())) if len(selected) else pd.Series(dtype="int64")

    result = pd.DataFrame(index=frames)
    result.index.name = "frame_idx"
    result[MIN_ENTRY_COLUMN] = min_entry.reindex(frames).astype("float64").to_numpy()
    result[UNKNOWN_ENTRY_COLUMN] = unknown.reindex(frames).fillna(0).astype("int64").to_numpy()
    return result.reset_index()
