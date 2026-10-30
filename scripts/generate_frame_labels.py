"""Build binary frame labels from manually annotated *source event* intervals.

Paths are read from the scenario YAML specified in scripts/settings.py
(CURRENT_CONFIGURATION), like the other pipeline scripts. Run from repository root:
 python scripts/generate_frame_labels.py

Required data_output keys: ml_frames_csv, risk_intervals_csv, frame_labels_csv.

Intervals have inclusive 0-based frame bounds. Only events explicitly present
in the annotations are assigned labels. Unannotated events remain unlabeled.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from settings import CURRENT_CONFIGURATION

OVERWRITE = False  # set to True to replace an existing output file


def load_config_paths(config_path: str) -> dict[str, str]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    data_output = config.get("data_output")
    if not isinstance(data_output, dict):
        raise ValueError(f"Missing 'data_output' section in {config_path}")

    required = ("ml_frames_csv", "frame_labels_csv")
    missing = [key for key in required if not data_output.get(key)]
    if missing:
        raise ValueError(
            f"Missing keys in 'data_output' of {config_path}: {', '.join(missing)}"
        )

    vqa = config.get("vqa")
    if not isinstance(vqa, dict):
        raise ValueError(f"Missing 'vqa' section in {config_path}")
    if not vqa.get("risk_intervals_csv"):
        raise ValueError(f"Missing key 'risk_intervals_csv' in 'vqa' of {config_path}")

    return {
        "frames": data_output["ml_frames_csv"],
        "intervals": vqa["risk_intervals_csv"],
        "output": data_output["frame_labels_csv"],
    }


def generate_labels(frames: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    frame_fields = {'video_id', 'source_event_id', 'frame_idx'}
    interval_fields = {'source_event_id', 'start_frame', 'end_frame'}
    if frame_fields - set(frames):
        raise ValueError(f"Frame data missing columns: {sorted(frame_fields - set(frames))}")
    if interval_fields - set(intervals):
        raise ValueError(f"Intervals missing columns: {sorted(interval_fields - set(intervals))}")

    f = frames[['video_id', 'source_event_id', 'frame_idx']].copy()
    a = intervals[list(interval_fields)].copy()
    for col in ['video_id', 'source_event_id']:
        if f[col].isna().any() or (f[col].astype(str).str.strip() == '').any():
            raise ValueError(f'Empty {col} in frames')
    if a['source_event_id'].isna().any() or (a['source_event_id'].astype(str).str.strip() == '').any():
        raise ValueError('Empty source_event_id in intervals')
    if f.duplicated(['video_id', 'frame_idx']).any():
        raise ValueError('Duplicated video_id + frame_idx in frame data')
    for column, table in [('frame_idx', f), ('start_frame', a), ('end_frame', a)]:
        numeric = pd.to_numeric(table[column], errors='coerce')
        if numeric.isna().any() or (numeric < 0).any() or (numeric % 1 != 0).any():
            raise ValueError(f'{column} must contain non-negative integer frame indices')
        table[column] = numeric.astype('int64')
    if (a['start_frame'] > a['end_frame']).any():
        raise ValueError('start_frame must be <= end_frame')

    # One source_event_id must correspond to a single video in this project.
    videos_per_event = f.groupby('source_event_id')['video_id'].nunique()
    if (videos_per_event > 1).any():
        raise ValueError('Multiple videos for the same source_event_id: annotate by video_id instead')

    output = f.copy()
    output['risk_gt'] = pd.Series(pd.NA, index=output.index, dtype='Int64')
    known_events = set(f.source_event_id)
    extras = set(a.source_event_id) - known_events
    if extras:
        print(f'INFO: {len(extras)} annotated events do not exist in input frames: {sorted(extras)}')

    for event_id, group in a.groupby('source_event_id'):
        if event_id not in known_events:
            continue
        ordered = group.sort_values('start_frame')
        if (ordered['start_frame'].iloc[1:].to_numpy() <= ordered['end_frame'].iloc[:-1].to_numpy()).any():
            raise ValueError(f'Overlapping danger intervals for {event_id}')
        mask = output.source_event_id == event_id
        frame_ids = output.loc[mask, 'frame_idx']
        labels = pd.Series(0, index=frame_ids.index, dtype='Int64')
        for start, end in zip(ordered.start_frame, ordered.end_frame):
            labels.loc[(frame_ids >= start) & (frame_ids <= end)] = 1
        output.loc[mask, 'risk_gt'] = labels
        min_frame, max_frame = frame_ids.min(), frame_ids.max()
        for start, end in zip(ordered.start_frame, ordered.end_frame):
            if start < min_frame or end > max_frame:
                print(f'WARNING: {event_id} interval [{start},{end}] exceeds available frame range [{min_frame},{max_frame}]')
    return output[['video_id','frame_idx','risk_gt']].sort_values(['video_id','frame_idx']).reset_index(drop=True)


def main():
    paths = load_config_paths(CURRENT_CONFIGURATION)
    print(f'Config: {CURRENT_CONFIGURATION}')
    output_path = Path(paths['output'])
    if output_path.exists() and not OVERWRITE:
        raise SystemExit(f'Output exists: {output_path} (set OVERWRITE = True to replace)')
    results = generate_labels(pd.read_csv(paths['frames']), pd.read_csv(paths['intervals']))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print('Saved:', output_path)
    print('Label counts including unannotated:', results.risk_gt.value_counts(dropna=False).to_dict())

if __name__ == '__main__':
    main()