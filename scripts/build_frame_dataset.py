#!/usr/bin/env python3
"""Build one ML record per frame using paths from a scenario YAML.

By default loads the YAML specified in scripts/settings.py, just like
analyze_tracks.py. Optional --config and individual file path overrides.

Expected data_output keys: csv, interactions_csv, interaction_frames_csv;
optional: risk_timeline_csv, ml_frames_csv, frame_labels_csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from settings import CURRENT_CONFIGURATION
import numpy as np
import pandas as pd


def load_config_paths(config_path: str) -> dict[str, str | None]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    data_output = config.get("data_output")
    if not isinstance(data_output, dict):
        raise ValueError(f"Missing 'data_output' section in {config_path}")

    required = ("csv", "interactions_csv", "interaction_frames_csv")
    missing = [key for key in required if not data_output.get(key)]
    if missing:
        raise ValueError(
            f"Missing keys in 'data_output' of {config_path}: {', '.join(missing)}"
        )

    # Keep the same convention as analyze_tracks.py: YAML paths are used
    # as provided, normally relative to the repository working directory.
    default_output = str(Path(data_output["interactions_csv"]).with_name(
        "scenario_ml_frames.csv"
    ))
    return {
        "tracks": data_output["csv"],
        "interactions": data_output["interactions_csv"],
        "frames": data_output["interaction_frames_csv"],
        "risk_timeline": data_output.get("risk_timeline_csv"),
        "labels": data_output.get("frame_labels_csv"),
        "output": data_output.get("ml_frames_csv", default_output),
    }


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def require(df: pd.DataFrame, columns: list[str], name: str) -> None:
    absent = sorted(set(columns) - set(df.columns))
    if absent:
        raise ValueError(f"{name}: missing columns: {absent}")


def verify_unique(df: pd.DataFrame, keys: list[str], name: str) -> None:
    if df.duplicated(keys).any():
        example = df.loc[df.duplicated(keys, keep=False), keys].head(3).to_dict('records')
        raise ValueError(f"{name}: duplicated {keys}; examples: {example}")


def numbers(df: pd.DataFrame, fields: list[str]) -> None:
    for col in fields:
        df[col] = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", default=CURRENT_CONFIGURATION,
        help="YAML config path (default from settings.py)",
    )
    parser.add_argument("--tracks", default=None, help="Override data_output.csv")
    parser.add_argument("--interactions", default=None, help="Override data_output.interactions_csv")
    parser.add_argument("--frames", default=None, help="Override data_output.interaction_frames_csv")
    parser.add_argument("--risk-timeline", default=None, help="Override data_output.risk_timeline_csv")
    parser.add_argument("--labels", default=None, help="Optional labels: video_id,frame_idx,risk_gt CSV")
    parser.add_argument("--output", default=None, help="Override data_output.ml_frames_csv")
    args = parser.parse_args()

    configured = load_config_paths(args.config)
    paths = {name: getattr(args, name) or value for name, value in configured.items()}
    print(f"Config: {args.config}")

    frames = read_csv(paths["frames"])
    frames = read_csv(paths["frames"])
    pairs = read_csv(paths["interactions"])
    tracks = read_csv(paths["tracks"])
    require(frames, ['video_id','frame_idx','timestamp_s','num_pedestrians','num_vehicles','num_interactions','pipeline_ok'], 'frames')
    require(pairs, ['video_id','source_event_id','frame_idx','pedestrian_track_id','vehicle_track_id',
                    'pedestrian_world_x_m','pedestrian_world_y_m','vehicle_world_x_m','vehicle_world_y_m'], 'interactions')
    require(tracks, ['frame_idx','track_id','category','state','zone','speed_kmh','distance_to_crosswalk_m','ttc_crosswalk_s'], 'tracks')
    verify_unique(frames, ['video_id','frame_idx'], 'frames')
    verify_unique(pairs, ['video_id','frame_idx','pedestrian_track_id','vehicle_track_id'], 'interactions')
    verify_unique(tracks, ['frame_idx','track_id'], 'tracks')

    # The current tracks CSV does not contain video_id. Treat inputs as ONE video.
    vids = frames['video_id'].dropna().astype(str).unique()
    pair_vids = pairs['video_id'].dropna().astype(str).unique()
    if len(vids) != 1 or (len(pair_vids) and (len(pair_vids) != 1 or pair_vids[0] != vids[0])):
        raise ValueError('Inputs must describe the same single video (run once per video).')
    video_id = vids[0]
    if tracks['frame_idx'].isin(frames['frame_idx']).sum() != len(tracks):
        raise ValueError('tracks contains frame indices not present in frames; check inputs')
    if not pairs['frame_idx'].isin(frames['frame_idx']).all():
        raise ValueError('interactions contains frame indices not present in frames')

    event_ids = pairs['source_event_id'].dropna().astype(str).unique()
    if len(event_ids) > 1:
        raise ValueError('Multiple source_event_id values in one interactions CSV')
    event_id = event_ids[0] if len(event_ids) else ''

    result = frames[['video_id','frame_idx','timestamp_s','pipeline_ok',
                     'num_pedestrians','num_vehicles','num_interactions']].copy()
    result.insert(1, 'source_event_id', event_id)
    result['frame_idx'] = pd.to_numeric(result['frame_idx'], errors='raise').astype('int64')
    tracks['frame_idx'] = pd.to_numeric(tracks['frame_idx'], errors='raise').astype('int64')
    pairs['frame_idx'] = pd.to_numeric(pairs['frame_idx'], errors='raise').astype('int64')
    numbers(tracks, ['speed_kmh','distance_to_crosswalk_m','ttc_crosswalk_s'])

    states = tracks['state'].fillna('').astype(str).str.lower().str.strip()
    categories = tracks['category'].fillna('').astype(str).str.lower().str.strip()
    zones = tracks['zone'].fillna('').astype(str).str.lower().str.strip()

    # A pedestrian on the marked crosswalk (zone OR crossing-like state).
    # Strict alternative: change to zones.eq('crosswalk') only.
    tracks['is_crossing'] = categories.eq('pedestrian') & (zones.eq('crosswalk') | states.isin(['entering','crossing']))
    tracks['is_waiting'] = categories.eq('pedestrian') & states.eq('waiting')
    tracks['is_approaching'] = categories.eq('vehicle') & states.eq('approaching')
    tracks['is_stopped'] = categories.eq('vehicle') & states.eq('stopped')
    counts = tracks.groupby('frame_idx', sort=False)[['is_crossing','is_waiting','is_approaching','is_stopped']].sum().rename(columns={
        'is_crossing':'num_pedestrians_crossing',
        'is_waiting':'num_pedestrians_waiting',
        'is_approaching':'num_vehicles_approaching',
        'is_stopped':'num_vehicles_stopped',
    }).reset_index()
    result = result.merge(counts, on='frame_idx', how='left', validate='one_to_one')
    for c in ['num_pedestrians_crossing','num_pedestrians_waiting','num_vehicles_approaching','num_vehicles_stopped']:
        result[c] = result[c].fillna(0).astype('int64')

    # Vehicle statistics taken from tracks, not interactions, to avoid repeating
    # the same vehicle once for each pedestrian.
    veh = tracks[categories.eq('vehicle')].copy()
    veh_stats = veh.groupby('frame_idx').agg(
        min_ttc_s=('ttc_crosswalk_s','min'),
        min_distance_crosswalk_m=('distance_to_crosswalk_m','min'),
        max_vehicle_speed_kmh=('speed_kmh','max'),
        mean_vehicle_speed_kmh=('speed_kmh','mean'),
        num_ttc_available=('ttc_crosswalk_s','count'),
    ).reset_index()
    result = result.merge(veh_stats, on='frame_idx', how='left', validate='one_to_one')
    result['num_ttc_available'] = result['num_ttc_available'].fillna(0).astype('int64')

    # Approximate point-to-point distance in ground-plane coordinates.
    # It is NOT minimum bounding-box/physical clearance.
    coords = ['pedestrian_world_x_m','pedestrian_world_y_m','vehicle_world_x_m','vehicle_world_y_m']
    numbers(pairs, coords)
    pairs['pair_distance_m'] = np.hypot(
        pairs['pedestrian_world_x_m'] - pairs['vehicle_world_x_m'],
        pairs['pedestrian_world_y_m'] - pairs['vehicle_world_y_m'],
    )
    pair_stats = pairs.groupby('frame_idx')['pair_distance_m'].min().rename('min_pedestrian_vehicle_distance_m').reset_index()
    result = result.merge(pair_stats, on='frame_idx', how='left', validate='one_to_one')

    # Optional baseline: 1 if at least one WARNING or DANGER in the frame.
    if paths["risk_timeline"]:
        risk = read_csv(paths["risk_timeline"])
        require(risk, ['frame_idx','risk_level'], 'risk timeline')
        verify_unique(risk, ['frame_idx'], 'risk timeline')
        levels = risk['risk_level'].fillna('').astype(str).str.lower().str.strip()
        bad = set(levels.unique()) - {'ok','safe','warning','danger'}
        if bad:
            raise ValueError(f'Unexpected risk timeline levels: {bad}')
        risk['rule_pred'] = levels.isin(['warning','danger']).astype('int64')
        result = result.merge(risk[['frame_idx','rule_pred']], on='frame_idx', how='left', validate='one_to_one')
        if result['rule_pred'].isna().any():
            raise ValueError('Missing rule predictions for some frames')
        result['rule_pred'] = result['rule_pred'].astype('int64')

    # Independent ground truth; optional until annotations are available.
    if paths["labels"]:
        labels = read_csv(paths["labels"])
        require(labels, ['video_id','frame_idx','risk_gt'], 'labels')
        verify_unique(labels, ['video_id','frame_idx'], 'labels')
        vals = pd.to_numeric(labels['risk_gt'], errors='coerce')
        if not vals.dropna().isin([0,1]).all():
            raise ValueError('risk_gt must contain 0, 1, or missing values')
        labels['risk_gt'] = vals.astype('Int64')
        result = result.merge(labels[['video_id','frame_idx','risk_gt']],
                              on=['video_id','frame_idx'], how='left', validate='one_to_one')
    else:
        result['risk_gt'] = pd.Series(pd.NA, index=result.index, dtype='Int64')

    # Do not silently interpret failed pipeline frames as true zero-object scenes.
    # Such records are retained but should be excluded from standard training.
    result['pipeline_ok'] = pd.to_numeric(result['pipeline_ok'], errors='coerce').astype('Int64')
    if not result['pipeline_ok'].isin([0,1]).all():
        raise ValueError('pipeline_ok must be 0 or 1')
    result = result.sort_values(['video_id','frame_idx']).reset_index(drop=True)
    out = Path(paths["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False, na_rep='', float_format='%.6f')
    print(f'Saved {len(result)} frame rows to {out}')
    print(f'Video: {video_id}; event: {event_id or "(not available)"}')
    print(f'With labels: {result["risk_gt"].notna().sum()}; pipeline failures: {(result["pipeline_ok"] == 0).sum()}')
    print('Use ONLY numeric feature columns for ML; exclude identifiers, pipeline_ok, rule_pred and risk_gt from X.')


if __name__ == '__main__':
    main()
