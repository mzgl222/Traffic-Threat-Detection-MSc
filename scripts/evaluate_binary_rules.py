#!/usr/bin/env python3
"""Pilot frame-level evaluation of the binary OK/DANGER rules (BinaryRuleEngine).

Run from the repository root:
    python scripts/evaluate_binary_rules.py
    python scripts/evaluate_binary_rules.py --configs-dir configs --output-dir outputs/ml_experiment/binary_rule_evaluation

The rules are replayed offline on the saved data_output.csv (scenario_tracks.csv):
no detection/tracking is re-run, so differences to the old system come only
from the rules, not from new tracks.

Frame universe and labels come from load_recording() in evaluate_rule_system.py
(ml_frames_csv + frame_labels_csv), so both systems are evaluated on exactly
the same frames: risk_gt present and pipeline_ok == 1. Frames with no tracked
objects are absent from the tracks CSV and are predicted OK (no pair exists).
Old system prediction = rule_pred (warning|danger -> 1), as in the existing
rule evaluation.

Limitation: tracks CSV values are rounded (speed 2 dp, TTA 3 dp), so values
exactly at a threshold may differ from an online run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluate_rule_system import load_recording, metrics, read_yaml  # noqa: E402
from traffic_safety.binary_rules import BinaryRule, BinaryRuleEngine  # noqa: E402

R1 = BinaryRule.VEHICLE_ON_CROSSWALK_WITH_PEDESTRIAN
R2 = BinaryRule.VEHICLE_ARRIVING_WHILE_PEDESTRIAN_CROSSING

TRACK_COLUMNS = ["frame_idx", "track_id", "category", "zone", "speed_kmh", "ttc_crosswalk_s"]


def _none_if_missing(value):
    return None if pd.isna(value) else value


def load_tracks(path: Path) -> pd.DataFrame:
    tracks = pd.read_csv(path, low_memory=False)
    absent = sorted(set(TRACK_COLUMNS) - set(tracks.columns))
    if absent:
        raise ValueError(f"{path}: missing columns required by BinaryRuleEngine: {absent}")
    if tracks.duplicated(["frame_idx", "track_id"]).any():
        raise ValueError(f"{path}: duplicated (frame_idx, track_id)")
    tracks["frame_idx"] = pd.to_numeric(tracks["frame_idx"], errors="raise").astype("int64")
    # Non-numeric / empty -> NaN; the engine treats NaN as missing.
    for column in ("speed_kmh", "ttc_crosswalk_s"):
        tracks[column] = pd.to_numeric(tracks[column], errors="coerce")
    return tracks


def predict_frames(tracks: pd.DataFrame, engine: BinaryRuleEngine) -> pd.DataFrame:
    """Return per-frame new_pred, r1, r2, n_r1_pairs, n_r2_pairs for frames with objects."""
    rows = []
    for frame_idx, group in tracks.groupby("frame_idx", sort=True):
        objects = [
            SimpleNamespace(
                track_id=int(row.track_id),
                category=str(row.category),
                zone=_none_if_missing(row.zone),
                speed_kmh=_none_if_missing(row.speed_kmh),
                ttc_crosswalk_s=_none_if_missing(row.ttc_crosswalk_s),
            )
            for row in group.itertuples(index=False)
        ]
        events = engine.evaluate(objects)
        n_r1 = sum(event.rule == R1 for event in events)
        n_r2 = sum(event.rule == R2 for event in events)
        rows.append({
            "frame_idx": int(frame_idx),
            "new_pred": engine.frame_label(events),
            "r1": int(n_r1 > 0),
            "r2": int(n_r2 > 0),
            "n_r1_pairs": n_r1,
            "n_r2_pairs": n_r2,
        })
    return pd.DataFrame(rows, columns=["frame_idx", "new_pred", "r1", "r2", "n_r1_pairs", "n_r2_pairs"])


def evaluate_recording(config_path: Path, engine: BinaryRuleEngine) -> pd.DataFrame:
    cfg = read_yaml(config_path)
    frames = load_recording(config_path)  # labels, pipeline_ok, rule_pred, source_event_id
    tracks = load_tracks(Path(cfg["data_output"]["csv"]))

    unknown = set(tracks["frame_idx"]) - set(frames["frame_idx"])
    if unknown:
        raise ValueError(f"{config_path}: {len(unknown)} track frames not present in frame list")

    preds = predict_frames(tracks, engine)
    merged = frames.merge(preds, on="frame_idx", how="left", validate="one_to_one")
    # Frames without any tracked object: no pedestrian-vehicle pair -> OK.
    for column in ("new_pred", "r1", "r2", "n_r1_pairs", "n_r2_pairs"):
        merged[column] = merged[column].fillna(0).astype("int64")

    merged = merged.rename(columns={"rule_pred": "old_pred"})
    return merged[[
        "video_id", "source_event_id", "frame_idx", "risk_gt", "pipeline_ok",
        "old_pred", "new_pred", "r1", "r2", "n_r1_pairs", "n_r2_pairs",
    ]]


def metric_rows(evaluated: pd.DataFrame, group_label: str, video_id: str, event_id: str) -> list[dict]:
    y_true = evaluated["risk_gt"].astype(int).to_numpy()
    rows = []
    for system, column in (("old", "old_pred"), ("new", "new_pred")):
        rows.append({
            "source_event_id": event_id,
            "video_id": video_id,
            "system": system,
            **metrics(y_true, evaluated[column].astype(int).to_numpy()),
        })
    return rows


def alarm_counts(evaluated: pd.DataFrame) -> dict:
    return {
        "frames_new_danger": int(evaluated["new_pred"].sum()),
        "frames_r1": int(evaluated["r1"].sum()),
        "frames_r2": int(evaluated["r2"].sum()),
        "frames_r1_and_r2": int((evaluated["r1"] & evaluated["r2"]).sum()),
        "frames_r1_only": int((evaluated["r1"] & ~evaluated["r2"].astype(bool)).sum()),
        "frames_r2_only": int((evaluated["r2"] & ~evaluated["r1"].astype(bool)).sum()),
        "pair_events_r1": int(evaluated["n_r1_pairs"].sum()),
        "pair_events_r2": int(evaluated["n_r2_pairs"].sum()),
        "frames_r1_tp": int((evaluated["r1"].astype(bool) & (evaluated["risk_gt"] == 1)).sum()),
        "frames_r2_tp": int((evaluated["r2"].astype(bool) & (evaluated["risk_gt"] == 1)).sum()),
        "frames_old_danger": int(evaluated["old_pred"].sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs-dir", type=Path, default=Path("configs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ml_experiment/binary_rule_evaluation"))
    parser.add_argument("--arrival-time-s", type=float, default=4.0)
    parser.add_argument("--moving-speed-kmh", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing output directory")
    args = parser.parse_args()

    if args.output_dir.exists() and not args.overwrite:
        raise SystemExit(f"Output directory exists: {args.output_dir} (use --overwrite to replace)")

    engine = BinaryRuleEngine(arrival_time_s=args.arrival_time_s, moving_speed_kmh=args.moving_speed_kmh)
    configs = sorted(set(args.configs_dir.rglob("*.yaml")) | set(args.configs_dir.rglob("*.yml")))
    if not configs:
        raise SystemExit(f"No YAML files in {args.configs_dir}")

    recordings = []
    for config in configs:
        rows = evaluate_recording(config, engine)
        print(f"OK {config}: {rows.video_id.iloc[0]}, {len(rows)} frames")
        recordings.append(rows)

    all_rows = pd.concat(recordings, ignore_index=True)
    if all_rows.duplicated(["video_id", "frame_idx"]).any():
        raise ValueError("Duplicate (video_id, frame_idx) across configs")

    eligible = all_rows["risk_gt"].notna() & (all_rows["pipeline_ok"] == 1)
    evaluated = all_rows.loc[eligible].copy()
    if evaluated["old_pred"].isna().any():
        raise ValueError("Missing old rule prediction for some eligible frames")
    evaluated["risk_gt"] = evaluated["risk_gt"].astype(int)
    evaluated["old_pred"] = evaluated["old_pred"].astype(int)
    print(f"Frames: {len(all_rows)} | evaluated: {len(evaluated)}")

    metric_table, alarm_table = [], []
    for (event_id, video_id), part in evaluated.groupby(["source_event_id", "video_id"], sort=True):
        metric_table += metric_rows(part, "video", video_id, event_id)
        alarm_table.append({"source_event_id": event_id, "video_id": video_id, **alarm_counts(part)})
    metric_table += metric_rows(evaluated, "all", "ALL", "ALL")
    alarm_table.append({"source_event_id": "ALL", "video_id": "ALL", **alarm_counts(evaluated)})

    metrics_df = pd.DataFrame(metric_table)
    alarms_df = pd.DataFrame(alarm_table)
    comparison = metrics_df.pivot_table(
        index=["source_event_id"], columns="system",
        values=["precision", "recall", "f1", "TP", "FP", "FN", "TN"], sort=False,
    )
    comparison.columns = [f"{metric}_{system}" for metric, system in comparison.columns]
    comparison = comparison.reset_index()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(args.output_dir / "binary_metrics_per_video.csv", index=False)
    alarms_df.to_csv(args.output_dir / "binary_rule_alarm_counts.csv", index=False)
    comparison.to_csv(args.output_dir / "old_vs_new_comparison.csv", index=False)
    evaluated[[
        "video_id", "source_event_id", "frame_idx", "risk_gt",
        "old_pred", "new_pred", "r1", "r2", "n_r1_pairs", "n_r2_pairs",
    ]].to_csv(args.output_dir / "binary_frame_predictions.csv", index=False)
    (args.output_dir / "run_parameters.txt").write_text(
        f"arrival_time_s={engine.arrival_time_s}\nmoving_speed_kmh={engine.moving_speed_kmh}\n"
        f"configs={[str(c) for c in configs]}\n",
        encoding="utf-8",
    )

    fmt = lambda x: f"{x:.3f}"
    print("\nOLD vs NEW per recording:")
    print(comparison[["source_event_id", "precision_old", "precision_new", "recall_old",
                      "recall_new", "f1_old", "f1_new"]].to_string(index=False, float_format=fmt))
    print("\nConfusion matrices (pooled), rows=true [0,1], cols=pred [0,1]:")
    for _, m in metrics_df[metrics_df.video_id == "ALL"].iterrows():
        print(f"  {m.system}: [[{m.TN}, {m.FP}], [{m.FN}, {m.TP}]]")
    print("\nAlarm counts:")
    print(alarms_df.to_string(index=False))
    print(f"\nSaved to: {args.output_dir}")


if __name__ == "__main__":
    main()
