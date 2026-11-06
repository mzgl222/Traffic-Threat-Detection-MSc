#!/usr/bin/env python3
"""Frame-level evaluation of rule-based risk detection against manual labels.

Run from the repository root:
    python scripts/evaluate_rule_system.py --configs-dir configs --skip-missing
or for one recording:
    python scripts/evaluate_rule_system.py --config configs/20230707_12_SN17_T1.yaml

For each YAML, reads:
  data_output.ml_frames_csv        (feature rows, preferably containing rule_pred)
  data_output.frame_labels_csv     (manual 0/1 labels)
  data_output.risk_timeline_csv    (fallback when rule_pred is absent)
  study_metadata.source_event_id   (optional validation)

Only frames with a valid risk_gt and pipeline_ok==1 enter the metrics.
Unlabeled frames are NEVER assumed to be safe. Aggregated frame-level
results can overrepresent long events; inspect per-event results too.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

KEYS = ["video_id", "frame_idx"]


def read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict) or not isinstance(cfg.get("data_output"), dict):
        raise ValueError(f"{path}: missing data_output section")
    return cfg


def required_columns(df: pd.DataFrame, names: list[str], source: Path) -> None:
    absent = sorted(set(names) - set(df.columns))
    if absent:
        raise ValueError(f"{source}: missing columns: {absent}")


def numeric_binary(series: pd.Series, name: str, path: Path) -> pd.Series:
    result = pd.to_numeric(series, errors="coerce")
    invalid = series.notna() & result.isna()
    if invalid.any() or not result.dropna().isin([0, 1]).all():
        raise ValueError(f"{path}: {name} must be 0, 1 or empty")
    return result


def read_csv(path: Path, cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, dtype={"video_id": "string"})
    required_columns(df, cols, path)
    df["frame_idx"] = pd.to_numeric(df["frame_idx"], errors="raise")
    if df["frame_idx"].isna().any() or (df["frame_idx"] % 1 != 0).any():
        raise ValueError(f"{path}: frame_idx must be integer")
    df["frame_idx"] = df["frame_idx"].astype("int64")
    return df


def load_recording(config_path: Path) -> pd.DataFrame:
    cfg = read_yaml(config_path)
    data = cfg["data_output"]
    for name in ("ml_frames_csv", "frame_labels_csv"):
        if not data.get(name):
            raise ValueError(f"{config_path}: missing data_output.{name}")
    # Same convention as the existing scripts: paths in YAML are relative to repo cwd.
    frame_path = Path(data["ml_frames_csv"])
    label_path = Path(data["frame_labels_csv"])
    frames = read_csv(frame_path, KEYS)
    labels = read_csv(label_path, KEYS + ["risk_gt"])
    for table, path in ((frames, frame_path), (labels, label_path)):
        if table[KEYS].isna().any().any() or table.duplicated(KEYS).any():
            raise ValueError(f"{path}: empty or duplicated (video_id, frame_idx)")
    if frames.empty:
        raise ValueError(f"{frame_path}: no rows")
    vids = frames.video_id.dropna().unique()
    if len(vids) != 1:
        raise ValueError(f"{frame_path}: expected exactly one video_id, got {list(vids)}")
    if not set(labels.video_id.dropna().unique()).issubset(set(vids)):
        raise ValueError(f"{label_path}: video_id disagrees with {frame_path}")
    if "rule_pred" in frames:
        frames["rule_pred"] = numeric_binary(frames["rule_pred"], "rule_pred", frame_path)
    else:
        timeline_path = data.get("risk_timeline_csv")
        if not timeline_path:
            raise ValueError(f"{config_path}: rule_pred absent; define risk_timeline_csv")
        timeline_path = Path(timeline_path)
        timeline = read_csv(timeline_path, ["frame_idx", "risk_level"])
        if timeline.frame_idx.duplicated().any():
            raise ValueError(f"{timeline_path}: duplicated frame_idx")
        level = timeline.risk_level.astype("string").str.strip().str.lower()
        if not level.isin(["ok", "safe", "warning", "danger"]).all():
            raise ValueError(f"{timeline_path}: unknown or missing risk_level")
        timeline["rule_pred"] = level.isin(["warning", "danger"]).astype(int)
        frames = frames.merge(timeline[["frame_idx", "rule_pred"]], on="frame_idx", how="left", validate="one_to_one")
    labels["risk_gt"] = numeric_binary(labels["risk_gt"], "risk_gt", label_path)
    merged = frames.merge(labels[KEYS + ["risk_gt"]], on=KEYS, how="left", validate="one_to_one", suffixes=("", "_labels"))
    # Defensive handling of old feature exports that included an empty risk_gt.
    if "risk_gt_labels" in merged:
        merged = merged.drop(columns=["risk_gt"]).rename(columns={"risk_gt_labels": "risk_gt"})
    merged["pipeline_ok"] = numeric_binary(merged["pipeline_ok"], "pipeline_ok", frame_path) if "pipeline_ok" in merged else 1
    event_in_yaml = str(cfg.get("study_metadata", {}).get("source_event_id", "")).strip()
    if "source_event_id" in merged:
        events = merged.source_event_id.dropna().astype(str).unique()
        if len(events) > 1 or (event_in_yaml and len(events) == 1 and events[0] != event_in_yaml):
            raise ValueError(f"{frame_path}: inconsistent source_event_id compared with YAML")
        event_id = events[0] if len(events) else event_in_yaml
    else:
        event_id = event_in_yaml
    if not event_id:
        raise ValueError(f"{config_path}: missing source_event_id in both YAML and features")
    merged["source_event_id"] = event_id
    merged["config_file"] = str(config_path)
    return merged


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    safe_div = lambda numerator, denominator: numerator / denominator if denominator else 0.0
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    return {
        "frames": len(y_true), "positive_frames": int(sum(y_true)),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": precision, "recall": recall,
        "f1": safe_div(2 * precision * recall, precision + recall),
        "accuracy": safe_div(tp + tn, len(y_true)),
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--config", type=Path, help="Evaluate exactly one scenario YAML")
    group.add_argument("--configs-dir", type=Path, default=Path("configs"), help="Find *.yaml and *.yml recursively (default: configs)")
    p.add_argument("--skip-missing", action="store_true", help="Skip YAMLs whose input CSVs are not yet generated")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/ml_experiment/rule_evaluation"))
    args = p.parse_args()
    configs = [args.config] if args.config else sorted(set(args.configs_dir.rglob("*.yaml")) | set(args.configs_dir.rglob("*.yml")))
    if not configs:
        p.error("No YAML files found")

    recordings = []
    seen_videos: set[str] = set()
    for config in configs:
        try:
            rows = load_recording(config)
        except FileNotFoundError as exc:
            if args.skip_missing:
                print(f"SKIP {config}: {exc.filename} not found")
                continue
            raise
        video_id = str(rows.video_id.iloc[0])
        if video_id in seen_videos:
            raise ValueError(f"Duplicate video_id across YAMLs: {video_id}")
        seen_videos.add(video_id)
        print(f"OK {config}: {video_id}, {len(rows)} frames")
        recordings.append(rows)
    if not recordings:
        raise SystemExit("No recordings loaded. Check YAML paths and --skip-missing.")

    all_rows = pd.concat(recordings, ignore_index=True)
    eligible = (all_rows.risk_gt.notna()) & (all_rows.pipeline_ok == 1)
    print(f"Total frames: {len(all_rows)} | unlabeled: {all_rows.risk_gt.isna().sum()} | "
          f"pipeline != 1: {(all_rows.pipeline_ok != 1).sum()} | evaluated: {eligible.sum()}")
    if not eligible.any():
        raise SystemExit("No labeled, successfully processed frames to evaluate")
    evaluated = all_rows.loc[eligible].copy()
    if evaluated.rule_pred.isna().any():
        raise ValueError("Missing rule prediction for some eligible frames")
    evaluated["risk_gt"] = evaluated["risk_gt"].astype(int)
    evaluated["rule_pred"] = evaluated["rule_pred"].astype(int)
    per_video = []
    for (event_id, video_id), part in evaluated.groupby(["source_event_id", "video_id"], sort=True):
        per_video.append({"source_event_id": event_id, "video_id": video_id,
                          **metrics(part.risk_gt.to_numpy(), part.rule_pred.to_numpy())})
    result = pd.DataFrame(per_video)
    overall = pd.DataFrame([{"source_event_id": "ALL", "video_id": "ALL",
                             **metrics(evaluated.risk_gt.to_numpy(), evaluated.rule_pred.to_numpy())}])
    print("\nPER RECORDING:")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nALL EVALUATED FRAMES (pooled):")
    print(overall.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nConfusion matrix: rows=true [0,1], columns=predicted [0,1]")
    m = overall.iloc[0]
    print(f"[[{m.TN}, {m.FP}], [{m.FN}, {m.TP}]]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "rule_metrics_per_video.csv", index=False)
    overall.to_csv(args.output_dir / "rule_metrics_overall.csv", index=False)
    evaluated[["video_id", "source_event_id", "frame_idx", "risk_gt", "rule_pred"]].to_csv(
        args.output_dir / "rule_frame_predictions.csv", index=False)
    print(f"\nSaved CSV reports to: {args.output_dir}")


if __name__ == "__main__":
    main()
