#!/usr/bin/env python3
"""Phase II, B4: frame-level dataset = baseline 13 features + crosswalk-entry aggregates.

Inputs (read only):
  - per scene: data_output.ml_frames_csv (baseline frame dataset),
    data_output.frame_labels_csv (GT), data_output.csv (tracks, for checks),
  - outputs/features_v2/<source_event_id>/scenario_tracks_features_v2.csv (B3).

Output (new directory, refuses to overwrite):
  <output-dir>/frames_features_v2.csv   all scenes; baseline columns copied as
      text (byte-identical values), risk_gt joined from frame_labels_csv as in
      the ML notebook, plus source_file and the two new columns.
  qc_per_scene.csv, qc_by_gt.csv, qc_by_scene_gt.csv, qc_spot_checks.csv,
  manifest.json, qc_report.md

The file is deliberately NOT named scenario_ml_frames.csv, so that the
notebook's rglob over outputs/ does not pick it up twice.
No model is trained here.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from traffic_safety.motion_aggregation import (  # noqa: E402
    FRAME_FEATURE_COLUMNS,
    MIN_ENTRY_COLUMN,
    UNKNOWN_ENTRY_COLUMN,
    aggregate_crosswalk_entry,
)
from traffic_safety.motion_features import compute_motion_features  # noqa: E402
from compute_motion_features import git_state, markdown_table, sha256  # noqa: E402

KEY = ["video_id", "frame_idx"]
OUTPUT_NAME = "frames_features_v2.csv"
FEATURES_V2_DIR = Path("outputs/features_v2")
FEATURES_V2_NAME = "scenario_tracks_features_v2.csv"
BASELINE_FEATURES = [
    "num_pedestrians", "num_vehicles", "num_interactions",
    "num_pedestrians_crossing", "num_pedestrians_waiting",
    "num_vehicles_approaching", "num_vehicles_stopped",
    "min_ttc_s", "min_distance_crosswalk_m",
    "max_vehicle_speed_kmh", "mean_vehicle_speed_kmh",
    "num_ttc_available", "min_pedestrian_vehicle_distance_m",
]
PREFIX_CUT_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def read_text(path: Path) -> pd.DataFrame:
    """Read every value as the exact text stored in the file."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)


def independent_aggregate(features: pd.DataFrame, frames) -> pd.DataFrame:
    """Plain-loop re-implementation used only to cross-check the module."""
    known: dict[int, float] = {}
    unknown: dict[int, int] = {}
    for row in features.itertuples(index=False):
        if row.category != "pedestrian" or row.motion_stable_zone != "crosswalk":
            continue
        frame = int(row.frame_idx)
        value = row.time_since_crosswalk_entry_s
        if value is None or (isinstance(value, float) and np.isnan(value)):
            unknown[frame] = unknown.get(frame, 0) + 1
        else:
            known[frame] = min(known.get(frame, np.inf), float(value))
    return pd.DataFrame(
        {
            "frame_idx": list(frames),
            MIN_ENTRY_COLUMN: [known.get(int(f), np.nan) for f in frames],
            UNKNOWN_ENTRY_COLUMN: [unknown.get(int(f), 0) for f in frames],
        }
    )


def aggregate_from_tracks(tracks: pd.DataFrame, frames: list[int]) -> pd.DataFrame:
    """Raw tracks -> motion features -> frame aggregates (no CSV round trip)."""
    feats = compute_motion_features(tracks)
    table = tracks.assign(
        motion_stable_zone=feats["stable_zone"],
        time_since_crosswalk_entry_s=feats["time_since_crosswalk_entry_s"],
    )
    return aggregate_crosswalk_entry(table, frames).set_index("frame_idx")


def prefix_check(tracks: pd.DataFrame, frames: list[int]) -> tuple[int, int, pd.DataFrame]:
    """Aggregates for frames <= cut computed from truncated raw tracks must be
    exactly equal to those computed from the full recording (same code path,
    so exact comparison is valid)."""
    full = aggregate_from_tracks(tracks, frames)
    mismatches = 0
    cuts = sorted({int(np.quantile(frames, q)) for q in PREFIX_CUT_QUANTILES})
    for cut in cuts:
        prefix_frames = [f for f in frames if f <= cut]
        agg = aggregate_from_tracks(tracks[tracks["frame_idx"] <= cut], prefix_frames)
        ref = full.loc[prefix_frames]
        same_min = (agg[MIN_ENTRY_COLUMN].isna() & ref[MIN_ENTRY_COLUMN].isna()) | (
            agg[MIN_ENTRY_COLUMN] == ref[MIN_ENTRY_COLUMN]
        )
        same_unknown = agg[UNKNOWN_ENTRY_COLUMN] == ref[UNKNOWN_ENTRY_COLUMN]
        mismatches += int((~(same_min & same_unknown)).sum())
    return len(cuts), mismatches, full


FILE_TOLERANCE_S = 1e-9  # B3 CSV round trip changes floats by ~1e-16 s


def spot_checks(scene: str, features: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    """A few readable frames per scene: one per case, deterministic choice."""
    ped = features[(features["category"] == "pedestrian") & (features["motion_stable_zone"] == "crosswalk")]
    by_frame = agg.set_index("frame_idx")
    cases = {
        "nobody_on_crosswalk": by_frame.index[by_frame[MIN_ENTRY_COLUMN].isna() & (by_frame[UNKNOWN_ENTRY_COLUMN] == 0)],
        "only_unknown_entries": by_frame.index[by_frame[MIN_ENTRY_COLUMN].isna() & (by_frame[UNKNOWN_ENTRY_COLUMN] > 0)],
        "known_and_unknown": by_frame.index[by_frame[MIN_ENTRY_COLUMN].notna() & (by_frame[UNKNOWN_ENTRY_COLUMN] > 0)],
        "several_known": by_frame.index[
            by_frame.index.isin(
                ped[ped["time_since_crosswalk_entry_s"].notna()].groupby("frame_idx").size().loc[lambda s: s > 1].index
            )
        ],
    }
    rows = []
    for case, frames in cases.items():
        if len(frames) == 0:
            continue
        frame = int(frames[len(frames) // 2])
        here = ped[ped["frame_idx"] == frame].sort_values("track_id")
        rows.append(
            {
                "scene": scene,
                "case": case,
                "frame_idx": frame,
                "crosswalk_pedestrians": "; ".join(
                    f"id {int(r.track_id)}: "
                    + ("unknown" if pd.isna(r.time_since_crosswalk_entry_s) else f"{r.time_since_crosswalk_entry_s:.3f}s")
                    for r in here.itertuples()
                ) or "(none)",
                MIN_ENTRY_COLUMN: by_frame.loc[frame, MIN_ENTRY_COLUMN],
                UNKNOWN_ENTRY_COLUMN: int(by_frame.loc[frame, UNKNOWN_ENTRY_COLUMN]),
            }
        )
    return pd.DataFrame(rows)


def process_scene(config_path: Path):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out_cfg = config["data_output"]
    frames_path = Path(out_cfg["ml_frames_csv"])
    labels_path = Path(out_cfg["frame_labels_csv"])
    tracks_path = Path(out_cfg["csv"])

    hashes = {p: sha256(p) for p in (frames_path, labels_path, tracks_path)}

    base = read_text(frames_path)
    scene = base["source_event_id"].iloc[0]
    if base["source_event_id"].nunique() != 1 or base["video_id"].nunique() != 1:
        raise SystemExit(f"{frames_path}: expected one video / event")
    if base.duplicated(KEY).any():
        raise SystemExit(f"{frames_path}: duplicated keys")
    if (base["risk_gt"] != "").any():
        raise SystemExit(f"{frames_path}: risk_gt is unexpectedly filled; refusing to replace it")

    features_path = FEATURES_V2_DIR / scene / FEATURES_V2_NAME
    features = pd.read_csv(features_path, low_memory=False)
    if set(features["video_id"].astype(str)) != set(base["video_id"]):
        raise SystemExit(f"{features_path}: video_id differs from {frames_path}")

    b3_manifest = json.loads((FEATURES_V2_DIR / "manifest.json").read_text())
    b3_entry = next(e for e in b3_manifest["files"] if e["output_csv"] == str(features_path))
    b3_consistent = (
        b3_entry["source_sha256"] == hashes[tracks_path]
        and b3_entry["output_sha256"] == sha256(features_path)
    )

    frame_ids = [int(v) for v in base["frame_idx"]]
    agg = aggregate_crosswalk_entry(features, frame_ids)
    check = independent_aggregate(features, frame_ids)
    independent_ok = bool(
        np.array_equal(agg[UNKNOWN_ENTRY_COLUMN].to_numpy(), check[UNKNOWN_ENTRY_COLUMN].to_numpy())
        and np.allclose(agg[MIN_ENTRY_COLUMN], check[MIN_ENTRY_COLUMN], equal_nan=True, rtol=0, atol=0)
    )

    tracks = pd.read_csv(tracks_path, low_memory=False)
    cuts, prefix_mismatches, fresh = prefix_check(tracks, frame_ids)
    diff = (agg.set_index("frame_idx")[MIN_ENTRY_COLUMN] - fresh[MIN_ENTRY_COLUMN]).abs()
    file_vs_fresh_max_diff = float(diff.max()) if diff.notna().any() else 0.0
    file_matches_fresh = bool(
        agg.set_index("frame_idx")[MIN_ENTRY_COLUMN].isna().equals(fresh[MIN_ENTRY_COLUMN].isna())
        and file_vs_fresh_max_diff <= FILE_TOLERANCE_S
        and np.array_equal(agg[UNKNOWN_ENTRY_COLUMN].to_numpy(), fresh[UNKNOWN_ENTRY_COLUMN].to_numpy())
    )

    labels = read_text(labels_path)
    if labels.duplicated(KEY).any():
        raise SystemExit(f"{labels_path}: duplicated keys")
    if not set(labels["risk_gt"]) <= {"0", "1", ""}:
        raise SystemExit(f"{labels_path}: unexpected label values")

    # Same join as the ML notebook (cell 9): drop empty risk_gt, left join labels.
    merged = base.merge(labels[KEY + ["risk_gt"]].rename(columns={"risk_gt": "_gt"}), on=KEY, how="left", validate="one_to_one")
    merged["risk_gt"] = merged.pop("_gt").fillna("")
    merged["source_file"] = str(frames_path)
    new = agg.assign(frame_idx=agg["frame_idx"].astype(str))
    merged = merged.merge(new, on="frame_idx", how="left", validate="one_to_one")

    with_ped = merged[UNKNOWN_ENTRY_COLUMN].gt(0) | merged[MIN_ENTRY_COLUMN].notna()
    row = {
        "scene": scene,
        "video_id": base["video_id"].iloc[0],
        "frames_before": len(base),
        "frames_after": len(merged),
        "label_rows": len(labels),
        "labels_matched": int((merged["risk_gt"] != "").sum()),
        "gt1_frames": int((merged["risk_gt"] == "1").sum()),
        "b3_features_match_current_tracks": b3_consistent,
        "aggregation_matches_independent_loop": independent_ok,
        "b3_file_matches_fresh_computation": file_matches_fresh,
        "b3_file_vs_fresh_max_abs_diff_s": file_vs_fresh_max_diff,
        "prefix_cuts": cuts,
        "prefix_mismatches": prefix_mismatches,
        "frames_with_crosswalk_pedestrian": int(with_ped.sum()),
        "frames_min_entry_available": int(merged[MIN_ENTRY_COLUMN].notna().sum()),
        "frames_only_unknown_entry": int((merged[MIN_ENTRY_COLUMN].isna() & merged[UNKNOWN_ENTRY_COLUMN].gt(0)).sum()),
        "frames_min_entry_available_pct": round(100 * merged[MIN_ENTRY_COLUMN].notna().mean(), 1),
        "pct_of_crosswalk_frames_with_min": round(
            100 * merged[MIN_ENTRY_COLUMN].notna().sum() / max(int(with_ped.sum()), 1), 1
        ),
        "frames_num_pedestrians_crossing_gt0": int((pd.to_numeric(base["num_pedestrians_crossing"]) > 0).sum()),
        "frames_crossing_gt0_but_no_stable_crosswalk_ped": int(
            ((pd.to_numeric(merged["num_pedestrians_crossing"]) > 0) & ~with_ped).sum()
        ),
        "frames_crossing_eq0_but_stable_crosswalk_ped": int(
            ((pd.to_numeric(merged["num_pedestrians_crossing"]) == 0) & with_ped).sum()
        ),
    }
    sources = {
        "config": str(config_path),
        "ml_frames_csv": str(frames_path),
        "frame_labels_csv": str(labels_path),
        "tracks_csv": str(tracks_path),
        "features_v2_csv": str(features_path),
        "sha256": {str(p): h for p, h in hashes.items()} | {str(features_path): sha256(features_path)},
    }
    return row, base, labels, merged, spot_checks(scene, features, agg), sources, hashes


def describe(values: pd.Series) -> dict:
    values = values.dropna()
    if values.empty:
        return {"n": 0}
    q = values.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    return {
        "n": len(values), "mean": round(values.mean(), 3),
        "p10": round(q[0.1], 3), "p25": round(q[0.25], 3), "median": round(q[0.5], 3),
        "p75": round(q[0.75], 3), "p90": round(q[0.9], 3), "max": round(values.max(), 3),
    }


def by_gt(data: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, part in data.groupby(group_cols, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        unknown = part[UNKNOWN_ENTRY_COLUMN]
        with_ped = unknown.gt(0) | part[MIN_ENTRY_COLUMN].notna()
        row = dict(zip(group_cols, keys))
        row |= {
            "frames": len(part),
            "pct_with_crosswalk_pedestrian": round(100 * with_ped.mean(), 1),
            "pct_min_entry_available": round(100 * part[MIN_ENTRY_COLUMN].notna().mean(), 1),
            "pct_only_unknown_entry": round(100 * (part[MIN_ENTRY_COLUMN].isna() & unknown.gt(0)).mean(), 1),
            "unknown_0": int(unknown.eq(0).sum()),
            "unknown_1": int(unknown.eq(1).sum()),
            "unknown_2": int(unknown.eq(2).sum()),
            "unknown_3plus": int(unknown.ge(3).sum()),
        }
        row |= {f"min_entry_{k}": v for k, v in describe(part[MIN_ENTRY_COLUMN]).items()}
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(path: Path, per_scene, gt_table, scene_gt, spots, manifest) -> None:
    checks = per_scene[[
        "scene", "frames_before", "frames_after", "labels_matched", "gt1_frames",
        "b3_features_match_current_tracks", "aggregation_matches_independent_loop",
        "b3_file_matches_fresh_computation", "b3_file_vs_fresh_max_abs_diff_s",
        "prefix_cuts", "prefix_mismatches",
    ]]
    avail = per_scene[[
        "scene", "frames_with_crosswalk_pedestrian", "frames_min_entry_available",
        "frames_only_unknown_entry", "frames_min_entry_available_pct", "pct_of_crosswalk_frames_with_min",
        "frames_crossing_gt0_but_no_stable_crosswalk_ped", "frames_crossing_eq0_but_stable_crosswalk_ped",
    ]]
    text = f"""# Frame dataset features_v2 — QC report (Phase II, B4)

Created: {manifest['created_utc']}; code commit {manifest['git']['commit']} (dirty: {manifest['git']['code_dirty']}).

Baseline: 13 frame features from `scenario_ml_frames.csv` (values copied as text, unchanged).
Labels: `risk_gt` joined from `frame_labels_csv` exactly as in the ML notebook.
New columns: `{MIN_ENTRY_COLUMN}`, `{UNKNOWN_ENTRY_COLUMN}`.

Encoding of the three cases (no extra column needed):
- nobody on crosswalk: min empty, unknown = 0
- only unknown entries: min empty, unknown > 0
- at least one known entry: min set

**Distributions by GT are diagnostics only. No thresholds were chosen from them.**

## Integrity checks

All global checks: {manifest['checks']}

{markdown_table(checks)}

## Availability per scene

{markdown_table(avail)}

## Diagnostics by GT (all scenes)

{markdown_table(gt_table)}

## Diagnostics by scene and GT

{markdown_table(scene_gt)}

## Spot checks (pedestrians with stable zone = crosswalk)

{markdown_table(spots)}
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs-dir", default="configs")
    parser.add_argument("--output-dir", default="outputs/ml_experiment/features_v2")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise SystemExit(f"{output_dir} already exists; choose another --output-dir (nothing is overwritten)")

    configs = sorted(Path(args.configs_dir).glob("*.yaml"))
    results = [process_scene(path) for path in configs]
    per_scene = pd.DataFrame([r[0] for r in results])

    base_all = pd.concat([r[1] for r in results], ignore_index=True)
    labels_all = pd.concat([r[2] for r in results], ignore_index=True)
    data = pd.concat([r[3] for r in results], ignore_index=True)
    spots = pd.concat([r[4] for r in results], ignore_index=True)

    # Notebook order: sorted by video_id, frame_idx (numeric).
    data = data.assign(_f=data["frame_idx"].astype(int)).sort_values(["video_id", "_f"]).drop(columns="_f").reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=False)
    out_path = output_dir / OUTPUT_NAME
    to_write = data.copy()
    to_write[MIN_ENTRY_COLUMN] = to_write[MIN_ENTRY_COLUMN].map(lambda v: "" if pd.isna(v) else f"{v:.6f}")
    to_write[UNKNOWN_ENTRY_COLUMN] = to_write[UNKNOWN_ENTRY_COLUMN].astype(int).astype(str)
    to_write.to_csv(out_path, index=False)

    # --- Global QC on the written file -----------------------------------
    reread = read_text(out_path)
    base_cols = [c for c in base_all.columns if c != "risk_gt"]
    left = reread.set_index(KEY)
    base_idx = base_all.set_index(KEY)
    label_idx = labels_all.set_index(KEY)
    checks = {
        "rows_equal": len(reread) == len(base_all),
        "no_duplicate_keys": not reread.duplicated(KEY).any(),
        "no_lost_or_added_frames": set(left.index) == set(base_idx.index),
        "baseline_columns_identical_text": bool(
            left.loc[base_idx.index, [c for c in base_cols if c not in KEY]].equals(
                base_idx[[c for c in base_cols if c not in KEY]]
            )
        ),
        "labels_identical_to_frame_labels_csv": bool(
            left.loc[label_idx.index, "risk_gt"].equals(label_idx["risk_gt"])
        ),
        "every_frame_labelled": bool((reread["risk_gt"] != "").all()),
        "unknown_count_never_missing": bool((reread[UNKNOWN_ENTRY_COLUMN] != "").all()),
        "column_order_baseline_first": list(reread.columns[: len(base_all.columns)]) == list(base_all.columns),
    }
    for flag in ("b3_features_match_current_tracks", "aggregation_matches_independent_loop", "b3_file_matches_fresh_computation"):
        checks[f"all_{flag}"] = bool(per_scene[flag].all())
    checks["prefix_mismatches_total"] = int(per_scene["prefix_mismatches"].sum())
    checks["frames_equal_per_scene"] = bool((per_scene["frames_before"] == per_scene["frames_after"]).all())

    source_hashes = {}
    for r in results:
        source_hashes |= r[6]
    checks["sources_unchanged"] = all(sha256(p) == h for p, h in source_hashes.items())

    numeric = data.assign(risk_gt=data["risk_gt"].astype(int))
    gt_table = by_gt(numeric, ["risk_gt"])
    scene_gt = by_gt(numeric.rename(columns={"source_event_id": "scene"}), ["scene", "risk_gt"])

    per_scene.to_csv(output_dir / "qc_per_scene.csv", index=False)
    gt_table.to_csv(output_dir / "qc_by_gt.csv", index=False)
    scene_gt.to_csv(output_dir / "qc_by_scene_gt.csv", index=False)
    spots.to_csv(output_dir / "qc_spot_checks.csv", index=False)

    manifest = {
        "version": "frames_features_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "baseline_features": BASELINE_FEATURES,
        "new_features": list(FRAME_FEATURE_COLUMNS),
        "label_column": "risk_gt",
        "exclude_from_X": ["video_id", "source_event_id", "frame_idx", "timestamp_s", "pipeline_ok", "rule_pred", "risk_gt", "source_file"],
        "aggregation": "pedestrians with motion_stable_zone == crosswalk in the frame; min of known time_since_crosswalk_entry_s; count of unknown",
        "checks": checks,
        "output_csv": str(out_path),
        "output_sha256": sha256(out_path),
        "sources": [r[5] for r in results],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    write_report(output_dir / "qc_report.md", per_scene, gt_table, scene_gt, spots, manifest)

    print(json.dumps(checks, indent=2))
    failed = [k for k, v in checks.items() if (v is False) or (k == "prefix_mismatches_total" and v != 0)]
    if failed:
        raise SystemExit(f"QC FAILED: {failed}")
    print(f"QC passed. Saved {len(reread)} frames to {out_path}")


if __name__ == "__main__":
    main()
