#!/usr/bin/env python3
"""B3: offline computation of temporal motion features (features_v2).

Run from the repository root:
    python scripts/compute_motion_features.py
    python scripts/compute_motion_features.py --configs-dir configs --output-dir outputs/features_v2

For every scenario YAML, reads the saved data_output.csv (scenario_tracks.csv),
computes the features with traffic_safety.motion_features.compute_motion_features
and writes:

    <output-dir>/<scene>/scenario_tracks_features_v2.csv
        video_id, source_event_id + all original track columns (unchanged)
        + motion_stable_zone (auxiliary, for auditing)
        + time_in_current_zone_s, time_since_crosswalk_entry_s,
          came_from_waiting, came_from_crosswalk   (missing values left empty)

    <output-dir>/qc_per_scene.csv, qc_feature_distributions.csv,
    qc_crosswalk_entries.csv, manifest.json, qc_report.md

No detection / tracking is re-run. Input files are only read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from traffic_safety.motion_features import (  # noqa: E402
    CROSSWALK_ZONE,
    FEATURE_COLUMNS,
    compute_motion_features,
)
from traffic_safety.state import PEDESTRIAN_WAITING_ZONES  # noqa: E402

OUTPUT_NAME = "scenario_tracks_features_v2.csv"
AUX_COLUMN = "motion_stable_zone"
TIME_FEATURES = ["time_in_current_zone_s", "time_since_crosswalk_entry_s"]
BOOL_FEATURES = ["came_from_waiting", "came_from_crosswalk"]
NOT_RECOMMENDED = {"came_from_waiting"}
PREFIX_CUT_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> dict:
    def run(*args):
        try:
            return subprocess.run(
                ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:
            return None

    status = run("status", "--porcelain", "--", "traffic_safety", "scripts")
    return {"commit": run("rev-parse", "HEAD"), "code_dirty": bool(status)}


def read_tracks(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def prefix_check(tracks: pd.DataFrame, full: pd.DataFrame) -> tuple[int, int]:
    """Recompute on prefixes ending at several frames; return (cuts, mismatching rows)."""
    frames = np.sort(tracks["frame_idx"].unique())
    cuts = sorted({int(frames[int(q * (len(frames) - 1))]) for q in PREFIX_CUT_QUANTILES})
    mismatches = 0
    for cut in cuts:
        prefix = tracks[tracks["frame_idx"] <= cut]
        partial = compute_motion_features(prefix)
        expected = full.loc[prefix.index]
        for column in FEATURE_COLUMNS + ("stable_zone",):
            left, right = partial[column], expected[column]
            same = (left == right) | (left.isna() & right.isna())
            mismatches += int((~same.fillna(False)).sum())
    return len(cuts), mismatches


def crosswalk_entries(scene: str, extended: pd.DataFrame) -> pd.DataFrame:
    """One row per observed (confirmed) crosswalk stay with a known start."""
    on = extended[
        (extended[AUX_COLUMN] == CROSSWALK_ZONE)
        & extended["time_since_crosswalk_entry_s"].notna()
    ].copy()
    if on.empty:
        return pd.DataFrame()
    on["entry_time_s"] = (on["timestamp_s"] - on["time_since_crosswalk_entry_s"]).round(4)
    entries = (
        on.groupby(["category", "track_id", "entry_time_s"], sort=True)
        .agg(
            first_confirmed_frame=("frame_idx", "min"),
            last_frame=("frame_idx", "max"),
            max_time_on_crosswalk_s=("time_since_crosswalk_entry_s", "max"),
            came_from_waiting=("came_from_waiting", "first"),
            came_from_crosswalk=("came_from_crosswalk", "first"),
        )
        .reset_index()
    )
    entries.insert(0, "scene", scene)
    return entries


def process_scene(config_path: Path, output_dir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    meta = config.get("study_metadata") or {}
    tracks_path = Path(config["data_output"]["csv"])
    scene = meta.get("source_event_id") or config_path.stem

    source_hash = sha256(tracks_path)
    tracks = read_tracks(tracks_path)
    features = compute_motion_features(tracks)

    extended = tracks.copy()
    extended.insert(0, "source_event_id", scene)
    extended.insert(0, "video_id", meta.get("video_id", ""))
    extended[AUX_COLUMN] = features["stable_zone"]
    for column in FEATURE_COLUMNS:
        extended[column] = features[column]

    scene_dir = output_dir / scene
    scene_dir.mkdir(parents=True, exist_ok=False)
    out_path = scene_dir / OUTPUT_NAME
    extended.to_csv(out_path, index=False, na_rep="")

    # --- QC: written file re-read; original columns unchanged ----------
    reread = read_tracks(out_path)
    original_unchanged = True
    try:
        pd.testing.assert_frame_equal(
            reread[list(tracks.columns)], tracks, check_dtype=False, check_exact=True
        )
    except AssertionError:
        original_unchanged = False
    source_unchanged = sha256(tracks_path) == source_hash

    cuts, prefix_mismatches = prefix_check(tracks, features)

    zones = set((config.get("zones") or {}).keys())
    ped = extended[extended["category"] == "pedestrian"]
    entries = crosswalk_entries(scene, extended)
    ped_entries = entries[entries["category"] == "pedestrian"] if not entries.empty else entries

    row = {
        "scene": scene,
        "video_id": meta.get("video_id", ""),
        "has_waiting_zones": bool(zones & PEDESTRIAN_WAITING_ZONES),
        "waiting_zones": ";".join(sorted(zones & PEDESTRIAN_WAITING_ZONES)),
        "rows_before": len(tracks),
        "rows_after": len(reread),
        "original_columns_unchanged": original_unchanged,
        "source_file_unchanged": source_unchanged,
        "prefix_cuts": cuts,
        "prefix_mismatches": prefix_mismatches,
        "pedestrian_rows": len(ped),
        "pedestrian_rows_on_crosswalk": int((ped[AUX_COLUMN] == CROSSWALK_ZONE).sum()),
        "pedestrian_crosswalk_entries": len(ped_entries),
        "pedestrian_entries_came_from_waiting": int(
            ped_entries["came_from_waiting"].fillna(False).astype(bool).sum()
        ) if len(ped_entries) else 0,
    }
    for column in FEATURE_COLUMNS:
        row[f"ped_{column}_available"] = int(ped[column].notna().sum())
        row[f"ped_{column}_missing"] = int(ped[column].isna().sum())
        row[f"ped_{column}_available_pct"] = round(100 * ped[column].notna().mean(), 1) if len(ped) else np.nan
    for column in BOOL_FEATURES:
        row[f"ped_{column}_true"] = int(ped[column].fillna(False).astype(bool).sum())

    manifest_entry = {
        "config": str(config_path),
        "source_tracks_csv": str(tracks_path),
        "source_sha256": source_hash,
        "output_csv": str(out_path),
        "output_sha256": sha256(out_path),
    }
    return row, extended, entries, manifest_entry


def distributions(all_rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for category, part in all_rows.groupby("category"):
        for column in TIME_FEATURES:
            values = part[column].dropna()
            desc = values.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
            records.append({
                "category": category, "feature": column,
                "rows": len(part), "available": len(values), "missing": int(part[column].isna().sum()),
                **{k: round(float(v), 4) for k, v in desc.drop("count").items()},
            })
        for column in BOOL_FEATURES:
            values = part[column]
            records.append({
                "category": category, "feature": column,
                "rows": len(part), "available": int(values.notna().sum()), "missing": int(values.isna().sum()),
                "true": int(values.fillna(False).astype(bool).sum()),
                "false": int((values == False).sum()),  # noqa: E712
            })
    return pd.DataFrame(records)


def markdown_table(frame: pd.DataFrame) -> str:
    """Minimal Markdown table (no optional dependencies)."""
    def cell(value):
        return "" if pd.isna(value) else str(value)

    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def write_report(path: Path, per_scene: pd.DataFrame, dist: pd.DataFrame, entries: pd.DataFrame, manifest: dict) -> None:
    ped_dist = dist[dist.category == "pedestrian"]
    lines = [
        "# features_v2: raport jakości (B3)",
        "",
        f"Wygenerowano: {manifest['created_utc']} | commit kodu: `{manifest['git']['commit']}`"
        f" | niezacommitowane zmiany w kodzie: {manifest['git']['code_dirty']}",
        "",
        "Cechy liczone offline z zapisanych `scenario_tracks.csv` (bez YOLO/ByteTrack), "
        "przyczynowo, z filtrem migotania 3 obserwacji. Braki pozostawione puste (nie 0).",
        "",
        "## Ważne: cechy niezalecane do ML",
        "",
        "- **`came_from_waiting` — NIE rekomendowana do pierwszego eksperymentu ML.** "
        f"Strefy oczekiwania ma tylko {int(per_scene.has_waiting_zones.sum())} z {len(per_scene)} scen; "
        "w pozostałych cecha jest strukturalnie False i kodowałaby konfigurację sceny, a nie zachowanie pieszego.",
        "",
        "## Kontrole",
        "",
        "| scena | strefy oczekiwania | wiersze przed | wiersze po | kolumny oryginalne bez zmian | plik źródłowy bez zmian | test prefiksu (cięcia / rozbieżności) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in per_scene.itertuples():
        lines.append(
            f"| {r.scene} | {'tak' if r.has_waiting_zones else '**nie**'} | {r.rows_before} | {r.rows_after} | "
            f"{r.original_columns_unchanged} | {r.source_file_unchanged} | {r.prefix_cuts} / {r.prefix_mismatches} |"
        )
    lines += [
        "",
        "## Dostępność cech u pieszych (% wierszy z wartością)",
        "",
        "| scena | wiersze pieszych | na przejściu | wejścia na przejście | w tym ze strefy oczekiwania | time_in_current_zone_s | time_since_crosswalk_entry_s | came_from_waiting | came_from_crosswalk |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in per_scene.itertuples():
        lines.append(
            f"| {r.scene} | {r.pedestrian_rows} | {r.pedestrian_rows_on_crosswalk} | {r.pedestrian_crosswalk_entries} | "
            f"{r.pedestrian_entries_came_from_waiting} | {r.ped_time_in_current_zone_s_available_pct} | "
            f"{r.ped_time_since_crosswalk_entry_s_available_pct} | {r.ped_came_from_waiting_available_pct} | "
            f"{r.ped_came_from_crosswalk_available_pct} |"
        )
    lines += ["", "## Rozkłady (piesi)", "", markdown_table(ped_dist.dropna(axis=1, how="all"))]
    lines += ["", f"Zaobserwowane wejścia na przejście (wszystkie kategorie): {len(entries)}; "
              f"piesi: {int((entries.category == 'pedestrian').sum()) if len(entries) else 0}.", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs-dir", type=Path, default=Path("configs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/features_v2"))
    args = parser.parse_args()

    if args.output_dir.exists():
        raise SystemExit(f"Output directory exists: {args.output_dir} (choose another --output-dir)")

    configs = sorted(set(args.configs_dir.rglob("*.yaml")) | set(args.configs_dir.rglob("*.yml")))
    if not configs:
        raise SystemExit(f"No YAML files in {args.configs_dir}")

    args.output_dir.mkdir(parents=True)
    rows, all_rows, all_entries, manifest_files = [], [], [], []
    for config in configs:
        row, extended, entries, manifest_entry = process_scene(config, args.output_dir)
        print(f"OK {row['scene']}: {row['rows_after']} rows, prefix mismatches {row['prefix_mismatches']}")
        rows.append(row)
        all_rows.append(extended)
        if len(entries):
            all_entries.append(entries)
        manifest_files.append(manifest_entry)

    per_scene = pd.DataFrame(rows)
    combined = pd.concat(all_rows, ignore_index=True)
    dist = distributions(combined)
    entries = pd.concat(all_entries, ignore_index=True) if all_entries else pd.DataFrame()

    manifest = {
        "version": "features_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "features": list(FEATURE_COLUMNS),
        "auxiliary_columns": [AUX_COLUMN],
        "confirm_observations": 3,
        "not_recommended_for_ml": sorted(NOT_RECOMMENDED),
        "files": manifest_files,
    }

    per_scene.to_csv(args.output_dir / "qc_per_scene.csv", index=False)
    dist.to_csv(args.output_dir / "qc_feature_distributions.csv", index=False)
    entries.to_csv(args.output_dir / "qc_crosswalk_entries.csv", index=False)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(args.output_dir / "qc_report.md", per_scene, dist, entries, manifest)

    failed = per_scene[
        (per_scene.rows_before != per_scene.rows_after)
        | ~per_scene.original_columns_unchanged
        | ~per_scene.source_file_unchanged
        | (per_scene.prefix_mismatches != 0)
    ]
    print(f"\nSaved to {args.output_dir}")
    if len(failed):
        raise SystemExit(f"QC FAILED for: {failed.scene.tolist()}")
    print("QC passed for all scenes.")


if __name__ == "__main__":
    main()
