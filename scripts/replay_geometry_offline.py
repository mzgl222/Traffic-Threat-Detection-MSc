#!/usr/bin/env python3
"""Offline replay of the geometric part of the pipeline from saved tracks CSV.

Chain replayed per frame (same order and modules as traffic_safety/pipeline.py):
    image_x, image_y (saved, 0.01 px)
      -> HomographyTransformer            -> world_x_m, world_y_m
      -> ZoneManager (image space)         -> zone
      -> SpeedEstimator (vehicles)         -> speed_kmh
      -> distance_to_line + RiskMetricsEstimator (vehicles in vehicle_approach)
                                           -> distance_to_crosswalk_m, ttc_crosswalk_s
      -> StateEstimator                    -> state
      -> SafetyRuleEngine (legacy timeline), BinaryRuleEngine (V1)
      -> interaction pairs (world coords)  -> build_frame_dataset.py -> ML frame features
Detections, track IDs, bounding boxes and timestamps are taken from the saved CSV
(no YOLO / ByteTrack). Nothing in the repository's outputs is modified.

Modes:
--mode recompute
    Recomputes the scenes assigned to homography B with the homography from their
    config, the historical StateEstimator (4d182cc) and zones taken from the saved
    CSV. Every scene first passes a gate: the same code path with homography A must
    reproduce the original rule decisions exactly. Output: outputs/geometry_v2/<scene>/
    with *_geomv2.csv names (never scenario_*.csv, to stay invisible to rglob-based
    consumers such as the ML notebook).
--mode reproduce
    Uses the homography that produced the saved files (A) and the StateEstimator
    from git commit 4d182cc (the version used when the tracks were generated),
    and compares every replayed quantity with the saved outputs.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from traffic_safety.binary_rules import BinaryRuleEngine  # noqa: E402
from traffic_safety.homography import HomographyTransformer  # noqa: E402
from traffic_safety.risk_metrics import RiskMetricsEstimator, distance_to_line  # noqa: E402
from traffic_safety.safety_rules import SafetyRuleEngine  # noqa: E402
from traffic_safety.speed import SpeedEstimator  # noqa: E402
from traffic_safety.zones import ZoneManager  # noqa: E402
from evaluate_binary_rules import predict_frames  # noqa: E402

HOMOGRAPHY_A = {
    "image_points": [[866, 52], [1074, 55], [1031, 429], [165, 411]],
    "world_points": [[0.0, 0.0], [10.0, 0.0], [10.0, 30.0], [0.0, 30.0]],
}
LEGACY_STATE_COMMIT = "4d182cc"
FPS = 30.0
# Parameters exactly as in run.py (unchanged since the tracks were generated).
STATE_PARAMS = dict(stopped_speed_threshold=2.0, slowing_threshold=-2.0,
                    state_confirm_frames=3, stopped_confirm_frames=5, speed_history_length=6)
RISK_PARAMS = dict(history_window_s=0.7, min_samples=5, min_closing_speed_m_s=0.5)


def load_state_estimator_class(commit: str):
    source = subprocess.run(["git", "show", f"{commit}:traffic_safety/state.py"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout
    spec = importlib.util.spec_from_loader(f"legacy_state_{commit}", loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, f"git:{commit}:traffic_safety/state.py", "exec"), module.__dict__)
    return module.StateEstimator


def state_value(state):
    return None if state is None else getattr(state, "value", state)


def replay(tracks: pd.DataFrame, config: dict, homography: dict, state_cls, zone_source: str = "recompute") -> pd.DataFrame:
    """Replay one recording. Returns replayed columns (unrounded).

    zone_source="recompute": zone from ZoneManager on the saved (0.01 px rounded) image point;
    zone_source="saved": zone column of the saved CSV (computed by the original pipeline on
    unrounded coordinates; zones do not depend on the homography)."""
    H = HomographyTransformer(homography["image_points"], homography["world_points"])
    zones = ZoneManager(config["zones"])
    speed = SpeedEstimator(history_length=config["tracking"]["history_length"],
                           speed_history_length=config["tracking"]["speed_history_length"])
    risk = RiskMetricsEstimator(**RISK_PARAMS)
    states = state_cls(**STATE_PARAMS)
    ref = config["crosswalk"]["reference_line_image"]
    line = (H.image_to_world(*ref[0]), H.image_to_world(*ref[1]))

    out = []
    # Saved row order within a frame = processing order of the pipeline.
    for row in tracks.itertuples(index=False):
        timestamp = row.frame_idx / FPS
        wx, wy = H.image_to_world(row.image_x, row.image_y)
        if zone_source == "saved":
            zone = None if pd.isna(row.zone) or row.zone == "" else row.zone
        else:
            zone = zones.get_zone(row.image_x, row.image_y)
        spd = dist = tta = None
        if row.category == "vehicle":
            spd = speed.update(track_id=row.track_id, timestamp=timestamp, world_x=wx, world_y=wy)
            if zone == "vehicle_approach":
                dist = distance_to_line(wx, wy, *line)
                tta = risk.update_ttc(track_id=row.track_id, timestamp=timestamp, distance_to_crosswalk_m=dist)
        obj = SimpleNamespace(track_id=row.track_id, category=row.category, zone=zone, speed_kmh=spd)
        st = state_value(states.update(obj))
        out.append((wx, wy, zone, spd, dist, tta, st))
    rep = pd.DataFrame(out, columns=["world_x_m", "world_y_m", "zone", "speed_kmh",
                                     "distance_to_crosswalk_m", "ttc_crosswalk_s", "state"], index=tracks.index)
    return rep


def legacy_timeline(tracks: pd.DataFrame, config: dict) -> pd.DataFrame:
    engine = SafetyRuleEngine(**config["risk"])
    rows = []
    for frame_idx, group in tracks.groupby("frame_idx", sort=True):
        objects = [SimpleNamespace(track_id=int(r.track_id), category=r.category,
                                   state=None if pd.isna(r.state) or r.state == "" else r.state,
                                   ttc_crosswalk_s=None if pd.isna(r.ttc_crosswalk_s) else r.ttc_crosswalk_s,
                                   distance_to_crosswalk_m=None if pd.isna(r.distance_to_crosswalk_m) else r.distance_to_crosswalk_m)
                   for r in group.itertuples(index=False)]
        events = engine.evaluate(objects)
        level = "danger" if any(e.level.value == "danger" for e in events) else (
            "warning" if any(e.level.value == "warning" for e in events) else "ok")
        rows.append({"frame_idx": int(frame_idx), "risk_level": level})
    return pd.DataFrame(rows)


def rounded_like_writer(tracks: pd.DataFrame, rep: pd.DataFrame) -> pd.DataFrame:
    """Saved-tracks schema with replayed values rounded exactly as CsvResultWriter does."""
    out = tracks.copy()
    out["world_x_m"] = rep["world_x_m"].round(3)
    out["world_y_m"] = rep["world_y_m"].round(3)
    out["zone"] = rep["zone"]
    out["state"] = rep["state"]
    out["speed_kmh"] = pd.to_numeric(rep["speed_kmh"]).round(2)
    out["distance_to_crosswalk_m"] = pd.to_numeric(rep["distance_to_crosswalk_m"]).round(3)
    out["ttc_crosswalk_s"] = pd.to_numeric(rep["ttc_crosswalk_s"]).round(3)
    return out


def interactions_from_tracks(tracks: pd.DataFrame, video_id: str, event_id: str) -> pd.DataFrame:
    ped = tracks[tracks.category == "pedestrian"][["frame_idx", "track_id", "world_x_m", "world_y_m"]]
    veh = tracks[tracks.category == "vehicle"][["frame_idx", "track_id", "world_x_m", "world_y_m"]]
    pairs = ped.merge(veh, on="frame_idx", suffixes=("_p", "_v"))
    return pd.DataFrame({
        "video_id": video_id, "source_event_id": event_id, "frame_idx": pairs.frame_idx,
        "pedestrian_track_id": pairs.track_id_p, "vehicle_track_id": pairs.track_id_v,
        "pedestrian_world_x_m": pairs.world_x_m_p, "pedestrian_world_y_m": pairs.world_y_m_p,
        "vehicle_world_x_m": pairs.world_x_m_v, "vehicle_world_y_m": pairs.world_y_m_v,
    })


def compare_numeric(saved: pd.Series, new: pd.Series, tol: float) -> dict:
    saved = pd.to_numeric(saved, errors="coerce")
    new = pd.to_numeric(new, errors="coerce")
    both = saved.notna() & new.notna()
    diff = (saved - new).abs()[both]
    return {
        "rows": int(len(saved)),
        "missing_pattern_mismatch": int((saved.isna() != new.isna()).sum()),
        "max_abs_diff": float(diff.max()) if len(diff) else 0.0,
        "p99_abs_diff": float(diff.quantile(0.99)) if len(diff) else 0.0,
        f"rows_over_tol_{tol}": int((diff > tol).sum()),
    }


def text_equal(saved: pd.Series, new: pd.Series) -> dict:
    a = saved.fillna("").astype(str)
    b = new.fillna("").astype(str)
    mism = a != b
    return {"rows": int(len(a)), "mismatches": int(mism.sum()),
            "examples": [f"{x}->{y}" for x, y in list(zip(a[mism], b[mism]))[:5]]}


def reproduce_scene(config_path: Path, out_dir: Path, state_cls) -> dict:
    config = yaml.safe_load(config_path.read_text())
    paths = config["data_output"]
    tracks = pd.read_csv(paths["csv"], low_memory=False)
    frames = pd.read_csv(paths["interaction_frames_csv"])
    video_id = config["study_metadata"]["video_id"]
    event_id = config["study_metadata"]["source_event_id"]
    scene = event_id

    # timestamps: saved = round(frame_idx / fps, 4)
    ts_ok = bool(np.allclose(tracks.timestamp_s, (tracks.frame_idx / FPS).round(4), atol=1e-9))
    rep = replay(tracks, config, HOMOGRAPHY_A, state_cls)
    new = rounded_like_writer(tracks, rep)
    result = {"scene": scene, "timestamps_equal_frame_over_30fps": ts_ok}

    # world-coordinate tolerance: 0.5 mm writer rounding + image rounding (0.005 px) * local scale
    H = HomographyTransformer(HOMOGRAPHY_A["image_points"], HOMOGRAPHY_A["world_points"])
    def local_scale(x, y):
        p = np.array(H.image_to_world(x, y))
        return np.linalg.norm(np.array(H.image_to_world(x + 1, y)) - p) + np.linalg.norm(np.array(H.image_to_world(x, y + 1)) - p)
    scale = np.array([local_scale(x, y) for x, y in zip(tracks.image_x, tracks.image_y)])
    werr = np.hypot(rep.world_x_m - tracks.world_x_m, rep.world_y_m - tracks.world_y_m)
    bound = 0.0008 + 0.005 * scale
    result["world"] = {"max_abs_diff_m": float(werr.max()), "rows_over_rounding_bound": int((werr > bound).sum())}
    result["zone"] = text_equal(tracks.zone, new.zone)
    veh = tracks.category == "vehicle"
    result["speed_kmh"] = compare_numeric(tracks.speed_kmh[veh], new.speed_kmh[veh], 0.15)
    result["distance_to_crosswalk_m"] = compare_numeric(tracks.distance_to_crosswalk_m[veh], new.distance_to_crosswalk_m[veh], 0.01)
    result["ttc_crosswalk_s"] = compare_numeric(tracks.ttc_crosswalk_s[veh], new.ttc_crosswalk_s[veh], 0.01)
    result["state"] = text_equal(tracks.state, new.state)
    st = tracks.state.fillna("").astype(str) != new.state.fillna("").astype(str)
    if st.any():
        bad = tracks[st]
        result["state"]["by_category"] = bad.category.value_counts().to_dict()
        # is the mismatch explained by speed rounding near the 2.0 km/h threshold?
        result["state"]["first_frames"] = bad.frame_idx.head(5).tolist()

    # Binary V1 on saved vs replayed columns
    eng = BinaryRuleEngine(arrival_time_s=4.0, moving_speed_kmh=2.0)
    b_saved = predict_frames(tracks, eng).set_index("frame_idx")
    b_new = predict_frames(new.assign(zone=new.zone.where(new.zone.notna(), np.nan)), eng).set_index("frame_idx")
    j = b_saved.join(b_new, lsuffix="_saved", rsuffix="_new", how="outer").fillna(0)
    result["binary_v1"] = {c: int((j[f"{c}_saved"] != j[f"{c}_new"]).sum()) for c in ("new_pred", "r1", "r2", "n_r1_pairs", "n_r2_pairs")}
    result["binary_v1"]["frames_danger_saved"] = int(j.new_pred_saved.sum())
    # compare also with the stored Phase I prediction file
    stored = pd.read_csv(ROOT / "outputs/ml_experiment/binary_rule_evaluation/binary_frame_predictions.csv")
    stored = stored[stored.source_event_id == event_id].set_index("frame_idx")
    jj = stored[["new_pred"]].join(b_new[["new_pred"]].rename(columns={"new_pred": "rep"}), how="left").fillna(0)
    result["binary_v1"]["vs_stored_phase1_predictions_mismatch"] = int((jj.new_pred != jj.rep).sum())

    # Legacy timeline (SafetyRuleEngine) vs saved scenario_risk_timeline.csv
    saved_tl = pd.read_csv(paths["risk_timeline_csv"]).set_index("frame_idx")
    new_tl = legacy_timeline(new, config).set_index("frame_idx").reindex(saved_tl.index).fillna("ok")
    result["legacy_timeline"] = text_equal(saved_tl.risk_level, new_tl.risk_level)

    # Interactions + ML frame features via the unchanged build_frame_dataset.py
    scene_dir = out_dir / scene
    scene_dir.mkdir(parents=True, exist_ok=False)
    new.to_csv(scene_dir / "replayed_tracks_A.csv", index=False)
    inter = interactions_from_tracks(new, video_id, event_id)
    saved_inter = pd.read_csv(paths["interactions_csv"], low_memory=False)
    result["interaction_rows"] = {"saved": len(saved_inter), "replayed": len(inter)}
    inter.to_csv(scene_dir / "replayed_interactions_A.csv", index=False)
    pd.DataFrame({"frame_idx": new_tl.index, "risk_level": new_tl.risk_level.values}).to_csv(scene_dir / "replayed_risk_timeline_A.csv", index=False)
    subprocess.run([sys.executable, str(ROOT / "scripts/build_frame_dataset.py"), "--config", str(config_path),
                    "--tracks", str(scene_dir / "replayed_tracks_A.csv"),
                    "--interactions", str(scene_dir / "replayed_interactions_A.csv"),
                    "--frames", paths["interaction_frames_csv"],
                    "--risk-timeline", str(scene_dir / "replayed_risk_timeline_A.csv"),
                    "--output", str(scene_dir / "replayed_ml_frames_A.csv")],
                   cwd=ROOT, check=True, capture_output=True, text=True)
    ml_saved = pd.read_csv(paths["ml_frames_csv"]).set_index("frame_idx")
    ml_new = pd.read_csv(scene_dir / "replayed_ml_frames_A.csv").set_index("frame_idx")
    ml = {}
    for col in ml_saved.columns:
        if col in ("video_id", "source_event_id"):
            continue
        if pd.api.types.is_numeric_dtype(ml_saved[col]) or col == "risk_gt":
            tol = {"max_vehicle_speed_kmh": 0.15, "mean_vehicle_speed_kmh": 0.15, "min_ttc_s": 0.01,
                   "min_distance_crosswalk_m": 0.01, "min_pedestrian_vehicle_distance_m": 0.01}.get(col, 1e-6)
            ml[col] = compare_numeric(ml_saved[col], ml_new[col].reindex(ml_saved.index), tol)
    result["ml_frames"] = ml
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["reproduce", "recompute"], required=True)
    ap.add_argument("--configs-dir", default="configs")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    if args.mode == "recompute":
        return recompute(args)
    out = Path(args.output_dir)
    if out.exists():
        raise SystemExit(f"{out} exists; choose a new --output-dir")
    out.mkdir(parents=True)
    state_cls = load_state_estimator_class(LEGACY_STATE_COMMIT)
    results = [reproduce_scene(p, out, state_cls) for p in sorted(Path(args.configs_dir).glob("*.yaml"))]
    (out / "reproduction_results.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=1, default=str))



# ======================================================================
# --mode recompute
# ======================================================================
import hashlib  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

from traffic_safety.csv_writer import CsvResultWriter  # noqa: E402
from traffic_safety.interaction_writer import InteractionCsvWriter  # noqa: E402
from traffic_safety.risk_event_writer import RiskEventCsvWriter  # noqa: E402
from traffic_safety.risk_timeline_writer import RiskTimelineCsvWriter  # noqa: E402

B_SCENES = ["20230922_16_CN5_T1", "20230728_31_CN37_T1", "20230922_37_CN9_T1",
            "20230922_42_CN37_T1", "20230728_32_CN4_T1"]
HOMOGRAPHY_B = {
    "image_points": [[540, 421], [1451, 437], [1746, 923], [65, 903]],
    "world_points": [[0.0, 0.0], [12.0, 0.0], [12.0, 8.2], [0.0, 8.2]],
}
BINARY_PARAMS = dict(arrival_time_s=4.0, moving_speed_kmh=2.0)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def risk_level(events) -> str:
    if any(e.level.value == "danger" for e in events):
        return "danger"
    if any(e.level.value == "warning" for e in events):
        return "warning"
    return "ok"


def run_scene(config: dict, homography: dict, state_cls, out_dir: Path, tag: str, config_path: Path) -> dict:
    """Replay one scene and write all per-scene outputs with the original writers."""
    paths = config["data_output"]
    meta = config["study_metadata"]
    tracks = pd.read_csv(paths["csv"], low_memory=False)
    frames = pd.read_csv(paths["interaction_frames_csv"])
    rep = replay(tracks, config, homography, state_cls, zone_source="saved")

    out_dir.mkdir(parents=True, exist_ok=False)
    f = {k: out_dir / f"{k}_{tag}.csv" for k in
         ("tracks", "interactions", "interaction_frames", "risk_events", "risk_timeline", "ml_frames", "binary_frame_predictions")}
    tw = CsvResultWriter(str(f["tracks"]))
    iw = InteractionCsvWriter(output_path=str(f["interactions"]), frame_metadata_path=str(f["interaction_frames"]),
                              video_id=meta["video_id"], scene_id=meta["scene_id"],
                              source_event_id=meta["source_event_id"], camera_id=meta["camera_id"])
    ew = RiskEventCsvWriter(str(f["risk_events"]))
    lw = RiskTimelineCsvWriter(str(f["risk_timeline"]))
    legacy = SafetyRuleEngine(**config["risk"])
    binary = BinaryRuleEngine(**BINARY_PARAMS)

    def none(v):
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else v

    by_frame = {k: g for k, g in tracks.join(rep, rsuffix="_new").groupby("frame_idx", sort=True)}
    bin_rows = []
    for frame_idx in frames["frame_idx"].astype(int):
        timestamp = frame_idx / FPS
        objects = []
        if frame_idx in by_frame:
            for r in by_frame[frame_idx].itertuples(index=False):
                objects.append(SimpleNamespace(
                    track_id=int(r.track_id), class_id=int(r.class_id), class_name=r.class_name,
                    category=r.category, confidence=float(r.confidence),
                    bbox=(r.bbox_x1, r.bbox_y1, r.bbox_x2, r.bbox_y2),
                    image_x=float(r.image_x), image_y=float(r.image_y),
                    world_x=float(r.world_x_m_new), world_y=float(r.world_y_m_new),
                    zone=none(r.zone_new), state=none(r.state_new), speed_kmh=none(r.speed_kmh_new),
                    distance_to_crosswalk_m=none(r.distance_to_crosswalk_m_new),
                    ttc_crosswalk_s=none(r.ttc_crosswalk_s_new)))
        events = legacy.evaluate(objects)
        iw.write_frame(frame_idx=frame_idx, timestamp_s=timestamp, objects=objects, risk_events=events, pipeline_ok=True)
        tw.write_frame(frame_idx=frame_idx, timestamp=timestamp, objects=objects)
        ew.write_frame(frame_idx=frame_idx, timestamp=timestamp, events=events)
        lw.write_frame(frame_idx=frame_idx, timestamp=timestamp, risk_level=risk_level(events), risk_events=events)
        # BinaryRuleEngine on the values as written to CSV (rounded), exactly like the offline evaluation
        bevents = binary.evaluate([SimpleNamespace(
            track_id=o.track_id, category=o.category, zone=o.zone,
            speed_kmh=None if o.speed_kmh is None else round(o.speed_kmh, 2),
            ttc_crosswalk_s=None if o.ttc_crosswalk_s is None else round(o.ttc_crosswalk_s, 3)) for o in objects])
        from evaluate_binary_rules import R1, R2
        bin_rows.append({"frame_idx": frame_idx, "new_pred": binary.frame_label(bevents),
                         "r1": int(any(e.rule == R1 for e in bevents)), "r2": int(any(e.rule == R2 for e in bevents)),
                         "n_r1_pairs": sum(e.rule == R1 for e in bevents), "n_r2_pairs": sum(e.rule == R2 for e in bevents)})
    for w in (tw, iw, ew, lw):
        w.close()
    pd.DataFrame(bin_rows).to_csv(f["binary_frame_predictions"], index=False)
    subprocess.run([sys.executable, str(ROOT / "scripts/build_frame_dataset.py"), "--config", str(config_path),
                    "--tracks", str(f["tracks"]), "--interactions", str(f["interactions"]),
                    "--frames", str(f["interaction_frames"]), "--risk-timeline", str(f["risk_timeline"]),
                    "--output", str(f["ml_frames"])], cwd=ROOT, check=True, capture_output=True, text=True)
    return f


ORIG_KEYS = {"tracks": "csv", "interactions": "interactions_csv", "interaction_frames": "interaction_frames_csv",
             "risk_events": "risk_csv", "risk_timeline": "risk_timeline_csv", "ml_frames": "ml_frames_csv"}
GEOM_TRACK_COLS = ["world_x_m", "world_y_m", "speed_kmh", "distance_to_crosswalk_m", "ttc_crosswalk_s"]
ML_GEOM_COLS = ["num_vehicles_approaching", "num_vehicles_stopped", "min_ttc_s", "min_distance_crosswalk_m",
                "max_vehicle_speed_kmh", "mean_vehicle_speed_kmh", "num_ttc_available",
                "min_pedestrian_vehicle_distance_m", "rule_pred"]


def diff_stats(a: pd.Series, b: pd.Series) -> dict:
    a = pd.to_numeric(a, errors="coerce"); b = pd.to_numeric(b, errors="coerce")
    both = a.notna() & b.notna(); d = (b - a)[both]
    return {"n_both": int(both.sum()), "old_only": int((a.notna() & b.isna()).sum()), "new_only": int((a.isna() & b.notna()).sum()),
            "old_median": float(a.median()) if a.notna().any() else None, "new_median": float(b.median()) if b.notna().any() else None,
            "old_p90": float(a.quantile(.9)) if a.notna().any() else None, "new_p90": float(b.quantile(.9)) if b.notna().any() else None,
            "diff_median": float(d.median()) if len(d) else None, "absdiff_max": float(d.abs().max()) if len(d) else None,
            "ratio_new_old_median": float((b[both] / a[both].replace(0, np.nan)).median()) if len(d) else None}


def gate_check(config: dict, f: dict, event_id: str) -> dict:
    """Homography A through the recompute path must reproduce the original decisions."""
    p = config["data_output"]
    res = {}
    for key in ("risk_timeline",):
        o = pd.read_csv(p[ORIG_KEYS[key]]); n = pd.read_csv(f[key])
        res["legacy_timeline_mismatch"] = int((o.risk_level != n.risk_level).sum()) if len(o) == len(n) else -1
    stored = pd.read_csv(ROOT / "outputs/ml_experiment/binary_rule_evaluation/binary_frame_predictions.csv")
    stored = stored[stored.source_event_id == event_id].set_index("frame_idx")
    n = pd.read_csv(f["binary_frame_predictions"]).set_index("frame_idx").reindex(stored.index)
    res["binary_mismatch"] = {c: int((stored[c] != n[c]).sum()) for c in ("new_pred", "r1", "r2")}
    o = pd.read_csv(p["csv"], low_memory=False); n = pd.read_csv(f["tracks"], low_memory=False)
    res["tracks_rows_equal"] = len(o) == len(n)
    res["state_mismatch"] = int((o.state.fillna("") != n.state.fillna("")).sum())
    res["zone_mismatch"] = int((o.zone.fillna("") != n.zone.fillna("")).sum())
    res["speed_absdiff_max"] = float((o.speed_kmh - n.speed_kmh).abs().max())
    oi = pd.read_csv(p["interactions_csv"], low_memory=False); ni = pd.read_csv(f["interactions"], low_memory=False)
    res["interactions_rows_equal"] = len(oi) == len(ni)
    res["interactions_rule_level_mismatch"] = int((oi.rule_risk_level.fillna("") != ni.rule_risk_level.fillna("")).sum()) if len(oi) == len(ni) else -1
    oe = pd.read_csv(p["risk_csv"]); ne = pd.read_csv(f["risk_events"])
    res["risk_events_rows"] = [len(oe), len(ne)]
    of = pd.read_csv(p["interaction_frames_csv"]); nf = pd.read_csv(f["interaction_frames"])
    res["interaction_frames_identical"] = bool(of.equals(nf))
    om = pd.read_csv(p["ml_frames_csv"]); nm = pd.read_csv(f["ml_frames"])
    res["ml_rule_pred_mismatch"] = int((om.rule_pred != nm.rule_pred).sum())
    res["passed"] = (res["legacy_timeline_mismatch"] == 0 and not any(res["binary_mismatch"].values())
                     and res["tracks_rows_equal"] and res["state_mismatch"] == 0 and res["zone_mismatch"] == 0
                     and res["interactions_rows_equal"] and res["interactions_rule_level_mismatch"] == 0
                     and res["risk_events_rows"][0] == res["risk_events_rows"][1]
                     and res["interaction_frames_identical"] and res["ml_rule_pred_mismatch"] == 0)
    return res


def qc_scene(config: dict, f: dict, event_id: str) -> tuple[dict, pd.DataFrame]:
    p = config["data_output"]
    o = pd.read_csv(p["csv"], low_memory=False); n = pd.read_csv(f["tracks"], low_memory=False)
    q = {"scene": event_id}
    # 1-2 identity of everything that must not change
    same_cols = ["frame_idx", "timestamp_s", "track_id", "class_id", "class_name", "category", "confidence",
                 "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "image_x", "image_y", "zone"]
    q["rows_old_new"] = [len(o), len(n)]
    q["frames_old_new"] = [int(o.frame_idx.nunique()), int(n.frame_idx.nunique())]
    q["track_ids_identical"] = set(zip(o.category, o.track_id)) == set(zip(n.category, n.track_id))
    q["unchanged_columns_identical"] = {c: bool(o[c].fillna("").astype(str).equals(n[c].fillna("").astype(str))) for c in same_cols}
    # 3 pedestrian states
    ped = o.category == "pedestrian"
    q["pedestrian_state_mismatch"] = int((o.state[ped].fillna("") != n.state[ped].fillna("")).sum())
    # 4 geometry
    veh = o.category == "vehicle"
    q["geometry"] = {c: diff_stats(o[c][veh] if c not in ("world_x_m", "world_y_m") else o[c],
                                   n[c][veh] if c not in ("world_x_m", "world_y_m") else n[c]) for c in GEOM_TRACK_COLS}
    # 5 vehicle states
    vs = pd.crosstab(o.state[veh].fillna(""), n.state[veh].fillna(""))
    q["vehicle_state_changed_rows"] = int((o.state[veh].fillna("") != n.state[veh].fillna("")).sum())
    q["vehicle_state_transition"] = {f"{a}->{b}": int(vs.loc[a, b]) for a in vs.index for b in vs.columns if vs.loc[a, b] and a != b}
    om = pd.read_csv(p["ml_frames_csv"]); nm = pd.read_csv(f["ml_frames"])
    q["ml_frames_rows_equal"] = len(om) == len(nm)
    q["ml_unchanged_columns_identical"] = {c: bool(om[c].fillna(-999).equals(nm[c].fillna(-999)))
                                           for c in om.columns if c not in ML_GEOM_COLS + ["risk_gt"]}
    q["ml_geometry_columns"] = {c: diff_stats(om[c], nm[c]) | {"frames_changed": int((om[c].fillna(-999) != nm[c].fillna(-999)).sum())} for c in ML_GEOM_COLS}
    # 6 finiteness / jumps / missing
    num = n[GEOM_TRACK_COLS].apply(pd.to_numeric, errors="coerce")
    q["non_finite_values"] = int(np.isinf(num.to_numpy(dtype=float)).sum())
    q["missing_pattern_changes"] = {c: int((o[c].isna() != n[c].isna()).sum()) for c in GEOM_TRACK_COLS}
    nn = n[veh].sort_values(["track_id", "frame_idx"]); g = nn.groupby("track_id")
    raw = np.hypot(g.world_x_m.diff(), g.world_y_m.diff()) / g.timestamp_s.diff() * 3.6
    oo = o[veh].sort_values(["track_id", "frame_idx"]); go = oo.groupby("track_id")
    raw_o = np.hypot(go.world_x_m.diff(), go.world_y_m.diff()) / go.timestamp_s.diff() * 3.6
    q["raw_step_speed_over_50kmh"] = {"old": int((raw_o > 50).sum()), "new": int((raw > 50).sum())}
    q["smoothed_speed_jump_over_5kmh"] = {"old": int((go.speed_kmh.diff().abs() > 5).sum()), "new": int((g.speed_kmh.diff().abs() > 5).sum())}
    q["max_speed_kmh"] = {"old": float(o.speed_kmh.max()), "new": float(n.speed_kmh.max())}
    # 7 binary decisions
    stored = pd.read_csv(ROOT / "outputs/ml_experiment/binary_rule_evaluation/binary_frame_predictions.csv")
    stored = stored[stored.source_event_id == event_id].set_index("frame_idx")
    b = pd.read_csv(f["binary_frame_predictions"]).set_index("frame_idx").reindex(stored.index)
    changes = []
    for c in ("new_pred", "r1", "r2"):
        q[f"binary_{c}"] = {"old_frames": int(stored[c].sum()), "new_frames": int(b[c].sum()),
                            "0->1": int(((stored[c] == 0) & (b[c] == 1)).sum()), "1->0": int(((stored[c] == 1) & (b[c] == 0)).sum())}
    ch = stored.join(b, lsuffix="_old", rsuffix="_new")
    ch = ch[(ch.new_pred_old != ch.new_pred_new) | (ch.r1_old != ch.r1_new) | (ch.r2_old != ch.r2_new)]
    ch = ch.reset_index()[["frame_idx", "new_pred_old", "new_pred_new", "r1_old", "r1_new", "r2_old", "r2_new"]].assign(scene=event_id)
    ot = pd.read_csv(p["risk_timeline_csv"]); nt = pd.read_csv(f["risk_timeline"])
    q["legacy_timeline_changed_frames"] = int((ot.risk_level != nt.risk_level).sum())
    q["legacy_timeline_levels"] = {"old": ot.risk_level.value_counts().to_dict(), "new": nt.risk_level.value_counts().to_dict()}
    return q, ch


def recompute(args) -> None:
    out = Path(args.output_dir)
    if out.exists():
        raise SystemExit(f"{out} exists; nothing is overwritten")
    commit, dirty = git("rev-parse", "HEAD"), git("status", "--porcelain", "--", "traffic_safety", "scripts")
    state_src = git("show", f"{LEGACY_STATE_COMMIT}:traffic_safety/state.py")
    state_cls = load_state_estimator_class(LEGACY_STATE_COMMIT)
    configs = {p.stem: p for p in sorted(Path(args.configs_dir).glob("*.yaml"))}
    manifest = {"version": "geometry_v2", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "command": " ".join([Path(sys.executable).name, *sys.argv]),
                "git_commit": commit, "code_dirty": bool(dirty), "dirty_files": dirty.splitlines(),
                "script_sha256": sha256(Path(__file__)),
                "state_estimator": {"commit": LEGACY_STATE_COMMIT, "source_sha256": hashlib.sha256(state_src.encode()).hexdigest()},
                "homography": HOMOGRAPHY_B, "fps": FPS, "state_params": STATE_PARAMS, "risk_metrics_params": RISK_PARAMS,
                "binary_params": BINARY_PARAMS, "zones": "taken from saved scenario_tracks.csv",
                "scenes": {}}
    out.mkdir(parents=True)
    summary, changes = [], []
    for scene in B_SCENES:
        cfg_path = configs[scene]
        config = yaml.safe_load(cfg_path.read_text())
        cal = config["calibration"]
        if cal["image_points"] != HOMOGRAPHY_B["image_points"] or [list(map(float, w)) for w in cal["world_points"]] != HOMOGRAPHY_B["world_points"]:
            raise SystemExit(f"{scene}: config calibration is not homography B")
        event_id = config["study_metadata"]["source_event_id"]
        inputs = {k: config["data_output"][v] for k, v in ORIG_KEYS.items()} | {"frame_labels": config["data_output"]["frame_labels_csv"], "config": str(cfg_path)}
        in_sha = {k: sha256(ROOT / v) for k, v in inputs.items()}
        gate_dir = out / scene / "gate_homography_A"
        gf = run_scene(config, HOMOGRAPHY_A, state_cls, gate_dir, "gateA", cfg_path)
        gate = gate_check(config, gf, event_id)
        if not gate["passed"]:
            (out / f"GATE_FAILED_{scene}.json").write_text(json.dumps(gate, indent=2, default=str))
            raise SystemExit(f"{scene}: reproduction gate failed: {gate}")
        f = run_scene(config, HOMOGRAPHY_B, state_cls, out / scene / "csv_geomv2", "geomv2", cfg_path)
        q, ch = qc_scene(config, f, event_id)
        q["gate_homography_A"] = gate
        q["inputs_unchanged"] = all(sha256(ROOT / v) == in_sha[k] for k, v in inputs.items())
        (out / scene / "qc_geomv2.json").write_text(json.dumps(q, indent=2, default=str))
        summary.append(q); changes.append(ch)
        manifest["scenes"][scene] = {"inputs": {k: {"path": v, "sha256": in_sha[k]} for k, v in inputs.items()},
                                     "outputs": {k: {"path": str(v), "sha256": sha256(v)} for k, v in f.items()},
                                     "gate_outputs_dir": str(gate_dir)}
    pd.concat(changes, ignore_index=True).to_csv(out / "binary_decision_changes_geomv2.csv", index=False)
    (out / "qc_summary_geomv2.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "manifest_geomv2.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
