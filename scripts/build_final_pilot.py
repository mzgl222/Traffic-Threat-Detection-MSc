#!/usr/bin/env python3
"""Final pilot evaluation (8 scenes) for the current pipeline version.

Configuration (fixed, not tuned):
  homography A (10 x 30 m): SN17, CN8, CN38; homography B (12 x 8.2 m): CN5, CN37_31, CN9, CN37_42, CN4
  SpeedEstimator(method="regression", window_s=0.5); current StateEstimator (S1/B1);
  current TTA / zones / interactions / rules; BinaryRuleEngine V1 (R1 > 2.0 km/h, R2 TTA <= 4.0 s); original GT.

Inputs: saved tracks (A: outputs/<scene>/csv/scenario_tracks.csv, B: outputs/geometry_v2/<scene>/csv_geomv2/
tracks_geomv2.csv). Detections, track IDs, pixel positions, timestamps and zones come from these files; the
dependent chain is replayed with replay_geometry_offline.run_scene (same classes and writers as run.py) and
build_frame_dataset.py. No YOLO / ByteTrack. Output names avoid scenario_*.csv and *.yaml (rglob consumers).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from traffic_safety.event_metrics import event_metrics, frame_metrics  # noqa: E402
from traffic_safety.motion_aggregation import aggregate_crosswalk_entry  # noqa: E402
from traffic_safety.motion_features import compute_motion_features  # noqa: E402
from traffic_safety.state import StateEstimator  # noqa: E402
from replay_geometry_offline import (  # noqa: E402
    B_SCENES, HOMOGRAPHY_A, HOMOGRAPHY_B, LEGACY_STATE_COMMIT, RISK_PARAMS, STATE_PARAMS,
    load_state_estimator_class, replay, run_scene,
)

SPEED = {"method": "regression", "window_s": 0.5}
BINARY = {"arrival_time_s": 4.0, "moving_speed_kmh": 2.0}
KEY = ["video_id", "frame_idx"]
BASELINE = ["num_pedestrians", "num_vehicles", "num_interactions", "num_pedestrians_crossing",
            "num_pedestrians_waiting", "num_vehicles_approaching", "num_vehicles_stopped", "min_ttc_s",
            "min_distance_crosswalk_m", "max_vehicle_speed_kmh", "mean_vehicle_speed_kmh",
            "num_ttc_available", "min_pedestrian_vehicle_distance_m"]
TEMPORAL = ["min_time_since_crosswalk_entry_s", "num_crosswalk_pedestrians_unknown_entry"]
PREV_DATASET = ROOT / "outputs/ml_experiment/features_v3_geomv2/frames_features_v3_geomv2.csv"
PRED_A = ROOT / "outputs/ml_experiment/binary_rule_evaluation/binary_frame_predictions.csv"
PRED_B = ROOT / "outputs/ml_experiment/binary_rule_evaluation_geomv2/frame_predictions_before_after.csv"
CHANGES_C = ROOT / "outputs/diagnostics/speed_regression_v1/rule_changes_saved_vs_regression.csv"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def md(df: pd.DataFrame) -> str:
    cell = lambda v: "" if pd.isna(v) else (f"{v:.3f}" if isinstance(v, float) else str(v))
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
                     + ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)])


def scenes():
    for p in sorted((ROOT / "configs").glob("*.yaml")):
        cfg = yaml.safe_load(p.read_text())
        ev = cfg["study_metadata"]["source_event_id"]
        hom = "B" if ev in B_SCENES else "A"
        expected = HOMOGRAPHY_B if hom == "B" else HOMOGRAPHY_A
        cal = cfg["calibration"]
        if cal["image_points"] != expected["image_points"] or \
                [list(map(float, w)) for w in cal["world_points"]] != expected["world_points"]:
            raise SystemExit(f"{ev}: config calibration does not match homography {hom}")
        src = (ROOT / f"outputs/geometry_v2/{ev}/csv_geomv2/tracks_geomv2.csv") if hom == "B" else ROOT / cfg["data_output"]["csv"]
        yield ev, hom, p, cfg, src


def scene_qc(ev, cfg, src: Path, f: dict) -> dict:
    orig = pd.read_csv(ROOT / cfg["data_output"]["csv"], low_memory=False)
    srcd = pd.read_csv(src, low_memory=False)
    new = pd.read_csv(f["tracks"], low_memory=False)
    ident = ["frame_idx", "timestamp_s", "track_id", "class_id", "class_name", "category", "confidence",
             "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "image_x", "image_y", "zone"]
    q = {"rows": [len(orig), len(srcd), len(new)],
         "identity_columns_equal_original": {c: bool(orig[c].fillna("").astype(str).equals(new[c].fillna("").astype(str))) for c in ident},
         "identity_columns_equal_source": all(srcd[c].fillna("").astype(str).equals(new[c].fillna("").astype(str)) for c in ident)}
    veh = new.category == "vehicle"
    num = new[["world_x_m", "world_y_m", "speed_kmh", "distance_to_crosswalk_m", "ttc_crosswalk_s"]]
    q["non_finite"] = int(np.isinf(num.to_numpy(dtype=float)).sum())
    q["world_nan"] = int(new[["world_x_m", "world_y_m"]].isna().sum().sum())
    q["vehicle_speed_missing"] = int(new.speed_kmh[veh].isna().sum())
    q["vehicle_speed_missing_first_0.3s_of_track"] = int((new[veh].speed_kmh.isna() &
        ((new[veh].frame_idx - new[veh].groupby("track_id").frame_idx.transform("min")) / 30 < 0.3 + 1e-9)).sum())
    q["approach_rows"] = int((veh & (new.zone == "vehicle_approach")).sum())
    q["tta_available"] = int(new.ttc_crosswalk_s.notna().sum())
    q["speed_outside_vehicles"] = int(new.speed_kmh[~veh].notna().sum())
    q["state_values"] = new.groupby("category").state.apply(lambda s: sorted(s.fillna("").unique())).to_dict()
    # pedestrian states depend only on zones -> must match an independent replay of the zone column
    ped = new[new.category == "pedestrian"]
    est = StateEstimator(**STATE_PARAMS)
    from types import SimpleNamespace
    exp = [getattr(est.update(SimpleNamespace(track_id=r.track_id, category="pedestrian", zone=None if pd.isna(r.zone) else r.zone, speed_kmh=None)), "value", None)
           for r in ped.itertuples()]
    q["pedestrian_state_independent_check_mismatch"] = int((ped.state.fillna("").to_numpy() != np.array([e or "" for e in exp])).sum())
    return q


def prefix_check(cfg, src: Path) -> int:
    t = pd.read_csv(src, low_memory=False)
    full = replay(t, cfg, cfg["calibration"], StateEstimator, zone_source="saved", speed_method=SPEED["method"], speed_window_s=SPEED["window_s"])
    mism = 0
    for cut in np.quantile(t.frame_idx, [0.25, 0.5, 0.75]).astype(int):
        part = t[t.frame_idx <= cut]
        p = replay(part, cfg, cfg["calibration"], StateEstimator, zone_source="saved", speed_method=SPEED["method"], speed_window_s=SPEED["window_s"])
        fpart = full.loc[part.index]
        for c in ("world_x_m", "world_y_m", "speed_kmh", "distance_to_crosswalk_m", "ttc_crosswalk_s"):
            a, b = pd.to_numeric(p[c]), pd.to_numeric(fpart[c])
            mism += int((~((a.isna() & b.isna()) | (a == b))).sum())
        mism += int((p.state.fillna("") != fpart.state.fillna("")).sum())
    return mism


def legacy_state_binary(cfg, src: Path) -> pd.Series:
    """Same chain with the historical StateEstimator: V1 decisions must not change (V1 ignores states)."""
    from evaluate_binary_rules import predict_frames
    from traffic_safety.binary_rules import BinaryRuleEngine
    t = pd.read_csv(src, low_memory=False)
    rep = replay(t, cfg, cfg["calibration"], load_state_estimator_class(LEGACY_STATE_COMMIT), zone_source="saved",
                 speed_method=SPEED["method"], speed_window_s=SPEED["window_s"])
    tt = t.assign(zone=rep.zone, speed_kmh=pd.to_numeric(rep.speed_kmh).round(2), ttc_crosswalk_s=pd.to_numeric(rep.ttc_crosswalk_s).round(3))
    return predict_frames(tt, BinaryRuleEngine(**BINARY)).set_index("frame_idx")


def evaluate(preds: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for sc, g in preds.groupby("source_event_id", sort=True):
        r = {"scene": sc} | frame_metrics(g.risk_gt, g[col]) | event_metrics(g, col)
        r["events_missed"] = r["gt_events"] - r["events_detected"]
        rows.append(r)
    per = pd.DataFrame(rows)
    tot = {"scene": "ALL"} | frame_metrics(preds.risk_gt, preds[col])
    for c in ("gt_events", "events_detected", "events_missed", "alarm_episodes", "false_alarm_episodes", "false_alarm_duration_s"):
        tot[c] = per[c].sum()
    tot["first_alarm_delay_s"] = per.first_alarm_delay_s.mean()
    tot["event_coverage"] = per.event_coverage.mean()
    return pd.concat([per, pd.DataFrame([tot])], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default="outputs/final_pilot_v1")
    args = ap.parse_args()
    out = ROOT / args.output_dir
    if out.exists():
        raise SystemExit(f"{out} exists; nothing is overwritten")
    out.mkdir(parents=True)

    inputs, qc, per_scene_files, final_parts, ml_parts = {}, {"scenes": {}}, {}, [], []
    for ev, hom, cfg_path, cfg, src in scenes():
        for p in (src, ROOT / cfg["data_output"]["frame_labels_csv"], ROOT / cfg["data_output"]["interaction_frames_csv"], cfg_path):
            inputs[str(p.relative_to(ROOT))] = sha256(p)
        f = run_scene(cfg, cfg["calibration"], StateEstimator, out / "scenes" / ev / "csv_final", "final", cfg_path,
                      tracks_path=src, speed_method=SPEED["method"], speed_window_s=SPEED["window_s"])
        per_scene_files[ev] = f
        q = scene_qc(ev, cfg, src, f)
        q["homography"] = hom
        q["prefix_mismatches"] = prefix_check(cfg, src)
        b = pd.read_csv(f["binary_frame_predictions"]).set_index("frame_idx")
        q["v1_identical_with_historical_state_estimator"] = bool(
            legacy_state_binary(cfg, src).reindex(b.index, fill_value=0)[["new_pred", "r1", "r2"]].equals(b[["new_pred", "r1", "r2"]]))
        gt = pd.read_csv(ROOT / cfg["data_output"]["frame_labels_csv"]).set_index("frame_idx").risk_gt
        frames = pd.read_csv(ROOT / cfg["data_output"]["interaction_frames_csv"])
        q["frames"] = len(frames); q["binary_frames"] = len(b); q["label_frames"] = len(gt)
        q["frames_equal_labels"] = set(frames.frame_idx) == set(gt.index) == set(b.index)
        qc["scenes"][ev] = q
        final_parts.append(pd.DataFrame({"video_id": cfg["study_metadata"]["video_id"], "source_event_id": ev,
                                         "frame_idx": b.index, "timestamp_s": b.index / 30.0,
                                         "risk_gt": gt.reindex(b.index).astype(int).to_numpy(),
                                         "pred_D": b.new_pred.to_numpy(), "r1_D": b.r1.to_numpy(), "r2_D": b.r2.to_numpy()}))
        # temporal B4 features recomputed from the final tracks (zones + timestamps only)
        tr = pd.read_csv(f["tracks"], low_memory=False)
        feats = compute_motion_features(tr)
        agg = aggregate_crosswalk_entry(tr.assign(motion_stable_zone=feats["stable_zone"],
                                                  time_since_crosswalk_entry_s=feats["time_since_crosswalk_entry_s"]), frames.frame_idx)
        ml = pd.read_csv(f["ml_frames"], dtype=str, keep_default_na=False)
        ml_parts.append((ev, ml, agg))

    # ------------------------------------------------ V1 evaluation and comparison A/B/C/D
    P = pd.concat(final_parts, ignore_index=True).sort_values(["source_event_id", "frame_idx"]).reset_index(drop=True)
    a = pd.read_csv(PRED_A).rename(columns={"new_pred": "pred_A", "r1": "r1_A", "r2": "r2_A"})
    bb = pd.read_csv(PRED_B).rename(columns={"pred_after": "pred_B", "r1_after": "r1_B", "r2_after": "r2_B"})
    P = P.merge(a[KEY + ["pred_A", "r1_A", "r2_A", "risk_gt"]].rename(columns={"risk_gt": "gt_A"}), on=KEY, how="left", validate="one_to_one")
    P = P.merge(bb[KEY + ["pred_B", "r1_B", "r2_B"]], on=KEY, how="left", validate="one_to_one")
    ch = pd.read_csv(CHANGES_C).set_index(["scene", "frame_idx"])
    P["pred_C"], P["r1_C"], P["r2_C"] = P.pred_B, P.r1_B, P.r2_B
    idx = pd.MultiIndex.from_frame(P[["source_event_id", "frame_idx"]])
    hit = idx.isin(ch.index)
    sel = ch.reindex(idx[hit])
    P.loc[hit, "pred_C"] = sel.danger_reg.to_numpy(); P.loc[hit, "r1_C"] = sel.r1_reg.to_numpy(); P.loc[hit, "r2_C"] = sel.r2_reg.to_numpy()
    qc["gt_identical_to_previous_evaluations"] = bool((P.risk_gt == P.gt_A).all())
    rules_dir = out / "rules_v1"; rules_dir.mkdir()
    P.drop(columns="gt_A").to_csv(rules_dir / "frame_predictions_A_B_C_D.csv", index=False)
    versions = {"A": "old geometry + median_step (stored Phase I)", "B": "corrected geometry + median_step (geometry_v2)",
                "C": "corrected geometry + regression + historical StateEstimator (offline diagnostic, speed_regression_v1)",
                "D": "FINAL: corrected geometry + regression 0.5 s + current StateEstimator"}
    metrics = []
    for v in "ABCD":
        m = evaluate(P, f"pred_{v}"); m.insert(0, "version", v)
        m["r1_frames"] = [int(P.loc[P.source_event_id == s, f"r1_{v}"].sum()) if s != "ALL" else int(P[f"r1_{v}"].sum()) for s in m.scene]
        m["r2_frames"] = [int(P.loc[P.source_event_id == s, f"r2_{v}"].sum()) if s != "ALL" else int(P[f"r2_{v}"].sum()) for s in m.scene]
        metrics.append(m)
    M = pd.concat(metrics, ignore_index=True); M.to_csv(rules_dir / "metrics_A_B_C_D.csv", index=False)
    qc["D_equals_C_frame_level"] = {c: int((P[f"{c}_D"] != P[f"{c}_C"]).sum()) for c in ("pred", "r1", "r2")}
    dif = P[(P.pred_D != P.pred_B)].assign(change=lambda d: np.where(d.pred_D > d.pred_B, "new_alarm", "removed_alarm"))
    dif["outcome"] = np.select([(dif.change == "new_alarm") & (dif.risk_gt == 1), (dif.change == "new_alarm") & (dif.risk_gt == 0),
                                (dif.change == "removed_alarm") & (dif.risk_gt == 1)], ["new_TP", "new_FP", "lost_TP"], "removed_FP")
    dif.to_csv(rules_dir / "changes_D_vs_B.csv", index=False)

    # ------------------------------------------------ ML dataset
    prev = pd.read_csv(PREV_DATASET, dtype=str, keep_default_na=False)
    rows, temporal_equal = [], {}
    for ev, ml, agg in ml_parts:
        p = prev[prev.source_event_id == ev].set_index("frame_idx")
        ml = ml.set_index("frame_idx")
        if set(ml.index) != set(p.index):
            raise SystemExit(f"{ev}: frame sets differ from previous dataset")
        agg = agg.set_index("frame_idx")
        agg.index = agg.index.astype(str)
        new_min = agg.min_time_since_crosswalk_entry_s.map(lambda v: "" if pd.isna(v) else f"{v:.6f}")
        new_unk = agg.num_crosswalk_pedestrians_unknown_entry.astype(int).astype(str)
        temporal_equal[ev] = bool((new_min.reindex(p.index) == p.min_time_since_crosswalk_entry_s).all()
                                  and (new_unk.reindex(p.index) == p.num_crosswalk_pedestrians_unknown_entry).all())
        d = ml.reindex(p.index).reset_index()
        d["risk_gt"] = p.risk_gt.to_numpy()
        d["source_file"] = str(per_scene_files[ev]["ml_frames"].relative_to(ROOT))
        d["min_time_since_crosswalk_entry_s"] = new_min.reindex(p.index).to_numpy()
        d["num_crosswalk_pedestrians_unknown_entry"] = new_unk.reindex(p.index).to_numpy()
        d["geometry_version"] = "B_12x8.2" if ev in B_SCENES else "A_10x30"
        d["speed_method"] = "regression_0.5s"
        d["state_estimator"] = "current_S1_B1"
        rows.append(d)
    D = pd.concat(rows, ignore_index=True)
    D = D.assign(_f=D.frame_idx.astype(int)).sort_values(["video_id", "_f"]).drop(columns="_f").reset_index(drop=True)
    D = D[list(prev.columns) + ["geometry_version", "speed_method", "state_estimator"]]
    data_dir = out / "dataset"; data_dir.mkdir()
    ds = data_dir / "frames_final_pilot_v1.csv"; D.to_csv(ds, index=False)
    re = pd.read_csv(ds, low_memory=False)
    qc["dataset"] = {
        "rows": len(re), "gt_positive": int(re.risk_gt.sum()), "duplicate_keys": int(re.duplicated(KEY).sum()),
        "same_keys_as_previous": set(zip(re.video_id, re.frame_idx.astype(str))) == set(zip(prev.video_id, prev.frame_idx)),
        "gt_identical_to_previous": bool((re.risk_gt.astype(str).to_numpy() == prev.sort_values(["video_id"]).set_index(KEY).loc[list(zip(re.video_id, re.frame_idx.astype(str)))].risk_gt.to_numpy()).all()),
        "temporal_features_recomputed_equal_previous": temporal_equal,
        "columns_superset_of_previous": set(prev.columns) <= set(re.columns),
        "non_finite_features": int(np.isinf(re[BASELINE + TEMPORAL].to_numpy(dtype=float)).sum()),
        "missing_per_feature": re[BASELINE + TEMPORAL].isna().sum().astype(int).to_dict(),
        "baseline_changed_frames_vs_previous": {c: int((D[c].to_numpy() != prev.set_index(KEY).loc[list(zip(D.video_id, D.frame_idx))][c].to_numpy()).sum()) for c in BASELINE + ["rule_pred"]},
    }
    qc["totals"] = {"frames": int(sum(q["frames"] for q in qc["scenes"].values())), "gt_positive": int(P.risk_gt.sum())}

    # ------------------------------------------------ checks summary, manifest, report
    sc = qc["scenes"].values()
    checks = {
        "1_frames_18187": qc["totals"]["frames"] == 18187 and qc["dataset"]["rows"] == 18187,
        "2_gt_positive_1026": qc["totals"]["gt_positive"] == 1026 and qc["dataset"]["gt_positive"] == 1026,
        "3_identity_columns_equal_original": all(all(q["identity_columns_equal_original"].values()) and q["identity_columns_equal_source"] for q in sc),
        "4_homography_assignment": {ev: q["homography"] for ev, q in qc["scenes"].items()},
        "5_no_duplicates_no_missing_frames": qc["dataset"]["duplicate_keys"] == 0 and qc["dataset"]["same_keys_as_previous"] and all(q["frames_equal_labels"] for q in sc),
        "6_prefix_mismatches": sum(q["prefix_mismatches"] for q in sc),
        "7_pedestrian_states_independent_check_mismatch": sum(q["pedestrian_state_independent_check_mismatch"] for q in sc),
        "8_non_finite": sum(q["non_finite"] for q in sc) + qc["dataset"]["non_finite_features"],
        "8_world_nan": sum(q["world_nan"] for q in sc),
        "9_dataset_schema_superset_of_previous": qc["dataset"]["columns_superset_of_previous"],
        "gt_identical": qc["gt_identical_to_previous_evaluations"] and qc["dataset"]["gt_identical_to_previous"],
        "temporal_features_equal_previous": all(temporal_equal.values()),
        "v1_independent_of_state_estimator_version": all(q["v1_identical_with_historical_state_estimator"] for q in sc),
    }
    qc["checks"] = checks
    (out / "qc.json").write_text(json.dumps(qc, indent=2, default=str))
    outputs = {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()}
    manifest = {
        "name": "final_pilot_v1", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "git_commit": git("rev-parse", "HEAD"), "uncommitted_changes": git("status", "--porcelain").splitlines(),
        "python": platform.python_version(),
        "module_sha256": {m: sha256(ROOT / m) for m in ("traffic_safety/speed.py", "traffic_safety/state.py", "traffic_safety/risk_metrics.py",
                                                         "traffic_safety/zones.py", "traffic_safety/binary_rules.py", "traffic_safety/safety_rules.py",
                                                         "traffic_safety/interaction_writer.py", "traffic_safety/csv_writer.py",
                                                         "scripts/replay_geometry_offline.py", "scripts/build_frame_dataset.py",
                                                         "scripts/build_final_pilot.py", "traffic_safety/motion_features.py",
                                                         "traffic_safety/motion_aggregation.py")},
        "homography": {"A": {"scenes": [e for e, q in qc["scenes"].items() if q["homography"] == "A"], **HOMOGRAPHY_A},
                       "B": {"scenes": [e for e, q in qc["scenes"].items() if q["homography"] == "B"], **HOMOGRAPHY_B}},
        "speed_estimator": SPEED | {"min_samples": 5, "min_coverage": 0.6},
        "state_estimator": {"version": "current (S1/B1)", "params": STATE_PARAMS},
        "risk_metrics_params": RISK_PARAMS, "binary_rules": BINARY, "zones": "from input tracks (pixel space)",
        "frames": qc["totals"]["frames"], "gt_positive": qc["totals"]["gt_positive"],
        "dataset": {"file": str(ds.relative_to(ROOT)), "sha256": sha256(ds), "baseline_features": BASELINE, "temporal_features": TEMPORAL,
                    "metadata_columns": ["geometry_version", "speed_method", "state_estimator", "source_file"]},
        "output_sha256": sha256(ds), "inputs_sha256": inputs, "outputs_sha256": outputs,
        "reproduce": "python scripts/build_final_pilot.py --output-dir <new dir>",
        "checks": checks,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    show = ["version", "scene", "precision", "recall", "f1", "tp", "fp", "fn", "tn", "alarm_frames", "r1_frames", "r2_frames",
            "events_detected", "events_missed", "false_alarm_episodes", "false_alarm_duration_s", "first_alarm_delay_s", "event_coverage"]
    (out / "report.md").write_text(f"""# Final pilot evaluation v1 (rules V1)

Versions: {json.dumps(versions, indent=1)}

Version C comes from an offline diagnostic experiment (speed_regression_v1), not a full final evaluation.

## Checks
{json.dumps(checks, indent=1)}

## Metrics
{md(M[show])}

## D vs B changed frames
{md(dif.groupby(['source_event_id', 'outcome']).size().reset_index(name='frames'))}
""")
    print(json.dumps(checks, indent=1, default=str))
    bad = [k for k, v in checks.items() if v is False or (k[:2] in ("6_", "7_", "8_") and v != 0)]
    if bad:
        raise SystemExit(f"QC FAILED: {bad}")


if __name__ == "__main__":
    main()
