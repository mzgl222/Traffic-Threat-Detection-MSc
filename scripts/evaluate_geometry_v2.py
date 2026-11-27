#!/usr/bin/env python3
"""Rule evaluation before/after the homography correction and frame dataset v3 (geometry_v2).

Task 1 (--rules-dir): BinaryRuleEngine V1 (4.0 s, 2.0 km/h), frame and event metrics,
    before = outputs/ml_experiment/binary_rule_evaluation/binary_frame_predictions.csv
    after  = scenes A unchanged + scenes B from outputs/geometry_v2/<scene>/csv_geomv2/.
Task 2 (--dataset-dir): frames_features_v3_geomv2.csv
    = outputs/ml_experiment/features_v2_r2/frames_features_v2.csv with, for scenes B only,
      the columns of ml_frames_geomv2.csv (baseline features + rule_pred) substituted.
      Temporal B4 aggregates and GT are carried over unchanged (they do not depend on the
      homography). Values are copied as text.
Nothing is overwritten; both output directories must not exist. No model is trained.
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
sys.path.insert(0, str(ROOT / "scripts"))

from traffic_safety.event_metrics import event_metrics, frame_metrics  # noqa: E402
from replay_geometry_offline import (  # noqa: E402
    B_SCENES, HOMOGRAPHY_B, LEGACY_STATE_COMMIT, load_state_estimator_class, replay,
)

KEY = ["video_id", "frame_idx"]
OLD_PRED = Path("outputs/ml_experiment/binary_rule_evaluation/binary_frame_predictions.csv")
OLD_DATASET = Path("outputs/ml_experiment/features_v2_r2/frames_features_v2.csv")
OLD_DATASET_MANIFEST = Path("outputs/ml_experiment/features_v2_r2/manifest.json")
GEOM = Path("outputs/geometry_v2")
GEOM_MANIFEST = GEOM / "manifest_geomv2.json"
BASELINE = ["num_pedestrians", "num_vehicles", "num_interactions", "num_pedestrians_crossing",
            "num_pedestrians_waiting", "num_vehicles_approaching", "num_vehicles_stopped", "min_ttc_s",
            "min_distance_crosswalk_m", "max_vehicle_speed_kmh", "mean_vehicle_speed_kmh",
            "num_ttc_available", "min_pedestrian_vehicle_distance_m"]
GEOM_DEPENDENT = ["num_vehicles_approaching", "num_vehicles_stopped", "min_ttc_s", "min_distance_crosswalk_m",
                  "max_vehicle_speed_kmh", "mean_vehicle_speed_kmh", "num_ttc_available",
                  "min_pedestrian_vehicle_distance_m", "rule_pred"]
TEMPORAL = ["min_time_since_crosswalk_entry_s", "num_crosswalk_pedestrians_unknown_entry"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def md(df: pd.DataFrame) -> str:
    cell = lambda v: "" if pd.isna(v) else (f"{v:.3f}" if isinstance(v, float) else str(v))
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
                     + ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)])


def configs_by_event() -> dict:
    out = {}
    for p in sorted(Path("configs").glob("*.yaml")):
        c = yaml.safe_load(p.read_text())
        out[c["study_metadata"]["source_event_id"]] = (p, c)
    return out


# ---------------------------------------------------------------- Task 1
def load_predictions(cfgs: dict) -> pd.DataFrame:
    old = pd.read_csv(OLD_PRED)
    frames = pd.read_csv(OLD_DATASET, usecols=KEY + ["source_event_id", "timestamp_s", "risk_gt"])
    parts = []
    for event, (path, cfg) in cfgs.items():
        o = old[old.source_event_id == event].set_index("frame_idx")
        labels = pd.read_csv(cfg["data_output"]["frame_labels_csv"]).set_index("frame_idx")
        if event in B_SCENES:
            n = pd.read_csv(GEOM / event / "csv_geomv2" / "binary_frame_predictions_geomv2.csv").set_index("frame_idx")
        else:
            n = o[["new_pred", "r1", "r2", "n_r1_pairs", "n_r2_pairs"]]
        f = frames[frames.source_event_id == event].set_index("frame_idx")
        if not (o.index.sort_values().equals(f.index.sort_values()) and n.index.sort_values().equals(f.index.sort_values())):
            raise SystemExit(f"{event}: frame sets differ")
        if not (o.risk_gt.reindex(f.index) == f.risk_gt).all() or not (labels.risk_gt.reindex(f.index) == f.risk_gt).all():
            raise SystemExit(f"{event}: GT differs between sources")
        part = pd.DataFrame({
            "video_id": f.video_id, "source_event_id": event, "frame_idx": f.index, "timestamp_s": f.timestamp_s,
            "risk_gt": f.risk_gt.astype(int), "geometry": "B_geomv2" if event in B_SCENES else "A_original",
            "pred_before": o.new_pred.reindex(f.index).astype(int), "r1_before": o.r1.reindex(f.index).astype(int),
            "r2_before": o.r2.reindex(f.index).astype(int),
            "pred_after": n.new_pred.reindex(f.index).astype(int), "r1_after": n.r1.reindex(f.index).astype(int),
            "r2_after": n.r2.reindex(f.index).astype(int)}).reset_index(drop=True)
        parts.append(part)
    return pd.concat(parts, ignore_index=True).sort_values(["source_event_id", "frame_idx"]).reset_index(drop=True)


def metric_rows(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scene, geom), g in pred.groupby(["source_event_id", "geometry"], sort=True):
        for version in ("before", "after"):
            r = {"scene": scene, "geometry": geom, "version": version}
            r |= frame_metrics(g.risk_gt, g[f"pred_{version}"])
            r |= event_metrics(g, f"pred_{version}")
            r["events_missed"] = r["gt_events"] - r["events_detected"]
            r["r1_frames"] = int(g[f"r1_{version}"].sum()); r["r2_frames"] = int(g[f"r2_{version}"].sum())
            rows.append(r)
    per = pd.DataFrame(rows)
    pooled = []
    for version in ("before", "after"):
        s = per[per.version == version]
        r = {"scene": "ALL", "geometry": "all", "version": version}
        r |= frame_metrics(pred.risk_gt, pred[f"pred_{version}"])
        for c in ("gt_events", "events_detected", "events_missed", "alarm_episodes", "false_alarm_episodes",
                  "false_alarm_duration_s", "r1_frames", "r2_frames"):
            r[c] = s[c].sum()
        r["first_alarm_delay_s"] = s.first_alarm_delay_s.mean()
        r["event_coverage"] = s.event_coverage.mean()
        pooled.append(r)
    return pd.concat([per, pd.DataFrame(pooled)], ignore_index=True)


def rule_attribution(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scene, geom), g in list(pred.groupby(["source_event_id", "geometry"])) + [(("ALL", "all"), pred)]:
        for rule in ("r1", "r2"):
            b, a = g[f"{rule}_before"], g[f"{rule}_after"]
            for label in (1, 0):
                m = g.risk_gt == label
                rows.append({"scene": scene, "geometry": geom, "rule": rule.upper(),
                             "gt": label, "frames_before": int(b[m].sum()), "frames_after": int(a[m].sum()),
                             "0->1": int(((b == 0) & (a == 1) & m).sum()), "1->0": int(((b == 1) & (a == 0) & m).sum())})
    return pd.DataFrame(rows)


def changed_frames(pred: pd.DataFrame) -> pd.DataFrame:
    c = pred[(pred.pred_before != pred.pred_after) | (pred.r1_before != pred.r1_after) | (pred.r2_before != pred.r2_after)].copy()
    c["change"] = np.select([(c.pred_before == 0) & (c.pred_after == 1), (c.pred_before == 1) & (c.pred_after == 0)],
                            ["new_alarm", "removed_alarm"], "rule_changed_same_decision")
    c["outcome"] = np.select([(c.change == "new_alarm") & (c.risk_gt == 1), (c.change == "new_alarm") & (c.risk_gt == 0),
                              (c.change == "removed_alarm") & (c.risk_gt == 1), (c.change == "removed_alarm") & (c.risk_gt == 0)],
                             ["new_TP", "new_FP", "lost_TP", "removed_FP"], "")
    return c


def runs(c: pd.DataFrame) -> pd.DataFrame:
    """Group changed frames into contiguous runs for readability."""
    out = []
    for (scene, outcome), g in c[c.outcome != ""].groupby(["source_event_id", "outcome"]):
        f = g.frame_idx.to_numpy(); t = g.timestamp_s.to_numpy(); start = 0
        for i in range(1, len(f) + 1):
            if i == len(f) or f[i] != f[i - 1] + 1:
                gg = g.iloc[start:i]
                out.append({"scene": scene, "outcome": outcome, "frame_start": int(f[start]), "frame_end": int(f[i - 1]),
                            "t_start_s": round(float(t[start]), 2), "frames": i - start,
                            "rule_after": "+".join(r for r in ("R1", "R2") if gg[f"{r.lower()}_after"].any())})
                start = i
    return pd.DataFrame(out)


# ---------------------------------------------------------------- Task 2
def build_dataset(cfgs: dict) -> tuple[pd.DataFrame, dict]:
    old = pd.read_csv(OLD_DATASET, dtype=str, keep_default_na=False)
    new = old.copy()
    qc = {}
    for event in B_SCENES:
        g = pd.read_csv(GEOM / event / "csv_geomv2" / "ml_frames_geomv2.csv", dtype=str, keep_default_na=False)
        g = g.set_index("frame_idx")
        rows = new.index[new.source_event_id == event]
        fi = new.loc[rows, "frame_idx"]
        if g.index.has_duplicates or set(g.index) != set(fi) or (g.video_id.unique().tolist() != new.loc[rows, "video_id"].unique().tolist()):
            raise SystemExit(f"{event}: key mismatch between dataset and geometry_v2")
        independent = [c for c in BASELINE + ["timestamp_s", "pipeline_ok"] if c not in GEOM_DEPENDENT]
        qc[event] = {"geometry_independent_columns_identical": bool(
            (new.loc[rows, independent].to_numpy() == g.loc[fi, independent].to_numpy()).all())}
        for c in BASELINE + ["rule_pred"]:
            new.loc[rows, c] = g.loc[fi, c].to_numpy()
        new.loc[rows, "source_file"] = str(GEOM / event / "csv_geomv2" / "ml_frames_geomv2.csv")
    new["geometry_version"] = np.where(new.source_event_id.isin(B_SCENES), "B_geomv2", "A_original")
    return old, new, qc


def prefix_check_geometry(cfgs: dict) -> dict:
    """Replay with homography B on truncated tracks must equal the full replay for earlier frames."""
    state_cls = load_state_estimator_class(LEGACY_STATE_COMMIT)
    out = {}
    for event in B_SCENES:
        _, cfg = cfgs[event]
        t = pd.read_csv(cfg["data_output"]["csv"], low_memory=False)
        full = replay(t, cfg, HOMOGRAPHY_B, state_cls, zone_source="saved")
        mism, cuts = 0, sorted({int(np.quantile(t.frame_idx, q)) for q in (0.25, 0.5, 0.75)})
        for cut in cuts:
            part = t[t.frame_idx <= cut]
            p = replay(part, cfg, HOMOGRAPHY_B, state_cls, zone_source="saved")
            f = full.loc[part.index]
            for c in ("speed_kmh", "distance_to_crosswalk_m", "ttc_crosswalk_s", "world_x_m", "world_y_m"):
                a, b = pd.to_numeric(p[c]), pd.to_numeric(f[c])
                mism += int((~((a.isna() & b.isna()) | (a == b))).sum())
            mism += int((p.state.fillna("") != f.state.fillna("")).sum())
        out[event] = {"cuts": cuts, "mismatches": mism}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rules-dir", default="outputs/ml_experiment/binary_rule_evaluation_geomv2")
    ap.add_argument("--dataset-dir", default="outputs/ml_experiment/features_v3_geomv2")
    args = ap.parse_args()
    rules_dir, data_dir = Path(args.rules_dir), Path(args.dataset_dir)
    for d in (rules_dir, data_dir):
        if d.exists():
            raise SystemExit(f"{d} exists; nothing is overwritten")
    cfgs = configs_by_event()
    inputs = [OLD_PRED, OLD_DATASET, GEOM_MANIFEST] + [Path(c["data_output"]["frame_labels_csv"]) for _, c in cfgs.values()] \
        + [GEOM / e / "csv_geomv2" / f for e in B_SCENES for f in ("binary_frame_predictions_geomv2.csv", "ml_frames_geomv2.csv")]
    in_sha = {str(p): sha256(p) for p in inputs}
    # geometry_v2 outputs must be the files recorded in its manifest
    gm = json.loads(GEOM_MANIFEST.read_text())
    geom_ok = all(sha256(Path(v["path"])) == v["sha256"] for s in gm["scenes"].values() for v in s["outputs"].values())
    if not geom_ok:
        raise SystemExit("geometry_v2 files differ from manifest_geomv2.json")

    # ---- Task 1
    pred = load_predictions(cfgs)
    metrics = metric_rows(pred)
    attrib = rule_attribution(pred)
    changes = changed_frames(pred)
    change_runs = runs(changes)
    rules_dir.mkdir(parents=True)
    pred.to_csv(rules_dir / "frame_predictions_before_after.csv", index=False)
    metrics.to_csv(rules_dir / "metrics_before_after.csv", index=False)
    attrib.to_csv(rules_dir / "rule_attribution_before_after.csv", index=False)
    changes.to_csv(rules_dir / "changed_frames.csv", index=False)
    change_runs.to_csv(rules_dir / "changed_frame_runs.csv", index=False)

    # ---- Task 2
    old, new, qc_b = build_dataset(cfgs)
    data_dir.mkdir(parents=True)
    out_csv = data_dir / "frames_features_v3_geomv2.csv"
    new.to_csv(out_csv, index=False)
    re = pd.read_csv(out_csv, dtype=str, keep_default_na=False)
    a_rows = ~re.source_event_id.isin(B_SCENES)
    b_rows = ~a_rows
    old_cols = list(old.columns)
    geom_ok_b = True
    for e in B_SCENES:
        g = pd.read_csv(GEOM / e / "csv_geomv2" / "ml_frames_geomv2.csv", dtype=str, keep_default_na=False).set_index("frame_idx")
        r = re[re.source_event_id == e].set_index("frame_idx")
        geom_ok_b &= bool((r[BASELINE + ["rule_pred"]].to_numpy() == g.loc[r.index, BASELINE + ["rule_pred"]].to_numpy()).all())
    prefix = prefix_check_geometry(cfgs)
    changed_cols_b = {c: int((re.loc[b_rows, c].to_numpy() != old.loc[b_rows.to_numpy(), c].to_numpy()).sum()) for c in BASELINE + ["rule_pred"]}
    checks = {
        "1_rows_18187": len(re) == 18187,
        "2_gt_identical_to_previous_dataset": bool((re.risk_gt == old.risk_gt).all()),
        "2_gt_identical_to_frame_labels": bool(all(
            (pd.read_csv(c["data_output"]["frame_labels_csv"]).set_index("frame_idx").risk_gt.astype(str)
             .reindex(re[re.source_event_id == e].frame_idx.astype(int)).to_numpy()
             == re[re.source_event_id == e].risk_gt.to_numpy()).all() for e, (_, c) in cfgs.items())),
        "3_no_duplicate_keys": not re.duplicated(KEY).any(),
        "3_same_key_set_as_previous": set(zip(re.video_id, re.frame_idx)) == set(zip(old.video_id, old.frame_idx)),
        "4_scenes_A_identical_all_previous_columns": bool(re.loc[a_rows, old_cols].equals(old.loc[a_rows.to_numpy(), old_cols])),
        "5_scenes_B_geometry_from_geometry_v2": geom_ok_b,
        "5_scenes_B_geometry_independent_columns_unchanged": all(v["geometry_independent_columns_identical"] for v in qc_b.values()),
        "6_temporal_features_identical": bool(re[TEMPORAL].equals(old[TEMPORAL])),
        "7_row_order_and_keys_aligned": bool((re[KEY + ["source_event_id"]].to_numpy() == old[KEY + ["source_event_id"]].to_numpy()).all()),
        "8_geometry_replay_prefix_mismatches": sum(v["mismatches"] for v in prefix.values()),
        "geometry_v2_matches_its_manifest": geom_ok,
        "inputs_unchanged": all(sha256(Path(p)) == h for p, h in in_sha.items()),
    }
    feats_old = old.loc[b_rows.to_numpy(), BASELINE].replace("", np.nan).astype(float)
    feats_new = re.loc[b_rows, BASELINE].replace("", np.nan).astype(float)
    feat_cmp = pd.DataFrame({"feature": BASELINE,
                             "frames_changed_B": [changed_cols_b[c] for c in BASELINE],
                             "median_before_B": feats_old.median().values, "median_after_B": feats_new.median().values,
                             "missing_before_B": feats_old.isna().sum().values, "missing_after_B": feats_new.isna().sum().values})
    feat_cmp.to_csv(data_dir / "qc_feature_changes_scenes_B.csv", index=False)

    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "command": " ".join([Path(sys.executable).name, *sys.argv]), "git_commit": git("rev-parse", "HEAD"),
                "code_dirty": bool(git("status", "--porcelain", "--", "traffic_safety", "scripts")),
                "binary_params": {"arrival_time_s": 4.0, "moving_speed_kmh": 2.0},
                "homography": {"A": ["SN17", "CN8", "CN38"], "B": B_SCENES}, "state_estimator_commit": LEGACY_STATE_COMMIT,
                "dataset": {"file": str(out_csv), "sha256": sha256(out_csv), "baseline_features": BASELINE,
                            "temporal_features": TEMPORAL, "label": "risk_gt", "added_columns": ["geometry_version"]},
                "checks": checks, "prefix_check": prefix, "inputs_sha256": in_sha,
                "rule_outputs_sha256": {p.name: sha256(p) for p in sorted(rules_dir.glob("*.csv"))}}
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (rules_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    show = ["scene", "version", "precision", "recall", "f1", "tp", "fp", "fn", "tn", "alarm_frames", "r1_frames", "r2_frames",
            "events_detected", "events_missed", "false_alarm_episodes", "false_alarm_duration_s", "first_alarm_delay_s"]
    outcomes = changes[changes.outcome != ""].groupby(["source_event_id", "outcome"]).size().unstack(fill_value=0)
    report = f"""# Rules V1 before/after homography correction + dataset features_v3_geomv2

BinaryRuleEngine V1, arrival 4.0 s, moving 2.0 km/h. Scenes A unchanged; scenes B from outputs/geometry_v2.
No thresholds tuned, no model trained.

## Metrics (frame + event)

{md(metrics[show])}

## Rule attribution (frames by GT)

{md(attrib)}

## Changed decisions (new alarms classified with GT)

{md(outcomes.reset_index())}

{md(change_runs)}

## Dataset checks

{json.dumps(checks, indent=2)}

{md(feat_cmp)}
"""
    (rules_dir / "report.md").write_text(report)
    (data_dir / "report.md").write_text(report)
    print(json.dumps(checks, indent=2))
    failed = [k for k, v in checks.items() if v is False or (k.startswith("8_") and v != 0)]
    if failed:
        raise SystemExit(f"QC FAILED: {failed}")


if __name__ == "__main__":
    main()
