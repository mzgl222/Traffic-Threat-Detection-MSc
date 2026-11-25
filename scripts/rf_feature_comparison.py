#!/usr/bin/env python3
"""Phase III, experiment 1: Random Forest, feature set A vs B, leave-one-scene-out.

A = 13 baseline frame features (as in notebooks/wts_ml_experiment_yaml_labels.ipynb)
B = A + min_time_since_crosswalk_entry_s + num_crosswalk_pedestrians_unknown_entry

Model, imputation and hyper-parameters are copied from the notebook
(SimpleImputer(median, add_indicator=True, keep_empty_features=True) +
RandomForestClassifier(200 trees, max_depth=8, min_samples_leaf=5,
class_weight="balanced")), threshold 0.5 (model.predict). The pipeline is
fitted on the 7 training scenes only in every fold. Seeds are fixed in
advance; all of them are reported (no selection). No tuning.

Reference: saved BinaryRuleEngine V1 frame predictions (new_pred).

Pilot experiment on 8 scenes; NOT a final generalisation test.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from traffic_safety.event_metrics import event_metrics, frame_metrics  # noqa: E402
from compute_motion_features import git_state, markdown_table, sha256  # noqa: E402

DATASET = Path("outputs/ml_experiment/features_v2_r2/frames_features_v2.csv")
DATASET_MANIFEST = Path("outputs/ml_experiment/features_v2_r2/manifest.json")
RULES_V1 = Path("outputs/ml_experiment/binary_rule_evaluation/binary_frame_predictions.csv")
KEY = ["video_id", "frame_idx"]

FEATURES_A = [
    "num_pedestrians", "num_vehicles", "num_interactions",
    "num_pedestrians_crossing", "num_pedestrians_waiting",
    "num_vehicles_approaching", "num_vehicles_stopped",
    "min_ttc_s", "min_distance_crosswalk_m",
    "max_vehicle_speed_kmh", "mean_vehicle_speed_kmh",
    "num_ttc_available", "min_pedestrian_vehicle_distance_m",
]
NEW_FEATURES = ["min_time_since_crosswalk_entry_s", "num_crosswalk_pedestrians_unknown_entry"]
VARIANTS = {"RF_A": FEATURES_A, "RF_B": FEATURES_A + NEW_FEATURES}
SEEDS = [42, 43, 44, 45, 46]  # fixed before running; 42 = notebook seed
RF_PARAMS = dict(n_estimators=200, max_depth=8, min_samples_leaf=5, class_weight="balanced", n_jobs=-1)
IMPUTER_PARAMS = dict(strategy="median", add_indicator=True, keep_empty_features=True)
RULES_NAME = "RULES_V1"


def make_model(seed: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(**IMPUTER_PARAMS)),
        ("rf", RandomForestClassifier(random_state=seed, **RF_PARAMS)),
    ])


def load_data() -> pd.DataFrame:
    expected = json.loads(DATASET_MANIFEST.read_text())["output_sha256"]
    if sha256(DATASET) != expected:
        raise SystemExit(f"{DATASET} differs from its B4 manifest")
    data = pd.read_csv(DATASET, low_memory=False)
    # Same filter as the notebook (cell 13).
    data = data[data["risk_gt"].notna() & (data["pipeline_ok"] == 1)].copy()
    data["risk_gt"] = data["risk_gt"].astype(int)
    if data.duplicated(KEY).any():
        raise SystemExit("duplicated keys")

    rules = pd.read_csv(RULES_V1)
    merged = data.merge(
        rules[KEY + ["risk_gt", "new_pred"]].rename(columns={"risk_gt": "_gt_rules", "new_pred": "pred_rules_v1"}),
        on=KEY, how="left", validate="one_to_one",
    )
    if merged["pred_rules_v1"].isna().any() or (merged["_gt_rules"] != merged["risk_gt"]).any():
        raise SystemExit("rule predictions do not cover the same frames / GT")
    return merged.drop(columns="_gt_rules").sort_values(KEY).reset_index(drop=True)


def run_folds(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenes = sorted(data["source_event_id"].unique())
    predictions, folds = [], []
    for seed in SEEDS:
        for test_scene in scenes:
            train = data[data["source_event_id"] != test_scene]
            test = data[data["source_event_id"] == test_scene]
            assert set(train["source_event_id"]).isdisjoint(set(test["source_event_id"]))
            for variant, features in VARIANTS.items():
                model = make_model(seed)
                model.fit(train[features], train["risk_gt"])
                proba = model.predict_proba(test[features])[:, list(model.classes_).index(1)]
                pred = model.predict(test[features])
                indicator = model.named_steps["imputer"].indicator_
                folds.append({
                    "seed": seed, "variant": variant, "test_scene": test_scene,
                    "train_rows": len(train), "train_positive": int(train["risk_gt"].sum()),
                    "test_rows": len(test), "test_positive": int(test["risk_gt"].sum()),
                    "missing_indicators": ";".join(np.array(features)[indicator.features_]) if indicator is not None else "",
                })
                predictions.append(pd.DataFrame({
                    "video_id": test["video_id"].to_numpy(),
                    "source_event_id": test_scene,
                    "frame_idx": test["frame_idx"].to_numpy(),
                    "timestamp_s": test["timestamp_s"].to_numpy(),
                    "risk_gt": test["risk_gt"].to_numpy(),
                    "system": variant, "seed": seed,
                    "proba_danger": proba, "pred": pred.astype(int),
                }))
    rules = data[["video_id", "source_event_id", "frame_idx", "timestamp_s", "risk_gt"]].copy()
    rules["system"] = RULES_NAME
    rules["seed"] = -1  # deterministic, no seed
    rules["proba_danger"] = np.nan
    rules["pred"] = data["pred_rules_v1"].astype(int).to_numpy()
    predictions.append(rules)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(folds)


def per_scene_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (system, seed, scene), part in pred.groupby(["system", "seed", "source_event_id"], sort=True):
        row = {"system": system, "seed": seed, "scene": scene}
        row |= frame_metrics(part["risk_gt"], part["pred"])
        row |= event_metrics(part, "pred")
        rows.append(row)
    return pd.DataFrame(rows)


def pooled_metrics(pred: pd.DataFrame, scene_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (system, seed), part in pred.groupby(["system", "seed"], sort=True):
        scenes = scene_table[(scene_table["system"] == system) & (scene_table["seed"] == seed)]
        row = {"system": system, "seed": seed}
        row |= {f"pooled_{k}": v for k, v in frame_metrics(part["risk_gt"], part["pred"]).items()}
        row |= {
            "macro_f1_over_scenes": scenes["f1"].mean(),
            "macro_precision_over_scenes": scenes["precision"].mean(),  # NaN scenes skipped
            "scenes_without_alarms": int(scenes["alarm_frames"].eq(0).sum()),
            "events_detected": int(scenes["events_detected"].sum()),
            "gt_events": int(scenes["gt_events"].sum()),
            "mean_first_alarm_delay_s": scenes["first_alarm_delay_s"].mean(),
            "median_first_alarm_delay_s": scenes["first_alarm_delay_s"].median(),
            "mean_event_coverage": scenes["event_coverage"].mean(),
            "false_alarm_episodes": int(scenes["false_alarm_episodes"].sum()),
            "false_alarm_duration_s": scenes["false_alarm_duration_s"].sum(),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def across_seeds(table: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    numeric = [c for c in table.columns if c not in group + ["seed"] and pd.api.types.is_numeric_dtype(table[c])]
    stats = table.groupby(group)[numeric].agg(["mean", "std", "min", "max"])
    stats.columns = [f"{c}_{s}" for c, s in stats.columns]
    return stats.reset_index()


PAIR_METRICS = ["f1", "precision", "recall", "alarm_frames", "fp", "tp", "events_detected",
                "first_alarm_delay_s", "event_coverage", "false_alarm_episodes", "false_alarm_duration_s"]


def paired_differences(scene_table: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    left = scene_table[scene_table["system"] == a].set_index(["seed", "scene"])[PAIR_METRICS]
    right = scene_table[scene_table["system"] == b].set_index(["seed", "scene"])[PAIR_METRICS]
    diff = (right - left).add_suffix("_diff").reset_index()
    diff.insert(0, "comparison", f"{b}-{a}")
    return diff


def paired_vs_rules(scene_table: pd.DataFrame, system: str) -> pd.DataFrame:
    ml = scene_table[scene_table["system"] == system].groupby("scene")[PAIR_METRICS].mean()
    rules = scene_table[scene_table["system"] == RULES_NAME].set_index("scene")[PAIR_METRICS]
    diff = (ml - rules).add_suffix("_diff").reset_index()
    diff.insert(0, "comparison", f"{system}(seed mean)-{RULES_NAME}")
    return diff


def sign_summary(diff: pd.DataFrame, metric: str, tol: float = 1e-9) -> dict:
    per_scene = diff.groupby("scene")[f"{metric}_diff"].mean()
    return {
        "metric": metric,
        "mean_diff_over_scenes": per_scene.mean(),
        "median_diff_over_scenes": per_scene.median(),
        "scenes_better": int((per_scene > tol).sum()),
        "scenes_worse": int((per_scene < -tol).sum()),
        "scenes_equal": int((per_scene.abs() <= tol).sum()),
        "scenes_undefined": int(per_scene.isna().sum()),
    }


def fmt(frame: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    out = frame.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(digits)
    return out


def write_report(path: Path, manifest, pooled_summary, scene_mean, pair_summary, rules_summary, seed_pooled_diff) -> None:
    keep = ["system", "pooled_precision_mean", "pooled_recall_mean", "pooled_f1_mean", "pooled_f1_std",
            "macro_f1_over_scenes_mean", "pooled_alarm_frames_mean", "events_detected_mean",
            "mean_first_alarm_delay_s_mean", "false_alarm_episodes_mean", "false_alarm_duration_s_mean"]
    frame_cols = ["system", "scene", "precision", "recall", "f1", "tp", "fp", "fn", "tn", "alarm_frames"]
    event_cols = ["system", "scene", "events_detected", "first_alarm_delay_s", "event_coverage",
                  "false_alarm_episodes", "false_alarm_duration_s"]
    text = f"""# RF feature comparison v1 — report (Phase III, pilot)

Created {manifest['created_utc']}; commit {manifest['git']['commit']} (dirty: {manifest['git']['code_dirty']}).
Leave-one-scene-out over {len(manifest['scenes'])} scenes; seeds {SEEDS} (all reported, none selected).
RF params {RF_PARAMS}; imputer {IMPUTER_PARAMS}; threshold 0.5. No tuning.

**Pilot only.** 8 scenes = 8 independent units; frames within a scene are strongly correlated.

## Pooled over folds (mean over seeds; RULES_V1 deterministic)

{markdown_table(fmt(pooled_summary[[c for c in keep if c in pooled_summary.columns]]))}

## Paired differences RF_B − RF_A per scene (seed-averaged), sign counts over 8 scenes

{markdown_table(fmt(pair_summary))}

Pooled F1 difference per seed (RF_B − RF_A): {seed_pooled_diff}

## ML (seed mean) − RULES_V1, per-scene sign counts

{markdown_table(fmt(rules_summary))}

## Frame-level metrics per scene (ML: mean over seeds)

{markdown_table(fmt(scene_mean[frame_cols]))}

## Event-level metrics per scene (ML: mean over seeds)

{markdown_table(fmt(scene_mean[event_cols]))}
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="outputs/ml_experiment/rf_feature_comparison_v1")
    args = parser.parse_args()
    out = Path(args.output_dir)
    if out.exists():
        raise SystemExit(f"{out} already exists; choose another --output-dir")

    inputs = {str(p): sha256(p) for p in (DATASET, RULES_V1)}
    data = load_data()
    predictions, folds = run_folds(data)

    scene_table = per_scene_metrics(predictions)
    pooled = pooled_metrics(predictions, scene_table)
    pooled_summary = across_seeds(pooled, ["system"])
    scene_mean = scene_table.groupby(["system", "scene"])[
        [c for c in scene_table.columns if c not in ("system", "seed", "scene")]
    ].mean().reset_index()

    ab = paired_differences(scene_table, "RF_A", "RF_B")
    pair_summary = pd.DataFrame([sign_summary(ab, m) for m in PAIR_METRICS])
    vs_rules = pd.concat([paired_vs_rules(scene_table, s) for s in VARIANTS], ignore_index=True)
    rules_summary = pd.concat([
        pd.DataFrame([{"comparison": comp, **sign_summary(part, m)} for m in PAIR_METRICS])
        for comp, part in vs_rules.groupby("comparison")
    ], ignore_index=True)
    pf = pooled.set_index(["system", "seed"])["pooled_f1"]
    seed_pooled_diff = {int(s): round(float(pf[("RF_B", s)] - pf[("RF_A", s)]), 4) for s in SEEDS}

    out.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(out / "frame_predictions.csv", index=False)
    folds.to_csv(out / "folds.csv", index=False)
    scene_table.to_csv(out / "metrics_per_scene_per_seed.csv", index=False)
    scene_mean.to_csv(out / "metrics_per_scene_seed_mean.csv", index=False)
    pooled.to_csv(out / "metrics_pooled_per_seed.csv", index=False)
    pooled_summary.to_csv(out / "metrics_pooled_summary.csv", index=False)
    ab.to_csv(out / "paired_diff_B_minus_A.csv", index=False)
    pair_summary.to_csv(out / "paired_diff_B_minus_A_summary.csv", index=False)
    vs_rules.to_csv(out / "paired_diff_ml_minus_rules.csv", index=False)
    rules_summary.to_csv(out / "paired_diff_ml_minus_rules_summary.csv", index=False)

    unchanged = all(sha256(Path(p)) == h for p, h in inputs.items())
    manifest = {
        "experiment": "rf_feature_comparison_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "git": git_state(),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "inputs_sha256": inputs,
        "inputs_unchanged_after_run": unchanged,
        "rows": len(data),
        "scenes": sorted(data["source_event_id"].unique()),
        "validation": "leave-one-scene-out; identical folds for all variants and seeds",
        "variants": VARIANTS,
        "label": "risk_gt (frame_labels_csv)",
        "excluded_from_X": ["video_id", "source_event_id", "frame_idx", "timestamp_s", "pipeline_ok",
                            "rule_pred", "risk_gt", "source_file", "pred_rules_v1"],
        "seeds": SEEDS,
        "rf_params": RF_PARAMS,
        "imputer_params": IMPUTER_PARAMS,
        "decision_threshold": 0.5,
        "reference": {"system": RULES_NAME, "file": str(RULES_V1), "column": "new_pred",
                      "params": {"arrival_time_s": 4.0, "moving_speed_kmh": 2.0}},
        "event_definitions": "traffic_safety/event_metrics.py docstring",
        "tuning": "none",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    write_report(out / "report.md", manifest, pooled_summary, scene_mean, pair_summary, rules_summary, seed_pooled_diff)
    print(f"inputs unchanged: {unchanged}; saved to {out}")


if __name__ == "__main__":
    main()
