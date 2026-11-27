#!/usr/bin/env python3
"""Phase III, diagnostic: decision threshold of the Random Forest chosen on training scenes only.

Pre-specified procedure (fixed before running):
  outer: leave-one-scene-out over the 8 pilot scenes (as rf_feature_comparison_v1);
  inner: leave-one-scene-out over the 7 training scenes -> out-of-fold
         probabilities for every training frame (each inner model = same
         Pipeline, same hyper-parameters, same seed, fitted on 6 scenes);
  rule:  threshold t in {0.01, 0.02, ..., 0.99} maximising pooled F1 of
         (proba >= t) on the inner out-of-fold predictions; ties -> highest t;
  test:  outer model fitted on all 7 training scenes (identical to
         rf_feature_comparison_v1), DANGER = proba >= t on the test scene.
The test scene's labels/predictions are never used for choosing t.

Diagnostics only (NOT used for any choice): per-scene oracle thresholds
on test scenes and on inner validation scenes, threshold curves.
Feature sets, model, hyper-parameters and seeds are unchanged.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from traffic_safety.event_metrics import event_metrics, frame_metrics  # noqa: E402
from compute_motion_features import git_state, markdown_table, sha256  # noqa: E402
import rf_feature_comparison as rfc  # noqa: E402
from rf_feature_comparison import (  # noqa: E402
    IMPUTER_PARAMS, RF_PARAMS, RULES_NAME, SEEDS, VARIANTS,
    load_data, make_model,
)

PHASE3 = Path("outputs/ml_experiment/rf_feature_comparison_v1/frame_predictions.csv")
GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)


def f1_curve(gt: np.ndarray, proba: np.ndarray) -> np.ndarray:
    gt = gt.astype(bool)
    out = np.empty(len(GRID))
    for i, t in enumerate(GRID):
        pred = proba >= t
        tp = np.sum(pred & gt)
        fp = np.sum(pred & ~gt)
        fn = np.sum(~pred & gt)
        out[i] = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return out


def best_threshold(gt, proba) -> tuple[float, float]:
    curve = f1_curve(np.asarray(gt), np.asarray(proba))
    best = curve.max()
    idx = np.flatnonzero(np.isclose(curve, best, rtol=0, atol=1e-12))[-1]  # ties -> highest t
    return float(GRID[idx]), float(best)


def proba_of(model, frame, features):
    return model.predict_proba(frame[features])[:, list(model.classes_).index(1)]


def run(data: pd.DataFrame):
    scenes = sorted(data["source_event_id"].unique())
    preds, folds, inner_oracle = [], [], []
    for seed in SEEDS:
        for test_scene in scenes:
            train = data[data["source_event_id"] != test_scene]
            test = data[data["source_event_id"] == test_scene]
            train_scenes = sorted(train["source_event_id"].unique())
            assert test_scene not in train_scenes
            for variant, features in VARIANTS.items():
                # ---- inner LOSO on the 7 training scenes only ----
                oof = pd.Series(np.nan, index=train.index)
                for val_scene in train_scenes:
                    inner_train = train[train["source_event_id"] != val_scene]
                    inner_val = train[train["source_event_id"] == val_scene]
                    model = make_model(seed).fit(inner_train[features], inner_train["risk_gt"])
                    oof.loc[inner_val.index] = proba_of(model, inner_val, features)
                    t_s, f1_s = best_threshold(inner_val["risk_gt"], oof.loc[inner_val.index])
                    inner_oracle.append({"seed": seed, "variant": variant, "outer_test_scene": test_scene,
                                         "inner_val_scene": val_scene, "oracle_threshold": t_s, "oracle_f1": f1_s})
                assert oof.notna().all()
                threshold, inner_f1 = best_threshold(train["risk_gt"], oof)
                inner_f1_at_05 = frame_metrics(train["risk_gt"], (oof >= 0.5).astype(int))["f1"]

                # ---- outer model: same as rf_feature_comparison_v1 ----
                model = make_model(seed).fit(train[features], train["risk_gt"])
                proba = proba_of(model, test, features)
                pred05 = model.predict(test[features]).astype(int)
                oracle_t, oracle_f1 = best_threshold(test["risk_gt"], proba)  # diagnostic only
                folds.append({"seed": seed, "variant": variant, "test_scene": test_scene,
                              "selected_threshold": threshold, "inner_oof_f1_selected": inner_f1,
                              "inner_oof_f1_at_0.5": inner_f1_at_05,
                              "test_oracle_threshold_DIAGNOSTIC": oracle_t, "test_oracle_f1_DIAGNOSTIC": oracle_f1})
                base = {"video_id": test["video_id"].to_numpy(), "source_event_id": test_scene,
                        "frame_idx": test["frame_idx"].to_numpy(), "timestamp_s": test["timestamp_s"].to_numpy(),
                        "risk_gt": test["risk_gt"].to_numpy(), "seed": seed, "proba_danger": proba}
                preds.append(pd.DataFrame(base | {"system": f"{variant}_t0.5", "threshold": 0.5, "pred": pred05}))
                preds.append(pd.DataFrame(base | {"system": f"{variant}_tsel", "threshold": threshold,
                                                  "pred": (proba >= threshold).astype(int)}))
    rules = data[["video_id", "source_event_id", "frame_idx", "timestamp_s", "risk_gt"]].copy()
    rules = rules.assign(seed=-1, proba_danger=np.nan, system=RULES_NAME, threshold=np.nan,
                         pred=data["pred_rules_v1"].astype(int).to_numpy())
    preds.append(rules)
    return pd.concat(preds, ignore_index=True), pd.DataFrame(folds), pd.DataFrame(inner_oracle)


def scene_metrics(pred):
    rows = []
    for (system, seed, scene), part in pred.groupby(["system", "seed", "source_event_id"], sort=True):
        rows.append({"system": system, "seed": seed, "scene": scene, "threshold": part["threshold"].iloc[0]}
                    | frame_metrics(part["risk_gt"], part["pred"]) | event_metrics(part, "pred"))
    return pd.DataFrame(rows)


def pooled(pred, scenes):
    rows = []
    for (system, seed), part in pred.groupby(["system", "seed"], sort=True):
        s = scenes[(scenes["system"] == system) & (scenes["seed"] == seed)]
        rows.append({"system": system, "seed": seed}
                    | {f"pooled_{k}": v for k, v in frame_metrics(part["risk_gt"], part["pred"]).items()}
                    | {"macro_f1": s["f1"].mean(), "scene_f1_sd": s["f1"].std(),
                       "events_detected": int(s["events_detected"].sum()),
                       "mean_first_alarm_delay_s": s["first_alarm_delay_s"].mean(),
                       "mean_event_coverage": s["event_coverage"].mean(),
                       "false_alarm_episodes": int(s["false_alarm_episodes"].sum()),
                       "false_alarm_duration_s": s["false_alarm_duration_s"].sum()})
    return pd.DataFrame(rows)


def seed_summary(table):
    cols = [c for c in table.columns if c not in ("system", "seed") and pd.api.types.is_numeric_dtype(table[c])]
    s = table.groupby("system")[cols].agg(["mean", "std"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    return s.reset_index()


METRICS = ["f1", "precision", "recall", "alarm_frames", "fp", "tp", "false_alarm_episodes",
           "false_alarm_duration_s", "first_alarm_delay_s", "event_coverage"]


def paired(scenes, a, b):
    """Per-scene seed-averaged difference b - a (rules have one 'seed')."""
    ma = scenes[scenes["system"] == a].groupby("scene")[METRICS].mean()
    mb = scenes[scenes["system"] == b].groupby("scene")[METRICS].mean()
    d = mb - ma
    rows = []
    for m in METRICS:
        col = d[m]
        rows.append({"comparison": f"{b} - {a}", "metric": m, "mean_diff": col.mean(), "median_diff": col.median(),
                     "scenes_higher": int((col > 1e-9).sum()), "scenes_lower": int((col < -1e-9).sum()),
                     "scenes_equal": int((col.abs() <= 1e-9).sum()), "scenes_undefined": int(col.isna().sum())})
    return pd.DataFrame(rows), d.add_suffix("_diff").reset_index().assign(comparison=f"{b} - {a}")


def threshold_curves(pred):
    """Diagnostic: pooled metrics on outer TEST predictions for every grid threshold (not used for selection)."""
    rows = []
    ml = pred[pred["system"].str.endswith("_t0.5")]
    for (system, seed), part in ml.groupby(["system", "seed"]):
        gt = part["risk_gt"].to_numpy()
        for t in GRID:
            rows.append({"variant": system.replace("_t0.5", ""), "seed": seed, "threshold": t}
                        | frame_metrics(gt, (part["proba_danger"].to_numpy() >= t).astype(int)))
    return pd.DataFrame(rows)


def fmt(df, d=3):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].round(d)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default="outputs/ml_experiment/rf_threshold_selection_v1")
    ap.add_argument("--dataset", default=str(rfc.DATASET))
    ap.add_argument("--dataset-manifest", default=str(rfc.DATASET_MANIFEST))
    ap.add_argument("--rules-v1", default=str(rfc.RULES_V1))
    ap.add_argument("--rules-column", default=rfc.RULES_COLUMN)
    ap.add_argument("--phase3-predictions", default=str(PHASE3),
                    help="frame_predictions.csv of the matching rf_feature_comparison run (t=0.5 reproducibility gate)")
    args = ap.parse_args()
    rfc.configure_inputs(args)
    phase3 = Path(args.phase3_predictions)
    out = Path(args.output_dir)
    if out.exists():
        raise SystemExit(f"{out} already exists; choose another --output-dir")

    inputs = {str(p): sha256(p) for p in (rfc.DATASET, rfc.RULES_V1, phase3)}
    data = load_data()
    pred, folds, inner_oracle = run(data)

    # Reproducibility gate: t=0.5 predictions must equal rf_feature_comparison_v1.
    old = pd.read_csv(phase3)
    old = old[old["system"] != RULES_NAME]
    new05 = pred[pred["system"].str.endswith("_t0.5")].assign(system=lambda d: d["system"].str.replace("_t0.5", "", regex=False))
    chk = old.merge(new05, on=["system", "seed", "video_id", "frame_idx"], suffixes=("_old", "_new"), validate="one_to_one")
    reproduced = bool(len(chk) == len(old) == len(new05) and (chk["pred_old"] == chk["pred_new"]).all()
                      and np.allclose(chk["proba_danger_old"], chk["proba_danger_new"], rtol=0, atol=1e-12))
    if not reproduced:
        raise SystemExit("t=0.5 predictions differ from rf_feature_comparison_v1")

    scenes = scene_metrics(pred)
    pool = pooled(pred, scenes)
    summary = seed_summary(pool)
    comparisons = [("RF_A_t0.5", "RF_A_tsel"), ("RF_B_t0.5", "RF_B_tsel"), ("RF_A_tsel", "RF_B_tsel"),
                   (RULES_NAME, "RF_A_tsel"), (RULES_NAME, "RF_B_tsel")]
    pair_summ, pair_detail = zip(*[paired(scenes, a, b) for a, b in comparisons])
    pair_summ = pd.concat(pair_summ, ignore_index=True)
    pair_detail = pd.concat(pair_detail, ignore_index=True)

    thr = folds.groupby("variant").agg(
        selected_mean=("selected_threshold", "mean"), selected_sd=("selected_threshold", "std"),
        selected_min=("selected_threshold", "min"), selected_max=("selected_threshold", "max"),
        oracle_test_mean=("test_oracle_threshold_DIAGNOSTIC", "mean"),
        oracle_test_sd=("test_oracle_threshold_DIAGNOSTIC", "std"),
        oracle_test_min=("test_oracle_threshold_DIAGNOSTIC", "min"),
        oracle_test_max=("test_oracle_threshold_DIAGNOSTIC", "max")).reset_index()
    inner_thr = inner_oracle.groupby("variant")["oracle_threshold"].describe().reset_index()
    per_fold_thr = folds.pivot_table(index="test_scene", columns="variant",
                                     values=["selected_threshold", "test_oracle_threshold_DIAGNOSTIC"],
                                     aggfunc=["mean", "min", "max"])
    per_fold_thr.columns = ["_".join(map(str, c)) for c in per_fold_thr.columns]
    per_fold_thr = per_fold_thr.reset_index()
    scene_mean = scenes.groupby(["system", "scene"])[["threshold"] + METRICS + ["tn", "fn"]].mean().reset_index()
    scene_sd = scenes.groupby(["system", "scene"])["f1"].std().rename("f1_sd_over_seeds").reset_index()
    scene_mean = scene_mean.merge(scene_sd, on=["system", "scene"])

    out.mkdir(parents=True, exist_ok=False)
    pred.to_csv(out / "frame_predictions.csv", index=False)
    folds.to_csv(out / "thresholds_per_fold.csv", index=False)
    inner_oracle.to_csv(out / "inner_scene_oracle_thresholds_DIAGNOSTIC.csv", index=False)
    scenes.to_csv(out / "metrics_per_scene_per_seed.csv", index=False)
    scene_mean.to_csv(out / "metrics_per_scene_seed_mean.csv", index=False)
    pool.to_csv(out / "metrics_pooled_per_seed.csv", index=False)
    summary.to_csv(out / "metrics_pooled_summary.csv", index=False)
    pair_summ.to_csv(out / "paired_summary.csv", index=False)
    pair_detail.to_csv(out / "paired_per_scene.csv", index=False)
    thr.to_csv(out / "threshold_stability.csv", index=False)
    per_fold_thr.to_csv(out / "thresholds_by_test_scene.csv", index=False)
    threshold_curves(pred).to_csv(out / "test_threshold_curves_DIAGNOSTIC.csv", index=False)

    unchanged = all(sha256(Path(p)) == h for p, h in inputs.items())
    manifest = {
        "experiment": "rf_threshold_selection_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "git": git_state(),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "inputs_sha256": inputs, "inputs_unchanged_after_run": unchanged,
        "reproduces_rf_feature_comparison_v1_at_0.5": reproduced,
        "variants": VARIANTS, "seeds": SEEDS, "rf_params": RF_PARAMS, "imputer_params": IMPUTER_PARAMS,
        "threshold_rule": "max pooled F1 of proba>=t on inner leave-one-scene-out OOF predictions of the 7 training scenes; grid 0.01..0.99 step 0.01; ties -> highest t",
        "diagnostic_only": ["test_oracle_threshold_DIAGNOSTIC", "inner_scene_oracle_thresholds_DIAGNOSTIC.csv",
                            "test_threshold_curves_DIAGNOSTIC.csv"],
        "baseline_t0.5": "model.predict (argmax), identical to rf_feature_comparison_v1",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    keep = ["system", "pooled_precision_mean", "pooled_recall_mean", "pooled_f1_mean", "pooled_f1_std",
            "macro_f1_mean", "pooled_alarm_frames_mean", "false_alarm_episodes_mean", "false_alarm_duration_s_mean",
            "mean_first_alarm_delay_s_mean", "mean_event_coverage_mean"]
    report = f"""# RF threshold selection v1 (Phase III, diagnostic)

{manifest['threshold_rule']}. Seeds {SEEDS}. Reproduces t=0.5 results of rf_feature_comparison_v1: {reproduced}.

## Pooled (mean/sd over seeds)

{markdown_table(fmt(summary[keep]))}

## Threshold stability

{markdown_table(fmt(thr))}

Inner validation scenes, per-scene oracle thresholds (diagnostic):

{markdown_table(fmt(inner_thr))}

## Thresholds by outer test scene (selected = training only; oracle = diagnostic)

{markdown_table(fmt(per_fold_thr))}

## Paired per-scene differences (seed-averaged)

{markdown_table(fmt(pair_summ))}

## Per scene (mean over seeds)

{markdown_table(fmt(scene_mean))}
"""
    (out / "report.md").write_text(report, encoding="utf-8")
    print(f"reproduced={reproduced} inputs_unchanged={unchanged} -> {out}")


if __name__ == "__main__":
    main()
