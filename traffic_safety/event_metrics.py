"""Frame- and event-level metrics for binary per-frame alarms of ONE recording.

Definitions (used for rules and ML alike):
- GT event: maximal run of consecutive frames with gt == 1.
- Alarm episode: maximal run of consecutive frames with pred == 1
  (consecutive = frame_idx differs by 1).
- Event detected: at least one alarm frame inside the GT event.
- Detection delay: timestamp of the first alarm frame inside the GT event
  minus timestamp of the event's first frame (>= 0; an alarm that started
  before the event and continues into it gives 0).
- False-alarm episode: alarm episode containing no gt == 1 frame.
  Duration = number of frames * frame period (median timestamp step).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def frame_metrics(gt, pred) -> dict:
    gt = np.asarray(gt, dtype=int)
    pred = np.asarray(pred, dtype=int)
    tp = int(((gt == 1) & (pred == 1)).sum())
    fp = int(((gt == 0) & (pred == 1)).sum())
    fn = int(((gt == 1) & (pred == 0)).sum())
    tn = int(((gt == 0) & (pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else np.nan  # undefined without alarms
    recall = tp / (tp + fn) if tp + fn else np.nan
    if np.isnan(precision) or np.isnan(recall) or precision + recall == 0:
        f1 = 0.0 if (tp + fn) else np.nan
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "alarm_frames": tp + fp,
    }


def _runs(frame_idx: np.ndarray, mask: np.ndarray) -> list[tuple[int, int]]:
    """(start_pos, end_pos) inclusive positions of runs of True over consecutive frames."""
    runs = []
    start = None
    for i, flag in enumerate(mask):
        contiguous = i > 0 and frame_idx[i] == frame_idx[i - 1] + 1
        if flag and (start is None or not contiguous):
            if start is not None:
                runs.append((start, i - 1))
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def event_metrics(frames: pd.DataFrame, pred_col: str, gt_col: str = "risk_gt") -> dict:
    """frames: one recording with frame_idx, timestamp_s, gt and prediction columns."""
    data = frames.sort_values("frame_idx")
    if data["frame_idx"].duplicated().any():
        raise ValueError("duplicated frame_idx")
    idx = data["frame_idx"].to_numpy(dtype=int)
    ts = data["timestamp_s"].to_numpy(dtype=float)
    gt = data[gt_col].to_numpy(dtype=int) == 1
    pred = data[pred_col].to_numpy(dtype=int) == 1
    period = float(np.median(np.diff(ts))) if len(ts) > 1 else np.nan

    events = _runs(idx, gt)
    detected, delays, coverage = 0, [], []
    for start, end in events:
        hits = np.flatnonzero(pred[start:end + 1])
        coverage.append(len(hits) / (end - start + 1))
        if len(hits):
            detected += 1
            delays.append(ts[start + hits[0]] - ts[start])

    episodes = _runs(idx, pred)
    fa = [(s, e) for s, e in episodes if not gt[s:e + 1].any()]
    fa_frames = sum(e - s + 1 for s, e in fa)
    return {
        "gt_events": len(events),
        "events_detected": detected,
        "first_alarm_delay_s": delays[0] if len(events) == 1 and delays else (
            float(np.mean(delays)) if delays else np.nan
        ),
        "event_coverage": float(np.mean(coverage)) if coverage else np.nan,
        "alarm_episodes": len(episodes),
        "false_alarm_episodes": len(fa),
        "false_alarm_duration_s": fa_frames * period if len(ts) > 1 else np.nan,
    }
