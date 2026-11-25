import math
import unittest

import pandas as pd

from traffic_safety.event_metrics import event_metrics, frame_metrics


def recording(gt, pred, period=0.1, start=0):
    n = len(gt)
    return pd.DataFrame({
        "frame_idx": range(start, start + n),
        "timestamp_s": [i * period for i in range(n)],
        "risk_gt": gt,
        "pred": pred,
    })


class FrameMetricsTest(unittest.TestCase):
    def test_counts_and_scores(self):
        m = frame_metrics([1, 1, 0, 0, 1], [1, 0, 1, 0, 1])
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["tn"]), (2, 1, 1, 1))
        self.assertAlmostEqual(m["precision"], 2 / 3)
        self.assertAlmostEqual(m["recall"], 2 / 3)
        self.assertAlmostEqual(m["f1"], 2 / 3)
        self.assertEqual(m["alarm_frames"], 3)

    def test_no_alarms_precision_undefined_f1_zero(self):
        m = frame_metrics([1, 0], [0, 0])
        self.assertTrue(math.isnan(m["precision"]))
        self.assertEqual(m["f1"], 0.0)


class EventMetricsTest(unittest.TestCase):
    def test_detection_delay_and_false_alarm_episodes(self):
        gt = [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        pd_ = [1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1]
        m = event_metrics(recording(gt, pd_), "pred")
        self.assertEqual(m["gt_events"], 1)
        self.assertEqual(m["events_detected"], 1)
        self.assertAlmostEqual(m["first_alarm_delay_s"], 0.2)
        self.assertAlmostEqual(m["event_coverage"], 0.5)
        self.assertEqual(m["alarm_episodes"], 4)
        # episodes [0,1], [9], [11] are outside GT; [5..7] overlaps GT.
        self.assertEqual(m["false_alarm_episodes"], 3)
        self.assertAlmostEqual(m["false_alarm_duration_s"], 0.4)

    def test_alarm_starting_before_event_gives_zero_delay(self):
        m = event_metrics(recording([0, 0, 1, 1], [0, 1, 1, 0]), "pred")
        self.assertEqual(m["first_alarm_delay_s"], 0.0)
        self.assertEqual(m["false_alarm_episodes"], 0)

    def test_missed_event(self):
        m = event_metrics(recording([0, 1, 1, 0], [0, 0, 0, 0]), "pred")
        self.assertEqual(m["events_detected"], 0)
        self.assertTrue(math.isnan(m["first_alarm_delay_s"]))
        self.assertEqual(m["event_coverage"], 0.0)

    def test_frame_gap_splits_episodes(self):
        frames = pd.DataFrame({
            "frame_idx": [0, 1, 3, 4],
            "timestamp_s": [0.0, 0.1, 0.3, 0.4],
            "risk_gt": [0, 0, 0, 0],
            "pred": [1, 1, 1, 1],
        })
        self.assertEqual(event_metrics(frames, "pred")["false_alarm_episodes"], 2)

    def test_unsorted_input(self):
        rec = recording([0, 1, 1, 0], [0, 0, 1, 0]).iloc[::-1]
        self.assertAlmostEqual(event_metrics(rec, "pred")["first_alarm_delay_s"], 0.1)


if __name__ == "__main__":
    unittest.main()
