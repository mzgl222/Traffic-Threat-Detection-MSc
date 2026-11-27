import math
import random
import unittest
from collections import deque
from statistics import median

from traffic_safety.speed import SpeedEstimator

FPS = 30.0
DT = 1.0 / FPS
KMH = 3.6


def regression(**kwargs):
    return SpeedEstimator(method="regression", window_s=0.5, **kwargs)


def feed(est, points, track_id=1):
    """points: iterable of (t, x, y). Returns list of outputs."""
    return [est.update(track_id, t, x, y) for t, x, y in points]


def times(duration_s, start=0.0):
    n = int(round(duration_s * FPS))
    return [start + i * DT for i in range(n)]


def valid(values):
    return [v for v in values if v is not None]


class ReferenceMedianStep:
    """Frozen copy of the original algorithm (before P1) for regression tests."""

    def __init__(self, history_length=15, speed_history_length=21, max_speed_kmh=180.0):
        self.pos = {}
        self.spd = {}
        self.hl, self.shl, self.max = history_length, speed_history_length, max_speed_kmh

    def update(self, track_id, t, x, y):
        h = self.pos.setdefault(track_id, deque(maxlen=self.hl))
        s = self.spd.setdefault(track_id, deque(maxlen=self.shl))
        h.append((t, x, y))
        if len(h) < 2:
            return None
        (t0, x0, y0), (t1, x1, y1) = h[-2], h[-1]
        dt = t1 - t0
        if dt <= 0:
            return None
        v = math.hypot(x1 - x0, y1 - y0) / dt * 3.6
        if not (v < 0 or v > self.max):
            s.append(v)
        return None if len(s) < 3 else float(median(s))


class DefaultMethodRegressionTest(unittest.TestCase):
    def test_default_is_median_step(self):
        self.assertEqual(SpeedEstimator().method, "median_step")

    def test_default_reproduces_original_algorithm_exactly(self):
        rng = random.Random(0)
        for seed in range(20):
            rng.seed(seed)
            est = SpeedEstimator(history_length=15, speed_history_length=21)
            ref = ReferenceMedianStep(15, 21)
            t = 0.0
            for _ in range(400):
                track = rng.choice([1, 2, 3])
                t += rng.choice([DT, DT, DT, 0.0, 2 * DT, -DT])  # repeated / backwards timestamps too
                x = rng.uniform(-5, 5) if rng.random() < 0.02 else rng.gauss(0, 0.05)  # occasional big jump
                y = rng.gauss(0, 0.05)
                self.assertEqual(est.update(track, t, x, y), ref.update(track, t, x, y))


class RegressionBasicsTest(unittest.TestCase):
    def test_parameter_validation(self):
        for kwargs in ({"method": "foo"}, {"method": "regression", "window_s": 0},
                       {"method": "regression", "window_s": float("nan")},
                       {"method": "regression", "min_samples": 1},
                       {"method": "regression", "min_coverage": 0}):
            with self.assertRaises(ValueError):
                SpeedEstimator(**kwargs)

    def test_first_observations_and_coverage(self):
        est = regression()
        out = feed(est, [(t, 0.0, 0.0) for t in times(0.5)])
        # needs >= 5 observations AND span >= 0.3 s -> first value at index 9 (t = 0.3 s)
        self.assertTrue(all(v is None for v in out[:9]))
        self.assertIsNotNone(out[9])

    def test_too_few_samples_in_window(self):
        est = regression()
        # 4 observations spread over 0.45 s: span ok, samples not
        out = feed(est, [(t, 0.0, 0.0) for t in (0.0, 0.15, 0.3, 0.45)])
        self.assertTrue(all(v is None for v in out))

    def test_constant_speed_exact(self):
        v = 30 / KMH
        est = regression()
        out = valid(feed(est, [(t, v * t * 0.6, v * t * 0.8) for t in times(3)]))
        for s in out:
            self.assertAlmostEqual(s, 30.0, places=6)

    def test_uses_real_timestamps_not_frame_count(self):
        v = 10 / KMH
        est = regression()
        # 15 fps instead of 30
        out = valid(feed(est, [(i / 15, v * i / 15, 0.0) for i in range(45)]))
        self.assertTrue(out)
        for s in out:
            self.assertAlmostEqual(s, 10.0, places=6)


class StationaryTest(unittest.TestCase):
    def test_stationary_white_jitter(self):
        rng = random.Random(1)
        pts = [(t, rng.gauss(0, 0.01), rng.gauss(0, 0.01)) for t in times(20)]
        reg = sorted(valid(feed(regression(), pts)))
        med = valid(feed(SpeedEstimator(), pts))
        p99 = reg[int(0.99 * len(reg))]
        self.assertLess(p99, 0.5)  # theory: ~0.19 km/h for sigma = 1 cm
        # the original method is biased by the same jitter
        self.assertGreater(median(med), 3 * median(reg))
        self.assertGreater(median(med), 1.0)

    def test_stationary_large_jitter_stays_below_threshold(self):
        rng = random.Random(2)
        pts = [(t, rng.gauss(0, 0.03), rng.gauss(0, 0.03)) for t in times(20)]
        reg = sorted(valid(feed(regression(), pts)))
        self.assertLess(reg[int(0.99 * len(reg))], 2.0)

    def test_stationary_slow_oscillation_is_bounded(self):
        # 5 cm, 1 Hz back-and-forth (e.g. bounding-box bottom oscillation)
        a, f = 0.05, 1.0
        peak_kmh = 2 * math.pi * f * a * KMH  # 1.13 km/h, max true |dx/dt|
        out = valid(feed(regression(), [(t, a * math.sin(2 * math.pi * f * t), 0.0) for t in times(10)]))
        # OLS slope is a positively weighted average of the derivative -> never above its peak
        self.assertLessEqual(max(out), peak_kmh + 1e-6)
        self.assertLess(max(out), 2.0)


class MotionTest(unittest.TestCase):
    def test_slow_motion_both_sides_of_threshold(self):
        rng = random.Random(3)
        for v_kmh in (1.5, 3.0):
            v = v_kmh / KMH
            pts = [(t, v * t + rng.gauss(0, 0.005), rng.gauss(0, 0.005)) for t in times(10)]
            out = valid(feed(regression(), pts))
            within = sum(abs(s - v_kmh) <= 0.3 for s in out) / len(out)
            right_side = sum((s > 2.0) == (v_kmh > 2.0) for s in out) / len(out)
            self.assertGreater(within, 0.99)
            self.assertGreater(right_side, 0.99)

    def test_acceleration_lags_by_half_window(self):
        a = 2.0  # m/s^2 from standstill
        pts = [(t, 0.5 * a * t * t, 0.0) for t in times(3)]
        out = feed(regression(), pts)
        for (t, _, _), s in zip(pts, out):
            if s is not None and t >= 0.5:
                # least-squares slope of a parabola = derivative at the window centre (t - 0.25 s)
                self.assertAlmostEqual(s, a * (t - 0.25) * KMH, places=6)

    def test_braking_overestimates_by_half_window(self):
        v0, a = 10.0, 2.0  # m/s, m/s^2 (still moving during the whole test)
        pts = [(t, v0 * t - 0.5 * a * t * t, 0.0) for t in times(3)]
        out = feed(regression(), pts)
        for (t, _, _), s in zip(pts, out):
            if s is not None and t >= 0.5:
                true_now = (v0 - a * t) * KMH
                self.assertAlmostEqual(s, true_now + a * 0.25 * KMH, places=6)
                self.assertGreater(s, true_now)

    def test_sudden_stop(self):
        v = 10 / KMH
        t_stop = 2.0
        pts = [(t, v * min(t, t_stop), 0.0) for t in times(4)]
        out = feed(regression(), pts)
        after = [(t, s) for (t, _, _), s in zip(pts, out) if t > t_stop]
        first_below = next(t for t, s in after if s < 2.0)
        self.assertLessEqual(first_below - t_stop, 0.4)
        # once the whole window is after the stop the speed is zero
        for t, s in after:
            if t - t_stop > 0.5 + DT:
                self.assertAlmostEqual(s, 0.0, places=9)

    def test_right_angle_turn(self):
        v = 10 / KMH
        t_turn = 2.0
        pts = []
        for t in times(4):
            if t <= t_turn:
                pts.append((t, v * t, 0.0))
            else:
                pts.append((t, v * t_turn, v * (t - t_turn)))
        out = valid(feed(regression(), pts))
        self.assertLessEqual(max(out), 10.0 + 1e-6)
        # weighted mean of two orthogonal unit vectors is at least sqrt(2)/2 long
        self.assertGreaterEqual(min(out), 10.0 * math.sqrt(2) / 2 - 1e-6)

    def test_reversal_gives_transient_dip(self):
        v = 10 / KMH
        t_rev = 2.0
        pts = [(t, v * t if t <= t_rev else v * (2 * t_rev - t), 0.0) for t in times(4)]
        out = feed(regression(), pts)
        during = [s for (t, _, _), s in zip(pts, out) if t_rev < t <= t_rev + 0.5]
        late = [s for (t, _, _), s in zip(pts, out) if t > t_rev + 0.5 + DT]
        self.assertLess(min(during), 2.0)  # documented limitation: vectors cancel
        for s in late:
            self.assertAlmostEqual(s, 10.0, places=6)


class IrregularDataTest(unittest.TestCase):
    def test_missing_detections_and_timestamp_jitter(self):
        rng = random.Random(4)
        v = 20 / KMH
        pts = [(t + rng.uniform(-0.003, 0.003), v * (t), 0.0) for t in times(10) if rng.random() > 0.3]
        # positions follow the jittered timestamp exactly -> speed must stay exact
        pts = [(t, v * t, 0.0) for t, _, _ in pts]
        out = valid(feed(regression(), pts))
        self.assertGreater(len(out), 0.8 * len(pts))
        for s in out:
            self.assertAlmostEqual(s, 20.0, places=6)

    def test_long_gap_does_not_mix_before_and_after(self):
        v = 10 / KMH
        before = [(t, v * t, 0.0) for t in times(1)]
        # 2 s without detections, object re-appears 50 m further and stands still
        after = [(t, 100.0, 0.0) for t in times(1, start=3.0)]
        est = regression()
        feed(est, before)
        out = feed(est, after)
        self.assertTrue(all(s is None for s in out[:9]))  # window must refill (>= 0.3 s)
        for s in valid(out):
            self.assertAlmostEqual(s, 0.0, places=9)

    def test_short_track_never_gets_speed(self):
        out = feed(regression(), [(t, 0.1 * t, 0.0) for t in times(0.25)])
        self.assertTrue(all(s is None for s in out))


class InvalidInputTest(unittest.TestCase):
    def _reference_outputs(self, pts):
        return feed(regression(), pts)

    def test_invalid_observations_are_ignored(self):
        v = 10 / KMH
        clean = [(t, v * t, 0.0) for t in times(2)]
        est = regression()
        outputs = []
        for i, (t, x, y) in enumerate(clean):
            if i == 20:
                self.assertIsNone(est.update(1, clean[i - 1][0], 99.0, 99.0))  # repeated timestamp
                self.assertIsNone(est.update(1, clean[i - 5][0], 99.0, 99.0))  # backwards
                self.assertIsNone(est.update(1, float("nan"), 99.0, 99.0))
                self.assertIsNone(est.update(1, float("inf"), 99.0, 99.0))
                self.assertIsNone(est.update(1, t, float("nan"), 0.0))
                self.assertIsNone(est.update(1, None, 0.0, 0.0))
            outputs.append(est.update(1, t, x, y))
        self.assertEqual(outputs, self._reference_outputs(clean))


class TrackIndependenceTest(unittest.TestCase):
    def test_tracks_are_independent(self):
        v1, v2 = 10 / KMH, 30 / KMH
        est = regression()
        out1, out2 = [], []
        for t in times(2):
            out1.append(est.update(1, t, v1 * t, 0.0))
            out2.append(est.update(2, t, 0.0, v2 * t))
        for s in valid(out1):
            self.assertAlmostEqual(s, 10.0, places=6)
        for s in valid(out2):
            self.assertAlmostEqual(s, 30.0, places=6)

    def test_remove_track(self):
        est = regression()
        feed(est, [(t, t, 0.0) for t in times(1)], track_id=7)
        feed(est, [(t, t, 0.0) for t in times(1)], track_id=8)
        est.remove_track(7)
        self.assertNotIn(7, est.window_history)
        self.assertIn(8, est.window_history)
        self.assertIsNone(est.update(7, 1.1, 0.0, 0.0))
        self.assertIsNotNone(est.update(8, 1.1, 1.1, 0.0))


class CausalityTest(unittest.TestCase):
    def test_outputs_do_not_depend_on_future_observations(self):
        rng = random.Random(5)
        pts, x, y = [], 0.0, 0.0
        for t in times(6):
            if rng.random() < 0.15:
                continue  # missing detection
            x += rng.uniform(0, 0.3) / FPS * 10
            y += rng.gauss(0, 0.01)
            pts.append((t, x, y))
        full = feed(regression(), pts)
        for cut in (10, 50, 100, len(pts) - 1):
            self.assertEqual(feed(regression(), pts[:cut]), full[:cut])

    def test_value_equals_fit_on_past_window_only(self):
        rng = random.Random(6)
        pts = [(t, rng.gauss(0, 0.02) + 0.1 * t, rng.gauss(0, 0.02)) for t in times(3)]
        out = feed(regression(), pts)
        for i in (30, 60, 89):
            t_i = pts[i][0]
            win = [p for p in pts[: i + 1] if t_i - p[0] <= 0.5 + 1e-9]
            mt = sum(p[0] for p in win) / len(win)
            mx = sum(p[1] for p in win) / len(win)
            my = sum(p[2] for p in win) / len(win)
            stt = sum((p[0] - mt) ** 2 for p in win)
            vx = sum((p[0] - mt) * (p[1] - mx) for p in win) / stt
            vy = sum((p[0] - mt) * (p[2] - my) for p in win) / stt
            self.assertAlmostEqual(out[i], math.hypot(vx, vy) * KMH, places=9)


if __name__ == "__main__":
    unittest.main()


class FactoryTest(unittest.TestCase):
    TRACKING = {"history_length": 15, "speed_history_length": 21}

    def test_default_is_median_step(self):
        from traffic_safety.speed import resolve_speed_settings, speed_estimator_from_config
        self.assertEqual(resolve_speed_settings(self.TRACKING), {"method": "median_step", "window_s": 0.5})
        est = speed_estimator_from_config(self.TRACKING)
        self.assertEqual((est.method, est.history_length, est.speed_history_length), ("median_step", 15, 21))

    def test_config_then_cli_precedence(self):
        from traffic_safety.speed import resolve_speed_settings
        cfg = dict(self.TRACKING, speed_method="regression", speed_window_s=0.7)
        self.assertEqual(resolve_speed_settings(cfg), {"method": "regression", "window_s": 0.7})
        self.assertEqual(resolve_speed_settings(cfg, "median_step", 0.5), {"method": "median_step", "window_s": 0.5})

    def test_factory_builds_regression_and_rejects_unknown(self):
        from traffic_safety.speed import speed_estimator_from_config
        est = speed_estimator_from_config(self.TRACKING, "regression", 0.5)
        self.assertEqual((est.method, est.window_s), ("regression", 0.5))
        with self.assertRaises(ValueError):
            speed_estimator_from_config(self.TRACKING, "foo")

    def test_factory_default_matches_original_outputs(self):
        from traffic_safety.speed import speed_estimator_from_config
        rng = random.Random(7)
        est, ref = speed_estimator_from_config(self.TRACKING), ReferenceMedianStep(15, 21)
        for i in range(300):
            t, x, y = i * DT, rng.gauss(0, 0.05), rng.gauss(0, 0.05)
            self.assertEqual(est.update(1, t, x, y), ref.update(1, t, x, y))
