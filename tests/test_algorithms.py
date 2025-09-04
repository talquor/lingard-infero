import math
import time
import unittest

import numpy as np

from algorithms.dms.face_mesh_utils import eye_aspect_ratio, mouth_aspect_ratio
from algorithms.dms.drowsiness import DrowsinessEstimator
from algorithms.dms.yawn import YawnEstimator
from algorithms.dms.head_gaze import head_pose_from_landmarks
from algorithms.dms.distraction import DistractionEstimator
from algorithms.scoring.ncap_scoring import NCAPScorer, NCAPConfig
from pathlib import Path


def make_landmarks(points: dict[int, tuple[float, float]], size: int = 468) -> np.ndarray:
    lm = np.zeros((size, 2), dtype=np.float32)
    for idx, (x, y) in points.items():
        lm[idx] = (x, y)
    return lm


class TestDrowsiness(unittest.TestCase):
    def test_ear_open_vs_closed(self):
        # Construct open eyes
        pts_open = {
            33: (100, 100), 133: (120, 100), 160: (110, 95), 153: (110, 105), 158: (112, 95), 144: (112, 105),
            263: (140, 100), 362: (120, 100), 387: (130, 95), 380: (130, 105), 385: (128, 95), 373: (128, 105),
        }
        lm_open = make_landmarks(pts_open)
        ear_l = eye_aspect_ratio(lm_open, left=True)
        ear_r = eye_aspect_ratio(lm_open, left=False)
        self.assertGreater(ear_l, 0.3)
        self.assertGreater(ear_r, 0.3)

        # Closed eyes (small vertical distances)
        pts_closed = {
            33: (100, 100), 133: (120, 100), 160: (110, 99), 153: (110, 101), 158: (112, 99), 144: (112, 101),
            263: (140, 100), 362: (120, 100), 387: (130, 99), 380: (130, 101), 385: (128, 99), 373: (128, 101),
        }
        lm_closed = make_landmarks(pts_closed)
        ear_l_c = eye_aspect_ratio(lm_closed, left=True)
        ear_r_c = eye_aspect_ratio(lm_closed, left=False)
        self.assertLess(ear_l_c, 0.2)
        self.assertLess(ear_r_c, 0.2)

    def test_perclos_blink_microsleep(self):
        d = DrowsinessEstimator(ear_thresh_closed=0.21, microsleep_ms=1000)
        # Simulate 2 seconds at 20 Hz: 1s open, 1s closed (PERCLOS ~50%)
        t0 = 0.0
        for i in range(20):
            ts = t0 + i * 0.05
            lm = make_landmarks({
                33: (100, 100), 133: (120, 100), 160: (110, 95), 153: (110, 105), 158: (112, 95), 144: (112, 105),
                263: (140, 100), 362: (120, 100), 387: (130, 95), 380: (130, 105), 385: (128, 95), 373: (128, 105),
            }) if i < 20 // 2 else make_landmarks({
                33: (100, 100), 133: (120, 100), 160: (110, 99), 153: (110, 101), 158: (112, 99), 144: (112, 101),
                263: (140, 100), 362: (120, 100), 387: (130, 99), 380: (130, 101), 385: (128, 99), 373: (128, 101),
            })
            out = d.update(ts, lm)
        self.assertTrue(40.0 <= out["perclos_pct"] <= 60.0)

        # Blink: 0.2s closure
        d = DrowsinessEstimator(ear_thresh_closed=0.21, microsleep_ms=1000)
        ts = 0.0
        # open
        for _ in range(5):
            out = d.update(ts, make_landmarks({
                33: (100, 100), 133: (120, 100), 160: (110, 95), 153: (110, 105), 158: (112, 95), 144: (112, 105),
                263: (140, 100), 362: (120, 100), 387: (130, 95), 380: (130, 105), 385: (128, 95), 373: (128, 105),
            }))
            ts += 0.05
        # closed 0.2s
        for _ in range(4):
            out = d.update(ts, make_landmarks({
                33: (100, 100), 133: (120, 100), 160: (110, 99), 153: (110, 101), 158: (112, 99), 144: (112, 101),
                263: (140, 100), 362: (120, 100), 387: (130, 99), 380: (130, 101), 385: (128, 99), 373: (128, 101),
            }))
            ts += 0.05
        # open again
        out = d.update(ts, make_landmarks({
            33: (100, 100), 133: (120, 100), 160: (110, 95), 153: (110, 105), 158: (112, 95), 144: (112, 105),
            263: (140, 100), 362: (120, 100), 387: (130, 95), 380: (130, 105), 385: (128, 95), 373: (128, 105),
        }))
        self.assertGreaterEqual(out["blinks_per_min"], 1.0)
        self.assertGreater(out["avg_blink_dur_ms"], 150.0)
        self.assertLess(out["avg_blink_dur_ms"], 400.0)

        # Microsleep (>1s)
        d = DrowsinessEstimator(ear_thresh_closed=0.21, microsleep_ms=1000)
        ts = 0.0
        # closed for 1.2s
        for _ in range(24):
            out = d.update(ts, make_landmarks({
                33: (100, 100), 133: (120, 100), 160: (110, 99), 153: (110, 101), 158: (112, 99), 144: (112, 101),
                263: (140, 100), 362: (120, 100), 387: (130, 99), 380: (130, 101), 385: (128, 99), 373: (128, 101),
            }))
            ts += 0.05
        self.assertTrue(out["microsleep"])


class TestYawn(unittest.TestCase):
    def test_yawn_detection(self):
        y = YawnEstimator(mar_thresh_open=0.6, min_duration_s=1.0)
        ts = 0.0
        # neutral mouth MAR ~0.5 for 0.5s
        for _ in range(10):
            lm = make_landmarks({61: (110, 135), 291: (130, 135), 13: (120, 130), 14: (120, 140)})
            out = y.update(ts, lm)
            ts += 0.05
        self.assertFalse(out["yawning"]) 
        # yawn >1s with MAR ~1.0
        for _ in range(22):
            lm = make_landmarks({61: (110, 135), 291: (130, 135), 13: (120, 130), 14: (120, 150)})
            out = y.update(ts, lm)
            ts += 0.05
        self.assertTrue(out["yawning"])  # ongoing yawn
        # close mouth to record event
        out = y.update(ts, make_landmarks({61: (110, 135), 291: (130, 135), 13: (120, 130), 14: (120, 140)}))
        self.assertGreaterEqual(out["yawns_per_min"], 0.15)  # approx 1 event within 5 min window scaled (1/300*60=0.2)


class TestHeadPose(unittest.TestCase):
    def test_look_direction(self):
        # Neutral
        base = {
            33: (100, 100), 263: (140, 100), 1: (120, 100), 13: (120, 130), 14: (120, 140)
        }
        lm = make_landmarks(base)
        hp = head_pose_from_landmarks(lm)
        self.assertEqual(hp["look_dir"], "Straight")

        # Look right: nose to the right
        lm = make_landmarks({**base, 1: (130, 100)})
        hp = head_pose_from_landmarks(lm)
        self.assertEqual(hp["look_dir"], "Right")
        self.assertGreater(hp["yaw_deg"], 0)

        # Look left
        lm = make_landmarks({**base, 1: (110, 100)})
        hp = head_pose_from_landmarks(lm)
        self.assertEqual(hp["look_dir"], "Left")
        self.assertLess(hp["yaw_deg"], 0)

        # Look up: nose above mid-eyes
        lm = make_landmarks({**base, 1: (120, 90)})
        hp = head_pose_from_landmarks(lm)
        self.assertEqual(hp["look_dir"], "Up")
        self.assertGreater(hp["pitch_deg"], 0)

        # Look down
        lm = make_landmarks({**base, 1: (120, 110)})
        hp = head_pose_from_landmarks(lm)
        self.assertEqual(hp["look_dir"], "Down")
        self.assertLess(hp["pitch_deg"], 0)


class TestDistractionNCAP(unittest.TestCase):
    def test_distraction_events(self):
        d = DistractionEstimator(onroad_threshold_pct=60.0, event_dwell_s=2.0, window_s=60.0)
        ts = 0.0
        # 3 seconds off-road to trigger one event
        for _ in range(60):
            out = d.update(ts, 30.0)
            ts += 0.05
        self.assertTrue(out["eyes_off_road_event_active"])  # still active
        # back on road to close the event
        out = d.update(ts, 90.0)
        self.assertGreaterEqual(out["eyes_off_road_events_per_min"], 1.0)

    def test_ncap_penalties(self):
        scorer = NCAPScorer()
        # Bad metrics should reduce scores
        metrics = {
            "dms_perclos_pct": 40.0,
            "dms_microsleep": True,
            "dms_blinks_per_min": 2.0,
            "dms_eyes_off_road_pct": 50.0,
            "dms_eyes_off_road_events_per_min": 2.0,
            "dms_yawns_per_min": 5.0,
            "oms_phone_use": True,
            "oms_smoking": True,
            "oms_seatbelt_fastened": False,
        }
        out = scorer.score(metrics)
        self.assertLess(out["sections"]["vigilance"], 100.0)
        self.assertLess(out["sections"]["distraction"], 100.0)
        self.assertLess(out["sections"]["impairment"], 100.0)

    def test_ncap_load_config(self):
        # Ensure loader merges/overrides defaults
        cfg_path = Path('config/ncap_config.json')
        self.assertTrue(cfg_path.exists())
        scorer = NCAPScorer(config_path=str(cfg_path))
        out = scorer.score({})
        self.assertIn('overall', out)


if __name__ == "__main__":
    unittest.main()
