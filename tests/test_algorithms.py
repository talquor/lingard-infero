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
from algorithms.utils.geometry import boxnorm_points, subsample, select_semantic_points
from algorithms.dms.face_mesh_utils import eye_aperture_points


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
        self.assertTrue(out["blink_detected"])  # should flag blink now
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

    def test_left_right_ear_and_closed_flags(self):
        d = DrowsinessEstimator(ear_thresh_closed=0.21, microsleep_ms=1000)
        ts = 0.0
        # Left eye closed, right eye open
        lm = make_landmarks({
            # Left eye closed
            33: (100, 100), 133: (120, 100), 160: (110, 99), 153: (110, 101), 158: (112, 99), 144: (112, 101),
            # Right eye open
            263: (140, 100), 362: (120, 100), 387: (130, 95), 380: (130, 105), 385: (128, 95), 373: (128, 105),
        })
        out = d.update(ts, lm)
        self.assertIsNotNone(out["left_ear"])
        self.assertIsNotNone(out["right_ear"])
        self.assertLess(out["left_ear"], 0.21)
        self.assertGreater(out["right_ear"], 0.21)
        self.assertTrue(out["left_eye_closed"])  # left closed
        self.assertFalse(out["right_eye_closed"])  # right open


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
        d = DistractionEstimator(onroad_threshold_pct=60.0, lizard_short_min_s=0.5, lizard_long_s=2.0, window_s=60.0,
                                 owl_yaw_th_deg=20.0, owl_short_min_s=0.5, owl_long_s=2.0)
        ts = 0.0
        # 1.0s off-road -> short lizard event when returning on-road
        for _ in range(20):
            out = d.update(ts, 30.0, yaw_deg=5.0)
            ts += 0.05
        # back on road to close
        out = d.update(ts, 90.0, yaw_deg=5.0)
        self.assertTrue(out["eyes_off_short_per_min"] > 0)

        # 2.5s off-road -> long lizard
        for _ in range(50):
            out = d.update(ts, 30.0, yaw_deg=5.0)
            ts += 0.05
        out = d.update(ts, 90.0, yaw_deg=5.0)
        self.assertTrue(out["eyes_off_long_per_min"] > 0)

        # Owl short: yaw above threshold for 1.0s
        for _ in range(20):
            out = d.update(ts, 90.0, yaw_deg=30.0)
            ts += 0.05
        out = d.update(ts, 90.0, yaw_deg=0.0)
        self.assertTrue(out["owl_short_per_min"] > 0)

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

class TestGeometry(unittest.TestCase):
    def test_boxnorm_and_subsample(self):
        w, h = 200, 100
        box = [0.25, 0.2, 0.5, 0.6]  # normalized
        # Create 100 random points inside the image
        pts = [[50 + i, 30 + (i % 10)] for i in range(100)]
        sub = subsample(pts, target=24)
        self.assertLessEqual(len(sub), 24)
        nb = boxnorm_points(sub, box, w, h)
        self.assertTrue(all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in nb))

    def test_eye_points_four(self):
        # Synthetic landmarks with known eye indices
        pts = {
            33:(100,100),133:(120,100),160:(110,95),158:(112,95),153:(110,105),144:(112,105),
            263:(140,100),362:(120,100),387:(130,95),385:(128,95),380:(130,105),373:(128,105),
        }
        lm = make_landmarks(pts)
        eyes = eye_aperture_points(lm)
        self.assertEqual(len(eyes), 4)
        # top y should be less than bottom y in pixel coords
        self.assertLess(eyes[0][1], eyes[1][1])  # left eye top < bottom
        self.assertLess(eyes[2][1], eyes[3][1])  # right eye top < bottom

    def test_ncap_load_config(self):
        # Ensure loader merges/overrides defaults
        cfg_path = Path('config/ncap_config.json')
        self.assertTrue(cfg_path.exists())
        scorer = NCAPScorer(config_path=str(cfg_path))
        out = scorer.score({})
        self.assertIn('overall', out)


if __name__ == "__main__":
    unittest.main()
