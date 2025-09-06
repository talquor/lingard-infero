from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from typing import Deque, Dict, Optional, List
import time

import numpy as np
from .face_mesh_utils import eye_aspect_ratio


@dataclass
class DrowsinessState:
    ear_hist: Deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=300))  # (ts, ear_avg)
    eye_closed_since: float | None = None
    blinks: Deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=300))  # (ts, duration)
    yawns: Deque[float] = field(default_factory=lambda: deque(maxlen=300))  # yawn timestamps (optional use here)


class DrowsinessEstimator:
    def __init__(
        self,
        ear_thresh_closed: float = 0.21,
        microsleep_ms: int = 1000,
        blink_min_s: float = 0.03,
        blink_max_s: float = 0.8,
        use_dynamic_threshold: bool = True,
        dynamic_rel_close: float = 0.85,
        dynamic_rel_open: float = 0.93,
        absolute_min_close: float = 0.15,
        perclos_window_s: float = 30.0,
        sleep_min_s: float = 3.0,
    ):
        self.state = DrowsinessState()
        self.ear_thresh_closed = ear_thresh_closed
        self.microsleep_ms = microsleep_ms
        self.blink_min_s = blink_min_s
        self.blink_max_s = blink_max_s
        self.use_dynamic_threshold = use_dynamic_threshold
        self.dynamic_rel_close = dynamic_rel_close
        self.dynamic_rel_open = dynamic_rel_open
        self.absolute_min_close = absolute_min_close
        self.perclos_window_s = perclos_window_s
        self.sleep_min_s = sleep_min_s

    def update(self, ts: float, landmarks: np.ndarray | None) -> Dict[str, float | bool]:
        # Compute EAR if landmarks are available; if not, skip updates that infer closure
        if landmarks is not None:
            ear_l = float(eye_aspect_ratio(landmarks, left=True))
            ear_r = float(eye_aspect_ratio(landmarks, left=False))
            ear = float((ear_l + ear_r) / 2.0)
        else:
            ear_l = None
            ear_r = None
            ear = None  # treat as missing data

        st = self.state
        # Add to history
        if ear is not None:
            st.ear_hist.append((ts, ear))

        # Blink detection: transition open->closed->open below/above thresholds
        blink_detected = False
        # Determine dynamic thresholds if enabled
        th_close = self.ear_thresh_closed
        th_open = self.ear_thresh_closed + 0.02  # small hysteresis by default
        if self.use_dynamic_threshold and len(st.ear_hist) >= 5:
            vals: List[float] = [v for _, v in st.ear_hist if v is not None]
            if vals:
                # 90th percentile as open baseline
                vs = sorted(vals)
                p90 = vs[int(0.9 * (len(vs) - 1))]
                base = p90
                th_close = max(self.absolute_min_close, min(base * self.dynamic_rel_close, base * 0.98))
                th_open = max(th_close + 0.005, min(base * self.dynamic_rel_open, base * 0.995))

        if ear is not None:
            prev_ear = st.ear_hist[-2][1] if len(st.ear_hist) >= 2 else None
            # entering closed state (either transition or starting closed)
            if ear < th_close and st.eye_closed_since is None:
                if prev_ear is None or prev_ear >= th_open:
                    st.eye_closed_since = ts
            # leaving closed state
            if prev_ear is not None and prev_ear < th_close and ear >= th_open and st.eye_closed_since is not None:
                duration = ts - st.eye_closed_since
                st.eye_closed_since = None
                # a quick closure is a blink
                # adapt max duration for low FPS by using ~2.5x average sample interval
                avg_dt = 0.0
                if len(st.ear_hist) >= 2:
                    dt_sum = 0.0
                    for i in range(1, len(st.ear_hist)):
                        dt_sum += (st.ear_hist[i][0] - st.ear_hist[i-1][0])
                    avg_dt = dt_sum / (len(st.ear_hist) - 1)
                dyn_max = max(self.blink_max_s, 2.5 * avg_dt) if avg_dt > 0 else self.blink_max_s
                if self.blink_min_s <= duration <= dyn_max:
                    blink_detected = True
                    st.blinks.append((ts, duration))

        # Microsleep and sleep-event if eyes closed continuously beyond thresholds
        microsleep = False
        microsleep_dwell_s = 0.0
        sleep_event_active = False
        sleep_dwell_s = 0.0
        if st.eye_closed_since is not None and ear is not None:
            dwell = ts - st.eye_closed_since
            microsleep_dwell_s = round(dwell, 2)
            if dwell * 1000.0 >= self.microsleep_ms:
                microsleep = True
            if dwell >= self.sleep_min_s:
                sleep_event_active = True
                sleep_dwell_s = round(dwell, 2)

        # Compute PERCLOS over configured window as percent of time eyes closed
        window_s = self.perclos_window_s
        now = ts
        while st.ear_hist and now - st.ear_hist[0][0] > window_s:
            st.ear_hist.popleft()
        # time-weighted PERCLOS over the window
        perclos = 0.0
        if len(st.ear_hist) >= 2:
            closed_dur = 0.0
            total_dur = st.ear_hist[-1][0] - st.ear_hist[0][0]
            for i in range(1, len(st.ear_hist)):
                ts_prev, ear_prev = st.ear_hist[i-1]
                ts_curr, _ = st.ear_hist[i]
                dt = ts_curr - ts_prev
                if ear_prev < th_close:
                    closed_dur += dt
            if total_dur > 0:
                perclos = (closed_dur / total_dur) * 100.0
        else:
            perclos = 0.0

        # Blink rate per minute over last 60s
        while st.blinks and now - st.blinks[0][0] > window_s:
            st.blinks.popleft()
        blinks_per_min = (len(st.blinks) / window_s) * 60.0
        avg_blink_dur_ms = round((sum(d for _, d in st.blinks) / len(st.blinks) * 1000.0), 1) if st.blinks else 0.0
        time_since_last_blink_s = round(now - st.blinks[-1][0], 2) if st.blinks else None

        # Aggregate drowsiness score (0-100) as a heuristic
        # Higher PERCLOS and microsleep increase risk; low blink rate also increases risk
        score = perclos * 0.6 + (10.0 if microsleep else 0.0) + max(0.0, 20.0 - blinks_per_min) * 1.0
        score = float(max(0.0, min(100.0, score)))

        left_closed = (ear_l is not None) and (ear_l < th_close)
        right_closed = (ear_r is not None) and (ear_r < th_close)

        # Estimate FPS from ear history spacing (for diagnostics)
        fps_est = None
        if len(st.ear_hist) >= 2:
            total = st.ear_hist[-1][0] - st.ear_hist[0][0]
            count = len(st.ear_hist) - 1
            if total > 0 and count > 0:
                fps_est = round(count / total, 1)

        return {
            "ear": round(ear, 3) if ear is not None else None,
            "left_ear": round(ear_l, 3) if ear_l is not None else None,
            "right_ear": round(ear_r, 3) if ear_r is not None else None,
            "left_eye_closed": left_closed,
            "right_eye_closed": right_closed,
            "blink_detected": blink_detected,
            "perclos_pct": round(perclos, 1),
            "blinks_per_min": round(blinks_per_min, 1),
            "avg_blink_dur_ms": avg_blink_dur_ms,
            "time_since_last_blink_s": time_since_last_blink_s,
            "microsleep": microsleep,
            "microsleep_dwell_s": microsleep_dwell_s,
            "sleep_event_active": sleep_event_active,
            "sleep_dwell_s": sleep_dwell_s,
            "drowsiness_score": round(score, 1),
            "fps_est": fps_est,
            "ear_thresh_closed": round(th_close, 3),
            "ear_thresh_open": round(th_open, 3),
            "perclos_window_s": window_s,
        }
