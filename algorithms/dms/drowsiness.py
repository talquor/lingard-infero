from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from typing import Deque, Dict
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
    def __init__(self, ear_thresh_closed: float = 0.21, microsleep_ms: int = 1000):
        self.state = DrowsinessState()
        self.ear_thresh_closed = ear_thresh_closed
        self.microsleep_ms = microsleep_ms

    def update(self, ts: float, landmarks: np.ndarray | None) -> Dict[str, float | bool]:
        # Compute EAR if landmarks are available; if not, skip updates that infer closure
        if landmarks is not None:
            ear_l = eye_aspect_ratio(landmarks, left=True)
            ear_r = eye_aspect_ratio(landmarks, left=False)
            ear = float((ear_l + ear_r) / 2.0)
        else:
            ear = None  # treat as missing data

        st = self.state
        # Add to history
        if ear is not None:
            st.ear_hist.append((ts, ear))

        # Blink detection: transition open->closed->open below/above thresholds
        blink_detected = False
        if ear is not None:
            prev_ear = st.ear_hist[-2][1] if len(st.ear_hist) >= 2 else None
            # entering closed state (either transition or starting closed)
            if ear < self.ear_thresh_closed and st.eye_closed_since is None:
                if prev_ear is None or prev_ear >= self.ear_thresh_closed:
                    st.eye_closed_since = ts
            # leaving closed state
            if prev_ear is not None and prev_ear < self.ear_thresh_closed and ear >= self.ear_thresh_closed and st.eye_closed_since is not None:
                duration = ts - st.eye_closed_since
                st.eye_closed_since = None
                # a quick closure is a blink
                if 0.05 <= duration <= 0.4:
                    blink_detected = True
                    st.blinks.append((ts, duration))

        # Microsleep if eyes closed continuously beyond threshold
        microsleep = False
        if st.eye_closed_since is not None and ear is not None:
            if (ts - st.eye_closed_since) * 1000.0 >= self.microsleep_ms:
                microsleep = True

        # Compute PERCLOS over 60s window as percent of time eyes closed
        window_s = 60.0
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
                if ear_prev < self.ear_thresh_closed:
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

        return {
            "ear": round(ear, 3) if ear is not None else None,
            "perclos_pct": round(perclos, 1),
            "blinks_per_min": round(blinks_per_min, 1),
            "avg_blink_dur_ms": avg_blink_dur_ms,
            "time_since_last_blink_s": time_since_last_blink_s,
            "microsleep": microsleep,
            "drowsiness_score": round(score, 1),
        }
