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
    blinks: Deque[float] = field(default_factory=lambda: deque(maxlen=300))  # blink timestamps
    yawns: Deque[float] = field(default_factory=lambda: deque(maxlen=300))  # yawn timestamps (optional use here)


class DrowsinessEstimator:
    def __init__(self, ear_thresh_closed: float = 0.21, microsleep_ms: int = 1000):
        self.state = DrowsinessState()
        self.ear_thresh_closed = ear_thresh_closed
        self.microsleep_ms = microsleep_ms

    def update(self, ts: float, landmarks: np.ndarray | None) -> Dict[str, float | bool]:
        # Compute EAR if landmarks are available
        if landmarks is not None:
            ear_l = eye_aspect_ratio(landmarks, left=True)
            ear_r = eye_aspect_ratio(landmarks, left=False)
            ear = float((ear_l + ear_r) / 2.0)
        else:
            ear = 0.0

        st = self.state
        # Add to history
        st.ear_hist.append((ts, ear))

        # Blink detection: transition open->closed->open below/above thresholds
        blink_detected = False
        if len(st.ear_hist) >= 2:
            prev_ear = st.ear_hist[-2][1]
            # entering closed state
            if prev_ear >= self.ear_thresh_closed and ear < self.ear_thresh_closed and st.eye_closed_since is None:
                st.eye_closed_since = ts
            # leaving closed state
            if prev_ear < self.ear_thresh_closed and ear >= self.ear_thresh_closed and st.eye_closed_since is not None:
                duration = ts - st.eye_closed_since
                st.eye_closed_since = None
                # a quick closure is a blink
                if 0.05 <= duration <= 0.4:
                    blink_detected = True
                    st.blinks.append(ts)

        # Microsleep if eyes closed continuously beyond threshold
        microsleep = False
        if st.eye_closed_since is not None:
            if (ts - st.eye_closed_since) * 1000.0 >= self.microsleep_ms:
                microsleep = True

        # Compute PERCLOS over 60s window as percent of time eyes closed
        window_s = 60.0
        now = ts
        while st.ear_hist and now - st.ear_hist[0][0] > window_s:
            st.ear_hist.popleft()
        closed_time = 0.0
        if st.ear_hist:
            # approximate closed fraction by counting samples below threshold and dividing by sample count
            closed_samples = sum(1 for _, v in st.ear_hist if v < self.ear_thresh_closed)
            perclos = (closed_samples / len(st.ear_hist)) * 100.0
        else:
            perclos = 0.0

        # Blink rate per minute over last 60s
        while st.blinks and now - st.blinks[0] > window_s:
            st.blinks.popleft()
        blinks_per_min = (len(st.blinks) / window_s) * 60.0

        # Aggregate drowsiness score (0-100) as a heuristic
        # Higher PERCLOS and microsleep increase risk; low blink rate also increases risk
        score = perclos * 0.6 + (10.0 if microsleep else 0.0) + max(0.0, 20.0 - blinks_per_min) * 1.0
        score = float(max(0.0, min(100.0, score)))

        return {
            "ear": round(ear, 3),
            "perclos_pct": round(perclos, 1),
            "blinks_per_min": round(blinks_per_min, 1),
            "microsleep": microsleep,
            "drowsiness_score": round(score, 1),
        }

