from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from typing import Deque, Dict
import time
import numpy as np
from .face_mesh_utils import mouth_aspect_ratio


@dataclass
class YawnState:
    mar_hist: Deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=300))
    mouth_open_since: float | None = None
    yawns: Deque[float] = field(default_factory=lambda: deque(maxlen=120))  # timestamps


class YawnEstimator:
    def __init__(self, mar_thresh_open: float = 0.6, min_duration_s: float = 1.0):
        self.mar_thresh_open = mar_thresh_open
        self.min_duration_s = min_duration_s
        self.state = YawnState()

    def update(self, ts: float, landmarks: np.ndarray | None) -> Dict[str, float | bool]:
        if landmarks is not None:
            mar = float(mouth_aspect_ratio(landmarks))
        else:
            mar = 0.0

        st = self.state
        st.mar_hist.append((ts, mar))

        # Detect yawn as sustained open mouth above threshold for >= min_duration
        if mar >= self.mar_thresh_open:
            if st.mouth_open_since is None:
                st.mouth_open_since = ts
        else:
            if st.mouth_open_since is not None:
                dur = ts - st.mouth_open_since
                if dur >= self.min_duration_s:
                    st.yawns.append(ts)
                st.mouth_open_since = None

        # Compute yawns per minute over last 5 minutes (cap at recent window)
        window_s = 300.0
        while st.yawns and ts - st.yawns[0] > window_s:
            st.yawns.popleft()
        yawns_per_min = (len(st.yawns) / window_s) * 60.0 if st.yawns else 0.0

        is_yawning_now = st.mouth_open_since is not None and (ts - st.mouth_open_since) >= self.min_duration_s

        return {
            "mar": round(mar, 3),
            "yawning": is_yawning_now,
            "yawns_per_min": round(yawns_per_min, 2),
        }

