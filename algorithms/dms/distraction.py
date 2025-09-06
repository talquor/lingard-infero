from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from typing import Deque, Dict, Optional


@dataclass
class DistractionState:
    # Lizard (eyes-off-road) state
    offroad_since: float | None = None
    off_short_events: Deque[float] = field(default_factory=lambda: deque(maxlen=600))
    off_long_events: Deque[float] = field(default_factory=lambda: deque(maxlen=600))
    gaze_hist: Deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=600))  # (ts, onroad_pct)

    # Owl (head yaw) state
    owl_since: float | None = None
    owl_short_events: Deque[float] = field(default_factory=lambda: deque(maxlen=600))
    owl_long_events: Deque[float] = field(default_factory=lambda: deque(maxlen=600))


class DistractionEstimator:
    def __init__(
        self,
        onroad_threshold_pct: float = 60.0,
        lizard_short_min_s: float = 0.2,
        lizard_long_s: float = 2.0,
        window_s: float = 60.0,
        owl_yaw_th_deg: float = 20.0,
        owl_short_min_s: float = 0.2,
        owl_long_s: float = 2.0,
        owl_pitch_th_deg: float = 12.0,
    ):
        self.onroad_threshold_pct = onroad_threshold_pct
        self.lizard_short_min_s = lizard_short_min_s
        self.lizard_long_s = lizard_long_s
        self.window_s = window_s
        self.owl_yaw_th_deg = owl_yaw_th_deg
        self.owl_short_min_s = owl_short_min_s
        self.owl_long_s = owl_long_s
        self.owl_pitch_th_deg = owl_pitch_th_deg
        self.state = DistractionState()

    def update(self, ts: float, onroad_pct: Optional[float], yaw_deg: Optional[float] = None, pitch_deg: Optional[float] = None) -> Dict[str, float | int | bool | str]:
        st = self.state

        # Add to history
        if onroad_pct is not None:
            st.gaze_hist.append((ts, onroad_pct))
            # Lizard: eyes off-road dwell
            if onroad_pct < self.onroad_threshold_pct:
                if st.offroad_since is None:
                    st.offroad_since = ts
            else:
                if st.offroad_since is not None:
                    dwell = ts - st.offroad_since
                    if dwell >= self.lizard_long_s:
                        st.off_long_events.append(ts)
                    elif dwell >= self.lizard_short_min_s:
                        st.off_short_events.append(ts)
                    st.offroad_since = None

        # Owl: head yaw dwell
        if yaw_deg is not None:
            if abs(yaw_deg) >= self.owl_yaw_th_deg:
                if st.owl_since is None:
                    st.owl_since = ts
            else:
                if st.owl_since is not None:
                    dwell = ts - st.owl_since
                    if dwell >= self.owl_long_s:
                        st.owl_long_events.append(ts)
                    elif dwell >= self.owl_short_min_s:
                        st.owl_short_events.append(ts)
                    st.owl_since = None

        # Trim windows
        while st.gaze_hist and ts - st.gaze_hist[0][0] > self.window_s:
            st.gaze_hist.popleft()
        for dq in (st.off_short_events, st.off_long_events, st.owl_short_events, st.owl_long_events):
            while dq and ts - dq[0] > self.window_s:
                dq.popleft()

        # Aggregates
        if st.gaze_hist:
            off_samples = sum(1 for _, v in st.gaze_hist if v < self.onroad_threshold_pct)
            offroad_pct = (off_samples / len(st.gaze_hist)) * 100.0
        else:
            offroad_pct = 0.0

        # Current dwells
        lizard_dwell = (ts - st.offroad_since) if st.offroad_since is not None else 0.0
        owl_dwell = (ts - st.owl_since) if st.owl_since is not None else 0.0
        lizard_short_active = st.offroad_since is not None and (self.lizard_short_min_s <= lizard_dwell < self.lizard_long_s)
        lizard_long_active = st.offroad_since is not None and (lizard_dwell >= self.lizard_long_s)
        owl_short_active = st.owl_since is not None and (self.owl_short_min_s <= owl_dwell < self.owl_long_s)
        owl_long_active = st.owl_since is not None and (owl_dwell >= self.owl_long_s)

        # Rates per minute
        to_rate = lambda n: round((n / self.window_s) * 60.0, 2)
        lizard_short_rpm = to_rate(len(st.off_short_events))
        lizard_long_rpm = to_rate(len(st.off_long_events))
        owl_short_rpm = to_rate(len(st.owl_short_events))
        owl_long_rpm = to_rate(len(st.owl_long_events))

        # Direction classification
        def dir_from_yaw_pitch(yaw: Optional[float], pitch: Optional[float]) -> str:
            if yaw is None and pitch is None:
                return "Unknown"
            y = yaw or 0.0
            p = pitch or 0.0
            if abs(y) < self.owl_yaw_th_deg and abs(p) < self.owl_pitch_th_deg:
                return "Focused"
            if abs(y) >= abs(p):
                return "Right" if y > 0 else "Left"
            else:
                return "Up" if p > 0 else "Down"

        owl_direction = dir_from_yaw_pitch(yaw_deg, pitch_deg)
        if onroad_pct is None:
            lizard_direction = "Unknown"
        elif onroad_pct >= self.onroad_threshold_pct:
            lizard_direction = "Focused"
        else:
            lizard_direction = owl_direction  # proxy until eye-gaze specific estimator is available

        return {
            "eyes_off_road_pct": round(offroad_pct, 1),
            "eyes_off_road_dwell_s": round(lizard_dwell, 2),
            "eyes_off_short_active": lizard_short_active,
            "eyes_off_long_active": lizard_long_active,
            "eyes_off_short_per_min": lizard_short_rpm,
            "eyes_off_long_per_min": lizard_long_rpm,
            "owl_yaw_dwell_s": round(owl_dwell, 2),
            "owl_short_active": owl_short_active,
            "owl_long_active": owl_long_active,
            "owl_short_per_min": owl_short_rpm,
            "owl_long_per_min": owl_long_rpm,
            "owl_direction": owl_direction,
            "lizard_direction": lizard_direction,
        }
