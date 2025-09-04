from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from typing import Deque, Dict


@dataclass
class DistractionState:
    offroad_since: float | None = None
    offroad_events: Deque[float] = field(default_factory=lambda: deque(maxlen=600))  # event timestamps
    gaze_hist: Deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=600))  # (ts, onroad_pct)


class DistractionEstimator:
    def __init__(self, onroad_threshold_pct: float = 60.0, event_dwell_s: float = 2.0, window_s: float = 60.0):
        self.thresh = onroad_threshold_pct
        self.event_dwell_s = event_dwell_s
        self.window_s = window_s
        self.state = DistractionState()

    def update(self, ts: float, onroad_pct: float | None) -> Dict[str, float | int | bool]:
        st = self.state

        # Add to history if valid
        if onroad_pct is not None:
            st.gaze_hist.append((ts, onroad_pct))

            # off-road detection
            if onroad_pct < self.thresh:
                if st.offroad_since is None:
                    st.offroad_since = ts
            else:
                if st.offroad_since is not None:
                    dwell = ts - st.offroad_since
                    if dwell >= self.event_dwell_s:
                        st.offroad_events.append(ts)
                    st.offroad_since = None

        # trim windows
        while st.gaze_hist and ts - st.gaze_hist[0][0] > self.window_s:
            st.gaze_hist.popleft()
        while st.offroad_events and ts - st.offroad_events[0] > self.window_s:
            st.offroad_events.popleft()

        # compute aggregates
        if st.gaze_hist:
            off_samples = sum(1 for _, v in st.gaze_hist if v < self.thresh)
            offroad_pct = (off_samples / len(st.gaze_hist)) * 100.0
        else:
            offroad_pct = 0.0

        events_per_min = (len(st.offroad_events) / self.window_s) * 60.0
        offroad_dwell_now_s = (ts - st.offroad_since) if st.offroad_since is not None else 0.0
        offroad_event_active = st.offroad_since is not None and offroad_dwell_now_s >= self.event_dwell_s

        return {
            "eyes_off_road_pct": round(offroad_pct, 1),
            "eyes_off_road_events_per_min": round(events_per_min, 2),
            "eyes_off_road_dwell_s": round(offroad_dwell_now_s, 2),
            "eyes_off_road_event_active": offroad_event_active,
        }

