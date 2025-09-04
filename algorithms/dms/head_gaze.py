from __future__ import annotations
from typing import Dict, List
import numpy as np


def head_pose_from_box(box_rel: list[float] | None) -> Dict[str, float]:
    """Approximate head yaw/pitch from relative bbox center.
    box_rel: [xmin, ymin, width, height] in 0..1. Returns degrees.
    """
    if not box_rel:
        return {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}
    cx = box_rel[0] + box_rel[2] / 2.0
    cy = box_rel[1] + box_rel[3] / 2.0
    yaw = (cx - 0.5) * 60.0
    pitch = (0.5 - cy) * 40.0
    return {"yaw_deg": float(yaw), "pitch_deg": float(pitch), "roll_deg": 0.0}


def gaze_zone_from_head(yaw_deg: float, pitch_deg: float) -> Dict[str, str | float]:
    """Coarse gaze/on-road estimation from head pose."""
    yaw_abs = abs(yaw_deg)
    pitch_abs = abs(pitch_deg)

    if yaw_abs < 10 and pitch_abs < 10:
        zone = "Road"
        on_road = 95.0
    elif yaw_abs < 25 and pitch_abs < 15:
        zone = "Near-Road"
        on_road = 75.0
    else:
        zone = "Off-Road"
        on_road = 30.0
    return {"gaze_zone": zone, "gaze_on_road_pct": on_road}

