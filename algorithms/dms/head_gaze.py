from __future__ import annotations
from typing import Dict, List
import numpy as np
import math


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


def _euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def head_pose_from_landmarks(landmarks: np.ndarray) -> Dict[str, float | str]:
    """Estimate head yaw/pitch/roll from 2D FaceMesh landmarks (pixel coords).
    Uses relative geometry of eyes and nose; returns degrees and a coarse look direction.
    """
    # Landmarks used
    try:
        left_eye_outer = landmarks[33]
        right_eye_outer = landmarks[263]
        nose_tip = landmarks[1] if landmarks.shape[0] > 1 else landmarks[4]
        mouth_top = landmarks[13]
        mouth_bottom = landmarks[14]
    except Exception:
        # Fallback neutral
        return {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0, "look_dir": "Straight"}

    mid_eyes = (left_eye_outer + right_eye_outer) / 2.0
    mouth_mid = (mouth_top + mouth_bottom) / 2.0

    face_w = max(1e-6, _euclid(left_eye_outer, right_eye_outer))
    face_h = max(1e-6, _euclid(mid_eyes, mouth_mid))

    # Roll: tilt of eye line
    dy = right_eye_outer[1] - left_eye_outer[1]
    dx = right_eye_outer[0] - left_eye_outer[0]
    roll = math.degrees(math.atan2(dy, dx))

    # Yaw: horizontal offset of nose from mid-eye center (normalized by face width)
    yaw = (nose_tip[0] - mid_eyes[0]) / face_w
    yaw_deg = float(np.clip(yaw * 90.0, -60.0, 60.0))

    # Pitch: vertical offset of nose from mid-eye center (normalized by inter-eye to mouth distance)
    pitch = (mid_eyes[1] - nose_tip[1]) / face_h
    pitch_deg = float(np.clip(pitch * 70.0, -50.0, 50.0))

    # Look direction classification
    yaw_th = 15.0
    pitch_th = 12.0
    if abs(yaw_deg) <= yaw_th and abs(pitch_deg) <= pitch_th:
        look = "Straight"
    elif abs(yaw_deg) >= abs(pitch_deg):
        look = "Right" if yaw_deg > 0 else "Left"
    else:
        look = "Up" if pitch_deg > 0 else "Down"

    return {
        "yaw_deg": round(yaw_deg, 1),
        "pitch_deg": round(pitch_deg, 1),
        "roll_deg": round(float(roll), 1),
        "look_dir": look,
    }
