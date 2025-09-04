from __future__ import annotations
from typing import List, Sequence
import numpy as np


def boxnorm_points(points_px: Sequence[Sequence[float]], box: Sequence[float], w: int, h: int) -> List[List[float]]:
    """Normalize pixel points to the given box (all in [0,1]).
    - points_px: list of [x_px, y_px]
    - box: [bx, by, bw, bh] normalized to image
    - w, h: image width/height
    Returns list of [px, py] each clamped into [0,1].
    """
    bx, by, bw, bh = box
    out: List[List[float]] = []
    for p in points_px:
        x, y = float(p[0]), float(p[1])
        xn = x / float(w)
        yn = y / float(h)
        if bw > 1e-6 and bh > 1e-6:
            px = (xn - bx) / bw
            py = (yn - by) / bh
            px = 0.0 if px < 0.0 else (1.0 if px > 1.0 else px)
            py = 0.0 if py < 0.0 else (1.0 if py > 1.0 else py)
            out.append([px, py])
    return out


def subsample(points_px: Sequence[Sequence[float]], target: int = 24) -> List[List[float]]:
    if not points_px:
        return []
    n = len(points_px)
    step = max(1, n // max(1, target))
    return [list(points_px[i]) for i in range(0, n, step)][:target]


SEMANTIC_IDX = {
    # Eyes
    "left_eye_outer": 33,
    "left_eye_inner": 133,
    "right_eye_inner": 362,
    "right_eye_outer": 263,
    # Nose
    "nose_tip": 1,
    "nose_bridge": 168,
    # Mouth
    "mouth_left": 61,
    "mouth_right": 291,
    "mouth_top": 13,
    "mouth_bottom": 14,
    # Chin / jaw
    "chin": 152,
}


def select_semantic_points(landmarks_px: np.ndarray) -> List[List[float]]:
    """Pick a compact set of semantic landmarks in pixel coordinates.
    Falls back to empty list if indices not present.
    """
    pts: List[List[float]] = []
    n = landmarks_px.shape[0] if hasattr(landmarks_px, 'shape') else 0
    for name, idx in SEMANTIC_IDX.items():
        if 0 <= idx < n:
            x, y = float(landmarks_px[idx][0]), float(landmarks_px[idx][1])
            pts.append([x, y])
    return pts

