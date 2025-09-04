from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

# Indices for MediaPipe FaceMesh landmarks
# Using eye and mouth landmark groups for EAR and MAR
# Left/Right eye landmarks chosen to approximate eye aspect ratio
LEFT_EYE = [33, 160, 158, 133, 153, 144]   # [outer, top1, top2, inner, bottom1, bottom2]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

# Mouth landmarks for MAR (outer lips): left, top, right, bottom
MOUTH = [61, 13, 291, 14]


def _euclidean(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))


def eye_aspect_ratio(landmarks: np.ndarray, left: bool = True) -> float:
    idx = LEFT_EYE if left else RIGHT_EYE
    pts = landmarks[idx]
    # EAR = (||p2-p5|| + ||p3-p6||) / (2 * ||p1-p4||)
    p1, p2, p3, p4, p5, p6 = pts
    vert = _euclidean(p2, p5) + _euclidean(p3, p6)
    horiz = 2.0 * _euclidean(p1, p4)
    if horiz <= 1e-6:
        return 0.0
    return float(vert / horiz)


def mouth_aspect_ratio(landmarks: np.ndarray) -> float:
    # MAR = ||top-bottom|| / ||left-right||
    l, t, r, b = landmarks[MOUTH]
    horiz = _euclidean(l, r)
    if horiz <= 1e-6:
        return 0.0
    return float(_euclidean(t, b) / horiz)


def eye_aperture_points(landmarks: np.ndarray) -> list[list[float]]:
    """Return four keypoints for blink computation in pixel coords:
    [left_top, left_bottom, right_top, right_bottom]. Each is [x,y].
    Uses the average of two top and two bottom eyelid points per eye for stability.
    """
    lt = landmarks[LEFT_EYE[1:3]].mean(axis=0)
    lb = landmarks[LEFT_EYE[4:6]].mean(axis=0)
    rt = landmarks[RIGHT_EYE[1:3]].mean(axis=0)
    rb = landmarks[RIGHT_EYE[4:6]].mean(axis=0)
    return [[float(lt[0]), float(lt[1])],
            [float(lb[0]), float(lb[1])],
            [float(rt[0]), float(rt[1])],
            [float(rb[0]), float(rb[1])]]


def extract_landmarks(mp_results, image_width: int, image_height: int) -> np.ndarray | None:
    """Return Nx2 array of pixel coords for the first face landmarks, or None."""
    if not getattr(mp_results, 'multi_face_landmarks', None):
        return None
    lm = mp_results.multi_face_landmarks[0].landmark
    pts = np.array([[p.x * image_width, p.y * image_height] for p in lm], dtype=np.float32)
    return pts


def extract_all_landmarks(mp_results, image_width: int, image_height: int) -> List[np.ndarray]:
    """Return list of Nx2 arrays for each face; empty list if none."""
    out: List[np.ndarray] = []
    if not getattr(mp_results, 'multi_face_landmarks', None):
        return out
    for face in mp_results.multi_face_landmarks:
        lm = face.landmark
        pts = np.array([[p.x * image_width, p.y * image_height] for p in lm], dtype=np.float32)
        out.append(pts)
    return out
