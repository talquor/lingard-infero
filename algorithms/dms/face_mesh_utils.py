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


# Iris landmark indices when FaceMesh refine_landmarks=True
# Right iris: 469-472; Left iris: 474-477 (MediaPipe convention)
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_IRIS = [474, 475, 476, 477]


def iris_centers(landmarks: np.ndarray) -> tuple[list[float] | None, list[float] | None]:
    """Return (left_iris_center[x,y], right_iris_center[x,y]) in pixel coords, or (None,None) if unavailable."""
    n = landmarks.shape[0] if hasattr(landmarks, 'shape') else 0
    def _center(idx):
        pts = [landmarks[i] for i in idx if 0 <= i < n]
        if len(pts) < 3:
            return None
        c = np.mean(np.array(pts, dtype=np.float32), axis=0)
        return [float(c[0]), float(c[1])]
    r = _center(RIGHT_IRIS)
    l = _center(LEFT_IRIS)
    return l, r


def eye_boxes(landmarks: np.ndarray, w: int, h: int) -> tuple[list[float], list[float]]:
    """Approximate eye boxes in normalized image coords [bx,by,bw,bh] using outer/inner eye corners and eyelids."""
    le_idx = [33, 133, 160, 158, 153, 144]
    re_idx = [362, 263, 387, 385, 380, 373]
    def _box(idx):
        pts = np.array([landmarks[i] for i in idx], dtype=np.float32)
        xs = pts[:,0] / float(w)
        ys = pts[:,1] / float(h)
        minx, maxx = max(0.0, float(xs.min())), min(1.0, float(xs.max()))
        miny, maxy = max(0.0, float(ys.min())), min(1.0, float(ys.max()))
        return [minx, miny, max(1e-6, maxx-minx), max(1e-6, maxy-miny)]
    return _box(le_idx), _box(re_idx)


def lizard_direction_from_iris(landmarks: np.ndarray, w: int, h: int, yaw_pitch_dir: str | None = None,
                               focus_th: float = 0.3) -> tuple[str, dict]:
    """Estimate eye-gaze direction via iris centers relative to eye boxes.
    Returns (direction, debug) where direction in {Left,Right,Up,Down,Focused,Unknown}.
    focus_th: abs offset under which we consider Focused.
    """
    li, ri = iris_centers(landmarks)
    if li is None or ri is None:
        return (yaw_pitch_dir or "Unknown", {"method": "fallback"})
    le_box, re_box = eye_boxes(landmarks, w, h)
    def _norm(c, b):
        bx, by, bw, bh = b
        xn, yn = c[0]/float(w), c[1]/float(h)
        return ((xn - bx) / bw, (yn - by) / bh)
    lix, liy = _norm(li, le_box)
    rix, riy = _norm(ri, re_box)
    # center offsets [-0.5,0.5] within box aligned space
    dx = ((lix - 0.5) + (rix - 0.5)) * 0.5
    dy = ((liy - 0.5) + (riy - 0.5)) * 0.5
    adx, ady = abs(dx), abs(dy)
    if adx < focus_th and ady < focus_th:
        dir = "Focused"
    elif adx >= ady:
        dir = "Right" if dx > 0 else "Left"
    else:
        dir = "Down" if dy > 0 else "Up"
    dbg = {"li": [lix, liy], "ri": [rix, riy], "dx": dx, "dy": dy}
    return dir, dbg


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
