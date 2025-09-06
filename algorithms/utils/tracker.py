from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple


def iou(a: List[float], b: List[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = aw * ah
    area_b = bw * bh
    return inter / (area_a + area_b - inter + 1e-12)


def center(box: List[float]) -> Tuple[float, float]:
    x, y, w, h = box
    return (x + w * 0.5, y + h * 0.5)


@dataclass
class Track:
    tid: int
    box: List[float]
    last_ts: float
    miss: int = 0


class BoxTracker:
    def __init__(
        self,
        iou_th: float = 0.2,
        center_th: float = 0.15,
        max_miss: int = 5,
        match_by_center: bool = True,
    ):
        self.iou_th = float(iou_th)
        self.center_th = float(center_th)
        self.max_miss = int(max_miss)
        self.match_by_center = bool(match_by_center)
        self.next_id = 1
        self.tracks: List[Track] = []

    def configure(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k) and v is not None:
                setattr(self, k, v)

    def assign(self, boxes: List[List[float]], ts: float) -> List[int]:
        n = len(boxes)
        ids: List[int] = [-1] * n
        if not self.tracks:
            for i in range(n):
                tid = self.next_id; self.next_id += 1
                self.tracks.append(Track(tid, boxes[i], ts))
                ids[i] = tid
            return ids

        # Stage 1: IoU-based greedy matching
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_dets = set(range(n))
        pairs: List[Tuple[int, int, float]] = []  # (di, tj, iou)
        for i in range(n):
            for j in range(len(self.tracks)):
                u = iou(boxes[i], self.tracks[j].box)
                if u >= self.iou_th:
                    pairs.append((i, j, u))
        pairs.sort(key=lambda x: x[2], reverse=True)
        for i, j, u in pairs:
            if i in unmatched_dets and j in unmatched_tracks:
                tr = self.tracks[j]
                tr.box = boxes[i]
                tr.last_ts = ts
                tr.miss = 0
                ids[i] = tr.tid
                unmatched_dets.remove(i)
                unmatched_tracks.remove(j)

        # Stage 2: center-distance fallback for remaining
        if self.match_by_center and unmatched_dets and unmatched_tracks:
            # Build candidate list with distances
            cand: List[Tuple[float, int, int]] = []  # (dist, di, tj)
            for i in unmatched_dets:
                cx, cy = center(boxes[i])
                for j in unmatched_tracks:
                    tx, ty = center(self.tracks[j].box)
                    dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
                    if dist <= self.center_th:
                        cand.append((dist, i, j))
            cand.sort(key=lambda x: x[0])
            for dist, i, j in cand:
                if i in unmatched_dets and j in unmatched_tracks:
                    tr = self.tracks[j]
                    tr.box = boxes[i]
                    tr.last_ts = ts
                    tr.miss = 0
                    ids[i] = tr.tid
                    unmatched_dets.remove(i)
                    unmatched_tracks.remove(j)

        # New tracks for any still unmatched detections
        for i in list(unmatched_dets):
            tid = self.next_id; self.next_id += 1
            self.tracks.append(Track(tid, boxes[i], ts))
            ids[i] = tid

        # Age out unmatched tracks
        alive: List[Track] = []
        for j, tr in enumerate(self.tracks):
            if j in unmatched_tracks:
                tr.miss += 1
                if tr.miss <= self.max_miss:
                    alive.append(tr)
            else:
                alive.append(tr)
        self.tracks = alive
        return ids
