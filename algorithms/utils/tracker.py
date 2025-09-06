from __future__ import annotations
from dataclasses import dataclass, field
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


@dataclass
class Track:
    tid: int
    box: List[float]
    last_ts: float
    miss: int = 0


class BoxTracker:
    def __init__(self, iou_th: float = 0.2, max_miss: int = 5):
        self.iou_th = iou_th
        self.max_miss = max_miss
        self.next_id = 1
        self.tracks: List[Track] = []

    def assign(self, boxes: List[List[float]], ts: float) -> List[int]:
        ids: List[int] = [-1] * len(boxes)
        used = set()
        # Greedy matching by IoU
        for i, b in enumerate(boxes):
            best, best_j = 0.0, -1
            for j, tr in enumerate(self.tracks):
                if j in used:
                    continue
                u = iou(b, tr.box)
                if u > best:
                    best, best_j = u, j
            if best_j >= 0 and best >= self.iou_th:
                tr = self.tracks[best_j]
                tr.box = b
                tr.last_ts = ts
                tr.miss = 0
                ids[i] = tr.tid
                used.add(best_j)

        # New tracks for unmatched boxes
        for i, b in enumerate(boxes):
            if ids[i] == -1:
                tid = self.next_id
                self.next_id += 1
                self.tracks.append(Track(tid=tid, box=b, last_ts=ts))
                ids[i] = tid

        # Age out unmatched tracks
        alive: List[Track] = []
        for j, tr in enumerate(self.tracks):
            if j in used:
                alive.append(tr)
            else:
                tr.miss += 1
                if tr.miss <= self.max_miss:
                    alive.append(tr)
        self.tracks = alive
        return ids

