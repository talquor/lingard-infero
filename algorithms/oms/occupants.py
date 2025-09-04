from __future__ import annotations
from typing import Dict, List


def occupant_metrics_from_faces(face_boxes: List[list]) -> Dict[str, int | float | bool]:
    """Basic OMS metrics derived from detected faces."""
    count = len(face_boxes)
    # Placeholder occupancy utilization
    cabin_occupied = count > 0
    return {
        "occupant_count": count,
        "cabin_occupied": cabin_occupied,
    }

