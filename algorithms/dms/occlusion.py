from __future__ import annotations
from dataclasses import dataclass
from typing import Dict


@dataclass
class OcclusionState:
    pass


class OcclusionEstimator:
    def update(self, ts: float, landmarks=None) -> Dict[str, bool | None]:
        # Placeholder: requires face parts visibility checks
        return {
            "face_occluded": None
        }

