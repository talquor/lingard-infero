from __future__ import annotations
from dataclasses import dataclass
from typing import Dict


@dataclass
class HandsOnWheelState:
    pass


class HandsOnWheelEstimator:
    def update(self, ts: float, frame=None) -> Dict[str, bool | None]:
        # Placeholder: requires hand keypoints + wheel region
        return {
            "hands_on_wheel": None
        }

