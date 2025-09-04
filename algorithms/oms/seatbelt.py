from __future__ import annotations
from dataclasses import dataclass
from typing import Dict


@dataclass
class SeatbeltState:
    pass


class SeatbeltEstimator:
    def update(self, ts: float, frame=None) -> Dict[str, bool | None]:
        # Placeholder: needs belt latch or vision model
        return {
            "seatbelt_fastened": None
        }

