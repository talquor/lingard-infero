from __future__ import annotations
from dataclasses import dataclass
from typing import Dict


@dataclass
class SmokingState:
    pass


class SmokingEstimator:
    def update(self, ts: float, frame=None) -> Dict[str, bool | None]:
        # Placeholder: requires object detector for cigarette/vape
        return {
            "smoking": None
        }

