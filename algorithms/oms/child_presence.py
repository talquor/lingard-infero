from __future__ import annotations
from dataclasses import dataclass
from typing import Dict


@dataclass
class ChildPresenceState:
    pass


class ChildPresenceEstimator:
    def update(self, ts: float, frame=None) -> Dict[str, bool | None]:
        # Placeholder: requires child classifier and seat localization
        return {
            "child_present": None
        }

