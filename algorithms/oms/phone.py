from __future__ import annotations
from dataclasses import dataclass
from typing import Dict


@dataclass
class PhoneUseState:
    # Placeholder; in production use a hand-object detector + phone classifier
    last_seen_ts: float | None = None


class PhoneUseEstimator:
    def update(self, ts: float, frame=None) -> Dict[str, bool | float | None]:
        # Placeholder: unknown unless integrated with detector
        return {
            "phone_use": None,
        }

