from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import json
from pathlib import Path


@dataclass
class NCAPConfig:
    # Placeholder weights and thresholds — replace with official protocol values.
    perclos_max_ok: float = 15.0
    microsleep_zero_required: bool = True
    eyes_off_road_pct_max: float = 20.0
    eyes_off_events_per_min_max: float = 1.0
    yawn_rate_per_min_max: float = 2.0
    blink_rate_min_ok: float = 6.0
    blink_rate_max_ok: float = 30.0

    # Section weights (heuristic)
    w_vigilance: float = 0.5
    w_distraction: float = 0.4
    w_impairment: float = 0.1


class NCAPScorer:
    def __init__(self, cfg: Optional[NCAPConfig] = None, config_path: Optional[str | Path] = None):
        if config_path:
            loaded = self.load_config(config_path)
            self.cfg = loaded
            self.mode = "config"
            self.ncap_version = "custom"
        else:
            self.cfg = cfg or NCAPConfig()
            self.mode = "heuristic"
            self.ncap_version = "unknown"

    @staticmethod
    def load_config(path: str | Path) -> NCAPConfig:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"NCAP config not found: {p}")
        data = json.loads(p.read_text())
        base = asdict(NCAPConfig())
        base.update(data or {})
        return NCAPConfig(**base)

    def score(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        c = self.cfg

        # Vigilance (drowsiness)
        perclos = float(metrics.get("dms_perclos_pct", 0.0) or 0.0)
        microsleep = bool(metrics.get("dms_microsleep", False))
        blink_rate = float(metrics.get("dms_blinks_per_min", 0.0) or 0.0)
        vigilance = 100.0
        if perclos > c.perclos_max_ok:
            vigilance -= min(100.0, (perclos - c.perclos_max_ok) * 2.0)
        if c.microsleep_zero_required and microsleep:
            vigilance -= 30.0
        if blink_rate < c.blink_rate_min_ok:
            vigilance -= (c.blink_rate_min_ok - blink_rate) * 2.0
        if blink_rate > c.blink_rate_max_ok:
            vigilance -= (blink_rate - c.blink_rate_max_ok) * 1.0
        vigilance = max(0.0, min(100.0, vigilance))

        # Distraction (eyes-off-road)
        off_pct = float(metrics.get("dms_eyes_off_road_pct", 0.0) or 0.0)
        off_ev = float(metrics.get("dms_eyes_off_road_events_per_min", 0.0) or 0.0)
        distraction = 100.0
        if off_pct > c.eyes_off_road_pct_max:
            distraction -= min(100.0, (off_pct - c.eyes_off_road_pct_max) * 1.5)
        if off_ev > c.eyes_off_events_per_min_max:
            distraction -= min(100.0, (off_ev - c.eyes_off_events_per_min_max) * 20.0)
        distraction = max(0.0, min(100.0, distraction))

        # Impairment (yawns as proxy; in reality would need alcohol/drug impairment tests)
        ypm = float(metrics.get("dms_yawns_per_min", 0.0) or 0.0)
        impairment = 100.0
        if ypm > c.yawn_rate_per_min_max:
            impairment -= min(100.0, (ypm - c.yawn_rate_per_min_max) * 5.0)
        impairment = max(0.0, min(100.0, impairment))

        # Optional penalties (if perception provides these flags)
        if metrics.get("oms_phone_use") is True:
            distraction -= 25.0
        if metrics.get("oms_smoking") is True:
            impairment -= 10.0
        if metrics.get("oms_seatbelt_fastened") is False:
            impairment -= 30.0
        # clamp after penalties
        vigilance = max(0.0, min(100.0, vigilance))
        distraction = max(0.0, min(100.0, distraction))
        impairment = max(0.0, min(100.0, impairment))

        overall = (
            vigilance * c.w_vigilance
            + distraction * c.w_distraction
            + impairment * c.w_impairment
        )
        overall = round(overall / (c.w_vigilance + c.w_distraction + c.w_impairment), 1)

        return {
            "mode": getattr(self, 'mode', 'heuristic'),
            "ncap_version": getattr(self, 'ncap_version', 'unknown'),
            "sections": {
                "vigilance": round(vigilance, 1),
                "distraction": round(distraction, 1),
                "impairment": round(impairment, 1),
            },
            "overall": overall,
            "notes": "Heuristic NCAP-like score. Replace config with official protocol to claim compliance.",
        }
