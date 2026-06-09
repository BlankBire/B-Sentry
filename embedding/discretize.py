from __future__ import annotations

import json
import os
from typing import Dict


DEFAULTS = {"suspicious_low": 0.40, "high_risk_high": 0.75}


def load_thresholds(model_dir: str | None = None) -> Dict[str, float]:
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(__file__), "models")
    path = os.path.join(model_dir, "threshold.json")
    if not os.path.exists(path):
        return DEFAULTS.copy()
    with open(path, "r", encoding="utf-8") as f:
        thr = json.load(f)
    return {
        "suspicious_low": float(thr.get("suspicious_low", DEFAULTS["suspicious_low"])),
        "high_risk_high": float(thr.get("high_risk_high", DEFAULTS["high_risk_high"])),
    }


def discretize(score: float, thresholds: Dict[str, float] | None = None) -> str:
    thr = thresholds or DEFAULTS
    if score >= thr["high_risk_high"]:
        return "high_risk"
    if score >= thr["suspicious_low"]:
        return "suspicious"
    return "normal"
