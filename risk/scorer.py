from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


SEVERITY: Dict[str, int] = {
    "forged_withdrawal":     3,
    "forged_deposit":        2,
    "finality_breach":       2,
    "peckshield_listed":     2,
    "invalid_token_mapping": 1,
}


def datalog_severity(violations: Iterable[str]) -> Tuple[int, str | None]:
    best = 0
    kind = None
    for v in violations:
        s = SEVERITY.get(v, 0)
        if s > best:
            best, kind = s, v
    return best, kind


def risk_score(ml_score: float, violations: Iterable[str],
               alpha: float, beta: float) -> Tuple[float, int, str | None]:
    sigma, kind = datalog_severity(violations)
    r = alpha * float(ml_score) + beta * (sigma / 3.0)
    return float(min(1.0, max(0.0, r))), sigma, kind


def grid_search_alpha_beta(
    ml_scores: np.ndarray,
    violation_kinds: List[List[str]],
    ml_labels: List[str],
    y_true: np.ndarray,
    alpha_grid: Iterable[float] = (0.3, 0.4, 0.5, 0.6, 0.7),
) -> Dict[str, Any]:
    from sklearn.metrics import f1_score, precision_score, recall_score
    n = len(ml_scores)
    best = {"f1": -1.0}
    curve = []
    sigmas = [datalog_severity(k)[0] for k in violation_kinds]
    from risk.alert import classify
    for alpha in alpha_grid:
        beta = 1.0 - float(alpha)
        risks = np.array([alpha * float(ml_scores[i])
                          + beta * (sigmas[i] / 3.0)
                          for i in range(n)])
        tiers = [classify(ml_labels[i], sigmas[i]) for i in range(n)]
        preds = np.array([1 if t in ("critical", "suspicious") else 0
                          for t in tiers])
        f1 = float(f1_score(y_true, preds, zero_division=0))
        p  = float(precision_score(y_true, preds, zero_division=0))
        r  = float(recall_score(y_true, preds, zero_division=0))
        curve.append({"alpha": alpha, "beta": beta, "f1": f1,
                      "precision": p, "recall": r,
                      "mean_risk_mal": float(np.mean(risks[y_true == 1])
                                              if (y_true == 1).any() else 0.0),
                      "mean_risk_norm": float(np.mean(risks[y_true == 0])
                                               if (y_true == 0).any() else 0.0)})
        if f1 > best["f1"] + 1e-9:
            best = {"alpha": float(alpha), "beta": float(beta),
                    "f1": f1, "precision": p, "recall": r}
    return {"best": best, "curve": curve}


def save_params(path: str, params: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)


def load_params(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"alpha": 0.4, "beta": 0.6}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
