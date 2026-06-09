from __future__ import annotations

from typing import Any, Dict, List

from risk.scorer import datalog_severity, risk_score


def classify(lambda_tau: str, sigma_tau: int) -> str:
    ml_active = (lambda_tau == "high_risk")
    dl_active = (sigma_tau >= 2)
    if ml_active and dl_active:
        return "critical"
    if ml_active ^ dl_active:
        return "suspicious"
    return "normal"


def score_and_alert(
    ml_rows: List[Dict[str, Any]],
    violations_by_tx: Dict[str, List[str]],
    params: Dict[str, float],
) -> List[Dict[str, Any]]:
    alpha = float(params.get("alpha", 0.4))
    beta = float(params.get("beta", 1.0 - alpha))

    out: List[Dict[str, Any]] = []
    for row in ml_rows:
        tx = row["tx"]
        ml_dec = row.get("decision", "normal")
        ml_score = row.get("anomaly_score", 0.0)
        kinds = violations_by_tx.get(tx, [])
        r, sigma, top = risk_score(ml_score, kinds, alpha, beta)
        tier = classify(ml_dec, sigma)
        out.append({
            "tx": tx,
            "label": row.get("label"),
            "ml_score": float(ml_score),
            "ml_decision": ml_dec,
            "violations": kinds,
            "top_violation": top,
            "sigma": sigma,
            "risk_score": r,
            "alert": tier,
        })
    return out
