from __future__ import annotations

import csv
import json
import os
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score,
    roc_auc_score,
)


def f1(y, p):       return float(f1_score(y, p, zero_division=0))
def precision(y, p): return float(precision_score(y, p, zero_division=0))
def recall(y, p):    return float(recall_score(y, p, zero_division=0))


def fpr(y, p):
    y = np.asarray(y); p = np.asarray(p)
    fp = int(((p == 1) & (y == 0)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    return float(fp) / float(fp + tn) if (fp + tn) else 0.0


def auroc(y, s):
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return float("nan")


def average_precision(y, s):
    try:
        return float(average_precision_score(y, s))
    except ValueError:
        return float("nan")


def paired_bootstrap(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float] = f1,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    diffs = np.empty(n_resamples, dtype=np.float64)
    for k in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        ya, yb, yc = y_true[idx], pred_a[idx], pred_b[idx]
        diffs[k] = metric(ya, yc) - metric(ya, yb)


    p_pos = float((diffs <= 0).mean())
    p_neg = float((diffs >= 0).mean())
    pval = min(1.0, 2.0 * min(p_pos, p_neg))
    return {
        "diff_mean": float(diffs.mean()),
        "diff_std":  float(diffs.std(ddof=1)),
        "ci95_lo":   float(np.quantile(diffs, 0.025)),
        "ci95_hi":   float(np.quantile(diffs, 0.975)),
        "p_value":   pval,
    }


def write_csv(path: str, rows: Sequence[Dict[str, Any]],
              field_order: Sequence[str] | None = None) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = list(field_order) if field_order else list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def dump_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
