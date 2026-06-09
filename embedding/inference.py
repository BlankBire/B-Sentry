from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from karateclub import Graph2Vec
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from collection.decoder import (
    load_nomad, stratified_split_70_15_15,
)
from embedding.build_graph import build_graphs
from embedding.discretize import discretize, load_thresholds


MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def _load_train() -> Tuple[List[Dict[str, Any]], Set[str], Dict[str, Any], int]:
    with open(os.path.join(MODEL_DIR, "train_records.pkl"), "rb") as f:
        bundle = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "graph2vec_params.json"), "r",
              encoding="utf-8") as f:
        params = json.load(f)
    return (bundle["records"], set(bundle.get("peckshield") or []),
            params, int(bundle.get("max_neighbours", 5)))


def _build_classifier(kind: str = "lr"):
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=400, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
    return LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)


def _score(
    train_records: List[Dict[str, Any]],
    eval_records: List[Dict[str, Any]],
    *,
    peckshield: Set[str],
    g2v_params: Dict[str, Any],
    max_neighbours: int = 5,
    classifier_kind: str = "lr",
) -> List[Dict[str, Any]]:
    if not eval_records:
        return []
    union = list(train_records) + list(eval_records)
    graphs, kept = build_graphs(union, peckshield=peckshield,
                                 max_neighbours=max_neighbours)
    if not graphs:
        return [{"tx": r.get("tx"), "anomaly_score": 0.0,
                 "decision": "normal", "label": r.get("label")}
                for r in eval_records]
    embedder = Graph2Vec(**g2v_params)
    embedder.fit(graphs)
    X = embedder.get_embedding()

    n_train = len(train_records)
    train_idx = [i for i, k in enumerate(kept) if k < n_train]
    eval_idx  = [i for i, k in enumerate(kept) if k >= n_train]
    X_tr = X[train_idx]
    X_ev = X[eval_idx]
    y_tr = np.array([1 if train_records[kept[i]]["label"] == "malicious" else 0
                     for i in train_idx])

    clf = _build_classifier(classifier_kind)
    if len(set(y_tr)) < 2:
        iso = IsolationForest(n_estimators=200, contamination=0.05,
                              random_state=42, n_jobs=-1)
        iso.fit(X_tr if len(X_tr) else X_ev)
        raw = -iso.decision_function(X_ev)
        lo, hi = float(np.min(raw)), float(np.max(raw))
        probs = ((raw - lo) / (hi - lo + 1e-9)).clip(0.0, 1.0)
    else:
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_ev)[:, 1]

    thr = load_thresholds()
    by_pos = {kept[i] - n_train: float(probs[j])
              for j, i in enumerate(eval_idx)}
    out: List[Dict[str, Any]] = []
    for k, r in enumerate(eval_records):
        s = by_pos.get(k, 0.0)
        out.append({
            "tx": r.get("tx"),
            "block": r.get("block"),
            "anomaly_score": s,
            "decision": discretize(s, thr),
            "label": r.get("label"),
        })
    return out


def score_records(eval_records: List[Dict[str, Any]],
                  classifier_kind: str = "lr") -> List[Dict[str, Any]]:
    train_records, peckshield, params, mn = _load_train()
    return _score(train_records, eval_records,
                  peckshield=peckshield, g2v_params=params,
                  max_neighbours=mn, classifier_kind=classifier_kind)


def score_records_with_train(
    train_records: List[Dict[str, Any]],
    eval_records: List[Dict[str, Any]],
    *,
    peckshield: Set[str],
    g2v_params: Optional[Dict[str, Any]] = None,
    max_neighbours: int = 5,
    classifier_kind: str = "lr",
) -> List[Dict[str, Any]]:
    if g2v_params is None:
        g2v_params = {"dimensions": 128, "wl_iterations": 2,
                      "attributed": True, "min_count": 1,
                      "epochs": 50, "learning_rate": 0.025,
                      "workers": 2, "seed": 42}
    return _score(train_records, eval_records,
                  peckshield=peckshield, g2v_params=g2v_params,
                  max_neighbours=max_neighbours,
                  classifier_kind=classifier_kind)


def write_ml_facts(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row['tx']}\t{row['anomaly_score']:.6f}\t{row['decision']}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="datalog/facts/ml_result.facts")
    args = ap.parse_args()

    b = load_nomad(args.data_dir)
    parts = stratified_split_70_15_15(b["withdrawals"], seed=args.seed)
    t0 = time.time()
    rows = score_records(parts[args.split])
    write_ml_facts(rows, args.out)
    high = sum(r["decision"] == "high_risk" for r in rows)
    susp = sum(r["decision"] == "suspicious" for r in rows)
    norm = sum(r["decision"] == "normal" for r in rows)
    print(f"[infer] scored {len(rows)} records in {time.time()-t0:.2f}s")
    print(f"[infer] decisions: high={high} suspicious={susp} normal={norm}")
    print(f"[infer] -> {args.out}")
