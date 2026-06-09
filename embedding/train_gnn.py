from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from karateclub import Graph2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, classification_report, f1_score,
    precision_recall_curve, roc_auc_score,
)

from collection.decoder import (
    load_nomad, stratified_split_70_15_15,
)
from embedding.build_graph import build_graphs


MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def _y(records, kept):
    return np.array([1 if records[i]["label"] == "malicious" else 0 for i in kept])


def _pick_thresholds(probs: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if y.sum() < 5 or (y == 0).sum() < 5:
        return 0.40, 0.75
    q95_norm = float(np.quantile(probs[y == 0], 0.95))
    q25_mal  = float(np.quantile(probs[y == 1], 0.25))
    theta1 = max(0.30, q95_norm)
    theta2 = max(theta1 + 0.05, q25_mal)
    theta1 = min(theta1, 0.70)
    theta2 = min(theta2, 0.90)
    return float(theta1), float(theta2)


def _build_classifier(kind: str):
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=400, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
    return LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)


def train(
    data_dir: str = "data",
    dimensions: int = 128,
    wl_iterations: int = 2,
    epochs: int = 50,
    seed: int = 42,
    classifier_kind: str = "lr",
    max_neighbours: int = 5,
) -> Dict[str, Any]:
    os.makedirs(MODEL_DIR, exist_ok=True)
    bundle = load_nomad(data_dir)
    parts = stratified_split_70_15_15(bundle["withdrawals"], seed=seed)
    peckshield = bundle["peckshield"]
    tr, vl = parts["train"], parts["val"]
    print(f"[train] split (seed={seed})  train={len(tr)}  val={len(vl)}  "
          f"peckshield={len(peckshield)}")

    t0 = time.time()
    graphs_tr, kept_tr = build_graphs(tr, peckshield=peckshield,
                                        max_neighbours=max_neighbours)
    graphs_vl, kept_vl = build_graphs(vl, peckshield=peckshield,
                                        max_neighbours=max_neighbours)
    y_tr = _y(tr, kept_tr); y_vl = _y(vl, kept_vl)
    print(f"[train] train graphs={len(graphs_tr)} (mal={int(y_tr.sum())})  "
          f"val graphs={len(graphs_vl)} (mal={int(y_vl.sum())})")

    params = {
        "dimensions": dimensions, "wl_iterations": wl_iterations,
        "attributed": True, "min_count": 1,
        "epochs": epochs, "learning_rate": 0.025,
        "workers": 2, "seed": seed,
    }
    embedder = Graph2Vec(**params)
    embedder.fit(graphs_tr + graphs_vl)
    emb = embedder.get_embedding()
    X_tr = emb[: len(graphs_tr)]
    X_vl = emb[len(graphs_tr):]

    clf = _build_classifier(classifier_kind)
    clf.fit(X_tr, y_tr)
    probs_vl = clf.predict_proba(X_vl)[:, 1]
    pred_vl = (probs_vl >= 0.5).astype(int)
    f1 = float(f1_score(y_vl, pred_vl, zero_division=0))
    try:
        auroc = float(roc_auc_score(y_vl, probs_vl))
        ap    = float(average_precision_score(y_vl, probs_vl))
    except ValueError:
        auroc = ap = float("nan")
    print("[train] val report:\n"
          + classification_report(y_vl, pred_vl, digits=4, zero_division=0))
    print(f"[train] val AUROC={auroc:.4f}  AP={ap:.4f}  F1={f1:.4f}")

    theta1, theta2 = _pick_thresholds(probs_vl, y_vl)
    print(f"[train] thresholds  theta1={theta1:.3f}  theta2={theta2:.3f}")

    with open(os.path.join(MODEL_DIR, "train_records.pkl"), "wb") as f:
        pickle.dump({"records": tr,
                     "labels": [1 if r["label"] == "malicious" else 0 for r in tr],
                     "peckshield": sorted(peckshield),
                     "max_neighbours": max_neighbours}, f)
    with open(os.path.join(MODEL_DIR, "graph2vec_params.json"), "w",
              encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    with open(os.path.join(MODEL_DIR, "threshold.json"), "w",
              encoding="utf-8") as f:
        json.dump({"suspicious_low": theta1, "high_risk_high": theta2}, f, indent=2)

    meta = {"params": params, "classifier_kind": classifier_kind,
            "n_train": len(tr), "n_val": len(vl),
            "val_f1": f1, "val_auroc": auroc, "val_ap": ap,
            "elapsed_sec": float(time.time() - t0),
            "max_neighbours": max_neighbours}
    with open(os.path.join(MODEL_DIR, "train_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[train] done in {time.time()-t0:.1f}s -> {MODEL_DIR}")
    return meta


def train_from_records(
    train_records: List[Dict[str, Any]],
    peckshield: Set[str],
    *,
    dimensions: int = 128,
    wl_iterations: int = 2,
    epochs: int = 50,
    seed: int = 42,
    classifier_kind: str = "lr",
    max_neighbours: int = 5,
) -> Dict[str, Any]:
    os.makedirs(MODEL_DIR, exist_ok=True)
    params = {
        "dimensions": dimensions, "wl_iterations": wl_iterations,
        "attributed": True, "min_count": 1,
        "epochs": epochs, "learning_rate": 0.025,
        "workers": 2, "seed": seed,
    }
    with open(os.path.join(MODEL_DIR, "train_records.pkl"), "wb") as f:
        pickle.dump({"records": train_records,
                     "labels": [1 if r["label"] == "malicious" else 0
                                for r in train_records],
                     "peckshield": sorted(peckshield),
                     "max_neighbours": max_neighbours}, f)
    with open(os.path.join(MODEL_DIR, "graph2vec_params.json"), "w",
              encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    return {"params": params, "classifier_kind": classifier_kind,
            "n_train": len(train_records),
            "max_neighbours": max_neighbours}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--wl", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--classifier", choices=["lr", "rf"], default="lr")
    ap.add_argument("--neighbours", type=int, default=5)
    args = ap.parse_args()
    train(args.data_dir, dimensions=args.dim, wl_iterations=args.wl,
          epochs=args.epochs, seed=args.seed,
          classifier_kind=args.classifier,
          max_neighbours=args.neighbours)
