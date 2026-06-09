from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

from collection.decoder import load_nomad, stratified_split_70_15_15
from embedding import inference as l2i
from embedding import train_gnn as l2t
from datalog import facts_generator, rule_engine
from risk import alert as l4a
from risk import scorer as l4s


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(REPO_ROOT, "outputs")
FACTS_DIR   = os.path.join(REPO_ROOT, "datalog", "facts")
RULE_OUT_DIR = os.path.join(REPO_ROOT, "datalog", "out")
RISK_PARAMS = os.path.join(REPO_ROOT, "risk", "models", "risk_params.json")


def _violations_by_tx(viol_rows):
    out: Dict[str, List[str]] = {}
    for tx, kind, _sev in viol_rows:
        out.setdefault(tx, []).append(kind)
    return out


def _tune_alpha_beta(
    train_records: List[Dict[str, Any]],
    val_records: List[Dict[str, Any]],
    deposits: List[Dict[str, Any]],
    peckshield,
    g2v_params: Dict[str, Any],
) -> Dict[str, Any]:
    ml_val = l2i.score_records_with_train(
        train_records, val_records,
        peckshield=peckshield, g2v_params=g2v_params,
    )
    tokens = facts_generator.derive_token_map(train_records, deposits)
    facts_generator.write_facts(
        val_records, train_records + val_records, deposits,
        tokens, peckshield, FACTS_DIR,
    )
    l2i.write_ml_facts(ml_val, os.path.join(FACTS_DIR, "ml_result.facts"))
    facts = rule_engine.load_facts(FACTS_DIR)
    res = rule_engine.evaluate(facts)
    viol_by_tx = _violations_by_tx(res["logic_violation"])

    ms = np.array([r["anomaly_score"] for r in ml_val])
    lams = [r["decision"] for r in ml_val]
    kinds = [viol_by_tx.get(r["tx"], []) for r in ml_val]
    y = np.array([1 if r["label"] == "malicious" else 0 for r in ml_val])
    grid = l4s.grid_search_alpha_beta(ms, kinds, lams, y)
    best = grid["best"]
    params = {"alpha": best["alpha"], "beta": best["beta"],
              "val_f1": best["f1"]}
    l4s.save_params(RISK_PARAMS, params)
    print(f"[tune] best alpha={params['alpha']:.2f} beta={params['beta']:.2f}  "
          f"val F1={best['f1']:.4f}")
    return params


def run_pipeline(
    data_dir: str = "data",
    split_name: str = "test",
    seed: int = 42,
    g2v_params: Optional[Dict[str, Any]] = None,
    classifier_kind: str = "lr",
    max_neighbours: int = 5,
    tune: bool = True,
    out_tag: Optional[str] = None,
    write_outputs: bool = True,
) -> Dict[str, Any]:
    if g2v_params is None:
        g2v_params = {"dimensions": 128, "wl_iterations": 2,
                      "attributed": True, "min_count": 1,
                      "epochs": 50, "learning_rate": 0.025,
                      "workers": 2, "seed": seed}
    b = load_nomad(data_dir)
    parts = stratified_split_70_15_15(b["withdrawals"], seed=seed)
    tr, vl, te = parts["train"], parts["val"], parts[split_name]
    peckshield = b["peckshield"]
    deposits = b["deposits"]

    l2t.train_from_records(
        tr, peckshield, dimensions=g2v_params["dimensions"],
        wl_iterations=g2v_params["wl_iterations"],
        epochs=g2v_params["epochs"], seed=g2v_params["seed"],
        classifier_kind=classifier_kind, max_neighbours=max_neighbours,
    )

    if tune:
        params = _tune_alpha_beta(tr, vl, deposits, peckshield, g2v_params)
    else:
        params = l4s.load_params(RISK_PARAMS)

    t0 = time.time()
    ml_rows = l2i.score_records_with_train(
        tr, te, peckshield=peckshield, g2v_params=g2v_params,
        max_neighbours=max_neighbours, classifier_kind=classifier_kind,
    )
    t_l2 = time.time() - t0

    t1 = time.time()
    tokens = facts_generator.derive_token_map(tr, deposits)
    facts_generator.write_facts(
        te, tr + te, deposits, tokens, peckshield, FACTS_DIR,
    )
    l2i.write_ml_facts(ml_rows, os.path.join(FACTS_DIR, "ml_result.facts"))
    facts = rule_engine.load_facts(FACTS_DIR)
    results = rule_engine.evaluate(facts)
    rule_engine.write_outputs(RULE_OUT_DIR, results)
    t_l3 = time.time() - t1

    viol_by_tx = _violations_by_tx(results["logic_violation"])

    t2 = time.time()
    alerts = l4a.score_and_alert(ml_rows, viol_by_tx, params)
    t_l4 = time.time() - t2

    out: Dict[str, Any] = {
        "split": split_name, "seed": seed,
        "n_train": len(tr), "n_val": len(vl), "n_test": len(te),
        "n_train_mal": sum(1 for r in tr if r["label"] == "malicious"),
        "n_test_mal":  sum(1 for r in te if r["label"] == "malicious"),
        "params": params,
        "timing_sec": {"layer2": t_l2, "layer3": t_l3, "layer4": t_l4,
                       "total": t_l2 + t_l3 + t_l4},
        "alerts": alerts,
    }
    if write_outputs:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        tag = out_tag or f"seed{seed}_{split_name}"
        with open(os.path.join(OUTPUT_DIR, f"alerts_{tag}.json"),
                  "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "eval", "all"], default="all")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--classifier", choices=["lr", "rf"], default="lr")
    args = ap.parse_args()

    if args.mode in ("train", "all"):
        l2t.train(args.data_dir, seed=args.seed,
                  classifier_kind=args.classifier)

    if args.mode in ("eval", "all"):
        run_pipeline(args.data_dir, split_name=args.split, seed=args.seed,
                     classifier_kind=args.classifier,
                     tune=(args.mode != "eval"))
