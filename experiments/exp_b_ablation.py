from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np

from collection.decoder import load_nomad, stratified_split_70_15_15
from embedding import inference as l2i
from datalog import facts_generator, rule_engine
from risk import alert as l4a
from risk import scorer as l4s
from experiments.stats import (
    auroc, average_precision, dump_json, f1, fpr, precision, recall, write_csv,
)
from pipeline import FACTS_DIR


_SEEDS_ENV = os.environ.get("EXP_SEEDS")
SEEDS = ([int(s) for s in _SEEDS_ENV.split(",")] if _SEEDS_ENV
         else [1, 2, 3, 4, 5])
CONFIGS = ("B-Sentry", "no-L2", "no-L3")


def _datalog_violations(eval_records, train_records, deposits, peckshield):
    tokens = facts_generator.derive_token_map(train_records, deposits)
    facts_generator.write_facts(
        eval_records, train_records + eval_records, deposits,
        tokens, peckshield, FACTS_DIR,
    )
    open(os.path.join(FACTS_DIR, "ml_result.facts"), "w").close()
    facts = rule_engine.load_facts(FACTS_DIR)
    res = rule_engine.evaluate(facts)
    out: Dict[str, List[str]] = {}
    for tx, k, _v in res["logic_violation"]:
        out.setdefault(tx, []).append(k)
    return out


def _one_seed(bundle, seed: int) -> Dict[str, Any]:
    parts = stratified_split_70_15_15(bundle["withdrawals"], seed=seed)
    tr, vl, te = parts["train"], parts["val"], parts["test"]
    peckshield = bundle["peckshield"]; deposits = bundle["deposits"]
    y = np.array([1 if r["label"] == "malicious" else 0 for r in te])

    g2v = {"dimensions": 128, "wl_iterations": 2, "attributed": True,
           "min_count": 1, "epochs": 50, "learning_rate": 0.025,
           "workers": 2, "seed": seed}

    g2v_rows = l2i.score_records_with_train(
        tr, te, peckshield=peckshield, g2v_params=g2v, max_neighbours=5,
    )
    g2v_score = np.array([r["anomaly_score"] for r in g2v_rows])
    g2v_dec = np.array([1 if r["decision"] == "high_risk" else 0
                        for r in g2v_rows])

    viol_by_tx = _datalog_violations(te, tr, deposits, peckshield)
    dl_pred = np.array([1 if r["tx"] in viol_by_tx else 0 for r in te])

    val_rows = l2i.score_records_with_train(
        tr, vl, peckshield=peckshield, g2v_params=g2v, max_neighbours=5,
    )
    viol_val = _datalog_violations(vl, tr, deposits, peckshield)
    ms_val = np.array([r["anomaly_score"] for r in val_rows])
    lam_val = [r["decision"] for r in val_rows]
    kinds_val = [viol_val.get(r["tx"], []) for r in val_rows]
    y_val = np.array([1 if r["label"] == "malicious" else 0 for r in val_rows])
    best = l4s.grid_search_alpha_beta(ms_val, kinds_val, lam_val, y_val)["best"]
    params = {"alpha": best["alpha"], "beta": best["beta"]}
    alerts = l4a.score_and_alert(g2v_rows, viol_by_tx, params)

    cs_pred = np.array([1 if a["alert"] == "critical" else 0 for a in alerts])
    cs_risk = np.array([a["risk_score"] for a in alerts])

    return {
        "y_true": y,
        "preds": {"B-Sentry": cs_pred, "no-L2": dl_pred, "no-L3": g2v_dec},
        "scores": {"B-Sentry": cs_risk, "no-L2": dl_pred.astype(float),
                   "no-L3": g2v_score},
    }


def main():
    bundle = load_nomad("data")
    rows_per_seed: List[Dict[str, Any]] = []
    for s in SEEDS:
        print(f"--- seed {s} ---", flush=True)
        r = _one_seed(bundle, s)
        y = r["y_true"]
        for cfg in CONFIGS:
            p = r["preds"][cfg]; sc = r["scores"][cfg]
            rows_per_seed.append({
                "config": cfg, "seed": s,
                "precision": precision(y, p), "recall": recall(y, p),
                "f1": f1(y, p), "fpr": fpr(y, p),
                "auroc": auroc(y, sc), "ap": average_precision(y, sc),
            })

    summary: List[Dict[str, Any]] = []
    for cfg in CONFIGS:
        vals = {k: [] for k in ("precision", "recall", "f1", "fpr",
                                "auroc", "ap")}
        for r in rows_per_seed:
            if r["config"] == cfg:
                for k in vals:
                    vals[k].append(r[k])
        row = {"config": cfg}
        for k, v in vals.items():
            arr = np.asarray(v, dtype=float); arr = arr[~np.isnan(arr)]
            row[f"{k}_mean"] = float(arr.mean()) if arr.size else float("nan")
            row[f"{k}_std"]  = float(arr.std(ddof=0)) if arr.size else float("nan")
        summary.append(row)

    os.makedirs("results", exist_ok=True)
    write_csv("results/exp_b_ablation.csv", summary,
              field_order=["config",
                           "precision_mean", "precision_std",
                           "recall_mean", "recall_std",
                           "f1_mean", "f1_std",
                           "fpr_mean", "fpr_std",
                           "auroc_mean", "auroc_std",
                           "ap_mean", "ap_std"])
    write_csv("results/exp_b_per_seed.csv", rows_per_seed,
              field_order=["config", "seed", "precision", "recall", "f1",
                           "fpr", "auroc", "ap"])
    dump_json("outputs/exp_b_ablation.json", {
        "seeds": SEEDS, "configs": list(CONFIGS),
        "per_seed": rows_per_seed, "summary": summary,
    })

    print("\n=== Exp-B ablation summary ===")
    h = f"{'config':<16} {'F1':>14} {'P':>14} {'R':>14} {'FPR':>14}"
    print(h); print("-" * len(h))
    for r in summary:
        print(f"{r['config']:<16} "
              f"{r['f1_mean']:>7.4f}+-{r['f1_std']:.4f} "
              f"{r['precision_mean']:>7.4f}+-{r['precision_std']:.4f} "
              f"{r['recall_mean']:>7.4f}+-{r['recall_std']:.4f} "
              f"{r['fpr_mean']:>7.4f}+-{r['fpr_std']:.4f}")
    print("\n -> results/exp_b_ablation.csv (+ exp_b_per_seed.csv)")


if __name__ == "__main__":
    main()
