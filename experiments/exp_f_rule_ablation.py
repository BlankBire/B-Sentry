from __future__ import annotations

import os
from typing import Any, Dict, List, Set, Tuple

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


R_KINDS = {
    "R1_forged_withdrawal":     {"forged_withdrawal"},
    "R2_invalid_token_mapping": {"invalid_token_mapping"},
    "R3_peckshield_listed":     {"peckshield_listed"},
}


SUBSETS: List[Tuple[str, Set[str]]] = [
    ("R1",               {"forged_withdrawal"}),
    ("R1+R2",            {"forged_withdrawal", "invalid_token_mapping"}),
    ("R1+R2+R3 (full)",  {"forged_withdrawal", "invalid_token_mapping",
                          "peckshield_listed"}),


]


def _filter_violations(viol_by_tx: Dict[str, List[str]],
                       allowed: Set[str]) -> Dict[str, List[str]]:
    return {tx: [k for k in kinds if k in allowed]
            for tx, kinds in viol_by_tx.items()
            if any(k in allowed for k in kinds)}


def _datalog_run(eval_records, train_records, deposits, peckshield):
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


def _eval_config(y_true, dl_pred, ml_rows, viol_filtered, alpha, beta):
    p_dl = {
        "precision": precision(y_true, dl_pred),
        "recall":    recall(y_true, dl_pred),
        "f1":        f1(y_true, dl_pred),
        "fpr":       fpr(y_true, dl_pred),
        "auroc":     auroc(y_true, dl_pred.astype(float)),
        "ap":        average_precision(y_true, dl_pred.astype(float)),
    }
    alerts = l4a.score_and_alert(ml_rows, viol_filtered,
                                  {"alpha": alpha, "beta": beta})
    cs_pred = np.array([1 if a["alert"] == "critical" else 0 for a in alerts])
    cs_risk = np.array([a["risk_score"] for a in alerts])
    p_cs = {
        "precision": precision(y_true, cs_pred),
        "recall":    recall(y_true, cs_pred),
        "f1":        f1(y_true, cs_pred),
        "fpr":       fpr(y_true, cs_pred),
        "auroc":     auroc(y_true, cs_risk),
        "ap":        average_precision(y_true, cs_risk),
    }
    return p_dl, p_cs


def _one_seed(bundle, seed: int) -> List[Dict[str, Any]]:
    parts = stratified_split_70_15_15(bundle["withdrawals"], seed=seed)
    tr, vl, te = parts["train"], parts["val"], parts["test"]
    peckshield = bundle["peckshield"]
    deposits = bundle["deposits"]
    y_true = np.array([1 if r["label"] == "malicious" else 0 for r in te])

    g2v = {"dimensions": 128, "wl_iterations": 2, "attributed": True,
           "min_count": 1, "epochs": 50, "learning_rate": 0.025,
           "workers": 2, "seed": seed}


    ml_rows = l2i.score_records_with_train(
        tr, te, peckshield=peckshield, g2v_params=g2v, max_neighbours=5,
    )
    viol_all_te = _datalog_run(te, tr, deposits, peckshield)


    val_rows = l2i.score_records_with_train(
        tr, vl, peckshield=peckshield, g2v_params=g2v, max_neighbours=5,
    )
    viol_all_vl = _datalog_run(vl, tr, deposits, peckshield)
    ms_val = np.array([r["anomaly_score"] for r in val_rows])
    lam_val = [r["decision"] for r in val_rows]
    kinds_val = [viol_all_vl.get(r["tx"], []) for r in val_rows]
    y_val = np.array([1 if r["label"] == "malicious" else 0 for r in val_rows])
    best = l4s.grid_search_alpha_beta(ms_val, kinds_val, lam_val, y_val)["best"]
    alpha, beta = best["alpha"], best["beta"]

    rows: List[Dict[str, Any]] = []
    for subset_name, allowed in SUBSETS:
        viol_filtered = _filter_violations(viol_all_te, allowed)
        dl_pred = np.array(
            [1 if r["tx"] in viol_filtered else 0 for r in te])
        p_dl, p_cs = _eval_config(
            y_true, dl_pred, ml_rows, viol_filtered, alpha, beta,
        )
        rows.append({"subset": subset_name, "seed": seed,
                     "config": "Datalog-only", **p_dl})
        rows.append({"subset": subset_name, "seed": seed,
                     "config": "B-Sentry-critical", **p_cs})
    return rows


def main():
    bundle = load_nomad("data")
    all_rows: List[Dict[str, Any]] = []
    for s in SEEDS:
        print(f"--- seed {s} ---", flush=True)
        all_rows.extend(_one_seed(bundle, s))


    summary: List[Dict[str, Any]] = []
    for subset_name, _ in SUBSETS:
        for config in ("Datalog-only", "B-Sentry-critical"):
            sel = [r for r in all_rows
                   if r["subset"] == subset_name and r["config"] == config]
            row = {"subset": subset_name, "config": config}
            for k in ("precision", "recall", "f1", "fpr", "auroc", "ap"):
                vals = np.array([s[k] for s in sel], dtype=float)
                vals = vals[~np.isnan(vals)]
                row[f"{k}_mean"] = float(vals.mean()) if vals.size else float("nan")
                row[f"{k}_std"]  = float(vals.std(ddof=0)) if vals.size else float("nan")
            summary.append(row)

    os.makedirs("results", exist_ok=True)
    write_csv("results/exp_f_rule_ablation.csv", all_rows,
              field_order=["subset", "config", "seed", "precision", "recall",
                           "f1", "fpr", "auroc", "ap"])
    write_csv("results/exp_f_rule_ablation_summary.csv", summary,
              field_order=["subset", "config",
                           "precision_mean", "precision_std",
                           "recall_mean", "recall_std",
                           "f1_mean", "f1_std",
                           "fpr_mean", "fpr_std",
                           "auroc_mean", "auroc_std",
                           "ap_mean", "ap_std"])
    dump_json("outputs/exp_f_rule_ablation.json", {
        "seeds": SEEDS,
        "subsets": [name for name, _ in SUBSETS],
        "per_seed": all_rows, "summary": summary,
    })


    print("\n=== Exp-F per-rule contribution (mean over seeds) ===")
    head = f"{'subset':<22} {'config':<22} {'P':>14} {'R':>14} {'F1':>14} {'FPR':>14}"
    print(head); print("-" * len(head))
    for row in summary:
        print(f"{row['subset']:<22} {row['config']:<22} "
              f"{row['precision_mean']:.4f}+-{row['precision_std']:.4f} "
              f"{row['recall_mean']:.4f}+-{row['recall_std']:.4f} "
              f"{row['f1_mean']:.4f}+-{row['f1_std']:.4f} "
              f"{row['fpr_mean']:.4f}+-{row['fpr_std']:.4f}")
    print("\n -> results/exp_f_rule_ablation.csv")


if __name__ == "__main__":
    main()
