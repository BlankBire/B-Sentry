from __future__ import annotations

import os
import time
from typing import Any, Dict, List

import numpy as np

from collection.decoder import load_nomad, stratified_split_70_15_15
from embedding import inference as l2i
from embedding.gnn_baselines import score_with_gcn, score_with_sage
from datalog import facts_generator, rule_engine
from risk import alert as l4a
from risk import scorer as l4s
from experiments.stats import (
    auroc, average_precision, dump_json, f1, fpr, paired_bootstrap,
    precision, recall, write_csv,
)
from pipeline import FACTS_DIR


_SEEDS_ENV = os.environ.get("EXP_SEEDS")
SEEDS = ([int(s) for s in _SEEDS_ENV.split(",")] if _SEEDS_ENV
         else [1, 2, 3, 4, 5])
MODELS = ("XChainWatcher", "Graph2Vec", "GCN", "GraphSAGE", "B-Sentry")


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


def _score_one_seed(bundle, seed: int) -> Dict[str, Any]:
    parts = stratified_split_70_15_15(bundle["withdrawals"], seed=seed)
    tr, vl, te = parts["train"], parts["val"], parts["test"]
    peckshield = bundle["peckshield"]; deposits = bundle["deposits"]
    y_true = np.array([1 if r["label"] == "malicious" else 0 for r in te])

    g2v_params = {"dimensions": 128, "wl_iterations": 2, "attributed": True,
                  "min_count": 1, "epochs": 50, "learning_rate": 0.025,
                  "workers": 2, "seed": seed}

    out: Dict[str, Any] = {
        "seed": seed, "n_test": len(te), "n_test_mal": int(y_true.sum()),
        "n_train": len(tr),
        "n_train_mal": sum(1 for r in tr if r["label"] == "malicious"),
    }


    viol_by_tx = _datalog_violations(te, tr, deposits, peckshield)
    dl_pred = np.array([1 if r["tx"] in viol_by_tx else 0 for r in te])


    t0 = time.time()
    g2v_rows = l2i.score_records_with_train(
        tr, te, peckshield=peckshield, g2v_params=g2v_params, max_neighbours=5,
    )
    out["timing_g2v_sec"] = time.time() - t0
    g2v_scores = np.array([r["anomaly_score"] for r in g2v_rows])
    g2v_pred = np.array([1 if r["decision"] == "high_risk" else 0
                         for r in g2v_rows])


    t0 = time.time()
    gcn_rows = score_with_gcn(
        tr, te, peckshield=peckshield, max_neighbours=5,
        seed=seed, epochs=30, hidden=64,
    )
    out["timing_gcn_sec"] = time.time() - t0
    gcn_scores = np.array([r["anomaly_score"] for r in gcn_rows])
    gcn_pred = np.array([1 if r["decision"] == "high_risk" else 0
                         for r in gcn_rows])


    t0 = time.time()
    sage_rows = score_with_sage(
        tr, te, peckshield=peckshield, max_neighbours=5,
        seed=seed, epochs=30, hidden=64,
    )
    out["timing_sage_sec"] = time.time() - t0
    sage_scores = np.array([r["anomaly_score"] for r in sage_rows])
    sage_pred = np.array([1 if r["decision"] == "high_risk" else 0
                          for r in sage_rows])


    val_rows = l2i.score_records_with_train(
        tr, vl, peckshield=peckshield, g2v_params=g2v_params, max_neighbours=5,
    )
    viol_val = _datalog_violations(vl, tr, deposits, peckshield)
    ms_val = np.array([r["anomaly_score"] for r in val_rows])
    lam_val = [r["decision"] for r in val_rows]
    kinds_val = [viol_val.get(r["tx"], []) for r in val_rows]
    y_val = np.array([1 if r["label"] == "malicious" else 0 for r in val_rows])
    best = l4s.grid_search_alpha_beta(ms_val, kinds_val, lam_val, y_val)["best"]
    params = {"alpha": best["alpha"], "beta": best["beta"]}
    alerts = l4a.score_and_alert(g2v_rows, viol_by_tx, params)
    cs_risk = np.array([a["risk_score"] for a in alerts])


    cs_pred = np.array([1 if a["alert"] == "critical" else 0
                        for a in alerts])
    out["alpha"] = params["alpha"]; out["beta"] = params["beta"]

    out["preds"] = {
        "XChainWatcher": dl_pred.tolist(),
        "Graph2Vec":     g2v_pred.tolist(),
        "GCN":           gcn_pred.tolist(),
        "GraphSAGE":     sage_pred.tolist(),
        "B-Sentry":      cs_pred.tolist(),
    }
    out["scores"] = {
        "XChainWatcher": dl_pred.tolist(),
        "Graph2Vec":     g2v_scores.tolist(),
        "GCN":           gcn_scores.tolist(),
        "GraphSAGE":     sage_scores.tolist(),
        "B-Sentry":      cs_risk.tolist(),
    }
    out["y_true"] = y_true.tolist()
    return out


def _per_seed_metrics(seed_run: Dict[str, Any]) -> List[Dict[str, Any]]:
    y = np.asarray(seed_run["y_true"])
    rows = []
    for m in MODELS:
        p = np.asarray(seed_run["preds"][m])
        s = np.asarray(seed_run["scores"][m])
        rows.append({
            "model": m, "seed": seed_run["seed"],
            "precision": precision(y, p), "recall": recall(y, p),
            "f1": f1(y, p), "fpr": fpr(y, p),
            "auroc": auroc(y, s), "ap": average_precision(y, s),
            "n_test": int(seed_run["n_test"]),
            "n_test_mal": int(seed_run["n_test_mal"]),
        })
    return rows


def _summary(per_seed_rows, seed_runs):
    summary: Dict[str, Dict[str, List[float]]] = {
        m: {k: [] for k in ("precision", "recall", "f1", "fpr", "auroc", "ap")}
        for m in MODELS
    }
    for r in per_seed_rows:
        for k in summary[r["model"]]:
            summary[r["model"]][k].append(r[k])

    boot: Dict[str, List[Dict[str, float]]] = {m: [] for m in MODELS
                                                if m != "B-Sentry"}
    for run in seed_runs:
        y = np.asarray(run["y_true"])
        cs = np.asarray(run["preds"]["B-Sentry"])
        for m in boot.keys():
            other = np.asarray(run["preds"][m])
            boot[m].append(paired_bootstrap(
                y, other, cs, metric=f1,
                n_resamples=10_000, seed=run["seed"] * 100 + 7,
            ))

    rows = []
    for m in MODELS:
        d = summary[m]
        row = {"model": m}
        for k in ("precision", "recall", "f1", "fpr", "auroc", "ap"):
            vals = np.asarray(d[k], dtype=float)
            vals = vals[~np.isnan(vals)]
            row[f"{k}_mean"] = float(vals.mean()) if vals.size else float("nan")
            row[f"{k}_std"]  = float(vals.std(ddof=0)) if vals.size else float("nan")
        if m != "B-Sentry":
            ps = [b["p_value"] for b in boot[m]]
            row["p_value_vs_BS_F1_min"]  = float(min(ps))
            row["p_value_vs_BS_F1_mean"] = float(np.mean(ps))
            row["F1_diff_mean_BS_minus"] = float(np.mean(
                [b["diff_mean"] for b in boot[m]]))
        rows.append(row)
    return rows


def main():
    bundle = load_nomad("data")

    seed_runs: List[Dict[str, Any]] = []
    per_seed_rows: List[Dict[str, Any]] = []
    for s in SEEDS:
        print(f"\n--- seed {s} ---", flush=True)
        sr = _score_one_seed(bundle, seed=s)
        seed_runs.append(sr)
        per_seed_rows.extend(_per_seed_metrics(sr))

    summary_rows = _summary(per_seed_rows, seed_runs)
    os.makedirs("results", exist_ok=True)
    write_csv("results/exp_a_metrics.csv", per_seed_rows,
              field_order=["model", "seed", "precision", "recall", "f1", "fpr",
                           "auroc", "ap", "n_test", "n_test_mal"])
    write_csv("results/exp_a_summary.csv", summary_rows,
              field_order=["model",
                           "precision_mean", "precision_std",
                           "recall_mean", "recall_std",
                           "f1_mean", "f1_std",
                           "fpr_mean", "fpr_std",
                           "auroc_mean", "auroc_std",
                           "ap_mean", "ap_std",
                           "p_value_vs_BS_F1_min",
                           "p_value_vs_BS_F1_mean",
                           "F1_diff_mean_BS_minus"])
    dump_json("outputs/exp_a_metrics.json", {
        "seeds": SEEDS, "models": list(MODELS),
        "per_seed_rows": per_seed_rows,
        "summary": summary_rows,
    })

    print("\n=== Exp-A summary ===")
    h = f"{'model':<16} {'F1':>14} {'P':>14} {'R':>14} {'FPR':>14} {'AUROC':>14} {'AP':>14}  p(vs BS, F1)"
    print(h); print("-" * len(h))
    for row in summary_rows:
        p_note = (f" p>={row['p_value_vs_BS_F1_min']:.3f}"
                  if "p_value_vs_BS_F1_min" in row else "")
        print(f"{row['model']:<16} "
              f"{row['f1_mean']:>7.4f}+-{row['f1_std']:.4f} "
              f"{row['precision_mean']:>7.4f}+-{row['precision_std']:.4f} "
              f"{row['recall_mean']:>7.4f}+-{row['recall_std']:.4f} "
              f"{row['fpr_mean']:>7.4f}+-{row['fpr_std']:.4f} "
              f"{row['auroc_mean']:>7.4f}+-{row['auroc_std']:.4f} "
              f"{row['ap_mean']:>7.4f}+-{row['ap_std']:.4f}"
              f"{p_note}")
    print("\n -> results/exp_a_metrics.csv, results/exp_a_summary.csv")


if __name__ == "__main__":
    main()
