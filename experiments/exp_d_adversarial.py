from __future__ import annotations

import copy
import os
import random
import time
from typing import Any, Dict, List, Set

import numpy as np

from collection.decoder import load_nomad, stratified_split_70_15_15
from embedding import inference as l2i
from embedding.gnn_baselines import score_with_gcn, score_with_sage
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
DROP_PS = [0.00, 0.10, 0.25, 0.50, 0.75, 1.00]


_DEFAULT_ORIG_CHAIN = 1650811245


def _synthesize_orig_observation(attack_rec: Dict[str, Any],
                                  origin_token_pool: List[str],
                                  rng: random.Random) -> Dict[str, Any]:
    fake = copy.deepcopy(attack_rec)
    fake["origin_observed"] = True
    fake["orig_chain"] = _DEFAULT_ORIG_CHAIN

    fake["orig_tx"] = "0xfake" + format(rng.getrandbits(208), "052x")

    fake["orig_ts"] = max(1, int(attack_rec.get("dst_ts") or 1) - 1000)
    fake["orig_token"] = (rng.choice(origin_token_pool)
                          if origin_token_pool else "0x0")

    fake["sender"] = attack_rec.get("beneficiary")
    return fake


def _perturb_test(test_records: List[Dict[str, Any]],
                  drop_p: float,
                  origin_token_pool: List[str],
                  seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed * 9973 + int(drop_p * 1000))
    attacks_idx = [i for i, r in enumerate(test_records)
                   if r["label"] == "malicious"]
    rng.shuffle(attacks_idx)
    n_perturb = int(round(len(attacks_idx) * drop_p))
    to_perturb = set(attacks_idx[:n_perturb])

    out: List[Dict[str, Any]] = []
    for i, r in enumerate(test_records):
        if i in to_perturb:
            out.append(_synthesize_orig_observation(r, origin_token_pool, rng))
        else:
            out.append(r)
    return out


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


def _one_seed_one_p(bundle, seed: int, drop_p: float) -> List[Dict[str, Any]]:
    parts = stratified_split_70_15_15(bundle["withdrawals"], seed=seed)
    tr, vl, te = parts["train"], parts["val"], parts["test"]
    peckshield = bundle["peckshield"]
    deposits = bundle["deposits"]


    tmap = facts_generator.derive_token_map(tr, deposits)
    origin_tokens = sorted({ot for (_oc, ot, _dc, _dt) in tmap})


    te_pert = _perturb_test(te, drop_p, origin_tokens, seed)
    y_true = np.array([1 if r["label"] == "malicious" else 0 for r in te_pert])

    g2v_params = {"dimensions": 128, "wl_iterations": 2, "attributed": True,
                  "min_count": 1, "epochs": 50, "learning_rate": 0.025,
                  "workers": 2, "seed": seed}


    viol_by_tx = _datalog_violations(te_pert, tr, deposits, peckshield)
    dl_pred = np.array([1 if r["tx"] in viol_by_tx else 0 for r in te_pert])


    g2v_rows = l2i.score_records_with_train(
        tr, te_pert, peckshield=peckshield, g2v_params=g2v_params,
        max_neighbours=5,
    )
    g2v_scores = np.array([r["anomaly_score"] for r in g2v_rows])
    g2v_pred = np.array([1 if r["decision"] == "high_risk" else 0
                         for r in g2v_rows])


    gcn_rows = score_with_gcn(tr, te_pert, peckshield=peckshield,
                               max_neighbours=5, seed=seed, epochs=30)
    gcn_scores = np.array([r["anomaly_score"] for r in gcn_rows])
    gcn_pred = np.array([1 if r["decision"] == "high_risk" else 0
                         for r in gcn_rows])


    sage_rows = score_with_sage(tr, te_pert, peckshield=peckshield,
                                 max_neighbours=5, seed=seed, epochs=30)
    sage_scores = np.array([r["anomaly_score"] for r in sage_rows])
    sage_pred = np.array([1 if r["decision"] == "high_risk" else 0
                          for r in sage_rows])


    val_rows = l2i.score_records_with_train(
        tr, vl, peckshield=peckshield, g2v_params=g2v_params,
        max_neighbours=5,
    )
    viol_val = _datalog_violations(vl, tr, deposits, peckshield)
    ms_val = np.array([r["anomaly_score"] for r in val_rows])
    lam_val = [r["decision"] for r in val_rows]
    kinds_val = [viol_val.get(r["tx"], []) for r in val_rows]
    y_val = np.array([1 if r["label"] == "malicious" else 0 for r in val_rows])
    best = l4s.grid_search_alpha_beta(ms_val, kinds_val, lam_val, y_val)["best"]
    params = {"alpha": best["alpha"], "beta": best["beta"]}
    alerts = l4a.score_and_alert(g2v_rows, viol_by_tx, params)
    cs_crit_pred = np.array([1 if a["alert"] == "critical" else 0
                              for a in alerts])
    cs_tier_pred = np.array([1 if a["alert"] in ("critical", "suspicious") else 0
                              for a in alerts])
    cs_risk = np.array([a["risk_score"] for a in alerts])

    rows = []
    for model, p, sc in [
        ("XChainWatcher",      dl_pred,      dl_pred.astype(float)),
        ("Graph2Vec",          g2v_pred,     g2v_scores),
        ("GCN",                gcn_pred,     gcn_scores),
        ("GraphSAGE",          sage_pred,    sage_scores),
        ("B-Sentry-critical",  cs_crit_pred, cs_risk),
        ("B-Sentry-tiered",    cs_tier_pred, cs_risk),
    ]:
        rows.append({
            "drop_p": drop_p, "seed": seed, "model": model,
            "precision": precision(y_true, p),
            "recall":    recall(y_true, p),
            "f1":        f1(y_true, p),
            "fpr":       fpr(y_true, p),
            "auroc":     auroc(y_true, sc),
            "ap":        average_precision(y_true, sc),
            "n_test":    int(len(te_pert)),
            "n_test_mal":int(y_true.sum()),
            "n_perturbed": int(round(
                sum(1 for r in te if r["label"] == "malicious") * drop_p)),
        })
    return rows


def main():
    bundle = load_nomad("data")
    all_rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for p in DROP_PS:
        for s in SEEDS:
            print(f"--- drop_p={p}  seed={s} ---", flush=True)
            all_rows.extend(_one_seed_one_p(bundle, seed=s, drop_p=p))
    elapsed = time.time() - t0
    print(f"[exp-d] total elapsed {elapsed:.1f}s")


    summary: List[Dict[str, Any]] = []
    for p in DROP_PS:
        for m in ("XChainWatcher", "Graph2Vec", "GCN", "GraphSAGE",
                  "B-Sentry-critical", "B-Sentry-tiered"):
            sel = [r for r in all_rows
                   if abs(r["drop_p"] - p) < 1e-9 and r["model"] == m]
            row = {"drop_p": p, "model": m}
            for k in ("precision", "recall", "f1", "fpr", "auroc", "ap"):
                vals = np.array([s[k] for s in sel], dtype=float)
                vals = vals[~np.isnan(vals)]
                row[f"{k}_mean"] = float(vals.mean()) if vals.size else float("nan")
                row[f"{k}_std"]  = float(vals.std(ddof=0)) if vals.size else float("nan")
            summary.append(row)

    os.makedirs("results", exist_ok=True)
    write_csv("results/exp_d_adversarial.csv", all_rows,
              field_order=["drop_p", "model", "seed", "precision", "recall",
                           "f1", "fpr", "auroc", "ap", "n_test", "n_test_mal",
                           "n_perturbed"])
    write_csv("results/exp_d_adversarial_summary.csv", summary,
              field_order=["drop_p", "model",
                           "precision_mean", "precision_std",
                           "recall_mean", "recall_std",
                           "f1_mean", "f1_std",
                           "fpr_mean", "fpr_std",
                           "auroc_mean", "auroc_std",
                           "ap_mean", "ap_std"])
    dump_json("outputs/exp_d_adversarial.json", {
        "drop_ps": DROP_PS, "seeds": SEEDS,
        "per_seed": all_rows, "summary": summary,
    })


    print("\n=== Exp-D adversarial robustness — F1 (mean over seeds) ===")
    models = ("XChainWatcher", "Graph2Vec", "GCN", "GraphSAGE",
              "B-Sentry-critical", "B-Sentry-tiered")
    head = "drop_p " + "  ".join(f"{m:>20}" for m in models)
    print(head); print("-" * len(head))
    for p in DROP_PS:
        cells = []
        for m in models:
            row = next(s for s in summary
                        if abs(s["drop_p"]-p) < 1e-9 and s["model"] == m)
            cells.append(f"{row['f1_mean']:>20.4f}")
        print(f"{p:>6.2f} " + "  ".join(cells))


if __name__ == "__main__":
    main()
