from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from collection.decoder import load_nomad, stratified_split_70_15_15
from embedding import inference as l2i
from embedding.gnn_baselines import score_with_gcn, score_with_sage
from datalog import facts_generator, rule_engine
from risk import alert as l4a
from risk import scorer as l4s
from pipeline import FACTS_DIR


SEED = 1


def _datalog_full(eval_records, train_records, deposits, peckshield):
    tokens = facts_generator.derive_token_map(train_records, deposits)
    facts_generator.write_facts(
        eval_records, train_records + eval_records, deposits,
        tokens, peckshield, FACTS_DIR,
    )
    open(os.path.join(FACTS_DIR, "ml_result.facts"), "w").close()
    facts = rule_engine.load_facts(FACTS_DIR)
    res = rule_engine.evaluate(facts)
    valid = {dtx for _wid, dtx in res["cctx_valid_withdrawal"]}
    by_tx: Dict[str, List[str]] = defaultdict(list)
    sigma: Dict[str, int] = defaultdict(int)
    for tx, k, v in res["logic_violation"]:
        by_tx[tx].append(k)
        if v > sigma[tx]:
            sigma[tx] = v
    return by_tx, sigma, valid, tokens


def _pick_cases(alerts, records, viol_by_tx, sigma_by_tx):
    a_by_tx = {a["tx"]: a for a in alerts}
    r_by_tx = {r["tx"]: r for r in records}

    def _try(predicate):
        for tx, a in a_by_tx.items():
            if tx not in r_by_tx:
                continue
            sig = sigma_by_tx.get(tx, 0)
            ml = a["ml_decision"]
            if predicate(ml, sig, r_by_tx[tx]["label"]):
                return tx
        return None

    cases = {
        "A_critical_attack":
            _try(lambda ml, s, lab:
                 ml == "high_risk" and s >= 2 and lab == "malicious"),
        "B_logic_only_attack":
            _try(lambda ml, s, lab:
                 ml != "high_risk" and s >= 2 and lab == "malicious"),
        "C_ml_only_false_positive":
            _try(lambda ml, s, lab:
                 ml == "high_risk" and s < 2 and lab == "normal"),
        "D_clean_normal":
            _try(lambda ml, s, lab:
                 ml != "high_risk" and s == 0 and lab == "normal"),
    }
    return cases


def main():
    bundle = load_nomad("data")
    parts = stratified_split_70_15_15(bundle["withdrawals"], seed=SEED)
    tr, vl, te = parts["train"], parts["val"], parts["test"]
    peckshield = bundle["peckshield"]
    deposits = bundle["deposits"]

    g2v = {"dimensions": 128, "wl_iterations": 2, "attributed": True,
           "min_count": 1, "epochs": 50, "learning_rate": 0.025,
           "workers": 2, "seed": SEED}

    print(f"[exp-e] seed={SEED}  scoring test...")
    ml_rows = l2i.score_records_with_train(
        tr, te, peckshield=peckshield, g2v_params=g2v, max_neighbours=5,
    )
    viol_by_tx, sigma_by_tx, valid_set, tmap = _datalog_full(
        te, tr, deposits, peckshield,
    )

    val_rows = l2i.score_records_with_train(
        tr, vl, peckshield=peckshield, g2v_params=g2v, max_neighbours=5,
    )
    viol_val, _sig_val, _valid_val, _ = _datalog_full(
        vl, tr, deposits, peckshield,
    )
    ms_val = np.array([r["anomaly_score"] for r in val_rows])
    lam_val = [r["decision"] for r in val_rows]
    kinds_val = [viol_val.get(r["tx"], []) for r in val_rows]
    y_val = np.array([1 if r["label"] == "malicious" else 0 for r in val_rows])
    best = l4s.grid_search_alpha_beta(ms_val, kinds_val, lam_val, y_val)["best"]
    params = {"alpha": best["alpha"], "beta": best["beta"]}
    alerts = l4a.score_and_alert(ml_rows, viol_by_tx, params)


    gcn_rows = score_with_gcn(tr, te, peckshield=peckshield,
                               max_neighbours=5, seed=SEED, epochs=30)
    sage_rows = score_with_sage(tr, te, peckshield=peckshield,
                                 max_neighbours=5, seed=SEED, epochs=30)
    gcn_by_tx = {r["tx"]: r for r in gcn_rows}
    sage_by_tx = {r["tx"]: r for r in sage_rows}

    cases = _pick_cases(alerts, te, viol_by_tx, sigma_by_tx)
    print("\n[exp-e] selected cases:")
    for k, tx in cases.items():
        print(f"  {k:<35s} {tx}")

    a_by_tx = {a["tx"]: a for a in alerts}
    r_by_tx = {r["tx"]: r for r in te}

    case_records = []
    for case_name, tx in cases.items():
        if tx is None:
            continue
        rec = r_by_tx[tx]
        a   = a_by_tx[tx]
        gcn = gcn_by_tx.get(tx)
        sage = sage_by_tx.get(tx)

        explanation: List[str] = []
        for k in viol_by_tx.get(tx, []):
            if k == "forged_withdrawal":
                explanation.append(
                    f"forged_withdrawal (σ=3): no origin-side burn observation "
                    f"matching (withdrawal_id={rec['withdrawal_id']}, "
                    f"beneficiary={rec['beneficiary']}, amount={rec['amount']}) "
                    f"was found before dst_ts={rec['dst_ts']}."
                )
            elif k == "invalid_token_mapping":
                explanation.append(
                    f"invalid_token_mapping (σ=1): origin observation exists "
                    f"but (orig_token, dst_token) = ({rec.get('orig_token')}, "
                    f"{rec['dst_token']}) is not in the training-derived "
                    f"token registry."
                )
            elif k == "peckshield_listed":
                explanation.append(
                    f"peckshield_listed (σ=2): beneficiary {rec['beneficiary']} "
                    f"is on the PeckShield attacker EOA list."
                )
            elif k == "finality_breach":
                explanation.append(
                    f"finality_breach (σ=2): a re-org / processing failure was "
                    f"observed for this transaction."
                )
        if not explanation:
            explanation.append("(no Layer 3 rule fired — Datalog signal silent.)")

        case_records.append({
            "case": case_name,
            "tx": tx,
            "label": rec["label"],
            "dst_chain": rec["dst_chain"],
            "dst_token": rec["dst_token"],
            "beneficiary": rec["beneficiary"],
            "amount": rec["amount"],
            "value_usd": rec["value_usd"],
            "origin_observed": rec["origin_observed"],
            "withdrawal_id": rec["withdrawal_id"],
            "in_peckshield": (rec["beneficiary"] or "").lower() in peckshield,
            "ml": {
                "anomaly_score": a["ml_score"],
                "lambda_tau":    a["ml_decision"],
            },
            "datalog": {
                "violations": viol_by_tx.get(tx, []),
                "sigma_tau":  sigma_by_tx.get(tx, 0),
                "in_cctx_valid_withdrawal": tx in valid_set,
                "explanation": explanation,
            },
            "layer4": {
                "alpha": params["alpha"], "beta": params["beta"],
                "risk_score": a["risk_score"],
                "alert_tier": a["alert"],
            },
            "blackbox_reference": {
                "GCN_score":       (gcn["anomaly_score"] if gcn else None),
                "GCN_decision":    (gcn["decision"] if gcn else None),
                "GraphSAGE_score": (sage["anomaly_score"] if sage else None),
                "GraphSAGE_decision": (sage["decision"] if sage else None),
                "note": ("Black-box GNNs emit a numerical anomaly score with no "
                          "per-alert rule explanation; B-Sentry's Layer 3 "
                          "rule labels (above) are the structural answer to "
                          "why a tier was assigned."),
            },
        })

    os.makedirs("results", exist_ok=True)
    with open("results/exp_e_case_study.json", "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "alpha": params["alpha"],
                    "beta": params["beta"], "cases": case_records},
                  f, indent=2)


    md = ["# Exp-E — Interpretability case study", "",
          f"Seed = {SEED}. Layer 4 α = {params['alpha']:.2f}, "
          f"β = {params['beta']:.2f}. Four cases, one per Lemma-1 tier "
          f"quadrant.", ""]
    for cr in case_records:
        md.append(f"## Case {cr['case']}")
        md.append("")
        md.append(f"- **tx**: `{cr['tx']}` — ground truth = **{cr['label']}**.")
        md.append(f"- **dst-side observation**: chain={cr['dst_chain']}, "
                   f"token=`{cr['dst_token']}`, "
                   f"beneficiary=`{cr['beneficiary']}` "
                   f"(in PeckShield: **{cr['in_peckshield']}**), "
                   f"amount={cr['amount']}, value≈${cr['value_usd']:.0f}, "
                   f"origin_observed={cr['origin_observed']}.")
        md.append(f"- **Layer 2 (Graph2Vec)**: anomaly_score = "
                   f"{cr['ml']['anomaly_score']:.4f} → "
                   f"λ_τ = `{cr['ml']['lambda_tau']}`.")
        md.append(f"- **Layer 3 (Datalog)**: σ_τ = {cr['datalog']['sigma_tau']}; "
                   f"matched origin? **{cr['datalog']['in_cctx_valid_withdrawal']}**.")
        md.append("    - Rule trace:")
        for e in cr['datalog']['explanation']:
            md.append(f"        - {e}")
        md.append(f"- **Layer 4**: R_τ = α·s + β·σ/3 = "
                   f"{cr['layer4']['alpha']:.2f}·{cr['ml']['anomaly_score']:.3f} "
                   f"+ {cr['layer4']['beta']:.2f}·{cr['datalog']['sigma_tau']}/3 "
                   f"= {cr['layer4']['risk_score']:.4f} → "
                   f"tier = **{cr['layer4']['alert_tier']}**.")
        ref = cr['blackbox_reference']
        if ref['GCN_score'] is not None:
            md.append(f"- **Black-box reference**: GCN score = "
                       f"{ref['GCN_score']:.4f} (`{ref['GCN_decision']}`), "
                       f"GraphSAGE score = {ref['GraphSAGE_score']:.4f} "
                       f"(`{ref['GraphSAGE_decision']}`). These outputs lack "
                       f"any per-alert structural explanation analogous to "
                       f"the Layer 3 rule trace above.")
        md.append("")
    with open("results/exp_e_case_study.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print("\n -> results/exp_e_case_study.json")
    print(" -> results/exp_e_case_study.md")


if __name__ == "__main__":
    main()
