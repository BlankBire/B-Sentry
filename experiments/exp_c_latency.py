from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List

import numpy as np

from collection.decoder import load_nomad, stratified_split_70_15_15
from embedding import inference as l2i
from datalog import facts_generator, rule_engine
from risk import alert as l4a
from risk import scorer as l4s
from experiments.stats import dump_json, write_csv
from pipeline import FACTS_DIR, RISK_PARAMS


def _set_threads(n: int) -> None:
    n = max(1, int(n))
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = str(n)
    try:
        import torch
        torch.set_num_threads(n)
    except Exception:
        pass


def _one_iter(bundle, seed: int) -> Dict[str, float]:
    parts = stratified_split_70_15_15(bundle["withdrawals"], seed=seed)
    tr, te = parts["train"], parts["test"]
    peckshield = bundle["peckshield"]; deposits = bundle["deposits"]
    g2v = {"dimensions": 128, "wl_iterations": 2, "attributed": True,
           "min_count": 1, "epochs": 50, "learning_rate": 0.025,
           "workers": 2, "seed": seed}


    t0 = time.perf_counter()
    _ = [dict(r) for r in te]
    t_l1 = time.perf_counter() - t0


    t0 = time.perf_counter()
    g2v_rows = l2i.score_records_with_train(
        tr, te, peckshield=peckshield, g2v_params=g2v, max_neighbours=5,
    )
    t_l2 = time.perf_counter() - t0


    t0 = time.perf_counter()
    tokens = facts_generator.derive_token_map(tr, deposits)
    facts_generator.write_facts(
        te, tr + te, deposits, tokens, peckshield, FACTS_DIR,
    )
    l2i.write_ml_facts(g2v_rows, os.path.join(FACTS_DIR, "ml_result.facts"))
    facts = rule_engine.load_facts(FACTS_DIR)
    results = rule_engine.evaluate(facts)
    t_l3 = time.perf_counter() - t0

    viol_by_tx: Dict[str, List[str]] = {}
    for tx, kind, _v in results["logic_violation"]:
        viol_by_tx.setdefault(tx, []).append(kind)


    t0 = time.perf_counter()
    p = l4s.load_params(RISK_PARAMS)
    _ = l4a.score_and_alert(g2v_rows, viol_by_tx, p)
    t_l4 = time.perf_counter() - t0

    return {"L1": t_l1, "L2": t_l2, "L3": t_l3, "L4": t_l4,
            "total": t_l1 + t_l2 + t_l3 + t_l4,
            "n_test": len(te)}


def _measure(profile: str, threads: int, bundle, warmup: int, iters: int):
    _set_threads(threads)
    print(f"\n--- profile={profile} threads={threads} warmup={warmup} iters={iters} ---")
    for w in range(warmup):
        _one_iter(bundle, seed=42 + w)
        print(f"[warmup {w+1}/{warmup}]", flush=True)
    runs = []
    for i in range(iters):
        runs.append(_one_iter(bundle, seed=42 + i))
        print(f"[iter {i+1}/{iters}] total={runs[-1]['total']:.3f}s", flush=True)

    rows: List[Dict[str, Any]] = []
    n_test = int(runs[0]["n_test"])
    for layer in ("L1", "L2", "L3", "L4", "total"):
        vals = np.array([r[layer] for r in runs], dtype=float)
        rows.append({
            "profile": profile, "threads": threads, "layer": layer,
            "mean_ms": float(vals.mean() * 1000),
            "std_ms":  float(vals.std(ddof=0) * 1000),
            "per_record_ms_mean": float(vals.mean() * 1000 / max(n_test, 1)),
            "iters": iters, "n_test": n_test,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters",  type=int, default=3)
    ap.add_argument("--profiles", default="workstation,commodity")
    args = ap.parse_args()

    bundle = load_nomad("data")
    profile_to_threads = {
        "workstation": min(8, os.cpu_count() or 4),
        "commodity":   1,
    }

    rows: List[Dict[str, Any]] = []
    for prof in args.profiles.split(","):
        prof = prof.strip()
        if prof not in profile_to_threads:
            print(f"skipping unknown profile {prof}")
            continue
        rows.extend(_measure(prof, profile_to_threads[prof], bundle,
                             args.warmup, args.iters))

    os.makedirs("results", exist_ok=True)
    write_csv("results/exp_c_latency.csv", rows,
              field_order=["profile", "threads", "layer",
                           "mean_ms", "std_ms",
                           "per_record_ms_mean", "iters", "n_test"])
    dump_json("outputs/exp_c_latency.json", {"rows": rows})

    print("\n=== Exp-C latency summary ===")
    h = f"{'profile':<12} {'thr':>4} {'layer':<8} {'mean_ms':>14} {'ms/record':>14}"
    print(h); print("-" * len(h))
    for r in rows:
        print(f"{r['profile']:<12} {r['threads']:>4} {r['layer']:<8} "
              f"{r['mean_ms']:>7.1f}+-{r['std_ms']:>6.1f} "
              f"{r['per_record_ms_mean']:>14.3f}")
    print("\n -> results/exp_c_latency.csv")


if __name__ == "__main__":
    main()
