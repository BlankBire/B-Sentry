from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple


def _read_facts(path: str) -> List[List[str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n").split("\t") for line in f if line.strip()]


def load_facts(facts_dir: str) -> Dict[str, Any]:
    def _i(x: str) -> int:
        try: return int(x)
        except ValueError: return 0
    def _f(x: str) -> float:
        try: return float(x)
        except ValueError: return 0.0

    D = _read_facts(os.path.join(facts_dir, "withdrawal_dst.facts"))
    O = _read_facts(os.path.join(facts_dir, "withdrawal_orig.facts"))
    P = _read_facts(os.path.join(facts_dir, "deposit.facts"))
    K = _read_facts(os.path.join(facts_dir, "token_map.facts"))
    H = _read_facts(os.path.join(facts_dir, "peckshield.facts"))
    L = _read_facts(os.path.join(facts_dir, "ml_result.facts"))


    return {
        "wd_dst":  [(w, t, _i(ts), _i(c), tok, b, _i(a), _f(v))
                    for w, t, ts, c, tok, b, a, v in D],
        "wd_orig": [(w, t, _i(ts), _i(c), tok, b, _i(a),
                     (row[7] if len(row) > 7 else ""))
                    for row in O
                    for (w, t, ts, c, tok, b, a) in [row[:7]]],
        "deposits":[(_i(oc), tx, _i(ots), otok, snd, _i(dc), dtok, ben, _i(amt))
                    for oc, tx, ots, otok, snd, dc, dtok, ben, amt in P],
        "tmap":    {(int(r[0]), r[1], int(r[2]), r[3]) for r in K},
        "peck":    {r[0] for r in H},
        "ml":      [(t, _f(s), d) for t, s, d in L],
    }


WINDOW_SEC = 7 * 24 * 3600


def evaluate(facts: Dict[str, Any]) -> Dict[str, List[Tuple[Any, ...]]]:
    dst = facts["wd_dst"]
    orig = facts["wd_orig"]
    tmap: Set[Tuple[int, str, int, str]] = facts["tmap"]
    peck: Set[str] = facts["peck"]
    ml = facts["ml"]


    orig_by_keys: Dict[Tuple[str, str, int], List[Tuple]] = defaultdict(list)
    for o in orig:
        wid, _otx, _ots, _oc, _otok, beneficiary, amt = o[:7]
        orig_by_keys[(wid, beneficiary, amt)].append(o)

    valid: Set[Tuple[str, str]] = set()


    for d in dst:
        wid, dtx, dts, _dc, _dtok, ben, amt, _v = d
        for o in orig_by_keys.get((wid, ben, amt), ()):
            if o[2] < dts:
                valid.add((wid, dtx)); break


    violations: List[Tuple[str, str, int]] = []
    dst_by_tx: Dict[str, Tuple[str, int, str, str]] = {}
    for d in dst:
        wid, dtx, _dts, dc, dtok, ben, _amt, _v = d
        dst_by_tx[dtx] = (wid, dc, dtok, ben)

        if (wid, dtx) not in valid:
            violations.append((dtx, "forged_withdrawal", 3))

        if ben in peck:
            violations.append((dtx, "peckshield_listed", 2))


    orig_by_wid: Dict[str, List[Tuple]] = defaultdict(list)
    for o in orig:
        orig_by_wid[o[0]].append(o)
    for d in dst:
        wid, dtx, _dts, dc, dtok, _ben, _amt, _v = d
        os_ = orig_by_wid.get(wid, [])
        if not os_:
            continue


        if any((o[3], o[4], dc, dtok) in tmap for o in os_):
            continue
        violations.append((dtx, "invalid_token_mapping", 1))

    viol_set = set(violations)
    by_tx_kinds: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    sigma_by_tx: Dict[str, int] = defaultdict(int)
    for tx, k, v in viol_set:
        by_tx_kinds[tx].append((k, v))
        if v > sigma_by_tx[tx]:
            sigma_by_tx[tx] = v
    max_severity_rows = [(tx, sigma_by_tx[tx]) for tx in by_tx_kinds]


    ml_by_tx: Dict[str, Tuple[float, str]] = {t: (s, d) for t, s, d in ml}
    all_txs = set(dst_by_tx) | set(ml_by_tx) | set(by_tx_kinds)

    alert_critical: List[Tuple[str, str]] = []
    alert_suspicious: List[Tuple[str, str]] = []
    alert_normal: List[Tuple[str]] = []
    for tx in all_txs:
        ml_dec = ml_by_tx.get(tx, (0.0, "normal"))[1]
        sigma = sigma_by_tx.get(tx, 0)
        ml_strong = (ml_dec == "high_risk")
        dl_strong = (sigma >= 2)
        if ml_strong and dl_strong:
            for k, v in by_tx_kinds.get(tx, []):
                if v >= 2:
                    alert_critical.append((tx, k))
        elif ml_strong:
            alert_suspicious.append((tx, "ml_only"))
        elif dl_strong:
            alert_suspicious.append((tx, "logic_only"))
        else:
            alert_normal.append((tx,))

    return {
        "cctx_valid_withdrawal": sorted(valid),
        "logic_violation": sorted(viol_set),
        "max_severity":    sorted(max_severity_rows),
        "alert_critical":  sorted(set(alert_critical)),
        "alert_suspicious": sorted(set(alert_suspicious)),
        "alert_normal":    sorted(set(alert_normal)),
    }


def write_outputs(out_dir: str,
                  results: Dict[str, List[Tuple[Any, ...]]]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for name, rows in results.items():
        path = os.path.join(out_dir, f"{name}.csv")
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            for row in rows:
                w.writerow(row)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default="datalog/facts")
    ap.add_argument("--out", default="datalog/out")
    args = ap.parse_args()
    facts = load_facts(args.facts)
    results = evaluate(facts)
    write_outputs(args.out, results)
    for k, v in results.items():
        print(f"  {k:18s} {len(v)}")
    print(f"[rule_engine] -> {args.out}")
