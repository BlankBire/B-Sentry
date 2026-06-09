from __future__ import annotations

import csv
import os
import random
from typing import Any, Dict, Iterable, List, Optional, Set


def _safe_int(x: Any, default: int = 0) -> int:
    if x is None or x == "":
        return default
    try:
        return int(float(x))
    except (ValueError, TypeError):
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    if x is None or x == "":
        return default
    try:
        return float(x)
    except (ValueError, TypeError):
        return default


def _lower(x: Optional[str]) -> Optional[str]:
    return x.lower() if isinstance(x, str) and x else None


def load_peckshield_addresses(path: str) -> Set[str]:
    out: Set[str] = set()
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            a = _lower((r.get("address") or "").strip())
            if a:
                out.add(a)
    return out


def _attack_records(path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tx = (row.get("dst_transaction_hash") or "").lower()
            if tx and tx not in out:
                out[tx] = row
    return out


def _matched_withdrawal(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "tx":              _lower(row.get("dst_tx_hash")),
        "block":           _safe_int(row.get("dst_timestamp")),
        "withdrawal_id":   str(row.get("withdrawal_id") or ""),
        "orig_chain":      _safe_int(row.get("orig_chain_id")),
        "orig_tx":         _lower(row.get("orig_tx_hash")),
        "orig_ts":         _safe_int(row.get("orig_timestamp")),
        "orig_token":      _lower(row.get("origin_token")),
        "sender":          _lower(row.get("sender")),
        "dst_chain":       _safe_int(row.get("dst_chain_id")),
        "dst_ts":          _safe_int(row.get("dst_timestamp")),
        "dst_token":       _lower(row.get("dst_token")),
        "beneficiary":     _lower(row.get("beneficiary")),
        "amount":          _safe_int(row.get("amount")),
        "value_usd":       _safe_float(row.get("value_usd")),
        "time_difference": _safe_int(row.get("time_difference")),
        "origin_observed": True,
        "label":           "normal",
    }


def _attack_withdrawal(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "tx":              _lower(row.get("dst_transaction_hash")),
        "block":           _safe_int(row.get("dst_timestamp")),
        "withdrawal_id":   str(row.get("withdrawal_id") or ""),
        "orig_chain":      None,
        "orig_tx":         None,
        "orig_ts":         None,
        "orig_token":      None,
        "sender":          None,
        "dst_chain":       _safe_int(row.get("dst_chain_id")),
        "dst_ts":          _safe_int(row.get("dst_timestamp")),
        "dst_token":       _lower(row.get("dst_token")),
        "beneficiary":     _lower(row.get("beneficiary")),
        "amount":          _safe_int(row.get("amount")),
        "value_usd":       _safe_float(row.get("value_usd")),
        "time_difference": 0,
        "origin_observed": False,
        "label":           "malicious",
    }


def _deposit_record(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "tx":           _lower(row.get("orig_tx_hash")),
        "orig_chain":   _safe_int(row.get("orig_chain_id")),
        "orig_ts":      _safe_int(row.get("orig_timestamp")),
        "orig_token":   _lower(row.get("origin_token")),
        "sender":       _lower(row.get("sender")),
        "dst_chain":    _safe_int(row.get("dst_chain_id")),
        "dst_ts":       _safe_int(row.get("dst_timestamp")),
        "dst_token":    _lower(row.get("dst_token")),
        "beneficiary":  _lower(row.get("beneficiary")),
        "amount":       _safe_int(row.get("amount")),
    }


def load_nomad(data_dir: str = "data") -> Dict[str, Any]:
    attacks = _attack_records(os.path.join(data_dir, "attacks.csv"))
    matched_path = os.path.join(data_dir, "cctxs_withdrawals.csv")

    withdrawals: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    if os.path.exists(matched_path):
        with open(matched_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rec = _matched_withdrawal(row)
                tx = rec["tx"]
                if not tx or tx in seen:
                    continue
                if tx in attacks:


                    rec["label"] = "malicious"
                seen.add(tx); withdrawals.append(rec)
    for tx, row in attacks.items():
        if tx in seen:
            continue
        seen.add(tx); withdrawals.append(_attack_withdrawal(row))

    deposits: List[Dict[str, Any]] = []
    dpath = os.path.join(data_dir, "cctxs_deposits.csv")
    if os.path.exists(dpath):
        with open(dpath, "r", encoding="utf-8") as f:
            seen_d: Set[str] = set()
            for row in csv.DictReader(f):
                rec = _deposit_record(row)
                key = rec["tx"]
                if not key or key in seen_d:
                    continue
                seen_d.add(key); deposits.append(rec)

    return {
        "withdrawals": withdrawals,
        "deposits":    deposits,
        "peckshield":  load_peckshield_addresses(
            os.path.join(data_dir, "all_addresses_peckshield.csv")),
    }


def stratified_split_70_15_15(
    records: List[Dict[str, Any]],
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    rng = random.Random(seed)
    by_lab: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        by_lab.setdefault(r["label"], []).append(r)
    train, val, test = [], [], []
    for lab, rs in by_lab.items():
        rs2 = list(rs); rng.shuffle(rs2)
        n = len(rs2)
        n_test = max(1, int(round(n * 0.15)))
        n_val = max(1, int(round(n * 0.15)))
        n_train = n - n_val - n_test
        train.extend(rs2[:n_train])
        val.extend(rs2[n_train:n_train + n_val])
        test.extend(rs2[n_train + n_val:])
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return {"train": train, "val": val, "test": test}


if __name__ == "__main__":
    b = load_nomad("data")
    ws = b["withdrawals"]
    mal = sum(1 for r in ws if r["label"] == "malicious")
    norm = sum(1 for r in ws if r["label"] == "normal")
    obs = sum(1 for r in ws if r["origin_observed"])
    print(f"Nomad CCTX: withdrawals={len(ws)}  malicious={mal}  normal={norm}  "
          f"origin_observed={obs}")
    print(f"            deposits={len(b['deposits'])}  peckshield={len(b['peckshield'])}")
    parts = stratified_split_70_15_15(ws, seed=42)
    for k, v in parts.items():
        m = sum(1 for r in v if r["label"] == "malicious")
        print(f"  {k:6s} n={len(v):4d}  malicious={m:3d}  normal={len(v)-m}")
