from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Iterable, List, Set, Tuple

from collection.decoder import (
    load_nomad, stratified_split_70_15_15,
)


def _lower(addr: Any, default: str = "0x0") -> str:
    return (str(addr) if addr else default).lower()


def derive_token_map(
    train_records: Iterable[Dict[str, Any]],
    deposits: Iterable[Dict[str, Any]] = (),
) -> Set[Tuple[int, str, int, str]]:
    out: Set[Tuple[int, str, int, str]] = set()
    for r in train_records:
        if r.get("label") == "malicious" or not r.get("origin_observed"):
            continue
        oc = int(r.get("orig_chain") or 0)
        dc = int(r.get("dst_chain") or 0)
        ot = _lower(r.get("orig_token"))
        dt = _lower(r.get("dst_token"))
        if ot != "0x0" and dt != "0x0":
            out.add((oc, ot, dc, dt))
    for d in deposits:
        oc = int(d.get("orig_chain") or 0)
        dc = int(d.get("dst_chain") or 0)
        ot = _lower(d.get("orig_token"))
        dt = _lower(d.get("dst_token"))
        if ot != "0x0" and dt != "0x0":
            out.add((oc, ot, dc, dt))
    return out


def write_facts(
    eval_records: List[Dict[str, Any]],
    orig_corpus: List[Dict[str, Any]],
    deposits: List[Dict[str, Any]],
    token_map: Set[Tuple[int, str, int, str]],
    peckshield: Iterable[str],
    out_dir: str,
) -> Dict[str, int]:
    os.makedirs(out_dir, exist_ok=True)
    counts = {"withdrawal_dst": 0, "withdrawal_orig": 0, "deposit": 0,
              "token_map": 0, "peckshield": 0}

    p_dst   = os.path.join(out_dir, "withdrawal_dst.facts")
    p_orig  = os.path.join(out_dir, "withdrawal_orig.facts")
    p_dep   = os.path.join(out_dir, "deposit.facts")
    p_tmap  = os.path.join(out_dir, "token_map.facts")
    p_peck  = os.path.join(out_dir, "peckshield.facts")

    with open(p_dst, "w", encoding="utf-8") as f:
        for r in eval_records:
            wid = str(r.get("withdrawal_id") or "")
            f.write(f"{wid}\t{_lower(r.get('tx'))}\t{int(r.get('dst_ts') or 0)}\t"
                    f"{int(r.get('dst_chain') or 0)}\t"
                    f"{_lower(r.get('dst_token'))}\t"
                    f"{_lower(r.get('beneficiary'))}\t"
                    f"{int(r.get('amount') or 0)}\t"
                    f"{float(r.get('value_usd') or 0.0):.6f}\n")
            counts["withdrawal_dst"] += 1

    seen_orig: Set[Tuple[str, str]] = set()
    with open(p_orig, "w", encoding="utf-8") as f:
        for r in orig_corpus:
            if not r.get("origin_observed"):
                continue
            wid = str(r.get("withdrawal_id") or "")
            otx = _lower(r.get("orig_tx"))
            key = (wid, otx)
            if key in seen_orig:
                continue
            seen_orig.add(key)


            f.write(f"{wid}\t{otx}\t{int(r.get('orig_ts') or 0)}\t"
                    f"{int(r.get('orig_chain') or 0)}\t"
                    f"{_lower(r.get('orig_token'))}\t"
                    f"{_lower(r.get('beneficiary'))}\t"
                    f"{int(r.get('amount') or 0)}\t"
                    f"{_lower(r.get('sender'))}\n")
            counts["withdrawal_orig"] += 1

    seen_dep: Set[str] = set()
    with open(p_dep, "w", encoding="utf-8") as f:
        for d in deposits:
            tx = _lower(d.get("tx"))
            if not tx or tx in seen_dep:
                continue
            seen_dep.add(tx)
            f.write(f"{int(d.get('orig_chain') or 0)}\t{tx}\t"
                    f"{int(d.get('orig_ts') or 0)}\t"
                    f"{_lower(d.get('orig_token'))}\t"
                    f"{_lower(d.get('sender'))}\t"
                    f"{int(d.get('dst_chain') or 0)}\t"
                    f"{_lower(d.get('dst_token'))}\t"
                    f"{_lower(d.get('beneficiary'))}\t"
                    f"{int(d.get('amount') or 0)}\n")
            counts["deposit"] += 1

    with open(p_tmap, "w", encoding="utf-8") as f:
        for oc, ot, dc, dt in sorted(token_map):
            f.write(f"{oc}\t{ot}\t{dc}\t{dt}\n")
            counts["token_map"] += 1

    with open(p_peck, "w", encoding="utf-8") as f:
        for a in sorted(set(_lower(x) for x in peckshield if x)):
            f.write(f"{a}\n")
            counts["peckshield"] += 1

    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="datalog/facts")
    args = ap.parse_args()
    b = load_nomad(args.data_dir)
    parts = stratified_split_70_15_15(b["withdrawals"], seed=args.seed)
    tmap = derive_token_map(parts["train"], b["deposits"])
    counts = write_facts(parts[args.split],
                          orig_corpus=parts["train"] + parts[args.split],
                          deposits=b["deposits"],
                          token_map=tmap, peckshield=b["peckshield"],
                          out_dir=args.out)
    print(f"[facts] split={args.split} seed={args.seed}  "
          f"registered_tokens={len(tmap)} -> {args.out}")
    for k, v in counts.items():
        print(f"  {k:18s} {v}")
