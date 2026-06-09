from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx


_STABLE_TOKENS: Set[str] = {
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
    "0xdac17f958d2ee523a2206206994597c13d831ec7",
    "0x6b175474e89094c44da98b954eedeac495271d0f",
    "0x853d955acef822db058eb8505911ed77f175b99e",
}
_ETH_LIKE: Set[str] = {
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599",
}


def _token_class(addr: Optional[str]) -> str:
    if not addr:
        return "TKN_NONE"
    a = addr.lower()
    if a in _STABLE_TOKENS:
        return "TKN_USD"
    if a in _ETH_LIKE:
        return "TKN_ETH"
    return "TKN_OTH"


def _amount_bucket(amount: int) -> str:
    if amount <= 0:
        return "AMT_0"
    bl = max(0, amount.bit_length() - 1)
    if bl < 20:  return "AMT_S"
    if bl < 40:  return "AMT_M"
    if bl < 60:  return "AMT_L"
    if bl < 80:  return "AMT_XL"
    return "AMT_XXL"


def _value_usd_bucket(v: float) -> str:
    if v <= 0:        return "USD_0"
    if v < 100:       return "USD_S"
    if v < 10_000:    return "USD_M"
    if v < 1_000_000: return "USD_L"
    return "USD_XL"


def _hour_bucket(ts: int) -> str:
    if ts <= 0:
        return "TIME_0"
    return f"TIME_{(ts // 3600) % 24:02d}H"


def _set_feat(g: nx.DiGraph, nid: int, *parts: str) -> None:
    cur = g.nodes[nid].get("feature", "")
    existing = set(cur.split("|")) if cur else set()
    existing.update(p for p in parts if p)
    g.nodes[nid]["feature"] = "|".join(sorted(existing))


def _add_node(g: nx.DiGraph, idx_map: Dict[str, int], key: str,
              *features: str) -> int:
    if key in idx_map:
        nid = idx_map[key]
        _set_feat(g, nid, *features)
        return nid
    nid = len(idx_map); idx_map[key] = nid
    g.add_node(nid, feature="|".join(sorted(set(features))), key=key)
    return nid


def _log_weight(amount: int) -> float:
    if amount is None or amount <= 0:
        return 1.0
    try:
        return 1.0 + float(amount.bit_length())
    except AttributeError:
        return 1.0


def build_xteg(
    record: Dict[str, Any],
    *,
    neighbour_records: Iterable[Dict[str, Any]] = (),
    peckshield: Optional[Set[str]] = None,
    max_neighbours: int = 5,
) -> Tuple[nx.DiGraph, Dict[str, int]]:
    g = nx.DiGraph()
    idx_map: Dict[str, int] = {}
    peckshield = peckshield or set()


    w_key = f"W:{record.get('tx', '?')}"
    w_feats = [
        "ACT_W",
        _amount_bucket(int(record.get("amount") or 0)),
        _value_usd_bucket(float(record.get("value_usd") or 0.0)),
        _hour_bucket(int(record.get("dst_ts") or 0)),
        "OBS_Y" if record.get("origin_observed") else "OBS_N",
    ]
    w_id = _add_node(g, idx_map, w_key, *w_feats)

    benef = record.get("beneficiary") or "0x0"
    benef_feats = ["ROLE_B"]
    if benef.lower() in peckshield:
        benef_feats.append("PECK")
    benef_id = _add_node(g, idx_map, f"ADDR:{benef}", *benef_feats)

    dst_tok = record.get("dst_token") or "0x0"
    dst_tok_id = _add_node(g, idx_map, f"TOK:{dst_tok}",
                            "ROLE_DT", _token_class(dst_tok))

    dst_chain = record.get("dst_chain") or 0
    dst_chain_id = _add_node(g, idx_map, f"CHAIN:{dst_chain}",
                              "ROLE_DC", f"CID_{dst_chain}")

    g.add_edge(w_id, benef_id, kind="W_TO_B",
               weight=_log_weight(record.get("amount") or 0))
    g.add_edge(w_id, dst_tok_id, kind="W_OF_TOK",
               weight=_log_weight(record.get("amount") or 0))
    g.add_edge(dst_tok_id, benef_id, kind="TOK_TO_B")
    g.add_edge(w_id, dst_chain_id, kind="W_ON_CHAIN")

    if record.get("origin_observed"):
        sender = record.get("sender") or "0x0"
        sender_feats = ["ROLE_S"]
        if sender and sender.lower() in peckshield:
            sender_feats.append("PECK")
        sender_id = _add_node(g, idx_map, f"ADDR:{sender}", *sender_feats)

        orig_tok = record.get("orig_token") or "0x0"
        orig_tok_id = _add_node(g, idx_map, f"TOK:{orig_tok}",
                                 "ROLE_OT", _token_class(orig_tok))
        orig_chain = record.get("orig_chain") or 0
        orig_chain_id = _add_node(g, idx_map, f"CHAIN:{orig_chain}",
                                    "ROLE_OC", f"CID_{orig_chain}")

        g.add_edge(sender_id, w_id, kind="S_INIT_W")
        g.add_edge(orig_tok_id, dst_tok_id, kind="TOK_MAP")
        g.add_edge(orig_chain_id, dst_chain_id, kind="CHAIN_HOP")
        g.add_edge(sender_id, orig_tok_id, kind="S_BURNS")


    n_added = 0
    for nrec in neighbour_records:
        if n_added >= max_neighbours:
            break
        if nrec.get("tx") == record.get("tx"):
            continue
        share_b = nrec.get("beneficiary") == record.get("beneficiary")
        share_t = nrec.get("dst_token") == record.get("dst_token")
        if not (share_b or share_t):
            continue
        nkey = f"NW:{(nrec.get('tx') or '?')[:10]}"
        feats = ["ACT_NW",
                 _amount_bucket(int(nrec.get("amount") or 0)),
                 _value_usd_bucket(float(nrec.get("value_usd") or 0.0))]
        nid = _add_node(g, idx_map, nkey, *feats)
        if share_b:
            g.add_edge(nid, benef_id, kind="NW_SHARE_B")
        if share_t:
            g.add_edge(nid, dst_tok_id, kind="NW_SHARE_T")
        n_added += 1

    return g, idx_map


def _neighbour_index(records: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    idx: Dict[str, List[int]] = {}
    for i, r in enumerate(records):
        b = r.get("beneficiary") or "?"
        t = r.get("dst_token") or "?"
        idx.setdefault(f"B:{b}", []).append(i)
        idx.setdefault(f"T:{t}", []).append(i)
    return idx


def build_graphs(
    records: List[Dict[str, Any]],
    *,
    peckshield: Optional[Set[str]] = None,
    max_neighbours: int = 5,
) -> Tuple[List[nx.DiGraph], List[int]]:
    idx = _neighbour_index(records)
    graphs: List[nx.DiGraph] = []
    kept: List[int] = []
    for i, r in enumerate(records):
        b = r.get("beneficiary") or "?"
        t = r.get("dst_token") or "?"
        cand_ids: List[int] = []
        seen: Set[int] = {i}
        for k in (f"B:{b}", f"T:{t}"):
            for j in idx.get(k, ()):
                if j in seen:
                    continue
                cand_ids.append(j); seen.add(j)
                if len(cand_ids) >= max_neighbours * 2:
                    break
            if len(cand_ids) >= max_neighbours * 2:
                break
        neighbours = [records[j] for j in cand_ids[:max_neighbours]]
        g, _ = build_xteg(r, neighbour_records=neighbours,
                          peckshield=peckshield,
                          max_neighbours=max_neighbours)
        if g.number_of_nodes() < 2:
            continue
        graphs.append(g); kept.append(i)
    return graphs, kept


if __name__ == "__main__":
    from collection.decoder import load_nomad
    b = load_nomad("data")
    gs, kept = build_graphs(b["withdrawals"][:20], peckshield=b["peckshield"])
    for i, g in zip(kept[:6], gs[:6]):
        r = b["withdrawals"][i]
        print(r["label"], "obs=", r["origin_observed"],
              "nodes=", g.number_of_nodes(), "edges=", g.number_of_edges())
