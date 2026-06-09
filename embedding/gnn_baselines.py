from __future__ import annotations

import os
import random
import time
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool

from collection.decoder import load_nomad, stratified_split_70_15_15
from embedding.build_graph import build_graphs
from embedding.discretize import discretize, load_thresholds


_DEVICE = torch.device("cuda" if (os.environ.get("BSENTRY_GPU") == "1"
                                   and torch.cuda.is_available()) else "cpu")


def build_feature_vocab(graphs) -> Dict[str, int]:
    vocab: Dict[str, int] = {}
    for g in graphs:
        for _, attr in g.nodes(data=True):
            for tok in (attr.get("feature") or "").split("|"):
                if tok and tok not in vocab:
                    vocab[tok] = len(vocab)
    return vocab


def nx_to_pyg(g, vocab: Dict[str, int]) -> Data:
    nodes = list(g.nodes())
    nid = {n: i for i, n in enumerate(nodes)}
    F_dim = max(1, len(vocab))
    X = torch.zeros((len(nodes), F_dim), dtype=torch.float32)
    for n in nodes:
        for tok in (g.nodes[n].get("feature") or "").split("|"):
            j = vocab.get(tok)
            if j is not None:
                X[nid[n], j] = 1.0
    if g.number_of_edges() == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        ei: List[List[int]] = []
        for u, v in g.edges():
            ei.append([nid[u], nid[v]])
            ei.append([nid[v], nid[u]])
        edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous()
    return Data(x=X, edge_index=edge_index)


class _GNN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, kind: str):
        super().__init__()
        ConvCls = GCNConv if kind == "gcn" else SAGEConv
        self.conv1 = ConvCls(in_dim, hidden)
        self.conv2 = ConvCls(hidden, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, edge_index, batch):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h = global_mean_pool(h, batch)
        return self.head(h).squeeze(-1)


def _set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_eval(
    kind: str,
    train_records: List[Dict[str, Any]],
    eval_records: List[Dict[str, Any]],
    *,
    peckshield: Optional[Set[str]] = None,
    max_neighbours: int = 5,
    hidden: int = 64,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 5e-5,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    if not eval_records:
        return []
    _set_seed(seed)
    train_graphs, kept_tr = build_graphs(train_records, peckshield=peckshield,
                                          max_neighbours=max_neighbours)
    eval_graphs,  kept_ev = build_graphs(eval_records, peckshield=peckshield,
                                          max_neighbours=max_neighbours)

    vocab = build_feature_vocab(train_graphs)
    Xtr = [nx_to_pyg(g, vocab) for g in train_graphs]
    Xev = [nx_to_pyg(g, vocab) for g in eval_graphs]
    ytr = torch.tensor(
        [1.0 if train_records[i]["label"] == "malicious" else 0.0 for i in kept_tr],
        dtype=torch.float32,
    )
    for d, y in zip(Xtr, ytr):
        d.y = y.view(1)

    pos = int(ytr.sum().item()); neg = len(ytr) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32).to(_DEVICE)

    in_dim = max(1, len(vocab))
    model = _GNN(in_dim, hidden, kind).to(_DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loader = DataLoader(Xtr, batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for batch in loader:
            batch = batch.to(_DEVICE)
            optim.zero_grad()
            logit = model(batch.x, batch.edge_index, batch.batch)
            loss = F.binary_cross_entropy_with_logits(
                logit, batch.y, pos_weight=pos_weight)
            loss.backward(); optim.step()

    model.eval()
    probs_ev = np.zeros(len(Xev), dtype=np.float32)
    with torch.no_grad():
        eval_loader = DataLoader(Xev, batch_size=batch_size, shuffle=False)
        i = 0
        for batch in eval_loader:
            batch = batch.to(_DEVICE)
            logit = model(batch.x, batch.edge_index, batch.batch)
            p = torch.sigmoid(logit).cpu().numpy()
            probs_ev[i:i + len(p)] = p
            i += len(p)

    thr = load_thresholds()
    by_pos = {kept_ev[i]: float(probs_ev[i]) for i in range(len(kept_ev))}
    out: List[Dict[str, Any]] = []
    for k, r in enumerate(eval_records):
        s = by_pos.get(k, 0.0)
        out.append({
            "tx": r.get("tx"),
            "anomaly_score": s,
            "decision": discretize(s, thr),
            "label": r.get("label"),
        })
    return out


def score_with_gcn(train_records, eval_records, **kw):
    return _train_eval("gcn", train_records, eval_records, **kw)


def score_with_sage(train_records, eval_records, **kw):
    return _train_eval("sage", train_records, eval_records, **kw)


if __name__ == "__main__":
    b = load_nomad("data")
    parts = stratified_split_70_15_15(b["withdrawals"], seed=42)
    for k in ("gcn", "sage"):
        t0 = time.time()
        rows = _train_eval(k, parts["train"], parts["test"],
                            peckshield=b["peckshield"], epochs=10)
        scores = np.array([r["anomaly_score"] for r in rows])
        y = np.array([1 if r["label"] == "malicious" else 0 for r in rows])
        print(f"{k:5s}  n={len(rows)}  mean_mal_score={scores[y==1].mean():.3f}  "
              f"mean_norm_score={scores[y==0].mean():.3f}  "
              f"elapsed={time.time()-t0:.1f}s")
