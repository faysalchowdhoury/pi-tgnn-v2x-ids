"""PI-TGNN: Consequence-aware Temporal Graph Neural Network with kinematic
consistency.

Edge-conditioned graph attention over the vehicle-RSU interaction graph, a GRU
node memory for temporal context, and two heads:

* **anomaly** head  -> spoofing probability (BCE),
* **consequence** head -> normalised physical hazard in [0,1] (MSE),

trained with a light **kinematic-consistency regulariser** that discourages
alarms on dead-reckoning-consistent nodes (reducing false alarms on benign GPS
drift) -- an explicit, reported trade-off against ultra-stealthy low-consequence
attacks. Implemented in plain PyTorch (no torch-geometric dependency).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from data_gen.graph_builder import TemporalGraph, N_NODE_FEAT, N_EDGE_FEAT, NODE_FEATURES

RES_IDX = NODE_FEATURES.index("residual")


def _segment_softmax(score: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    score = score - score.max()
    e = torch.exp(score)
    denom = torch.zeros(n, device=e.device).index_add_(0, index, e) + 1e-9
    return e / denom[index]


class EdgeAttention(nn.Module):
    def __init__(self, dim: int, edim: int):
        super().__init__()
        self.msg = nn.Linear(dim + edim, dim)
        self.att = nn.Linear(2 * dim + edim, 1)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index, edge_feat, n):
        if edge_index.shape[1] == 0:
            return torch.zeros_like(x)
        src, dst = edge_index[0], edge_index[1]
        m = self.msg(torch.cat([x[src], edge_feat], dim=-1))
        s = self.act(self.att(torch.cat([x[dst], x[src], edge_feat], dim=-1))).squeeze(-1)
        alpha = _segment_softmax(s, dst, n).unsqueeze(-1)
        out = torch.zeros(n, x.shape[1], device=x.device)
        out.index_add_(0, dst, alpha * m)
        return out


class PITGNN(nn.Module):
    def __init__(self, hidden: int = 64, use_edge: bool = True, use_residual: bool = True,
                 dual_head: bool = True, tbptt: int = 20):
        super().__init__()
        self.hidden = hidden
        self.use_edge = use_edge
        self.use_residual = use_residual
        self.dual_head = dual_head
        self.tbptt = tbptt
        self.enc = nn.Linear(N_NODE_FEAT, hidden)
        self.att = EdgeAttention(hidden, N_EDGE_FEAT if use_edge else 0)
        self.gru = nn.GRUCell(hidden, hidden)
        self.head_a = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                    nn.Linear(hidden // 2, 1))
        self.head_c = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                    nn.Linear(hidden // 2, 1))

    def forward(self, tg: TemporalGraph, mu: torch.Tensor, sd: torch.Tensor
                ) -> Tuple[torch.Tensor, ...]:
        H = torch.zeros(tg.n_nodes, self.hidden)
        P, C, Y, CS, CL, R = [], [], [], [], [], []
        for k, s in enumerate(tg.steps):
            idx = torch.from_numpy(s["idx"])
            feat = (torch.from_numpy(s["feat"]) - mu) / sd
            if not self.use_residual:
                feat = feat.clone(); feat[:, RES_IDX] = 0.0
            x = torch.relu(self.enc(feat))
            n = x.shape[0]
            if self.use_edge:
                ei = torch.from_numpy(s["edge_index"])
                ef = torch.from_numpy(s["edge_feat"])
                agg = self.att(x, ei, ef, n)
            else:                                   # plain mean aggregation, no edge features
                ei = torch.from_numpy(s["edge_index"])
                agg = torch.zeros_like(x)
                if ei.shape[1] > 0:
                    deg = torch.zeros(n).index_add_(0, ei[1], torch.ones(ei.shape[1]))
                    agg.index_add_(0, ei[1], x[ei[0]])
                    agg = agg / deg.clamp(min=1).unsqueeze(-1)
            h_new = self.gru(agg + x, H[idx])
            H = H.index_copy(0, idx, h_new)
            P.append(self.head_a(h_new).squeeze(-1))
            C.append(torch.sigmoid(self.head_c(h_new)).squeeze(-1))
            Y.append(torch.from_numpy(s["y"]))
            CS.append(torch.from_numpy(s["cons"]))
            CL.append(torch.from_numpy(s["cls"]))
            R.append(torch.from_numpy(s["feat"][:, RES_IDX]))
            if (k + 1) % self.tbptt == 0:
                H = H.detach()
        return (torch.cat(P), torch.cat(C), torch.cat(Y), torch.cat(CS),
                torch.cat(CL), torch.cat(R))


def pitgnn_loss(logit_p, c_pred, y, cons, residual,
                alpha: float = 1.0, beta: float = 0.1, res_tol: float = 0.8):
    bce = nn.functional.binary_cross_entropy_with_logits(logit_p, y)
    mse = nn.functional.mse_loss(c_pred, cons)
    # kinematic-consistency regulariser: discourage alarms on dead-reckoning-consistent nodes
    consistent = (residual < res_tol).float()
    phys = (torch.sigmoid(logit_p) * consistent).sum() / (consistent.sum() + 1e-6)
    total = bce + alpha * mse + beta * phys
    return total, {"bce": float(bce.detach()), "mse": float(mse.detach()),
                   "phys": float(phys.detach())}
