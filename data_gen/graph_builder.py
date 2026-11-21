"""Convert a scenario's BSM/SPaT message table into a temporal sequence of
spatiotemporal interaction graphs for the (T)GNN models.

Per timestep: nodes = transmitting vehicles/RSUs; edges connect nodes within
DSRC/C-V2X range (``DSRC_RANGE``). Node features are
``[x, y, v, a, heading, dtau, residual]``; edge features are
``[Δspacing, Δclosing-speed, kinematic-residual(source)]``. Node indices are
global within the scenario so a temporal model can carry per-vehicle memory.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from data_gen.v2x_corridor_generator import DSRC_RANGE, DT, ATTACK_CLASSES

NODE_FEATURES = ["x_rep", "y_rep", "v_rep", "a_rep", "heading", "dtau", "residual"]
N_NODE_FEAT = len(NODE_FEATURES)
N_EDGE_FEAT = 3
CLASS_IDX = {c: i for i, c in enumerate(ATTACK_CLASSES)}


@dataclass
class TemporalGraph:
    vids: np.ndarray                    # global node id list (sorted unique vids)
    vid2idx: Dict[int, int]
    steps: List[dict]                   # per-timestep graph dicts
    n_nodes: int


def _dtau(df: pd.DataFrame) -> np.ndarray:
    out = np.full(len(df), DT)
    for _, g in df.groupby("vid"):
        idx = g.index.to_numpy()
        tt = g.t.to_numpy()
        out[idx[1:]] = np.clip((tt[1:] - tt[:-1]) * DT, DT, 2.0)
    return out


def build_temporal_graph(scenario_df: pd.DataFrame) -> TemporalGraph:
    df = scenario_df.sort_values(["vid", "t"]).reset_index(drop=True).copy()
    df["dtau"] = _dtau(df)
    vids = np.sort(df.vid.unique())
    vid2idx = {int(v): i for i, v in enumerate(vids)}
    steps: List[dict] = []
    for t, g in df.groupby("t"):
        g = g.reset_index(drop=True)
        gidx = np.array([vid2idx[int(v)] for v in g.vid])
        feat = g[NODE_FEATURES].to_numpy(dtype=np.float32)
        pos = g[["x_rep", "y_rep"]].to_numpy()
        vel = g["v_rep"].to_numpy()
        res = g["residual"].to_numpy()
        # edges within range (undirected -> both directions)
        src, dst, efeat = [], [], []
        n = len(g)
        for a in range(n):
            d = np.hypot(pos[:, 0] - pos[a, 0], pos[:, 1] - pos[a, 1])
            nbr = np.where((d <= DSRC_RANGE) & (np.arange(n) != a))[0]
            for b in nbr:
                src.append(int(b)); dst.append(int(a))             # local indices, message b -> a
                efeat.append([abs(pos[b, 0] - pos[a, 0]),          # Δ spacing
                              vel[a] - vel[b],                      # closing speed
                              res[b]])                              # source kinematic residual
        steps.append(dict(
            idx=gidx.astype(np.int64),
            feat=feat,
            edge_index=np.array([src, dst], dtype=np.int64) if src else np.zeros((2, 0), np.int64),
            edge_feat=np.array(efeat, dtype=np.float32) if efeat else np.zeros((0, N_EDGE_FEAT), np.float32),
            y=g["y_attack"].to_numpy(dtype=np.float32),
            cons=g["consequence"].to_numpy(dtype=np.float32),
            cls=np.array([CLASS_IDX[c] for c in g["attack_class"]], dtype=np.int64),
        ))
    return TemporalGraph(vids, vid2idx, steps, len(vids))


def feature_normalizer(graphs: List[TemporalGraph]):
    """Fit per-feature mean/std over node features for standardisation."""
    allf = np.concatenate([s["feat"] for g in graphs for s in g.steps], axis=0)
    mu, sd = allf.mean(0), allf.std(0) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)
