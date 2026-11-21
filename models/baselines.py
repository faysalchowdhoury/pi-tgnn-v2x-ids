"""Baselines for V2X intrusion detection.

* ``RuleBasedKinematicFilter`` -- SAE-J2945/1-style plausibility checks
  (dead-reckoning residual + kinematic-bound violations), parameter-free.
* ``TabularRandomForest`` / ``TabularIsolationForest`` -- packet-level tabular
  IDS with no graph topology.
* The ``StandardGNN`` baseline is the :class:`PITGNN` with edges, kinematic
  residual, and the consequence head disabled (see ``experiments/train_and_eval``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier

from data_gen.graph_builder import NODE_FEATURES
from data_gen.v2x_corridor_generator import A_MAX, B_MAX, V_MAX

RES_TOL = 1.2       # m, dead-reckoning residual alarm threshold (~2.4 sigma GPS)


class RuleBasedKinematicFilter:
    name = "Rule-Based Kinematic"

    def score(self, df: pd.DataFrame) -> np.ndarray:
        res = df["residual"].to_numpy()
        a, v = df["a_rep"].to_numpy(), df["v_rep"].to_numpy()
        bound = ((a < -B_MAX - 1e-6) | (a > A_MAX + 1e-6) |
                 (v < -1e-6) | (v > V_MAX + 1e-6)).astype(float)
        res_score = res / (res + RES_TOL)                      # in [0,1), monotone in residual
        return np.clip(np.maximum(res_score, bound), 0.0, 1.0)


class TabularRandomForest:
    name = "Random Forest (tabular)"

    def __init__(self, seed: int = 0):
        self.clf = RandomForestClassifier(n_estimators=200, max_depth=12,
                                          class_weight="balanced", random_state=seed, n_jobs=-1)

    def fit(self, df: pd.DataFrame) -> "TabularRandomForest":
        self.clf.fit(df[NODE_FEATURES].to_numpy(), df["y_attack"].to_numpy())
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(df[NODE_FEATURES].to_numpy())[:, 1]


class TabularIsolationForest:
    name = "Isolation Forest (tabular)"

    def __init__(self, seed: int = 0):
        self.clf = IsolationForest(n_estimators=200, contamination="auto", random_state=seed)

    def fit(self, df: pd.DataFrame) -> "TabularIsolationForest":
        self.clf.fit(df[NODE_FEATURES].to_numpy())
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        s = -self.clf.score_samples(df[NODE_FEATURES].to_numpy())   # higher = more anomalous
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo + 1e-9)
