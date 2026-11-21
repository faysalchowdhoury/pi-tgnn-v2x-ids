"""Zero-shot generalisation to unseen attack types.

Supervised packet-level IDS memorises the signatures of attacks seen in
training; a physics-informed detector should transfer to novel attacks. For each
held-out attack class, every model is trained on the *other* three attacks (plus
benign) and evaluated on the held-out class. Reports detection recall at a 1%
false-alarm budget. Outputs results/zero_shot.csv.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data_gen.v2x_corridor_generator import generate_scenario, ATTACK_CLASSES
from data_gen.graph_builder import build_temporal_graph, feature_normalizer
from models.pi_tgnn import PITGNN
from models.baselines import TabularRandomForest, RuleBasedKinematicFilter
import experiments.train_and_eval as te

RESULTS = os.path.join(ROOT, "results")
ATTACKS = ATTACK_CLASSES[1:]


def _scenarios(seeds, classes):
    out = []
    for sd in seeds:
        for atk in ["benign"] + list(classes):
            sc = generate_scenario(sd, atk, n_steps=te.N_STEPS)
            out.append((sc.messages, build_temporal_graph(sc.messages), atk, sc))
    return out


def _recall_at_fpr(score, y, fpr=0.01):
    t = float(np.quantile(score[y == 0], 1.0 - fpr))
    atk = y == 1
    return float(((score >= t) & atk).sum() / max(atk.sum(), 1))


def main():
    torch.manual_seed(0); np.random.seed(0)
    rows = []
    for held in ATTACKS:
        seen = [a for a in ATTACKS if a != held]
        print(f"[holdout] unseen = {held}  (train on {seen})")
        train = _scenarios(te.TRAIN_SEEDS, seen)
        test = _scenarios(te.TEST_SEEDS, [held])             # benign + the unseen attack only
        mu, sd = feature_normalizer([tg for _, tg, _, _ in train])
        train_df = pd.concat([d for d, _, _, _ in train], ignore_index=True)

        pit = te.train_gnn(PITGNN(use_edge=True, use_residual=True, dual_head=True),
                           train, mu, sd, epochs=6, dual_head=True, phys=True)
        rf = TabularRandomForest().fit(train_df)
        rule = RuleBasedKinematicFilter()

        s_pit, _, y, cons, cl, _ = te.score_gnn(pit, test, mu, sd)
        s_rf, y2, _, _, _ = te.score_tabular(rf, test)
        s_ru, y3, _, _, _ = te.score_tabular(rule, test)
        rows.append(dict(unseen_attack=held,
                         PI_TGNN=_recall_at_fpr(s_pit, y),
                         Random_Forest=_recall_at_fpr(s_rf, y2),
                         Rule_Based=_recall_at_fpr(s_ru, y3)))

    df = pd.DataFrame(rows)
    df.loc["mean"] = {"unseen_attack": "MEAN", "PI_TGNN": df.PI_TGNN.mean(),
                      "Random_Forest": df.Random_Forest.mean(), "Rule_Based": df.Rule_Based.mean()}
    df.to_csv(os.path.join(RESULTS, "zero_shot.csv"), index=False)
    print("\n======== ZERO-SHOT detection recall @1% FPR on the UNSEEN attack ========")
    print(df.round(3).to_string(index=False))
    print(f"\nWrote {os.path.join(RESULTS, 'zero_shot.csv')}")


if __name__ == "__main__":
    main()
