"""Train PI-TGNN and baselines, then benchmark V2X intrusion detection with
consequence-aware metrics.

Metrics
-------
* ROC-AUC, PR-AUC                 -- packet-level detection quality
* FAR@90%TPR                      -- false-alarm rate on benign messages at 90% recall
* CWDR@1%FPR                      -- consequence-weighted detection recall within a
                                     1% false-alarm budget (rewards catching high-hazard attacks)
* Recall@1%FPR, latency (ms/graph)

Outputs results/ids_benchmark.csv, results/pr_curves.npz, results/scored_messages.csv
and results/shockwave_timeline.csv.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data_gen.v2x_corridor_generator import (generate_scenario, ATTACK_CLASSES, HIGH_HAZARD, DT)
from data_gen.graph_builder import build_temporal_graph, feature_normalizer, CLASS_IDX
from models.pi_tgnn import PITGNN, pitgnn_loss
from models.baselines import (RuleBasedKinematicFilter, TabularRandomForest, TabularIsolationForest)

RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)
TRAIN_SEEDS = [1, 2, 3, 4, 5]
TEST_SEEDS = [11, 12, 13]
N_STEPS = 200
EPOCHS = 10


def make_scenarios(seeds: List[int], n_steps: int = N_STEPS):
    out = []
    for sd in seeds:
        for atk in ATTACK_CLASSES:                 # benign + 4 attack classes
            sc = generate_scenario(sd, atk, n_steps=n_steps)
            out.append((sc.messages, build_temporal_graph(sc.messages), atk, sc))
    return out


# --------------------------------------------------------------------------- #
# Training (PI-TGNN and the ablated Standard GNN)
# --------------------------------------------------------------------------- #
def train_gnn(model: PITGNN, scenarios, mu, sd, epochs=EPOCHS, lr=3e-3,
              dual_head=True, phys=True) -> PITGNN:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mu_t, sd_t = torch.tensor(mu), torch.tensor(sd)
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for df, tg, atk, sc in scenarios:
            opt.zero_grad()
            logit, c, y, cons, cls, res = model(tg, mu_t, sd_t)
            if dual_head:
                loss, _ = pitgnn_loss(logit, c, y, cons, res,
                                      beta=(0.1 if phys else 0.0))
            else:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y)
            loss.backward(); opt.step(); tot += float(loss.detach())
        if ep == 0 or (ep + 1) % 5 == 0:
            print(f"      epoch {ep+1:2d}/{epochs}  loss={tot/len(scenarios):.4f}")
    model.eval()
    return model


@torch.no_grad()
def score_gnn(model, scenarios, mu, sd):
    mu_t, sd_t = torch.tensor(mu), torch.tensor(sd)
    P, Cc, Y, CS, CL = [], [], [], [], []
    lat = []
    for df, tg, atk, sc in scenarios:
        t0 = time.perf_counter()
        logit, c, y, cons, cls, res = model(tg, mu_t, sd_t)
        lat.append((time.perf_counter() - t0) / max(len(tg.steps), 1) * 1e3)
        P.append(torch.sigmoid(logit).numpy()); Cc.append(c.numpy())
        Y.append(y.numpy()); CS.append(cons.numpy()); CL.append(cls.numpy())
    return (np.concatenate(P), np.concatenate(Cc), np.concatenate(Y),
            np.concatenate(CS), np.concatenate(CL), float(np.mean(lat)))


def score_tabular(model, scenarios):
    S, Y, CS, CL, lat = [], [], [], [], []
    for df, tg, atk, sc in scenarios:
        t0 = time.perf_counter()
        s = model.score(df)
        lat.append((time.perf_counter() - t0) / max(df.t.nunique(), 1) * 1e3)
        S.append(s); Y.append(df.y_attack.to_numpy()); CS.append(df.consequence.to_numpy())
        CL.append(df.attack_class.map(CLASS_IDX).to_numpy())
    return (np.concatenate(S), np.concatenate(Y), np.concatenate(CS),
            np.concatenate(CL), float(np.mean(lat)))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def threshold_at_fpr(score, y, fpr_budget):
    benign = score[y == 0]
    return float(np.quantile(benign, 1.0 - fpr_budget))


def metrics(score, y, cons, fpr_budget=0.01):
    roc = roc_auc_score(y, score)
    pr = average_precision_score(y, score)
    # FAR @ 90% TPR
    fpr, tpr, thr = roc_curve(y, score)
    i = np.argmax(tpr >= 0.90)
    far90 = float(fpr[i])
    # CWDR and recall at a 1% false-alarm budget
    t = threshold_at_fpr(score, y, fpr_budget)
    atk = y == 1
    det = (score >= t) & atk
    recall = float(det.sum() / max(atk.sum(), 1))
    cwdr = float((cons[atk] * (score[atk] >= t)).sum() / max(cons[atk].sum(), 1e-9))
    return {"roc_auc": roc, "pr_auc": pr, "far_at_90tpr": far90,
            "recall_at_1pct_fpr": recall, "cwdr_at_1pct_fpr": cwdr}


# --------------------------------------------------------------------------- #
def main():
    torch.manual_seed(0); np.random.seed(0)
    print("[1/4] generating scenarios ...")
    train = make_scenarios(TRAIN_SEEDS)
    test = make_scenarios(TEST_SEEDS)
    mu, sd = feature_normalizer([tg for _, tg, _, _ in train])
    train_df = pd.concat([d for d, _, _, _ in train], ignore_index=True)

    print("[2/4] training PI-TGNN (ours) + Standard GNN baseline ...")
    pit = train_gnn(PITGNN(use_edge=True, use_residual=True, dual_head=True),
                    train, mu, sd, dual_head=True, phys=True)
    print("    standard GNN (no edges / no residual / no consequence) ...")
    stdgnn = train_gnn(PITGNN(use_edge=False, use_residual=False, dual_head=False),
                       train, mu, sd, dual_head=False, phys=False)

    print("[3/4] fitting tabular baselines ...")
    rf = TabularRandomForest().fit(train_df)
    iso = TabularIsolationForest().fit(train_df)
    rule = RuleBasedKinematicFilter()

    print("[4/4] scoring + metrics ...")
    rows, pr_store, scored_all = [], {}, {}
    # GNN models
    for name, model in [("PI-TGNN (ours)", pit), ("Standard GNN", stdgnn)]:
        s, c, y, cons, cl, lat = score_gnn(model, test, mu, sd)
        m = metrics(s, y, cons); m.update(method=name, latency_ms=lat)
        rows.append(m); scored_all[name] = (s, y, cons)
        pr, rc, _ = precision_recall_curve(y, s); pr_store[name + "_p"] = pr; pr_store[name + "_r"] = rc
        if name.startswith("PI-TGNN"):
            scored = pd.DataFrame({"y": y, "consequence_true": cons, "attack_cls": cl,
                                   "score_anom": s, "score_cons": c})
            scored.to_csv(os.path.join(RESULTS, "scored_messages.csv"), index=False)
            atk = y == 1
            cons_r = float(np.corrcoef(c[atk], cons[atk])[0, 1])
            cons_mae = float(np.mean(np.abs(c[atk] - cons[atk])))
            print(f"    consequence-head: Pearson r={cons_r:.3f}, MAE={cons_mae:.3f} "
                  f"(unique to PI-TGNN; baselines output no severity)")
    # tabular / rule
    for name, model in [("Random Forest", rf), ("Isolation Forest", iso), ("Rule-Based", rule)]:
        s, y, cons, cl, lat = score_tabular(model, test)
        m = metrics(s, y, cons); m.update(method=name, latency_ms=lat)
        rows.append(m); scored_all[name] = (s, y, cons)
        pr, rc, _ = precision_recall_curve(y, s); pr_store[name + "_p"] = pr; pr_store[name + "_r"] = rc

    df = pd.DataFrame(rows)[["method", "roc_auc", "pr_auc", "far_at_90tpr",
                             "recall_at_1pct_fpr", "cwdr_at_1pct_fpr", "latency_ms"]]
    df.to_csv(os.path.join(RESULTS, "ids_benchmark.csv"), index=False)
    np.savez(os.path.join(RESULTS, "pr_curves.npz"), **pr_store)
    # CWDR across false-alarm budgets (fig2)
    budgets = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    crows = []
    for name, (s, y, cons) in scored_all.items():
        atk = y == 1
        for b in budgets:
            t = threshold_at_fpr(s, y, b)
            cwdr = float((cons[atk] * (s[atk] >= t)).sum() / max(cons[atk].sum(), 1e-9))
            crows.append({"method": name, "fpr_budget": b, "cwdr": cwdr})
    pd.DataFrame(crows).to_csv(os.path.join(RESULTS, "cwdr_curves.csv"), index=False)
    _shockwave_timeline(pit, mu, sd)

    pd.set_option("display.width", 160)
    print("\n================ IDS BENCHMARK (test set) ================")
    print(df.round(3).to_string(index=False))
    print(f"\nWrote results to {RESULTS}")


def _shockwave_timeline(model, mu, sd):
    """Per-timestep platoon hazard for one ghost-braking scenario (fig3)."""
    sc = generate_scenario(11, "Ghost_Braking_Spoof", n_steps=N_STEPS)
    tg = build_temporal_graph(sc.messages)
    mu_t, sd_t = torch.tensor(mu), torch.tensor(sd)
    with torch.no_grad():
        logit, c, y, cons, cls, res = model(tg, mu_t, sd_t)
    # recover per-step max consequence prediction + any-attack alert
    probs = torch.sigmoid(logit).numpy()
    cpred = c.numpy()
    rows, ptr = [], 0
    victim_v = sc.true_v[min(sc.true_v)]
    for k, s in enumerate(tg.steps):
        n = len(s["idx"])
        seg = slice(ptr, ptr + n); ptr += n
        # platoon min TTC proxy from true speeds/positions
        rows.append(dict(t=k, tsec=k * DT,
                         mean_speed=float(np.mean([v[k] for v in sc.true_v.values()])),
                         hazard_pred=float(cpred[seg].max()),
                         alert=float(probs[seg].max()),
                         attack_present=int(s["y"].max() > 0)))
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "shockwave_timeline.csv"), index=False)


if __name__ == "__main__":
    main()
