"""Pytest suite: generator physics, graph construction, kinematic residuals,
dual-head shapes, a training step, the VeReMi parser, and figure export."""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data_gen.v2x_corridor_generator import (generate_scenario, ATTACK_CLASSES, HIGH_HAZARD,
                                             A_MAX, B_MAX, V_MAX)
from data_gen.graph_builder import (build_temporal_graph, feature_normalizer,
                                    N_NODE_FEAT, N_EDGE_FEAT)
from data_gen.veremi_loader import parse_records
from models.pi_tgnn import PITGNN, pitgnn_loss
from models.baselines import RuleBasedKinematicFilter, TabularRandomForest


def test_generator_bounds_and_labels():
    sc = generate_scenario(1, "Ghost_Braking_Spoof", n_steps=120)
    d = sc.messages
    veh = d[d.is_rsu == 0]
    assert veh.v_rep.min() >= -1e-6 and veh.v_rep.max() <= V_MAX + 1e-6
    assert (d.consequence.between(0.0, 1.0)).all()
    assert d.y_attack.isin([0, 1]).all()
    assert d[d.y_attack == 1].attack_class.eq("Ghost_Braking_Spoof").all()


def test_consequence_ordering():
    for atk in ["Sybil_Phantom_Vehicle", "SPaT_Green_Spoof"]:
        d = generate_scenario(2, atk, n_steps=120).messages
        hi = d[d.y_attack == 1].consequence.mean()
        lo = d[d.y_attack == 0].consequence.mean()
        assert hi > lo + 0.2, f"{atk}: attack consequence should exceed benign"
    # stealthy drift is a true attack but LOW consequence (core thesis)
    drift = generate_scenario(2, "Gradual_Position_Drift", n_steps=120).messages
    assert drift[drift.y_attack == 1].consequence.mean() < 0.3


def test_kinematic_residual_detects_spoof():
    d = generate_scenario(3, "Ghost_Braking_Spoof", n_steps=120).messages
    atk_res = d[d.y_attack == 1].residual.mean()
    benign_res = d[(d.y_attack == 0) & (d.is_rsu == 0)].residual.mean()
    assert atk_res > benign_res


def test_graph_construction_shapes():
    sc = generate_scenario(1, "Sybil_Phantom_Vehicle", n_steps=60)
    tg = build_temporal_graph(sc.messages)
    assert tg.n_nodes > 0 and len(tg.steps) == 60
    s = tg.steps[30]
    assert s["feat"].shape[1] == N_NODE_FEAT
    assert s["edge_feat"].shape[1] == N_EDGE_FEAT
    assert s["edge_index"].shape[0] == 2
    assert len(s["y"]) == len(s["idx"]) == s["feat"].shape[0]


def test_pitgnn_forward_shapes():
    sc = generate_scenario(1, "benign", n_steps=40)
    tg = build_temporal_graph(sc.messages)
    mu, sd = feature_normalizer([tg])
    model = PITGNN()
    logit, c, y, cons, cls, res = model(tg, torch.tensor(mu), torch.tensor(sd))
    assert logit.shape == c.shape == y.shape == cons.shape == res.shape
    assert (c >= 0).all() and (c <= 1).all()               # consequence head bounded


def test_training_step_reduces_loss():
    scs = [build_temporal_graph(generate_scenario(s, "Ghost_Braking_Spoof", n_steps=80).messages)
           for s in (1, 2)]
    mu, sd = feature_normalizer(scs)
    mu_t, sd_t = torch.tensor(mu), torch.tensor(sd)
    model = PITGNN()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = last = None
    for it in range(6):
        opt.zero_grad(); tot = 0.0
        for tg in scs:
            logit, c, y, cons, cls, res = model(tg, mu_t, sd_t)
            loss, _ = pitgnn_loss(logit, c, y, cons, res)
            loss.backward(); tot += float(loss)
        opt.step()
        if it == 0:
            first = tot
        last = tot
    assert last < first


def test_ablation_variant_runs():
    sc = generate_scenario(1, "benign", n_steps=40)
    tg = build_temporal_graph(sc.messages)
    mu, sd = feature_normalizer([tg])
    std = PITGNN(use_edge=False, use_residual=False, dual_head=False)
    logit, c, y, cons, cls, res = std(tg, torch.tensor(mu), torch.tensor(sd))
    assert torch.isfinite(logit).all()


def test_veremi_parser_on_fixture():
    records = []
    for t in range(5):
        for sender in (1, 2):
            records.append({"type": 3, "rcvTime": t * 0.1, "sender": sender,
                            "messageID": t * 10 + sender,
                            "pos": [sender * 20 + t * 2.5, 0.0, 0.0],
                            "spd": [25.0, 0.0, 0.0]})
    truth = {k: (1 if (k % 10) == 2 else 0) for k in [r["messageID"] for r in records]}
    df = parse_records(records, truth)
    assert not df.empty
    assert {"residual", "v_rep", "a_rep", "y_attack"} <= set(df.columns)
    assert df.y_attack.sum() == 5           # sender 2's messages flagged


def test_figures_from_fixtures(tmp_path, monkeypatch):
    import pandas as pd
    import visualization.make_pub_figures as mpf
    res = tmp_path / "results"; fig = tmp_path / "figures"
    res.mkdir(); fig.mkdir()
    n = 200
    rng = np.random.default_rng(0)
    cls = rng.integers(0, len(ATTACK_CLASSES), n)
    pd.DataFrame({"y": (cls > 0).astype(int), "consequence_true": rng.random(n),
                  "attack_cls": cls, "score_anom": rng.random(n),
                  "score_cons": rng.random(n)}).to_csv(res / "scored_messages.csv", index=False)
    methods = ["PI-TGNN (ours)", "Standard GNN", "Random Forest", "Isolation Forest", "Rule-Based"]
    np.savez(res / "pr_curves.npz", **{f"{m}_{k}": np.linspace(0, 1, 10)
                                       for m in methods for k in ("p", "r")})
    pd.DataFrame([{"method": m, "fpr_budget": b, "cwdr": rng.random()}
                 for m in methods for b in (0.001, 0.01, 0.1)]).to_csv(res / "cwdr_curves.csv", index=False)
    pd.DataFrame({"t": range(50), "tsec": np.arange(50) * 0.1,
                  "mean_speed": 25 + rng.normal(0, 1, 50), "hazard_pred": rng.random(50),
                  "alert": rng.random(50),
                  "attack_present": (np.arange(50) > 20) & (np.arange(50) < 35)}).to_csv(
        res / "shockwave_timeline.csv", index=False)
    pd.DataFrame([{"method": m, "roc_auc": 0.9, "pr_auc": 0.6, "far_at_90tpr": 0.1,
                   "recall_at_1pct_fpr": 0.7, "cwdr_at_1pct_fpr": rng.random(),
                   "latency_ms": rng.random()} for m in methods]).to_csv(
        res / "ids_benchmark.csv", index=False)
    monkeypatch.setattr(mpf, "RESULTS", str(res))
    monkeypatch.setattr(mpf, "FIGDIR", str(fig))
    mpf.main()
    for name in ("fig1_consequence_vs_detection", "fig2_pr_and_cwdr_curves",
                 "fig3_platoon_shockwave_timeline", "fig4_ablation_and_latency"):
        for ext in ("svg", "pdf", "jpg"):
            assert (fig / f"{name}.{ext}").exists()
