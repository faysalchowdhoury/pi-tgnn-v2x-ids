"""Publication figures (dark theme, Times New Roman, 300 DPI, svg/pdf/jpg)."""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from data_gen.v2x_corridor_generator import ATTACK_CLASSES, HIGH_HAZARD

RESULTS = os.path.join(ROOT, "results")
FIGDIR = os.path.join(ROOT, "figures")
os.makedirs(FIGDIR, exist_ok=True)

FIG_BG, AX_BG, FG, GRID = "#0f0f1a", "#1a1a2e", "#d6d6e0", "#33334d"
C_OURS, C_STD, C_RF, C_IF, C_RULE = "#ffd166", "#06d6a0", "#4cc9f0", "#ff6b6b", "#b983ff"
METHOD_COLOR = {"PI-TGNN (ours)": C_OURS, "Standard GNN": C_STD, "Random Forest": C_RF,
                "Isolation Forest": C_IF, "Rule-Based": C_RULE}
CLASS_COLOR = {"benign": "#7a7a99", "Ghost_Braking_Spoof": "#ff6b6b",
               "Sybil_Phantom_Vehicle": "#ffd166", "Gradual_Position_Drift": "#4cc9f0",
               "SPaT_Green_Spoof": "#06d6a0"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "figure.facecolor": FIG_BG, "axes.facecolor": AX_BG, "savefig.facecolor": FIG_BG,
    "axes.edgecolor": GRID, "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": FG, "ytick.color": FG, "grid.color": GRID, "grid.alpha": 0.5,
    "axes.grid": True, "axes.axisbelow": True, "font.size": 11,
})


def _save(fig, name):
    for ext in ("svg", "pdf", "jpg"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return ax


def fig1_consequence_vs_detection():
    df = pd.read_csv(os.path.join(RESULTS, "scored_messages.csv"))
    df["cls"] = df.attack_cls.map({i: c for i, c in enumerate(ATTACK_CLASSES)})
    fig, ax = plt.subplots(figsize=(8, 6))
    for cls in ATTACK_CLASSES:
        g = df[df.cls == cls]
        if len(g) == 0:
            continue
        ax.scatter(g.score_anom, g.consequence_true, s=14, alpha=0.5,
                   color=CLASS_COLOR[cls], label=cls.replace("_", " "),
                   edgecolors="none", rasterized=True)
    ax.axhline(0.5, color=FG, ls=":", lw=1, alpha=0.5)
    ax.axvline(0.5, color=FG, ls=":", lw=1, alpha=0.5)
    ax.text(0.97, 0.52, "high-hazard zone", color=FG, fontsize=9, ha="right", alpha=0.8)
    _style(ax)
    ax.set_xlabel("PI-TGNN anomaly score")
    ax.set_ylabel("physical consequence severity")
    ax.set_title("Anomaly score vs. physical consequence by attack type\n"
                 "high-hazard spoofs flagged (right of threshold); benign and\n"
                 "low-consequence drift suppressed (deprioritised)", color=FG, fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(1.01, 0.5))
    _save(fig, "fig1_consequence_vs_detection")


def fig2_pr_and_cwdr():
    pr = np.load(os.path.join(RESULTS, "pr_curves.npz"))
    cw = pd.read_csv(os.path.join(RESULTS, "cwdr_curves.csv"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for name, c in METHOD_COLOR.items():
        if name + "_p" in pr:
            lw = 2.6 if name.startswith("PI-TGNN") else 1.6
            ax1.plot(pr[name + "_r"], pr[name + "_p"], color=c, lw=lw,
                     label=name.replace(" (ours)", ""))
    _style(ax1); ax1.set_xlabel("recall"); ax1.set_ylabel("precision")
    ax1.set_title("Precision-Recall", color=FG); ax1.legend(frameon=False, fontsize=8.5)
    for name, g in cw.groupby("method"):
        c = METHOD_COLOR.get(name, FG)
        lw = 2.6 if name.startswith("PI-TGNN") else 1.6
        ax2.plot(g.fpr_budget, g.cwdr, "-o", ms=4, color=c, lw=lw,
                 label=name.replace(" (ours)", ""))
    ax2.set_xscale("log"); _style(ax2)
    ax2.set_xlabel("false-alarm budget (FPR, log)")
    ax2.set_ylabel("consequence-weighted detection recall")
    ax2.set_title("CWDR vs false-alarm budget", color=FG)
    fig.suptitle("Detection quality and consequence-weighted recall", color=FG, fontsize=13)
    fig.tight_layout()
    _save(fig, "fig2_pr_and_cwdr_curves")


def fig3_shockwave():
    df = pd.read_csv(os.path.join(RESULTS, "shockwave_timeline.csv"))
    atk = df[df.attack_present == 1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax2 = ax.twinx()
    ax.plot(df.tsec, df.mean_speed, color=C_RF, lw=2, label="platoon mean speed")
    ax2.plot(df.tsec, df.hazard_pred, color=C_OURS, lw=2.2, label="PI-TGNN hazard")
    ax2.plot(df.tsec, df.alert, color=C_STD, lw=1.6, ls="--", label="PI-TGNN anomaly alert")
    if len(atk):
        t0, t1 = atk.tsec.min(), atk.tsec.max()
        ax.axvspan(t0, t1, color="#ff6b6b", alpha=0.12)
        ax.text(t0, ax.get_ylim()[1] * 0.96, " attack window", color="#ff6b6b", fontsize=9)
        first_alert = df[(df.alert > 0.5)]
        if len(first_alert):
            ta = first_alert.tsec.min()
            ax.axvline(ta, color=C_STD, ls=":", lw=1.3)
            ax.text(ta, ax.get_ylim()[0] + 1, f" first alert @ {ta:.1f}s", color=C_STD, fontsize=9)
    _style(ax)
    for sp in ("top",):
        ax2.spines[sp].set_visible(False)
    ax.set_xlabel("time (s)"); ax.set_ylabel("speed (m/s)", color=C_RF)
    ax2.set_ylabel("hazard / anomaly score", color=C_OURS); ax2.set_ylim(0, 1.05)
    ax.set_title("Platoon shockwave timeline under ghost-braking spoof", color=FG)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=9, loc="center right")
    _save(fig, "fig3_platoon_shockwave_timeline")


def fig4_ablation_latency():
    df = pd.read_csv(os.path.join(RESULTS, "ids_benchmark.csv"))
    order = ["Rule-Based", "Isolation Forest", "Random Forest", "Standard GNN", "PI-TGNN (ours)"]
    df = df.set_index("method").loc[[m for m in order if m in df.method.values]].reset_index()
    colors = [METHOD_COLOR[m] for m in df.method]
    labels = [m.replace(" (ours)", "") for m in df.method]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    x = np.arange(len(df))
    ax1.bar(x, df.cwdr_at_1pct_fpr, color=colors)
    _style(ax1); ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylabel("CWDR @ 1% FPR"); ax1.set_title("Consequence-weighted recall", color=FG)
    ax1.set_ylim(0, 1.05)
    ax2.bar(x, df.latency_ms, color=colors)
    ax2.set_yscale("log"); _style(ax2)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.set_ylabel("latency (ms/graph, log)"); ax2.set_title("Inference latency", color=FG)
    fig.suptitle("Ablation and latency across detectors", color=FG, fontsize=13)
    fig.tight_layout()
    _save(fig, "fig4_ablation_and_latency")


def fig5_veremi_realdata():
    p = os.path.join(RESULTS, "veremi_eval.csv")
    if not os.path.exists(p):
        print("  (skipping fig5: run experiments/eval_veremi.py with the VeReMi dataset)")
        return
    df = pd.read_csv(p)
    df = df[df.attacker_type != "MEAN"].copy()
    df["short"] = df.attacker_type.str.split(":").str[1]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    colors = [C_OURS if v >= 0.7 else C_IF for v in df.residual_roc_auc]
    ax.bar(df.short, df.residual_roc_auc, color=colors)
    ax.axhline(0.5, color=FG, ls=":", lw=1, alpha=0.6)
    ax.text(0.02, 0.52, "chance", color=FG, fontsize=9, transform=ax.get_yaxis_transform())
    for i, v in enumerate(df.residual_roc_auc):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", color=FG, fontsize=9)
    _style(ax)
    ax.set_ylim(0, 1.05); ax.set_ylabel("residual detector ROC-AUC")
    ax.set_xlabel("VeReMi attack type (real public benchmark)")
    ax.set_title("Physics core on REAL VeReMi data: kinematic residual catches 4/5\n"
                 "attack types (~0.96); misses constant-offset (needs graph context)", color=FG)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    _save(fig, "fig5_veremi_realdata")


def fig6_zero_shot():
    p = os.path.join(RESULTS, "zero_shot.csv")
    if not os.path.exists(p):
        print("  (skipping fig6: run experiments/zero_shot.py first)")
        return
    df = pd.read_csv(p)
    df = df[df.unseen_attack != "MEAN"].copy()
    df["short"] = df.unseen_attack.str.replace("_", " ")
    x = np.arange(len(df)); w = 0.26
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(x - w, df.PI_TGNN, w, color=C_OURS, label="PI-TGNN (ours)")
    ax.bar(x, df.Random_Forest, w, color=C_RF, label="Random Forest")
    ax.bar(x + w, df.Rule_Based, w, color=C_RULE, label="Rule-Based")
    _style(ax)
    ax.set_xticks(x); ax.set_xticklabels(df.short, rotation=12, ha="right")
    ax.set_ylabel("recall on the UNSEEN attack @1% FPR"); ax.set_ylim(0, 1.08)
    ax.set_title("Zero-shot generalisation: trained on 3 attacks, tested on the 4th\n"
                 "PI-TGNN transfers to unseen kinematic attacks; supervised RF memorises",
                 color=FG, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    _save(fig, "fig6_zero_shot_generalization")


def main():
    print("Generating publication figures (300 DPI) ...")
    fig1_consequence_vs_detection()
    fig2_pr_and_cwdr()
    fig3_shockwave()
    fig4_ablation_latency()
    fig5_veremi_realdata()
    fig6_zero_shot()
    print("Figures written to", FIGDIR)


if __name__ == "__main__":
    main()
