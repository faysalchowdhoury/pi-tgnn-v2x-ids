"""Validate the physics core of PI-TGNN on the real public VeReMi benchmark.

PI-TGNN's kinematic-consistency residual is dataset-agnostic (it only needs
reported position + speed), so we evaluate it directly on VeReMi's falsified-BSM
position attacks without domain-transfer assumptions. Reports ROC-AUC / PR-AUC
of the rule-based residual detector on real data. Skips gracefully if the
dataset is not present.
"""
from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data_gen.veremi_loader import load_veremi
from models.baselines import RuleBasedKinematicFilter

RESULTS = os.path.join(ROOT, "results")
DEFAULT_PATH = os.path.join(ROOT, "data", "veremi")
# VeReMi attacker-type codes (van der Heijden et al., 2018)
ATTACK_NAME = {1: "ConstantPos", 2: "ConstantPosOffset", 4: "RandomPos",
               8: "RandomPosOffset", 16: "EventualStop"}


def _attacker_type(scenario_dir: str) -> str:
    hits = glob.glob(os.path.join(scenario_dir, "**", "AttackerType*.sca"), recursive=True)
    for h in hits:
        m = re.search(r"AttackerType(\d+)", os.path.basename(h))
        if m:
            code = int(m.group(1))
            return f"{code}:{ATTACK_NAME.get(code, '?')}"
    return "unknown"


def _scenario_dirs(path: str):
    dirs = set()
    for gt in glob.glob(os.path.join(path, "**", "*GroundTruth*"), recursive=True):
        # scenario root = two levels up from results/GroundTruthJSONlog.json's top veins_maat dir
        p = gt
        while os.path.dirname(p) != path and os.path.dirname(p) != p:
            p = os.path.dirname(p)
        dirs.add(p)
    return sorted(dirs)


def main(path: str = None):
    path = path or os.environ.get("VEREMI_PATH", DEFAULT_PATH)
    scen = _scenario_dirs(path)
    if not scen:
        print("VeReMi dataset not found at", path)
        print("Download it from https://github.com/VeReMi-dataset/VeReMi and extract into")
        print(f"  {DEFAULT_PATH}/<scenario>/   (or set $VEREMI_PATH). Skipping real-data eval.")
        return
    rule = RuleBasedKinematicFilter()
    rows = []
    for d in scen:
        df = load_veremi(d, max_files=60)
        if df.empty or df.y_attack.nunique() < 2:
            continue
        y = df.y_attack.to_numpy()
        rows.append({"attacker_type": _attacker_type(d), "messages": int(len(df)),
                     "attacks": int(y.sum()),
                     "residual_roc_auc": float(roc_auc_score(y, df.residual.to_numpy())),
                     "rule_roc_auc": float(roc_auc_score(y, rule.score(df))),
                     "rule_pr_auc": float(average_precision_score(y, rule.score(df)))})
    if not rows:
        print("No usable VeReMi scenarios (need benign + attack). Skipping.")
        return
    df_out = pd.DataFrame(rows).sort_values("attacker_type")
    mean = {"attacker_type": "MEAN", "messages": int(df_out.messages.sum()),
            "attacks": int(df_out.attacks.sum()),
            "residual_roc_auc": float(df_out.residual_roc_auc.mean()),
            "rule_roc_auc": float(df_out.rule_roc_auc.mean()),
            "rule_pr_auc": float(df_out.rule_pr_auc.mean())}
    df_out = pd.concat([df_out, pd.DataFrame([mean])], ignore_index=True)
    df_out.to_csv(os.path.join(RESULTS, "veremi_eval.csv"), index=False)
    print("======== VeReMi (real public benchmark) -- kinematic-residual detector ========")
    print(df_out.round(3).to_string(index=False))
    print(f"\nWrote {os.path.join(RESULTS, 'veremi_eval.csv')}")


if __name__ == "__main__":
    main()
