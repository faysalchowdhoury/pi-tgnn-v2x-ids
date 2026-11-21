"""V2X corridor generator.

Simulates multi-lane platoon trajectories with the Intelligent Driver Model
(IDM), emits SAE-J2735-style Basic Safety Messages (BSMs) at 10 Hz with
realistic GPS noise, and injects four cyber-attack classes:

  1. Ghost_Braking_Spoof   -- false sudden deceleration inside a fast platoon
  2. Sybil_Phantom_Vehicle -- injected non-existent lead vehicle
  3. Gradual_Position_Drift -- stealthy low-consequence GPS bias
  4. SPaT_Green_Spoof       -- falsified signal phase (red-light conflict)

For every message a **physical consequence severity** in [0,1] is computed from
the hazard a trusting follower would incur (required braking vs. comfortable
limit, and induced inverse time-to-collision), clipped to kinematic bounds.
Benign GPS drift is low-consequence by construction; ghost-braking / Sybil /
SPaT spoofs are high-consequence. This label supervises the consequence head.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# --- physical constants (SI) --------------------------------------------- #
DT = 0.1                      # s, 10 Hz BSM rate
V0 = 30.0                     # m/s desired speed (~108 km/h)
T_HEADWAY = 1.2               # s
A_MAX = 1.5                   # m/s^2 IDM accel
B_COMF = 2.0                  # m/s^2 comfortable decel
B_MAX = 8.0                   # m/s^2 physical max decel
S0 = 2.0                      # m minimum gap
DELTA = 4.0
LANE_WIDTH = 3.5
GPS_SIGMA = 0.5               # m benign position noise (the drift that must NOT alarm)
DSRC_RANGE = 150.0            # m
V_MAX = 40.0                  # m/s kinematic bound
RSU_SPACING = 400.0          # m

ATTACK_CLASSES = ["benign", "Ghost_Braking_Spoof", "Sybil_Phantom_Vehicle",
                  "Gradual_Position_Drift", "SPaT_Green_Spoof"]
HIGH_HAZARD = {"Ghost_Braking_Spoof", "Sybil_Phantom_Vehicle", "SPaT_Green_Spoof"}


def idm_accel(v: float, gap: float, dv: float) -> float:
    """IDM acceleration. gap = bumper-to-bumper distance, dv = v - v_lead."""
    gap = max(gap, 0.1)
    s_star = S0 + max(0.0, v * T_HEADWAY + v * dv / (2.0 * np.sqrt(A_MAX * B_COMF)))
    return A_MAX * (1.0 - (v / V0) ** DELTA - (s_star / gap) ** 2)


@dataclass
class Scenario:
    messages: pd.DataFrame                 # one row per emitted BSM/SPaT
    true_xy: Dict[int, np.ndarray]         # vid -> [T,2] true positions (for figures)
    true_v: Dict[int, np.ndarray]          # vid -> [T] true speed
    n_steps: int
    attack_class: str
    victim_vid: int = -1


def _simulate_platoons(rng, n_lanes: int, veh_per_lane: int, n_steps: int):
    """Deterministic IDM rollout; returns per-vehicle true x,v,a arrays."""
    vids, lane_of, x, v = [], {}, {}, {}
    vid = 0
    for ln in range(n_lanes):
        start = 0.0
        for k in range(veh_per_lane):
            vids.append(vid); lane_of[vid] = ln
            x[vid] = -start - rng.uniform(0, 4)
            v[vid] = V0 - rng.uniform(0, 4)
            start += rng.uniform(22, 32)
            vid += 1
    X = {i: np.zeros(n_steps) for i in vids}
    Vv = {i: np.zeros(n_steps) for i in vids}
    Aa = {i: np.zeros(n_steps) for i in vids}
    lane_members = {ln: sorted([i for i in vids if lane_of[i] == ln], key=lambda i: -x[i])
                    for ln in range(n_lanes)}
    for t in range(n_steps):
        for ln, members in lane_members.items():
            for idx, i in enumerate(members):
                if idx == 0:
                    a = idm_accel(v[i], 1e4, 0.0)           # free leader
                else:
                    lead = members[idx - 1]
                    gap = x[lead] - x[i] - 5.0              # 5 m vehicle length
                    a = idm_accel(v[i], gap, v[i] - v[lead])
                a = float(np.clip(a, -B_MAX, A_MAX))
                X[i][t], Vv[i][t], Aa[i][t] = x[i], v[i], a
                v[i] = float(np.clip(v[i] + a * DT, 0.0, V_MAX))
                x[i] = x[i] + v[i] * DT
    return vids, lane_of, X, Vv, Aa


def _consequence(a_req: float, ttc: float) -> float:
    """Normalised physical hazard a trusting follower would incur."""
    brake = np.clip(a_req / B_MAX, 0.0, 1.0)
    ttc_term = np.clip(1.5 / max(ttc, 0.3), 0.0, 1.0) if ttc < 6.0 else 0.0
    return float(np.clip(0.6 * brake + 0.4 * ttc_term, 0.0, 1.0))


def _follower_hazard(gap: float, v_follow: float, v_lead_reported: float) -> float:
    """Hazard if a follower reacts to a lead reported at (gap, v_lead_reported)."""
    closing = v_follow - v_lead_reported
    if closing <= 0 or gap <= 0:
        a_req, ttc = 0.3, 99.0
    else:
        a_req = closing ** 2 / (2.0 * max(gap, 0.5))
        ttc = gap / closing
    return _consequence(a_req, ttc)


def generate_scenario(seed: int, attack_class: str = "benign",
                      n_lanes: int = 3, veh_per_lane: int = 8,
                      n_steps: int = 300) -> Scenario:
    rng = np.random.default_rng(seed)
    vids, lane_of, X, Vv, Aa = _simulate_platoons(rng, n_lanes, veh_per_lane, n_steps)
    lane_members = {ln: sorted([i for i in vids if lane_of[i] == ln], key=lambda i: -X[i][0])
                    for ln in range(n_lanes)}

    onset = int(n_steps * 0.45)
    duration = int(n_steps * 0.25)
    attackers, phantom_vid = set(), max(vids) + 1
    victim = -1
    if attack_class in ("Ghost_Braking_Spoof", "Gradual_Position_Drift"):
        ln = rng.integers(0, n_lanes)
        mem = lane_members[int(ln)]
        attackers = {mem[len(mem) // 2]}                    # mid-platoon attacker
    elif attack_class == "Sybil_Phantom_Vehicle":
        ln = rng.integers(0, n_lanes)
        victim = lane_members[int(ln)][min(2, veh_per_lane - 1)]
    elif attack_class == "SPaT_Green_Spoof":
        victim = lane_members[0][-1]                        # tail vehicle approaching signal

    rows: List[dict] = []
    prev_rep: Dict[int, Tuple[float, float, float, float]] = {}   # vid -> x,y,v,a last reported

    def lead_of(i, t):
        mem = lane_members[lane_of[i]]
        k = mem.index(i)
        return mem[k - 1] if k > 0 else None

    for t in range(n_steps):
        tsec = t * DT
        for i in vids:
            atk = "benign"
            x_true, y_true = X[i][t], lane_of[i] * LANE_WIDTH
            v_true, a_true = Vv[i][t], Aa[i][t]
            x_rep = x_true + rng.normal(0, GPS_SIGMA)
            y_rep = y_true + rng.normal(0, GPS_SIGMA)
            v_rep, a_rep = v_true, a_true
            # inject attacks
            if i in attackers and onset <= t < onset + duration:
                if attack_class == "Ghost_Braking_Spoof":
                    v_rep = max(0.0, v_true - 12.0)
                    a_rep = -B_MAX
                    atk = "Ghost_Braking_Spoof"
                elif attack_class == "Gradual_Position_Drift":
                    x_rep = x_true + 0.4 * (t - onset)       # stealthy growing bias
                    atk = "Gradual_Position_Drift"
            # consequence from a trusting follower
            cons = 0.0
            f = None
            mem = lane_members[lane_of[i]]
            k = mem.index(i)
            if k < len(mem) - 1:
                f = mem[k + 1]
            if f is not None:
                gap = x_rep - X[f][t] - 5.0
                cons = _follower_hazard(gap, Vv[f][t], v_rep)
            if atk == "benign":
                cons = min(cons, 0.25)                        # benign stays low-hazard
            rows.append(dict(scenario=seed, t=t, tsec=tsec, vid=i, lane=lane_of[i],
                             is_rsu=0, x_true=x_true, y_true=y_true,
                             x_rep=x_rep, y_rep=y_rep, v_rep=v_rep, a_rep=a_rep,
                             heading=0.0, y_attack=int(atk != "benign"),
                             attack_class=atk, consequence=cons))

        # Sybil phantom vehicle just ahead of the victim
        if attack_class == "Sybil_Phantom_Vehicle" and victim >= 0 and onset <= t < onset + duration:
            x_ph = X[victim][t] + 12.0
            y_ph = lane_of[victim] * LANE_WIDTH
            gap = x_ph - X[victim][t] - 5.0
            cons = _follower_hazard(gap, Vv[victim][t], 2.0)   # phantom reports near-stop
            rows.append(dict(scenario=seed, t=t, tsec=tsec, vid=phantom_vid, lane=lane_of[victim],
                             is_rsu=0, x_true=x_ph, y_true=y_ph, x_rep=x_ph, y_rep=y_ph,
                             v_rep=2.0, a_rep=-B_MAX, heading=0.0, y_attack=1,
                             attack_class="Sybil_Phantom_Vehicle", consequence=max(cons, 0.6)))

        # RSU SPaT messages (one per RSU) -- optionally spoofed green
        max_x = max(X[i][t] for i in vids)
        for r, rsu_x in enumerate(np.arange(0, max_x + RSU_SPACING, RSU_SPACING)):
            spoof = (attack_class == "SPaT_Green_Spoof" and onset <= t < onset + duration and r == 1)
            atk = "SPaT_Green_Spoof" if spoof else "benign"
            cons = 0.85 if spoof else 0.0
            rows.append(dict(scenario=seed, t=t, tsec=tsec, vid=10000 + r, lane=-1,
                             is_rsu=1, x_true=rsu_x, y_true=0.0, x_rep=rsu_x, y_rep=0.0,
                             v_rep=0.0, a_rep=0.0, heading=0.0, y_attack=int(spoof),
                             attack_class=atk, consequence=cons))

    df = pd.DataFrame(rows)
    # per-node dead-reckoning kinematic residual (sorted by vid,t)
    df = df.sort_values(["vid", "t"]).reset_index(drop=True)
    res = np.zeros(len(df))
    for vid, g in df.groupby("vid"):
        idx = g.index.to_numpy()
        xr, vr, ar = g.x_rep.to_numpy(), g.v_rep.to_numpy(), g.a_rep.to_numpy()
        for j in range(1, len(idx)):
            pred = xr[j - 1] + vr[j - 1] * DT + 0.5 * ar[j - 1] * DT * DT
            res[idx[j]] = abs(xr[j] - pred)
    df["residual"] = res
    # message inter-arrival interval per transmitter (s)
    dtau = np.full(len(df), DT)
    for _, g in df.groupby("vid"):
        gi = g.index.to_numpy()
        tt = g.t.to_numpy()
        dtau[gi[1:]] = np.clip((tt[1:] - tt[:-1]) * DT, DT, 2.0)
    df["dtau"] = dtau
    true_xy = {i: np.stack([X[i], np.full(n_steps, lane_of[i] * LANE_WIDTH)], 1) for i in vids}
    true_v = {i: Vv[i] for i in vids}
    return Scenario(df, true_xy, true_v, n_steps, attack_class, victim)


def build_dataset(seeds: List[int], attacks: List[str] = None) -> pd.DataFrame:
    """One scenario per (seed, attack class); concatenated message table."""
    attacks = attacks or ATTACK_CLASSES[1:]       # all four attack classes
    frames = []
    for sd in seeds:
        for atk in ["benign"] + list(attacks):
            sc = generate_scenario(sd, atk)
            d = sc.messages.copy()
            d["scenario_id"] = f"{sd}_{atk}"
            frames.append(d)
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    for atk in ATTACK_CLASSES[1:]:
        sc = generate_scenario(1, atk)
        d = sc.messages
        atk_rows = d[d.y_attack == 1]
        print(f"{atk:24s} msgs={len(d):5d} attack={len(atk_rows):4d} "
              f"cons(attack)={atk_rows.consequence.mean():.2f} "
              f"cons(benign)={d[d.y_attack==0].consequence.mean():.2f} "
              f"resid(attack)={atk_rows.residual.mean():.2f}")
