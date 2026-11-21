"""Loader for the public **VeReMi** vehicular-misbehavior dataset
(van der Heijden et al., 2018) -- a real, widely used benchmark of falsified
BSM position attacks generated with VEINS/SUMO/OMNeT++.

VeReMi ships per-receiver ``traceJSON-*`` logs (received BSMs) and a
``GroundTruth*`` log labelling each message's ``attackerType``. This module
parses that format into the same message table the rest of the pipeline uses, so
the **kinematic-consistency residual** -- the physics core of PI-TGNN -- can be
validated on real public data (see ``experiments/eval_veremi.py``).

Download (not redistributed here):
  https://github.com/VeReMi-dataset/VeReMi   (or the VeReMi-Extension)
Extract into ``data/veremi/<scenario>/`` and point the loader at it.
"""
from __future__ import annotations

import glob
import json
import math
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

MSG_TYPE_BSM = 3          # VeReMi: type 3 = received BSM record (type 2 = own GPS)


def _iter_json_records(path: str):
    """Yield dict records from a VeReMi log (JSON-lines or a JSON array)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        head = fh.read(1)
        fh.seek(0)
        if head == "[":
            for rec in json.load(fh):
                yield rec
        else:
            for line in fh:
                line = line.strip().rstrip(",")
                if line and line[0] == "{":
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


def parse_records(records: List[dict], truth: Optional[Dict] = None) -> pd.DataFrame:
    """Build a message table from in-memory VeReMi records.

    ``truth`` maps messageID -> attackerType (0 = benign). If absent, each
    record's own ``attackerType``/``attack`` field is used."""
    rows = []
    for r in records:
        if r.get("type") != MSG_TYPE_BSM:          # type 3 = received BSM (type 2 = own GPS)
            continue
        if "sender" not in r or "messageID" not in r:
            continue
        pos = r.get("pos") or r.get("position") or [0, 0, 0]
        spd = r.get("spd") or r.get("speed") or [0, 0, 0]
        sender = int(r.get("sender", r.get("senderPseudo", -1)))
        mid = r.get("messageID", r.get("messageId", -1))
        at = 0
        if truth is not None and mid in truth:
            at = int(truth[mid])
        else:
            at = int(r.get("attackerType", r.get("attack", 0)))
        rows.append(dict(t=float(r.get("rcvTime", r.get("sendTime", 0.0))),
                         vid=sender, x_rep=float(pos[0]), y_rep=float(pos[1]),
                         vx=float(spd[0]), vy=float(spd[1]), y_attack=int(at > 0)))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["v_rep"] = np.hypot(df.vx, df.vy)
    df["heading"] = np.arctan2(df.vy, df.vx)
    # discretise time into 0.1 s BSM frames and derive accel / residual per sender
    df = df.sort_values(["vid", "t"]).reset_index(drop=True)
    df["frame"] = (df.t / 0.1).round().astype(int)
    a = np.zeros(len(df)); res = np.zeros(len(df)); dtau = np.full(len(df), 0.1)
    for _, g in df.groupby("vid"):
        gi = g.index.to_numpy()
        tt, vv, xx = g.t.to_numpy(), g.v_rep.to_numpy(), g.x_rep.to_numpy()
        vxv, vyv = g.vx.to_numpy(), g.vy.to_numpy()
        for j in range(1, len(gi)):
            dt = max(tt[j] - tt[j - 1], 1e-3)
            dtau[gi[j]] = min(dt, 2.0)
            a[gi[j]] = (vv[j] - vv[j - 1]) / dt
            predx = xx[j - 1] + vxv[j - 1] * dt
            predy = g.y_rep.to_numpy()[j - 1] + vyv[j - 1] * dt
            res[gi[j]] = math.hypot(xx[j] - predx, g.y_rep.to_numpy()[j] - predy)
    df["a_rep"] = np.clip(a, -12.0, 6.0)
    df["residual"] = res
    df["dtau"] = dtau
    df["attack_class"] = np.where(df.y_attack == 1, "veremi_position_falsification", "benign")
    df["consequence"] = 0.0
    df["t"] = df["frame"]
    df["lane"] = 0; df["is_rsu"] = 0
    return df


def load_veremi(root: str, max_files: int = 40) -> pd.DataFrame:
    """Parse a VeReMi scenario directory into a message table (empty if absent)."""
    if not root or not os.path.isdir(root):
        return pd.DataFrame()
    gt_files = glob.glob(os.path.join(root, "**", "*GroundTruth*"), recursive=True)
    truth: Dict = {}
    for gf in gt_files:
        for rec in _iter_json_records(gf):
            mid = rec.get("messageID", rec.get("messageId"))
            if mid is not None:
                truth[mid] = rec.get("attackerType", rec.get("attack", 0))
    trace_files = [f for f in glob.glob(os.path.join(root, "**", "JSONlog-*.json"), recursive=True)
                   if "GroundTruth" not in os.path.basename(f)]
    records: List[dict] = []
    seen = set()
    for tf in sorted(trace_files)[:max_files]:
        for rec in _iter_json_records(tf):
            mid = rec.get("messageID")
            if rec.get("type") == MSG_TYPE_BSM and mid not in seen:   # one row per transmitted BSM
                seen.add(mid); records.append(rec)
    return parse_records(records, truth if truth else None)
