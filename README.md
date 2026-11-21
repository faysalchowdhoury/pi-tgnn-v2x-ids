# Consequence-Aware Temporal Graph Neural Networks with Kinematic Consistency Constraints for V2X Intrusion Detection (PI-TGNN)

A reproducible, open-source framework for **consequence-aware** V2X intrusion
detection. Existing IDS score every spoofed BSM equally by packet-level F1, so a
benign GPS drift can out-alarm a dangerous phantom-braking spoof. **PI-TGNN**
instead pairs anomaly detection with a **physical-consequence severity** head on
a temporal vehicle-RSU interaction graph, using a **kinematic-consistency
residual** as a core signal — and we benchmark it honestly against rule-based,
tabular, and standard-GNN baselines, plus a real-data check on the public
**VeReMi** benchmark.

> **Scope & honesty.** The consequence-severity ground truth (time-to-collision,
> induced shockwave) requires a simulator, so the core experiments use an IDM
> traffic generator with four injected J2735 attack classes. The **physics core
> is additionally validated on the real public VeReMi dataset** (§ Real-data).
> Results include honest limitations (a deprioritised low-consequence attack, an
> underestimated signal-phase severity, and a real-data constant-offset miss) —
> reported, not spun.

## Method

A dynamic interaction graph `G_t = (V_t, E_t)` connects vehicles within 150 m
DSRC/C-V2X range and to roadside units (RSUs).

- **Node features:** position `(x, y)`, speed `v`, acceleration `a`, heading `θ`,
  message inter-arrival `Δτ`, and the **kinematic-consistency residual**
  `r_i = |x_i(t) − x_i(t−Δt) − v_i(t−Δt)Δt − ½ a_i(t−Δt)Δt²|` (dead-reckoning
  mismatch — large when position/speed are falsified).
- **Edge features:** relative spacing `Δs`, closing speed `Δv`, and the source
  node's residual.
- **Model:** edge-conditioned graph attention + a GRU node memory (temporal
  context), with **two heads** — anomaly probability (BCE) and normalised
  physical consequence `ĉ ∈ [0,1]` (MSE) — and a light **kinematic-consistency
  regulariser** discouraging alarms on dead-reckoning-consistent nodes. Plain
  PyTorch, no torch-geometric dependency.

### Attack classes (injected into IDM platoons)

| Class | Description | Consequence |
|---|---|---|
| `Ghost_Braking_Spoof` | false sudden deceleration inside a fast platoon | high |
| `Sybil_Phantom_Vehicle` | injected non-existent lead vehicle → emergency braking | high |
| `Gradual_Position_Drift` | stealthy slow GPS bias | **low (by design)** |
| `SPaT_Green_Spoof` | falsified signal phase (red-light conflict) | high |

## Benchmark (test set, synthetic consequence-labelled corridor)

| Method | ROC-AUC | PR-AUC | FAR@90%TPR | Recall@1%FPR | CWDR@1%FPR | Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|
| **PI-TGNN (ours)** | **0.993** | **0.724** | **0.016** | 0.757 | **0.990** | 2.91 |
| Standard GNN (ablation) | 0.945 | 0.630 | 0.229 | 0.698 | 0.912 | 1.29 |
| Random Forest (tabular) | 0.981 | 0.615 | 0.036 | **0.803** | 0.990 | 0.41 |
| Isolation Forest (tabular) | 0.766 | 0.361 | 0.959 | 0.500 | 0.612 | 0.29 |
| Rule-Based (J2945/1) | 0.581 | 0.224 | 1.000 | 0.298 | 0.471 | 0.003 |

- **Consequence head (unique to PI-TGNN):** predicted vs. true severity Pearson
  `r = 0.64`, MAE `0.21`. Baselines output no severity at all.
- **Per-class recall @1% FPR:** Ghost-braking 1.00, Sybil 1.00, SPaT 1.00,
  Gradual drift **0.03** — the low-consequence drift is deliberately
  **deprioritised** (the core thesis: spend the false-alarm budget on hazard).

### What holds, and what does not (honest)

- **PI-TGNN leads on ROC-AUC, PR-AUC, and false-alarm rate**, with the best
  consequence-weighted recall (CWDR 0.990) and lowest benign false alarms
  (FAR 0.016 vs 0.036 for RF).
- **Random Forest is a strong competitor** — it edges PI-TGNN on raw
  recall@1%FPR (0.803 vs 0.757) and ties on CWDR, at the cost of 2× the false
  alarms. For these attack types, packet-level features carry a lot of signal.
- **Ablation confirms the design.** Removing edges, the kinematic residual, and
  the consequence head (Standard GNN) drops ROC-AUC 0.993 → 0.945 and raises
  FAR 7× (0.23) — the physics-informed components matter.
- **Consequence estimation is a capability, not just a metric.** Only PI-TGNN
  ranks attacks by physical hazard; `fig1` shows high-hazard spoofs separated
  from benign and low-consequence drift.
- **Limitations (reported):** the consequence head *underestimates* SPaT
  signal-phase severity (RSU nodes carry no kinematic hazard signal; SPaT is
  detected but scored low-severity), and the kinematic residual alone cannot see
  a *constant spatial offset* (see Real-data).

## Zero-shot generalisation to unseen attacks (`fig6`)

Supervised packet-level IDS memorise seen attack signatures. Trained on three
attack classes and tested on the held-out fourth (recall @1% FPR on the unseen
attack):

| Unseen attack | PI-TGNN | Random Forest | Rule-Based |
|---|---:|---:|---:|
| Ghost-braking | **0.99** | 0.00 | 0.21 |
| Sybil phantom | **1.00** | 0.83 | 0.98 |
| Gradual drift | 0.00 | 0.00 | 0.00 |
| SPaT green-spoof | 0.00 | 0.00 | 0.00 |
| **mean** | **0.50** | 0.21 | 0.30 |

**PI-TGNN generalises to unseen *kinematic* attacks** (ghost-braking 0.99 vs. RF
0.00 — RF simply memorised the three it saw), averaging 2.4× RF's unseen-attack
recall. Honest caveat: **no method transfers to unseen drift or SPaT** — both
lack a kinematic footprint, so they must be seen (or need semantic/map features).

## Real-data validation on VeReMi (`fig5`)

The kinematic-consistency residual — the physics core of PI-TGNN — needs only
reported position and speed, so it is evaluated directly on the real public
**VeReMi** benchmark (van der Heijden et al., 2018; VEINS/SUMO/OMNeT++ traces)
without domain-transfer assumptions. One scenario per attacker type:

| VeReMi attack | Residual ROC-AUC |
|---|---:|
| ConstantPos | 0.96 |
| RandomPos | 0.97 |
| RandomPosOffset | 0.96 |
| EventualStop | 0.96 |
| **ConstantPosOffset** | **0.46 (≈ chance)** |
| **mean** | **0.86** |

The residual catches 4 of 5 real attack types at ROC-AUC ~0.96, and **misses the
constant-offset attack** — a constant spatial bias is temporally self-consistent,
so a dead-reckoning check cannot see it; it needs *neighbour/graph* consistency.
This real-data limitation directly motivates the graph structure in PI-TGNN.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The VeReMi real-data check is optional. Download the dataset from
<https://github.com/VeReMi-dataset/VeReMi> (git-LFS; one scenario per attacker
type is ~12 MB extracted) into `data/veremi/<scenario>/`, or set `$VEREMI_PATH`.

## Reproduction

```bash
pytest -q                                   # 9 tests: physics, graph, residual, heads, export
python experiments/train_and_eval.py        # train + benchmark -> results/*.csv (+ console table)
python experiments/zero_shot.py             # unseen-attack generalisation -> results/zero_shot.csv
python visualization/make_pub_figures.py    # figures/ at 300 DPI (svg/pdf/jpg)
python experiments/eval_veremi.py           # real VeReMi residual eval (if dataset present)
```

Deterministic given the seeds in the scripts.

## Figures (`figures/`, 300 DPI, svg/pdf/jpg)

1. `fig1_consequence_vs_detection` — anomaly score vs. physical consequence by
   attack type.
2. `fig2_pr_and_cwdr_curves` — precision-recall and consequence-weighted recall
   vs. false-alarm budget.
3. `fig3_platoon_shockwave_timeline` — platoon speed, hazard, and alert lead time
   under a ghost-braking spoof.
4. `fig4_ablation_and_latency` — CWDR and inference latency across detectors.
5. `fig5_veremi_realdata` — physics core on real VeReMi data (per attack type).
6. `fig6_zero_shot_generalization` — recall on unseen attacks (train on 3, test on 4th).

## Repository structure

```
data_gen/v2x_corridor_generator.py  IDM platoons + 4 J2735 attacks + consequence ground truth
data_gen/graph_builder.py           message table -> temporal interaction graphs
data_gen/veremi_loader.py           parser for the real VeReMi dataset
models/pi_tgnn.py                   PI-TGNN (edge attention + GRU memory + dual heads + phys reg)
models/baselines.py                 rule-based, Random Forest, Isolation Forest
experiments/train_and_eval.py       training + consequence-aware benchmark
experiments/zero_shot.py            generalisation to unseen attack types
experiments/eval_veremi.py          real-data validation on VeReMi
visualization/make_pub_figures.py   six 300-DPI figures
tests/test_pi_tgnn.py               physics, graph, residual, dual-head, training, VeReMi, export
```

## Limitations & threats to validity

- **Simulation for the consequence core.** Consequence labels require a
  simulator; absolute attack rates and the benign/attack mix are assumptions.
  The real-data check validates the physics core, not the attack realism.
- **Strong tabular baseline.** For these attack signatures, Random Forest is
  competitive; PI-TGNN's edge is in false alarms, consequence ranking, and
  (see `zero_shot.py`) unseen-attack generalisation.
- **Signal-phase severity.** SPaT spoofs are detected but their consequence is
  underestimated (no kinematic hazard signal on RSU nodes).
- **Constant-offset attacks** evade a temporal-consistency residual (real VeReMi
  result), motivating neighbour/graph context.

## License

MIT — see [LICENSE](LICENSE).
