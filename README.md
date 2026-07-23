# Greenhouse interior climate — clustering, forecasting, what-if & RL control

A small analysis pipeline for the Osiris greenhouse (Universidad de Murcia) that
predicts **interior temperature and humidity** and explores how the greenhouse's
climatic systems relate to them. It builds on the resampled hourly data already
in this repo and adds four notebooks plus a shared data-assembly module.

> **Scope.** This README covers the *modelling* part only. Raw-data extraction
> from InfluxDB and the resampling that produces `inside data resampled/` live in
> the pre-existing `downloadData.ipynb` and `ResampleClimateDeviceData.ipynb`
> (see `CLAUDE.md`); those are inputs to this pipeline, not part of it.

---

## What it does

| Objective | Where |
|---|---|
| Group hours into recurring **operating states** ("what goes together") | `01.clustering.ipynb` |
| **Forecast** interior temperature *and* humidity at **+1 h / +6 h / +12 h** | `02. prediction models.ipynb` |
| **What-if / sensitivity**: how a climatic system moves T & H, and which settings reach a target | `03. what-if control.ipynb` |
| **Automatic control**: a basic RL (Q-learning) agent that picks a system to run each hour | `04.RL control.ipynb` |

Everything is *multi-objective* (temperature **and** humidity together) and
*multi-horizon*. Predictions target the **centre** sensors (`temp_centro`,
`hum_centro`) as the representative greenhouse climate.

---

## Files

```
greenhouse_dataset.py          Shared module — assembles one tidy hourly table
01.clustering.ipynb            Step 1: KMeans operating states (k = 8)
02. prediction models.ipynb    Step 2: multi-output / multi-horizon forecasting
03. what-if control.ipynb      Step 3: actuator sensitivity + target search
04.RL control.ipynb            Step 4: model-based Q-learning controller (demo)
model data/                    Generated outputs (see below)
├── clusters.csv               Cluster label per hour  (written by notebook 1)
├── cluster_model.joblib       Fitted scaler + KMeans   (written by notebook 1)
└── rl_policy.csv              Learned state->action policy (written by notebook 4)
```

Data consumed (already in the repo):

```
inside data resampled/   interior climate sensors + climatic-system actuators (hourly)
outside data/            exterior weather station (raw, resampled in the module)
forecast data/           Open-Meteo hourly forecast
```

---

## Setup

Python 3.10+. Install the pinned dependencies (the versions the notebooks were
verified against, so the seeded results reproduce exactly):

```bash
pip install -r requirements.txt
```

If your system Python blocks `pip` (PEP 668, "externally-managed
environment"), install into a virtual environment that inherits the system
scientific stack, and register it as a Jupyter kernel:

```bash
python3 -m venv --system-site-packages ~/.venvs/greenhouse
~/.venvs/greenhouse/bin/pip install -r requirements.txt
~/.venvs/greenhouse/bin/python -m ipykernel install --user \
    --name greenhouse --display-name "Python (greenhouse)"
```

Then pick **Python (greenhouse)** as the kernel in Jupyter / VS Code.

(`plotly` is **not** required — these notebooks plot with matplotlib. The
extraction notebooks need extra packages; see the commented section in
`requirements.txt`.)

---

## How to run

Run the notebooks **in order** — notebook 1 writes the cluster labels that
notebooks 2 and 3 consume:

```bash
jupyter lab      # then run 1 → 2 → 3 → 4, "Restart & Run All"
```

Everything is seeded (`random_state=0`), so results reproduce exactly. To run
non-interactively:

```bash
jupyter nbconvert --to notebook --execute --inplace "01.clustering.ipynb"
jupyter nbconvert --to notebook --execute --inplace "02. prediction models.ipynb"
jupyter nbconvert --to notebook --execute --inplace "03. what-if control.ipynb"
jupyter nbconvert --to notebook --execute --inplace "04.RL control.ipynb"
```

Notebook 4 is self-contained (it does not read notebooks 1–3's outputs) but is
compute-heavy: tabular Q-learning runs tens of thousands of one-step simulator
predictions, so a full execute takes ~25–30 min. Notebooks 1–3 run in seconds.

---

## The shared module (`greenhouse_dataset.py`)

One function assembles the whole feature table; the notebooks never re-implement
the merge:

- `build_hourly_dataset()` — merges every interior sensor, every actuator, the
  exterior station (resampled to hourly) and the Open-Meteo forecast (shifted
  −2 h to match local time) onto one regular hourly index. Actuator gaps → `0`
  (device off / not installed); short continuous-sensor gaps → time-interpolated.
- `add_actuator_history(df)` — adds rolling 3 h / 6 h "minutes run" per actuator
  (`act_*_r3`, `act_*_r6`), because thermal/humidity inertia means *recent*
  operation matters, not just the current hour.
- `make_targets(df)` — builds the future targets by shifting `temp_centro` /
  `hum_centro` back by each horizon.
- `ACTIVE_START = "2026-04-24"` — first hour at which *all* climatic systems are
  online; the forecasting/what-if notebooks scope to this so the systems are
  present in both train and test.

Two things it fixes at the source:

- **Sentinel cleaning.** The exterior files use `-999999` as a "no-data" code.
  `outside data/Radiación_solar_global_0.csv` is **100 % sentinel** (a dead
  channel) and is dropped; stray spikes in the other files are masked. (The
  original repo notebooks do not clean these.)
- **No leakage.** Every feature is known at prediction time (current sensors,
  actuators, exterior weather, the *forecast*, the cluster label). Targets are
  strictly future values.

---

## Method & key results

### 1 — Clustering (operating states)

Each hour is described by its interior climate + actuator activity + exterior
weather + time of day, standardised, then **KMeans with k = 8**. `k` was chosen
for *control granularity and interpretability*, not the raw silhouette (which
prefers 2–3 coarse buckets). Each state gets a profile (typical T/H + the
actuator pattern that accompanies it), and a `nearest_state(T, H)` helper maps a
desired climate to the closest state.

### 2 — Forecasting

Persistence baseline vs **Ridge / Random Forest / HistGradientBoosting**, multi-
output over the 6 targets, evaluated with **MAE** and **CVRMSE** on a
**chronological** 80/20 split (never shuffled). An **ablation ladder**
(*no actuators → + current → + history → + cluster*) measures what each block of
climatic-system information adds.

- All learned models beat naïve persistence, and the advantage **grows with the
  horizon** (at +1 h persistence is already strong; at +6 h/+12 h the models pull
  clearly ahead using the daily cycle and the weather forecast).
- Single-target accuracy on the active period (from notebook 3): **temp +1 h MAE
  ≈ 0.7 °C, +6 h ≈ 1.1 °C; humidity +1 h ≈ 2.2 %, +6 h ≈ 4.1 %.**
- The **climatic-system** features are fully included and, once scoped and given
  rolling history, become the most-used engineered features (~11 % of RF
  importance). **But the ablation shows they add little forecasting accuracy** —
  their effect is already visible in the interior sensors (redundant *for
  forecasting*).

### 3 — What-if / sensitivity ⚠️

Treats actuators as knobs: change one, hold everything else fixed, read the
predicted T/H; plus a target-search that ranks settings by how close they land to
a desired climate.

**Important honesty caveat.** This data is **observational** — the systems were
switched on *in response to* the climate (fog, shade screens and the recirculator
correlate **+0.76 to +0.78** with interior temperature). So the model learns
"screen-on ↔ hot", and the sensitivity sweeps come out flat or even wrong-signed.
**The independent causal effect of the actuators is not identifiable from this
data**, and the notebook diagnoses this openly. Use it as an accurate
**forecaster + advisory**, not a calibrated controller.

To obtain trustworthy control effects you would need one of:
1. **Controlled tests** — deliberately vary one setting under similar weather;
2. a **physical / energy-balance model** calibrated to the sensors;
3. **causal-inference** methods (e.g. conditioning on the control rules in
   `inside data/rules/`).

### 4 — Automatic control with Reinforcement Learning ⚠️

Frames regulation as a Markov decision process and solves it with **tabular
Q-learning**. The **environment** is a one-step (+1 h) simulator trained here
(temp MAE ≈ 0.68 °C / CVRMSE 0.033, humidity ≈ 2.1 % / 0.040), stepped forward
while **real exterior weather is replayed** from history. **State** = interior
temperature band × humidity band × daypart (80 discrete states); **actions** =
six interpretable macro-presets (`passive`, `heat`, `ventilate`, `fog_cool`,
`shade+vent`, `recirculate`); **reward** = closeness to a 24 °C / 65 % RH
setpoint. The learned policy is saved to `model data/rl_policy.csv` as a
readable state → action table.

**This is a demonstration of the RL machinery, not a deployable controller —
and the results show exactly why.** The agent converges and produces a policy,
but on unseen simulated days it only matches the *do-nothing* baseline (combined
cost 3.09 vs 3.11 passive, 2.98 always-ventilate) and trails a simple 1-step
MPC reference (2.68); its policy table even contains physically wrong choices
(e.g. `heat` at 26–29 °C). The reason is the **same confounding as notebook 3**:
the agent optimises a world-model built from observational data where the
systems ran *because* it was hot, so it receives weak/wrong-signed gradients.
The fix is identical — replace the simulator with one built from **controlled
tests** or a **calibrated energy-balance model**; the RL code above it does not
change.

---

## Limitations & next steps

- Scoping to the active period (~1,900 hours) gives a summer-only test window —
  less seasonal diversity is the price of a fair test for the actuators.
- Biggest likely accuracy gain not yet done: **lagged / rolling climate**
  features (temperature 1–3 h ago, 3 h means).
- Consider `TimeSeriesSplit` cross-validation + hyper-parameter tuning, and
  modelling the three sensor positions (alta/centro/baja) jointly.
