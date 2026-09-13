"""
campaign_rl.py
==============

Interventional simulator and control environment built on the **August-2026
empty-greenhouse campaign** (``climate devices experiments data/``).

Why this exists alongside ``04.RL control.ipynb``
-------------------------------------------------
Notebook 4 learned a Q-table on a simulator fitted to the *observational*
history and the result was not usable: the policy valued heating at 26 C and
matched the do-nothing baseline. Two separate causes were established on
2026-09-10, and this module fixes both.

1. **The training data were confounded.** Actuators were switched on *because*
   it was hot, so the simulator's actuator gradients were flat or wrong-signed.
   The campaign fixes this at the source: every level was set by a pre-generated
   seeded schedule, and over the campaign the correlation between a commanded
   level and the outside temperature is |r| <= 0.10 for all seven groups
   (against +0.76..+0.78 in the observational data).

2. **The 1 h step was itself fatal.** Phase A puts the interior time constant at
   a median of **16 min** (range 2-39). One hour is ~4 tau: by then the state
   has re-equilibrated and, worse, ``T_t`` already contains the effect of the
   actuator that has been running through the previous hour, so a model
   conditioned on ``T_t`` has nothing left to attribute. Measured on the clean
   campaign data, a +1 h simulator's do-gradients collapse and flip sign
   (VENT_ROOF +0.39 C, i.e. "opening the roof warms the greenhouse"); the same
   data at a **+15 min** step give VENT_ROOF -1.71 C, VENT_SIDE -2.08 C,
   FOG -3.05 C / +12.7 %RH. Predicting the **increment** rather than the next
   level roughly halves the error as well (MAE 0.53 vs 0.73 C).

So: 15 min step, delta target, campaign data. The reward is kept identical to
notebook 4 (``-(|T-24|/2 + |H-65|/10)``) so the two are directly comparable.

What this module deliberately does *not* claim
----------------------------------------------
The greenhouse was **empty**: no transpiration, so humidity and the fog balance
are not transferable to a planted house. The campaign ran in **August only**
(outside 18.5-43 C) and the heaters were repurposed as RECIRC2, so there is no
heating regime in the data at all. Anything learned here is a **summer cooling**
controller for an empty house, and the 45 C interlock is part of the plant, not
something the agent is asked to discover.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

import protocol_check as pc

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

#: Simulator / control step. Chosen as ~tau: Phase A gives a median tau of
#: 16 min, so 15 min resolves the response instead of stepping over it.
STEP = "15min"

#: Campaign window used for learning: day 1 to day 31 of the protocol.
CAMPAIGN = ("2026-08-03", "2026-09-02")

#: The seven protocol groups, in a fixed order — this order *is* the layout of
#: the action part of the feature vector.
GROUPS = ["LIGHTS", "FOG", "VENT_ROOF", "VENT_SIDE", "SHADE", "RECIRC", "RECIRC2"]

#: Full-on level per group, in that group's own units (0-10 intensity, seconds
#: of spray per minute, aperture %, on/off).
FULL = {"LIGHTS": 10.0, "FOG": 20.0, "VENT_ROOF": 100.0, "VENT_SIDE": 100.0,
        "SHADE": 100.0, "RECIRC": 1.0, "RECIRC2": 1.0}

STATE_COLS = ["temp_centro", "hum_centro"]
EXO_COLS = ["ext_temp", "ext_hum", "ext_par", "ext_wind", "hour_sin", "hour_cos"]
#: ``dT_ext`` is derived from the state, so it is rebuilt at every simulated
#: step rather than replayed.
FEATURES = STATE_COLS + ["dT_ext"] + EXO_COLS + GROUPS

#: Setpoint and cost weights, copied from ``04.RL control.ipynb`` so the two
#: notebooks' numbers sit on the same scale.
T_SET, H_SET = 24.0, 65.0
COST_W = (2.0, 10.0)

#: Interior envelope actually observed during the campaign, widened slightly.
#: The simulator is a regression, not physics: without a clamp a long rollout
#: can wander somewhere the greenhouse cannot go.
T_CLIP = (10.0, 60.0)
H_CLIP = (5.0, 100.0)

#: Section 8 of the protocol. Above this the plant overrides the program: both
#: window banks to 100 %, screens and lamps off. The agent does not get a vote.
SAFETY_T = pc.SAFETY["overtemp_c"]          # 45 C


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

def group_levels(data_dir=pc.DATA_DIR, freq: str = STEP) -> pd.DataFrame:
    """Realized level per group on a regular grid, local time.

    A group's level is the mean over its devices, ignoring negative status
    codes. FOG is the decoded duty cycle (seconds spraying per minute), not a
    position, so it comes from :func:`protocol_check.load_fog_duty`.
    """
    act = pc.load_actuators(data_dir)
    out = {}
    for g in GROUPS:
        cols = [c for c in act.columns if c.startswith(g + "::")]
        out[g] = act[cols].where(act[cols] >= 0).mean(axis=1).resample(freq).mean()
    out["FOG"] = pc.load_fog_duty(data_dir).resample(freq).mean()
    return pd.DataFrame(out)


def build_frame(data_dir=pc.DATA_DIR, freq: str = STEP,
                window: tuple[str, str] = CAMPAIGN) -> pd.DataFrame:
    """Interior state + realized actuator levels + exterior weather, one grid.

    This is the interventional analogue of
    ``greenhouse_dataset.build_hourly_dataset()``: same idea, campaign data,
    campaign time base.
    """
    sens = pc.load_sensors(data_dir).resample(freq).mean()
    lev = group_levels(data_dir, freq)
    wx = pc.load_weather(data_dir, freq=freq)
    df = sens.join(lev).join(wx).loc[window[0]:window[1]].copy()
    hour = df.index.hour + df.index.minute / 60.0
    df["hour"] = df.index.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dT_ext"] = df["temp_centro"] - df["ext_temp"]
    return df


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #

@dataclass
class Simulator:
    """One-step (+``STEP``) model of the interior state.

    Predicts the **increment** in temperature and humidity, which is what makes
    the actuator effects visible: a model of the next *level* spends its whole
    capacity on ``T_t`` (r ~ 0.97 with the target) and leaves the actuators
    almost no gradient to carry.
    """

    model_t: HistGradientBoostingRegressor
    model_h: HistGradientBoostingRegressor
    features: list[str] = field(default_factory=lambda: list(FEATURES))
    scores: dict = field(default_factory=dict)
    #: ``"delta"`` (the increment) or ``"level"`` (the next value directly).
    #: Kept configurable only so the ablation can show why ``"level"`` fails.
    target: str = "delta"
    step: str = STEP

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Next ``(temp, hum)`` for a batch of feature rows, already clipped."""
        X = np.asarray(X, dtype=float)
        base = X[:, :2] if self.target == "delta" else 0.0
        t = (base[:, 0] if self.target == "delta" else 0.0) + self.model_t.predict(X)
        h = (base[:, 1] if self.target == "delta" else 0.0) + self.model_h.predict(X)
        return np.column_stack([np.clip(t, *T_CLIP), np.clip(h, *H_CLIP)])

    # -- causal read-out ---------------------------------------------------- #
    def do_effect(self, X: np.ndarray, levels: dict[str, float] | None = None,
                  baseline: dict[str, float] | None = None) -> tuple[float, float]:
        """Mean one-step effect of forcing ``levels`` versus ``baseline``.

        This is a ``do()`` query, not a correlation: the exogenous and interior
        columns are held fixed and only the actuator block is rewritten. On
        observational data it returned nonsense; on campaign data it is the
        number that says whether an RL agent can possibly learn anything.
        """
        idx = {c: i for i, c in enumerate(self.features)}
        base = np.asarray(X, dtype=float).copy()
        for g in GROUPS:
            base[:, idx[g]] = (baseline or {}).get(g, 0.0)
        alt = base.copy()
        for g, v in (levels or {}).items():
            alt[:, idx[g]] = v
        b, a = self.predict(base), self.predict(alt)
        return float(np.mean(a[:, 0] - b[:, 0])), float(np.mean(a[:, 1] - b[:, 1]))


def fit_simulator(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 0,
                  step: str = STEP, target: str = "delta", **kw) -> Simulator:
    """Fit the one-step simulator, scoring it on a chronological hold-out.

    The reported baseline is *persistence* (predict no change), which at a
    15 min step is a genuinely strong reference — beating it by 35 % is a much
    harder test than beating hourly persistence.
    """
    d = df.dropna(subset=FEATURES).copy()
    off = d["temp_centro"] if target == "delta" else 0.0
    offh = d["hum_centro"] if target == "delta" else 0.0
    d["y_t"] = d["temp_centro"].shift(-1) - off
    d["y_h"] = d["hum_centro"].shift(-1) - offh
    # A shift across a gap in the index is not a one-step increment.
    ok = d.index.to_series().shift(-1) - d.index.to_series() == pd.Timedelta(step)
    d = d[ok].dropna(subset=["y_t", "y_h"])

    cut = int(len(d) * (1 - test_frac))
    tr, te = d.iloc[:cut], d.iloc[cut:]
    params = dict(max_iter=400, learning_rate=0.06, random_state=seed)
    params.update(kw)

    scores = {}
    models = {}
    for tag, y in (("temp", "y_t"), ("hum", "y_h")):
        m = HistGradientBoostingRegressor(**params).fit(tr[FEATURES].values, tr[y].values)
        pred = m.predict(te[FEATURES].values)
        naive = np.zeros(len(te)) if target == "delta" else te[
            "temp_centro" if tag == "temp" else "hum_centro"].values
        scores[tag] = dict(
            mae=float(mean_absolute_error(te[y], pred)),
            mae_persistence=float(mean_absolute_error(te[y], naive)),
            n_train=len(tr), n_test=len(te),
            test_from=str(te.index.min()), test_to=str(te.index.max()))
        scores[tag]["gain_pct"] = 100 * (1 - scores[tag]["mae"] / scores[tag]["mae_persistence"])
        # refit on everything for use as an environment
        models[tag] = HistGradientBoostingRegressor(**params).fit(
            d[FEATURES].values, d[y].values)
    return Simulator(model_t=models["temp"], model_h=models["hum"], scores=scores,
                     target=target, step=step)


# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #

def _act(**kw) -> dict[str, float]:
    return {g: FULL[g] for g in kw if kw[g]}


#: Macro-actions. Deliberately close to notebook 4's list so the comparison is
#: like-for-like, minus ``heat`` (no heaters in this campaign — they were wired
#: as RECIRC2) and plus the roof/side split, which the campaign shows behave
#: very differently.
ACTIONS: dict[str, dict[str, float]] = {
    "passive":      {},
    "vent_roof":    _act(VENT_ROOF=1),
    "vent_side":    _act(VENT_SIDE=1),
    "vent_all":     _act(VENT_ROOF=1, VENT_SIDE=1),
    "fog":          _act(FOG=1),
    "fog+vent":     _act(FOG=1, VENT_ROOF=1, VENT_SIDE=1),
    "shade":        _act(SHADE=1),
    "shade+vent":   _act(SHADE=1, VENT_ROOF=1, VENT_SIDE=1),
    "lights(heat)": _act(LIGHTS=1),
    "recirculate":  _act(RECIRC=1, RECIRC2=1),
}
ACTION_NAMES = list(ACTIONS)
N_ACTIONS = len(ACTION_NAMES)

#: ``ACTION_VECS[a]`` is the actuator block of the feature vector for action a.
ACTION_VECS = np.array([[ACTIONS[n].get(g, 0.0) for g in GROUPS] for n in ACTION_NAMES])


def cost(temp: np.ndarray, hum: np.ndarray) -> np.ndarray:
    """Distance to setpoint, notebook-4 scaling. Reward is ``-cost``."""
    return np.abs(temp - T_SET) / COST_W[0] + np.abs(hum - H_SET) / COST_W[1]


# --------------------------------------------------------------------------- #
# Vectorised environment
# --------------------------------------------------------------------------- #

class CampaignEnv:
    """Batch of greenhouse episodes replaying real exterior weather.

    Every episode is one trajectory through the campaign's own weather: the
    exogenous columns come from the historical row at that timestamp, and only
    the interior state and the actuators are simulated. Running a whole batch of
    episodes in lock-step matters — the simulator is a gradient-boosted tree, so
    one batched ``predict`` over 256 episodes costs about what a single-row
    predict costs, and that is the difference between a 30 min training run and
    a 30 s one.

    ``hold`` keeps an action for several consecutive steps. The campaign
    commanded levels in 3 h blocks, so a controller that re-decides every 15 min
    is asking the simulator about switching rates it never saw; holding for
    30 min (the default) stays much closer to the training distribution while
    still being far below the 1 h step that broke notebook 4.
    """

    def __init__(self, sim: Simulator, df: pd.DataFrame, ep_len: int = 96,
                 hold: int = 2, safety: bool = True):
        d = df.dropna(subset=FEATURES).copy()
        self.index = d.index
        self.base = d[FEATURES].values.astype(float)
        self.hours = d.index.hour.values
        self.sim = sim
        self.ep_len = ep_len
        self.hold = max(int(hold), 1)
        self.safety = safety
        self.act_slice = slice(len(FEATURES) - len(GROUPS), len(FEATURES))
        self.gi = {g: FEATURES.index(g) for g in GROUPS}

    # -- episode bookkeeping ------------------------------------------------ #
    def starts(self, test_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
        """Chronological split of episode start rows: train first, then eval."""
        usable = len(self.base) - self.ep_len - 1
        cut = int(usable * (1 - test_frac))
        return np.arange(0, cut), np.arange(cut, usable)

    def reset(self, starts: np.ndarray):
        self.i0 = np.asarray(starts, dtype=int)
        self.t = 0
        rows = self.base[self.i0]
        self.temp = rows[:, 0].copy()
        self.hum = rows[:, 1].copy()
        return self.temp, self.hum

    def rows(self) -> np.ndarray:
        return self.i0 + self.t

    def step(self, actions: np.ndarray):
        """Advance every episode one step under its own action.

        Returns ``(temp, hum, reward, forced)`` where ``forced`` flags the
        episodes whose action was overridden by the 45 C interlock.
        """
        i = self.rows()
        x = self.base[i].copy()
        x[:, 0] = self.temp
        x[:, 1] = self.hum
        x[:, FEATURES.index("dT_ext")] = self.temp - x[:, FEATURES.index("ext_temp")]
        x[:, self.act_slice] = ACTION_VECS[np.asarray(actions, dtype=int)]

        forced = np.zeros(len(i), dtype=bool)
        if self.safety:
            forced = self.temp > SAFETY_T
            if forced.any():
                x[forced, self.gi["VENT_ROOF"]] = FULL["VENT_ROOF"]
                x[forced, self.gi["VENT_SIDE"]] = FULL["VENT_SIDE"]
                x[forced, self.gi["SHADE"]] = 0.0
                x[forced, self.gi["LIGHTS"]] = 0.0

        nxt = self.sim.predict(x)
        self.temp, self.hum = nxt[:, 0], nxt[:, 1]
        self.t += 1
        return self.temp, self.hum, -cost(self.temp, self.hum), forced


# --------------------------------------------------------------------------- #
# State discretisation (tabular agent)
# --------------------------------------------------------------------------- #

#: Bands chosen on the campaign's own envelope (interior 17.6-52.4 C): fine
#: where control decisions actually change, coarse at the extremes.
TEMP_EDGES = np.array([22, 26, 30, 34, 38, 42, 46])
HUM_EDGES = np.array([30, 50, 70, 90])
N_TB, N_HB, N_PB = len(TEMP_EDGES) + 1, len(HUM_EDGES) + 1, 4
N_STATES = N_TB * N_HB * N_PB


def discretise(temp, hum, hour) -> np.ndarray:
    tb = np.digitize(temp, TEMP_EDGES)
    hb = np.digitize(hum, HUM_EDGES)
    pb = np.asarray(hour, dtype=int) // 6
    return (tb * N_HB + hb) * N_PB + pb


# --------------------------------------------------------------------------- #
# Policies
# --------------------------------------------------------------------------- #

def policy_fixed(name: str):
    """Always take one macro-action."""
    a = ACTION_NAMES.index(name)

    def pol(env, temp, hum, hour):
        return np.full(len(temp), a, dtype=int)
    return pol


def policy_thermostat(hi: float = 26.0, lo: float = 22.0):
    """The obvious hand-written baseline: vent when hot, fog when hot and dry.

    Included because "does RL beat a thermostat" is a much more informative
    question than "does RL beat doing nothing".
    """
    a_pass = ACTION_NAMES.index("passive")
    a_vent = ACTION_NAMES.index("vent_all")
    a_fogv = ACTION_NAMES.index("fog+vent")
    a_light = ACTION_NAMES.index("lights(heat)")

    def pol(env, temp, hum, hour):
        a = np.full(len(temp), a_pass, dtype=int)
        a[temp > hi] = a_vent
        a[(temp > hi) & (hum < 60)] = a_fogv
        a[temp < lo] = a_light
        return a
    return pol


def policy_mpc(sim: Simulator):
    """One-step greedy lookahead: try all actions, keep the cheapest next state.

    The reference an RL agent has to justify itself against. It is not free —
    it needs the model online at every decision — but it uses exactly the same
    model the agent was trained on, so any gap is about the *policy*, not the
    world model.
    """
    def pol(env, temp, hum, hour):
        i = env.rows()
        n = len(i)
        x = env.base[i].copy()
        x[:, 0] = temp
        x[:, 1] = hum
        x[:, FEATURES.index("dT_ext")] = temp - x[:, FEATURES.index("ext_temp")]
        big = np.repeat(x, N_ACTIONS, axis=0)
        big[:, env.act_slice] = np.tile(ACTION_VECS, (n, 1))
        nxt = sim.predict(big)
        c = cost(nxt[:, 0], nxt[:, 1]).reshape(n, N_ACTIONS)
        return c.argmin(axis=1)
    return pol


def policy_table(Q: np.ndarray):
    """Greedy w.r.t. a learned Q-table."""
    def pol(env, temp, hum, hour):
        return Q[discretise(temp, hum, hour)].argmax(axis=1)
    return pol


# --------------------------------------------------------------------------- #
# Rollout & training
# --------------------------------------------------------------------------- #

def rollout(env: CampaignEnv, policy, starts: np.ndarray) -> dict:
    """Run a policy over a set of episode starts and summarise the result."""
    env.reset(starts)
    T, H, C, A, F = [], [], [], [], []
    a = None
    for t in range(env.ep_len):
        hour = env.hours[env.rows()]
        if a is None or t % env.hold == 0:
            a = policy(env, env.temp, env.hum, hour)
        temp, hum, r, forced = env.step(a)
        T.append(temp.copy()); H.append(hum.copy()); C.append(-r)
        A.append(a.copy()); F.append(forced.copy())
    T, H, C = np.array(T), np.array(H), np.array(C)
    A, F = np.array(A), np.array(F)
    return dict(temp=T, hum=H, cost=C, actions=A, forced=F,
                mean_cost=float(C.mean()),
                mae_t=float(np.abs(T - T_SET).mean()),
                mae_h=float(np.abs(H - H_SET).mean()),
                pct_forced=float(F.mean() * 100),
                action_mix={ACTION_NAMES[k]: float((A == k).mean() * 100)
                            for k in range(N_ACTIONS) if (A == k).any()})


def _apply_td(Q: np.ndarray, s: np.ndarray, a: np.ndarray, td: np.ndarray,
              alpha: float) -> None:
    """One averaged Q update per distinct ``(s, a)`` seen in the batch.

    Running 256 episodes at once means the same ``(s, a)`` shows up many times
    in a single step, and ``np.add.at`` would then apply the learning rate once
    per occurrence — an effective alpha of 0.15 x 50. That diverges, quietly:
    the Q-table blows up and the greedy policy ends up worse than doing nothing.
    Averaging the TD error over the duplicates keeps the update equivalent to
    one properly-sized step.
    """
    flat = s * N_ACTIONS + a
    sums = np.zeros(Q.size)
    counts = np.zeros(Q.size)
    np.add.at(sums, flat, td)
    np.add.at(counts, flat, 1.0)
    hit = counts > 0
    Q.reshape(-1)[hit] += alpha * sums[hit] / counts[hit]


def train_q(env: CampaignEnv, starts: np.ndarray, episodes: int = 12288,
            batch: int = 256, alpha: float = 0.2, gamma: float = 0.95,
            eps0: float = 1.0, eps1: float = 0.05, seed: int = 0,
            log_every: int = 0) -> tuple[np.ndarray, list[float]]:
    """Tabular Q-learning, episodes run in parallel batches.

    Same learner as notebook 4 — the point of this module is that the *world
    model* changed, not the algorithm, so keeping the algorithm recognisable is
    what makes the comparison mean anything.

    One deliberate difference: because ``env.hold`` keeps an action for several
    simulator steps, the agent is a semi-MDP. The update is made once per
    *decision*, against the discounted reward accumulated over the whole hold
    window and bootstrapped with ``gamma ** hold`` — not once per simulator
    step, which would credit the same choice several times over.
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((N_STATES, N_ACTIONS))
    curve = []
    n_batches = max(int(np.ceil(episodes / batch)), 1)
    for b in range(n_batches):
        eps = eps0 + (eps1 - eps0) * b / max(n_batches - 1, 1)
        env.reset(rng.choice(starts, size=batch, replace=True))
        s = discretise(env.temp, env.hum, env.hours[env.rows()])
        total = np.zeros(batch)
        t = 0
        while t < env.ep_len:
            greedy = Q[s].argmax(axis=1)
            rand = rng.integers(0, N_ACTIONS, size=batch)
            a = np.where(rng.random(batch) < eps, rand, greedy)

            g = np.zeros(batch)
            k = 0
            while k < env.hold and t < env.ep_len:
                temp, hum, r, _ = env.step(a)
                g += (gamma ** k) * r
                total += -r
                k += 1
                t += 1
            s2 = discretise(temp, hum, env.hours[np.minimum(env.rows(), len(env.base) - 1)])
            td = g + (gamma ** k) * Q[s2].max(axis=1) - Q[s, a]
            _apply_td(Q, s, a, td, alpha)
            s = s2
        curve.append(float(total.mean() / env.ep_len))
        if log_every and (b % log_every == 0 or b == n_batches - 1):
            print(f"  batch {b + 1}/{n_batches}  eps={eps:.2f}  mean cost={curve[-1]:.3f}")
    return Q, curve


def policy_grid(Q: np.ndarray, hum: float = 60.0) -> pd.DataFrame:
    """Greedy action per temperature band and daypart, as names.

    The readable form of the Q-table, and the place notebook 4's policy gave
    itself away (it heated at 26-29 C). Read down a column: the action should
    get more aggressive as the band gets hotter, and heating should appear only
    in the coldest band, if at all.
    """
    lo = np.r_[TEMP_EDGES[0] - 4, TEMP_EDGES]
    hi = np.r_[TEMP_EDGES, TEMP_EDGES[-1] + 4]
    mid = (lo + hi) / 2
    parts = ["night 0-6", "morning 6-12", "midday 12-18", "evening 18-24"]
    rows = {}
    for t, m in zip(mid, mid):
        s = discretise(np.full(4, m), np.full(4, hum), np.array([3, 9, 15, 21]))
        rows[f"{t:.0f} degC"] = [ACTION_NAMES[k] for k in Q[s].argmax(axis=1)]
    return pd.DataFrame(rows, index=parts).T


def compare(env: CampaignEnv, sim: Simulator, Q: np.ndarray,
            starts: np.ndarray) -> pd.DataFrame:
    """Evaluate every policy on the same unseen episode starts."""
    policies = {
        "do nothing (passive)": policy_fixed("passive"),
        "always vent_all": policy_fixed("vent_all"),
        "always fog+vent": policy_fixed("fog+vent"),
        "thermostat (rule)": policy_thermostat(),
        "RL (learned Q)": policy_table(Q),
        "MPC-greedy 1-step": policy_mpc(sim),
    }
    rows = {}
    for name, pol in policies.items():
        r = rollout(env, pol, starts)
        rows[name] = dict(**{"|T-24| (degC)": r["mae_t"], "|H-65| (%RH)": r["mae_h"],
                             "combined cost": r["mean_cost"],
                             "% steps interlocked": r["pct_forced"]})
    out = pd.DataFrame(rows).T
    return out.sort_values("combined cost")


# --------------------------------------------------------------------------- #
# Coverage: where the data lived vs where the policy would operate
# --------------------------------------------------------------------------- #

def nearest_action(levels: np.ndarray) -> np.ndarray:
    """Map observed actuator levels onto the closest macro-action.

    The campaign commanded arbitrary level combinations, the agent picks from
    ten macro-actions, and to ask "how much data is there for what the agent
    wants to do" the two have to be spoken in the same vocabulary. Matching is
    on the on/off pattern rather than the magnitudes: an hour with the fog at
    10 s/min and both window banks half open is evidence about ``fog+vent``,
    even though the agent's ``fog+vent`` runs them flat out.
    """
    lv = np.atleast_2d(np.asarray(levels, dtype=float))
    thr = np.array([1, 1, 5, 5, 5, .5, .5])          # "on" per group, its own units
    on = (lv > thr).astype(float)
    va = (ACTION_VECS > 0).astype(float)
    return np.argmin(((on[:, None, :] - va[None, :, :]) ** 2).sum(axis=2), axis=1)


def band_labels(edges: np.ndarray = TEMP_EDGES) -> list[str]:
    return ([f"<{edges[0]:.0f}"]
            + [f"{a:.0f}-{b:.0f}" for a, b in zip(edges[:-1], edges[1:])]
            + [f">{edges[-1]:.0f}"])


def coverage_vs_policy(df: pd.DataFrame, roll: dict) -> pd.DataFrame:
    """Share of time per interior-temperature band: measured vs simulated.

    The two columns are **not** the same kind of number and the caller should
    say so: ``medido_pct`` is the greenhouse's own record, ``simulado_pct`` is
    where the learned policy would take it *according to this simulator* - the
    controller has never run for real. The comparison is a warning flag, not a
    measurement: if the policy behaves as the model predicts, it operates
    exactly where the model is least informed.
    """
    lab = band_labels()
    d = df.dropna(subset=FEATURES)
    med = np.bincount(np.digitize(d.temp_centro.values, TEMP_EDGES), minlength=len(lab))
    sim = np.bincount(np.digitize(np.ravel(roll["temp"]), TEMP_EDGES), minlength=len(lab))
    out = pd.DataFrame({"banda": lab,
                        "medido_pct": 100 * med / med.sum(),
                        "simulado_pct": 100 * sim / sim.sum()}).set_index("banda")
    out.attrs["n_medido"], out.attrs["n_simulado"] = int(med.sum()), int(sim.sum())
    return out


def support_gaps(df: pd.DataFrame, roll: dict, env: CampaignEnv,
                 min_minutes: int = 30) -> pd.DataFrame:
    """Cells the policy uses a lot and the campaign barely visited.

    A cell is (temperature band, daypart, macro-action). ``pasos_agente`` counts
    the policy's simulated steps in it; ``min_observados`` counts the real
    15-min rows whose state and nearest macro-action land in the same cell. The
    rows that come back are the shopping list for the next campaign: not a
    generic design, but the conditions the controller will actually meet.
    """
    lab = band_labels()
    parts = ["noche 0-6", "mañana 6-12", "mediodía 12-18", "tarde 18-24"]
    d = df.dropna(subset=FEATURES)
    obs = pd.DataFrame({"tb": np.digitize(d.temp_centro.values, TEMP_EDGES),
                        "pb": d.index.hour.values // 6,
                        "a": nearest_action(d[GROUPS].values)})
    seen = obs.groupby(["tb", "pb", "a"]).size()

    rows = env.i0 + np.arange(env.ep_len)[:, None]
    hours = env.hours[np.minimum(rows, len(env.base) - 1)]
    want = pd.DataFrame({"tb": np.digitize(np.ravel(roll["temp"]), TEMP_EDGES),
                         "pb": np.ravel(hours) // 6,
                         "a": np.ravel(roll["actions"])})
    g = want.groupby(["tb", "pb", "a"]).size().rename("pasos_agente").reset_index()
    g["min_observados"] = [int(seen.get((r.tb, r.pb, r.a), 0)) for r in g.itertuples()]
    g["pct_agente"] = 100 * g.pasos_agente / g.pasos_agente.sum()
    g["banda"] = [lab[min(i, len(lab) - 1)] for i in g.tb]
    g["franja"] = [parts[i] for i in g.pb]
    g["accion"] = [ACTION_NAMES[i] for i in g.a]
    out = g[g.min_observados < min_minutes].sort_values("pasos_agente", ascending=False)
    out = out[["banda", "franja", "accion", "pasos_agente", "pct_agente", "min_observados"]]
    out.attrs["pct_sin_soporte"] = float(out.pct_agente.sum())
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Open-loop check
# --------------------------------------------------------------------------- #

def replay_check(sim: Simulator, df: pd.DataFrame, ep_len: int = 96,
                 n: int = 40, seed: int = 0) -> pd.DataFrame:
    """Drift of a free-running simulation against what actually happened.

    The agent is only as good as the world it is trained in, and a one-step MAE
    says nothing about a 24 h rollout. Here the *recorded* actuator levels are
    replayed and the simulator runs open-loop from a real starting state, so any
    number below is pure accumulated model error.
    """
    d = df.dropna(subset=FEATURES)
    X = d[FEATURES].values.astype(float)
    rng = np.random.default_rng(seed)
    starts = rng.choice(len(d) - ep_len - 1, size=min(n, len(d) - ep_len - 1),
                        replace=False)
    temp = X[starts, 0].copy()
    hum = X[starts, 1].copy()
    rows = []
    for t in range(ep_len):
        i = starts + t
        x = X[i].copy()
        x[:, 0], x[:, 1] = temp, hum
        x[:, FEATURES.index("dT_ext")] = temp - x[:, FEATURES.index("ext_temp")]
        nxt = sim.predict(x)
        temp, hum = nxt[:, 0], nxt[:, 1]
        truth = X[i + 1]
        rows.append(dict(step=t + 1, minutes=(t + 1) * 15,
                         mae_temp=float(np.abs(temp - truth[:, 0]).mean()),
                         mae_hum=float(np.abs(hum - truth[:, 1]).mean())))
    return pd.DataFrame(rows)
