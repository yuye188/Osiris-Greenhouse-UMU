"""Shared forecasting utilities for the rolling-origin evaluation (notebook 02b).

The functions reproduce the protocol of notebook 02 (feature sets, 24 h common anchor set,
two fitting stages per split, validation-based selection with the mean CVRMSE criterion,
tabular grids, KerasTuner search for the neural networks, persistence baselines and the
Diebold-Mariano / block-bootstrap statistics), but every split-dependent step takes the
split as an argument, so that the same code can run on several chronological folds.

Notebook 02 keeps its own copy of this code for now; its cached results are not affected.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from greenhouse_dataset import (
    ACTIVE_START, ACTUATOR_COLS, CONTINUOUS_COLS, EXTERIOR_COLS, TARGET_COLS,
    add_actuator_history, build_hourly_dataset, make_forecast_horizon_features, make_targets,
)

HORIZONS = (1, 6, 12)
MAX_LOOKBACK = 24
TABULAR = ["Ridge", "RandomForest", "HistGB"]
NEURAL_ARCHS = ["LSTM", "GRU", "Attention", "MLP"]
MODEL_ORDER = ["Ridge", "RandomForest", "HistGB",
               "LSTM (t only)", "GRU (t only)", "Attention (t only)", "MLP (t only)",
               "Ridge (hist)", "RandomForest (hist)", "HistGB (hist)",
               "LSTM", "GRU", "Attention", "MLP"]
BASELINES = ("persistence", "seasonal persistence")
TEMP_TARGET_COLS = [c for c in TARGET_COLS if c.startswith("temp_")]
HUM_TARGET_COLS = [c for c in TARGET_COLS if c.startswith("hum_")]


@dataclass
class Budget:
    """Training budget. The defaults are the reduced rolling-origin budget."""
    max_trials: int = 10          # KerasTuner BayesianOptimization trials per (model, feature set, fold)
    max_epochs: int = 100
    patience: int = 8
    seeds: tuple = (0, 1, 2)      # final refits per neural model
    rf_trees: int = 200
    hgb_iter: int = 300
    lookback_grid: tuple = (3, 6, 12, 24)
    kt_dir: str = "kt_tuning_ro"


# ---------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------
TIME_FEATS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "hour", "month"]
CLUSTER_TIME_FEATS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]


def load_data(root: str = ".", k_clusters: int = 8) -> dict:
    """Build the modelling table exactly as notebook 02 (sections 2.1 to 2.4), without cluster
    labels. Clusters are fitted per fold and stage with `cluster_frame`."""
    df_full = build_hourly_dataset(root=root)
    hist_cols = add_actuator_history(df_full, windows=(3, 6))
    df = df_full.loc[ACTIVE_START:].copy()
    fc_hist_cols = make_forecast_horizon_features(df, horizons=HORIZONS)
    oracle_cols = make_forecast_horizon_features(df, horizons=HORIZONS, source_cols=EXTERIOR_COLS)
    targets = make_targets(df, temp_col="temp_centro", hum_col="hum_centro", horizons=HORIZONS)
    climate = CONTINUOUS_COLS + EXTERIOR_COLS + TIME_FEATS
    cluster_cols = [f"clu_{k}" for k in range(k_clusters)]
    feature_sets = {
        "no actuators": climate,
        "+ actuators (current)": climate + ACTUATOR_COLS,
        "+ actuator history": climate + ACTUATOR_COLS + hist_cols,
        "+ cluster": climate + ACTUATOR_COLS + hist_cols + cluster_cols,
        "+ day-ahead forecast": climate + ACTUATOR_COLS + hist_cols + cluster_cols + fc_hist_cols,
    }
    base_cols = climate + ACTUATOR_COLS + hist_cols + fc_hist_cols + oracle_cols
    data = df[base_cols].join(targets).dropna(subset=base_cols + TARGET_COLS)
    # common anchor set: prediction times whose 24 h history is complete and consecutive
    ts, pos = data.index.values, np.arange(len(data))
    valid = np.zeros(len(data), dtype=bool)
    ok = pos >= MAX_LOOKBACK - 1
    valid[ok] = (ts[pos[ok]] - ts[pos[ok] - MAX_LOOKBACK + 1]) == np.timedelta64(MAX_LOOKBACK - 1, "h")
    # clustering input: same features and rows as notebook 01 (whole record, not only the active period)
    cluster_features = CONTINUOUS_COLS + ACTUATOR_COLS + EXTERIOR_COLS + CLUSTER_TIME_FEATS
    return dict(df=df, data=data, anchors=data.index[valid], feature_sets=feature_sets,
                cluster_cols=cluster_cols, k_clusters=k_clusters,
                cluster_input=df_full[cluster_features].dropna())


def cluster_frame(D: dict, fit_end) -> pd.DataFrame:
    """One-hot cluster labels for every modelling row, with the scaler and K-means fitted
    only on hours before `fit_end` (as in notebook 01)."""
    C = D["cluster_input"]
    fit = C[C.index < fit_end]
    sc = StandardScaler().fit(fit)
    km = KMeans(n_clusters=D["k_clusters"], n_init=10, random_state=0).fit(sc.transform(fit))
    lab = pd.Series(km.predict(sc.transform(C)), index=C.index)
    dummies = pd.get_dummies(pd.Categorical(lab, categories=range(D["k_clusters"])), prefix="clu").astype(float)
    dummies.index = lab.index
    return dummies.reindex(D["data"].index).fillna(0.0)


# ---------------------------------------------------------------------------------------
# folds
# ---------------------------------------------------------------------------------------
@dataclass
class Fold:
    name: str
    inner_train_end: pd.Timestamp
    val_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp            # exclusive
    idx: dict = field(default_factory=dict)


def make_folds(D: dict, test_starts, val_starts=None, data_end=None, val_days: int = 7,
               embargo_hours: int = 12) -> list:
    """Expanding-window folds. Fold k tests on [test_starts[k], test_starts[k+1]) (the last one
    up to the end of the data). Its validation period starts at val_starts[k] if given, otherwise
    `val_days` days before train_end = test_start - embargo, and ends at train_end. The inner
    training period ends one embargo before the validation period."""
    emb, anchors = pd.Timedelta(hours=embargo_hours), D["anchors"]
    data_end = data_end or anchors.max() + pd.Timedelta(hours=1)
    val_starts = val_starts or [None] * len(test_starts)
    folds = []
    for k, ts in enumerate(test_starts):
        ts = pd.Timestamp(ts)
        te = pd.Timestamp(test_starts[k + 1]) if k + 1 < len(test_starts) else data_end
        train_end = ts - emb
        val_start = pd.Timestamp(val_starts[k]) if val_starts[k] is not None else train_end - pd.Timedelta(days=val_days)
        f = Fold(f"fold {k + 1}", val_start - emb, val_start, train_end, ts, te)
        f.idx = dict(inner=anchors[anchors < f.inner_train_end],
                     val=anchors[(anchors >= f.val_start) & (anchors < f.train_end)],
                     train=anchors[anchors < f.train_end],
                     test=anchors[(anchors >= f.test_start) & (anchors < f.test_end)])
        h = pd.Timedelta(hours=max(HORIZONS))
        assert (f.idx["inner"] + h < f.val_start).all() and (f.idx["train"] + h < f.test_start).all()
        folds.append(f)
    return folds


def folds_table(folds) -> pd.DataFrame:
    return pd.DataFrame({f.name: {
        "inner train end": f.inner_train_end, "validation": f"{f.val_start} -> {f.train_end}",
        "test": f"{f.test_start} -> {f.test_end}",
        "n inner train": len(f.idx["inner"]), "n validation": len(f.idx["val"]),
        "n outer train": len(f.idx["train"]), "n test": len(f.idx["test"]),
        "test days": round(len(f.idx["test"]) / 24, 1)} for f in folds}).T


# ---------------------------------------------------------------------------------------
# metrics and baselines
# ---------------------------------------------------------------------------------------
def cvrmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred) / np.mean(y_true)


def score(y_true_df, y_pred_df):
    y_pred_df = y_pred_df.loc[y_true_df.index]
    return pd.DataFrame({
        "MAE": {c: mean_absolute_error(y_true_df[c], y_pred_df[c]) for c in TARGET_COLS},
        "CVRMSE": {c: cvrmse(y_true_df[c], y_pred_df[c]) for c in TARGET_COLS},
        "R2": {c: r2_score(y_true_df[c], y_pred_df[c]) for c in TARGET_COLS},
    })


def mean_cvrmse(y_true_df, y_pred_df):
    """Selection criterion: mean CVRMSE over the six targets, physical units."""
    return float(score(y_true_df, y_pred_df)["CVRMSE"].mean())


def baseline_frame(df, index, kind):
    out = pd.DataFrame(index=index)
    for h in HORIZONS:
        for v, col in (("temp", "temp_centro"), ("hum", "hum_centro")):
            src = df[col] if kind == "persistence" else df[col].shift(24 - h)
            out[f"{v}_h{h}"] = src.reindex(index).values
    return out[TARGET_COLS]


# ---------------------------------------------------------------------------------------
# one fold: stages, windows, models
# ---------------------------------------------------------------------------------------
def _windows(arr, positions, lookback):
    offs = np.arange(-lookback + 1, 1)
    return arr[positions[:, None] + offs[None, :]]


def last(X, L):
    return X[:, -L:, :]


def flat(X, L):
    return X[:, -L:, :].reshape(len(X), -1)


class FoldRunner:
    """Selection stage (fit on inner train, evaluate on validation) and final stage (refit on
    outer train, evaluate once on test) for one fold, as in notebook 02."""

    def __init__(self, D: dict, fold: Fold, budget: Budget):
        self.D, self.fold, self.budget = D, fold, budget
        data = D["data"]
        self.frames = {"sel": data.join(cluster_frame(D, fold.inner_train_end)),
                       "final": data.join(cluster_frame(D, fold.train_end))}
        self.stage_def = {
            "sel": dict(fit_end=fold.inner_train_end, fit_idx=fold.idx["inner"], eval_idx=fold.idx["val"]),
            "final": dict(fit_end=fold.train_end, fit_idx=fold.idx["train"], eval_idx=fold.idx["test"]),
        }
        self.y_val = data.loc[fold.idx["val"], TARGET_COLS]
        self.y_test = data.loc[fold.idx["test"], TARGET_COLS]
        self._prep = {}

    # -- inputs ---------------------------------------------------------------------------
    def prepare(self, cols, stage):
        key = (tuple(cols), stage)
        if key in self._prep:
            return self._prep[key]
        fr, S = self.frames[stage], self.stage_def[stage]
        X = fr[cols].to_numpy("float64")
        xs = StandardScaler().fit(fr.loc[fr.index < S["fit_end"], cols].to_numpy("float64"))
        Xs = xs.transform(X).astype("float32")
        ys = StandardScaler().fit(fr.loc[S["fit_idx"], TARGET_COLS].to_numpy("float64"))
        p_fit, p_ev = fr.index.get_indexer(S["fit_idx"]), fr.index.get_indexer(S["eval_idx"])
        out = dict(Xtr=_windows(Xs, p_fit, MAX_LOOKBACK),
                   ytr=ys.transform(fr.loc[S["fit_idx"], TARGET_COLS].to_numpy("float64")).astype("float32"),
                   Xev=_windows(Xs, p_ev, MAX_LOOKBACK),
                   yev=fr.loc[S["eval_idx"], TARGET_COLS],
                   yev_s=ys.transform(fr.loc[S["eval_idx"], TARGET_COLS].to_numpy("float64")).astype("float32"),
                   y_scaler=ys, ev_index=S["eval_idx"])
        self._prep[key] = out
        return out

    @staticmethod
    def to_frame(P, pred_scaled):
        return pd.DataFrame(P["y_scaler"].inverse_transform(pred_scaled), index=P["ev_index"], columns=TARGET_COLS)

    # -- tabular --------------------------------------------------------------------------
    def search_space(self):
        b = self.budget
        return {
            "Ridge": (lambda p: make_pipeline(StandardScaler(), Ridge(alpha=p["alpha"])),
                      [{"alpha": a} for a in (0.1, 1.0, 10.0, 100.0, 1000.0)]),
            "RandomForest": (lambda p: RandomForestRegressor(n_estimators=b.rf_trees, min_samples_leaf=p["min_samples_leaf"],
                                                             n_jobs=-1, random_state=0),
                             [{"min_samples_leaf": v} for v in (1, 2, 5)]),
            "HistGB": (lambda p: MultiOutputRegressor(HistGradientBoostingRegressor(
                max_iter=b.hgb_iter, learning_rate=p["learning_rate"], random_state=0)),
                [{"learning_rate": v} for v in (0.03, 0.05, 0.1)]),
        }

    def run_tabular(self, arch, cols, lookbacks):
        ctor, grid = self.search_space()[arch]
        S = self.prepare(cols, "sel")
        best = (np.inf, None, None, None)
        for L in lookbacks:
            Xtr, Xev = flat(S["Xtr"], L), flat(S["Xev"], L)
            for p in grid:
                vp = self.to_frame(S, ctor(p).fit(Xtr, S["ytr"]).predict(Xev))
                crit = mean_cvrmse(S["yev"], vp)
                if crit < best[0]:
                    best = (crit, p, L, vp)
        crit, p, L, vp = best
        F = self.prepare(cols, "final")
        m = ctor(p).fit(flat(F["Xtr"], L), F["ytr"])
        return dict(val_pred=vp, val_crit=crit, test_preds=[self.to_frame(F, m.predict(flat(F["Xev"], L)))],
                    params={**p, "lookback": L})

    # -- neural ---------------------------------------------------------------------------
    def run_neural(self, arch, cols, lookbacks, label):
        import keras_tuner as kt
        from tensorflow import keras
        b, S = self.budget, self.prepare(cols, "sel")
        make_fn = NEURAL_BUILDERS[arch]
        to_frame = self.to_frame

        class HM(kt.HyperModel):
            def __init__(self):
                super().__init__()
                self.log = []

            def build(self, hp):
                L = hp.Choice("lookback", list(lookbacks)) if len(lookbacks) > 1 else lookbacks[0]
                return make_fn(n_features=S["Xtr"].shape[-1], n_targets=len(TARGET_COLS), lookback=L,
                               units=hp.Int("units", 16, 128, step=16), dropout=hp.Float("dropout", 0.0, 0.4),
                               lr=hp.Float("lr", 1e-4, 1e-2, sampling="log"),
                               optimizer=hp.Choice("optimizer", ["adam", "rmsprop"]))

            def fit(self, hp, model, *args, **kwargs):
                L = hp.get("lookback") if len(lookbacks) > 1 else lookbacks[0]
                es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=b.patience, restore_best_weights=True)
                h = model.fit(last(S["Xtr"], L), S["ytr"], validation_data=(last(S["Xev"], L), S["yev_s"]),
                              epochs=b.max_epochs, batch_size=32, callbacks=[es], verbose=0)
                vp = to_frame(S, model.predict(last(S["Xev"], L), verbose=0))
                crit = mean_cvrmse(S["yev"], vp)
                self.log.append(dict(hp=dict(hp.values), epochs=int(np.argmin(h.history["val_loss"])) + 1,
                                     val_pred=vp, crit=crit))
                return {"val_cvrmse": crit}

        hm = HM()
        tuner = kt.BayesianOptimization(hypermodel=hm, objective=kt.Objective("val_cvrmse", direction="min"),
                                        max_trials=b.max_trials, seed=0, directory=b.kt_dir,
                                        project_name="".join(c if c.isalnum() else "_" for c in label),
                                        overwrite=True)
        tuner.search(verbose=0)
        assert hm.log, f"{label}: no KerasTuner trial completed"
        best = min(hm.log, key=lambda e: e["crit"])
        params = {k: best["hp"][k] for k in ("units", "dropout", "lr", "optimizer")}
        params["lookback"] = best["hp"].get("lookback", lookbacks[0])
        params["epochs"] = best["epochs"]
        F, L, preds = self.prepare(cols, "final"), params["lookback"], []
        for seed in b.seeds:
            m = make_fn(n_features=F["Xtr"].shape[-1], n_targets=len(TARGET_COLS), lookback=L, units=params["units"],
                        dropout=params["dropout"], lr=params["lr"], optimizer=params["optimizer"], seed=seed)
            m.fit(last(F["Xtr"], L), F["ytr"], epochs=params["epochs"], batch_size=32, verbose=0)
            preds.append(self.to_frame(F, m.predict(last(F["Xev"], L), verbose=0)))
        return dict(val_pred=best["val_pred"], val_crit=best["crit"], test_preds=preds, params=params)

    # -- everything for one fold ----------------------------------------------------------
    def run(self, feature_set_names, models=MODEL_ORDER, log=print):
        res = {"pred": {}, "val_pred": {}, "val_crit": {}, "params": {}}
        for fs in feature_set_names:
            cols = self.D["feature_sets"][fs]
            for name in models:
                arch, hist = name.split(" ")[0], not name.endswith("(t only)") and (
                    name in NEURAL_ARCHS or name.endswith("(hist)"))
                lookbacks = self.budget.lookback_grid if hist else (1,)
                if arch in TABULAR:
                    out = self.run_tabular(arch, cols, lookbacks)
                else:
                    out = self.run_neural(arch, cols, lookbacks, label=f"{self.fold.name}_{name}_{fs}")
                key = (name, fs)
                res["pred"][key], res["val_pred"][key] = out["test_preds"], out["val_pred"]
                res["val_crit"][key], res["params"][key] = out["val_crit"], out["params"]
                log(f"{self.fold.name} | {name:>20} | {fs:<22} | val mean CVRMSE {out['val_crit']:.4f} | {out['params']}")
        df = self.D["df"]
        res["baselines_test"] = {k: baseline_frame(df, self.fold.idx["test"], k) for k in BASELINES}
        res["baselines_val"] = {k: baseline_frame(df, self.fold.idx["val"], k) for k in BASELINES}
        val_mae = {k: score(self.y_val, b)["MAE"] for k, b in res["baselines_val"].items()}
        res["ref_choice"] = pd.Series({c: min(val_mae, key=lambda k: val_mae[k][c]) for c in TARGET_COLS})
        res["y_test"], res["y_val"] = self.y_test, self.y_val
        return res


# ---------------------------------------------------------------------------------------
# neural architectures (same as notebook 02, sections 2.7e to 2.7g)
# ---------------------------------------------------------------------------------------
def _opt(name, lr):
    from tensorflow import keras
    return {"adam": keras.optimizers.Adam, "rmsprop": keras.optimizers.RMSprop}[name](lr)


def make_lstm(n_features, n_targets, lookback=MAX_LOOKBACK, units=64, dropout=0.0, lr=1e-3, optimizer="adam", seed=0):
    from tensorflow import keras
    from tensorflow.keras import layers
    keras.utils.set_random_seed(seed)
    model = keras.Sequential([layers.Input(shape=(lookback, n_features)), layers.LSTM(units, dropout=dropout),
                              layers.Dense(32, activation="tanh"), layers.Dense(n_targets)])
    model.compile(optimizer=_opt(optimizer, lr), loss="mse")
    return model


def make_gru(n_features, n_targets, lookback=MAX_LOOKBACK, units=64, dropout=0.0, lr=1e-3, optimizer="adam", seed=0):
    from tensorflow import keras
    from tensorflow.keras import layers
    keras.utils.set_random_seed(seed)
    model = keras.Sequential([layers.Input(shape=(lookback, n_features)), layers.GRU(units, dropout=dropout),
                              layers.Dense(32, activation="tanh"), layers.Dense(n_targets)])
    model.compile(optimizer=_opt(optimizer, lr), loss="mse")
    return model


def make_attention(n_features, n_targets, lookback=MAX_LOOKBACK, units=64, dropout=0.0, lr=1e-3, optimizer="adam", seed=0):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    keras.utils.set_random_seed(seed)
    inputs = layers.Input(shape=(lookback, n_features))
    x = layers.Dense(units)(inputs)
    x = x + layers.Embedding(input_dim=lookback, output_dim=units)(tf.range(start=0, limit=lookback, delta=1))
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=max(units // 4, 8), dropout=dropout)(x, x)
    x = layers.LayerNormalization()(x + attn)
    ff = layers.Dense(units, activation="relu")(x)
    ff = layers.Dropout(dropout)(ff)
    ff = layers.Dense(units)(ff)
    x = layers.LayerNormalization()(x + ff)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="tanh")(x)
    model = keras.Model(inputs, layers.Dense(n_targets)(x))
    model.compile(optimizer=_opt(optimizer, lr), loss="mse")
    return model


def make_mlp(n_features, n_targets, lookback=MAX_LOOKBACK, units=64, dropout=0.0, lr=1e-3, optimizer="adam", seed=0):
    from tensorflow import keras
    from tensorflow.keras import layers
    keras.utils.set_random_seed(seed)
    model = keras.Sequential([layers.Input(shape=(lookback, n_features)), layers.Flatten(),
                              layers.Dense(units, activation="relu"), layers.Dropout(dropout),
                              layers.Dense(units, activation="relu"), layers.Dense(n_targets)])
    model.compile(optimizer=_opt(optimizer, lr), loss="mse")
    return model


NEURAL_BUILDERS = {"LSTM": make_lstm, "GRU": make_gru, "Attention": make_attention, "MLP": make_mlp}


# ---------------------------------------------------------------------------------------
# statistics (same as notebook 02, section 2.8d)
# ---------------------------------------------------------------------------------------
def horizon_of(col):
    return int(col.rsplit("_h", 1)[1])


def as_list(p):
    return p if isinstance(p, list) else [p]


def abs_err(preds, y):
    """Hourly absolute error per target; for several seeds, the mean absolute error over seeds."""
    preds = as_list(preds)
    return sum((p.loc[y.index, TARGET_COLS] - y[TARGET_COLS]).abs() for p in preds) / len(preds)


def lag1_autocorr(d):
    d = np.asarray(d, dtype=float) - np.mean(d)
    den = d @ d
    return float(d[1:] @ d[:-1] / den) if den > 0 else np.nan


def long_run_var(d, L):
    """Newey-West (Bartlett-kernel) long-run variance with L lags."""
    d = np.asarray(d, dtype=float) - np.mean(d)
    n = len(d)
    v = d @ d / n
    for k in range(1, min(L, n - 1) + 1):
        v += 2 * (1 - k / (L + 1)) * (d[k:] @ d[:-k]) / n
    return v


def dm_test(d, h, L=24):
    """Diebold-Mariano test on d = |e_A| - |e_B| with HAC variance (L lags) and the
    Harvey-Leybourne-Newbold correction. Returns (statistic, two-sided p-value)."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    lrv = long_run_var(d, L)
    if not np.isfinite(lrv) or lrv <= 0:
        return np.nan, np.nan
    stat = d.mean() / np.sqrt(lrv / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return stat, 2 * stats.t.sf(abs(stat), df=n - 1)


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.where(np.isfinite(p))[0]
    if len(ok) == 0:
        return out
    order = ok[np.argsort(p[ok])]
    m = len(order)
    out[order] = np.minimum(np.maximum.accumulate((m - np.arange(m)) * p[order]), 1.0)
    return out


def block_bootstrap_ci(d, block=24, n_boot=2000, seed=0):
    """95 % CI of the mean of d with a circular block bootstrap."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, int(np.ceil(n / block))))
    idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(n_boot, -1)[:, :n]
    bm = d[idx].mean(axis=1)
    return np.quantile(bm, 0.025), np.quantile(bm, 0.975)


def vs_baselines(losses, h, models, baselines=BASELINES, lags=(12, 24, 48), alpha=0.05):
    """Every model against both baselines on one loss series (hours x methods): mean difference
    (model - baseline), DM p-values for each HAC bandwidth, Holm over models x baselines for
    each bandwidth, and a verdict that compares the bandwidths."""
    rows = []
    for m in models:
        for b in baselines:
            d = (losses[m] - losses[b]).values
            row = {"model": m, "baseline": b, "diff": d.mean(), "rel diff (%)": 100 * d.mean() / losses[b].mean()}
            for L in lags:
                row[f"p (L={L})"] = dm_test(d, h, L=L)[1]
            rows.append(row)
    t = pd.DataFrame(rows)
    for L in lags:
        t[f"p Holm (L={L})"] = holm(t[f"p (L={L})"].values)
    sig = np.column_stack([t[f"p Holm (L={L})"] < alpha for L in lags])
    better, worse = t["diff"] < 0, t["diff"] > 0
    t["verdict"] = np.select([sig.all(1) & better, sig.all(1) & worse, sig.any(1) & better, sig.any(1) & worse],
                             ["better (all L)", "worse (all L)", "better (some L)", "worse (some L)"], default="n.s.")
    return t


def paired_rows(err_a, err_b, h_of=horizon_of, L=24, n_boot=2000):
    """Per target: MAE of A and B, difference, bootstrap CI, DM p-value; Holm over the targets."""
    rows = {}
    for c in TARGET_COLS:
        d = (err_a[c] - err_b[c]).values
        lo, hi = block_bootstrap_ci(d, n_boot=n_boot)
        rows[c] = {"MAE A": err_a[c].mean(), "MAE B": err_b[c].mean(), "dMAE (A-B)": d.mean(),
                   "rel (%)": 100 * d.mean() / err_b[c].mean(), "CI 2.5%": lo, "CI 97.5%": hi,
                   "p DM": dm_test(d, h_of(c), L=L)[1]}
    t = pd.DataFrame(rows).T
    t["p Holm"] = holm(t["p DM"].astype(float).values)
    return t


# ---------------------------------------------------------------------------------------
# pairwise Diebold-Mariano comparison of many methods, drawn as a CD-style diagram
# (same as notebook 02, section 2.17)
# ---------------------------------------------------------------------------------------
INK, INK2 = "#0b0b0b", "#52514e"
BLUE, RED = "#2a78d6", "#e34948"
BASELINE_TAGS = (("persistence", "P"), ("seasonal persistence", "S"))
DMD_KEY = ("Bars join methods consecutive on the axis that are not distinguishable; the absence of a bar does NOT imply "
           "a significant difference.\nP / S: test against persistence / seasonal persistence "
           "(● significantly better, ○ not distinguishable, ▼ significantly worse)")


def dm_pairwise(losses, h, L=24, alpha=0.05):
    """losses: DataFrame hours x methods (lower is better). DM test for every pair of methods
    (HAC with L lags, HLN correction), Holm over all pairs."""
    from itertools import combinations
    rows = []
    for a, b in combinations(losses.columns, 2):
        d = (losses[a] - losses[b]).values
        stat, p = dm_test(d, h, L=L)
        rows.append({"A": a, "B": b, "mean diff (A-B)": d.mean(), "DM": stat, "p DM": p})
    pw = pd.DataFrame(rows)
    pw["p Holm"] = holm(pw["p DM"].values)
    pw["significant"] = pw["p Holm"] < alpha
    return pw


def dm_groups(order, pw):
    """Maximal runs of consecutive methods (in score order) with no significant pair inside."""
    sig = {frozenset((a, b)) for a, b, s in zip(pw.A, pw.B, pw.significant) if s}
    runs = []
    for i in range(len(order)):
        j = i
        while j + 1 < len(order) and all(frozenset((order[a], order[j + 1])) not in sig for a in range(i, j + 1)):
            j += 1
        if j > i and not any(s <= i and j <= e for s, e in runs):
            runs.append((i, j))
    return runs


def baseline_status(pw, base):
    """+1 significantly better than `base`, -1 significantly worse, 0 not distinguishable."""
    st = {}
    for a, b, d, s in zip(pw.A, pw.B, pw["mean diff (A-B)"], pw.significant):
        if base in (a, b):
            other = b if a == base else a
            diff = d if a == other else -d
            st[other] = 0 if not s else (1 if diff < 0 else -1)
    return st


def dm_matrix(pw, order):
    """Square table of Holm-adjusted p-values."""
    m = pd.DataFrame(np.nan, index=order, columns=order)
    for a, b, p in zip(pw["A"], pw["B"], pw["p Holm"]):
        m.loc[a, b] = m.loc[b, a] = p
    return m


def draw_dm_diagram(score, pw, title, ax, xlabel):
    """CD-style diagram on a value axis: methods placed at their score (relative MAE), thick bars
    join rank-consecutive methods of which no pair differs (DM, Holm). Next to each label, two
    markers give the test against persistence (P) and seasonal persistence (S)."""
    from matplotlib.ticker import MaxNLocator
    score = score.sort_values()
    names, k = list(score.index), len(score)
    groups = dm_groups(names, pw)
    status = {tag: baseline_status(pw, base) for base, tag in BASELINE_TAGS}
    v = score.values
    cap = v[-2] + 0.10 * (v[-2] - v[0]) if k > 2 and v[-1] - v[-2] > 0.6 * (v[-2] - v[0]) else None
    pos = score.clip(upper=cap) if cap is not None else score
    lo, hi = pos.min(), pos.max()
    pad = 0.05 * (hi - lo)
    x0, x1 = lo - pad, hi + pad
    span = x1 - x0
    g = 0.035 * span
    half = int(np.ceil(k / 2))
    bar_step, row_step, y_axis = 0.22, 0.42, 0.0
    y_bars = [y_axis - 0.25 - bar_step * n for n in range(len(groups))]
    y_first = (y_bars[-1] if groups else y_axis) - 0.55
    ax.set_xlim(x0 - 0.55 * span, x1 + 0.55 * span)
    ax.set_ylim(y_first - row_step * (half - 1) - 0.35, 0.75)
    ax.axis("off")
    ax.hlines(y_axis, x0, x1, color=INK, lw=1)
    for t in MaxNLocator(nbins=6).tick_values(lo, hi if cap is None else v[-2]):
        if x0 <= t <= (x1 if cap is None else v[-2]):
            ax.vlines(t, y_axis, y_axis + 0.08, color=INK, lw=1)
            ax.text(t, y_axis + 0.13, f"{t:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    if cap is not None:
        ax.text(cap, y_axis + 0.13, "off scale ▸", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.text((x0 + x1) / 2, y_axis + 0.42, xlabel, ha="center", va="bottom", fontsize=8, color=INK2)
    xl, xr = x0 - 0.03 * span, x1 + 0.03 * span
    ntag = len(BASELINE_TAGS)

    def col(sgn, n):
        return n + 1 if sgn > 0 else ntag - n          # same P, S order on both sides

    for xe, sgn in ((xl, -1), (xr, 1)):
        for n, (_, tag) in enumerate(BASELINE_TAGS):
            ax.text(xe + sgn * g * col(sgn, n), y_first + 0.30, tag, ha="center", va="center",
                    fontsize=7.5, color=INK2, fontweight="bold")
    for idx, name in enumerate(names):
        left = idx < half
        y = y_first - row_step * (idx if left else k - 1 - idx)
        xe, sgn = (xl, -1) if left else (xr, 1)
        ax.plot([pos[name], pos[name], xe], [y_axis, y, y], color=INK, lw=0.9,
                ls=":" if cap is not None and score[name] > cap else "-")
        for n, (base, tag) in enumerate(BASELINE_TAGS):
            xm = xe + sgn * g * col(sgn, n)
            if name == base:
                ax.text(xm, y, "–", ha="center", va="center", fontsize=8, color=INK2)
                continue
            s = status[tag].get(name, 0)
            if s == 1:
                ax.plot(xm, y, "o", ms=5.5, mfc=BLUE, mec=BLUE)
            elif s == -1:
                ax.plot(xm, y, "v", ms=5.5, mfc=RED, mec=RED)
            else:
                ax.plot(xm, y, "o", ms=5.5, mfc="white", mec=INK2, mew=0.9)
        ax.text(xe + sgn * g * (ntag + 0.8), y, f"{name} ({score[name]:.2f})",
                ha="right" if left else "left", va="center", fontsize=8, color=INK)
    for (i, j), yb in zip(groups, y_bars):
        ax.hlines(yb, pos.iloc[i] - 0.004 * span, pos.iloc[j] + 0.004 * span, color=INK, lw=3.2)
    ax.set_title(title, loc="left", fontsize=9, color=INK)
    return groups
