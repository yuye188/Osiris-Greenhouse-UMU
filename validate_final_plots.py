display = print  # validation-only shim; Jupyter provides this normally

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import (mean_absolute_error, root_mean_squared_error, r2_score,
                            precision_score, recall_score, f1_score,
                            average_precision_score, confusion_matrix)


from greenhouse_dataset import (
    build_hourly_dataset, add_actuator_history, make_targets, TARGET_COLS,
    ACTIVE_START, CONTINUOUS_COLS, ACTUATOR_COLS, EXTERIOR_COLS,
    FORECAST_PREVDAY_FEATURE_COLS, make_forecast_horizon_features,
)

pd.set_option("display.width", 150)
plt.rcParams["figure.figsize"] = (11, 4)

df = build_hourly_dataset(root=".")
hist_cols = add_actuator_history(df, windows=(3, 6))     # act_*_r3, act_*_r6
print(f"added {len(hist_cols)} actuator-history features, e.g. {hist_cols[:4]}")

df = df.loc[ACTIVE_START:]                                # <-- re-scope
print("active-period span:", df.index.min(), "->", df.index.max(), "|", len(df), "hours")

clusters = pd.read_csv("model data/clusters.csv", index_col=0, parse_dates=True)["cluster"]
cluster_ohe = pd.get_dummies(clusters, prefix="clu").astype(float)
df = df.join(cluster_ohe)
cluster_cols = list(cluster_ohe.columns)
df[cluster_cols] = df[cluster_cols].fillna(0.0)
print("cluster dummies present in window:",
      [c for c in cluster_cols if df[c].sum() > 0])

fc_hist_cols = make_forecast_horizon_features(df, horizons=(1, 6, 12))
print(f"added {len(fc_hist_cols)} day-ahead forecast features, e.g. {fc_hist_cols[:4]}")


targets = make_targets(df, temp_col="temp_centro", hum_col="hum_centro",
                       horizons=(1, 6, 12))
print("targets:", TARGET_COLS)
targets.head()

time_feats = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "hour", "month"]
climate = CONTINUOUS_COLS + EXTERIOR_COLS + time_feats   # FORECAST_FEATURE_COLS dropped, see 2.3

feature_sets = {
    "no actuators":           climate,
    "+ actuators (current)":  climate + ACTUATOR_COLS,
    "+ actuator history":     climate + ACTUATOR_COLS + hist_cols,
    "+ cluster":              climate + ACTUATOR_COLS + hist_cols + cluster_cols,
    "+ day-ahead forecast":   climate + ACTUATOR_COLS + hist_cols + cluster_cols + fc_hist_cols,
}
actuator_features = ACTUATOR_COLS + hist_cols     # everything "climatic-system"

all_cols = climate + ACTUATOR_COLS + hist_cols + cluster_cols + fc_hist_cols
required = climate + ACTUATOR_COLS + hist_cols + fc_hist_cols   # cluster dummies may be 0
data = df[all_cols].join(targets).dropna(subset=required + TARGET_COLS)
print("modelling rows:", len(data), "| candidate features:", len(all_cols))


n = len(data); cut = int(n * 0.80)
train, test = data.iloc[:cut], data.iloc[cut:]
print(f"train: {train.index.min()} -> {train.index.max()}  ({len(train)} h)")
print(f"test : {test.index.min()} -> {test.index.max()}  ({len(test)} h)")
y_train, y_test = train[TARGET_COLS], test[TARGET_COLS]

def cvrmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred) / np.mean(y_true)

def score(y_true_df, y_pred_df):
    return pd.DataFrame({
        "MAE":    {c: mean_absolute_error(y_true_df[c], y_pred_df[c]) for c in TARGET_COLS},
        "CVRMSE": {c: cvrmse(y_true_df[c], y_pred_df[c]) for c in TARGET_COLS},
        "R2":     {c: r2_score(y_true_df[c], y_pred_df[c]) for c in TARGET_COLS},
    })

def cvrmse_np(y_true, y_pred):
    """Same recipe as cvrmse(), column-wise over a 2D array (no column labels
    needed) -- for use where y is a bare numpy array, e.g. inside fit_best()."""
    y_true = np.asarray(y_true, dtype="float64"); y_pred = np.asarray(y_pred, dtype="float64")
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    return rmse / np.mean(y_true, axis=0)

def mean_cvrmse_np(y_true, y_pred):
    """Mean CVRMSE across columns -- unitless, so temperature (degC) and
    humidity (%RH) targets are comparable. This is the single scalar used
    everywhere a model/hyperparameter ranking needs one number (fit_best(),
    the 'best model' picks in 2.8b/2.11/2.14), instead of averaging raw MAE
    across columns with different physical units (see 2.8's markdown)."""
    return np.mean(cvrmse_np(y_true, y_pred))

persist = pd.DataFrame(index=test.index)
for h in (1, 6, 12):
    persist[f"temp_h{h}"] = df["temp_centro"].reindex(test.index)
    persist[f"hum_h{h}"]  = df["hum_centro"].reindex(test.index)
baseline_scores = score(y_test, persist[TARGET_COLS])
baseline_scores.round(2)

LOOKBACK = 24  # hours of history fed to every "24h hist" model (tabular-flattened and LSTM alike)

def make_sequences(cols, lookback=LOOKBACK):
    """Slide a `lookback`-hour window over `data[cols]` -> `data[TARGET_COLS]`.

    Scalers are fit on the train rows only; sequences are built over the
    whole table so test-set windows can reach back into the train period
    without losing any test rows. Returns 3D arrays `(n, lookback, n_features)`
    — flatten to 2D for the tabular models (2.7c), keep 3D for the LSTM (2.7d).
    Same underlying data either way: that's the point.
    """
    X_all = data[cols].to_numpy(dtype="float32")
    y_all = data[TARGET_COLS].to_numpy(dtype="float32")

    x_scaler = StandardScaler().fit(X_all[:cut])
    y_scaler = StandardScaler().fit(y_all[:cut])
    X_scaled = x_scaler.transform(X_all).astype("float32")
    y_scaled = y_scaler.transform(y_all).astype("float32")

    idx = np.arange(lookback - 1, len(data))
    X_seq = np.stack([X_scaled[i - lookback + 1: i + 1] for i in idx])
    y_seq = y_scaled[idx]
    time_seq = data.index[idx]

    train_sel = time_seq <= train.index.max()
    test_sel = time_seq >= test.index.min()
    return X_seq[train_sel], y_seq[train_sel], X_seq[test_sel], y_scaler, time_seq[test_sel]


def model_search_space():
    """model name -> (constructor(params) -> unfitted estimator, small param grid)."""
    return {
        "Ridge": (
            lambda p: make_pipeline(StandardScaler(), Ridge(alpha=p["alpha"])),
            [{"alpha": a} for a in (0.1, 1.0, 10.0, 100.0)],
        ),
        "RandomForest": (
            lambda p: RandomForestRegressor(
                n_estimators=200, min_samples_leaf=p["min_samples_leaf"],
                n_jobs=-1, random_state=0),
            [{"min_samples_leaf": v} for v in (1, 2, 5)],
        ),
        "HistGB": (
            lambda p: MultiOutputRegressor(HistGradientBoostingRegressor(
                max_iter=300, learning_rate=p["learning_rate"], random_state=0)),
            [{"learning_rate": v} for v in (0.03, 0.05, 0.1)],
        ),
    }

def fit_best(model_ctor, param_grid, X, y, val_frac=0.15):
    """Chronologically hold out the last `val_frac` of (X, y) to score each
    candidate in `param_grid` by mean CVRMSE (unitless, see 2.5's
    mean_cvrmse_np), then refit the winner on the FULL (X, y). Never touches
    the real test set — this is a nested, in-train-only search, the tabular
    equivalent of the LSTM's `validation_split`.

    NOTE: this used to rank candidates by raw mean_absolute_error averaged
    across all 6 target columns -- which mixes degC (temp_*) and %RH (hum_*)
    in one number. Switched to mean CVRMSE so hyperparameter selection never
    silently favours whichever unit happens to have larger raw magnitude.
    """
    X = np.asarray(X); y = np.asarray(y)
    cut_inner = int(len(X) * (1 - val_frac))
    X_tr, y_tr = X[:cut_inner], y[:cut_inner]
    X_val, y_val = X[cut_inner:], y[cut_inner:]

    best_crit, best_params = np.inf, param_grid[0]
    for params in param_grid:
        m = model_ctor(params)
        m.fit(X_tr, y_tr)
        crit = mean_cvrmse_np(y_val, m.predict(X_val))
        if crit < best_crit:
            best_crit, best_params = crit, params

    final_model = model_ctor(best_params)
    final_model.fit(X, y)
    return final_model, best_params

results, predictions, best_params_log = {}, {}, {}
for fs_name, cols in feature_sets.items():
    for m_name, (ctor, grid) in model_search_space().items():
        model, best_p = fit_best(ctor, grid, train[cols], y_train)
        pred = pd.DataFrame(model.predict(np.asarray(test[cols])), index=test.index, columns=TARGET_COLS)
        results[(m_name, fs_name)] = score(y_test, pred)
        predictions[(m_name, fs_name)] = pred
        best_params_log[(m_name, fs_name)] = best_p
print("trained", len(results), "model x feature-set combinations (each tuned over a small grid)")


for fs_name, cols in feature_sets.items():
    Xtr_seq, ytr, Xte_seq, y_scaler, seq_test_time = make_sequences(cols)
    assert list(seq_test_time) == list(test.index), "24h-history sequences must line up with the tabular test split"
    mean_ = ytr.mean(axis=0)
    pred_scaled = np.tile(mean_, (len(Xte_seq), 1))
    for m_name, _ in model_search_space().items():
        pred = pd.DataFrame(y_scaler.inverse_transform(pred_scaled),
                            index=test.index, columns=TARGET_COLS)
        key_name = f"{m_name} (24h hist)"
        results[(key_name, fs_name)] = score(y_test, pred)
        predictions[(key_name, fs_name)] = pred
        best_params_log[(key_name, fs_name)] = {"stub": True}

n_hist = sum(1 for k in results if k[0].endswith("(24h hist)"))
print(f"[SPEED-CUT STUB] {n_hist} tabular (24h hist) entries filled with mean predictor")



# ==== VALIDATION STUB ====
def _safe_name(s):
    return "".join(c if c.isalnum() else "_" for c in s)

class _StubModel:
    def __init__(self):
        self._mean = None
    def compile(self, *a, **k): pass
    def fit(self, X, y, *a, **k):
        self._mean = np.asarray(y).mean(axis=0)
    def predict(self, X, verbose=0):
        return np.tile(self._mean, (len(X), 1))
    def count_params(self):
        return 12345

def make_lstm(*a, **k): return _StubModel()
def make_gru(*a, **k): return _StubModel()
def make_attention(*a, **k): return _StubModel()
def make_mlp(*a, **k): return _StubModel()

class _FakeEarlyStopping:
    def __init__(self, *a, **k): pass

class _FakeKerasUtils:
    @staticmethod
    def set_random_seed(seed): pass

class _FakeKerasCallbacks:
    EarlyStopping = _FakeEarlyStopping

class _FakeKeras:
    utils = _FakeKerasUtils
    callbacks = _FakeKerasCallbacks

keras = _FakeKeras()

def tune_neural(make_model_fn, n_features, n_targets, lookback, Xtr, ytr, project_name,
               max_trials=20, seed=0):
    model = make_model_fn(n_features=n_features, n_targets=n_targets, lookback=lookback)
    model.fit(Xtr, ytr)
    return model, {"units": 64, "dropout": 0.1, "lr": 1e-3, "optimizer": "adam"}


SEED_REPEATS = (0, 1, 2, 3, 4)
results_seeds, param_count_log = {}, {}

def refit_with_seeds(make_model_fn, best_params, n_features, n_targets, lookback,
                     Xtr, ytr, Xte, y_scaler, y_test_df, seeds=SEED_REPEATS):
    """Refit `make_model_fn(**best_params)` from scratch under several random
    seeds (same architecture, same winning hyperparameters, same data -- only
    the initialization/training randomness changes) and score each refit on
    the real test set. Returns (list of per-seed score() DataFrames, param count).
    """
    score_list, param_count = [], None
    for seed in seeds:
        keras.utils.set_random_seed(seed)
        m = make_model_fn(n_features=n_features, n_targets=n_targets, lookback=lookback,
                          units=best_params["units"], dropout=best_params["dropout"],
                          lr=best_params["lr"], optimizer=best_params["optimizer"])
        if param_count is None:
            param_count = m.count_params()
        es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
        m.fit(Xtr, ytr, validation_split=0.15, epochs=100, batch_size=32,
             callbacks=[es], verbose=0)
        pred = pd.DataFrame(y_scaler.inverse_transform(m.predict(Xte, verbose=0)),
                            index=y_test_df.index, columns=TARGET_COLS)
        score_list.append(score(y_test_df, pred))
    return score_list, param_count

for fs_name, cols in feature_sets.items():
    Xtr, ytr, Xte, y_scaler, seq_test_time = make_sequences(cols)
    assert list(seq_test_time) == list(test.index), "LSTM test sequences must line up with the tabular test split"

    model, best_p = tune_neural(
        make_lstm, n_features=Xtr.shape[-1], n_targets=len(TARGET_COLS), lookback=LOOKBACK,
        Xtr=Xtr, ytr=ytr, project_name=f"lstm_24h_{_safe_name(fs_name)}")

    pred_scaled = model.predict(Xte, verbose=0)
    pred = pd.DataFrame(y_scaler.inverse_transform(pred_scaled), index=test.index, columns=TARGET_COLS)

    results[("LSTM", fs_name)] = score(y_test, pred)
    predictions[("LSTM", fs_name)] = pred
    best_params_log[("LSTM", fs_name)] = best_p

    seed_scores, n_params = refit_with_seeds(
        make_lstm, best_p, n_features=Xtr.shape[-1], n_targets=len(TARGET_COLS), lookback=LOOKBACK,
        Xtr=Xtr, ytr=ytr, Xte=Xte, y_scaler=y_scaler, y_test_df=y_test)
    results_seeds[("LSTM", fs_name)] = seed_scores
    param_count_log[("LSTM", fs_name)] = n_params

print("trained LSTM on", len(feature_sets), "feature-set rungs (each tuned via KerasTuner BayesianOptimization)")


for fs_name, cols in feature_sets.items():
    Xtr0, ytr0, Xte0, y_scaler0, seq_test_time0 = make_sequences(cols, lookback=1)
    assert list(seq_test_time0) == list(test.index), "LSTM (t only) sequences must line up with the tabular test split"

    model, best_p0 = tune_neural(
        make_lstm, n_features=Xtr0.shape[-1], n_targets=len(TARGET_COLS), lookback=1,
        Xtr=Xtr0, ytr=ytr0, project_name=f"lstm_tonly_{_safe_name(fs_name)}")

    pred_scaled = model.predict(Xte0, verbose=0)
    pred = pd.DataFrame(y_scaler0.inverse_transform(pred_scaled), index=test.index, columns=TARGET_COLS)

    results[("LSTM (t only)", fs_name)] = score(y_test, pred)
    predictions[("LSTM (t only)", fs_name)] = pred
    best_params_log[("LSTM (t only)", fs_name)] = best_p0

    seed_scores, n_params = refit_with_seeds(
        make_lstm, best_p0, n_features=Xtr0.shape[-1], n_targets=len(TARGET_COLS), lookback=1,
        Xtr=Xtr0, ytr=ytr0, Xte=Xte0, y_scaler=y_scaler0, y_test_df=y_test)
    results_seeds[("LSTM (t only)", fs_name)] = seed_scores
    param_count_log[("LSTM (t only)", fs_name)] = n_params

print("trained LSTM (t only) on", len(feature_sets), "feature-set rungs (each tuned via KerasTuner BayesianOptimization)")


for fs_name, cols in feature_sets.items():
    Xtr, ytr, Xte, y_scaler, seq_test_time = make_sequences(cols)
    assert list(seq_test_time) == list(test.index), "Attention test sequences must line up with the tabular test split"

    model, best_p = tune_neural(
        make_attention, n_features=Xtr.shape[-1], n_targets=len(TARGET_COLS), lookback=LOOKBACK,
        Xtr=Xtr, ytr=ytr, project_name=f"attn_24h_{_safe_name(fs_name)}")

    pred_scaled = model.predict(Xte, verbose=0)
    pred = pd.DataFrame(y_scaler.inverse_transform(pred_scaled), index=test.index, columns=TARGET_COLS)

    results[("Attention", fs_name)] = score(y_test, pred)
    predictions[("Attention", fs_name)] = pred
    best_params_log[("Attention", fs_name)] = best_p

    seed_scores, n_params = refit_with_seeds(
        make_attention, best_p, n_features=Xtr.shape[-1], n_targets=len(TARGET_COLS), lookback=LOOKBACK,
        Xtr=Xtr, ytr=ytr, Xte=Xte, y_scaler=y_scaler, y_test_df=y_test)
    results_seeds[("Attention", fs_name)] = seed_scores
    param_count_log[("Attention", fs_name)] = n_params

print("trained Attention on", len(feature_sets), "feature-set rungs (each tuned via KerasTuner BayesianOptimization)")


for fs_name, cols in feature_sets.items():
    Xtr0, ytr0, Xte0, y_scaler0, seq_test_time0 = make_sequences(cols, lookback=1)
    assert list(seq_test_time0) == list(test.index), "Attention (t only) sequences must line up with the tabular test split"

    model, best_p0 = tune_neural(
        make_attention, n_features=Xtr0.shape[-1], n_targets=len(TARGET_COLS), lookback=1,
        Xtr=Xtr0, ytr=ytr0, project_name=f"attn_tonly_{_safe_name(fs_name)}")

    pred_scaled = model.predict(Xte0, verbose=0)
    pred = pd.DataFrame(y_scaler0.inverse_transform(pred_scaled), index=test.index, columns=TARGET_COLS)

    results[("Attention (t only)", fs_name)] = score(y_test, pred)
    predictions[("Attention (t only)", fs_name)] = pred
    best_params_log[("Attention (t only)", fs_name)] = best_p0

    seed_scores, n_params = refit_with_seeds(
        make_attention, best_p0, n_features=Xtr0.shape[-1], n_targets=len(TARGET_COLS), lookback=1,
        Xtr=Xtr0, ytr=ytr0, Xte=Xte0, y_scaler=y_scaler0, y_test_df=y_test)
    results_seeds[("Attention (t only)", fs_name)] = seed_scores
    param_count_log[("Attention (t only)", fs_name)] = n_params

print("trained Attention (t only) on", len(feature_sets), "feature-set rungs (each tuned via KerasTuner BayesianOptimization)")


for fs_name, cols in feature_sets.items():
    Xtr, ytr, Xte, y_scaler, seq_test_time = make_sequences(cols)
    assert list(seq_test_time) == list(test.index), "GRU test sequences must line up with the tabular test split"

    model, best_p = tune_neural(
        make_gru, n_features=Xtr.shape[-1], n_targets=len(TARGET_COLS), lookback=LOOKBACK,
        Xtr=Xtr, ytr=ytr, project_name=f"gru_24h_{_safe_name(fs_name)}")

    pred_scaled = model.predict(Xte, verbose=0)
    pred = pd.DataFrame(y_scaler.inverse_transform(pred_scaled), index=test.index, columns=TARGET_COLS)

    results[("GRU", fs_name)] = score(y_test, pred)
    predictions[("GRU", fs_name)] = pred
    best_params_log[("GRU", fs_name)] = best_p

    seed_scores, n_params = refit_with_seeds(
        make_gru, best_p, n_features=Xtr.shape[-1], n_targets=len(TARGET_COLS), lookback=LOOKBACK,
        Xtr=Xtr, ytr=ytr, Xte=Xte, y_scaler=y_scaler, y_test_df=y_test)
    results_seeds[("GRU", fs_name)] = seed_scores
    param_count_log[("GRU", fs_name)] = n_params

print("trained GRU on", len(feature_sets), "feature-set rungs (each tuned via KerasTuner BayesianOptimization)")

for fs_name, cols in feature_sets.items():
    Xtr0, ytr0, Xte0, y_scaler0, seq_test_time0 = make_sequences(cols, lookback=1)
    assert list(seq_test_time0) == list(test.index), "GRU (t only) sequences must line up with the tabular test split"

    model, best_p0 = tune_neural(
        make_gru, n_features=Xtr0.shape[-1], n_targets=len(TARGET_COLS), lookback=1,
        Xtr=Xtr0, ytr=ytr0, project_name=f"gru_tonly_{_safe_name(fs_name)}")

    pred_scaled = model.predict(Xte0, verbose=0)
    pred = pd.DataFrame(y_scaler0.inverse_transform(pred_scaled), index=test.index, columns=TARGET_COLS)

    results[("GRU (t only)", fs_name)] = score(y_test, pred)
    predictions[("GRU (t only)", fs_name)] = pred
    best_params_log[("GRU (t only)", fs_name)] = best_p0

    seed_scores, n_params = refit_with_seeds(
        make_gru, best_p0, n_features=Xtr0.shape[-1], n_targets=len(TARGET_COLS), lookback=1,
        Xtr=Xtr0, ytr=ytr0, Xte=Xte0, y_scaler=y_scaler0, y_test_df=y_test)
    results_seeds[("GRU (t only)", fs_name)] = seed_scores
    param_count_log[("GRU (t only)", fs_name)] = n_params

print("trained GRU (t only) on", len(feature_sets), "feature-set rungs (each tuned via KerasTuner BayesianOptimization)")

for fs_name, cols in feature_sets.items():
    Xtr, ytr, Xte, y_scaler, seq_test_time = make_sequences(cols)
    assert list(seq_test_time) == list(test.index), "MLP test sequences must line up with the tabular test split"

    model, best_p = tune_neural(
        make_mlp, n_features=Xtr.shape[-1], n_targets=len(TARGET_COLS), lookback=LOOKBACK,
        Xtr=Xtr, ytr=ytr, project_name=f"mlp_24h_{_safe_name(fs_name)}")

    pred_scaled = model.predict(Xte, verbose=0)
    pred = pd.DataFrame(y_scaler.inverse_transform(pred_scaled), index=test.index, columns=TARGET_COLS)

    results[("MLP", fs_name)] = score(y_test, pred)
    predictions[("MLP", fs_name)] = pred
    best_params_log[("MLP", fs_name)] = best_p

    seed_scores, n_params = refit_with_seeds(
        make_mlp, best_p, n_features=Xtr.shape[-1], n_targets=len(TARGET_COLS), lookback=LOOKBACK,
        Xtr=Xtr, ytr=ytr, Xte=Xte, y_scaler=y_scaler, y_test_df=y_test)
    results_seeds[("MLP", fs_name)] = seed_scores
    param_count_log[("MLP", fs_name)] = n_params

print("trained MLP on", len(feature_sets), "feature-set rungs (each tuned via KerasTuner BayesianOptimization)")

for fs_name, cols in feature_sets.items():
    Xtr0, ytr0, Xte0, y_scaler0, seq_test_time0 = make_sequences(cols, lookback=1)
    assert list(seq_test_time0) == list(test.index), "MLP (t only) sequences must line up with the tabular test split"

    model, best_p0 = tune_neural(
        make_mlp, n_features=Xtr0.shape[-1], n_targets=len(TARGET_COLS), lookback=1,
        Xtr=Xtr0, ytr=ytr0, project_name=f"mlp_tonly_{_safe_name(fs_name)}")

    pred_scaled = model.predict(Xte0, verbose=0)
    pred = pd.DataFrame(y_scaler0.inverse_transform(pred_scaled), index=test.index, columns=TARGET_COLS)

    results[("MLP (t only)", fs_name)] = score(y_test, pred)
    predictions[("MLP (t only)", fs_name)] = pred
    best_params_log[("MLP (t only)", fs_name)] = best_p0

    seed_scores, n_params = refit_with_seeds(
        make_mlp, best_p0, n_features=Xtr0.shape[-1], n_targets=len(TARGET_COLS), lookback=1,
        Xtr=Xtr0, ytr=ytr0, Xte=Xte0, y_scaler=y_scaler0, y_test_df=y_test)
    results_seeds[("MLP (t only)", fs_name)] = seed_scores
    param_count_log[("MLP (t only)", fs_name)] = n_params

print("trained MLP (t only) on", len(feature_sets), "feature-set rungs (each tuned via KerasTuner BayesianOptimization)")

params_table = pd.Series(best_params_log).apply(lambda p: ", ".join(f"{k}={v}" for k, v in p.items()))
params_table = params_table.unstack()[list(feature_sets)]
params_table


print("Parameter count (trainable weights) per neural model x rung:")
param_table = pd.Series(param_count_log).unstack()[list(feature_sets)]
display(param_table)

def seed_variability_table(results_seeds, feature_sets_keys, metric_fn):
    """One row per (model) that has seed-repeated results, one column per rung,
    cell = 'mean +/- std' of `metric_fn(score_df)` across SEED_REPEATS."""
    model_names = list(dict.fromkeys(k[0] for k in results_seeds))
    rows = {}
    for m in model_names:
        row = {}
        for fs in feature_sets_keys:
            key = (m, fs)
            if key not in results_seeds:
                continue
            vals = [metric_fn(s) for s in results_seeds[key]]
            row[fs] = f"{np.mean(vals):.3f} \u00b1 {np.std(vals):.3f}"
        rows[m] = row
    return pd.DataFrame(rows).T

print(f"Seed variability -- mean CVRMSE (unitless) across {len(SEED_REPEATS)} seeds, per model x rung:")
seed_variability_table(results_seeds, list(feature_sets), lambda s: s["CVRMSE"].mean())

MODEL_ORDER = ["Ridge", "RandomForest", "HistGB",
              "LSTM (t only)", "GRU (t only)", "Attention (t only)", "MLP (t only)",
              "Ridge (24h hist)", "RandomForest (24h hist)", "HistGB (24h hist)",
              "LSTM", "GRU", "Attention", "MLP"]

# Never average raw MAE across temperature (degC) and humidity (%RH) -- report
# each separately, in native units (see 2.8's markdown).
TEMP_TARGET_COLS = [c for c in TARGET_COLS if c.startswith("temp_")]
HUM_TARGET_COLS = [c for c in TARGET_COLS if c.startswith("hum_")]

print("Model ladder (mean MAE, temperature only, degC):")
ladder_mae_temp = pd.DataFrame(
    {fs: {m: results[(m, fs)]["MAE"].loc[TEMP_TARGET_COLS].mean() for m in MODEL_ORDER}
     for fs in feature_sets})
ladder_mae_temp.loc["persistence"] = baseline_scores["MAE"].loc[TEMP_TARGET_COLS].mean()
ladder_mae_temp = ladder_mae_temp[list(feature_sets)]
display(ladder_mae_temp.round(3))

print("Model ladder (mean MAE, humidity only, %RH):")
ladder_mae_hum = pd.DataFrame(
    {fs: {m: results[(m, fs)]["MAE"].loc[HUM_TARGET_COLS].mean() for m in MODEL_ORDER}
     for fs in feature_sets})
ladder_mae_hum.loc["persistence"] = baseline_scores["MAE"].loc[HUM_TARGET_COLS].mean()
ladder_mae_hum = ladder_mae_hum[list(feature_sets)]
ladder_mae_hum.round(3)


print("Model ladder (mean CVRMSE across targets -- unitless; this is the ranking metric, see 2.8's markdown):")
ladder = pd.DataFrame(
    {fs: {m: results[(m, fs)]["CVRMSE"].mean() for m in MODEL_ORDER}
     for fs in feature_sets})
ladder.loc["persistence"] = baseline_scores['CVRMSE'].mean()
ladder = ladder[list(feature_sets)]
ladder.round(3)


print("Model ladder (mean R² across targets):")
ladder = pd.DataFrame(
    {fs: {m: results[(m, fs)]["R2"].mean() for m in MODEL_ORDER}
     for fs in feature_sets})
ladder.loc["persistence"] = baseline_scores['R2'].mean()
ladder = ladder[list(feature_sets)]
ladder.round(3)


# pick the best model x feature-set by mean CVRMSE (unitless) over the 6
# targets -- never by mean MAE, which would average degC and %RH together
# (see 2.8's markdown / fit_best() in 2.7b).
best_key = min(results, key=lambda k: results[k]['CVRMSE'].mean())
best_scores = results[best_key]

report = pd.DataFrame({
    ('model',      'MAE'):    best_scores['MAE'],
    ('model',      'CVRMSE'): best_scores['CVRMSE'],
    ('model',      'R2'):     best_scores['R2'],
    ('persistence','MAE'):    baseline_scores['MAE'],
    ('persistence','CVRMSE'): baseline_scores['CVRMSE'],
    ('persistence','R2'):     baseline_scores['R2'],
})
report.columns = pd.MultiIndex.from_tuples(report.columns)
report[('improvement','CVRMSE %')] = (
    100 * (baseline_scores['CVRMSE'] - best_scores['CVRMSE']) / baseline_scores['CVRMSE'])
report[('improvement','R2 delta')] = best_scores['R2'] - baseline_scores['R2']

print(f'Best model/feature-set (ranked by mean CVRMSE): {best_key[0]}  |  {best_key[1]}')
print(f'(units: MAE in degC for temp_* and %RH for hum_*; CVRMSE and R2 are unitless -- CVRMSE is the ranking metric)')
report.round(3)

best_model_for_ablation = best_key[0]  # from the CVRMSE-ranked winner, 2.8b
no_act = results[(best_model_for_ablation, "no actuators")]["MAE"]
with_act = results[(best_model_for_ablation, "+ actuator history")]["MAE"]
ablation = pd.DataFrame({
    "MAE no actuators": no_act,
    "MAE + actuators":  with_act,
    "delta (neg=help)": with_act - no_act,
})
print(f"Ablation for {best_model_for_ablation}:")
ablation.round(3)

avg_cvrmse = {k: v["CVRMSE"].mean() for k, v in results.items()}
best_key = min(avg_cvrmse, key=avg_cvrmse.get)
print("best (ranked by mean CVRMSE):", best_key, "| mean CVRMSE =", round(avg_cvrmse[best_key], 3))
best_pred = predictions[best_key]

cols_to_plot = ["temp_h1", "temp_h6", "temp_h12", "hum_h1", "hum_h6", "hum_h12"]
labels = ["Temp +1h (°C)", "Temp +6h (°C)", "Temp +12h (°C)",
         "Humidity +1h (%)", "Humidity +6h (%)", "Humidity +12h (%)"]

fig, axes = plt.subplots(len(cols_to_plot), 1, figsize=(12, 16), sharex=True)
for ax, col, ylab in zip(axes, cols_to_plot, labels):
    ax.plot(test.index, y_test[col], label="actual", lw=1)
    ax.plot(test.index, best_pred[col], label="predicted", lw=1, alpha=0.8)
    ax.set_ylabel(ylab); ax.legend(loc="upper right")
axes[0].set_title(f"Best model: {best_key[0]} ({best_key[1]}) — test period")
plt.tight_layout(); plt.show()


def vpd_kpa(temp_c, rh_pct):
    """Vapour pressure deficit (kPa) from temperature (°C) and relative humidity (%), Tetens equation."""
    es = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))
    return es * (1 - rh_pct / 100.0)

horizons = (1, 6, 12)
vpd_real = pd.DataFrame({h: vpd_kpa(y_test[f"temp_h{h}"], y_test[f"hum_h{h}"]) for h in horizons})
vpd_best = pd.DataFrame({h: vpd_kpa(best_pred[f"temp_h{h}"], best_pred[f"hum_h{h}"]) for h in horizons})
vpd_real.columns.name = vpd_best.columns.name = "horizon (h)"

vpd_scores = pd.DataFrame({
    "MAE":    {h: mean_absolute_error(vpd_real[h], vpd_best[h]) for h in horizons},
    "CVRMSE": {h: cvrmse(vpd_real[h], vpd_best[h]) for h in horizons},
    "R2":     {h: r2_score(vpd_real[h], vpd_best[h]) for h in horizons},
})
vpd_scores.index.name = "horizon (h)"
print(f"VPD from best model: {best_key[0]} ({best_key[1]})  |  units: kPa for MAE, unitless for CVRMSE/R2")
vpd_scores.round(3)

# PLACEHOLDER thresholds -- typical greenhouse VPD guidance, NOT yet validated
# by the agronomy expert. Fixed as plain constants, independent of the
# train/test split, so swapping in the expert's numbers later only requires
# editing these two lines.
VPD_LOW_KPA = 0.4    # below this: high humidity, disease/fungal risk
VPD_HIGH_KPA = 1.6   # above this: water stress / stomatal closure risk

def vpd_event_scores(real, pred, low, high, kind):
    """kind: 'low' or 'high'. Binary event = real VPD crosses the fixed
    threshold; the model's own alarm uses the SAME threshold on its forecast."""
    real = np.asarray(real); pred = np.asarray(pred)
    if kind == "low":
        y_true = (real < low).astype(int)
        y_pred = (pred < low).astype(int)
        y_score = -pred   # more negative predicted VPD => more "low"-like
    else:
        y_true = (real > high).astype(int)
        y_pred = (pred > high).astype(int)
        y_score = pred    # higher predicted VPD => more "high"-like

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    n_events = int(y_true.sum())
    if n_events == 0 or n_events == len(y_true):
        auprc = np.nan   # undefined with a single-class horizon -- flag, don't fake a number
    else:
        auprc = average_precision_score(y_true, y_score)

    return {
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "False-alarm rate": far,
        "AUPRC": auprc,
        "n_events": n_events,
    }

vpd_event_low = pd.DataFrame(
    {h: vpd_event_scores(vpd_real[h], vpd_best[h], VPD_LOW_KPA, VPD_HIGH_KPA, "low") for h in horizons}).T
vpd_event_low.index.name = "horizon (h)"
print(f"Critical-VPD event detection -- LOW VPD (< {VPD_LOW_KPA} kPa), model: {best_key[0]} ({best_key[1]})")
display(vpd_event_low.round(3))

vpd_event_high = pd.DataFrame(
    {h: vpd_event_scores(vpd_real[h], vpd_best[h], VPD_LOW_KPA, VPD_HIGH_KPA, "high") for h in horizons}).T
vpd_event_high.index.name = "horizon (h)"
print(f"Critical-VPD event detection -- HIGH VPD (> {VPD_HIGH_KPA} kPa), model: {best_key[0]} ({best_key[1]})")
vpd_event_high.round(3)


MODEL_ORDER = ["Ridge", "RandomForest", "HistGB",
              "Ridge (24h hist)", "RandomForest (24h hist)", "HistGB (24h hist)"]


oracle_hist_cols = make_forecast_horizon_features(df, horizons=(1, 6, 12), source_cols=EXTERIOR_COLS)
print(f"added {len(oracle_hist_cols)} oracle (real future exterior weather) features, e.g. {oracle_hist_cols[:4]}")

oracle_na = df.loc[data.index, oracle_hist_cols].isna().any(axis=1)
dropped_idx = data.index[oracle_na]
print(f"{len(dropped_idx)} of {len(data)} modelling rows lack a genuine future exterior-weather "
      f"observation for at least one oracle horizon (raw station gaps occasionally exceed the "
      f"short-gap fill in greenhouse_dataset.py) -- excluded from ALL FIVE regimes below.")

n_train_before, n_test_before = len(train), len(test)
data = data.drop(index=dropped_idx).join(df.loc[:, oracle_hist_cols])
train = train.drop(index=dropped_idx, errors="ignore").join(df.loc[:, oracle_hist_cols])
test = test.drop(index=dropped_idx, errors="ignore").join(df.loc[:, oracle_hist_cols])
y_train, y_test = train[TARGET_COLS], test[TARGET_COLS]
cut = len(train)  # keep make_sequences() (2.7a) consistent with this section's reduced row set

print(f"train: {n_train_before} -> {len(train)}  |  test: {n_test_before} -> {len(test)}  "
      f"(same chronological split as the rest of the notebook, test now starts {test.index.min()})")

# Persistence, rescored on this section's (very slightly reduced) test rows -- same recipe as 2.6.
persist_r = pd.DataFrame(index=test.index)
for h in (1, 6, 12):
    persist_r[f"temp_h{h}"] = df["temp_centro"].reindex(test.index)
    persist_r[f"hum_h{h}"] = df["hum_centro"].reindex(test.index)
persistence_scores = score(y_test, persist_r[TARGET_COLS])

def fit_all_architectures(cols, label, include_24h=True):
    """Train every architecture in this notebook's grid on ONE feature list `cols`,
    for the information-regime comparison -- reuses every helper already defined
    in 2.7 (`model_search_space`/`fit_best`, `make_sequences`, `tune_neural`, and
    the four `make_*` builders) so the fitting logic is identical to 2.7b-2.7k,
    just parameterised by `cols` instead of `feature_sets`. Stores nothing in the
    global `results`/`predictions` dicts -- returns its own (r, p) dicts instead,
    kept separate from the feature-ablation ladder's results by design.

    `include_24h=False` (used for `snapshot`) trains ONLY the `(t only)` /
    `lookback=1` variant of every architecture: the `(24h hist)` tabular models
    and the 24h-lookback sequence models see 24 raw past hours by construction,
    which would silently turn "snapshot" into "historical" regardless of which
    engineered columns are in `cols`.
    """
    r, p = {}, {}

    for m_name, (ctor, grid) in model_search_space().items():
        model, _ = fit_best(ctor, grid, train[cols], y_train)
        pred = pd.DataFrame(model.predict(np.asarray(test[cols])), index=test.index, columns=TARGET_COLS)
        r[m_name] = score(y_test, pred); p[m_name] = pred

    if include_24h:
        Xtr_seq, ytr, Xte_seq, y_scaler, _ = make_sequences(cols)
        mean_ = ytr.mean(axis=0)
        pred_scaled = np.tile(mean_, (len(Xte_seq), 1))
        for m_name, _ in model_search_space().items():
            pred = pd.DataFrame(y_scaler.inverse_transform(pred_scaled),
                                index=test.index, columns=TARGET_COLS)
            key = f"{m_name} (24h hist)"
            r[key] = score(y_test, pred); p[key] = pred

    for arch_name, make_fn in [("LSTM", make_lstm), ("GRU", make_gru),
                               ("Attention", make_attention), ("MLP", make_mlp)]:
        if include_24h:
            Xtr, ytr, Xte, y_scaler, _ = make_sequences(cols)
            model, _ = tune_neural(make_fn, n_features=Xtr.shape[-1], n_targets=len(TARGET_COLS),
                                   lookback=LOOKBACK, Xtr=Xtr, ytr=ytr,
                                   project_name=f"{arch_name.lower()}_24h_regime_{_safe_name(label)}")
            pred = pd.DataFrame(y_scaler.inverse_transform(model.predict(Xte, verbose=0)),
                                index=test.index, columns=TARGET_COLS)
            r[arch_name] = score(y_test, pred); p[arch_name] = pred

        Xtr0, ytr0, Xte0, y_scaler0, _ = make_sequences(cols, lookback=1)
        model0, _ = tune_neural(make_fn, n_features=Xtr0.shape[-1], n_targets=len(TARGET_COLS),
                                lookback=1, Xtr=Xtr0, ytr=ytr0,
                                project_name=f"{arch_name.lower()}_tonly_regime_{_safe_name(label)}")
        pred0 = pd.DataFrame(y_scaler0.inverse_transform(model0.predict(Xte0, verbose=0)),
                             index=test.index, columns=TARGET_COLS)
        key0 = f"{arch_name} (t only)"
        r[key0] = score(y_test, pred0); p[key0] = pred0

    tag = "" if include_24h else " (t-only architectures only -- by regime definition, see markdown above)"
    print(f"[{label}] trained {len(r)} model variants{tag}")
    return r, p

regime_results, regime_predictions = {}, {}

r_snap, p_snap = fit_all_architectures(climate + ACTUATOR_COLS + cluster_cols, "snapshot", include_24h=False)
for m_name in r_snap:
    regime_results[(m_name, "snapshot")] = r_snap[m_name]
    regime_predictions[(m_name, "snapshot")] = p_snap[m_name]

r_orac, p_orac = fit_all_architectures(
    climate + ACTUATOR_COLS + hist_cols + cluster_cols + oracle_hist_cols, "oracle", include_24h=True)
for m_name in r_orac:
    regime_results[(m_name, "oracle")] = r_orac[m_name]
    regime_predictions[(m_name, "oracle")] = p_orac[m_name]

# historical / NWP: reuse the already-trained "+ cluster" / "+ day-ahead forecast" predictions
# (no retraining), rescored on this section's row set so all five regimes are comparable.
for m_name in MODEL_ORDER:
    hist_pred = predictions[(m_name, "+ cluster")].loc[test.index]
    nwp_pred = predictions[(m_name, "+ day-ahead forecast")].loc[test.index]
    regime_results[(m_name, "historical")] = score(y_test, hist_pred)
    regime_predictions[(m_name, "historical")] = hist_pred
    regime_results[(m_name, "NWP")] = score(y_test, nwp_pred)
    regime_predictions[(m_name, "NWP")] = nwp_pred

print("regime_results now holds", len(regime_results), "(model, regime) entries")

def regime_mean_metric(m, rg, metric, cols=None):
    key = (m, rg)
    if key not in regime_results:
        return np.nan
    s = regime_results[key][metric]
    return s.loc[cols].mean() if cols is not None else s.mean()

REGIME_ORDER = ["snapshot", "historical", "NWP", "oracle"]

# Never average raw MAE across temperature (degC) and humidity (%RH) -- same
# convention as 2.8. CVRMSE (next cell) is the unitless ranking metric.
print("Information-regime comparison (mean MAE, temperature only, degC):")
regime_ladder_mae_temp = pd.DataFrame(
    {rg: {m: regime_mean_metric(m, rg, "MAE", TEMP_TARGET_COLS) for m in MODEL_ORDER} for rg in REGIME_ORDER})
regime_ladder_mae_temp.loc["persistence"] = persistence_scores["MAE"].loc[TEMP_TARGET_COLS].mean()
display(regime_ladder_mae_temp.round(3))

print("Information-regime comparison (mean MAE, humidity only, %RH):")
regime_ladder_mae_hum = pd.DataFrame(
    {rg: {m: regime_mean_metric(m, rg, "MAE", HUM_TARGET_COLS) for m in MODEL_ORDER} for rg in REGIME_ORDER})
regime_ladder_mae_hum.loc["persistence"] = persistence_scores["MAE"].loc[HUM_TARGET_COLS].mean()
regime_ladder_mae_hum.round(3)

print("Information-regime comparison (mean CVRMSE across targets -- unitless, the ranking metric):")
regime_ladder_cvrmse = pd.DataFrame(
    {rg: {m: regime_mean_metric(m, rg, "CVRMSE") for m in MODEL_ORDER} for rg in REGIME_ORDER})
regime_ladder_cvrmse.loc["persistence"] = persistence_scores["CVRMSE"].mean()
regime_ladder_cvrmse.round(3)

print("Information-regime comparison (mean R2 across targets):")
regime_ladder_r2 = pd.DataFrame(
    {rg: {m: regime_mean_metric(m, rg, "R2") for m in MODEL_ORDER} for rg in REGIME_ORDER})
regime_ladder_r2.loc["persistence"] = persistence_scores["R2"].mean()
regime_ladder_r2.round(3)

fs_plot = fs_best  # richest feature-ablation rung, "+ day-ahead forecast" (2.8c)
cvrmse_final = pd.Series({m: results[(m, fs_plot)]["CVRMSE"].mean() for m in MODEL_ORDER})
cvrmse_final["persistence"] = baseline_scores["CVRMSE"].mean()

fig, ax = plt.subplots(figsize=(11, 5))
bar_colors = ["#4C72B0"] * len(MODEL_ORDER) + ["#888888"]
ax.bar(cvrmse_final.index.astype(str), cvrmse_final.values, color=bar_colors)
ax.set_ylabel("Mean CVRMSE (unitless)")
ax.set_title(f"Final model comparison -- richest rung ('{fs_plot}')")
ax.tick_params(axis="x", rotation=45)
for lbl in ax.get_xticklabels():
    lbl.set_ha("right")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(MODEL_ORDER))
width = 0.8 / len(REGIME_ORDER)
for i, rg in enumerate(REGIME_ORDER):
    vals = [regime_mean_metric(m, rg, "CVRMSE") for m in MODEL_ORDER]
    ax.bar(x + i * width, vals, width, label=rg)
ax.axhline(persistence_scores["CVRMSE"].mean(), color="k", linestyle="--", linewidth=1, label="persistence")
ax.set_xticks(x + width * (len(REGIME_ORDER) - 1) / 2)
ax.set_xticklabels(MODEL_ORDER, rotation=45, ha="right")
ax.set_ylabel("Mean CVRMSE (unitless)")
ax.set_title("Final model comparison across deployment information regimes (2.14b)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

neural_names = list(dict.fromkeys(k[0] for k in results_seeds))
fs_list = list(feature_sets)

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(neural_names))
width = 0.8 / len(fs_list)
for i, fs in enumerate(fs_list):
    means, stds = [], []
    for m in neural_names:
        key = (m, fs)
        if key in results_seeds:
            vals = [s["CVRMSE"].mean() for s in results_seeds[key]]
            means.append(np.mean(vals)); stds.append(np.std(vals))
        else:
            means.append(np.nan); stds.append(0.0)
    ax.bar(x + i * width, means, width, yerr=stds, capsize=3, label=fs)
ax.set_xticks(x + width * (len(fs_list) - 1) / 2)
ax.set_xticklabels(neural_names, rotation=45, ha="right")
ax.set_ylabel("Mean CVRMSE (unitless)")
ax.set_title(f"Neural-architecture comparison, mean +/- std across {len(SEED_REPEATS)} seeds (2.7m)")
ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
x = np.arange(len(horizons)); w = 0.35
for ax, metric in zip(axes, ["F1", "AUPRC"]):
    ax.bar(x - w / 2, vpd_event_low[metric], w, label=f"low VPD (< {VPD_LOW_KPA} kPa)")
    ax.bar(x + w / 2, vpd_event_high[metric], w, label=f"high VPD (> {VPD_HIGH_KPA} kPa)")
    ax.set_xticks(x); ax.set_xticklabels([f"+{h}h" for h in horizons])
    ax.set_title(metric); ax.set_ylim(0, 1)
axes[0].set_ylabel("score")
axes[1].legend(fontsize=8)
plt.suptitle("Critical-VPD event detection -- final scores (2.13b)")
plt.tight_layout()
plt.show()