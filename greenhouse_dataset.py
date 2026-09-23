"""
greenhouse_dataset.py
=====================

Shared data-assembly utilities for the greenhouse temperature/humidity
prediction project. Both notebooks import from here so the feature matrix is
built exactly the same way in the clustering step and in the modelling step.

The single entry point is `build_hourly_dataset()`, which merges every
resampled interior sensor, every actuator, the exterior weather station and
the Open-Meteo forecast onto one regular hourly index and returns a tidy
`pandas.DataFrame` indexed by timestamp.

Design decisions (kept deliberately explicit so they are easy to audit):

* **Interior climate** sensors are continuous; short gaps are filled by time
  interpolation. These are the greenhouse state we ultimately want to predict.
* **Actuators** are stored as "on_minutes" per hour. A missing value means the
  device was not reporting / not installed yet, which for control purposes is
  equivalent to "off" -> filled with 0.
* **Exterior** station data is raw (sub-hourly, tz-aware); we resample to hourly
  means and drop the timezone, matching the convention in the existing repo
  notebooks.
* **Forecast** timestamps are shifted by -2h to line the UTC-based forecast up
  with local greenhouse time, exactly as `ResampleClimateDeviceData.ipynb` does.

Nothing here looks at future values, so every returned column is safe to use as
a model input at prediction time (the forecast columns are, by construction,
known ahead of time).
"""

from __future__ import annotations

import os
import pandas as pd

# --------------------------------------------------------------------------- #
# Column maps: source file  ->  friendly column name
# --------------------------------------------------------------------------- #

# Interior continuous sensors (value column is `_value`)
INTERIOR_CONTINUOUS = {
    "temperatura_alta":            "temp_alta",
    "temperatura_baja":            "temp_baja",
    "temperatura_centro":          "temp_centro",
    "humedad_alta":                "hum_alta",
    "humedad_baja":                "hum_baja",
    "humedad_centro":              "hum_centro",
    "presion_barometrica_alta":    "pres_baro_alta",
    "presion_barometrica_baja":    "pres_baro_baja",
    "presion_barometrica_centro":  "pres_baro_centro",
    "presion_vapor_alta":          "pres_vapor_alta",
    "presion_vapor_baja":          "pres_vapor_baja",
    "par":                         "par_int",
    "radiacion_solar":             "rad_int",
    "velocidad_viento":            "viento_int",
}

# Interior actuators (value column is `on_minutes`); note the original repo
# filename typo "sobreo" (should be "sombreo") is preserved on purpose.
INTERIOR_ACTUATORS = {
    "calefactores_hourly_duration":       "act_calefactores",
    "fog_hourly_duration":                "act_fog",
    "bomba_fog_hourly_duration":          "act_bomba_fog",
    "bomba_soplante_hourly_duration":     "act_bomba_soplante",
    "recirculador_hourly_duration":       "act_recirculador",
    "recirculador2_hourly_duration":      "act_recirculador2",
    "pantalla_sobreo_cenital":            "act_pantalla_cenital",
    "pantalla_sobreo_lateral_sur":        "act_pantalla_lat_sur",
    "ventana_cenital_este":               "act_vent_cen_este",
    "ventana_cenital_oeste":              "act_vent_cen_oeste",
    "ventana_lateral_norte":              "act_vent_lat_norte",
    "ventana_lateral_oeste":              "act_vent_lat_oeste",
    "ventana_lateral_sur":                "act_vent_lat_sur",
}

# Exterior weather station (raw sub-hourly files with the widest coverage).
# NOTE: "Radiación_solar_global_0" is deliberately excluded — that file is 100 %
# sentinel (-999999), i.e. a dead channel. Exterior radiation still enters the
# model via `ext_rad_par` and the forecast's `fc_rad` (shortwave_radiation).
EXTERIOR_RAW = {
    "Temperatura_ambiente_0":       "ext_temp",
    "Humedad_ambiente_0":           "ext_hum",
    "Radiación_solar_par_0":        "ext_rad_par",
    "Velocidad_del_viento_0":       "ext_viento",
}

# Values at or below this are logger "no-data" sentinels (e.g. -999999), not
# real physical readings; none of the kept quantities can legitimately reach it.
SENTINEL_FLOOR = -100.0

# Open-Meteo forecast columns we keep
FORECAST_COLS = {
    "temperature_2m":        "fc_temp",
    "relative_humidity_2m":  "fc_hum",
    "precipitation":         "fc_precip",
    "wind_speed_10m":        "fc_viento",
    "shortwave_radiation":   "fc_rad",
}

# Open-Meteo "previous day" forecast columns — the value that was actually
# forecast about a day *before* the target hour (Open-Meteo's Previous Runs
# API: `<var>_previous_day1` = the run available roughly 1 calendar day ahead
# of the valid time). Unlike the un-suffixed columns above — which the
# Previous-Runs-API backfills from the *most recent* model run once a date is
# in the past, i.e. closer to the final analysis than to a real advance
# forecast — these are honestly "known ahead of time" and safe to use as a
# forecast feature for genuine multi-hour-ahead prediction.
FORECAST_PREVDAY_COLS = {
    "temperature_2m_previous_day1":       "fc_temp_prevday",
    "relative_humidity_2m_previous_day1": "fc_hum_prevday",
    "precipitation_previous_day1":        "fc_precip_prevday",
    "wind_speed_10m_previous_day1":       "fc_viento_prevday",
    "shortwave_radiation_previous_day1":  "fc_rad_prevday",
}

# Prefix groups, handy for feature selection in the notebooks
CONTINUOUS_COLS = list(INTERIOR_CONTINUOUS.values())
ACTUATOR_COLS = list(INTERIOR_ACTUATORS.values())
EXTERIOR_COLS = list(EXTERIOR_RAW.values())
FORECAST_FEATURE_COLS = list(FORECAST_COLS.values())
FORECAST_PREVDAY_FEATURE_COLS = list(FORECAST_PREVDAY_COLS.values())
# Companion "stale" flag per previous_day1 variable: 1 = this hour's value is a
# frozen forward-filled carry-over from a long forecast outage (see the note
# above the fill logic in `build_hourly_dataset`), 0 = a genuine fetched value.
FORECAST_PREVDAY_STALE_COLS = [f"{c}_stale" for c in FORECAST_PREVDAY_FEATURE_COLS]

# First hour at which *all* climatic systems (fog, shade screens, every roof and
# side vent) are reporting. Before this the actuator columns are mostly zero
# because the devices were not installed / logging yet, which unfairly dilutes
# their signal. Scope the modelling here so the systems are active in both the
# train and the test split. (Pumps/recirculators alone start 2026-02-26.)
ACTIVE_START = "2026-04-24"


# --------------------------------------------------------------------------- #
# Low-level readers
# --------------------------------------------------------------------------- #

def _read_interior(path: str, value_col: str, new_name: str) -> pd.Series:
    """Read one resampled interior CSV -> hourly Series named `new_name`."""
    df = pd.read_csv(path)
    df["_time"] = pd.to_datetime(df["_time"])
    s = df.set_index("_time")[value_col].rename(new_name)
    # collapse any accidental duplicate hours
    s = s[~s.index.duplicated(keep="first")].sort_index()
    return s


def _read_exterior(path: str, new_name: str) -> pd.Series:
    """Read a raw exterior CSV, resample to hourly mean, drop tz."""
    df = pd.read_csv(path)
    t = pd.to_datetime(df["_time"], format="mixed", utc=True).dt.tz_localize(None)
    s = pd.Series(df["_value"].values, index=t, name=new_name).sort_index()
    s = s.where(s > SENTINEL_FLOOR)      # drop -999999 no-data spikes before averaging
    s = s.resample("h").mean()
    return s


def _read_forecast(path: str) -> pd.DataFrame:
    """Read the Open-Meteo forecast, shift -2h, keep/rename selected columns.

    Keeps both the "latest run" columns (`FORECAST_COLS`) and the
    "previous_day1" columns (`FORECAST_PREVDAY_COLS`) — see the note above
    `FORECAST_PREVDAY_COLS` for why the two are not interchangeable.
    """
    all_cols = {**FORECAST_COLS, **FORECAST_PREVDAY_COLS}
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]) - pd.Timedelta(hours=2)
    df = df.set_index("date")[list(all_cols)].rename(columns=all_cols)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def build_hourly_dataset(
    root: str = ".",
    add_time_features: bool = True,
    interpolate_limit: int = 3,
) -> pd.DataFrame:
    """Assemble the full hourly feature frame.

    Parameters
    ----------
    root : project root containing the data folders.
    add_time_features : append hour / day-of-year sine-cosine encodings.
    interpolate_limit : max consecutive hours to fill in continuous sensors by
        time interpolation (longer gaps stay NaN and their rows are dropped by
        the caller as needed).

    Returns
    -------
    DataFrame indexed by an hourly DatetimeIndex.
    """
    resampled = os.path.join(root, "inside data resampled")
    outside = os.path.join(root, "outside data")
    forecast = os.path.join(root, "forecast data", "hourly_weather_data.csv")

    series = []

    # interior continuous
    for fname, col in INTERIOR_CONTINUOUS.items():
        series.append(_read_interior(os.path.join(resampled, f"{fname}.csv"),
                                     "_value", col))
    # interior actuators
    for fname, col in INTERIOR_ACTUATORS.items():
        series.append(_read_interior(os.path.join(resampled, f"{fname}.csv"),
                                     "on_minutes", col))
    # exterior
    for fname, col in EXTERIOR_RAW.items():
        series.append(_read_exterior(os.path.join(outside, f"{fname}.csv"), col))

    df = pd.concat(series, axis=1)

    # regular hourly index across the observed span
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full_index)
    df.index.name = "time"

    # merge forecast
    fc = _read_forecast(forecast)
    df = df.join(fc, how="left")

    # --- gap handling -----------------------------------------------------
    # actuators: absent report == off
    df[ACTUATOR_COLS] = df[ACTUATOR_COLS].fillna(0.0)
    # continuous interior + exterior + forecast: fill short gaps forward-only.
    # NOTE: this used to be `df[cont].interpolate(method="time", limit=...)`,
    # but time-interpolation fills the value at time t from the *next* known
    # observation too (t' > t) -- i.e. a future value. That is not available
    # in a deployable/online setting, so gaps are now bridged with a plain
    # forward-fill (uses only observations at or before t), capped at the
    # same `interpolate_limit` hours.
    cont = CONTINUOUS_COLS + EXTERIOR_COLS + FORECAST_FEATURE_COLS + FORECAST_PREVDAY_FEATURE_COLS
    df[cont] = df[cont].ffill(limit=interpolate_limit)

    # The `previous_day1` forecast has had at least one real multi-day outage
    # in the archive (observed: ~260-364 h across variables, Apr-May 2026)
    # that is far too long for `interpolate_limit` to bridge. This used to be
    # backfilled from the "latest run" `FORECAST_FEATURE_COLS` column for the
    # same variable -- but that column is the Previous-Runs API's
    # most-recent-run estimate, which does NOT carry the ~1-day lead-time
    # guarantee `previous_day1` is relied on for (see the note above
    # `FORECAST_PREVDAY_COLS`), so falling back to it silently broke that
    # guarantee for exactly the rows it patched.
    #
    # Instead we bridge the remaining gap with an UNLIMITED forward-fill of
    # the previous_day1 column's OWN past values -- still only ever pulling
    # from something known before hour t, so the lead-time guarantee holds;
    # it just means "assume the last real day-ahead forecast persists" through
    # the outage rather than dropping the hour entirely. Every hour bridged
    # this way is flagged 1 in a companion `*_stale` column (0 = a genuine
    # fetched value) so models can learn to trust a long-frozen forecast less;
    # `make_forecast_horizon_features` shifts these flags per horizon exactly
    # like the forecast values themselves, so row t also knows whether the
    # forecast it sees for t+h was stale.
    for prevday_col, stale_col in zip(FORECAST_PREVDAY_FEATURE_COLS, FORECAST_PREVDAY_STALE_COLS):
        still_missing = df[prevday_col].isna()
        n_missing = int(still_missing.sum())
        df[stale_col] = still_missing.astype(float)
        if n_missing:
            print(f"note: {prevday_col} still missing {n_missing}h after the short-gap fill; "
                  f"bridging with an unlimited forward-fill of its own past values "
                  f"(flagged in {stale_col})")
            df[prevday_col] = df[prevday_col].ffill()

    if add_time_features:
        import numpy as np
        hour = df.index.hour
        doy = df.index.dayofyear
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        df["doy_sin"] = np.sin(2 * np.pi * doy / 365)
        df["doy_cos"] = np.cos(2 * np.pi * doy / 365)
        df["hour"] = hour
        df["month"] = df.index.month

    return df


def make_targets(df: pd.DataFrame,
                 temp_col: str = "temp_centro",
                 hum_col: str = "hum_centro",
                 horizons=(1, 6, 12)) -> pd.DataFrame:
    """Return a DataFrame of future targets (shifted by -h for each horizon)."""
    out = {}
    for h in horizons:
        out[f"temp_h{h}"] = df[temp_col].shift(-h)
        out[f"hum_h{h}"] = df[hum_col].shift(-h)
    return pd.DataFrame(out, index=df.index)


TARGET_COLS = [f"{v}_h{h}" for h in (1, 6, 12) for v in ("temp", "hum")]


def add_actuator_history(df: pd.DataFrame, windows=(3, 6)) -> list[str]:
    """Add rolling-sum features capturing *recent* climatic-system activity.

    A single hour of "on_minutes" only says what a device is doing right now.
    But thermal/humidity inertia means what matters is how long it has been
    running. For each actuator we add the total minutes it ran over the last
    `w` hours (inclusive), e.g. `act_fog_r6` = fog minutes in the past 6h.

    Modifies `df` in place and returns the list of new column names.
    """
    new_cols = []
    for col in ACTUATOR_COLS:
        for w in windows:
            name = f"{col}_r{w}"
            df[name] = df[col].rolling(window=w, min_periods=1).sum()
            new_cols.append(name)
    return new_cols


def make_forecast_horizon_features(df: pd.DataFrame, horizons=(1, 6, 12),
                                   source_cols=None) -> list[str]:
    """Add, for each forecast variable and horizon, the forecast value that
    applies to the *target* hour t+h — i.e. what was actually known about
    that future hour's weather at prediction time.

    Uses the `previous_day1` forecast columns by default (`source_cols`),
    since those genuinely carry ~1-day lead time (see `FORECAST_PREVDAY_COLS`).
    Built the same way as `make_targets()`: shift back by h so row t carries
    the value forecast for t+h. Because the previous_day1 forecast for hour
    t+h was itself issued about a day before t+h, it was already known at t
    for every horizon used in this project (max 12h, well inside that ~1-day
    lead time), so this introduces no leakage.

    Modifies `df` in place and returns the list of new column names.
    """
    if source_cols is None:
        source_cols = FORECAST_PREVDAY_FEATURE_COLS
    new_cols = []
    for col in source_cols:
        for h in horizons:
            name = f"{col}_h{h}"
            df[name] = df[col].shift(-h)
            new_cols.append(name)
    return new_cols


def get_split_date(root: str = ".", active_start: str = ACTIVE_START,
                   horizons=(1, 6, 12), test_frac: float = 0.20) -> "pd.Timestamp":
    """Return the first timestamp of the chronological test period.

    Both notebooks must agree on exactly the same train/test boundary:
    notebook 2 (prediction) splits its modelling table 80/20 chronologically,
    and notebook 1's clustering scaler/KMeans has to be fit *only* on rows
    strictly before that same boundary -- otherwise the cluster label for a
    test-period hour would depend on test-period data (through the scaler's
    mean/std or the KMeans centroids), which is exactly the leakage the
    clustering notebook was flagged for.

    This replays the same scoping/dropna recipe notebook 2 uses to build its
    modelling table -- scope to `active_start`, add actuator-history and
    day-ahead forecast horizon features, then drop rows missing any required
    input or target -- *before* cluster columns are joined in. Cluster values
    never affect which rows survive that dropna (they are one-hot columns
    filled with 0 where absent), so this function has no dependency on
    `clusters.csv` and can run standalone, ahead of notebook 1 producing it.
    """
    df = build_hourly_dataset(root=root)
    hist_cols = add_actuator_history(df)
    df = df.loc[active_start:]
    fc_hist_cols = make_forecast_horizon_features(df, horizons=horizons)
    targets = make_targets(df, horizons=horizons)

    time_feats = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "hour", "month"]
    climate = CONTINUOUS_COLS + EXTERIOR_COLS + time_feats
    required = climate + ACTUATOR_COLS + hist_cols + fc_hist_cols

    idx = df[required].join(targets).dropna(subset=required + TARGET_COLS).index
    cut = int(len(idx) * (1 - test_frac))
    return idx[cut]


if __name__ == "__main__":
    d = build_hourly_dataset()
    print("shape:", d.shape)
    print("span:", d.index.min(), "->", d.index.max())
    print("\nmissing per column:")
    print(d.isna().sum().sort_values(ascending=False).head(20))
