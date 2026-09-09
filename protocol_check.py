"""
protocol_check.py
=================
Verify the Osiris greenhouse **August-2026 campaign** (empty greenhouse) against
the written protocol in ``REGLAS_EXPERIMENTO.pdf``, and provide the plotting
helpers used by ``checkExperimentsProtocol.ipynb``.

Why this module exists
----------------------
``climate devices experiments data/`` holds one CSV per InfluxDB series, in
**event-logged** form: a row is written only when the value *changes* (plus a
few sub-second echoes of the same value). Nothing in the folder states which
protocol block a row belongs to. To answer "did the experiment run as
designed?" the raw event streams have to be turned back into *commanded levels
per protocol block*, and only then compared with the prescription.

Three decoding facts, all established empirically (see the notebook):

1. **Timestamps are UTC; the protocol is written in local time.**
   ``_time`` carries a ``Z`` suffix. Murcia in Aug/Sep 2026 is CEST = UTC+2.
   Read naively, every block looks 1-2 h off the prescribed
   00:00/03:00/.../21:00 grid. Shifted by +2 h it lands exactly on it.

2. **FOG encodes a duty cycle, not a level.** ``_value == 1`` means the nozzles
   are spraying. Integrating that state over each minute recovers the protocol's
   axis directly: seconds-on-per-minute clusters at ~5, ~10 and ~19-20, i.e. the
   prescribed 5s / 10s / 20s. ``-15`` is the idle status code, not a level.

3. **Ventana / Pantalla report position, so they ramp.** A block commanded to
   50 % is logged as 0, 1, 2, ... 50 while the motor travels, then holds at 50.
   The commanded level is therefore the *sustained* value (time-weighted mode
   after the ramp), never the instantaneous reading. Negative values (-1, -3)
   are position-unknown/fault codes.

Everything below is a consequence of those three facts.

Author: generated for the Osiris-Greenhouse-UMU repo.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DATA_DIR = Path("climate devices experiments data")

#: The CSV timestamps are UTC; the protocol is written in Murcia local time.
UTC_OFFSET = pd.Timedelta(hours=2)  # CEST, valid for Aug-Sep 2026

#: Day 1 of the campaign, local time. Anchored empirically: every actuator's
#: first record lines up with its first prescribed Phase-A trial under this
#: assumption (windows d1, shade d3, fog+recirc d4, lights+recirc2 d5).
DAY1 = pd.Timestamp("2026-08-03")

#: Protocol section 3: 8 blocks/day, each 2 h treatment + 1 h washout.
BLOCK_HOURS = 3
TREATMENT_HOURS = 2

#: Skip the first ``RAMP_SKIP`` minutes of a window when reading its level, so
#: the motor travel of Ventana/Pantalla is not mistaken for the setpoint.
RAMP_SKIP = pd.Timedelta(minutes=5)

PHASES = {
    "A": (1, 5),    # Caracterizacion  - step trials
    "B": (6, 12),   # Dosis-respuesta  - one group, several levels
    "C": (13, 18),  # Interacciones    - fractional factorial, 32 runs
    "D": (19, 31),  # Combinaciones aleatorias - Sobol fill
}


@dataclass
class Group:
    """One of the protocol's 7 actuator groups."""

    name: str
    files: list[str]
    levels: list[float]           # levels the protocol allows
    tol: float                    # how far an observed level may sit from a legal one
    kind: str = "analog"          # 'analog' | 'binary' | 'duty'
    note: str = ""
    #: devices inside the group are commanded together, so they should agree
    expect_agreement: bool = True


GROUPS: dict[str, Group] = {
    "LIGHTS": Group(
        "LIGHTS",
        [f"Luz_artificial_{p}_0.csv" for p in
         ("centro-este", "centro-oeste", "nor-este", "nor-oeste", "sur-este", "sur-oeste")],
        levels=[0, 2, 4, 5, 6, 8, 10],
        tol=0.6,
        note="6 lamps driven in conjunction, analogue intensity 0-10",
    ),
    "FOG": Group(
        "FOG",
        ["Fog_0.csv"],
        levels=[0, 5, 10, 20],
        tol=2.0,
        kind="duty",
        note="seconds spraying per minute; -15 is the idle code",
    ),
    "VENT_ROOF": Group(
        "VENT_ROOF",
        ["Ventana_cenital_este_0.csv", "Ventana_cenital_oeste_0.csv"],
        levels=[0, 25, 50, 75, 100],
        tol=6.0,
        note="2 roof windows (east + west), aperture %",
    ),
    "VENT_SIDE": Group(
        "VENT_SIDE",
        ["Ventana_lateral_norte_0.csv", "Ventana_lateral_sur_0.csv", "Ventana_lateral_oeste_0.csv"],
        levels=[0, 50, 100],
        tol=6.0,
        note="3 side windows (north, south, west), aperture %",
    ),
    "SHADE": Group(
        "SHADE",
        ["Pantalla_de_sombreo_cenital_0.csv"],
        levels=[0, 50, 100],
        tol=6.0,
        note="zenithal shade screen only; the lateral-south screen file is ignored",
    ),
    "RECIRC": Group(
        "RECIRC",
        ["Recirculador_0.csv", "Recirculador_centro_0.csv"],
        levels=[0, 1],
        tol=0.01,
        kind="binary",
        note="1 recirculator; the two files are duplicate exports of one device",
        expect_agreement=True,
    ),
    "RECIRC2": Group(
        "RECIRC2",
        ["Calefactores_0.csv", "Calefactores_2.csv"],
        levels=[0, 1],
        tol=0.01,
        kind="binary",
        note="2 heaters used as recirculators; the two files are duplicate exports",
        expect_agreement=True,
    ),
}

#: Files deliberately excluded, with the reason (reported so the exclusion is visible).
IGNORED_FILES = {
    "Calefactores_1.csv": "2-row duplicate export fragment (user-confirmed)",
    "Recirculador_1.csv": "2-row duplicate export fragment (user-confirmed)",
    "Pantalla_de_sombreo_lateral_sur_0.csv": "18 rows over 13 s; screen not part of the protocol (user-confirmed)",
}

SENSORS: dict[str, str] = {
    "temp_alta": "Temperatura_Interior_Alta_(ºC)_0.csv",
    "temp_centro": "Temperatura_Interior_Centro_(ºC)_0.csv",
    "temp_baja": "Temperatura_Interior_Baja_(ºC)_0.csv",
    "hum_alta": "Humedad_Relativa_interior_Alta_(%)_0.csv",
    "hum_centro": "Humedad_Relativa_interior_Centro_(%)_0.csv",
    "hum_baja": "Humedad_Relativa_interior_Baja_(%)_0.csv",
    "co2": "CO2_(ppm)_0.csv",
    "dpv": "DPV_(kPa)_0.csv",
    "par": "PAR_interior_(µmol_m_-2_s_-1)_0.csv",
    "rad": "Radiación_Solar_Global_Interior_(W_m_-2)_0.csv",
}

#: Plausibility windows used by the data-quality report.
SENSOR_RANGE = {
    "temp_alta": (5, 60), "temp_centro": (5, 60), "temp_baja": (5, 60),
    "hum_alta": (5, 100), "hum_centro": (5, 100), "hum_baja": (5, 100),
    "co2": (0, 5000), "dpv": (0, 12), "par": (0, 2500), "rad": (0, 1400),
}


# --------------------------------------------------------------------------- #
# The prescription (transcribed from REGLAS_EXPERIMENTO.pdf)
# --------------------------------------------------------------------------- #

#: Section 4 - Phase A step trials. ``day`` is 1-based; ``hour`` is local.
PHASE_A = [
    dict(id="A1",  day=1, hour=None, group=None,        level=None, desc="Passive baseline, all off, 24 h"),
    dict(id="A2",  day=2, hour=6,  group="VENT_ROOF", level=100),
    dict(id="A3",  day=2, hour=18, group="VENT_ROOF", level=100),
    dict(id="A4",  day=3, hour=6,  group="VENT_SIDE", level=100),
    dict(id="A5",  day=3, hour=18, group="VENT_SIDE", level=100),
    dict(id="A6",  day=3, hour=8,  group="SHADE",     level=100),
    dict(id="A7",  day=4, hour=6,  group="FOG",       level=20),
    dict(id="A8",  day=4, hour=18, group="FOG",       level=20),
    dict(id="A9",  day=4, hour=8,  group="RECIRC",    level=1),
    dict(id="A10", day=5, hour=8,  group="RECIRC2",   level=1),
    dict(id="A11", day=5, hour=18, group="LIGHTS",    level=10),
]

#: Section 5 - Phase B dose-response, run back-to-back in this group order.
PHASE_B = [
    ("VENT_ROOF", [25, 0, 50, 100, 75, 75, 50, 25, 0, 100, 50, 75, 0, 100, 25]),
    ("VENT_SIDE", [100, 50, 0, 50, 100, 0, 0, 100, 50]),
    ("SHADE",     [50, 100, 0, 50, 100, 0, 0, 50, 100]),
    ("FOG",       [0, 20, 5, 10, 20, 10, 0, 5, 20, 5, 0, 10]),
    ("LIGHTS",    [0, 5, 8, 2, 10, 0, 10, 8, 5, 2, 0, 5, 2, 8, 10]),
]

_C_COLS = ["LIGHTS", "FOG", "VENT_ROOF", "VENT_SIDE", "SHADE", "RECIRC", "RECIRC2"]

#: Section 6 - Phase C resolution-IV fractional factorial, already randomised.
PHASE_C = [
    [10, 20, 100, 100, 100, 1, 1], [10,  0,   0, 100, 100, 1, 0],
    [ 0,  0, 100, 100, 100, 1, 1], [10, 20,   0, 100, 100, 0, 1],
    [10, 20,   0, 100,   0, 0, 0], [10,  0,   0, 100,   0, 1, 1],
    [ 0,  0, 100,   0, 100, 0, 0], [10,  0, 100, 100,   0, 0, 1],
    [10, 20, 100,   0,   0, 0, 1], [ 0, 20,   0, 100,   0, 1, 1],
    [10, 20,   0,   0,   0, 1, 1], [10,  0, 100, 100, 100, 0, 0],
    [10,  0, 100,   0,   0, 1, 0], [ 0, 20,   0,   0, 100, 0, 1],
    [ 0,  0, 100, 100,   0, 1, 0], [10,  0,   0,   0,   0, 0, 0],
    [ 0,  0,   0, 100, 100, 0, 1], [ 0,  0,   0,   0,   0, 1, 1],
    [ 0, 20, 100, 100, 100, 0, 0], [10,  0,   0,   0, 100, 0, 1],
    [10, 20,   0,   0, 100, 1, 0], [ 0, 20, 100,   0,   0, 1, 0],
    [10,  0, 100,   0, 100, 1, 1], [ 0, 20,   0, 100, 100, 1, 0],
    [ 0, 20, 100, 100,   0, 0, 1], [10, 20, 100, 100,   0, 1, 0],
    [ 0,  0,   0, 100,   0, 0, 0], [ 0,  0, 100,   0,   0, 0, 1],
    [10, 20, 100,   0, 100, 0, 0], [ 0,  0,   0,   0, 100, 1, 0],
    [ 0, 20, 100,   0, 100, 1, 1], [ 0, 20,   0,   0,   0, 0, 0],
]

#: Section 7 - Phase D, first 24 of 96 Sobol blocks (the only ones the PDF lists;
#: ``fase_D_programa.csv`` is NOT in the repo, so blocks 25-96 cannot be checked).
PHASE_D_SAMPLE = [
    [ 8,  5,   0,  50, 100, 1, 1], [ 4, 20, 100,  50,   0, 0, 0],
    [ 0,  0,  25, 100,  50, 0, 1], [ 8, 10,  75,   0, 100, 1, 0],
    [10,  0, 100,   0,  50, 1, 1], [ 0, 10,   0, 100,   0, 0, 0],
    [ 2,  5,  50,   0,   0, 0, 1], [ 6, 20,  50,  50, 100, 1, 0],
    [ 6,  0,  75, 100,   0, 1, 1], [ 2, 10,  25,   0, 100, 0, 0],
    [ 2,  5, 100, 100,  50, 0, 1], [10, 20,   0,  50,  50, 1, 0],
    [ 8,  5,  25,  50,   0, 1, 1], [ 0, 20,  75,  50,  50, 0, 0],
    [ 4,  0,  25,   0, 100, 0, 1], [ 6, 10,  75, 100,   0, 1, 0],
    [ 8,  0,  50,   0, 100, 0, 0], [ 4, 10,  50, 100,  50, 1, 1],
    [ 0,  5,   0,   0,   0, 1, 0], [10, 20, 100, 100, 100, 0, 1],
    [10,  5,  75, 100, 100, 0, 0], [ 2, 20,  25,   0,   0, 1, 1],
    [ 2,  0,  75,  50,   0, 1, 0], [ 6, 10,  25,  50,  50, 0, 1],
]

#: Section 8 - safety rules (the only sensor-triggered rules).
SAFETY = dict(
    overtemp_c=45.0,          # T_int > 45 C  -> windows 100, SHADE 0, LIGHTS 0
    condensation_rh=100.0,    # RH = 100 % sustained > 20 min with FOG on -> FOG off
    condensation_minutes=20,
)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def _read_events(path: Path) -> pd.Series:
    """Read one InfluxDB export as an event series in **local** time.

    Sub-second echoes of the same value are collapsed; negative status codes are
    turned into ``NaN`` (they are not levels). Returns a Series indexed by local
    timestamp. The value holds until the next index entry.
    """
    df = pd.read_csv(path)
    t = pd.to_datetime(df["_time"], format="mixed", utc=True).dt.tz_localize(None) + UTC_OFFSET
    v = pd.to_numeric(df["_value"], errors="coerce")
    s = pd.Series(v.values, index=t.values).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s


def count_status_codes(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """How many negative (status/fault) records each actuator file carries."""
    rows = []
    for g in GROUPS.values():
        for f in g.files:
            df = pd.read_csv(data_dir / f)
            v = pd.to_numeric(df["_value"], errors="coerce")
            neg = v[v < 0]
            rows.append(dict(group=g.name, file=f, n_rows=len(v),
                             n_negative=int(len(neg)),
                             codes=sorted(neg.unique().tolist()) or None))
    return pd.DataFrame(rows)


def load_actuators(data_dir: Path = DATA_DIR, freq: str = "1min") -> pd.DataFrame:
    """Reconstruct a regular step series per actuator device.

    Event logs are forward-filled (a value holds until it changes) onto a
    regular ``freq`` grid spanning the whole campaign. Returns one column per
    *device*, named ``GROUP::file-stem``.
    """
    raw: dict[str, pd.Series] = {}
    for g in GROUPS.values():
        for f in g.files:
            raw[f"{g.name}::{Path(f).stem}"] = _read_events(data_dir / f)

    lo = min(s.index.min() for s in raw.values()).floor("h")
    hi = max(s.index.max() for s in raw.values()).ceil("h")
    grid = pd.date_range(lo, hi, freq=freq)

    out = {}
    for k, s in raw.items():
        # reindex onto the union, ffill, then take the grid: preserves the
        # step semantics without inventing transitions between samples
        out[k] = s.reindex(s.index.union(grid)).ffill().reindex(grid)
    return pd.DataFrame(out, index=grid)


def load_fog_duty(data_dir: Path = DATA_DIR) -> pd.Series:
    """Seconds-per-minute the fog nozzles actually sprayed.

    This is the protocol's own intensity axis for FOG (0 / 5s / 10s / 20s), so
    it is computed at 1 s resolution rather than read off the raw level.
    """
    s = _read_events(data_dir / "Fog_0.csv")
    raw = pd.read_csv(data_dir / "Fog_0.csv")
    t = pd.to_datetime(raw["_time"], format="mixed", utc=True).dt.tz_localize(None) + UTC_OFFSET
    v = pd.to_numeric(raw["_value"], errors="coerce")
    spray = pd.Series((v == 1).astype(float).values, index=t.values).sort_index()
    spray = spray[~spray.index.duplicated(keep="last")]
    sec = spray.resample("1s").ffill()
    return sec.resample("1min").sum()


#: A value that never changes again for this many minutes is a frozen channel,
#: not a measurement. Six sensors froze within 6 min of each other on
#: 2026-08-22 (see ``detect_flatlines``); the frozen values sit inside the
#: plausible range, so nothing but this check catches them.
FLATLINE_MIN = 600


def detect_flatlines(data_dir: Path = DATA_DIR, min_minutes: int = FLATLINE_MIN) -> pd.DataFrame:
    """Find sensors whose value freezes and never moves again.

    This is the failure mode a range check cannot see: the channel keeps
    reporting, at a perfectly plausible value, for ever. Returns one row per
    sensor with the timestamp of the last real reading.
    """
    rows = []
    for name, f in SENSORS.items():
        s = _read_events(data_dir / f)
        m = s.resample("1min").mean()
        same = m.diff().abs() < 1e-9
        run = same.astype(int).groupby((~same).cumsum()).cumsum()
        longest = int(run.max()) if len(run) else 0
        end = run.idxmax() if len(run) else pd.NaT
        start = end - pd.Timedelta(minutes=longest) if longest else pd.NaT
        # "dead" = the frozen run is the longest one *and* it reaches the end of
        # the record; a long flat stretch in the middle is suspicious but alive.
        dead = bool(longest >= min_minutes and pd.notna(end)
                    and end >= m.index.max() - pd.Timedelta(hours=1))
        rows.append(dict(sensor=name, longest_flat_min=longest,
                         frozen_from=start if dead else pd.NaT,
                         frozen_value=round(float(m.loc[end]), 2) if dead else np.nan,
                         dead=dead))
    return pd.DataFrame(rows)


def load_sensors(data_dir: Path = DATA_DIR, freq: str = "1min",
                 mask_flatlines: bool = True) -> pd.DataFrame:
    """Interior climate sensors on a regular local-time grid (mean per bin).

    With ``mask_flatlines`` (the default) a frozen channel becomes ``NaN`` from
    the moment it froze, so a dead sensor shows as missing rather than silently
    contributing a constant to every later average.
    """
    flat = detect_flatlines(data_dir).set_index("sensor") if mask_flatlines else None
    cols = {}
    for name, f in SENSORS.items():
        s = _read_events(data_dir / f)
        lo, hi = SENSOR_RANGE.get(name, (-np.inf, np.inf))
        if name == "par":
            s = s.clip(lower=0)          # small negative dark offset at night
        s = s.where((s >= lo) & (s <= hi))
        if flat is not None and bool(flat.loc[name, "dead"]):
            s = s.loc[s.index < flat.loc[name, "frozen_from"]]
        cols[name] = s.resample(freq).mean()
    df = pd.DataFrame(cols)
    return df.interpolate(limit=15, limit_direction="both")


# --------------------------------------------------------------------------- #
# Block grid and level recovery
# --------------------------------------------------------------------------- #

def block_grid(start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> pd.DatetimeIndex:
    """The prescribed 3 h block starts (00:00, 03:00, ... 21:00 local)."""
    start = (start or DAY1).floor("D")
    end = end or (DAY1 + pd.Timedelta(days=35))
    return pd.date_range(start, end, freq=f"{BLOCK_HOURS}h")


def campaign_day(ts: pd.Timestamp) -> int:
    """1-based campaign day for a local timestamp."""
    return int((pd.Timestamp(ts).floor("D") - DAY1).days) + 1


def phase_of(ts: pd.Timestamp) -> str:
    d = campaign_day(ts)
    for ph, (a, b) in PHASES.items():
        if a <= d <= b:
            return ph
    return "post" if d > 31 else "pre"


def _weighted_mode(x: pd.Series) -> float:
    """Most-held value in a uniformly sampled window (NaN if all missing)."""
    x = x.dropna()
    if x.empty:
        return np.nan
    return float(x.round(2).value_counts().idxmax())


def _snap(value: float, group: Group) -> tuple[float, bool]:
    """Snap an observed level to the nearest legal one; flag if it is off-grid."""
    if not np.isfinite(value):
        return np.nan, False
    legal = min(group.levels, key=lambda L: abs(L - value))
    return float(legal), bool(abs(legal - value) <= group.tol)


def group_level(act: pd.DataFrame, fog: pd.Series, group: Group,
                t0: pd.Timestamp, t1: pd.Timestamp) -> dict:
    """Recover one group's commanded level over the window ``[t0, t1)``.

    For Ventana/Pantalla the ramp at the start of the window is skipped, so the
    figure returned is the *sustained* position, i.e. the setpoint.
    """
    w0 = t0 + RAMP_SKIP
    if group.name == "FOG":
        seg = fog.loc[(fog.index >= w0) & (fog.index < t1)]
        raw = float(seg.mean()) if len(seg) else np.nan
        spread = 0.0
    else:
        cols = [c for c in act.columns if c.startswith(group.name + "::")]
        seg = act.loc[(act.index >= w0) & (act.index < t1), cols]
        if seg.empty or seg.isna().all().all():
            raw, spread = np.nan, np.nan
        else:
            per_dev = {c: _weighted_mode(seg[c]) for c in cols}
            vals = [v for v in per_dev.values() if np.isfinite(v)]
            raw = float(np.median(vals)) if vals else np.nan
            spread = float(np.nanmax(vals) - np.nanmin(vals)) if len(vals) > 1 else 0.0

    level, on_grid = _snap(raw, group)
    return dict(raw=raw, level=level, on_grid=on_grid, device_spread=spread)


def block_table(act: pd.DataFrame, fog: pd.Series,
                grid: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """One row per 3 h block: treatment level and washout level for all 7 groups.

    Columns
    -------
    ``<GROUP>``            snapped treatment level (protocol units)
    ``<GROUP>_raw``        the measured sustained value before snapping
    ``<GROUP>_offgrid``    True if the measurement is not within tolerance of any legal level
    ``<GROUP>_spread``     max-min across devices in the group (0 = they agree)
    ``<GROUP>_wash``       snapped level during the 3rd (washout) hour
    """
    grid = grid if grid is not None else block_grid(act.index.min(), act.index.max())
    grid = grid[(grid >= act.index.min().floor("D")) & (grid <= act.index.max())]

    rows = []
    for t0 in grid:
        t_treat_end = t0 + pd.Timedelta(hours=TREATMENT_HOURS)
        t_block_end = t0 + pd.Timedelta(hours=BLOCK_HOURS)
        row = dict(block=t0, day=campaign_day(t0), phase=phase_of(t0))
        for g in GROUPS.values():
            tr = group_level(act, fog, g, t0, t_treat_end)
            wa = group_level(act, fog, g, t_treat_end, t_block_end)
            row[g.name] = tr["level"]
            row[f"{g.name}_raw"] = tr["raw"]
            row[f"{g.name}_offgrid"] = (not tr["on_grid"]) and np.isfinite(tr["raw"])
            row[f"{g.name}_spread"] = tr["device_spread"]
            row[f"{g.name}_wash"] = wa["level"]
        rows.append(row)

    bt = pd.DataFrame(rows).set_index("block")
    names = [g.name for g in GROUPS.values()]
    bt["n_active"] = (bt[names].fillna(0) > 0).sum(axis=1)
    bt["washout_clean"] = (bt[[f"{n}_wash" for n in names]].fillna(0) == 0).all(axis=1)
    return bt


# --------------------------------------------------------------------------- #
# Phase checks
# --------------------------------------------------------------------------- #

def detect_onsets(act: pd.DataFrame, fog: pd.Series, group: Group,
                  min_gap_h: float = 2.0) -> pd.DataFrame:
    """Find times where a group switches from off to a non-zero level.

    Phase A does not use the 3 h block grid ("hold ON until steady state, ~4 h"),
    so its trials are located by detecting the steps themselves and only then
    matched against the prescription.
    """
    if group.name == "FOG":
        s = fog.copy()
        on = (s.fillna(0) > 1.0).astype(int)          # >1 s/min of spraying
    else:
        cols = [c for c in act.columns if c.startswith(group.name + "::")]
        s = act[cols].median(axis=1)
        on = (s.fillna(0) > 0).astype(int)

    s5 = on.rolling("5min").mean()                    # ignore 1-sample glitches
    onmask = s5 > 0.6
    # A series whose first sample is already ON has no rising edge to find, but
    # that first sample *is* the onset (the device only began logging when it
    # was first commanded). Prepend an off sample so the edge appears.
    prev = onmask.shift(1)
    prev.iloc[0] = False
    edges = onmask.astype(int) - prev.astype(int)
    starts = onmask.index[edges == 1]

    kept: list[pd.Timestamp] = []
    for t in starts:
        if not kept or (t - kept[-1]) >= pd.Timedelta(hours=min_gap_h):
            kept.append(t)

    out = []
    for t in kept:
        seg = s.loc[t:t + pd.Timedelta(hours=4)]
        out.append(dict(group=group.name, onset=t, day=campaign_day(t),
                        phase=phase_of(t), peak=float(np.nanmax(seg)) if len(seg) else np.nan))
    return pd.DataFrame(out)


def check_phase_a(act: pd.DataFrame, fog: pd.Series, tol_hours: float = 1.0) -> pd.DataFrame:
    """Verify the 11 Phase-A step trials: right day, right hour, right level, alone."""
    onsets = pd.concat([detect_onsets(act, fog, g) for g in GROUPS.values()],
                       ignore_index=True)

    rows: list[dict] = []
    claimed: set = set()   # observed onsets already assigned to a trial
    for tr in PHASE_A:
        want_day, want_hour, gname = tr["day"], tr["hour"], tr["group"]

        if gname is None:  # A1 baseline: nothing may be on for 24 h
            d0 = DAY1 + pd.Timedelta(days=want_day - 1)
            d1 = d0 + pd.Timedelta(days=1)
            seg = act.loc[d0:d1]
            active = []
            for g in GROUPS.values():
                cols = [c for c in seg.columns if c.startswith(g.name + "::")]
                if not cols:
                    continue
                on = (seg[cols].fillna(0) > 0).any(axis=1)
                if on.any():
                    active.append(f"{g.name} on for {int(on.sum())} min "
                                  f"({on[on].index.min():%H:%M}-{on[on].index.max():%H:%M})")
            fseg = fog.loc[d0:d1]
            if len(fseg) and (fseg.fillna(0) > 1).any():
                active.append(f"FOG spraying for {int((fseg.fillna(0) > 1).sum())} min")
            covered = act.loc[d0:d1].notna().any(axis=1)
            gap = "" if covered.all() else \
                f"; no actuator logging before {covered[covered].index.min():%H:%M}"
            rows.append(dict(trial="A1", group="(passive)", prescribed_at=d0,
                             prescribed_level=None, observed_at=pd.NaT, observed_level=None,
                             verdict="PASS" if not active else "FAIL",
                             detail=("all groups off for 24 h" + gap) if not active
                                    else "; ".join(active) + gap))
            continue

        want_t = DAY1 + pd.Timedelta(days=want_day - 1, hours=want_hour)
        cand = onsets[(onsets.group == gname) & (~onsets.onset.isin(claimed))].copy()
        # Only Phase-A-era steps are candidates; a later Phase-B step of the same
        # group must never be offered as "the trial that ran 6 days late".
        cand = cand[cand.day <= PHASES["A"][1] + 1]
        if cand.empty:
            rows.append(dict(trial=tr["id"], group=gname, prescribed_at=want_t,
                             prescribed_level=tr["level"], observed_at=pd.NaT,
                             observed_level=np.nan, verdict="MISSING",
                             detail="no unclaimed step for this group within Phase A"))
            continue

        cand["dt_h"] = (cand.onset - want_t).dt.total_seconds() / 3600
        best = cand.loc[cand.dt_h.abs().idxmin()]
        dt = float(best.dt_h)
        claimed.add(best.onset)   # one observed step can satisfy only one trial

        g = GROUPS[gname]
        lev = group_level(act, fog, g, best.onset,
                          best.onset + pd.Timedelta(hours=TREATMENT_HOURS))
        got = lev["level"]

        # who else was on during this trial? Phase A demands a single group.
        others = []
        for og in GROUPS.values():
            if og.name == gname:
                continue
            ol = group_level(act, fog, og, best.onset,
                             best.onset + pd.Timedelta(hours=TREATMENT_HOURS))
            if np.isfinite(ol["level"]) and ol["level"] > 0:
                others.append(f"{og.name}={ol['level']:g}")

        ok_time = abs(dt) <= tol_hours
        ok_level = np.isfinite(got) and abs(got - tr["level"]) <= g.tol
        verdict = "PASS" if (ok_time and ok_level and not others) else \
                  ("FAIL" if not ok_level else "PARTIAL")
        detail = []
        if not ok_time:
            detail.append(f"onset {dt:+.1f} h from prescribed")
        if not ok_level:
            detail.append(f"level {got} != {tr['level']}")
        if others:
            detail.append("not isolated: " + ", ".join(others))
        rows.append(dict(trial=tr["id"], group=gname, prescribed_at=want_t,
                         prescribed_level=tr["level"], observed_at=best.onset,
                         observed_level=got, verdict=verdict,
                         detail="; ".join(detail) or "on time, right level, isolated"))
    return pd.DataFrame(rows)


def check_phase_b(bt: pd.DataFrame, search_days: tuple[int, int] = (6, 13)) -> tuple[pd.DataFrame, dict]:
    """Verify the Phase-B dose-response sequences.

    The protocol gives, per group, an exact randomised level order run in
    consecutive blocks. The start block is not stated, so it is found by sliding
    the concatenated expected sequence over the observed blocks and taking the
    best-matching alignment.
    """
    lo = DAY1 + pd.Timedelta(days=search_days[0] - 2)
    hi = DAY1 + pd.Timedelta(days=search_days[1] + 1)
    obs = bt.loc[(bt.index >= lo) & (bt.index <= hi)]

    expected: list[tuple[str, float]] = []
    for gname, seq in PHASE_B:
        expected += [(gname, lv) for lv in seq]

    def score(offset: int) -> float:
        hits = 0
        for i, (gname, lv) in enumerate(expected):
            if offset + i >= len(obs):
                return -1
            got = obs.iloc[offset + i][gname]
            if np.isfinite(got) and abs(got - lv) <= GROUPS[gname].tol:
                hits += 1
        return hits / len(expected)

    best_off, best = 0, -1.0
    for off in range(max(1, len(obs) - len(expected) + 1)):
        sc = score(off)
        if sc > best:
            best, best_off = sc, off

    rows = []
    for i, (gname, lv) in enumerate(expected):
        idx = best_off + i
        if idx >= len(obs):
            rows.append(dict(seq=i + 1, group=gname, block=pd.NaT, prescribed=lv,
                             observed=np.nan, match=False, isolated=None,
                             washout_clean=None))
            continue
        r = obs.iloc[idx]
        got = r[gname]
        others = [g.name for g in GROUPS.values()
                  if g.name != gname and np.isfinite(r[g.name]) and r[g.name] > 0]
        rows.append(dict(seq=i + 1, group=gname, block=obs.index[idx], prescribed=lv,
                         observed=got,
                         match=bool(np.isfinite(got) and abs(got - lv) <= GROUPS[gname].tol),
                         isolated=not others, other_active=",".join(others) or "",
                         washout_clean=bool(r["washout_clean"])))
    res = pd.DataFrame(rows)
    summary = dict(alignment_start=obs.index[best_off] if len(obs) else pd.NaT,
                   match_rate=round(float(res.match.mean()), 3),
                   isolated_rate=round(float(res.isolated.fillna(False).mean()), 3),
                   washout_rate=round(float(res.washout_clean.fillna(False).mean()), 3),
                   n_blocks=len(res))
    return res, summary


def _match_design(bt: pd.DataFrame, design: list[list[float]],
                  search_days: tuple[int, int], label: str) -> tuple[pd.DataFrame, dict]:
    """Slide a design matrix over the observed blocks and report row-by-row agreement."""
    lo = DAY1 + pd.Timedelta(days=search_days[0] - 2)
    hi = DAY1 + pd.Timedelta(days=search_days[1] + 2)
    obs = bt.loc[(bt.index >= lo) & (bt.index <= hi)]

    def score(off: int) -> float:
        tot = hit = 0
        for i, run in enumerate(design):
            if off + i >= len(obs):
                return -1
            r = obs.iloc[off + i]
            for c, want in zip(_C_COLS, run):
                got = r[c]
                tot += 1
                if np.isfinite(got) and abs(got - want) <= GROUPS[c].tol:
                    hit += 1
        return hit / max(tot, 1)

    best_off, best = 0, -1.0
    for off in range(max(1, len(obs) - len(design) + 1)):
        sc = score(off)
        if sc > best:
            best, best_off = sc, off

    rows = []
    for i, run in enumerate(design):
        idx = best_off + i
        if idx >= len(obs):
            rows.append(dict(run=i + 1, block=pd.NaT, cells_ok=0, cells=len(_C_COLS),
                             exact=False, mismatches="block missing"))
            continue
        r = obs.iloc[idx]
        bad = []
        ok = 0
        for c, want in zip(_C_COLS, run):
            got = r[c]
            if np.isfinite(got) and abs(got - want) <= GROUPS[c].tol:
                ok += 1
            else:
                bad.append(f"{c}: want {want:g}, got {got if np.isfinite(got) else 'NaN'}")
        rows.append(dict(run=i + 1, block=obs.index[idx], cells_ok=ok, cells=len(_C_COLS),
                         exact=(ok == len(_C_COLS)), mismatches="; ".join(bad),
                         washout_clean=bool(r["washout_clean"])))
    res = pd.DataFrame(rows)
    summary = dict(design=label,
                   alignment_start=obs.index[best_off] if len(obs) else pd.NaT,
                   exact_runs=int(res.exact.sum()), n_runs=len(res),
                   cell_match_rate=round(float(res.cells_ok.sum() / (len(res) * len(_C_COLS))), 3))
    return res, summary


def check_phase_c(bt: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Verify the 32-run resolution-IV factorial of Phase C."""
    return _match_design(bt, PHASE_C, PHASES["C"], "Phase C (32-run res-IV factorial)")


def check_phase_d(bt: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Verify the 24 Phase-D blocks the PDF lists (blocks 25-96 are unavailable)."""
    return _match_design(bt, PHASE_D_SAMPLE, (PHASES["D"][0], PHASES["D"][0] + 4),
                         "Phase D (first 24 of 96 Sobol blocks)")


# --------------------------------------------------------------------------- #
# Phase D against its generated programme (all 96 blocks)
# --------------------------------------------------------------------------- #

#: FOG in the programme CSV is written as the protocol writes it - seconds of
#: spraying per minute, or "off".
_FOG_WORDS = {"off": 0.0, "5s": 5.0, "10s": 10.0, "20s": 20.0}


def load_phase_d_programme(path: str | Path = "fase_D_programa.csv") -> pd.DataFrame:
    """Read the seeded Sobol programme for Phase D: one row per block.

    The CSV carries both rows the protocol describes - ``tratamiento`` (2 h) and
    ``lavado`` (1 h, everything to 0) - so the washout is prescribed explicitly
    and can be checked rather than assumed.

    Note the times in this file are the **planned** ones. Execution ran a day
    later (see ``find_programme_offset``), so nothing here should be joined to
    the logs on ``planned_start`` directly.
    """
    df = pd.read_csv(path)
    cols = ["LIGHTS", "FOG", "VENT_ROOF", "VENT_SIDE", "SHADE", "RECIRC", "RECIRC2"]
    tr = df[df.row_type == "tratamiento"].reset_index(drop=True)

    def _lv(col: str, v) -> float:
        s = str(v).strip()
        if col == "FOG":
            return _FOG_WORDS.get(s.lower(), np.nan)
        if col in ("RECIRC", "RECIRC2"):
            return 1.0 if s.upper() == "ON" else 0.0
        return float(s)

    out = pd.DataFrame({
        "block_id": tr.block_id,
        "planned_start": pd.to_datetime(tr.start_time),
        **{c: [_lv(c, v) for v in tr[c]] for c in cols},
    })
    out["n"] = range(1, len(out) + 1)
    return out


def find_programme_offset(bt: pd.DataFrame, prog: pd.DataFrame,
                          candidates_days: tuple[int, ...] = (0, 1, 2, -1)) -> dict:
    """Work out how the programme's planned times map onto the executed blocks.

    Rather than trusting either the CSV's dates or the protocol's day numbering,
    every candidate whole-day shift is scored on how many of the 7 x 96 commanded
    levels it reproduces, and the best is returned. A correct alignment scores
    near 1.0; a wrong one scores near chance.
    """
    names = [g.name for g in GROUPS.values()]
    scores = {}
    for d in candidates_days:
        hit = tot = 0
        for _, r in prog.iterrows():
            t0 = r.planned_start + pd.Timedelta(days=d)
            if t0 not in bt.index:
                continue
            for c in names:
                want, got = r[c], bt.at[t0, c]
                tot += 1
                if np.isfinite(got) and abs(got - want) <= GROUPS[c].tol:
                    hit += 1
        scores[d] = round(hit / tot, 4) if tot else 0.0
    best = max(scores, key=scores.get)
    return dict(offset_days=best, scores=scores, best_score=scores[best])


def check_phase_d_full(bt: pd.DataFrame, prog: pd.DataFrame, offset_days: int,
                       sens: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Verify all 96 Phase-D blocks - treatment levels and washout - against the programme.

    This is the check the PDF alone could not support: it lists only 24 of the 96
    blocks. Returns one row per block with the per-group comparison, whether the
    washout hour was actually clear, and (with ``sens``) the temperature reached.
    """
    names = [g.name for g in GROUPS.values()]
    rows = []
    for _, r in prog.iterrows():
        t0 = r.planned_start + pd.Timedelta(days=offset_days)
        d = dict(block_id=r.block_id, n=int(r.n), block=t0)
        if t0 not in bt.index:
            d.update(cells_ok=0, cells=len(names), exact=False, executed=None,
                     mismatches="block outside the logged period", washout_clean=None)
            rows.append(d)
            continue
        b = bt.loc[t0]
        bad, ok, got_any = [], 0, False
        for c in names:
            want, got = r[c], b[c]
            if np.isfinite(got) and got > 0:
                got_any = True
            if np.isfinite(got) and abs(got - want) <= GROUPS[c].tol:
                ok += 1
            else:
                bad.append(f"{c}: want {want:g}, got " +
                           ("NaN" if not np.isfinite(got) else f"{got:g}"))
        # A block whose prescription is non-trivial but which shows nothing at all
        # is a queue failure; one that is prescribed all-zero cannot be told apart
        # from a missing run, so it is marked rather than judged.
        prescribed_any = any(r[c] > 0 for c in names)
        d.update(cells_ok=ok, cells=len(names), exact=(ok == len(names)),
                 mismatches="; ".join(bad),
                 washout_clean=bool(b["washout_clean"]),
                 prescribed_all_zero=(not prescribed_any),
                 executed=(got_any if prescribed_any else None),
                 not_queued=bool(prescribed_any and not got_any))
        if sens is not None:
            seg = sens.loc[t0:t0 + pd.Timedelta(hours=TREATMENT_HOURS), "temp_centro"].dropna()
            d["centro_peak"] = round(float(seg.max()), 1) if len(seg) else np.nan
            d["min_over_50"] = int(pd.Series(seg[seg > 50].index).dt.floor("min").nunique()) \
                if len(seg) else 0
        rows.append(d)
    res = pd.DataFrame(rows)
    summary = dict(
        offset_days=offset_days,
        n_blocks=len(res),
        exact=int(res.exact.sum()),
        cell_match_rate=round(float(res.cells_ok.sum() / (len(res) * len(names))), 4),
        not_queued=int(res.not_queued.fillna(False).sum()),
        washout_clean=int(res.washout_clean.fillna(False).sum()),
        over_50=int((res.get("min_over_50", pd.Series(dtype=float)).fillna(0) > 0).sum()),
    )
    return res, summary


def slot_confounding(prog: pd.DataFrame) -> pd.DataFrame:
    """How much of each group's commanded level is determined by time of day?

    Protocol note 2 claims the Phase-D programme is "equilibrado por franja
    horaria". This tests it: ``var_explained_by_slot`` is the share of the level's
    variance that lies *between* the eight 3 h slots rather than within them, so
    0 means the level is independent of time of day and 1 means it is fully
    determined by it. ``levels_per_slot`` shows how many distinct levels each slot
    actually samples - the balance claim requires the whole range in every slot.

    Anything above ~0.3 here is a confounding warning: that group's effect cannot
    be separated from the diurnal cycle in Phase D, whatever the model.
    """
    names = [g.name for g in GROUPS.values()]
    p = prog.copy()
    p["slot"] = (p.n - 1) % (24 // BLOCK_HOURS)
    p["hour"] = p.planned_start.dt.hour
    p["parity"] = p.n % 2

    rows = []
    for c in names:
        v = p[c].astype(float)
        ss_tot = float(((v - v.mean()) ** 2).sum())
        ss_bet = float(sum(len(g) * (g[c].mean() - v.mean()) ** 2 for _, g in p.groupby("slot")))
        per_slot = p.groupby("slot")[c].nunique()
        rows.append(dict(
            group=c,
            levels_available=len(set(GROUPS[c].levels) & set(v.unique())) or v.nunique(),
            levels_per_slot_min=int(per_slot.min()), levels_per_slot_max=int(per_slot.max()),
            var_explained_by_slot=round(ss_bet / ss_tot, 3) if ss_tot else 0.0,
            r_with_block_parity=round(float(np.corrcoef(v, p.parity)[0, 1]), 3)
            if v.std() else 0.0,
            verdict=("CONFOUNDED with time of day" if ss_tot and ss_bet / ss_tot > 0.3
                     else "balanced"),
        ))
    return pd.DataFrame(rows)


def slot_risk(bt: pd.DataFrame, sens: pd.DataFrame, limit: float = 50.0) -> pd.DataFrame:
    """Per start-hour risk of breaching ``limit``, measured on the campaign itself.

    Used to pick a safe slot for a re-run: a block that overheated once will
    overheat again in the same slot, so the question is which slots never do.
    """
    rows = []
    for t0, b in bt.iterrows():
        if b.phase not in ("A", "B", "C", "D"):
            continue
        seg = sens.loc[t0:t0 + pd.Timedelta(hours=TREATMENT_HOURS), "temp_centro"].dropna()
        if not len(seg):
            continue
        rows.append(dict(hour=t0.hour, peak=float(seg.max()),
                         over=bool((seg > limit).any())))
    d = pd.DataFrame(rows)
    out = d.groupby("hour").agg(blocks=("peak", "size"), median_peak=("peak", "median"),
                                max_peak=("peak", "max"), pct_over=("over", "mean"))
    out["median_peak"] = out.median_peak.round(1)
    out["max_peak"] = out.max_peak.round(1)
    out["pct_over"] = (100 * out.pct_over).round(0)
    out["safe"] = out.max_peak <= limit
    return out.reset_index()


# --------------------------------------------------------------------------- #
# Safety rules
# --------------------------------------------------------------------------- #

def annotate_safety(res: pd.DataFrame, sens: pd.DataFrame,
                    block_col: str = "block") -> pd.DataFrame:
    """Add ``peak_temp`` and ``safety_hot`` to a per-block result table.

    A block that breached the 45 °C limit had its program legitimately overridden
    by the safety rules (section 8), so a level mismatch there is the safety
    system working, not the protocol being broken. Every mismatch table should be
    read with this column beside it.
    """
    res = res.copy()
    tcols = [c for c in ("temp_alta", "temp_centro", "temp_baja") if c in sens]
    tmax = sens[tcols].max(axis=1)
    peaks, hot = [], []
    for t0 in res[block_col]:
        if pd.isna(t0):
            peaks.append(np.nan); hot.append(None); continue
        seg = tmax.loc[t0:t0 + pd.Timedelta(hours=TREATMENT_HOURS)]
        p = float(seg.max()) if len(seg) else np.nan
        peaks.append(round(p, 1) if np.isfinite(p) else np.nan)
        hot.append(bool(np.isfinite(p) and p > SAFETY["overtemp_c"]))
    res["peak_temp"] = peaks
    res["safety_hot"] = hot
    return res


def check_safety(sens: pd.DataFrame, act: pd.DataFrame, fog: pd.Series) -> pd.DataFrame:
    """Find the events section 8 covers and check the prescribed reaction.

    Overtemperature: any minute with an interior temperature above the limit
    should be followed, within 15 min, by windows at 100 %, SHADE at 0 and
    LIGHTS at 0. Condensation: RH pinned at 100 % for more than 20 min while the
    fog is spraying should be followed by the fog going off.
    """
    rows = []
    tcols = [c for c in ("temp_alta", "temp_centro", "temp_baja") if c in sens]
    tmax = sens[tcols].max(axis=1)
    hot = tmax > SAFETY["overtemp_c"]

    roof_cols = [c for c in act.columns if c.startswith("VENT_ROOF::")]
    side_cols = [c for c in act.columns if c.startswith("VENT_SIDE::")]
    shade_cols = [c for c in act.columns if c.startswith("SHADE::")]
    light_cols = [c for c in act.columns if c.startswith("LIGHTS::")]
    roof = act[roof_cols].median(axis=1)
    side = act[side_cols].median(axis=1)
    shade = act[shade_cols].median(axis=1)
    lights = act[light_cols].median(axis=1)

    grp = (hot != hot.shift()).cumsum()[hot]
    for _, idx in hot[hot].groupby(grp):
        t0, t1 = idx.index[0], idx.index[-1]
        # Judge the reaction over the whole episode plus a short tail, not just
        # the first minutes: the rule is satisfied if the vents ever reach 100 %
        # while it is too hot, however late.
        w1 = t1 + pd.Timedelta(minutes=30)
        r_seg, s_seg = roof.loc[t0:w1], side.loc[t0:w1]
        sh_seg, li_seg = shade.loc[t0:w1], lights.loc[t0:w1]
        roof_max = float(r_seg.max()) if len(r_seg.dropna()) else np.nan
        opened = r_seg[r_seg >= 100]
        delay = round((opened.index[0] - t0).total_seconds() / 60) if len(opened) else np.nan

        if not np.isfinite(roof_max):
            react = "no actuator data"
        elif not len(opened):
            react = "no reaction (vents never opened)"
        elif delay <= 30:
            react = "as prescribed"
        else:
            react = f"late ({delay:.0f} min)"

        # Attribution caveat: the scheduled program opens the roof vents to 100 %
        # routinely, so an opening during a hot spell cannot be credited to the
        # safety rule in general. The unambiguous cases are the negatives - the
        # vents stayed shut through a breach, which the rule would never allow.
        attribution = ("clean evidence the rule did NOT act" if not len(opened)
                       else "ambiguous: opening may be the scheduled program")

        rows.append(dict(rule="overtemp", start=t0, end=t1,
                         minutes=int((t1 - t0).total_seconds() // 60) + 1,
                         peak=round(float(tmax.loc[t0:t1].max()), 1),
                         roof_max=roof_max,
                         side_max=float(s_seg.max()) if len(s_seg.dropna()) else np.nan,
                         shade_min=float(sh_seg.min()) if len(sh_seg.dropna()) else np.nan,
                         lights_max=float(li_seg.max()) if len(li_seg.dropna()) else np.nan,
                         open_delay_min=delay, reaction=react,
                         attribution=attribution))

    hcols = [c for c in ("hum_alta", "hum_centro", "hum_baja") if c in sens]
    if hcols:
        hmax = sens[hcols].max(axis=1)
        sat = hmax >= SAFETY["condensation_rh"]
        grp = (sat != sat.shift()).cumsum()[sat]
        for _, idx in sat[sat].groupby(grp):
            t0, t1 = idx.index[0], idx.index[-1]
            mins = int((t1 - t0).total_seconds() // 60) + 1
            if mins <= SAFETY["condensation_minutes"]:
                continue
            fog_during = float(fog.loc[t0:t1].mean()) if len(fog.loc[t0:t1]) else np.nan
            fog_after = float(fog.loc[t1:t1 + pd.Timedelta(minutes=15)].mean()) \
                if len(fog.loc[t1:t1 + pd.Timedelta(minutes=15)]) else np.nan
            rows.append(dict(rule="condensation", start=t0, end=t1, minutes=mins,
                             peak=round(float(hmax.loc[t0:t1].max()), 1),
                             fog_during_s_per_min=round(fog_during, 1) if np.isfinite(fog_during) else np.nan,
                             fog_after_s_per_min=round(fog_after, 1) if np.isfinite(fog_after) else np.nan,
                             reaction="fog off" if (np.isfinite(fog_after) and fog_after < 1)
                                      else ("fog still on" if np.isfinite(fog_after) else "no fog data")))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Step-response identification (the point of Phase A)
# --------------------------------------------------------------------------- #

TAU_BOUNDS = (2.0, 600.0)
GAIN_BOUND = 60.0


def fit_tau(sens: pd.DataFrame, onset: pd.Timestamp, signal: str = "temp_centro",
            hours: float = 4.0, detrend: bool = True) -> dict:
    """Fit a first-order step response and return tau, the gain, and diagnostics.

    Phase A exists to measure tau, which then sets the block duration for every
    later phase (protocol section 3).

    The catch, and why ``detrend`` defaults to True: in an empty greenhouse in
    August the **diurnal swing is larger than any treatment effect** (~20 °C over
    a day against a few °C from an actuator). Fitting the bare step

        y = y0 + K (1 - exp(-t/tau))

    to a 4 h window starting at 06:00 mostly fits sunrise, which drives tau and K
    to their bounds and makes the number meaningless. So the fitted model adds a
    linear background term

        y = y0 + c·t + K (1 - exp(-t/tau))

    which absorbs the ambient ramp and leaves the step to the exponential.
    ``bound_hit`` marks a fit that ran into its limits — those numbers must not be
    used regardless of R², because R² here mostly measures the trend, not the step.
    """
    from scipy.optimize import curve_fit

    seg = sens.loc[onset:onset + pd.Timedelta(hours=hours), signal].dropna()
    if len(seg) < 30:
        return dict(signal=signal, onset=onset, baseline=np.nan, gain=np.nan,
                    trend_per_h=np.nan, tau_min=np.nan, r2=np.nan, bound_hit=None,
                    note="not enough samples")
    t = (seg.index - seg.index[0]).total_seconds().values / 60.0
    y = seg.values
    y0 = float(y[:5].mean())
    K0 = float(np.clip(y[-1] - y0, -GAIN_BOUND * 0.9, GAIN_BOUND * 0.9)) or 1.0

    if detrend:
        def model(tt, K, tau, c):
            return y0 + c * tt + K * (1 - np.exp(-tt / max(tau, 1e-6)))
        p0 = [K0, 30.0, 0.0]
        lo = [-GAIN_BOUND, TAU_BOUNDS[0], -1.0]
        hi = [GAIN_BOUND, TAU_BOUNDS[1], 1.0]
    else:
        def model(tt, K, tau):
            return y0 + K * (1 - np.exp(-tt / max(tau, 1e-6)))
        p0 = [K0, 30.0]
        lo = [-GAIN_BOUND, TAU_BOUNDS[0]]
        hi = [GAIN_BOUND, TAU_BOUNDS[1]]

    try:
        p, _ = curve_fit(model, t, y, p0=p0, bounds=(lo, hi), maxfev=40000)
    except Exception as e:  # pragma: no cover
        return dict(signal=signal, onset=onset, baseline=round(y0, 2), gain=np.nan,
                    trend_per_h=np.nan, tau_min=np.nan, r2=np.nan, bound_hit=None,
                    note=f"fit failed: {e}")

    K, tau = float(p[0]), float(p[1])
    c = float(p[2]) if detrend else 0.0
    resid = y - model(t, *p)
    r2 = float(1 - resid.var() / y.var()) if y.var() > 0 else np.nan
    hit = (abs(abs(K) - GAIN_BOUND) < 1e-3 or abs(tau - TAU_BOUNDS[1]) < 1e-2
           or abs(tau - TAU_BOUNDS[0]) < 1e-3)

    # A time constant longer than half the observation window cannot be
    # identified from that window: over 4 h, anything past ~120 min is
    # indistinguishable from a straight line, and the fit just trades the step
    # off against the trend term (which shows up as an implausibly large gain).
    half_window = hours * 60 / 2
    plausible_gain = 15.0 if signal.startswith("temp") else 40.0
    reasons = []
    if hit:
        reasons.append("fit hit its bounds")
    if tau > half_window:
        reasons.append(f"tau > half the {hours:g} h window, not identifiable")
    if abs(K) > plausible_gain:
        reasons.append(f"|gain| {abs(K):.0f} implausible for one actuator")

    return dict(signal=signal, onset=onset, baseline=round(y0, 2), gain=round(K, 2),
                trend_per_h=round(c * 60, 2), tau_min=round(tau, 1), r2=round(r2, 3),
                bound_hit=bool(hit), usable=not reasons,
                note="UNUSABLE: " + "; ".join(reasons) if reasons else "")


def phase_a_dynamics(sens: pd.DataFrame, pa: pd.DataFrame,
                     detrend: bool = True) -> pd.DataFrame:
    """tau and gain for temperature and humidity at every located Phase-A step.

    Only rows with ``bound_hit == False`` and a decent R² should be used to set
    the block duration; the rest tell you the trial cannot be identified from
    this window, not that the greenhouse is slow.
    """
    rows = []
    seen = set()
    for _, r in pa.iterrows():
        if pd.isna(r.observed_at) or r.observed_at in seen:
            continue
        seen.add(r.observed_at)
        for sig in ("temp_centro", "hum_centro"):
            d = fit_tau(sens, r.observed_at, sig, detrend=detrend)
            d.update(trial=r.trial, group=r.group)
            rows.append(d)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out[["trial", "group", "signal", "onset", "baseline", "gain",
                "trend_per_h", "tau_min", "r2", "usable", "note"]]


def tau_summary(dyn: pd.DataFrame) -> dict:
    """The one number Phase A was supposed to deliver, plus what it implies.

    Protocol note 1 leaves the 2 h + 1 h block duration provisional "until tau is
    measured in Phase A". This answers it from the usable fits only.
    """
    ok = dyn[dyn.usable & (dyn.r2 > 0.8)] if len(dyn) else dyn
    t = ok[ok.signal.str.startswith("temp")] if len(ok) else ok
    if not len(t):
        return dict(n_usable=0, note="no usable fit; tau not measurable from this data")
    tau_max = float(t.tau_min.max())
    return dict(n_usable=int(len(t)),
                tau_median_min=round(float(t.tau_min.median()), 1),
                tau_min_min=round(float(t.tau_min.min()), 1),
                tau_max_min=round(tau_max, 1),
                treatment_in_tau=round(TREATMENT_HOURS * 60 / tau_max, 1),
                washout_in_tau=round((BLOCK_HOURS - TREATMENT_HOURS) * 60 / tau_max, 1),
                verdict=("2 h treatment is >=3 tau and 1 h washout >=3 tau: block "
                         "duration adequate"
                         if (TREATMENT_HOURS * 60 / tau_max >= 3
                             and (BLOCK_HOURS - TREATMENT_HOURS) * 60 / tau_max >= 3)
                         else "block duration too short for the measured tau: "
                              "lengthen blocks (protocol section 3)"))


# --------------------------------------------------------------------------- #
# Data quality
# --------------------------------------------------------------------------- #

def data_quality(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Per-sensor coverage, gaps, stuck stretches and out-of-range samples."""
    rows = []
    for name, f in SENSORS.items():
        s = _read_events(data_dir / f)
        lo, hi = SENSOR_RANGE[name]
        oor = int(((s < lo) | (s > hi)).sum())
        dt = s.index.to_series().diff().dt.total_seconds()
        gaps = dt[dt > 300]
        rows.append(dict(sensor=name, n=len(s),
                         start=s.index.min(), end=s.index.max(),
                         median_dt_s=round(float(dt.median()), 1),
                         n_gaps_gt5min=int(len(gaps)),
                         max_gap_min=round(float(gaps.max() / 60), 1) if len(gaps) else 0.0,
                         out_of_range=oor))
    q = pd.DataFrame(rows)
    return q.merge(detect_flatlines(data_dir), on="sensor", how="left")


def coverage_report(bt: pd.DataFrame) -> pd.DataFrame:
    """Per-phase block counts, expected vs present, and washout compliance."""
    rows = []
    for ph, (d0, d1) in PHASES.items():
        seg = bt[(bt.day >= d0) & (bt.day <= d1)]
        expected = (d1 - d0 + 1) * (24 // BLOCK_HOURS)
        names = [g.name for g in GROUPS.values()]
        # Phase A holds a single group ON for ~4 h on purpose, so the 3 h block
        # grid - and therefore the washout column - does not apply there.
        wash = np.nan if ph == "A" else (
            round(100 * float(seg["washout_clean"].mean()), 1) if len(seg) else np.nan)
        rows.append(dict(phase=ph, days=f"{d0}-{d1}",
                         dates=f"{(DAY1 + pd.Timedelta(days=d0-1)).date()} to {(DAY1 + pd.Timedelta(days=d1-1)).date()}",
                         blocks_expected=expected, blocks_present=len(seg),
                         blocks_all_off=int((seg[names].fillna(0).sum(axis=1) == 0).sum()),
                         mean_groups_active=round(float(seg["n_active"].mean()), 2) if len(seg) else np.nan,
                         washout_clean_pct=wash))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# One-call pipeline
# --------------------------------------------------------------------------- #

@dataclass
class Report:
    act: pd.DataFrame
    fog: pd.Series
    sens: pd.DataFrame
    blocks: pd.DataFrame
    quality: pd.DataFrame
    flatlines: pd.DataFrame
    status_codes: pd.DataFrame
    coverage: pd.DataFrame
    phase_a: pd.DataFrame
    dynamics: pd.DataFrame
    phase_b: pd.DataFrame
    phase_b_summary: dict
    phase_c: pd.DataFrame
    phase_c_summary: dict
    phase_d: pd.DataFrame
    phase_d_summary: dict
    safety: pd.DataFrame


def run_all(data_dir: Path = DATA_DIR, verbose: bool = True) -> Report:
    """Load everything and run every check. ~30-60 s on the full campaign."""
    say = print if verbose else (lambda *a, **k: None)
    say("loading actuators ...")
    act = load_actuators(data_dir)
    say("decoding fog duty cycle ...")
    fog = load_fog_duty(data_dir)
    say("loading sensors ...")
    sens = load_sensors(data_dir)
    say("recovering per-block levels ...")
    bt = block_table(act, fog)
    say("checking phases ...")
    pa = check_phase_a(act, fog)
    pb, pbs = check_phase_b(bt)
    pc, pcs = check_phase_c(bt)
    pdz, pds = check_phase_d(bt)
    # A block that breached 45 C had its program legitimately overridden by the
    # safety rules, so every mismatch table carries that context.
    pb = annotate_safety(pb, sens)
    pc = annotate_safety(pc, sens)
    pdz = annotate_safety(pdz, sens)
    say("fitting step responses ...")
    dyn = phase_a_dynamics(sens, pa)
    say("scanning safety rules ...")
    saf = check_safety(sens, act, fog)
    return Report(act=act, fog=fog, sens=sens, blocks=bt,
                  quality=data_quality(data_dir), flatlines=detect_flatlines(data_dir),
                  status_codes=count_status_codes(data_dir),
                  coverage=coverage_report(bt), phase_a=pa, dynamics=dyn,
                  phase_b=pb, phase_b_summary=pbs, phase_c=pc, phase_c_summary=pcs,
                  phase_d=pdz, phase_d_summary=pds, safety=saf)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

# Colour-blind-safe categorical palette, one hue per actuator group.
GROUP_COLORS = {
    "LIGHTS":    "#B8860B",
    "FOG":       "#2E86AB",
    "VENT_ROOF": "#1B7F5F",
    "VENT_SIDE": "#7FB069",
    "SHADE":     "#8E5572",
    "RECIRC":    "#D97706",
    "RECIRC2":   "#A63D40",
}


def plot_campaign_overview(rep: Report, figsize=(15, 9)):
    """Whole campaign at a glance: the 7 actuator lanes over interior climate."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    names = list(GROUPS)
    fig, axes = plt.subplots(len(names) + 2, 1, figsize=figsize, sharex=True,
                             gridspec_kw=dict(height_ratios=[2, 1.4] + [1] * len(names)))

    t = rep.sens.index
    ax = axes[0]
    for c, lab, col in (("temp_alta", "high", "#F2B5A0"), ("temp_centro", "centre", "#C1443E"),
                        ("temp_baja", "low", "#7A2E2B")):
        if c in rep.sens:
            ax.plot(t, rep.sens[c], lw=0.5, color=col, label=lab)
    ax.axhline(SAFETY["overtemp_c"], color="#C1443E", ls=":", lw=1)
    ax.text(t[10], SAFETY["overtemp_c"] + 0.6, "safety limit 45 °C", fontsize=7, color="#C1443E")
    ax.set_ylabel("T (°C)", fontsize=8)
    ax.legend(fontsize=6, ncol=3, frameon=False, loc="lower right")
    ax.set_title("Osiris greenhouse — August 2026 campaign: commanded actuator levels vs interior climate",
                 fontsize=11, loc="left", pad=16)

    ax = axes[1]
    if "hum_centro" in rep.sens:
        ax.plot(t, rep.sens["hum_centro"], lw=0.5, color="#2E86AB")
    ax.set_ylabel("RH (%)", fontsize=8)

    for ax, g in zip(axes[2:], names):
        lv = rep.blocks[g]
        ax.step(rep.blocks.index, lv, where="post", lw=0.9, color=GROUP_COLORS[g])
        ax.fill_between(rep.blocks.index, 0, lv.fillna(0), step="post",
                        color=GROUP_COLORS[g], alpha=0.28)
        ax.set_ylabel(g, fontsize=7, rotation=0, ha="right", va="center")
        ax.set_yticks([0, max(GROUPS[g].levels)])
        ax.tick_params(labelsize=6)

    # phase bands
    for ax in axes:
        for ph, (d0, d1) in PHASES.items():
            a = DAY1 + pd.Timedelta(days=d0 - 1)
            b = DAY1 + pd.Timedelta(days=d1)
            ax.axvspan(a, b, color="#000000", alpha=0.04 if ph in "BD" else 0.0, lw=0)
            ax.axvline(a, color="#999999", lw=0.6, ls="--")
    # Phase labels go in the margin above the top axes, where nothing collides.
    import matplotlib.transforms as mtransforms
    tr = mtransforms.blended_transform_factory(axes[0].transData, axes[0].transAxes)
    for ph, (d0, d1) in PHASES.items():
        mid = DAY1 + pd.Timedelta(days=(d0 + d1) / 2 - 1)
        axes[0].text(mid, 1.03, f"Phase {ph}", transform=tr, ha="center", va="bottom",
                     fontsize=8, color="#555555", clip_on=False)

    # Mark the simultaneous sensor failure: after this instant the high/low
    # temperature, high/low humidity, PAR and radiation channels are dead, so
    # only the centre traces below are real.
    dead = detect_flatlines()
    dead = dead[dead.dead]
    if len(dead):
        t_dead = dead.frozen_from.min()
        for ax in axes:
            ax.axvline(t_dead, color="#C1443E", lw=1.1, ls="-", alpha=0.6)
        axes[0].text(t_dead, 1.03, "6 sensors freeze", transform=tr, ha="left",
                     va="bottom", fontsize=7, color="#C1443E", clip_on=False)

    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    for lbl in axes[-1].get_xticklabels():
        lbl.set_rotation(0)
    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout(h_pad=0.15)
    return fig


def plot_block(rep: Report, t0, hours: float = 3.0, figsize=(11, 6)):
    """Zoom on one block: raw device positions plus the climate response."""
    import matplotlib.pyplot as plt

    t0 = pd.Timestamp(t0)
    t1 = t0 + pd.Timedelta(hours=hours)
    fig, (a0, a1, a2) = plt.subplots(3, 1, figsize=figsize, sharex=True,
                                     gridspec_kw=dict(height_ratios=[2, 1.2, 1.6]))

    s = rep.sens.loc[t0:t1]
    for c, col in (("temp_alta", "#F2B5A0"), ("temp_centro", "#C1443E"), ("temp_baja", "#7A2E2B")):
        if c in s:
            a0.plot(s.index, s[c], lw=1.1, color=col, label=c.replace("temp_", ""))
    a0.set_ylabel("T (°C)", fontsize=8)
    a0.legend(fontsize=6, frameon=False, ncol=3)
    a0.set_title(f"Block {t0:%d %b %Y %H:%M} local  ·  day {campaign_day(t0)}  ·  phase {phase_of(t0)}",
                 fontsize=10, loc="left")

    if "hum_centro" in s:
        a1.plot(s.index, s["hum_centro"], lw=1.1, color="#2E86AB")
    a1.set_ylabel("RH (%)", fontsize=8)

    seg = rep.act.loc[t0:t1]
    for c in seg.columns:
        g = c.split("::")[0]
        if seg[c].fillna(0).abs().sum() == 0:
            continue
        a2.step(seg.index, seg[c], where="post", lw=1.0, color=GROUP_COLORS[g], alpha=0.85,
                label=c.split("::")[1][:26])
    f = rep.fog.loc[t0:t1]
    if len(f) and f.fillna(0).sum() > 0:
        a2.step(f.index, f, where="post", lw=1.2, color=GROUP_COLORS["FOG"], ls="--",
                label="FOG s/min")
    a2.set_ylabel("device level", fontsize=8)
    a2.legend(fontsize=6, frameon=False, ncol=3)

    for ax in (a0, a1, a2):
        ax.axvline(t0 + pd.Timedelta(hours=TREATMENT_HOURS), color="#999999", ls="--", lw=0.9)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    a2.text(t0 + pd.Timedelta(hours=TREATMENT_HOURS + 0.05), a2.get_ylim()[1] * 0.9,
            "washout", fontsize=7, color="#777777")
    fig.tight_layout(h_pad=0.2)
    return fig


def plot_step_fit(rep: Report, onset, signal="temp_centro", hours=4.0, figsize=(7.5, 3.6)):
    """The measured step response and its fitted first-order model."""
    import matplotlib.pyplot as plt

    onset = pd.Timestamp(onset)
    d = fit_tau(rep.sens, onset, signal, hours)
    seg = rep.sens.loc[onset:onset + pd.Timedelta(hours=hours), signal].dropna()
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(seg.index, seg.values, lw=1.0, color="#C1443E", label="measured")
    if np.isfinite(d["tau_min"]):
        tt = (seg.index - seg.index[0]).total_seconds().values / 60
        ax.plot(seg.index, d["baseline"] + d["gain"] * (1 - np.exp(-tt / d["tau_min"])),
                lw=1.6, color="#1B7F5F", ls="--", label="first-order fit")
        ax.axvline(onset + pd.Timedelta(minutes=d["tau_min"]), color="#1B7F5F", lw=0.8, ls=":")
        ax.set_title(f"{signal}  ·  τ = {d['tau_min']:.0f} min  ·  gain = {d['gain']:+.2f}  ·  R² = {d['r2']:.2f}",
                     fontsize=9, loc="left")
    ax.set_ylabel(signal, fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def plot_dose_response(rep: Report, group: str, figsize=(7.5, 4.0)):
    """Mean interior climate reached per commanded level (Phase-B style summary)."""
    import matplotlib.pyplot as plt

    bt = rep.blocks
    names = [g.name for g in GROUPS.values()]
    solo = bt[(bt[names].fillna(0) > 0).sum(axis=1).le(1)]
    rows = []
    for t0, r in solo.iterrows():
        lv = r[group]
        if not np.isfinite(lv):
            continue
        w = rep.sens.loc[t0 + pd.Timedelta(minutes=60):t0 + pd.Timedelta(hours=TREATMENT_HOURS)]
        if w.empty:
            continue
        rows.append(dict(level=lv, temp=w["temp_centro"].mean(), hum=w["hum_centro"].mean()))
    d = pd.DataFrame(rows)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=figsize)
    if not d.empty:
        for ax, col, lab, c in ((a0, "temp", "T centre (°C)", "#C1443E"),
                                (a1, "hum", "RH centre (%)", "#2E86AB")):
            m = d.groupby("level")[col].agg(["mean", "std", "count"])
            ax.errorbar(m.index, m["mean"], yerr=m["std"].fillna(0), marker="o", lw=1.4,
                        capsize=3, color=c)
            ax.set_xlabel(f"{group} level", fontsize=8)
            ax.set_ylabel(lab, fontsize=8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    fig.suptitle(f"{group}: observed climate by commanded level (single-group blocks only)",
                 fontsize=9.5)
    fig.tight_layout()
    return fig


__all__ = [n for n in dir() if not n.startswith("_")]
