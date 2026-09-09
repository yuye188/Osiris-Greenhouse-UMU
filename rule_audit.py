"""
rule_audit.py
=============
Audit the greenhouse controller's **own rule export** (``Reglas a repetir BD.xlsx``)
against the logged data.

Why this sits beside ``protocol_check.py``
------------------------------------------
``protocol_check.py`` compares the logs with the *protocol document*
(``REGLAS_EXPERIMENTO.pdf``): it works at the level of the 7 protocol groups, and
it has to infer which block belongs to which phase. The controller export is
better ground truth in three ways, so anything it covers should be checked here
instead:

* it names **every device individually** (``Ventana cenital oeste:50``), so a
  single window lagging its partner is visible;
* it carries the **exact scheduled window** (``FechaInicio`` in UTC alongside
  ``HoraInicio`` in local time — which is where the UTC+2 offset is confirmed
  outright rather than inferred);
* it carries the **guard** each rule ran under: ``Propiedad`` names the sensor,
  ``ValorObjetivo`` the threshold in tenths (500 → 50.0 °C).

That last column is the important one. A rule guarded on a sensor that has
stopped reporting cannot abort, so ``check_guard_health`` is the first thing to
run before trusting any note in the ``Nota repetición`` column.

Two kinds of failure, and only one is visible in the logs
---------------------------------------------------------
* **"N min +50º"** — the house breached the guard threshold during the block.
  Fully recomputable from the sensor data; ``audit_rules`` reproduces the minute
  counts.
* **"Fallo"** — the rule was not queued correctly. Detectable only when the rule
  commands a **non-zero** level: a rule that was supposed to set everything to 0
  and never ran looks exactly like one that ran perfectly. ``audit_rules`` marks
  those rows ``blind`` rather than pretending to verify them.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

import protocol_check as pk

#: Controller device name -> CSV in the experiment folder. Names are matched
#: case-insensitively and after collapsing whitespace, because the export is
#: inconsistent about "fog" vs "Fog".
DEVICE_FILES: dict[str, str] = {
    "luz artificial centro-este":   "Luz_artificial_centro-este_0.csv",
    "luz artificial centro-oeste":  "Luz_artificial_centro-oeste_0.csv",
    "luz artificial nor-este":      "Luz_artificial_nor-este_0.csv",
    "luz artificial nor-oeste":     "Luz_artificial_nor-oeste_0.csv",
    "luz artificial sur-este":      "Luz_artificial_sur-este_0.csv",
    "luz artificial sur-oeste":     "Luz_artificial_sur-oeste_0.csv",
    "fog":                          "Fog_0.csv",
    "ventana cenital este":         "Ventana_cenital_este_0.csv",
    "ventana cenital oeste":        "Ventana_cenital_oeste_0.csv",
    "ventana lateral norte":        "Ventana_lateral_norte_0.csv",
    "ventana lateral oeste":        "Ventana_lateral_oeste_0.csv",
    "ventana lateral sur":          "Ventana_lateral_sur_0.csv",
    "pantalla de sombreo cenital":  "Pantalla_de_sombreo_cenital_0.csv",
    "recirculador centro":          "Recirculador_centro_0.csv",
    "recirculador":                 "Recirculador_centro_0.csv",
    "calefactores":                 "Calefactores_2.csv",
}

#: Controller property name -> key in ``protocol_check.SENSORS``.
GUARD_SENSORS = {
    "temperatura interior baja (ºc)":   "temp_baja",
    "temperatura interior centro (ºc)": "temp_centro",
    "temperatura interior alta (ºc)":   "temp_alta",
}

#: Tolerance on a recovered device level, in that device's own units.
DEV_TOL = {"fog": 2.5}          # fog is a duty cycle; everything else is a position
DEFAULT_TOL = 6.0
LIGHT_TOL = 0.6


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _tol(device: str) -> float:
    n = _norm(device)
    if n in DEV_TOL:
        return DEV_TOL[n]
    if n.startswith("luz"):
        return LIGHT_TOL
    if n.startswith("recirculador") or n.startswith("calefactor"):
        return 0.01
    return DEFAULT_TOL


# --------------------------------------------------------------------------- #
# Loading the rule export
# --------------------------------------------------------------------------- #

def load_rules(xlsx: str | Path) -> pd.DataFrame:
    """Read the controller rule export into one row per rule.

    ``start`` / ``end`` are **local** times: ``FechaInicio`` is UTC and
    ``HoraInicio`` is the same instant in local time, so the pair is used to
    verify the offset instead of assuming it.
    """
    df = pd.read_excel(xlsx)
    out = []
    for _, r in df.iterrows():
        start_utc = pd.to_datetime(r["FechaInicio"]).tz_localize(None)
        end_utc = pd.to_datetime(r["FechaFin"]).tz_localize(None)
        start = start_utc + pk.UTC_OFFSET
        # offset implied by this row, from the local HoraInicio column
        hh = r["HoraInicio"]
        implied = (hh.hour - start_utc.hour) % 24 if hasattr(hh, "hour") else np.nan

        targets = {}
        for field in ("Sistemas", "GruposSistemas"):
            v = r.get(field)
            if isinstance(v, str):
                for part in v.split("|"):
                    if ":" not in part:
                        continue
                    dev, lv = part.rsplit(":", 1)
                    dev = _norm(dev)
                    if dev:
                        targets[dev] = float(lv)

        out.append(dict(
            name=str(r["Nombre"]).strip(),
            phase=str(r["Nombre"]).strip()[0],
            start=start, end=end_utc + pk.UTC_OFFSET,
            implied_offset_h=implied,
            guard_sensor=GUARD_SENSORS.get(_norm(r["Propiedad"])),
            guard_property=str(r["Propiedad"]).strip(),
            guard_threshold=float(r["ValorObjetivo"]) / 10.0,
            note=("" if pd.isna(r.get("Nota repetición")) else str(r["Nota repetición"]).strip()),
            targets=targets,
            n_devices=len(targets),
            all_zero=all(v == 0 for v in targets.values()) if targets else None,
        ))
    return pd.DataFrame(out)


def check_offset(rules: pd.DataFrame) -> dict:
    """Confirm the UTC->local offset from the export itself."""
    vals = rules.implied_offset_h.dropna().unique()
    return dict(implied_offsets=sorted(int(v) for v in vals),
                module_offset=int(pk.UTC_OFFSET.total_seconds() // 3600),
                agrees=len(vals) == 1 and int(vals[0]) == int(pk.UTC_OFFSET.total_seconds() // 3600))


# --------------------------------------------------------------------------- #
# The guard
# --------------------------------------------------------------------------- #

def check_guard_health(rules: pd.DataFrame, data_dir: Path = pk.DATA_DIR) -> pd.DataFrame:
    """For each rule, was the sensor its guard watches still reporting?

    A frozen channel keeps returning a plausible constant, so a guard on it
    silently stops protecting anything. This is why a run can look clean.
    """
    flat = pk.detect_flatlines(data_dir).set_index("sensor")
    rows = []
    for _, r in rules.iterrows():
        s = r.guard_sensor
        dead_from = flat.loc[s, "frozen_from"] if (s in flat.index and flat.loc[s, "dead"]) else pd.NaT
        live = pd.isna(dead_from) or r.start < dead_from
        rows.append(dict(name=r["name"], start=r.start, guard=r.guard_property,
                         threshold=r.guard_threshold, guard_dead_from=dead_from,
                         guard_live_at_rule=live,
                         note=r.note))
    return pd.DataFrame(rows)


def minutes_above(sens: pd.DataFrame, col: str, thresh: float,
                  t0: pd.Timestamp, t1: pd.Timestamp) -> float:
    seg = sens.loc[t0:t1, col].dropna() if col in sens else pd.Series(dtype=float)
    return float((seg > thresh).sum()) if len(seg) else np.nan


# --------------------------------------------------------------------------- #
# Device-level verification
# --------------------------------------------------------------------------- #

def _device_series(data_dir: Path, device: str, fog: pd.Series | None) -> pd.Series | None:
    n = _norm(device)
    if n == "fog":
        return fog
    f = DEVICE_FILES.get(n)
    if f is None:
        return None
    s = pk._read_events(data_dir / f)
    return s.where(s >= 0)


def observed_level(series: pd.Series, t0: pd.Timestamp, t1: pd.Timestamp,
                   is_fog: bool = False) -> float:
    """Sustained (time-weighted modal) level over ``[t0, t1)``, skipping the ramp."""
    w0 = t0 + pk.RAMP_SKIP
    if is_fog:
        seg = series.loc[(series.index >= w0) & (series.index < t1)]
        return float(seg.mean()) if len(seg) else np.nan
    grid = pd.date_range(w0, t1, freq="1min", inclusive="left")
    if not len(grid):
        return np.nan
    r = series.reindex(series.index.union(grid)).ffill().reindex(grid).dropna()
    return np.nan if r.empty else float(r.round(2).value_counts().idxmax())


def check_window(rule: pd.Series, t0: pd.Timestamp, cache: dict,
                 data_dir: Path = pk.DATA_DIR, fog: pd.Series | None = None,
                 sens: pd.DataFrame | None = None) -> dict:
    """Verify one rule's device targets over a window starting at ``t0``.

    Returns per-device agreement plus the temperature the block actually reached,
    on the rule's own guard sensor and on a sensor known to be alive.
    """
    t1 = t0 + pd.Timedelta(hours=pk.TREATMENT_HOURS)
    bad, ok = [], 0
    for dev, want in rule.targets.items():
        if dev not in cache:
            cache[dev] = _device_series(data_dir, dev, fog)
        s = cache[dev]
        if s is None:
            bad.append(f"{dev}: no data file")
            continue
        got = observed_level(s, t0, t1, is_fog=(_norm(dev) == "fog"))
        if np.isfinite(got) and abs(got - want) <= _tol(dev):
            ok += 1
        else:
            bad.append(f"{dev}: want {want:g}, got " +
                       ("no data" if not np.isfinite(got) else f"{got:g}"))

    res = dict(cells_ok=ok, cells=rule.n_devices, exact=(ok == rule.n_devices),
               mismatches="; ".join(bad))
    if sens is not None:
        g = rule.guard_sensor
        res["guard_min_above"] = minutes_above(sens, g, rule.guard_threshold, t0, t1) if g else np.nan
        res["centro_min_above"] = minutes_above(sens, "temp_centro", rule.guard_threshold, t0, t1)
        seg = sens.loc[t0:t1, "temp_centro"].dropna()
        res["centro_peak"] = round(float(seg.max()), 1) if len(seg) else np.nan
    return res


def audit_rules(rules: pd.DataFrame, data_dir: Path = pk.DATA_DIR,
                fog: pd.Series | None = None, sens: pd.DataFrame | None = None) -> pd.DataFrame:
    """Verify every rule over its **originally scheduled** window.

    ``blind`` marks a rule whose targets are all zero: if such a rule was never
    queued the logs look identical to a perfect execution, so a "Fallo" note on
    it can be neither confirmed nor contradicted here.
    """
    fog = fog if fog is not None else pk.load_fog_duty(data_dir)
    sens = sens if sens is not None else pk.load_sensors(data_dir)
    cache: dict = {}
    rows = []
    for _, r in rules.iterrows():
        d = dict(name=r["name"], start=r.start, note=r.note,
                 devices=r.n_devices, all_zero=r.all_zero)
        d.update(check_window(r, r.start, cache, data_dir, fog, sens))
        d["blind"] = bool(r.all_zero)
        d["note_kind"] = ("guard" if "+50" in r.note else
                          "queue" if r.note.lower().startswith("fallo") else "")
        d["note_minutes"] = (int(m.group(1)) if (m := re.match(r"(\d+)\s*min", r.note)) else np.nan)
        rows.append(d)
    out = pd.DataFrame(rows)
    out["guard_agrees"] = out.apply(
        lambda r: (abs(r.guard_min_above - r.note_minutes) <= 3
                   if np.isfinite(r.note_minutes) and np.isfinite(r.guard_min_above) else None),
        axis=1)
    return out


def audit_repeats(rules: pd.DataFrame, order: list[str], first_block: pd.Timestamp,
                  slots: dict[str, int] | None = None,
                  data_dir: Path = pk.DATA_DIR, fog: pd.Series | None = None,
                  sens: pd.DataFrame | None = None) -> pd.DataFrame:
    """Verify the repeat round, laying ``order`` onto the 3 h grid from ``first_block``.

    ``slots`` overrides how many 3 h blocks a rule occupies (the Phase-A step
    trials are held for ~4 h, so they take two). The alignment is asserted by the
    non-zero rules matching device-for-device; all-zero rules ride along.
    """
    fog = fog if fog is not None else pk.load_fog_duty(data_dir)
    sens = sens if sens is not None else pk.load_sensors(data_dir)
    slots = slots or {}
    by_name = rules.set_index("name")
    cache: dict = {}
    rows, k = [], 0
    for nm in order:
        t0 = first_block + pd.Timedelta(hours=pk.BLOCK_HOURS * k)
        k += slots.get(nm, 1)
        if nm not in by_name.index:
            rows.append(dict(name=nm, block=t0, note="(not in export)", exact=None))
            continue
        r = by_name.loc[nm].copy()
        r["name"] = nm
        d = dict(name=nm, block=t0, note=r.note, devices=r.n_devices, all_zero=r.all_zero)
        d.update(check_window(r, t0, cache, data_dir, fog, sens))
        d["blind"] = bool(r.all_zero)
        rows.append(d)
    return pd.DataFrame(rows)


def observed_vectors(data_dir: Path = pk.DATA_DIR, t0: str = "2026-09-03",
                     t1: str = "2026-09-08", fog: pd.Series | None = None) -> pd.DataFrame:
    """Per-device sustained level for every 3 h block in a window.

    Used to read the repeat round straight off the logs, independently of any
    assumed rule order.
    """
    fog = fog if fog is not None else pk.load_fog_duty(data_dir)
    devs = ["luz artificial centro-este", "fog", "ventana cenital este", "ventana cenital oeste",
            "ventana lateral norte", "ventana lateral oeste", "ventana lateral sur",
            "pantalla de sombreo cenital", "recirculador centro", "calefactores"]
    cache = {d: _device_series(data_dir, d, fog) for d in devs}
    grid = pd.date_range(pd.Timestamp(t0), pd.Timestamp(t1), freq=f"{pk.BLOCK_HOURS}h",
                         inclusive="left")
    rows = []
    for t in grid:
        row = {"block": t}
        for d in devs:
            row[d.replace("luz artificial centro-este", "luz (c-este)")
                 .replace("pantalla de sombreo cenital", "pantalla")
                 .replace("recirculador centro", "recirc")] = observed_level(
                     cache[d], t, t + pd.Timedelta(hours=pk.TREATMENT_HOURS),
                     is_fog=(d == "fog")) if cache[d] is not None else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("block")


__all__ = [n for n in dir() if not n.startswith("_")]
