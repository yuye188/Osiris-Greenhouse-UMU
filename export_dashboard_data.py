"""
export_dashboard_data.py
========================
Run every protocol check and write the results twice:

* ``protocol check output/*.csv``  — one CSV per result table, for diffing
  against a re-run after the next campaign;
* ``protocol check output/dash.json`` — one compact JSON bundle that backs the
  HTML audit dashboard.

Usage (from the repo root):

    python export_dashboard_data.py
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

import protocol_check as pk
import rule_audit as ra

OUT = "protocol check output"
RULES_XLSX = "Reglas a repetir BD.xlsx"
PROG_CSV = "fase_D_programa.csv"

#: Order the 22 repeat rules were laid onto the 3 h grid from 4 Sep 00:00.
#: Derived from the logs, not from a document: the non-zero rules match
#: device-for-device at these positions (18 of 22 exactly), which pins the
#: alignment, and the two Phase-A step trials occupy two blocks each because
#: they are held for 4 h.
REPEAT_ORDER = ["B3", "B13", "B18", "B21", "B22", "B24", "B29", "A9",
                "B30", "B38", "B46", "B53", "B54", "C30", "A10",
                "D05", "D06", "D25", "D61", "D80", "D82", "D95"]
REPEAT_SLOTS = {"A9": 2, "A10": 2}
REPEAT_START = pd.Timestamp("2026-09-04 00:00")


def _records(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    """JSON-safe records, with timestamps as short ISO strings."""
    d = df[[c for c in cols if c in df.columns]].copy()
    for c in d.columns:
        if pd.api.types.is_datetime64_any_dtype(d[c]):
            d[c] = d[c].dt.strftime("%Y-%m-%dT%H:%M")
    return json.loads(d.where(pd.notna(d), None).to_json(orient="records"))


def _plain(d: dict) -> dict:
    return {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in d.items()}


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    rep = pk.run_all()

    tables = [
        ("blocks", rep.blocks), ("quality", rep.quality), ("flatlines", rep.flatlines),
        ("status_codes", rep.status_codes), ("coverage", rep.coverage),
        ("phase_a", rep.phase_a), ("dynamics", rep.dynamics), ("phase_b", rep.phase_b),
        ("phase_c", rep.phase_c), ("phase_d", rep.phase_d), ("safety", rep.safety),
    ]
    for name, df in tables:
        df.to_csv(f"{OUT}/{name}.csv", index=(name == "blocks"))

    names = list(pk.GROUPS)
    bt = rep.blocks

    ot = rep.safety[rep.safety.rule == "overtemp"]
    otc = ot[ot.start.map(pk.campaign_day) <= 31]
    clean = otc[otc.attribution.str.startswith("clean")]
    cond = rep.safety[rep.safety.rule == "condensation"]

    blocks = [
        dict(t=t0.strftime("%Y-%m-%dT%H:%M"), d=int(r.day), p=r.phase, n=int(r.n_active),
             L=[None if not np.isfinite(r[n]) else float(r[n]) for n in names])
        for t0, r in bt.iterrows()
    ]

    h = rep.sens[["temp_centro", "hum_centro", "temp_alta", "temp_baja"]].resample("1h").mean().round(2)
    climate = [
        dict(t=i.strftime("%Y-%m-%dT%H:%M"),
             tc=None if pd.isna(v.temp_centro) else float(v.temp_centro),
             hc=None if pd.isna(v.hum_centro) else float(v.hum_centro),
             ta=None if pd.isna(v.temp_alta) else float(v.temp_alta),
             tb=None if pd.isna(v.temp_baja) else float(v.temp_baja))
        for i, v in h.iterrows()
    ]

    data = dict(
        meta=dict(day1=str(pk.DAY1.date()), groups=names,
                  levels={n: pk.GROUPS[n].levels for n in names},
                  phases={k: list(v) for k, v in pk.PHASES.items()},
                  offset_h=int(pk.UTC_OFFSET.total_seconds() // 3600),
                  generated=str(pd.Timestamp.now().round("min"))),
        coverage=_records(rep.coverage, list(rep.coverage.columns)),
        quality=_records(rep.quality, ["sensor", "n", "median_dt_s", "n_gaps_gt5min",
                                       "out_of_range", "longest_flat_min", "frozen_from",
                                       "frozen_value", "dead"]),
        status=_records(rep.status_codes, ["group", "file", "n_rows", "n_negative"]),
        phase_a=_records(rep.phase_a, ["trial", "group", "prescribed_at", "prescribed_level",
                                       "observed_at", "observed_level", "verdict", "detail"]),
        dynamics=_records(rep.dynamics, ["trial", "group", "signal", "gain", "trend_per_h",
                                         "tau_min", "r2", "usable", "note"]),
        tau=pk.tau_summary(rep.dynamics),
        phase_b=_records(rep.phase_b, ["seq", "group", "block", "prescribed", "observed",
                                       "match", "isolated", "washout_clean", "peak_temp",
                                       "safety_hot"]),
        phase_b_summary=_plain(rep.phase_b_summary),
        phase_c=_records(rep.phase_c, ["run", "block", "cells_ok", "exact", "mismatches",
                                       "peak_temp", "safety_hot"]),
        phase_c_summary=_plain(rep.phase_c_summary),
        phase_c_design=pk.PHASE_C,
        phase_d=_records(rep.phase_d, ["run", "block", "cells_ok", "exact", "mismatches",
                                       "peak_temp", "safety_hot"]),
        phase_d_summary=_plain(rep.phase_d_summary),
        phase_d_design=pk.PHASE_D_SAMPLE,
        safety=dict(
            n_ep=int(len(otc)), minutes=int(otc.minutes.sum()), peak=float(otc.peak.max()),
            n_clean_fail=int(len(clean)), clean_minutes=int(clean.minutes.sum()),
            n_ambiguous=int(otc.attribution.str.startswith("ambiguous").sum()),
            lights_on=int((clean.lights_max.fillna(0) > 0).sum()),
            by_day=[dict(d=int(k), m=int(v)) for k, v in
                    ot.assign(d=ot.start.map(pk.campaign_day)).groupby("d").minutes.sum().items()],
            episodes=_records(clean, ["start", "minutes", "peak", "roof_max", "side_max",
                                      "shade_min", "lights_max"]),
            cond=_records(cond, ["start", "minutes", "peak", "fog_during_s_per_min",
                                 "fog_after_s_per_min", "reaction"]),
            cond_counts=json.loads(cond.reaction.value_counts().to_json()) if len(cond) else {},
        ),
        blocks=blocks, climate=climate,
    )

    # ---- Phase D against its full generated programme (all 96 blocks) ----
    if os.path.exists(PROG_CSV):
        prog = pk.load_phase_d_programme(PROG_CSV)
        off = pk.find_programme_offset(rep.blocks, prog)
        full, fsum = pk.check_phase_d_full(rep.blocks, prog, off["offset_days"], sens=rep.sens)
        conf = pk.slot_confounding(prog)
        risk = pk.slot_risk(rep.blocks, rep.sens)
        full.to_csv(f"{OUT}/phase_d_full.csv", index=False)
        conf.to_csv(f"{OUT}/phase_d_slot_confounding.csv", index=False)
        risk.to_csv(f"{OUT}/slot_risk.csv", index=False)
        names7 = list(pk.GROUPS)
        data["phase_d_full"] = dict(
            summary=fsum, offset=off,
            blocks=_records(full, ["block_id", "n", "block", "cells_ok", "cells", "exact",
                                   "not_queued", "washout_clean", "mismatches",
                                   "centro_peak", "min_over_50"]),
            prescribed=[[float(r[c]) for c in names7] for _, r in prog.iterrows()],
            confounding=_records(conf, list(conf.columns)),
            risk=_records(risk, list(risk.columns)),
        )

    # ---- the controller's own rule export: failures and the repeat round ----
    if os.path.exists(RULES_XLSX):
        rules = ra.load_rules(RULES_XLSX)
        orig = ra.audit_rules(rules, fog=rep.fog, sens=rep.sens)
        reps = ra.audit_repeats(rules, REPEAT_ORDER, REPEAT_START, REPEAT_SLOTS,
                                fog=rep.fog, sens=rep.sens)
        guard = ra.check_guard_health(rules)
        orig.to_csv(f"{OUT}/rules_original.csv", index=False)
        reps.to_csv(f"{OUT}/rules_repeats.csv", index=False)
        guard.to_csv(f"{OUT}/rules_guard_health.csv", index=False)

        # peak on each sensor in every flagged block, so a "+50" note can be
        # corroborated even where the exact minute count cannot be reproduced
        raws = {k: pk._read_events(pk.DATA_DIR / pk.SENSORS[k])
                for k in ("temp_baja", "temp_centro", "temp_alta")}
        peaks = []
        for _, r in rules.iterrows():
            d = dict(name=r["name"], note=r.note)
            for k, s in raws.items():
                seg = s.loc[r.start:r.start + pd.Timedelta(hours=pk.TREATMENT_HOURS)]
                d[k] = round(float(seg.max()), 1) if len(seg) else None
            d["any_over_50"] = any(v is not None and v > 50
                                   for v in (d["temp_baja"], d["temp_centro"], d["temp_alta"]))
            peaks.append(d)

        # every unguarded 50 C breach: guard dead, centre sensor still live
        tc = raws["temp_centro"]
        dead_from = rep.flatlines.loc[rep.flatlines.sensor == "temp_baja", "frozen_from"].iloc[0]
        d0 = pd.Timestamp("2026-08-20 00:00")          # Phase D block 1, from the rule times
        rep_lbl = {}
        k = 0
        for nm in REPEAT_ORDER:
            for _ in range(REPEAT_SLOTS.get(nm, 1)):
                rep_lbl[k] = nm
                k += 1

        def _label(t):
            if d0 <= t < pd.Timestamp("2026-09-01"):
                n = int((t - d0).total_seconds() // 10800) + 1
                return f"D{n:02d}" if n <= 96 else "—"
            if t >= REPEAT_START:
                return "repeat " + rep_lbl.get(int((t - REPEAT_START).total_seconds() // 10800), "?")
            return "—"

        unguarded = []
        for t0 in pd.date_range(dead_from.floor("3h"), pd.Timestamp("2026-09-06 18:00"), freq="3h"):
            seg = tc.loc[t0:t0 + pd.Timedelta(hours=pk.TREATMENT_HOURS)]
            if not len(seg):
                continue
            m = pd.Series(seg[seg > 50].index).dt.floor("min").nunique()
            if m:
                unguarded.append(dict(block=t0.strftime("%Y-%m-%dT%H:%M"), rule=_label(t0),
                                      peak=round(float(seg.max()), 1), min_over_50=int(m)))

        # The two Phase-A step trials are not in the export (it covers B-D only),
        # so verify them on their own terms: held ~4 h, single group, no overheat.
        a_reps = []
        for nm, t0, grp in (("A9", pd.Timestamp("2026-09-04 21:00"), "RECIRC"),
                            ("A10", pd.Timestamp("2026-09-05 21:00"), "RECIRC2")):
            t_end = t0 + pd.Timedelta(hours=6)
            cols = [c for c in rep.act.columns if c.startswith(grp + "::")]
            on = (rep.act.loc[t0:t_end, cols].fillna(0) > 0).any(axis=1)
            others = []
            for g in pk.GROUPS.values():
                if g.name == grp:
                    continue
                lv = pk.group_level(rep.act, rep.fog, g, t0, t0 + pd.Timedelta(hours=4))
                if np.isfinite(lv["level"]) and lv["level"] > 0:
                    others.append(f"{g.name}={lv['level']:g}")
            tc_seg = rep.sens.loc[t0:t0 + pd.Timedelta(hours=4), "temp_centro"].dropna()
            a_reps.append(dict(name=nm, block=t0.strftime("%Y-%m-%dT%H:%M"), group=grp,
                               minutes_on=int(on.sum()), isolated=(not others),
                               other_active="; ".join(others),
                               t_min=round(float(tc_seg.min()), 1) if len(tc_seg) else None,
                               t_max=round(float(tc_seg.max()), 1) if len(tc_seg) else None,
                               over_50=int((tc_seg > 50).sum()) if len(tc_seg) else None))

        pre = rep.sens.loc[:dead_from, ["temp_centro", "temp_baja"]].dropna()
        hot = pre[pre.temp_baja > 48]
        data["rules"] = dict(
            guard_property=str(rules.guard_property.iloc[0]),
            guard_threshold=float(rules.guard_threshold.iloc[0]),
            guard_dead_from=dead_from.strftime("%Y-%m-%dT%H:%M"),
            n_guard_dead=int((~guard.guard_live_at_rule).sum()),
            offset_check=ra.check_offset(rules),
            original=_records(orig, ["name", "start", "note", "note_kind", "devices", "cells_ok",
                                     "cells", "exact", "blind", "mismatches", "guard_min_above",
                                     "note_minutes", "centro_peak"]),
            repeats=_records(reps, ["name", "block", "note", "devices", "cells_ok", "cells",
                                    "exact", "blind", "mismatches", "centro_peak",
                                    "centro_min_above"]),
            peaks=peaks, unguarded=unguarded, phase_a_repeats=a_reps,
            centro_minus_baja_hot=round(float((hot.temp_centro - hot.temp_baja).mean()), 2),
            repeat_order=REPEAT_ORDER, repeat_slots=REPEAT_SLOTS,
            repeat_start=REPEAT_START.strftime("%Y-%m-%dT%H:%M"),
            actuator_log_ends=max(
                pk._read_events(pk.DATA_DIR / f).index.max()
                for g in pk.GROUPS.values() for f in g.files).strftime("%Y-%m-%dT%H:%M"),
        )

    with open(f"{OUT}/dash.json", "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))

    # figures used by the notebook narrative
    pk.plot_campaign_overview(rep).savefig(f"{OUT}/fig_campaign_overview.png", dpi=110,
                                           bbox_inches="tight")
    pk.plot_block(rep, "2026-08-19 15:00").savefig(f"{OUT}/fig_block_run30_override.png",
                                                   dpi=110, bbox_inches="tight")
    pk.plot_step_fit(rep, "2026-08-05 18:04").savefig(f"{OUT}/fig_step_A5.png", dpi=110,
                                                      bbox_inches="tight")
    pk.plot_dose_response(rep, "VENT_ROOF").savefig(f"{OUT}/fig_dose_vent_roof.png", dpi=110,
                                                    bbox_inches="tight")

    print("phase A verdicts :", rep.phase_a.verdict.value_counts().to_dict())
    print("phase B          :", rep.phase_b_summary)
    print("phase C          :", rep.phase_c_summary)
    print("phase D          :", rep.phase_d_summary)
    print("tau              :", pk.tau_summary(rep.dynamics))
    print("overtemp (d1-31) :", len(otc), "episodes,", int(otc.minutes.sum()), "min, peak",
          otc.peak.max(), "| vents-shut:", len(clean), "episodes,", int(clean.minutes.sum()), "min")
    print("\nwritten to", OUT + "/:")
    print("  " + "\n  ".join(sorted(os.listdir(OUT))))


if __name__ == "__main__":
    main()
