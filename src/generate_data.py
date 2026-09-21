"""Generate a synthetic 30-day 4680 electrode-line dataset with a planted coating drift.
Data is SIMULATED to demonstrate method (SPC / OEE / DMAIC), not real Tesla data."""
import numpy as np, pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)      # fixed seed -> reproducible
DAYS, PPT, ICT = 30, 1260, 0.60      # days; planned min/day (21h); ideal sec/cell (~100/min)

downtime     = np.clip(rng.normal(255, 45, DAYS) - np.linspace(0, 30, DAYS), 110, None)
run_time     = PPT - downtime
availability = run_time / PPT
performance  = np.clip(rng.normal(0.84, 0.03, DAYS), 0.72, 0.93)

TARGET, USL, LSL = 20.0, 20.8, 19.2
load_mean = np.full(DAYS, TARGET); load_mean[15:] += np.linspace(0.08, 1.02, DAYS - 15)
loading = np.array([rng.normal(load_mean[d], 0.18, 5) for d in range(DAYS)])
xbar = loading.mean(axis=1)

over    = np.clip(xbar - TARGET, 0, None)          # heavier-than-target coating drives scrap
scrap   = np.clip(rng.normal(0.075, 0.007, DAYS) + over * 0.10, 0.01, 0.30)
quality = 1 - scrap
oee     = availability * performance * quality
actual  = (run_time * 60 / ICT) * performance
good    = actual * quality

daily = pd.DataFrame({"day": np.arange(1, DAYS+1), "downtime_min": downtime.round().astype(int),
    "availability": availability.round(4), "performance": performance.round(4),
    "quality": quality.round(4), "oee": oee.round(4), "coating_xbar": xbar.round(3),
    "scrap_pct": (scrap*100).round(2), "good_cells": good.round().astype(int)})
readings = pd.DataFrame([(d+1, g+1, round(float(loading[d][g]),3))
    for d in range(DAYS) for g in range(5)], columns=["day","subgroup","coating_loading_mgcm2"])

Path("data").mkdir(exist_ok=True)
daily.to_csv("data/daily_metrics.csv", index=False)
readings.to_csv("data/coating_readings.csv", index=False)
print(f"Generated: OEE {oee.mean():.1%} | A {availability.mean():.1%} "
      f"P {performance.mean():.1%} Q {quality.mean():.1%}")
