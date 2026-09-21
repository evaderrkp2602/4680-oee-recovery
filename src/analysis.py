"""DMAIC analysis on the synthetic 4680 line: OEE, SPC (X-bar/R), Cpk, Pareto,
the coating->scrap causal check, and a projected CAPA recovery. Writes results.md + figures."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

daily = pd.read_csv("data/daily_metrics.csv")
reads = pd.read_csv("data/coating_readings.csv")
Path("figures").mkdir(exist_ok=True)

# ---------- MEASURE: OEE ----------
A, P, Q = daily.availability.mean(), daily.performance.mean(), daily.quality.mean()
OEE = daily.oee.mean()

# ---------- ANALYZE: SPC X-bar/R (subgroup n=5), limits from in-control baseline (days 1-15) ----------
A2, D4, D3, d2 = 0.577, 2.114, 0.0, 2.326
sub = reads.pivot(index="day", columns="subgroup", values="coating_loading_mgcm2").values
base = sub[:15]
xbb  = base.mean()
Rbar = (base.max(1) - base.min(1)).mean()
UCL, LCL = xbb + A2*Rbar, xbb - A2*Rbar
sigma = Rbar / d2
USL, LSL = 20.8, 19.2
xbar = sub.mean(1)
cpk = lambda v: min(USL - v.mean(), v.mean() - LSL) / (3*v.std(ddof=1))
cpk_base, cpk_drift = cpk(base.ravel()), cpk(sub[22:].ravel())
ooc = [int(i+1) for i in range(30) if xbar[i] > UCL or xbar[i] < LCL]

# ---------- ANALYZE: coating -> scrap causal check ----------
r = np.corrcoef(daily.coating_xbar, daily.scrap_pct)[0,1]

# ---------- ANALYZE: Pareto ----------
loss = pd.DataFrame({"reason":["Coating loading out-of-spec (scrap)","Winder/jelly-roll tooling changes",
    "Formation & aging bay starvation","Calender roll adjust / breakdown","Electrode slitting web breaks",
    "EOL test station minor stops","Material (cathode) starvation","Other"],
    "loss_min":[3120,1980,1450,1210,760,540,430,310]}).sort_values("loss_min", ascending=False)
loss["cum_pct"] = (loss.loss_min.cumsum()/loss.loss_min.sum()*100).round(1)

# ---------- IMPROVE: projected recovery (close the quality gap; modest A/P ramp gains) ----------
A1, P1, Q1 = 0.83, 0.86, 0.98
OEE1 = A1*P1*Q1
gain = OEE1/OEE - 1

# ---------- results.md ----------
with open("results.md","w") as f:
    f.write(f"""# Results (auto-generated)

## Measure — current state (30-day mean)
| Metric | Value |
|---|---|
| **OEE** | **{OEE:.1%}** |
| Availability | {A:.1%} |
| Performance | {P:.1%} |
| Quality (first-pass yield) | {Q:.1%} (target 98%) |

## Analyze — SPC on coating areal loading (X-bar/R, n=5)
| Item | Value |
|---|---|
| Center line (baseline days 1-15) | {xbb:.2f} mg/cm2 |
| Control limits (UCL / LCL) | {UCL:.2f} / {LCL:.2f} |
| Spec limits (USL / LSL) | {USL} / {LSL} |
| **Cpk baseline (days 1-15)** | **{cpk_base:.2f}** (capable, >=1.33) |
| **Cpk drifted (days 23-30)** | **{cpk_drift:.2f}** (incapable) |
| First out-of-control day | {ooc[0]} |
| Out-of-control days | {len(ooc)} of 30 |
| Coating<->scrap correlation (r) | {r:.2f} |

## Improve — projected CAPA recovery
| Metric | Current | Target |
|---|---|---|
| Availability | {A:.1%} | {A1:.0%} |
| Performance | {P:.1%} | {P1:.0%} |
| Quality | {Q:.1%} | {Q1:.0%} |
| **OEE** | **{OEE:.1%}** | **{OEE1:.1%}** |
| Throughput vs same tooling/crew | — | **+{gain:.0%}** |

*Data simulated to demonstrate method; not Tesla data.*
""")

# ---------- figures ----------
plt.rcParams.update({"font.size":10,"axes.edgecolor":"#888","axes.grid":True,
    "grid.color":"#e5e5e5","figure.dpi":130})
BLUE,RED,GREEN,ORANGE,AMBER = "#2a78d6","#d5342a","#1a9e6f","#eb6834","#c98500"

# OEE trend
fig,ax=plt.subplots(figsize=(8,3.4))
ax.plot(daily.day, daily.oee*100, color=RED, lw=2.4, label="OEE")
ax.plot(daily.day, daily.availability*100, color=BLUE, lw=1.4, label="Availability")
ax.plot(daily.day, daily.performance*100, color=ORANGE, lw=1.4, label="Performance")
ax.plot(daily.day, daily.quality*100, color=GREEN, lw=1.4, label="Quality")
ax.set_title("OEE and its three loss buckets — 30 production days")
ax.set_xlabel("Production day"); ax.set_ylabel("%"); ax.legend(ncol=4, fontsize=8)
fig.tight_layout(); fig.savefig("figures/fig_oee_trend.png"); plt.close(fig)

# SPC chart
fig,ax=plt.subplots(figsize=(8,3.6))
colors=[RED if (v>UCL or v<LCL) else BLUE for v in xbar]
ax.plot(daily.day, xbar, color=BLUE, lw=1.6, zorder=1)
ax.scatter(daily.day, xbar, c=colors, s=28, zorder=2)
ax.axhline(xbb, color="#555", lw=1, label=f"CL {xbb:.2f}")
ax.axhline(UCL, color=RED, ls="--", lw=1.2, label=f"UCL/LCL")
ax.axhline(LCL, color=RED, ls="--", lw=1.2)
ax.axhline(USL, color=AMBER, ls=":", lw=1.3, label="USL/LSL (spec)")
ax.axhline(LSL, color=AMBER, ls=":", lw=1.3)
ax.axvspan(15.5,30.5,color=RED,alpha=0.05)
ax.set_title("X-bar control chart — coating areal loading (mg/cm2)")
ax.set_xlabel("Production day"); ax.set_ylabel("mg/cm2"); ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig("figures/fig_spc.png"); plt.close(fig)

# Pareto
fig,ax=plt.subplots(figsize=(8,4))
x=range(len(loss)); bars=[RED]+[BLUE]*(len(loss)-1)
ax.bar(x, loss.loss_min, color=bars)
ax.set_xticks(x); ax.set_xticklabels(loss.reason, rotation=38, ha="right", fontsize=7.5)
ax.set_ylabel("Loss (minutes)")
ax2=ax.twinx(); ax2.plot(x, loss.cum_pct, color=AMBER, marker="o", lw=1.8)
ax2.set_ylim(0,105); ax2.set_ylabel("Cumulative %"); ax2.grid(False)
ax.set_title("Pareto of line losses")
fig.tight_layout(); fig.savefig("figures/fig_pareto.png"); plt.close(fig)

print(f"OEE {OEE:.1%} | Q {Q:.1%} | Cpk {cpk_base:.2f}->{cpk_drift:.2f} | "
      f"1st OOC day {ooc[0]} | r={r:.2f} | target OEE {OEE1:.1%} (+{gain:.0%})")
print("Wrote results.md and 3 figures.")
