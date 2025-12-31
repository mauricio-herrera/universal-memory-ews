from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def decade_label(dt: pd.Timestamp) -> str:
    y = dt.year
    return f"{(y // 10) * 10}s"

def phase_space_plot(region: str, df: pd.DataFrame, ews_win_years: int, marker_scale=(20, 350), out_path=None, show=False):
    r = df["onset_rate"].values.astype(float)
    r_min, r_max = np.nanpercentile(r, 5), np.nanpercentile(r, 95)
    r_sc = (r - r_min) / (r_max - r_min + 1e-12)
    smin, smax = marker_scale
    sizes = smin + (smax - smin) * np.clip(r_sc, 0, 1)

    dec = df.index.to_series().apply(decade_label)
    g = df.groupby(dec).mean(numeric_only=True)
    order = sorted(g.index, key=lambda x: int(x[:-1]))
    g = g.loc[order]

    fig = plt.figure(figsize=(14, 5), constrained_layout=True)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.045])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    cmap = plt.cm.viridis
    sc1 = ax1.scatter(df["EWS_var"], df["M"], c=df["t_num"], s=sizes, cmap=cmap,
                      alpha=0.85, edgecolors="white", linewidths=0.3)
    sc2 = ax2.scatter(df["EWS_ac1"], df["M"], c=df["t_num"], s=sizes, cmap=cmap,
                      alpha=0.85, edgecolors="white", linewidths=0.3)

    ax1.set_title(f"M(t) vs EWS variance (rolling {ews_win_years}y)")
    ax2.set_title(f"M(t) vs EWS AC(1) (rolling {ews_win_years}y)")
    ax1.set_xlabel("EWS variance")
    ax2.set_xlabel("EWS AC(1)")
    ax1.set_ylabel("Memory index M(t) (endogenous share)")
    ax2.set_ylabel("Memory index M(t) (endogenous share)")
    ax1.grid(True, alpha=0.25)
    ax2.grid(True, alpha=0.25)

    ax1.plot(g["EWS_var"], g["M"], "-k", lw=2, alpha=0.85, zorder=5)
    ax2.plot(g["EWS_ac1"], g["M"], "-k", lw=2, alpha=0.85, zorder=5)
    ax1.scatter(g["EWS_var"], g["M"], s=200, facecolor="white", edgecolor="black", lw=2, zorder=6)
    ax2.scatter(g["EWS_ac1"], g["M"], s=200, facecolor="white", edgecolor="black", lw=2, zorder=6)
    for lab, row in g.iterrows():
        ax1.text(row["EWS_var"], row["M"], lab, fontsize=11, weight="bold")
        ax2.text(row["EWS_ac1"], row["M"], lab, fontsize=11, weight="bold")

    cb = fig.colorbar(sc2, cax=cax)
    cax.set_ylabel("Time (year + month/12)")

    fig.suptitle(
        f"{region}: regime geometry between EWS and universal memory index M(t)\n"
        f"Point size ∝ ONSET rate (rolling {ews_win_years}y; trailing)",
        fontsize=14
    )

    if out_path:
        fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)

def timeseries_plot(region: str, df: pd.DataFrame, ews_win_years: int, dry_thr_z: float, tau_months: float, out_path=None, show=False):
    fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True, constrained_layout=True)

    axes[0].plot(df.index, df["z"], lw=1.0, color="black", alpha=0.8, label="z anomaly (SPI-like)")
    axes[0].axhline(0, ls="--", lw=1, color="gray", alpha=0.7)
    axes[0].axhline(dry_thr_z, ls=":", lw=1, color="gray", alpha=0.7)
    axes[0].set_ylabel("z (seasonal standardized)")
    axes[0].legend(loc="upper right", frameon=True)
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(df.index, df["EWS_var"], lw=2.0, label=f"EWS variance ({ews_win_years}y)")
    ax1b = axes[1].twinx()
    ax1b.plot(df.index, df["EWS_ac1"], lw=2.0, label=f"EWS AC(1) ({ews_win_years}y)")
    axes[1].set_ylabel("EWS variance")
    ax1b.set_ylabel("EWS AC(1)")
    axes[1].grid(True, alpha=0.2)
    lines = axes[1].get_lines() + ax1b.get_lines()
    axes[1].legend(lines, [l.get_label() for l in lines], loc="upper right", frameon=True)

    axes[2].plot(df.index, df["M"], lw=2.5, label="M(t) (smoothed)")
    axes[2].plot(df.index, df["mu"], lw=2.0, alpha=0.9, label="μ(t) baseline onset rate")
    ax2b = axes[2].twinx()
    ax2b.plot(df.index, df["onset_rate"], lw=2.0, ls="--", label=f"ONSET rate (rolling {ews_win_years}y; trailing)")
    axes[2].set_ylabel("Memory / baseline")
    ax2b.set_ylabel("Onset rate (events/month)")
    axes[2].grid(True, alpha=0.2)
    lines2 = axes[2].get_lines() + ax2b.get_lines()
    axes[2].legend(lines2, [l.get_label() for l in lines2], loc="upper right", frameon=True)

    fig.suptitle(
        f"{region}: anomalies, EWS, and universal memory index M(t)\n"
        f"EWSwin={ews_win_years}y | dry thr z={dry_thr_z} | onset-real | tau={tau_months}mo",
        fontsize=14
    )

    if out_path:
        fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)
