from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
import xarray as xr

from memory_ews.config import PipelineConfig
from memory_ews.dataset import guess_lat_lon_time, ensure_lon_180, spatial_mean_box
from memory_ews.preprocess import ensure_monthly_index, seasonal_zscore_monthly
from memory_ews.memory_index import compute_onset_from_dry, estimate_mu_baseline, estimate_triggered_and_M
from memory_ews.ews import compute_ews
from memory_ews.bootstrap import block_bootstrap_leadlag, plot_leadlag_with_ci
from memory_ews.oos import oos_experiment_ridge


def build_region_df_from_nc(
    ds: xr.Dataset,
    region: str,
    cfg: PipelineConfig,
) -> tuple[pd.DataFrame, float]:
    """
    Recompute everything from NetCDF, producing a clean dataframe for the region:
      z, EWS_var, EWS_ac1, M (smoothed), mu, onset, onset_rate_trailing
    Returns (df, alpha).
    """
    cfg = cfg.with_default_regions()

    lat_name, lon_name, time_name = guess_lat_lon_time(ds)
    ds = ensure_lon_180(ds, lon_name=lon_name)

    if cfg.var_name not in ds:
        raise KeyError(f"Variable '{cfg.var_name}' not found. Available: {list(ds.data_vars)}")

    da = ds[cfg.var_name]
    bounds = cfg.regions[region]

    # area-weighted mean on box
    x = spatial_mean_box(da, lat_name=lat_name, lon_name=lon_name, bounds=bounds, area_weighted=True)

    t = ensure_monthly_index(x[time_name].values)
    s = pd.Series(x.values, index=t).sort_index().replace([np.inf, -np.inf], np.nan).dropna()

    if len(s) < cfg.min_valid_months:
        raise RuntimeError(f"{region}: not enough valid months after cleaning: {len(s)}")

    # anomalies + dry + onset-real
    z = seasonal_zscore_monthly(s).replace([np.inf, -np.inf], np.nan)
    dry = (z <= cfg.dry_thr_z).astype(float)
    onset = compute_onset_from_dry(dry)

    # baseline mu(t) on onset events
    slow_win = int(cfg.mu_slow_win_years * 12)
    mu = estimate_mu_baseline(onset, slow_win=slow_win, clip=cfg.mu_clip)

    # triggered + M(t)
    trig = estimate_triggered_and_M(
        e=onset,
        mu=mu,
        tau_months=cfg.tau_months,
        max_lag=cfg.kernel_max_lag_months,
        y_short_win=cfg.y_short_win_months,
        alpha_min=cfg.alpha_min,
        alpha_cap=cfg.alpha_cap,
        smooth_span_M=cfg.smooth_span_M_months,
    )
    alpha = float(trig["alpha"])

    # EWS on z
    win = int(cfg.ews_win_years * 12)
    minp = int(cfg.ews_min_frac * win)
    ews_var, ews_ac1 = compute_ews(z=z, win_months=win, min_periods=minp, smooth_span=cfg.smooth_span_ews_months)

    # trailing onset-rate (causal interpretation)
    onset_rate_tr = onset.rolling(window=win, min_periods=minp, center=not cfg.onset_rate_trailing).mean()

    df = pd.concat([z, ews_var, ews_ac1, trig["M_s"], mu, onset, onset_rate_tr], axis=1)
    df.columns = ["z", "EWS_var", "EWS_ac1", "M", "mu", "onset", "onset_rate_trailing"]
    df = df.dropna()

    if df.empty:
        raise RuntimeError(f"{region}: alignment produced empty dataframe.")

    return df, alpha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", required=True, help="Path to CHIRPS monthly netCDF")
    ap.add_argument("--out", default="outputs_memory_ews", help="Output directory")
    ap.add_argument("--region", default="Mediterranean", help="Region key in cfg.regions")

    # Bootstrap
    ap.add_argument("--bb_block_months", type=int, default=36)
    ap.add_argument("--bb_B", type=int, default=500)
    ap.add_argument("--bb_ci_lo", type=float, default=2.5)
    ap.add_argument("--bb_ci_hi", type=float, default=97.5)
    ap.add_argument("--bb_seed", type=int, default=123)

    # OOS
    ap.add_argument("--oos_splits", nargs="+", default=["2010-01-31", "2015-01-31"])
    ap.add_argument("--oos_horizons", nargs="+", type=int, default=[3, 12])
    ap.add_argument("--ridge_lambda", type=float, default=1e-2)

    # plotting
    ap.add_argument("--save_fig", action="store_true")
    ap.add_argument("--show_fig", action="store_true")

    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    cfg = PipelineConfig().with_default_regions()

    ds = xr.open_dataset(args.nc)
    region = args.region

    df, alpha = build_region_df_from_nc(ds, region=region, cfg=cfg)
    print(f"[OK] {region}: alpha={alpha:.6f} | df_len={len(df)}")

    # ------------------- (1) Block-bootstrap lead–lag -------------------
    rows = []
    max_lag = int(cfg.max_lag_leadlag_months)
    ci = (float(args.bb_ci_lo), float(args.bb_ci_hi))

    for xname in ["M", "EWS_var", "EWS_ac1"]:
        res = block_bootstrap_leadlag(
            a=df[xname],
            b=df["onset_rate_trailing"],
            max_lag=max_lag,
            block_len=int(args.bb_block_months),
            B=int(args.bb_B),
            ci=ci,
            seed=int(args.bb_seed),
        )

        fig_path = None
        if args.save_fig:
            fig_path = os.path.join(out_dir, f"{region}_leadlag_{xname}_vs_onset_rate_trailing_bootCI.png")

        plot_leadlag_with_ci(
            title_prefix=region,
            name_x=xname,
            name_y="onset_rate_trailing",
            res=res,
            block_len=int(args.bb_block_months),
            B=int(args.bb_B),
            out_path=fig_path,
            show=bool(args.show_fig),
        )

        rows.append(dict(
            region=region,
            x=xname,
            y="onset_rate_trailing",
            best_lag_months=int(res["best_lag"]),
            r_best=float(res["r_best"]),
            r_ci_lo=float(res["r_best_ci"][0]),
            r_ci_hi=float(res["r_best_ci"][1]),
            lag_ci_lo=float(res["best_lag_ci"][0]),
            lag_ci_hi=float(res["best_lag_ci"][1]),
            n_overlap=int(res["n0"]),
            block_months=int(args.bb_block_months),
            B=int(args.bb_B),
            alpha=float(alpha),
            dry_thr_z=float(cfg.dry_thr_z),
        ))

    df_bootsum = pd.DataFrame(rows)
    out_csv1 = os.path.join(out_dir, f"{region}_leadlag_bootstrap_summary_trailing.csv")
    df_bootsum.to_csv(out_csv1, index=False)
    print(f"[OK] Bootstrap lead–lag summary saved: {out_csv1}")
    print(df_bootsum)

    # ------------------- (2) Minimal OOS experiments -------------------
    feats = ["M", "EWS_var", "EWS_ac1"]
    target = "onset_rate_trailing"

    oos_rows = []
    for split_date in args.oos_splits:
        for h in args.oos_horizons:
            fig_path = None
            if args.save_fig:
                fig_path = os.path.join(out_dir, f"{region}_OOS_h{int(h)}m_split{split_date[:4]}_{split_date[5:7]}.png")

            met = oos_experiment_ridge(
                df=df,
                horizon=int(h),
                split_date=str(split_date),
                features=feats,
                target_col=target,
                ridge_lambda=float(args.ridge_lambda),
                out_path=fig_path,
                show=bool(args.show_fig),
            )
            oos_rows.append(dict(region=region, alpha=float(alpha), dry_thr_z=float(cfg.dry_thr_z), **met))

    df_oos = pd.DataFrame(oos_rows)
    out_csv2 = os.path.join(out_dir, f"{region}_OOS_summary_trailing.csv")
    df_oos.to_csv(out_csv2, index=False)
    print(f"[OK] OOS summary saved: {out_csv2}")
    print(df_oos)


if __name__ == "__main__":
    main()
