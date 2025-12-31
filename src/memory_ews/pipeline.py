from __future__ import annotations
from pathlib import Path
import os
import numpy as np
import pandas as pd
import xarray as xr

from .config import PipelineConfig
from .dataset import guess_lat_lon_time, ensure_lon_180, spatial_mean_box
from .preprocess import ensure_monthly_index, seasonal_zscore_monthly
from .memory_index import compute_onset_from_dry, estimate_mu_baseline, estimate_triggered_and_M
from .ews import compute_ews
from .plots import timeseries_plot, phase_space_plot
from .leadlag import lead_lag_corr, best_lag_stats

def run_region(region: str, ds: xr.Dataset, cfg: PipelineConfig, out_dir: Path) -> tuple[pd.DataFrame, dict, list[dict]]:
    lat_name, lon_name, time_name = guess_lat_lon_time(ds)
    ds = ensure_lon_180(ds, lon_name=lon_name)

    if cfg.var_name not in ds:
        raise KeyError(f"Variable '{cfg.var_name}' not found. Available: {list(ds.data_vars)}")

    da = ds[cfg.var_name]
    bounds = cfg.regions[region]
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

    # memory index
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

    # EWS on z
    win = int(cfg.ews_win_years * 12)
    minp = int(cfg.ews_min_frac * win)
    ews_var, ews_ac1 = compute_ews(z=z, win_months=win, min_periods=minp, smooth_span=cfg.smooth_span_ews_months)

    # trailing onset rate (causal)
    onset_rate = onset.rolling(window=win, min_periods=minp, center=not cfg.onset_rate_trailing).mean()

    df = pd.concat([z, ews_var, ews_ac1, trig["M_s"], mu, onset_rate, dry, onset], axis=1)
    df.columns = ["z", "EWS_var", "EWS_ac1", "M", "mu", "onset_rate", "dry", "onset"]
    df = df.dropna()
    df["t_num"] = df.index.year + (df.index.month - 1) / 12.0

    out_dir.mkdir(parents=True, exist_ok=True)

    # save per-region df
    df.to_csv(out_dir / f"{region}_df_region.csv", index=True)

    # figures
    timeseries_plot(region, df, cfg.ews_win_years, cfg.dry_thr_z, cfg.tau_months, out_path=out_dir / f"{region}_timeseries_onset.png", show=False)
    phase_space_plot(region, df[["EWS_var", "EWS_ac1", "M", "onset_rate", "t_num"]], cfg.ews_win_years, marker_scale=cfg.marker_scale,
                     out_path=out_dir / f"{region}_phase_onset.png", show=False)

    # lead-lag summaries
    rows = []
    for xname in ["M", "EWS_var", "EWS_ac1"]:
        lags, rs, ns = lead_lag_corr(df[xname], df["onset_rate"], max_lag=cfg.max_lag_leadlag_months)
        best_lag, best_r, n_best = best_lag_stats(lags, rs, ns)
        rows.append(dict(region=region, x=xname, y="onset_rate", best_lag_months=best_lag, r_best=best_r, n_overlap=n_best, alpha=trig["alpha"]))

    return df, trig, rows

def run_all_regions(nc_path: Path, out_dir: Path, cfg: PipelineConfig) -> pd.DataFrame:
    cfg = cfg.with_default_regions()
    ds = xr.open_dataset(nc_path)

    all_rows = []
    for region in cfg.regions.keys():
        print(f"[INFO] Region: {region}")
        try:
            _df, _trig, rows = run_region(region, ds, cfg, out_dir)
            all_rows.extend(rows)
            print(f"[OK] {region} | alpha={_trig['alpha']:.6f} | n={len(_df)}")
        except Exception as ex:
            print(f"[FAIL] {region}: {ex}")

    summary = pd.DataFrame(all_rows)
    summary.to_csv(out_dir / "leadlag_summary_all_regions_onset_rate_trailing.csv", index=False)
    return summary
