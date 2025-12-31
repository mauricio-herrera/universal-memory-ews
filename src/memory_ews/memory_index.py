from __future__ import annotations
import numpy as np
import pandas as pd
from .preprocess import ewma

def compute_onset_from_dry(dry: pd.Series) -> pd.Series:
    """
    "Real onset" definition:
      onset[t] = 1 if dry[t]==1 and dry[t-1]==0 else 0
    """
    dry = dry.astype(float)
    prev = dry.shift(1).fillna(0.0)
    return ((dry >= 0.5) & (prev < 0.5)).astype(float)

def exp_kernel(tau_months: float, max_lag: int) -> np.ndarray:
    """
    Normalized exponential kernel over lags 1..max_lag.
    """
    k = np.arange(1, max_lag + 1, dtype=float)
    g = np.exp(-k / float(tau_months))
    s = g.sum()
    if s <= 0:
        raise ValueError("Kernel sum is non-positive.")
    return g / s

def causal_conv(events: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Causal discrete convolution: (events * kernel)(t) using past lags only.
    """
    conv_full = np.convolve(events, kernel, mode="full")
    return conv_full[: len(events)]

def estimate_mu_baseline(e: pd.Series, slow_win: int, clip=(1e-5, 0.25)) -> pd.Series:
    """
    Baseline rate mu(t) combining:
      - seasonal component (month-wise mean of events)
      - slow component (rolling mean)
    """
    season = e.groupby(e.index.month).mean().reindex(range(1, 13)).fillna(e.mean())
    p_season = e.index.month.map(season.to_dict()).astype(float)

    # keep center=True as in your working version
    p_slow = e.rolling(window=slow_win, min_periods=int(0.7 * slow_win), center=True).mean()

    season_norm = p_season / np.nanmean(p_season)
    mu = season_norm.values * p_slow.values

    mu = pd.Series(mu, index=e.index).astype(float).clip(lower=clip[0], upper=clip[1])
    return mu

def estimate_triggered_and_M(
    e: pd.Series,
    mu: pd.Series,
    tau_months: float,
    max_lag: int,
    y_short_win: int,
    alpha_min: float,
    alpha_cap: float,
    smooth_span_M: int,
) -> dict:
    """
    Build triggered component m(t)=alpha*(e * kernel), intensity lambda=mu+m,
    and memory share M(t)=m/lambda (clipped), plus a smoothed M_s.
    """
    kernel = exp_kernel(tau_months=tau_months, max_lag=max_lag)
    conv = causal_conv(e.values.astype(float), kernel)
    conv = pd.Series(conv, index=e.index)

    # short-window intensity proxy (kept as-is)
    y = e.rolling(window=y_short_win, min_periods=max(3, y_short_win // 2), center=True).mean()

    mask = np.isfinite(y.values) & np.isfinite(mu.values) & np.isfinite(conv.values)
    if mask.sum() < 50 or np.nanvar(conv.values[mask]) < 1e-10:
        alpha = alpha_min
    else:
        num = np.dot((y.values[mask] - mu.values[mask]), conv.values[mask])
        den = np.dot(conv.values[mask], conv.values[mask]) + 1e-12
        alpha = max(0.0, num / den)

        if alpha < alpha_min:
            excess = np.nanmean(y.values[mask] - mu.values[mask])
            denom = np.nanmean(conv.values[mask]) + 1e-12
            alpha = max(alpha_min, excess / denom)

    alpha = float(np.clip(alpha, alpha_min, alpha_cap))

    m = alpha * conv
    lam = (mu + m).clip(lower=1e-8)
    M = (m / lam).clip(lower=0.0, upper=1.0)
    M_s = ewma(M, span=smooth_span_M)

    return dict(alpha=alpha, conv=conv, m=m, lam=lam, M=M, M_s=M_s)
