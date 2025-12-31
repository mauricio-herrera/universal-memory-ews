from __future__ import annotations
import numpy as np
import pandas as pd

def lead_lag_corr(a: pd.Series, b: pd.Series, max_lag: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    r(lag) = corr(a_t, b_{t+lag}), lag in [-max_lag, +max_lag]
    Convention:
      lag > 0: a leads b (predictive direction a -> future b)
      lag < 0: b leads a
    """
    lags = np.arange(-max_lag, max_lag + 1)
    rs = np.full_like(lags, np.nan, dtype=float)
    ns = np.zeros_like(lags, dtype=int)

    for i, L in enumerate(lags):
        bb = b.shift(-L)
        df = pd.concat([a, bb], axis=1).dropna()
        ns[i] = len(df)
        if ns[i] >= 20:
            rs[i] = df.iloc[:, 0].corr(df.iloc[:, 1])
    return lags, rs, ns

def best_lag_stats(lags: np.ndarray, rs: np.ndarray, ns: np.ndarray) -> tuple[int | float, float, int]:
    """
    Best lag by max |r|.
    """
    if np.all(~np.isfinite(rs)):
        return np.nan, np.nan, 0
    k = int(np.nanargmax(np.abs(rs)))
    return int(lags[k]), float(rs[k]), int(ns[k])
