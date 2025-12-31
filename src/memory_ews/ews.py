from __future__ import annotations
import numpy as np
import pandas as pd
from .preprocess import ewma

def rolling_ac1(x: pd.Series, window: int, min_periods: int) -> pd.Series:
    """
    Rolling lag-1 autocorrelation in a numerically safe way.
    """
    x = x.astype(float)
    x_lag = x.shift(1)

    def _corr(a, b):
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 3:
            return np.nan
        aa, bb = a[mask], b[mask]
        if np.nanstd(aa) == 0 or np.nanstd(bb) == 0:
            return np.nan
        return np.corrcoef(aa, bb)[0, 1]

    return x.rolling(window=window, min_periods=min_periods).apply(
        lambda arr: _corr(arr, x_lag.loc[arr.index].values), raw=False
    )

def compute_ews(z: pd.Series, win_months: int, min_periods: int, smooth_span: int) -> tuple[pd.Series, pd.Series]:
    """
    Classical EWS: rolling variance and rolling AC(1), then EWMA smoothing.
    """
    var = z.rolling(window=win_months, min_periods=min_periods).var()
    ac1 = rolling_ac1(z, window=win_months, min_periods=min_periods)
    return ewma(var, span=smooth_span), ewma(ac1, span=smooth_span)
