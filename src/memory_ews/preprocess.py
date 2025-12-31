from __future__ import annotations
import numpy as np
import pandas as pd

def ensure_monthly_index(time_values) -> pd.DatetimeIndex:
    """
    Force timestamps to monthly end (Period('M') → month-end timestamp).
    """
    t = pd.to_datetime(time_values)
    t = t.to_period("M").to_timestamp("M")
    return pd.DatetimeIndex(t)

def seasonal_zscore_monthly(x: pd.Series) -> pd.Series:
    """
    Seasonal (month-wise) z-score: z(t) = (x - mean_month) / std_month.
    """
    df = pd.DataFrame({"x": x, "m": x.index.month})
    mu = df.groupby("m")["x"].transform("mean")
    sd = df.groupby("m")["x"].transform("std").replace(0, np.nan)
    z = (df["x"] - mu) / sd
    return pd.Series(z.values, index=x.index)

def ewma(series: pd.Series, span: int) -> pd.Series:
    """
    Exponentially-weighted moving average with safe min_periods.
    """
    return series.ewm(span=span, adjust=False, min_periods=max(3, span // 3)).mean()
