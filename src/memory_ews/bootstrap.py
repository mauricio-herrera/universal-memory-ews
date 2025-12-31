from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .leadlag import lead_lag_corr


def _block_resample_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """
    Moving-block resampling: concatenate random contiguous blocks until length n, then truncate.
    This preserves short-range dependence approximately.
    """
    idx: list[int] = []
    block_len = int(max(1, block_len))
    while len(idx) < n:
        start = int(rng.integers(0, max(1, n - block_len + 1)))
        idx.extend(range(start, min(n, start + block_len)))
    return np.asarray(idx[:n], dtype=int)


def block_bootstrap_leadlag(
    a: pd.Series,
    b: pd.Series,
    max_lag: int,
    block_len: int,
    B: int,
    ci: tuple[float, float] = (2.5, 97.5),
    seed: int = 123,
) -> dict:
    """
    Block-bootstrap uncertainty for lead–lag curve r(lag)=corr(a_t, b_{t+lag}).

    Bootstrap is performed on the aligned (a,b) timeline at lag 0, resampling paired blocks,
    then recomputing lag correlations on the resampled arrays.

    Returns a dict with:
      lags, r_orig, r_lo, r_hi,
      best_lag, r_best,
      best_lag_ci, r_best_ci,
      n0 (aligned sample size used)
    """
    lags, r_orig, _ = lead_lag_corr(a, b, max_lag=max_lag)

    # aligned at lag 0
    df0 = pd.concat([a, b], axis=1).dropna()
    if len(df0) < 50:
        raise RuntimeError("Not enough overlap for bootstrap (need >=50).")

    a0 = df0.iloc[:, 0].to_numpy(dtype=float)
    b0 = df0.iloc[:, 1].to_numpy(dtype=float)
    n0 = len(df0)

    rng = np.random.default_rng(seed)
    r_boot = np.full((int(B), len(lags)), np.nan, float)

    for k in range(int(B)):
        idx = _block_resample_indices(n0, block_len=block_len, rng=rng)
        aa = a0[idx]
        bb = b0[idx]

        for j, L in enumerate(lags):
            # corr(aa_t, bb_{t+L})
            if L > 0:
                x = aa[:-L]
                y = bb[L:]
            elif L < 0:
                Lm = -L
                x = aa[Lm:]
                y = bb[:-Lm]
            else:
                x = aa
                y = bb

            if len(x) < 20:
                continue
            sx = np.nanstd(x)
            sy = np.nanstd(y)
            if sx == 0 or sy == 0:
                continue
            r_boot[k, j] = np.corrcoef(x, y)[0, 1]

    r_lo = np.nanpercentile(r_boot, ci[0], axis=0)
    r_hi = np.nanpercentile(r_boot, ci[1], axis=0)

    # best lag by max |r|
    j_best = int(np.nanargmax(np.abs(r_orig)))
    best_lag = int(lags[j_best])
    r_best = float(r_orig[j_best])

    best_lags_boot: list[int] = []
    r_best_boot: list[float] = []
    for k in range(int(B)):
        rr = r_boot[k, :]
        if np.all(~np.isfinite(rr)):
            continue
        jb = int(np.nanargmax(np.abs(rr)))
        best_lags_boot.append(int(lags[jb]))
        r_best_boot.append(float(rr[jb]))

    if len(best_lags_boot) < max(30, int(B) // 10):
        warnings.warn("Few valid bootstrap replicates for best-lag CI.", RuntimeWarning)

    lag_ci = (
        float(np.nanpercentile(best_lags_boot, ci[0])),
        float(np.nanpercentile(best_lags_boot, ci[1])),
    )
    rbest_ci = (
        float(np.nanpercentile(r_best_boot, ci[0])),
        float(np.nanpercentile(r_best_boot, ci[1])),
    )

    return dict(
        lags=lags,
        r_orig=r_orig,
        r_lo=r_lo,
        r_hi=r_hi,
        best_lag=best_lag,
        r_best=r_best,
        best_lag_ci=lag_ci,
        r_best_ci=rbest_ci,
        n0=int(n0),
    )


def plot_leadlag_with_ci(
    title_prefix: str,
    name_x: str,
    name_y: str,
    res: dict,
    block_len: int,
    B: int,
    out_path: str | None = None,
    show: bool = False,
):
    lags = res["lags"]
    r = res["r_orig"]
    lo = res["r_lo"]
    hi = res["r_hi"]

    best_lag = res["best_lag"]
    r_best = res["r_best"]
    lag_ci_lo, lag_ci_hi = res["best_lag_ci"]
    r_ci_lo, r_ci_hi = res["r_best_ci"]
    n0 = res["n0"]

    fig = plt.figure(figsize=(11, 4.8), constrained_layout=True)
    ax = plt.gca()
    ax.plot(lags, r, lw=2, label="r(lag) (original)")
    ax.fill_between(lags, lo, hi, alpha=0.18, label="block-bootstrap band")
    ax.axhline(0, lw=1, alpha=0.7)
    ax.axvline(0, lw=1, alpha=0.7)
    ax.axvline(best_lag, lw=1.5, alpha=0.85)

    ax.set_xlabel("lag (months)   [lag>0: x leads y]")
    ax.set_ylabel("Pearson r")
    ax.grid(True, alpha=0.25)

    ax.set_title(
        f"{title_prefix}: corr({name_x}_t, {name_y}_(t+lag)) with block-bootstrap CI\n"
        f"best lag={best_lag}m (CI[{lag_ci_lo:.0f},{lag_ci_hi:.0f}]), "
        f"r={r_best:.3f} (CI[{r_ci_lo:.3f},{r_ci_hi:.3f}]), "
        f"n={n0}, block={block_len}m, B={B}"
    )
    ax.legend(loc="upper left", frameon=True)

    if out_path:
        fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)
