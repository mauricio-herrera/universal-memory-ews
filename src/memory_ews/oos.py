from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """
    Ridge regression with intercept:
      y ≈ β0 + Xβ , penalize β (not β0).
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1, 1)

    X1 = np.column_stack([np.ones(len(X)), X])
    p = X1.shape[1]
    I = np.eye(p)
    I[0, 0] = 0.0  # do not penalize intercept
    beta = np.linalg.solve(X1.T @ X1 + lam * I, X1.T @ y)
    return beta.flatten()


def predict_ridge(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    beta = np.asarray(beta, float).flatten()
    X1 = np.column_stack([np.ones(len(X)), X])
    return (X1 @ beta.reshape(-1, 1)).flatten()


def oos_experiment_ridge(
    df: pd.DataFrame,
    horizon: int,
    split_date: str,
    features: list[str],
    target_col: str,
    ridge_lambda: float,
    out_path: str | None = None,
    show: bool = False,
) -> dict:
    """
    Predict target(t+horizon) from features at time t via ridge regression.

    - standardize features using TRAIN statistics only
    - baseline = train mean of target
    """
    if any(c not in df.columns for c in features + [target_col]):
        missing = [c for c in features + [target_col] if c not in df.columns]
        raise KeyError(f"Missing required columns: {missing}")

    y_future = df[target_col].shift(-int(horizon))
    X = df[features]
    D = pd.concat([X, y_future], axis=1).dropna()
    D.columns = features + ["y"]

    split_ts = pd.to_datetime(split_date)

    train = D.loc[D.index < split_ts].copy()
    test = D.loc[D.index >= split_ts].copy()

    if len(train) < 120 or len(test) < 36:
        warnings.warn(
            f"OOS split={split_date} horizon={horizon}: few samples "
            f"(train={len(train)}, test={len(test)})",
            RuntimeWarning,
        )

    mu = train[features].mean()
    sd = train[features].std().replace(0, np.nan)

    Xtr = ((train[features] - mu) / sd).to_numpy()
    Xte = ((test[features] - mu) / sd).to_numpy()
    ytr = train["y"].to_numpy()
    yte = test["y"].to_numpy()

    beta = fit_ridge(Xtr, ytr, lam=float(ridge_lambda))
    yhat = predict_ridge(Xte, beta)

    y_clim = np.full_like(yte, ytr.mean(), dtype=float)

    rmse = float(np.sqrt(np.mean((yte - yhat) ** 2)))
    rmse_clim = float(np.sqrt(np.mean((yte - y_clim) ** 2)))
    skill = float(1.0 - rmse / (rmse_clim + 1e-12))
    corr = (
        float(np.corrcoef(yte, yhat)[0, 1])
        if (np.nanstd(yte) > 0 and np.nanstd(yhat) > 0)
        else np.nan
    )

    if out_path is not None:
        fig = plt.figure(figsize=(12, 4.5), constrained_layout=True)
        ax = plt.gca()
        ax.plot(test.index, yte, lw=2.2, label=f"Observed {target_col}(t+{horizon}m)")
        ax.plot(test.index, yhat, lw=2.2, label="Predicted (ridge)")
        ax.axhline(ytr.mean(), lw=2, ls="--", label="Baseline (train mean)")
        ax.set_ylabel(target_col)
        ax.set_xlabel("time (test period)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", frameon=True)

        ax.set_title(
            f"OOS ridge: {target_col}(t+{horizon}m) | split={split_date}\n"
            f"RMSE={rmse:.4f} (clim {rmse_clim:.4f}) | skill={skill:.3f} | corr={corr:.3f} | lambda={ridge_lambda:g}"
        )
        fig.savefig(out_path, dpi=200)
        if show:
            plt.show()
        else:
            plt.close(fig)

    return dict(
        split_date=str(split_date),
        horizon=int(horizon),
        rmse=rmse,
        rmse_clim=rmse_clim,
        skill=skill,
        corr=corr,
    )
