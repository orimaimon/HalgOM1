"""Portfolio benchmark comparison module.

Provides pure functions for comparing portfolio daily returns against a benchmark series:
- compare(): Calculates summary statistics (beta, alpha, correlation, tracking error, up/down capture).
- rolling_beta(): Calculates vectorized rolling beta over a moving window.
"""

from typing import Any
import numpy as np
import pandas as pd


def compare(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_annual: float = 0.0,
) -> dict[str, Any]:
    """Compare portfolio daily returns to a benchmark series.

    Args:
        portfolio_returns: Daily simple returns as fractions (0.01 == 1%), indexed by date.
        benchmark_returns: Daily simple returns as fractions (0.01 == 1%), indexed by date.
        risk_free_annual: Annualized risk-free rate as a fraction (0.04 == 4%).

    Returns:
        dict with keys: n_obs, beta, alpha_annual, correlation, tracking_error, up_capture, down_capture.
        Missing/uncomputable statistics evaluate to None.
    """
    default_result: dict[str, Any] = {
        "n_obs": 0,
        "beta": None,
        "alpha_annual": None,
        "correlation": None,
        "tracking_error": None,
        "up_capture": None,
        "down_capture": None,
    }

    try:
        if portfolio_returns is None or benchmark_returns is None:
            return default_result

        # Align on common index and drop NaN pairs
        df = pd.DataFrame({"p": portfolio_returns, "b": benchmark_returns}).dropna()
        n_obs = int(len(df))
        result = dict(default_result)
        result["n_obs"] = n_obs

        if n_obs < 20:
            return result

        p = df["p"]
        b = df["b"]

        # 1. Beta & Annualized Alpha
        var_b = float(b.var(ddof=1))
        # Zero variance threshold for constant benchmark returns (floating point precision limit)
        if var_b > 1e-12 and not np.isnan(var_b):
            cov_pb = float(p.cov(b))
            if not np.isnan(cov_pb):
                beta_val = cov_pb / var_b
                result["beta"] = float(beta_val)

                rf_daily = float(risk_free_annual) / 252.0
                mean_p = float(p.mean())
                mean_b = float(b.mean())
                alpha_daily = (mean_p - rf_daily) - beta_val * (mean_b - rf_daily)
                result["alpha_annual"] = float(alpha_daily * 252.0)

        # 2. Correlation
        var_p = float(p.var(ddof=1))
        if var_p > 1e-12 and var_b > 1e-12:
            corr_val = float(p.corr(b))
            if not np.isnan(corr_val) and not np.isinf(corr_val):
                result["correlation"] = corr_val

        # 3. Tracking Error
        diff = p - b
        te_val = float(diff.std(ddof=1) * np.sqrt(252.0))
        if not np.isnan(te_val) and not np.isinf(te_val):
            result["tracking_error"] = te_val

        # 4. Up Capture Ratio
        df_up = df[df["b"] > 0]
        if len(df_up) > 0:
            mean_b_up = float(df_up["b"].mean())
            if mean_b_up > 1e-12 and not np.isnan(mean_b_up):
                uc_val = float(df_up["p"].mean()) / mean_b_up
                if not np.isnan(uc_val) and not np.isinf(uc_val):
                    result["up_capture"] = uc_val

        # 5. Down Capture Ratio
        df_down = df[df["b"] < 0]
        if len(df_down) > 0:
            mean_b_down = float(df_down["b"].mean())
            if mean_b_down < -1e-12 and not np.isnan(mean_b_down):
                dc_val = float(df_down["p"].mean()) / mean_b_down
                if not np.isnan(dc_val) and not np.isinf(dc_val):
                    result["down_capture"] = dc_val

        return result

    except Exception:
        return default_result


def rolling_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 60,
) -> pd.Series:
    """Compute rolling beta using vectorized pandas rolling covariance and variance.

    Args:
        portfolio_returns: Daily simple returns as fractions, indexed by date.
        benchmark_returns: Daily simple returns as fractions, indexed by date.
        window: Rolling window length in trading days (default 60).

    Returns:
        pd.Series indexed like portfolio_returns, with NaN values until the window fills.
    """
    try:
        if portfolio_returns is None or benchmark_returns is None:
            return pd.Series(dtype=float)

        df = pd.DataFrame({"p": portfolio_returns, "b": benchmark_returns}).dropna()
        if len(df) < window:
            res = pd.Series(np.nan, index=df.index)
            return res.reindex(portfolio_returns.index)

        cov = df["p"].rolling(window=window, min_periods=window).cov(df["b"])
        var = df["b"].rolling(window=window, min_periods=window).var()

        # Mask variance <= 1e-12 as NaN to prevent division by zero / constant benchmark explosion
        var = var.where(var > 1e-12, np.nan)

        beta = cov / var
        beta = beta.replace([np.inf, -np.inf], np.nan)
        return beta.reindex(portfolio_returns.index)
    except Exception:
        idx = getattr(portfolio_returns, "index", None)
        return pd.Series(np.nan, index=idx)
