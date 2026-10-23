"""Forecast accuracy metrics. MAPE is only reported when it is meaningful."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def mape_is_meaningful(actual, min_rel: float = 0.01) -> bool:
    """MAPE needs strictly positive actuals that are not tiny relative to the mean level."""
    a = _arr(actual)
    if a.size == 0 or np.any(~np.isfinite(a)):
        return False
    return bool(np.all(a > 0) and a.min() >= min_rel * a.mean())


def compute_metrics(actual, pred) -> dict:
    """MAE, RMSE, WAPE, MAPE (NaN if not meaningful), bias (mean error) and n.

    WAPE = sum|error| / sum|actual| (robust to small individual actuals).
    """
    a, p = _arr(actual), _arr(pred)
    mask = np.isfinite(a) & np.isfinite(p)
    a, p = a[mask], p[mask]
    if a.size == 0:
        return {"mae": np.nan, "rmse": np.nan, "wape": np.nan, "mape": np.nan, "bias": np.nan, "n": 0}
    err = p - a
    denom = np.abs(a).sum()
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "wape": float(np.abs(err).sum() / denom) if denom > 0 else np.nan,
        "mape": float((np.abs(err) / a).mean()) if mape_is_meaningful(a) else np.nan,
        "bias": float(err.mean()),
        "n": int(a.size),
    }


def interval_coverage(actual, lower, upper) -> float:
    """Share of actuals inside [lower, upper]; NaN when no intervals are available."""
    a, lo, hi = _arr(actual), _arr(lower), _arr(upper)
    mask = np.isfinite(a) & np.isfinite(lo) & np.isfinite(hi)
    if not mask.any():
        return float("nan")
    return float(((a[mask] >= lo[mask]) & (a[mask] <= hi[mask])).mean())


def metrics_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Metrics per group from a long frame with columns actual, pred (and optionally lower, upper)."""
    rows = []
    for key, g in df.groupby(group_col, sort=False):
        m = compute_metrics(g["actual"], g["pred"])
        m[group_col] = key
        m["coverage_95"] = interval_coverage(g["actual"], g["lower"], g["upper"]) if "lower" in g else np.nan
        rows.append(m)
    return pd.DataFrame(rows)
