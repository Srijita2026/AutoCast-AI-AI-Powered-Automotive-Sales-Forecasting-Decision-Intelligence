"""Series construction and time-series diagnostics (decomposition, ADF, ACF/PACF, residual checks)."""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, adfuller, pacf

from .config import ForecastSettings

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- series
def build_series(df: pd.DataFrame, settings: ForecastSettings) -> tuple[pd.Series, list[str]]:
    """Aggregate to a continuous monthly series for the chosen target.

    Months with no transactions are real zeros (no sales recorded) and are kept as 0; they are listed in notes.
    Excluded statuses (default: Cancelled) are removed *before* aggregation.
    """
    notes: list[str] = []
    d = df
    if settings.exclude_statuses and "status" in d.columns:
        d = d[~d["status"].isin(settings.exclude_statuses)]
        notes.append(f"Excluded statuses from the target: {list(settings.exclude_statuses)}.")
    if d.empty:
        raise ValueError("No rows left after applying the status exclusions.")
    if settings.target == "revenue":
        s = d.groupby("year_month")["sales"].sum()
    elif settings.target == "units":
        s = d.groupby("year_month")["quantity_ordered"].sum()
    elif settings.target == "orders":
        s = d.groupby("year_month")["order_number"].nunique()
    else:
        raise ValueError(f"Unknown target '{settings.target}'")
    idx = pd.date_range(s.index.min(), s.index.max(), freq=settings.freq)
    missing = idx.difference(s.index)
    s = s.reindex(idx)
    s = s.fillna(0.0).astype(float)
    s.index.freq = settings.freq
    s.name = settings.target
    if len(missing):
        notes.append(f"{len(missing)} month(s) had no transactions and were set to 0: "
                     f"{[f'{m:%Y-%m}' for m in missing]}.")
    first, last = d["order_date"].min(), d["order_date"].max()
    if first.day > 7:
        notes.append(f"First period ({first:%b %Y}) may be partial: first order is on day {first.day}.")
    if last.day < (last + pd.offsets.MonthEnd(0)).day - 6:
        notes.append(f"Last period ({last:%b %Y}) may be partial: last order is on {last:%d %b}.")
    notes.append(f"{len(s)} monthly observations ({s.index.min():%b %Y} to {s.index.max():%b %Y}).")
    return s, notes


def frequency_check(df: pd.DataFrame) -> pd.DataFrame:
    """Justify the aggregation frequency: periods available and share of empty periods per frequency."""
    rows = []
    d = df["order_date"]
    for label, rule in (("Daily", "D"), ("Weekly", "W-MON"), ("Monthly", "MS"), ("Quarterly", "QS")):
        counts = df.set_index("order_date")["sales"].resample(rule).size()
        rows.append({"frequency": label, "periods": len(counts),
                     "empty_periods_pct": round(100 * (counts == 0).mean(), 1),
                     "median_lines_per_period": float(counts.median())})
    out = pd.DataFrame(rows)
    out.attrs["span"] = (d.min(), d.max())
    return out


# ----------------------------------------------------------------------------- decomposition / seasonality
@dataclass
class Decomp:
    observed: pd.Series
    trend: pd.Series
    seasonal: pd.Series
    resid: pd.Series
    period: int
    strength: float


def seasonal_strength(resid: pd.Series, seasonal: pd.Series) -> float:
    """Hyndman's F_S = max(0, 1 - Var(R) / Var(S + R))."""
    r, sr = resid.dropna(), (seasonal + resid).dropna()
    if len(r) < 3 or sr.var() == 0:
        return 0.0
    return float(max(0.0, 1 - r.var() / sr.var()))


def decompose(y: pd.Series, period: int = 12) -> Decomp | None:
    """Additive classical decomposition; requires >= 2 full cycles, otherwise returns None."""
    if len(y) < 2 * period:
        return None
    try:
        r = seasonal_decompose(y, model="additive", period=period)
    except Exception as exc:  # noqa: BLE001
        log.warning("decomposition failed: %s", exc)
        return None
    return Decomp(y, r.trend, r.seasonal, r.resid, period, seasonal_strength(r.resid, r.seasonal))


def seasonal_evidence(y: pd.Series, settings: ForecastSettings) -> dict:
    """Decide whether seasonal (period s) models are justified. Never assumes s=12 blindly."""
    s = settings.seasonal_period
    n = len(y)
    ev = {"period": s, "n_obs": n, "full_cycles": n // s, "strength": None, "supported": False, "reason": ""}
    if n < 2 * s:
        ev["reason"] = f"Only {n} observations: need at least {2 * s} (two full cycles of {s}) for seasonal models."
        return ev
    dec = decompose(y, s)
    if dec is None:
        ev["reason"] = "Decomposition failed; seasonality could not be assessed."
        return ev
    ev["strength"] = dec.strength
    ev["supported"] = dec.strength >= settings.min_seasonal_strength
    ev["reason"] = (f"Seasonal strength {dec.strength:.2f} over {n // s} cycles "
                    + ("meets" if ev["supported"] else "is below") + f" the {settings.min_seasonal_strength:.2f} gate. "
                    "Caveat: with so few cycles seasonal estimates are fragile.")
    return ev


# ----------------------------------------------------------------------------- tests
def rolling_stats(y: pd.Series, window: int = 3) -> pd.DataFrame:
    """Rolling mean and std."""
    w = max(2, min(window, len(y)))
    return pd.DataFrame({"value": y, "rolling_mean": y.rolling(w).mean(), "rolling_std": y.rolling(w).std()})


def adf_test(y: pd.Series) -> dict:
    """Augmented Dickey-Fuller with a small lag cap (short series). Returns a dict, never raises."""
    y = y.dropna()
    out = {"n": len(y), "statistic": np.nan, "p_value": np.nan, "used_lag": None, "stationary_5pct": None,
           "critical_values": {}, "error": None}
    if len(y) < 8 or y.nunique() < 3:
        out["error"] = "Series too short or constant for an ADF test."
        return out
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p, lag, _, crit, _ = adfuller(y, maxlag=min(4, len(y) // 5), autolag="AIC")
        out.update(statistic=float(stat), p_value=float(p), used_lag=int(lag), critical_values=crit,
                   stationary_5pct=bool(p < 0.05))
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
    return out


def acf_pacf(y: pd.Series, nlags: int = 12) -> dict:
    """ACF/PACF values with approximate 95% bounds (+/-1.96/sqrt(n))."""
    y = y.dropna()
    n = len(y)
    k = max(1, min(nlags, n // 2 - 1))
    try:
        a = acf(y, nlags=k, fft=False)
        p = pacf(y, nlags=k, method="ywm")
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    return {"acf": a, "pacf": p, "lags": np.arange(k + 1), "bound": 1.96 / np.sqrt(n), "error": None}


def ljung_box(resid: pd.Series, lags: int | None = None, model_df: int = 0) -> pd.DataFrame:
    """Ljung-Box test table; returns empty frame if too few residuals."""
    r = pd.Series(resid).dropna()
    if len(r) < 8:
        return pd.DataFrame(columns=["lag", "lb_stat", "lb_pvalue"])
    lags = lags or max(1, min(10, len(r) // 3))
    try:
        res = acorr_ljungbox(r, lags=[lags], model_df=min(model_df, lags - 1) if lags > 1 else 0)
        return res.reset_index().rename(columns={"index": "lag"})
    except Exception as exc:  # noqa: BLE001
        log.warning("Ljung-Box failed: %s", exc)
        return pd.DataFrame(columns=["lag", "lb_stat", "lb_pvalue"])


def residual_diagnostics(resid: pd.Series, model_df: int = 0) -> dict:
    """Summary of residual behaviour: mean, std, Ljung-Box, Jarque-Bera, ACF of residuals."""
    r = pd.Series(resid).dropna()
    out: dict = {"n": len(r), "mean": float(r.mean()) if len(r) else np.nan,
                 "std": float(r.std()) if len(r) > 1 else np.nan}
    lb = ljung_box(r, model_df=model_df)
    out["ljung_box"] = lb
    out["ljung_box_p"] = float(lb["lb_pvalue"].iloc[0]) if len(lb) else np.nan
    if len(r) >= 8:
        jb, p = stats.jarque_bera(r)
        out["jarque_bera_p"] = float(p)
    else:
        out["jarque_bera_p"] = np.nan
    out["acf"] = acf_pacf(r, 12)
    out["white_noise_5pct"] = bool(out["ljung_box_p"] > 0.05) if not np.isnan(out["ljung_box_p"]) else None
    return out
