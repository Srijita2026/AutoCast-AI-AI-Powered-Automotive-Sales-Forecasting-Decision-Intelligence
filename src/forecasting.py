"""Forecasting engine: ARIMA, SARIMA, ARIMAX, SARIMAX with chronological validation.

Protocol
--------
1. The last ``test_periods`` observations are a hold-out that is NOT used for model selection.
2. Candidates are scored by expanding-window rolling-origin validation on the development part only.
3. The selected model is chosen from validation metrics (+ a parsimony rule), never from AIC alone.
4. Only after selection is the hold-out used, to report an honest out-of-sample score.
5. The final model is refitted on all data to forecast the future with 95% prediction intervals.

Exogenous regressors are restricted to *calendar* indicators whose future values are genuinely known.
Anything else raises ``FutureRegressorError`` rather than being zero-filled or leaked from actuals.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from . import diagnostics as dg
from .config import ForecastSettings
from .evaluation import compute_metrics, interval_coverage

log = logging.getLogger(__name__)

CALENDAR_EXOG: dict[str, int] = {"is_nov": 11, "is_oct": 10}
EXOG_SETS: dict[str, tuple[str, ...]] = {"nov": ("is_nov",), "oct_nov": ("is_oct", "is_nov")}


class FitFailure(Exception):
    """A model could not be fitted/forecast reliably (reason in message)."""


class FutureRegressorError(FitFailure):
    """Future values of an exogenous regressor are not available/known."""


class InsufficientDataError(ValueError):
    """Not enough history for the requested validation design."""


# ----------------------------------------------------------------------------- exogenous
def calendar_exog(index: pd.DatetimeIndex, cols: tuple[str, ...]) -> pd.DataFrame:
    """Calendar indicators derived purely from dates (known for any future period)."""
    unknown = [c for c in cols if c not in CALENDAR_EXOG]
    if unknown:
        raise FutureRegressorError(f"No known future values for regressors {unknown}; "
                                   "supply explicit scenario values or drop them.")
    return pd.DataFrame({c: (index.month == CALENDAR_EXOG[c]).astype(float) for c in cols}, index=index)


def build_future_exog(cols: tuple[str, ...], future_index: pd.DatetimeIndex,
                      scenarios: pd.DataFrame | None = None) -> pd.DataFrame:
    """Future regressors: calendar-derived, or user scenario values. Never zero-filled silently."""
    if not cols:
        return pd.DataFrame(index=future_index)
    known = [c for c in cols if c in CALENDAR_EXOG]
    other = [c for c in cols if c not in CALENDAR_EXOG]
    parts = [calendar_exog(future_index, tuple(known))] if known else []
    if other:
        if scenarios is None or not set(other) <= set(scenarios.columns):
            raise FutureRegressorError(f"Future values for {other} are unknown and no scenario was provided.")
        sc = scenarios.reindex(future_index)[other]
        if sc.isna().any().any():
            raise FutureRegressorError(f"Scenario values for {other} do not cover every forecast period.")
        parts.append(sc)
    return pd.concat(parts, axis=1)[list(cols)]


# ----------------------------------------------------------------------------- specs
@dataclass(frozen=True)
class ModelSpec:
    """A fully specified candidate model."""

    family: str                                   # ARIMA | SARIMA | ARIMAX | SARIMAX
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0)
    trend: str | None = None
    exog: tuple[str, ...] = ()
    exog_name: str = ""

    @property
    def label(self) -> str:
        s = f"{self.family}{self.order}"
        if self.seasonal_order[3]:
            s += f"x{self.seasonal_order}"
        if self.trend:
            s += f" +{self.trend}"
        if self.exog:
            s += f" [{self.exog_name or ','.join(self.exog)}]"
        return s

    @property
    def n_params(self) -> int:
        p, _, q = self.order
        P, _, Q, _ = self.seasonal_order
        return p + q + P + Q + len(self.exog) + (1 if self.trend else 0)


def build_candidates(seasonal_ok: bool, s: int = 12) -> list[ModelSpec]:
    """Small, defensible candidate grids per family (kept small: only ~26 months of history)."""
    base_pq = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2)]
    seas_pq = [(1, 0), (0, 1), (1, 1)]
    seas_orders = [(1, 0, 0, s), (0, 0, 1, s)]  # D=0: seasonal differencing would consume 12 of ~26 points
    out: list[ModelSpec] = []
    for d in (0, 1):
        trend = "c" if d == 0 else None
        for p, q in base_pq:
            out.append(ModelSpec("ARIMA", (p, d, q), trend=trend))
            for name, cols in EXOG_SETS.items():
                out.append(ModelSpec("ARIMAX", (p, d, q), trend=trend, exog=cols, exog_name=name))
        if seasonal_ok:
            for p, q in seas_pq:
                for so in seas_orders:
                    out.append(ModelSpec("SARIMA", (p, d, q), so, trend))
                    for name, cols in EXOG_SETS.items():
                        out.append(ModelSpec("SARIMAX", (p, d, q), so, trend, cols, name))
    return out


# ----------------------------------------------------------------------------- fitting
@dataclass
class FitForecast:
    """Result of fitting a spec on a training window and forecasting ``steps`` ahead."""

    mean: pd.Series
    lower: pd.Series
    upper: pd.Series
    aic: float
    bic: float
    nobs: int
    resid: pd.Series


def _freq(y: pd.Series, freq: str = "MS") -> pd.Series:
    y = y.copy()
    y.index = pd.DatetimeIndex(y.index, freq=freq)
    return y


def fit_forecast(spec: ModelSpec, y: pd.Series, steps: int, exog_train: pd.DataFrame | None = None,
                 exog_future: pd.DataFrame | None = None, alpha: float = 0.05) -> FitForecast:
    """Fit ``spec`` on ``y`` and forecast ``steps`` periods with (1-alpha) prediction intervals.

    The series is scaled internally for numerical stability; outputs are on the original scale.
    Raises FitFailure with a human-readable reason on any problem.
    """
    y = _freq(y)
    if spec.exog:
        if exog_train is None or exog_future is None:
            raise FutureRegressorError("exogenous regressors required but not provided")
        if exog_future.isna().any().any() or len(exog_future) != steps:
            raise FutureRegressorError("future regressors missing/incomplete for the forecast horizon")
        ex_tr, ex_fu = exog_train.loc[y.index, list(spec.exog)], exog_future[list(spec.exog)]
    else:
        ex_tr = ex_fu = None
    s_len = spec.seasonal_order[3]
    min_obs = max(8 + spec.n_params, int(1.5 * s_len))
    if len(y) < min_obs:
        raise FitFailure(f"insufficient history: {len(y)} observations, need >= {min_obs} for {spec.label}")
    scale = float(np.abs(y).mean()) or 1.0
    ys = y / scale
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mod = SARIMAX(ys, exog=ex_tr, order=spec.order, seasonal_order=spec.seasonal_order,
                          trend=spec.trend, enforce_stationarity=False, enforce_invertibility=False)
            res = mod.fit(disp=False, maxiter=300)
            fc = res.get_forecast(steps=steps, exog=ex_fu)
            ci = fc.conf_int(alpha=alpha)
    except FitFailure:
        raise
    except Exception as exc:  # noqa: BLE001 - statsmodels raises many types
        raise FitFailure(f"{type(exc).__name__}: {str(exc)[:120]}") from exc
    if not bool(res.mle_retvals.get("converged", True)):
        raise FitFailure("optimizer did not converge")
    if any("convergence" in str(w.message).lower() for w in caught):
        raise FitFailure("convergence warning from optimizer")
    mean = fc.predicted_mean * scale
    lower, upper = ci.iloc[:, 0] * scale, ci.iloc[:, 1] * scale
    if not (np.isfinite(mean).all() and np.isfinite(lower).all() and np.isfinite(upper).all()):
        raise FitFailure("non-finite forecast or interval")
    if np.abs(mean).max() > 20 * np.abs(y).max():
        raise FitFailure("forecast exploded (>20x historical maximum)")
    burn = spec.order[1] + spec.seasonal_order[1] * spec.seasonal_order[3]
    resid = (res.resid * scale).iloc[burn:]
    n = len(y)
    shift = 2 * n * np.log(scale)  # undo scaling effect on the likelihood so AIC/BIC are on the original scale
    return FitForecast(mean=mean, lower=lower.clip(lower=0), upper=upper, aic=float(res.aic + shift),
                       bic=float(res.bic + shift), nobs=n, resid=resid)


# ----------------------------------------------------------------------------- benchmarks
def naive_forecast(y: pd.Series, steps: int) -> pd.Series:
    """Repeat the last observation."""
    idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
    return pd.Series(float(y.iloc[-1]), index=idx)


def seasonal_naive_forecast(y: pd.Series, steps: int, s: int = 12) -> pd.Series:
    """Repeat the value from one season ago."""
    if len(y) < s:
        raise FitFailure(f"need at least {s} observations for seasonal naive")
    idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
    vals = [float(y.iloc[len(y) - s + (k % s)]) for k in range(steps)]
    return pd.Series(vals, index=idx)


# ----------------------------------------------------------------------------- validation
def rolling_origins(n_dev: int, min_train: int, horizon: int) -> list[int]:
    """Expanding-window training sizes such that each origin has a full ``horizon`` of actuals."""
    if n_dev < min_train + horizon:
        raise InsufficientDataError(
            f"{n_dev} development observations cannot support min_train={min_train} + horizon={horizon}.")
    return list(range(min_train, n_dev - horizon + 1))


def validate_spec(spec: ModelSpec, y_dev: pd.Series, exog_all: pd.DataFrame, origins: list[int],
                  horizon: int, alpha: float) -> tuple[pd.DataFrame, list[dict]]:
    """Rolling-origin validation of one spec. Returns (prediction rows, failure records)."""
    rows, fails = [], []
    for o in origins:
        tr, va = y_dev.iloc[:o], y_dev.iloc[o:o + horizon]
        try:
            ex_future = exog_all.loc[va.index] if spec.exog else None
            r = fit_forecast(spec, tr, len(va), exog_all.loc[tr.index] if spec.exog else None, ex_future, alpha)
        except FitFailure as exc:
            fails.append({"model": spec.label, "family": spec.family, "origin": tr.index[-1], "reason": str(exc)})
            continue
        for k, (dt, act) in enumerate(va.items(), start=1):
            rows.append({"model": spec.label, "family": spec.family, "origin": tr.index[-1], "step": k, "date": dt,
                         "actual": float(act), "pred": float(r.mean.iloc[k - 1]),
                         "lower": float(r.lower.iloc[k - 1]), "upper": float(r.upper.iloc[k - 1])})
    return pd.DataFrame(rows), fails


# ----------------------------------------------------------------------------- selection
@dataclass
class SelectionResult:
    """Everything the UI/report needs from one model-selection run."""

    settings: ForecastSettings
    series: pd.Series
    dev: pd.Series
    test: pd.Series
    seasonal_evidence: dict
    adf: dict
    origins: list[int]
    leaderboard: pd.DataFrame
    family_best: pd.DataFrame
    val_preds: pd.DataFrame            # rolling-origin predictions of each family's best + benchmarks
    test_table: pd.DataFrame           # hold-out metrics (reported AFTER selection)
    test_preds: pd.DataFrame
    failures: pd.DataFrame
    selected: ModelSpec | None
    family_specs: dict[str, ModelSpec] = field(default_factory=dict)
    rationale: str = ""
    notes: list[str] = field(default_factory=list)


def _primary(m: dict, key: str) -> float:
    v = m.get(key, np.nan)
    return float(v) if v is not None and np.isfinite(v) else np.inf


def run_selection(y: pd.Series, settings: ForecastSettings, progress=None) -> SelectionResult:
    """Run the full chronological model comparison and selection (see module docstring)."""
    y = _freq(y, settings.freq)
    n = len(y)
    if settings.test_periods < 1 or n <= settings.test_periods:
        raise InsufficientDataError("test_periods must be >=1 and smaller than the series length.")
    dev, test = y.iloc[:-settings.test_periods], y.iloc[-settings.test_periods:]
    notes: list[str] = []
    min_train = settings.min_train_periods
    if len(dev) < min_train + settings.val_horizon:
        min_train = len(dev) - settings.val_horizon - 2
        if min_train < 10:
            raise InsufficientDataError(
                f"Only {len(dev)} development periods: too short for rolling-origin validation "
                f"(horizon {settings.val_horizon}). Reduce the hold-out/horizon or add data.")
        notes.append(f"min_train reduced to {min_train} because the development set is short.")
    origins = rolling_origins(len(dev), min_train, settings.val_horizon)

    evidence = dg.seasonal_evidence(dev, settings)
    adf = dg.adf_test(dev)
    seasonal_ok = bool(evidence["supported"])
    if not seasonal_ok:
        notes.append("Seasonal models skipped: " + evidence["reason"])
    candidates = build_candidates(seasonal_ok, settings.seasonal_period)
    exog_cols = tuple(sorted({c for sp in candidates for c in sp.exog}))
    exog_all = calendar_exog(y.index, exog_cols)

    lb_rows, val_frames, fail_rows = [], {}, []
    for i, spec in enumerate(candidates):
        if progress:
            progress(i / len(candidates), f"Validating {spec.label}")
        vp, fails = validate_spec(spec, dev, exog_all, origins, settings.val_horizon, settings.alpha)
        fail_rows += fails
        failed_share = len(fails) / len(origins)
        m = compute_metrics(vp["actual"], vp["pred"]) if len(vp) else compute_metrics([], [])
        cov = interval_coverage(vp["actual"], vp["lower"], vp["upper"]) if len(vp) else np.nan
        aic = bic = np.nan
        try:  # information criteria from a fit on the full development set (diagnostic only)
            nxt = pd.date_range(dev.index[-1] + pd.offsets.MonthBegin(1), periods=1, freq="MS")
            ff = fit_forecast(spec, dev, 1, exog_all.loc[dev.index] if spec.exog else None,
                              exog_all.loc[nxt] if spec.exog else None, settings.alpha)
            aic, bic = ff.aic, ff.bic
        except FitFailure as exc:
            fail_rows.append({"model": spec.label, "family": spec.family, "origin": dev.index[-1],
                              "reason": f"full-development fit: {exc}"})
        lb_rows.append({"model": spec.label, "family": spec.family, "n_params": spec.n_params,
                        "folds": len(origins), "folds_failed": len(fails),
                        "eligible": failed_share <= settings.max_failed_fold_share and len(vp) > 0
                                    and np.isfinite(aic),
                        **{k: m[k] for k in ("mae", "rmse", "wape", "mape", "bias")},
                        "coverage_95": cov, "aic": aic, "bic": bic})
        if len(vp):
            val_frames[spec.label] = vp
    spec_by_label = {sp.label: sp for sp in candidates}
    lb = pd.DataFrame(lb_rows)
    pm = settings.primary_metric
    lb["_key"] = lb[pm].where(lb["eligible"], np.inf).fillna(np.inf)
    lb = lb.sort_values(["_key", "aic"]).reset_index(drop=True)

    # benchmarks (no intervals)
    bench_preds = []
    for name, fn in (("Benchmark: naive", lambda tr, h: naive_forecast(tr, h)),
                     ("Benchmark: seasonal naive", lambda tr, h: seasonal_naive_forecast(tr, h, settings.seasonal_period))):
        for o in origins:
            tr, va = dev.iloc[:o], dev.iloc[o:o + settings.val_horizon]
            try:
                p = fn(tr, len(va))
            except FitFailure:
                continue
            for k, (dt, act) in enumerate(va.items(), start=1):
                bench_preds.append({"model": name, "family": "Benchmark", "origin": tr.index[-1], "step": k,
                                    "date": dt, "actual": float(act), "pred": float(p.iloc[k - 1]),
                                    "lower": np.nan, "upper": np.nan})
    bench = pd.DataFrame(bench_preds)
    bench_metrics = {name: compute_metrics(g["actual"], g["pred"]) for name, g in bench.groupby("model")} if len(bench) else {}

    # best per family
    fam_rows, fam_specs = [], {}
    for fam in ("ARIMA", "SARIMA", "ARIMAX", "SARIMAX"):
        sub = lb[(lb["family"] == fam) & lb["eligible"]]
        if sub.empty:
            reason = ("seasonal evidence insufficient" if fam in ("SARIMA", "SARIMAX") and not seasonal_ok
                      else "no candidate passed validation")
            fam_rows.append({"family": fam, "model": None, "status": f"not available: {reason}"})
            continue
        best = sub.iloc[0]
        fam_specs[fam] = spec_by_label[best["model"]]
        fam_rows.append({"family": fam, "model": best["model"], "status": "ok", **{k: best[k] for k in
                         ("n_params", "mae", "rmse", "wape", "mape", "coverage_95", "aic", "bic", "folds_failed")}})
    family_best = pd.DataFrame(fam_rows)

    selected, rationale = None, ""
    ok = family_best[family_best["status"] == "ok"]
    if ok.empty:
        rationale = "No model passed validation; no forecast is offered. See failure log."
    else:
        best_val = float(ok[pm].min())
        close = ok[ok[pm] <= best_val * (1 + settings.parsimony_tolerance)].sort_values(["n_params", pm])
        pick = close.iloc[0]
        selected = fam_specs[pick["family"]]
        lowest = ok.sort_values(pm).iloc[0]
        rationale = (f"Selected {pick['model']} using rolling-origin validation ({len(origins)} folds x "
                     f"{settings.val_horizon}-month horizon), primary metric {pm.upper()}={pick[pm]:.3f}.")
        if pick["model"] != lowest["model"]:
            rationale += (f" It is within {settings.parsimony_tolerance:.0%} of the best {pm.upper()} "
                          f"({lowest['model']}: {lowest[pm]:.3f}) and has fewer parameters, so the simpler model was preferred.")
        if selected.exog:
            rationale += (f" Its regressors ({', '.join(selected.exog)}) are calendar indicators, whose future "
                          "values are known, so forecasts are valid without separate regressor forecasts.")
        if bench_metrics:
            bname, bm = min(bench_metrics.items(), key=lambda kv: _primary(kv[1], pm))
            if _primary(bm, pm) <= float(pick[pm]):
                notes.append(f"{bname} ({pm.upper()}={bm[pm]:.3f}) matched or beat the selected model on validation; "
                             "treat the added model complexity with caution.")
        if np.isfinite(pick["coverage_95"]) and pick["coverage_95"] < 0.85:
            notes.append(f"Nominal 95% prediction intervals covered only {pick['coverage_95']:.0%} of validation "
                         "actuals (few folds, short history): treat the intervals as optimistically narrow.")
        if pick["folds_failed"] > 0:
            notes.append("The selected model failed in some validation folds; metrics use the successful folds.")

    # hold-out evaluation, AFTER selection
    test_rows, test_pred_rows = [], []
    exog_test_hist = exog_all
    for fam, spec in fam_specs.items():
        try:
            r = fit_forecast(spec, dev, len(test), exog_test_hist.loc[dev.index] if spec.exog else None,
                             exog_test_hist.loc[test.index] if spec.exog else None, settings.alpha)
        except FitFailure as exc:
            fail_rows.append({"model": spec.label, "family": fam, "origin": dev.index[-1],
                              "reason": f"hold-out fit: {exc}"})
            continue
        m = compute_metrics(test.values, r.mean.values)
        test_rows.append({"family": fam, "model": spec.label, "selected": spec == selected, **m,
                          "coverage_95": interval_coverage(test.values, r.lower.values, r.upper.values)})
        for dt, a, p, lo, hi in zip(test.index, test.values, r.mean.values, r.lower.values, r.upper.values):
            test_pred_rows.append({"model": spec.label, "family": fam, "date": dt, "actual": a, "pred": p,
                                   "lower": lo, "upper": hi})
    for name, fn in (("Benchmark: naive", naive_forecast),
                     ("Benchmark: seasonal naive", lambda tr, h: seasonal_naive_forecast(tr, h, settings.seasonal_period))):
        try:
            p = fn(dev, len(test))
        except FitFailure:
            continue
        m = compute_metrics(test.values, p.values)
        test_rows.append({"family": "Benchmark", "model": name, "selected": False, **m, "coverage_95": np.nan})
        for dt, a, pv in zip(test.index, test.values, p.values):
            test_pred_rows.append({"model": name, "family": "Benchmark", "date": dt, "actual": a, "pred": pv,
                                   "lower": np.nan, "upper": np.nan})
    notes.append("The hold-out period contains no November, so it cannot confirm how well the November peak is "
                 "forecast; validation folds (which include Nov 2019) are the evidence for peak handling."
                 if not any(d.month == 11 for d in test.index) else
                 "The hold-out period includes November, a peak month, so hold-out scores are sensitive to it.")
    notes.append("Candidate regressors NOT used: quantity, price, order counts and days_since_last_order are not known "
                 "in advance, so ARIMAX/SARIMAX use only calendar indicators (Oct/Nov flags) as regressors.")

    keep = {sp.label for sp in fam_specs.values()}
    vp_all = pd.concat([val_frames[k] for k in keep if k in val_frames] + [bench], ignore_index=True) \
        if keep or len(bench) else pd.DataFrame()
    lb = lb.drop(columns="_key")
    return SelectionResult(settings=settings, series=y, dev=dev, test=test, seasonal_evidence=evidence, adf=adf,
                           origins=origins, leaderboard=lb, family_best=family_best, val_preds=vp_all,
                           test_table=pd.DataFrame(test_rows), test_preds=pd.DataFrame(test_pred_rows),
                           failures=pd.DataFrame(fail_rows), selected=selected, family_specs=fam_specs,
                           rationale=rationale, notes=notes)


# ----------------------------------------------------------------------------- final forecasts
def forecast_future(y: pd.Series, spec: ModelSpec, horizon: int, alpha: float = 0.05,
                    scenarios: pd.DataFrame | None = None) -> tuple[pd.DataFrame, FitForecast]:
    """Refit ``spec`` on ALL data and forecast ``horizon`` future periods with intervals.

    Raises FutureRegressorError if regressor values for the future are not known/provided.
    """
    if not 1 <= horizon <= 24:
        raise ValueError("horizon must be between 1 and 24 periods")
    y = _freq(y)
    fidx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    ex_future = build_future_exog(spec.exog, fidx, scenarios) if spec.exog else None
    ex_train = calendar_exog(y.index, spec.exog) if spec.exog else None
    ff = fit_forecast(spec, y, horizon, ex_train, ex_future, alpha)
    out = pd.DataFrame({"forecast": ff.mean, "lower_95": ff.lower, "upper_95": ff.upper})
    out.index.name = "month"
    return out, ff
