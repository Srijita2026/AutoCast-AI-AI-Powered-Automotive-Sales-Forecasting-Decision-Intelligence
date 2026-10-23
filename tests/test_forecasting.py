import numpy as np
import pandas as pd
import pytest

from src import forecasting as fc
from src.config import ForecastSettings


def test_rolling_origins_are_chronological_and_full_horizon():
    o = fc.rolling_origins(26, 18, 3)
    assert o == [18, 19, 20, 21, 22, 23]
    assert max(o) + 3 <= 26                      # every fold has a full horizon of actuals inside the dev set
    with pytest.raises(fc.InsufficientDataError):
        fc.rolling_origins(15, 18, 3)


def test_calendar_exog_known_future():
    idx = pd.date_range("2020-09-01", periods=4, freq="MS")
    ex = fc.calendar_exog(idx, ("is_oct", "is_nov"))
    assert ex["is_oct"].tolist() == [0, 1, 0, 0] and ex["is_nov"].tolist() == [0, 0, 1, 0]


def test_unknown_regressor_is_refused_not_zero_filled():
    idx = pd.date_range("2020-06-01", periods=3, freq="MS")
    with pytest.raises(fc.FutureRegressorError):
        fc.build_future_exog(("quantity",), idx)
    with pytest.raises(fc.FutureRegressorError):          # scenario that does not cover every period
        fc.build_future_exog(("quantity",), idx, pd.DataFrame({"quantity": [1, 2]}, index=idx[:2]))
    ok = fc.build_future_exog(("quantity",), idx, pd.DataFrame({"quantity": [1, 2, 3]}, index=idx))
    assert ok["quantity"].tolist() == [1, 2, 3]


def test_fit_forecast_intervals_ordered_and_finite():
    rng = np.random.default_rng(3)
    e, x = rng.standard_normal(30), [0.0]
    for t in range(1, 30):
        x.append(0.6 * x[-1] + e[t])
    y = pd.Series(100 + 5 * np.array(x), index=pd.date_range("2018-01-01", periods=30, freq="MS"))
    spec = fc.ModelSpec("ARIMA", (1, 0, 0), trend="c")
    r = fc.fit_forecast(spec, y, 6)
    assert len(r.mean) == 6 and np.isfinite(r.mean).all()
    assert (r.lower <= r.mean + 1e-9).all() and (r.mean <= r.upper + 1e-9).all()
    assert (r.lower >= 0).all()


def test_arimax_uses_calendar_flag_to_capture_november(series):
    spec = fc.ModelSpec("ARIMAX", (0, 0, 0), trend="c", exog=("is_nov",), exog_name="nov")
    tr = series.iloc[:34]                                   # Jan 2018 - Oct 2020
    ex = fc.calendar_exog(series.index, ("is_nov",))
    r = fc.fit_forecast(spec, tr, 2, ex.loc[tr.index], ex.loc[series.index[34:36]])
    assert r.mean.iloc[0] > r.mean.iloc[1] + 100            # Nov 2020 >> Dec 2020


def test_failed_fit_is_reported_gracefully():
    spec = fc.ModelSpec("SARIMA", (1, 0, 1), (1, 0, 1, 12), "c")
    short = pd.Series(np.arange(6, dtype=float), index=pd.date_range("2020-01-01", periods=6, freq="MS"))
    with pytest.raises(fc.FitFailure):
        fc.fit_forecast(spec, short, 3)
    with pytest.raises(fc.FutureRegressorError):            # exog model without future regressors
        fc.fit_forecast(fc.ModelSpec("ARIMAX", (1, 0, 0), exog=("is_nov",)), short, 3)


def test_benchmarks(series):
    n = fc.naive_forecast(series.iloc[:24], 3)
    assert (n == series.iloc[23]).all() and n.index[0] == series.index[24]
    sn = fc.seasonal_naive_forecast(series.iloc[:24], 3)
    assert sn.iloc[0] == series.iloc[12]
    with pytest.raises(fc.FitFailure):
        fc.seasonal_naive_forecast(series.iloc[:5], 3)


def test_candidates_gate_seasonal_models():
    assert not any(c.seasonal_order[3] for c in fc.build_candidates(False))
    assert any(c.family == "SARIMAX" for c in fc.build_candidates(True))
    assert {c.family for c in fc.build_candidates(False)} == {"ARIMA", "ARIMAX"}


def test_run_selection_keeps_holdout_untouched(series, monkeypatch):
    small = [s for s in fc.build_candidates(True)
             if s.order in {(1, 0, 0), (0, 1, 1)} and (not s.seasonal_order[3] or s.seasonal_order == (1, 0, 0, 12))
             and s.exog_name in ("", "nov")]
    monkeypatch.setattr(fc, "build_candidates", lambda ok, s=12: small)
    st = ForecastSettings(test_periods=3, min_train_periods=18)
    res = fc.run_selection(series, st)
    assert len(res.test) == 3 and res.dev.index.max() < res.test.index.min()
    model_preds = res.val_preds[res.val_preds["family"] != "Benchmark"]
    assert model_preds["date"].max() <= res.dev.index.max()            # no validation row touches the hold-out
    assert (model_preds["origin"] < model_preds["date"]).all()          # forecasts are strictly out-of-sample
    assert res.selected is not None and res.rationale
    assert set(res.test_table["family"]) >= {"ARIMA", "Benchmark"}
    # the selected model was chosen from validation, so the chosen one is eligible
    lb = res.leaderboard.set_index("model")
    assert bool(lb.loc[res.selected.label, "eligible"])


def test_run_selection_rejects_short_series(series):
    with pytest.raises(fc.InsufficientDataError):
        fc.run_selection(series.iloc[:14], ForecastSettings())


def test_forecast_future_horizon_and_validation(series):
    spec = fc.ModelSpec("ARIMAX", (1, 0, 0), trend="c", exog=("is_oct", "is_nov"), exog_name="oct_nov")
    out, _ = fc.forecast_future(series, spec, 6)
    assert list(out.columns) == ["forecast", "lower_95", "upper_95"] and out.index[0] == pd.Timestamp("2021-01-01")
    assert out.index.is_monotonic_increasing
    with pytest.raises(ValueError):
        fc.forecast_future(series, spec, 0)
    with pytest.raises(fc.FutureRegressorError):
        fc.forecast_future(series, fc.ModelSpec("ARIMAX", (1, 0, 0), exog=("price",)), 3)
