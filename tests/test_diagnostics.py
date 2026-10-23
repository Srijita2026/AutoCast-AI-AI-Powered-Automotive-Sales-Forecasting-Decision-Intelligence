import numpy as np
import pandas as pd

from src import diagnostics as dg
from src.config import ForecastSettings


def test_build_series_continuous_with_zero_months(clean_df, settings):
    d = clean_df[clean_df["year_month"] != pd.Timestamp("2019-05-01")]
    y, notes = dg.build_series(d, settings)
    assert y.index.freq is not None and len(y) == 36
    assert y.loc["2019-05-01"] == 0.0              # real zero-sales month preserved, not dropped/imputed
    assert any("no transactions" in n for n in notes)


def test_build_series_excludes_cancelled(clean_df):
    s_all, _ = dg.build_series(clean_df, ForecastSettings(exclude_statuses=()))
    s_ex, _ = dg.build_series(clean_df, ForecastSettings(exclude_statuses=("Cancelled",)))
    assert s_ex.sum() < s_all.sum()


def test_targets_differ(clean_df):
    rev, _ = dg.build_series(clean_df, ForecastSettings(target="revenue"))
    units, _ = dg.build_series(clean_df, ForecastSettings(target="units"))
    orders, _ = dg.build_series(clean_df, ForecastSettings(target="orders"))
    assert not np.allclose(rev, units) and orders.sum() < units.sum()


def test_seasonal_evidence_requires_two_cycles(series):
    ev = dg.seasonal_evidence(series.iloc[:20], ForecastSettings())
    assert not ev["supported"] and "two full cycles" in ev["reason"]
    assert dg.decompose(series.iloc[:20], 12) is None


def test_seasonal_evidence_detects_november_spike(series):
    ev = dg.seasonal_evidence(series, ForecastSettings())
    assert ev["supported"] and ev["strength"] > 0.5


def test_adf_handles_short_or_constant():
    assert dg.adf_test(pd.Series([1.0] * 20))["error"]
    assert dg.adf_test(pd.Series([1.0, 2.0, 3.0]))["error"]


def test_adf_stationary_noise():
    r = dg.adf_test(pd.Series(np.random.default_rng(0).standard_normal(60)))
    assert r["stationary_5pct"] is True and r["error"] is None


def test_acf_pacf_and_ljung_box(series):
    ap = dg.acf_pacf(series, 12)
    assert ap["acf"][0] == 1.0 and len(ap["acf"]) == len(ap["pacf"])
    rd = dg.residual_diagnostics(pd.Series(np.random.default_rng(2).standard_normal(40)))
    assert rd["white_noise_5pct"] in (True, False) and 0 <= rd["ljung_box_p"] <= 1


def test_frequency_check(clean_df):
    f = dg.frequency_check(clean_df)
    assert set(f["frequency"]) == {"Daily", "Weekly", "Monthly", "Quarterly"}
    assert f.set_index("frequency").loc["Monthly", "empty_periods_pct"] == 0
