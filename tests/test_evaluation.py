import numpy as np
import pytest

from src import evaluation as ev


def test_metrics_exact():
    m = ev.compute_metrics([100, 200, 300], [110, 180, 330])
    assert m["mae"] == pytest.approx(20.0)
    assert m["rmse"] == pytest.approx(np.sqrt((100 + 400 + 900) / 3))
    assert m["wape"] == pytest.approx(60 / 600)
    assert m["mape"] == pytest.approx((0.1 + 0.1 + 0.1) / 3)
    assert m["bias"] == pytest.approx(20 / 3)


def test_mape_not_reported_with_zero_actuals():
    m = ev.compute_metrics([0, 10, 20], [1, 9, 22])
    assert np.isnan(m["mape"]) and np.isfinite(m["wape"])


def test_empty_and_nan_inputs():
    assert ev.compute_metrics([], [])["n"] == 0
    assert ev.compute_metrics([1, np.nan], [1, 2])["n"] == 1


def test_interval_coverage():
    assert ev.interval_coverage([1, 5, 10], [0, 0, 0], [2, 6, 9]) == pytest.approx(2 / 3)
    assert np.isnan(ev.interval_coverage([1], [np.nan], [np.nan]))
