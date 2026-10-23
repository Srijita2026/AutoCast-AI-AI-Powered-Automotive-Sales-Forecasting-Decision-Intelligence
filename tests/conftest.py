"""Shared fixtures: small synthetic datasets with the same schema as the Kaggle file."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data as dm  # noqa: E402
from src.config import ForecastSettings  # noqa: E402


@pytest.fixture
def raw_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    months = pd.date_range("2018-01-01", periods=36, freq="MS")
    rows, order = [], 10000
    for m in months:
        n_orders = 3 + (6 if m.month == 11 else 0)
        for _ in range(n_orders):
            order += 1
            for line in range(1, 3):
                q, p = int(rng.integers(10, 50)), float(rng.integers(40, 150))
                day = int(rng.integers(13, 28))        # day > 12 => unambiguous dd/mm/YYYY
                rows.append({"ORDERNUMBER": order, "QUANTITYORDERED": q, "PRICEEACH": p, "ORDERLINENUMBER": line,
                             "SALES": round(q * p, 2), "ORDERDATE": f"{day:02d}/{m.month:02d}/{m.year}",
                             "DAYS_SINCE_LASTORDER": 100, "STATUS": "Cancelled" if order % 17 == 0 else "Shipped",
                             "PRODUCTLINE": ["Classic Cars", "Planes", "Ships"][order % 3], "MSRP": 100,
                             "PRODUCTCODE": "S10_1", "CUSTOMERNAME": f"Cust{order % 5}", "PHONE": "555",
                             "ADDRESSLINE1": "1 Main St", "CITY": "Paris", "POSTALCODE": "75000",
                             "COUNTRY": ["USA", "France", "UK"][order % 3], "CONTACTLASTNAME": "X",
                             "CONTACTFIRSTNAME": "Y", "DEALSIZE": "Small"})
    return pd.DataFrame(rows)


@pytest.fixture
def clean_df(raw_df):
    return dm.clean(raw_df)[0]


@pytest.fixture
def settings() -> ForecastSettings:
    return ForecastSettings(exclude_statuses=("Cancelled",), test_periods=3, min_train_periods=18)


@pytest.fixture
def series() -> pd.Series:
    """36 months: level 100 + yearly November spike + noise."""
    rng = np.random.default_rng(1)
    idx = pd.date_range("2018-01-01", periods=36, freq="MS")
    v = 100 + 5 * rng.standard_normal(36) + np.where(idx.month == 11, 150, 0)
    return pd.Series(v, index=idx, name="revenue")
