import numpy as np
import pandas as pd
import pytest

from src import data as dm


def test_to_snake_aliases():
    assert dm.to_snake("QUANTITYORDERED") == "quantity_ordered"
    assert dm.to_snake("Order Date") == "order_date"
    assert dm.to_snake("days_since_lastorder") == "days_since_last_order"


def test_detect_date_format_dayfirst(raw_df):
    assert dm.detect_date_format(raw_df["ORDERDATE"]) == "%d/%m/%Y"
    assert dm.detect_date_format(pd.Series(["nonsense", "foo"])) is None


def test_clean_does_not_mutate_raw_and_parses_dates(raw_df):
    before = raw_df.copy(deep=True)
    out, log = dm.clean(raw_df)
    pd.testing.assert_frame_equal(raw_df, before)
    assert pd.api.types.is_datetime64_any_dtype(out["order_date"])
    assert out["order_date"].dt.day.max() > 12 and out["month"].between(1, 12).all()
    assert any(r["step"] == "parse_dates" for r in log)


def test_clean_excludes_pii_but_keeps_outliers_and_zero_months(raw_df):
    raw_df.loc[0, "SALES"] = 1e6   # genuine spike must survive, but be flagged
    out, log = dm.clean(raw_df)
    assert len(out) == len(raw_df)
    assert out["flag_sales_outlier"].iloc[out["sales"].idxmax()]
    for c in ("phone", "address_line1", "contact_last_name", "contact_first_name"):
        assert c not in out.columns


def test_exact_duplicates_dropped_and_logged(raw_df):
    dup = pd.concat([raw_df, raw_df.iloc[[0, 1]]], ignore_index=True)
    out, log = dm.clean(dup)
    assert len(out) == len(raw_df)
    assert [r for r in log if r["step"] == "drop_exact_duplicates"][0]["rows_affected"] == 2


def test_missing_sales_dropped_missing_numeric_not_zero_filled(raw_df):
    raw_df = raw_df.copy()
    raw_df.loc[3, "SALES"] = np.nan
    raw_df.loc[5, "MSRP"] = np.nan
    out, log = dm.clean(raw_df)
    assert len(out) == len(raw_df) - 1
    assert out["msrp"].isna().sum() == 1          # left as NaN, not 0
    assert any(r["step"] == "drop_invalid_rows" for r in log)


def test_missing_date_column_raises():
    with pytest.raises(dm.DataLoadError):
        dm.clean(pd.DataFrame({"SALES": [1.0, 2.0], "X": ["a", "b"]}))


def test_audit_reports_shape_and_suspicious(raw_df):
    out, _ = dm.clean(raw_df)
    a = dm.audit(out, raw_df)
    assert a["shape"] == out.shape and a["duplicates_exact"] == 0
    assert {"check", "rows"} <= set(a["suspicious"].columns)
    assert "sales" in a["schema"]["target_candidates"]


def test_load_raw_missing_file(tmp_path):
    with pytest.raises(dm.DataLoadError):
        dm.load_raw(tmp_path / "nope.csv")


def test_find_csv(tmp_path):
    assert dm.find_csv(tmp_path) is None
    (tmp_path / "x.csv").write_text("a\n1\n")
    assert dm.find_csv(tmp_path).name == "x.csv"
