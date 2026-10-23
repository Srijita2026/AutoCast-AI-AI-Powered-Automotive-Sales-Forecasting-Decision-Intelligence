"""Exploratory analysis helpers (pure pandas, no UI). Revenue, units and orders are kept distinct."""
from __future__ import annotations

import numpy as np
import pandas as pd

METRICS: dict[str, tuple[str, str]] = {
    "revenue": ("sales", "sum"),
    "units": ("quantity_ordered", "sum"),
    "orders": ("order_number", "nunique"),
}
METRIC_NAMES = {"revenue": "Revenue", "units": "Units sold", "orders": "Orders (distinct)"}


def filter_df(df: pd.DataFrame, start=None, end=None, statuses=None, countries=None,
              product_lines=None, deal_sizes=None) -> pd.DataFrame:
    """Apply optional filters; ``None`` means no restriction."""
    m = pd.Series(True, index=df.index)
    if start is not None:
        m &= df["order_date"] >= pd.Timestamp(start)
    if end is not None:
        m &= df["order_date"] <= pd.Timestamp(end)
    for col, vals in (("status", statuses), ("country", countries), ("product_line", product_lines),
                      ("deal_size", deal_sizes)):
        if vals is not None and col in df.columns:
            m &= df[col].isin(list(vals))
    return df.loc[m]


def agg(df: pd.DataFrame, by: str | list[str], metric: str) -> pd.DataFrame:
    """Aggregate ``metric`` (revenue|units|orders) by one or more columns -> DataFrame with 'value'."""
    if metric not in METRICS:
        raise ValueError(f"Unknown metric '{metric}'. Choose from {list(METRICS)}")
    col, fn = METRICS[metric]
    out = df.groupby(by, observed=True)[col].agg(fn).rename("value").reset_index()
    return out.sort_values("value", ascending=False).reset_index(drop=True)


def kpis(df: pd.DataFrame) -> dict:
    """Headline KPIs. Orders = distinct order numbers; lines = rows."""
    if df.empty:
        return {k: 0 for k in ("revenue", "units", "orders", "lines", "aov", "avg_unit_price", "customers", "countries")}
    rev, units, orders = df["sales"].sum(), df["quantity_ordered"].sum(), df["order_number"].nunique()
    return {"revenue": float(rev), "units": int(units), "orders": int(orders), "lines": int(len(df)),
            "aov": float(rev / orders) if orders else 0.0, "avg_unit_price": float(rev / units) if units else 0.0,
            "customers": int(df["customer_name"].nunique()), "countries": int(df["country"].nunique())}


def monthly_table(df: pd.DataFrame) -> pd.DataFrame:
    """Continuous monthly table; months without transactions are true zeros (no sales recorded)."""
    if df.empty:
        return pd.DataFrame(columns=["revenue", "units", "orders", "lines", "aov"])
    g = df.groupby("year_month").agg(revenue=("sales", "sum"), units=("quantity_ordered", "sum"),
                                     orders=("order_number", "nunique"), lines=("sales", "size"))
    idx = pd.date_range(g.index.min(), g.index.max(), freq="MS")
    g = g.reindex(idx).fillna(0)
    g.index.name = "month"
    g["aov"] = np.where(g["orders"] > 0, g["revenue"] / g["orders"].replace(0, np.nan), np.nan)
    return g


def growth_table(monthly: pd.DataFrame, col: str = "revenue") -> pd.DataFrame:
    """MoM and YoY % change; only computed where the comparison period is non-zero (otherwise NaN)."""
    s = monthly[col]
    prev_m, prev_y = s.shift(1), s.shift(12)
    return pd.DataFrame({
        "value": s,
        "mom_pct": np.where(prev_m > 0, (s / prev_m - 1) * 100, np.nan),
        "yoy_pct": np.where(prev_y > 0, (s / prev_y - 1) * 100, np.nan),
    }, index=s.index)


def top_bottom(df: pd.DataFrame, dim: str, metric: str, n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Top-n and bottom-n values of ``dim`` by metric."""
    t = agg(df, dim, metric)
    return t.head(n), t.tail(n).sort_values("value")


def unusual_months(monthly: pd.DataFrame, col: str = "revenue", z_cut: float = 3.5) -> pd.DataFrame:
    """Months far from the median using a robust (MAD) z-score; flagged, never removed."""
    s = monthly[col]
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return pd.DataFrame(columns=["month", col, "robust_z", "x_median"])
    z = 0.6745 * (s - med) / mad
    out = pd.DataFrame({"month": s.index, col: s.values, "robust_z": z.values, "x_median": (s / med).values})
    return out[out["robust_z"].abs() > z_cut].sort_values("robust_z", ascending=False).reset_index(drop=True)


def top_orders(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Largest orders by total revenue (an order can have several lines)."""
    g = df.groupby("order_number").agg(order_date=("order_date", "first"), customer=("customer_name", "first"),
                                       country=("country", "first"), lines=("sales", "size"),
                                       units=("quantity_ordered", "sum"), revenue=("sales", "sum"))
    return g.sort_values("revenue", ascending=False).head(n).reset_index()


def observations(df: pd.DataFrame) -> list[str]:
    """Data-supported observations. Wording is descriptive; no causal claims are made."""
    if df.empty:
        return ["No data in the current selection."]
    obs: list[str] = []
    m = monthly_table(df)
    if len(m) >= 3:
        peak = m["revenue"].idxmax()
        med = m["revenue"].median()
        obs.append(f"Peak month by revenue is {peak:%b %Y} (${m['revenue'].max():,.0f}), "
                   f"{m['revenue'].max() / med:.1f}x the median month (${med:,.0f}).")
        nov = m[m.index.month == 11]["revenue"]
        if len(nov) >= 2 and (nov > 2 * med).all():
            obs.append(f"November exceeded 2x the median month in each of the {len(nov)} Novembers observed "
                       "(a repeated pattern; the data cannot say why).")
    years = sorted(df["year"].unique())
    if len(years) >= 2:
        y1, y0 = years[-1], years[-2]
        months_y1 = set(df.loc[df["year"] == y1, "month"].unique())
        a = df[(df["year"] == y0) & (df["month"].isin(months_y1))]["sales"].sum()
        b = df[df["year"] == y1]["sales"].sum()
        if a > 0:
            span = f"{min(months_y1)}-{max(months_y1)}" if len(months_y1) > 1 else str(min(months_y1))
            obs.append(f"Like-for-like (calendar months {span}), revenue in {y1} was {(b / a - 1) * 100:+.1f}% "
                       f"vs {y0} (${b:,.0f} vs ${a:,.0f}).")
    for dim, label in (("country", "country"), ("product_line", "product line"), ("customer_name", "customer"),
                       ("deal_size", "deal size")):
        if dim in df.columns:
            t = agg(df, dim, "revenue")
            if len(t):
                obs.append(f"Top {label}: {t.iloc[0][dim]} with {t.iloc[0]['value'] / t['value'].sum():.1%} "
                           f"of revenue; top 3 together hold {t['value'].head(3).sum() / t['value'].sum():.1%}.")
    if "status" in df.columns:
        canc = df.loc[df["status"] == "Cancelled", "sales"].sum() / df["sales"].sum()
        obs.append(f"Cancelled orders represent {canc:.1%} of gross order value in this selection.")
    return obs
