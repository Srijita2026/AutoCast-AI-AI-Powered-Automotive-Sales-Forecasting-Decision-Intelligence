"""Plotly chart builders with a consistent dark automotive theme."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .diagnostics import Decomp

ACCENT, ACCENT2, ACCENT3, MUTED = "#ff4b2b", "#2bb8ff", "#ffc83d", "#8b95a7"
PALETTE = [ACCENT, ACCENT2, ACCENT3, "#7ee081", "#b48bff", "#ff8fb1", "#5ee0d0"]
COUNTRY_ISO = {"USA": "United States", "UK": "United Kingdom"}


def theme(fig: go.Figure, title: str | None = None, height: int = 380) -> go.Figure:
    """Apply the dark theme."""
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=height, margin=dict(l=10, r=10, t=50 if title else 20, b=10),
                      title=dict(text=title, x=0.01, font=dict(size=16)) if title else None,
                      legend=dict(orientation="h", y=1.08, x=0), colorway=PALETTE,
                      font=dict(family="Inter, Segoe UI, sans-serif"))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
    return fig


def trend_line(s: pd.Series, title: str, y_label: str, rolling: int | None = None) -> go.Figure:
    fig = go.Figure(go.Scatter(x=s.index, y=s.values, mode="lines+markers", name=y_label, line=dict(color=ACCENT, width=2.5)))
    if rolling and len(s) >= rolling:
        fig.add_scatter(x=s.index, y=s.rolling(rolling).mean(), name=f"{rolling}-mo rolling mean",
                        line=dict(color=ACCENT2, dash="dot"))
    fig.update_yaxes(title=y_label)
    return theme(fig, title)


def bar_h(df: pd.DataFrame, x: str, y: str, title: str, color: str = ACCENT, n: int | None = None) -> go.Figure:
    d = df.head(n) if n else df
    fig = go.Figure(go.Bar(x=d[x], y=d[y], orientation="h", marker_color=color))
    fig.update_yaxes(autorange="reversed")
    return theme(fig, title, height=max(260, 28 * len(d) + 90))


def growth_bars(g: pd.DataFrame, col: str, title: str) -> go.Figure:
    v = g[col]
    fig = go.Figure(go.Bar(x=g.index, y=v, marker_color=np.where(v.fillna(0) >= 0, "#7ee081", ACCENT)))
    fig.update_yaxes(title="% change")
    return theme(fig, title)


def month_year_heatmap(monthly: pd.DataFrame, col: str, title: str) -> go.Figure:
    t = monthly[col].to_frame("v")
    t["year"], t["month"] = t.index.year, t.index.month
    p = t.pivot(index="year", columns="month", values="v").reindex(columns=range(1, 13))
    fig = go.Figure(go.Heatmap(z=p.values, x=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                               y=p.index.astype(str), colorscale="Inferno", hoverongaps=False))
    return theme(fig, title, height=260)


def country_map(df: pd.DataFrame, name: str = "country", value: str = "value", title: str = "") -> go.Figure:
    d = df.copy()
    d["loc"] = d[name].replace(COUNTRY_ISO)
    fig = go.Figure(go.Choropleth(locations=d["loc"], locationmode="country names", z=d[value], colorscale="Inferno",
                                  marker_line_color="#222", colorbar=dict(title="")))
    fig.update_geos(bgcolor="rgba(0,0,0,0)", showframe=False, showcoastlines=False, projection_type="natural earth")
    return theme(fig, title, height=380)


def decomposition_fig(dec: Decomp) -> go.Figure:
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=("Observed", "Trend", f"Seasonal (period {dec.period})", "Residual"))
    for i, (s, c) in enumerate(((dec.observed, ACCENT), (dec.trend, ACCENT2), (dec.seasonal, ACCENT3), (dec.resid, MUTED)), 1):
        fig.add_scatter(x=s.index, y=s.values, row=i, col=1, line=dict(color=c), showlegend=False)
    return theme(fig, None, 620)


def rolling_fig(rs: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("Value & rolling mean", "Rolling std"))
    fig.add_scatter(x=rs.index, y=rs["value"], name="value", row=1, col=1, line=dict(color=MUTED))
    fig.add_scatter(x=rs.index, y=rs["rolling_mean"], name="rolling mean", row=1, col=1, line=dict(color=ACCENT))
    fig.add_scatter(x=rs.index, y=rs["rolling_std"], name="rolling std", row=2, col=1, line=dict(color=ACCENT2))
    return theme(fig, None, 460)


def acf_pacf_fig(ap: dict) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("ACF", "PACF"))
    for i, k in enumerate(("acf", "pacf"), 1):
        fig.add_bar(x=ap["lags"], y=ap[k], marker_color=ACCENT if i == 1 else ACCENT2, row=1, col=i, showlegend=False)
        for sign in (1, -1):
            fig.add_hline(y=sign * ap["bound"], line_dash="dash", line_color=MUTED, row=1, col=i)
    return theme(fig, None, 320)


def residual_fig(resid: pd.Series, ap: dict | None) -> go.Figure:
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Residuals over time", "Histogram", "Residual ACF"))
    fig.add_scatter(x=resid.index, y=resid.values, mode="lines+markers", line=dict(color=ACCENT), row=1, col=1, showlegend=False)
    fig.add_histogram(x=resid.values, marker_color=ACCENT2, row=1, col=2, showlegend=False)
    if ap and not ap.get("error"):
        fig.add_bar(x=ap["lags"], y=ap["acf"], marker_color=ACCENT3, row=1, col=3, showlegend=False)
        for sign in (1, -1):
            fig.add_hline(y=sign * ap["bound"], line_dash="dash", line_color=MUTED, row=1, col=3)
    return theme(fig, None, 320)


def forecast_fig(history: pd.Series, fc: pd.DataFrame, title: str, y_label: str, test: pd.Series | None = None,
                 scenario: pd.Series | None = None, history_months: int = 36) -> go.Figure:
    h = history.tail(history_months)
    fig = go.Figure()
    fig.add_scatter(x=h.index, y=h.values, name="Actual", line=dict(color="#e6e9ef", width=2))
    fig.add_scatter(x=fc.index, y=fc["upper_95"], line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=fc.index, y=fc["lower_95"], fill="tonexty", fillcolor="rgba(255,75,43,0.18)", line=dict(width=0),
                    name="95% prediction interval", hoverinfo="skip")
    fig.add_scatter(x=fc.index, y=fc["forecast"], name="Forecast", line=dict(color=ACCENT, width=3), mode="lines+markers")
    if scenario is not None:
        fig.add_scatter(x=scenario.index, y=scenario.values, name="Scenario (assumption)", line=dict(color=ACCENT3, dash="dash"))
    fig.update_yaxes(title=y_label)
    return theme(fig, title, 430)


def actual_vs_pred(df: pd.DataFrame, title: str, y_label: str) -> go.Figure:
    """df columns: date, actual, pred, (lower, upper); several origins are drawn as separate prediction traces."""
    fig = go.Figure()
    act = df.drop_duplicates("date").sort_values("date")
    fig.add_scatter(x=act["date"], y=act["actual"], name="Actual", line=dict(color="#e6e9ef", width=2))
    if "origin" in df.columns:
        for i, (o, g) in enumerate(df.groupby("origin")):
            fig.add_scatter(x=g["date"], y=g["pred"], name=f"Forecast from {o:%b %Y}", mode="lines+markers",
                            line=dict(color=PALETTE[i % len(PALETTE)], dash="dot"), opacity=0.8)
    else:
        fig.add_scatter(x=df["date"], y=df["pred"], name="Predicted", mode="lines+markers", line=dict(color=ACCENT))
        if "lower" in df and df["lower"].notna().any():
            fig.add_scatter(x=df["date"], y=df["upper"], line=dict(width=0), showlegend=False, hoverinfo="skip")
            fig.add_scatter(x=df["date"], y=df["lower"], fill="tonexty", fillcolor="rgba(255,75,43,0.18)",
                            line=dict(width=0), name="95% interval", hoverinfo="skip")
    fig.update_yaxes(title=y_label)
    return theme(fig, title, 400)


def metric_bars(df: pd.DataFrame, metric: str, title: str) -> go.Figure:
    d = df.dropna(subset=[metric]).sort_values(metric)
    fig = go.Figure(go.Bar(x=d[metric], y=d["model"], orientation="h",
                           marker_color=[ACCENT if f != "Benchmark" else MUTED for f in d["family"]]))
    fig.update_yaxes(autorange="reversed")
    return theme(fig, title, height=max(260, 46 * len(d) + 90))
