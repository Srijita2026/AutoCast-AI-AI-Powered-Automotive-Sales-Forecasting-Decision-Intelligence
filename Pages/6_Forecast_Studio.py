"""Forecast Studio: choose horizon/model, view intervals, apply labelled what-if scenarios, download."""
import pandas as pd
import streamlit as st

from src import charts, ui

d = ui.ensure_data()
s = d["settings"]
ui.hero("Forecast Studio", "Future forecasts with 95% prediction intervals from validated models.")
sel = ui.get_selection(d, run=True)
if not sel.family_specs:
    st.error("No model passed validation, so no forecast is offered. See Model Lab for the failure log.")
    st.stop()

c1, c2 = st.columns([1, 2])
horizon = c1.slider("Forecast horizon (months)", 1, 24, 6)
labels = {sp.label: sp for sp in sel.family_specs.values()}
names = list(labels)
default = names.index(sel.selected.label) if sel.selected and sel.selected.label in names else 0
choice = c2.selectbox("Model", names, index=default,
                      format_func=lambda n: n + ("  (selected)" if sel.selected and n == sel.selected.label else ""))
spec = labels[choice]

res = ui.get_forecast(d, spec, horizon)
if res is None:
    st.stop()
fc, _, aic, bic = res

if spec.exog:
    st.info(f"Regressors **{', '.join(spec.exog)}** are calendar indicators (October/November flags). Their future values are "
            "known from the calendar, so no regressor needs to be forecast or guessed. Other candidate regressors "
            "(price, quantity, orders) are unknown in advance and were deliberately not used.")
if sel.selected and spec.label != sel.selected.label:
    st.warning("This is not the model chosen by validation; shown for comparison.")
if horizon > 6:
    st.caption("Long horizons with ~2 years of history are weakly supported; intervals widen but probably still understate risk.")

st.markdown(ui.tag("fc"), unsafe_allow_html=True)
adj = st.slider("What-if: apply an adjustment to the forecast (%)", -30, 30, 0,
                help="A flat multiplier you choose. It is an assumption, not a model estimate.")
scen = None
if adj != 0:
    scen = fc["forecast"] * (1 + adj / 100)
fig = charts.forecast_fig(sel.series, fc, f"{ui.target_name(s.target)} forecast - {spec.label}", ui.target_name(s.target), scenario=scen)
st.plotly_chart(fig, width="stretch")
if scen is not None:
    st.markdown(ui.tag("sc") + f" Scenario = forecast x {1 + adj / 100:.2f}. Interval bounds are shown for the unadjusted forecast only.",
                unsafe_allow_html=True)

k = st.columns(3)
ui.kpi(k[0], f"Total next {horizon} months", ui.fmt_target(fc["forecast"].sum(), s.target), "sum of monthly point forecasts")
if scen is not None:
    ui.kpi(k[1], "Scenario total", ui.fmt_target(scen.sum(), s.target), f"{adj:+d}% assumption")
ui.kpi(k[2], "Interval for first month", f"{ui.fmt_target(fc['lower_95'].iloc[0], s.target)} - {ui.fmt_target(fc['upper_95'].iloc[0], s.target)}", "95% prediction interval")

table = fc.round(0).copy()
if scen is not None:
    table["scenario_assumption"] = scen.round(0)
table.index = table.index.strftime("%Y-%m")
st.dataframe(table, width="stretch")
dl = fc.copy()
dl["model"] = spec.label
dl["target"] = s.target
if scen is not None:
    dl["scenario_assumption"] = scen
st.download_button("Download forecast (CSV)", dl.to_csv(), f"forecast_{s.target}_{horizon}m.csv", "text/csv")
st.caption("Prediction intervals assume the fitted model is correct. In rolling validation, nominal 95% intervals covered "
           "fewer actuals than 95% - see Model Lab.")
