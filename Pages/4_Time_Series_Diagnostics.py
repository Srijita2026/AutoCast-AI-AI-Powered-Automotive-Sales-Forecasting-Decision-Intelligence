"""Time-Series Diagnostics for the chosen target series."""
import pandas as pd
import streamlit as st

from src import charts, diagnostics as dg, ui
from src.config import TARGET_LABELS

d = ui.ensure_data()
s = d["settings"]
y, notes = ui.get_series(d)
ui.hero("Time-Series Diagnostics", f"Target: {TARGET_LABELS[s.target]} aggregated monthly ({len(y)} observations).")

st.markdown("##### Series construction")
for n in notes:
    st.markdown(f"- {n}")
st.plotly_chart(charts.trend_line(y, "Target series", ui.target_name(s.target)), width="stretch")
dev = y.iloc[:-s.test_periods]
st.caption(f"Diagnostics below use the development part only ({len(dev)} months); the last {s.test_periods} months are held out.")

ev = dg.seasonal_evidence(dev, s)
st.markdown("##### Seasonality evidence")
(st.success if ev["supported"] else st.warning)(ev["reason"])

t1, t2, t3, t4 = st.tabs(["Decomposition", "Rolling statistics", "Stationarity (ADF)", "ACF / PACF & residuals"])
with t1:
    dec = dg.decompose(dev, s.seasonal_period)
    if dec is None:
        st.info(f"Decomposition needs >= 2 full cycles ({2 * s.seasonal_period} points); only {len(dev)} available.")
    else:
        st.plotly_chart(charts.decomposition_fig(dec), width="stretch")
        st.caption(f"Additive decomposition, period {dec.period}. Seasonal strength {dec.strength:.2f}. With two cycles the "
                   "seasonal component is mostly the November peak and should be read cautiously.")
with t2:
    w = st.slider("Rolling window (months)", 2, 12, 3)
    st.plotly_chart(charts.rolling_fig(dg.rolling_stats(y, w)), width="stretch")
    st.caption("A rolling std that rises with spikes suggests variance is not constant (a log transform could be explored).")
with t3:
    rows = []
    for label, series in (("Level", dev), ("First difference", dev.diff().dropna())):
        r = dg.adf_test(series)
        rows.append({"series": label, "n": r["n"], "ADF statistic": r["statistic"], "p-value": r["p_value"],
                     "lags used": r["used_lag"], "stationary at 5%": r["stationary_5pct"], "note": r["error"] or ""})
    st.dataframe(pd.DataFrame(rows).round(4), width="stretch", hide_index=True)
    st.caption("ADF has low power on ~26 points and is distorted by the November spikes. Candidate models therefore "
               "include both d=0 and d=1 and are chosen by validation error, not by this test alone.")
with t4:
    ap = dg.acf_pacf(dev)
    if ap.get("error"):
        st.warning(f"ACF/PACF unavailable: {ap['error']}")
    else:
        st.plotly_chart(charts.acf_pacf_fig(ap), width="stretch")
        st.caption(f"Dashed lines: approximate 95% bounds (+/-{ap['bound']:.2f}); few lags ({len(ap['lags']) - 1}) are estimable.")
    if dec is not None:
        rd = dg.residual_diagnostics(dec.resid)
        st.markdown("**Decomposition residual checks**")
        st.write(f"Ljung-Box p = {rd['ljung_box_p']:.3f}; Jarque-Bera p = {rd['jarque_bera_p']:.3f}. "
                 + ("No significant leftover autocorrelation." if rd["white_noise_5pct"] else "Residual autocorrelation remains."))
        st.plotly_chart(charts.residual_fig(dec.resid.dropna(), rd["acf"]), width="stretch")
    st.caption("Residual diagnostics of the fitted forecasting model are on the Model Lab page.")
