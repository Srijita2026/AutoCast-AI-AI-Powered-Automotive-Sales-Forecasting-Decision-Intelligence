"""Executive Overview: headline KPIs, trend, observations and (if computed) the forecast headline."""
import streamlit as st

from src import charts, eda, ui

d = ui.ensure_data()
s = d["settings"]
df = ui.base_df(d)
ui.hero("Executive Overview", f"Vehicle-model sales, {d['df']['order_date'].min():%b %Y} - {d['df']['order_date'].max():%b %Y}. "
        f"Basis: {'excludes ' + ', '.join(s.exclude_statuses) if s.exclude_statuses else 'all statuses'}.")

k = eda.kpis(df)
c = st.columns(6)
ui.kpi(c[0], "Revenue", ui.money(k["revenue"]), "sum of line sales")
ui.kpi(c[1], "Units sold", f"{k['units']:,}", "sum of quantity")
ui.kpi(c[2], "Orders", f"{k['orders']:,}", "distinct order numbers")
ui.kpi(c[3], "Avg order value", ui.money(k["aov"]), "revenue / orders")
ui.kpi(c[4], "Customers", f"{k['customers']:,}", f"{k['lines']:,} order lines")
ui.kpi(c[5], "Countries", f"{k['countries']}", "with orders")

m = eda.monthly_table(df)
st.markdown(ui.tag("hist"), unsafe_allow_html=True)
st.plotly_chart(charts.trend_line(m["revenue"], "Monthly revenue", "Revenue ($)", rolling=3), width="stretch")

l, r = st.columns(2)
with l:
    st.markdown("##### Data-supported observations")
    for o in eda.observations(df):
        st.markdown(f"- {o}")
    st.caption("Observations are descriptive associations; they do not establish causes.")
with r:
    pl = eda.agg(df, "product_line", "revenue")
    st.plotly_chart(charts.bar_h(pl, "value", "product_line", "Revenue by product line"), width="stretch")

st.divider()
st.markdown("#### Forecast headline")
sel = ui.get_selection(d, run=False)
if sel is None or sel.selected is None:
    st.info("Forecasts have not been computed in this session. Open **Model Lab** to run the model comparison "
            "(about 30 s, then cached), or **Forecast Studio** to choose a horizon.")
else:
    out = ui.get_forecast(d, sel.selected, 3)
    if out is not None:
        fc, _, _, _ = out
        st.markdown(ui.tag("fc"), unsafe_allow_html=True)
        c = st.columns(4)
        ui.kpi(c[0], "Selected model", sel.selected.family, sel.selected.label)
        ui.kpi(c[1], f"Next 3 months ({ui.target_name(s.target)})", ui.fmt_target(fc["forecast"].sum(), s.target),
               "monthly 95% intervals in Forecast Studio")
        ui.kpi(c[2], "Validation WAPE", f"{sel.family_best.set_index('model').loc[sel.selected.label, 'wape']:.1%}", "rolling-origin")
        t = sel.test_table.set_index("model").loc[sel.selected.label]
        ui.kpi(c[3], "Hold-out WAPE", f"{t['wape']:.1%}", f"{len(sel.test)} months, reported after selection")
        st.plotly_chart(charts.forecast_fig(sel.series, fc, "Forecast (selected model)", ui.target_name(s.target)), width="stretch")
        for n in sel.notes[:2]:
            st.caption("⚠ " + n)
