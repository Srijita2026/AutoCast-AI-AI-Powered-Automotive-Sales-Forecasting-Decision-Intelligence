"""Sales Analytics: filterable EDA. Filters here never trigger model retraining."""
import pandas as pd
import streamlit as st

from src import charts, eda, ui

d = ui.ensure_data()
df0 = d["df"]
ui.hero("Sales Analytics", "Revenue, units and orders are different measures - pick one below; they are never mixed.")

with st.expander("Filters", expanded=True):
    f1, f2, f3, f4 = st.columns(4)
    metric = f1.selectbox("Measure", list(eda.METRICS), format_func=lambda k: eda.METRIC_NAMES[k])
    dmin, dmax = df0["order_date"].min().date(), df0["order_date"].max().date()
    rng = f2.date_input("Order date range", (dmin, dmax), min_value=dmin, max_value=dmax)
    default_status = [s for s in sorted(df0["status"].unique()) if s not in d["settings"].exclude_statuses]
    statuses = f3.multiselect("Status", sorted(df0["status"].unique()), default=default_status)
    deals = f4.multiselect("Deal size", sorted(df0["deal_size"].unique()), default=sorted(df0["deal_size"].unique()))
    g1, g2 = st.columns(2)
    countries = g1.multiselect("Country", sorted(df0["country"].unique()))
    lines = g2.multiselect("Product line", sorted(df0["product_line"].unique()))

start, end = (rng if isinstance(rng, tuple) and len(rng) == 2 else (dmin, dmax))
df = eda.filter_df(df0, start, end, statuses or None, countries or None, lines or None, deals or None)
if df.empty:
    st.warning("No rows match these filters.")
    st.stop()

k = eda.kpis(df)
c = st.columns(5)
ui.kpi(c[0], "Revenue", ui.money(k["revenue"]))
ui.kpi(c[1], "Units", f"{k['units']:,}")
ui.kpi(c[2], "Orders", f"{k['orders']:,}")
ui.kpi(c[3], "Avg order value", ui.money(k["aov"]))
ui.kpi(c[4], "Avg price / unit", f"${k['avg_unit_price']:,.2f}")
st.markdown(ui.tag("hist"), unsafe_allow_html=True)

m = eda.monthly_table(df)
name = eda.METRIC_NAMES[metric]
tab1, tab2, tab3, tab4 = st.tabs(["Trends", "Categories & geography", "Top / bottom", "Unusual observations"])
with tab1:
    st.plotly_chart(charts.trend_line(m[metric], f"Monthly {name.lower()}", name, rolling=3), width="stretch")
    g = eda.growth_table(m, metric)
    a1, a2 = st.columns(2)
    a1.plotly_chart(charts.growth_bars(g, "mom_pct", "Month-over-month % (blank where prior month is zero)"), width="stretch")
    a2.plotly_chart(charts.growth_bars(g, "yoy_pct", "Year-over-year % (needs same month a year earlier)"), width="stretch")
    st.plotly_chart(charts.month_year_heatmap(m, metric, f"{name} by month and year"), width="stretch")
    if m["lines"].gt(0).sum() < 13:
        st.caption("Filters leave fewer than 13 active months, so year-over-year views are limited.")
    q = df.groupby(df["order_date"].dt.to_period("Q").astype(str)).apply(lambda x: eda.kpis(x)[metric if metric != "units" else "units"], include_groups=False)
    st.dataframe(q.rename(name).to_frame().T, width="stretch")
with tab2:
    dim = st.selectbox("Compare by", ["product_line", "country", "deal_size", "customer_name", "status", "year"])
    t = eda.agg(df, dim, metric)
    t[dim] = t[dim].astype(str)
    l, r = st.columns(2)
    l.plotly_chart(charts.bar_h(t, "value", dim, f"{name} by {dim}", n=15), width="stretch")
    if dim == "country":
        r.plotly_chart(charts.country_map(t, "country", "value", f"{name} by country"), width="stretch")
    else:
        share = t.assign(share=lambda x: (x["value"] / x["value"].sum() * 100).round(1))
        r.dataframe(share, width="stretch", hide_index=True)
    st.markdown("**Mix over time**")
    mix = df.groupby(["year_month", "product_line"])[eda.METRICS[metric][0]].agg(eda.METRICS[metric][1]).reset_index()
    import plotly.express as px
    st.plotly_chart(charts.theme(px.area(mix, x="year_month", y=eda.METRICS[metric][0], color="product_line",
                                         color_discrete_sequence=charts.PALETTE), "Product-line mix"), width="stretch")
with tab3:
    dim2 = st.selectbox("Rank", ["customer_name", "product_code", "country", "product_line", "city"], key="rank")
    top, bot = eda.top_bottom(df, dim2, metric, 10)
    l, r = st.columns(2)
    l.markdown("**Top 10**"); l.dataframe(top, width="stretch", hide_index=True)
    r.markdown("**Bottom 10** (smallest non-zero totals)"); r.dataframe(bot, width="stretch", hide_index=True)
    st.markdown("**Largest orders**")
    st.dataframe(eda.top_orders(df, 10), width="stretch", hide_index=True)
with tab4:
    u = eda.unusual_months(m, metric)
    if u.empty:
        st.success("No months exceed the robust z-score threshold (3.5) in this selection.")
    else:
        st.dataframe(u, width="stretch", hide_index=True)
        st.caption("Flagged months are genuine data, kept as-is. The data alone cannot explain why they stand out.")
    out = df[df["flag_sales_outlier"]]
    st.write(f"Line-level sales outliers (IQR rule): **{len(out)}** lines, {out['sales'].sum() / df['sales'].sum():.1%} of revenue.")
    st.markdown("**Observations**")
    for o in eda.observations(df):
        st.markdown(f"- {o}")
