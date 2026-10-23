"""Data Quality: audit of the raw data and the cleaning log."""
import json

import pandas as pd
import plotly.express as px
import streamlit as st

from src import charts, diagnostics as dg, ui

d = ui.ensure_data()
a, raw, df = d["audit"], d["raw"], d["df"]
ui.hero("Data Quality", "What is in the file, what looked suspicious, and exactly what was changed.")

c = st.columns(5)
ui.kpi(c[0], "Rows x columns (raw)", f"{raw.shape[0]:,} x {raw.shape[1]}", "as loaded")
ui.kpi(c[1], "Date range", f"{a['date_range'][0]:%b %Y} - {a['date_range'][1]:%b %Y}", f"{a['n_distinct_dates']} distinct order dates")
ui.kpi(c[2], "Missing cells", f"{int(a['missing']['missing'].sum()):,}", "raw file")
ui.kpi(c[3], "Exact duplicate rows", f"{a['duplicates_exact']:,}", "raw file")
ui.kpi(c[4], "Rows after cleaning", f"{len(df):,}", "no rows dropped" if len(df) == len(raw) else f"{len(raw) - len(df)} dropped")

t1, t2, t3, t4, t5, t6 = st.tabs(["Schema", "Missing & duplicates", "Categories", "Suspicious values", "Distributions", "Cleaning audit log"])
with t1:
    sc = a["schema"]
    st.write("**Detected date columns:**", ", ".join(sc["date_columns"]))
    st.write("**Candidate sales targets:**", ", ".join(sc["target_candidates"]),
             "- chosen: `sales` (revenue); `quantity_ordered` (units); orders = distinct `order_number`.")
    st.dataframe(a["dtypes"], width="stretch", hide_index=True)
    st.markdown("**Frequency check** (why monthly aggregation):")
    st.dataframe(dg.frequency_check(df), width="stretch", hide_index=True)
    st.caption("Daily data is mostly empty periods and weekly is noisy; monthly has no gaps but only ~29 points.")
with t2:
    st.dataframe(a["missing"], width="stretch", hide_index=True)
    st.write(f"Exact duplicate rows: **{a['duplicates_exact']}**. Missing values are never blindly zero-filled; "
             "rows without a valid date/sales would be dropped and logged.")
with t3:
    cat = st.selectbox("Category column", list(a["categories"]))
    st.plotly_chart(charts.bar_h(a["categories"][cat], "rows", cat, f"Order lines by {cat}"), width="stretch")
with t4:
    st.dataframe(a["suspicious"], width="stretch", hide_index=True)
    for n in a["notes"]:
        st.markdown(f"- {n}")
    flag = st.selectbox("Inspect flagged rows", ["flag_price_high", "flag_price_low", "flag_sales_outlier"])
    cols = ["order_number", "order_date", "product_line", "customer_name", "quantity_ordered", "price_each", "msrp", "sales"]
    st.dataframe(df.loc[df[flag], cols].head(200), width="stretch", hide_index=True)
    st.caption("Flagged rows are kept: large lines and unusual prices can be genuine. They are flagged, never deleted.")
with t5:
    num = [c for c in a["schema"]["numeric_columns"] if c not in ("year", "month", "order_number")]
    col = st.selectbox("Numeric column", num, index=num.index("sales") if "sales" in num else 0)
    st.plotly_chart(charts.theme(px.histogram(df, x=col, nbins=50, color_discrete_sequence=[charts.ACCENT]), f"Distribution of {col}"),
                    width="stretch")
    st.dataframe(a["numeric_summary"], width="stretch")
with t6:
    log = pd.DataFrame(d["log"])
    st.dataframe(log, width="stretch", hide_index=True)
    st.download_button("Download audit log (JSON)", json.dumps(d["log"], indent=2), "cleaning_audit_log.json", "application/json")
    st.caption("The raw CSV in data/raw is never modified; cleaned output is written to data/processed.")
