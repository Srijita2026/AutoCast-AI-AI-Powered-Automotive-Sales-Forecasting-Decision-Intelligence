"""Reports & Downloads: cleaned data, forecasts, metrics and a PDF report."""
import json

import streamlit as st

from src import reporting, ui
from src.config import TARGET_LABELS

d = ui.ensure_data()
s = d["settings"]
ui.hero("Reports & Downloads", "Export cleaned data, model metrics, forecasts and a PDF summary.")

c1, c2 = st.columns(2)
c1.download_button("Cleaned data (CSV)", d["df"].to_csv(index=False), "auto_sales_clean.csv", "text/csv", width="stretch")
c2.download_button("Cleaning audit log (JSON)", json.dumps(d["log"], indent=2), "cleaning_audit_log.json", "application/json", width="stretch")
st.caption("Contact columns (phone, address, contact names) are excluded from the cleaned export.")

st.markdown("#### Forecasts and model metrics")
sel = ui.get_selection(d, run=False)
if sel is None:
    st.info("Run the model comparison to enable forecast, metric and PDF downloads.")
    if st.button("Run model comparison now"):
        ui.get_selection(d, run=True)
        st.rerun()
else:
    h = st.slider("Forecast horizon (months)", 1, 12, 6)
    forecast = None
    if sel.selected is not None:
        out = ui.get_forecast(d, sel.selected, h)
        if out is not None:
            forecast = out[0]
            dl = forecast.assign(model=sel.selected.label, target=s.target)
            st.download_button("Forecast with 95% intervals (CSV)", dl.to_csv(), f"forecast_{s.target}_{h}m.csv", "text/csv")
    else:
        st.warning("No model passed validation; no forecast to download.")
    st.download_button("Model comparison - all candidates (CSV)", sel.leaderboard.to_csv(index=False), "model_comparison.csv", "text/csv")
    st.download_button("Hold-out metrics (CSV)", sel.test_table.to_csv(index=False), "holdout_metrics.csv", "text/csv")

    st.markdown("#### PDF report")
    if st.button("Generate PDF report"):
        with st.spinner("Building PDF..."):
            _, notes = ui.get_series(d)
            pdf = reporting.build_pdf(d["df"], d["audit"], d["log"], sel, forecast, notes, TARGET_LABELS[s.target])
        st.session_state["pdf"] = pdf
    if st.session_state.get("pdf"):
        st.download_button("Download PDF report", st.session_state["pdf"], "autocast_report.pdf", "application/pdf")
