"""Model Lab: chronological model comparison (ARIMA / SARIMA / ARIMAX / SARIMAX)."""
import pandas as pd
import streamlit as st

from src import charts, diagnostics as dg, evaluation as ev, ui

d = ui.ensure_data()
s = d["settings"]
ui.hero("Model Lab", "Rolling-origin validation picks the model; the hold-out is scored only after selection.")
sel = ui.get_selection(d, run=True)

c = st.columns(4)
ui.kpi(c[0], "Selected model", sel.selected.family if sel.selected else "none", sel.selected.label if sel.selected else "no valid model")
ui.kpi(c[1], "Validation folds", f"{len(sel.origins)}", f"expanding window, {s.val_horizon}-month horizon")
ui.kpi(c[2], "Development / hold-out", f"{len(sel.dev)} / {len(sel.test)}", f"hold-out {sel.test.index.min():%b %Y} - {sel.test.index.max():%b %Y}")
ui.kpi(c[3], "Candidates fitted", f"{len(sel.leaderboard)}", f"{int(sel.leaderboard['eligible'].sum())} eligible")

st.markdown("##### Selection rationale")
st.info(sel.rationale)
for n in sel.notes:
    st.caption("⚠ " + n)
st.caption("Seasonality gate: " + sel.seasonal_evidence["reason"])

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Comparison", "Validation charts", "Hold-out test", "Residual diagnostics", "Leaderboard & failures"])
with tab1:
    fb = sel.family_best.copy()
    show = fb.drop(columns=[c for c in ("folds_failed",) if c in fb])
    st.dataframe(show.round(3), width="stretch", hide_index=True)
    st.caption("MAE/RMSE in target units. WAPE = sum|error| / sum|actual|. MAPE is shown only when all actuals are strictly "
               "positive. Coverage = share of validation actuals inside the 95% interval. AIC/BIC are diagnostics from a fit "
               "on the development set; they are not used to pick the model.")
    vm = sel.val_preds.groupby("model", sort=False).apply(lambda g: pd.Series(ev.compute_metrics(g["actual"], g["pred"])), include_groups=False)
    fam = sel.val_preds.drop_duplicates("model").set_index("model")["family"]
    vm = vm.assign(family=fam).reset_index()
    metric = st.radio("Metric chart", ["wape", "rmse", "mae"], horizontal=True, index=["wape", "rmse", "mae"].index(s.primary_metric))
    st.plotly_chart(charts.metric_bars(vm, metric, f"Validation {metric.upper()} (grey = naive benchmarks)"), width="stretch")
with tab2:
    models = list(dict.fromkeys(sel.val_preds["model"]))
    default = models.index(sel.selected.label) if sel.selected and sel.selected.label in models else 0
    m = st.selectbox("Model", models, index=default)
    st.plotly_chart(charts.actual_vs_pred(sel.val_preds[sel.val_preds["model"] == m], f"Rolling-origin forecasts - {m}",
                                          ui.target_name(s.target)), width="stretch")
    st.caption("Each dotted line is a 3-month forecast issued from a different origin using only data available then.")
with tab3:
    st.markdown("Scored **after** selection on months never used for choosing models.")
    tt = sel.test_table.copy()
    st.dataframe(tt.round(3), width="stretch", hide_index=True)
    m2 = st.selectbox("Model", list(dict.fromkeys(sel.test_preds["model"])), key="t")
    tp = sel.test_preds[sel.test_preds["model"] == m2]
    hist = sel.dev.tail(12)
    fig = charts.actual_vs_pred(tp, f"Hold-out: {m2}", ui.target_name(s.target))
    fig.add_scatter(x=hist.index, y=hist.values, name="Recent history", line=dict(color=charts.MUTED))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Only {len(sel.test)} hold-out points: treat these scores as indicative, not conclusive.")
with tab4:
    if sel.selected is None:
        st.info("No selected model.")
    else:
        out = ui.get_forecast(d, sel.selected, 1)
        if out is not None:
            _, resid, aic, bic = out
            rd = dg.residual_diagnostics(resid, model_df=sel.selected.n_params)
            a, b, c3 = st.columns(3)
            a.metric("Ljung-Box p-value", f"{rd['ljung_box_p']:.3f}" if pd.notna(rd["ljung_box_p"]) else "n/a")
            b.metric("Jarque-Bera p-value", f"{rd['jarque_bera_p']:.3f}" if pd.notna(rd["jarque_bera_p"]) else "n/a")
            c3.metric("AIC / BIC (full data)", f"{aic:,.0f} / {bic:,.0f}")
            if rd["white_noise_5pct"] is not None:
                (st.success if rd["white_noise_5pct"] else st.warning)(
                    "No significant residual autocorrelation (Ljung-Box, 5%)." if rd["white_noise_5pct"]
                    else "Residual autocorrelation remains: the model may be missing structure.")
            st.plotly_chart(charts.residual_fig(resid, rd["acf"]), width="stretch")
            st.caption("In-sample one-step residuals of the selected model refitted on all data. Large residuals at the "
                       "November peaks mean the size of the peak is only partly captured.")
with tab5:
    fams = st.multiselect("Family", ["ARIMA", "SARIMA", "ARIMAX", "SARIMAX"], default=["ARIMA", "SARIMA", "ARIMAX", "SARIMAX"])
    lb = sel.leaderboard[sel.leaderboard["family"].isin(fams)]
    st.dataframe(lb.round(3), width="stretch", hide_index=True)
    st.download_button("Download model comparison (CSV)", lb.to_csv(index=False), "model_comparison.csv", "text/csv")
    st.markdown("**Failed fits and reasons**")
    if sel.failures.empty:
        st.success("No fit failures recorded.")
    else:
        agg = sel.failures.groupby(["family", "model", "reason"]).size().rename("count").reset_index()
        st.dataframe(agg, width="stretch", hide_index=True)
        st.caption("Candidates that fail in any validation fold are ineligible so that all models are compared on the same folds.")
